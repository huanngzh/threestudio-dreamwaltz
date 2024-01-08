from dataclasses import dataclass, field

import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("dreamwaltz-system")
class DreamWaltz(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)

        stage: Optional[str] = None  # warmup, nerf
        skip_warmup: bool = False
        warmup_steps: int = 2000  # only used when skip_warmup is False

        controlnet_ref_types: List[str] = field(default_factory=lambda: ["pose"])

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()

        guidance_eval = (
            self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        loss = 0.0

        if self.cfg.stage == "warmup" or (
            self.true_global_step < self.cfg.warmup_steps and not self.cfg.skip_warmup
        ):
            # warmup
            assert (
                "mesh" in batch and "depth" in batch
            ), "Mesh and depth are required in warmup stage."
            gt_views = batch["mesh"].to(out["comp_rgb"].dtype)
            gt_depths = batch["depth"].to(out["comp_rgb"].dtype)
            if gt_views.shape[1] != out["comp_rgb"].shape[1]:
                # resize gt_views [B, H, W, C] to match the output size
                resize_ = lambda x: F.interpolate(
                    x.permute(0, 3, 1, 2),
                    size=out["comp_rgb"].shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                gt_views = resize_(gt_views)
                gt_depths = resize_(gt_depths)[..., :1]

            img_mask = (gt_depths.detach() > 1e-6).float()
            loss_l2 = F.mse_loss(
                out["comp_rgb"] * img_mask, gt_views.detach() * img_mask
            )
            self.log("train/loss_l2", loss_l2)
            loss += loss_l2 * self.C(self.cfg.loss.lambda_l2)

            if guidance_eval:
                # save warmup ground truth images
                self.save_image_grid(
                    f"it{self.true_global_step}-gt.png",
                    [
                        {
                            "type": "rgb",
                            "img": batch["mesh"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    + (
                        [
                            {
                                "type": "grayscale",
                                "img": batch["depth"][0, :, :, 0],
                                "kwargs": {"cmap": None, "data_range": (0, 1)},
                            }
                        ]
                        if "depth" in batch
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": batch["pose"][0],
                                "kwargs": {"data_format": "HWC"},
                            }
                        ]
                        if "pose" in batch
                        else []
                    ),
                )

        else:
            # prepare condition reference images
            cond_rgbs = []
            for ref_type in self.cfg.controlnet_ref_types:
                if ref_type not in ["pose", "depth"]:
                    raise ValueError(f"Invalid controlnet reference type: {ref_type}")
                if ref_type not in batch:
                    raise ValueError(
                        f"Controlnet reference type {ref_type} is not found in the batch. Please check the data config."
                    )
                cond_rgbs.append(batch[ref_type])

            # guidance
            guidance_out = self.guidance(
                out["comp_rgb"],
                cond_rgbs,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )

            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    self.log(f"train/{name}", value)
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

            if guidance_eval:
                # save guidance evaluation images
                self.guidance_evaluation_save(
                    out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                    guidance_out["eval"],
                )

                # visualize controlnet condition images
                if "controlnet_cond_images" in guidance_out:
                    controlnet_cond_images = [
                        {
                            "type": "rgb",
                            "img": cond_image[0],
                            "kwargs": {"data_format": "CHW"},
                        }
                        for cond_image in guidance_out["controlnet_cond_images"]
                    ]
                    self.save_image_grid(
                        f"it{self.true_global_step}-cond.png",
                        controlnet_cond_images,
                    )

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
