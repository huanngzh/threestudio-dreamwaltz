import functools
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import trimesh

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from ..utils import render_mesh as render_tools


@threestudio.register("dreamwaltz-system")
class DreamWaltz(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)

        warmup_steps: int = 2000

        prior_path: str = "custom/threestudio-dreamwaltz/load/priors/smpl_apose.obj"
        prior_radius: float = 0.7
        prior_rotate: str = "threestudio"  # in ['threestudio', 'meshlab', 'glb']
        prior_view_hw: Optional[Tuple[int, int]] = None

        controlnet_ref_types: List[str] = field(
            default_factory=lambda: ["rgb"]
        )  # in ['rgb', 'depth']

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        assert self.cfg.prior_path is not None, "Prior path is required."
        self.prior = trimesh.load(
            self.cfg.prior_path, force="scene", merge_primitives=True
        )
        self.prior_scene, nc, nl = render_tools.prepare_pyrender_scene(
            self.prior,
            preprocess_mesh=True,
            normalize_mesh=True,
            radius=self.cfg.prior_radius,
            source=self.cfg.prior_rotate,
        )
        self.render_prior = functools.partial(
            render_tools.render_scene, self.prior_scene, nc, nl
        )
        threestudio.info(f"Using observation prior: {self.cfg.prior_path}")

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
        def _get_prior_view(view_hw=None) -> Float[Tensor, "B H W C"]:
            if view_hw is not None:
                height, width = view_hw
            else:
                rgb_wh = out["comp_rgb"] if "comp_rgb" in out else out["comp_normal"]
                height, width = rgb_wh.shape[1:3]
            device = get_device()

            colors, depths = [], []
            for i in range(batch["fovy"].shape[0]):
                color, depth = self.render_prior(
                    batch["c2w"][i].cpu().numpy(),
                    batch["fovy"][i].item(),
                    width=width,
                    height=height,
                )
                color = torch.from_numpy(color.copy()) / 255.0
                depth = torch.from_numpy(depth.copy() / depth.max())
                depth = depth.unsqueeze(-1).repeat(1, 1, 3)
                colors.append(color)
                depths.append(depth)

            return torch.stack(colors, dim=0).to(device), torch.stack(depths, dim=0).to(
                device
            )

        out = self(batch)
        prompt_utils = self.prompt_processor()

        guidance_eval = (
            self.true_global_step >= self.cfg.warmup_steps
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        loss = 0.0

        if self.true_global_step < self.cfg.warmup_steps:
            # warmup
            gt_views, gt_depths = _get_prior_view()
            img_mask = (gt_depths.detach() > 0).float()
            loss_l2 = F.mse_loss(
                out["comp_rgb"] * img_mask, gt_views.detach() * img_mask
            )
            self.log("train/loss_l2", loss_l2)
            loss += loss_l2 * self.C(self.cfg.loss.lambda_l2)

        else:
            # prepare condition reference images
            cond_rgbs = []
            for ref_type in self.cfg.controlnet_ref_types:
                # use prior view as control
                cond_rgb, cond_depth = _get_prior_view(self.cfg.prior_view_hw)
                if ref_type == "depth":
                    cond_rgb = cond_depth
                cond_rgbs.append(cond_rgb)

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

        if guidance_eval:
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
