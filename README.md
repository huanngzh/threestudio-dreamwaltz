# threestudio-dreamwaltz

<img src="https://github.com/huanngzh/threestudio-dreamwaltz/assets/78398294/ad0a79ba-41c4-449b-96ee-a49c7591a94d" width="48%">
<img src="https://github.com/huanngzh/threestudio-dreamwaltz/assets/78398294/ebbab7a8-4182-4d0d-9a4c-b7615a2eeaa8" width="48%">

DreamWaltz extension of threestudio. To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

## Installation
```bash
cd custom
git clone https://github.com/huanngzh/threestudio-dreamwaltz.git
cd threestudio-dreamwaltz

pip install -r requirements.txt
```

If installing the pytorch3d package fails, please see the detailed instructions at [pytorch3d/INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Prepare SMPL Weights
We use smpl and vposer models for avatar creation and animation learning, please follow the instructions in [smplx](https://github.com/vchoutas/smplx#downloading-the-model) and [human_body_prior](https://github.com/nghorbani/human_body_prior) to download the model weights, and build a directory with the following structure:
```
smpl_models
├── smpl
│   ├── SMPL_FEMALE.pkl
│   └── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
└── vposer
    └── v2.0
        ├── snapshots
        ├── V02_05.yaml
        └── V02_05.log
```
Then, update the model paths `SMPL_ROOT` and `VPOSER_ROOT` in `utils/smpl/smpl_prompt.py`.

## Quick Start

### Static Avatar Creation
All in one (SMPL Initializaion + Canonical Avatar Creation):
```bash
python launch.py --config custom/threestudio-dreamwaltz/configs/dreamwaltz-static.yaml --train --gpu 0 system.prompt_processor.prompt="Naruto"
```

Divided into multiple stages:
```bash
# SMPL Initializaion
python launch.py --config custom/threestudio-dreamwaltz/configs/experimental/dreamwaltz-1-warmup.yaml --train --gpu 0 system.prompt_processor.prompt="Naruto"
# Canonical Avatar Creation
python launch.py --config custom/threestudio-dreamwaltz/configs/experimental/dreamwaltz-2-nerf.yaml --train --gpu 0 system.prompt_processor.prompt="Naruto" resume=path/to/trial/dir/ckpts/last.ckpt
```

### Animatable Avatar Learning
Not yet implemented!

## Citing
If you find DreamWaltz helpful, please consider citing:
```
@article{huang2023dreamwaltz,
    title={DreamWaltz: Make a Scene with Complex 3D Animatable Avatars},
    author={Yukun Huang and Jianan Wang and Ailing Zeng and He Cao and Xianbiao Qi and Yukai Shi and Zheng-Jun Zha and Lei Zhang},
    journal = {arXiv:2305.12529},
    year={2023},
}
```