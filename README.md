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

## Quick Start
```bash
# DreamWaltz - Static Avatar Creation
python launch.py --config custom/threestudio-dreamwaltz/configs/dreamwaltz-static.yaml --train --gpu 0 system.prompt_processor.prompt="Naruto" system.prior_path="custom/threestudio-dreamwaltz/load/priors/smpl_apose.obj"

# DreamWaltz - Animatable Avatar Learning
# !! Not yet implemented
```

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