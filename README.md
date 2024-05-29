

# Learn from Slo-Mo: State-Novelty Guided Adaptive Action Persistence in Deep Reinforcement Learning

This is an original PyTorch implementation of incorporating temporally persistent exploration in DrQ-v2 from

[//]: # ([[Learn from Slo-Mo: State-Novelty Guided Adaptive Action Persistence in Deep Reinforcement Learning]]&#40;https://openreview.net/forum?id=EGQBpkIEuu&#41; by)

Learn from Slo-Mo: State-Novelty Guided Adaptive Action Persistence in Deep Reinforcement Learning by
[Jianshu Hu](https://jianshu-hu.github.io/), [Paul Weng](https://weng.fr/) and [Yutong Ban](https://people.csail.mit.edu/yban/index.html).

## Method
We implement State-Novelty guided adaptive Action Persistence (SNAP) based on DrQv2.

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this repo in your research, please consider citing the paper as follows:)

[//]: # (```)

[//]: # (@inproceedings{)

[//]: # (hu2024revisiting,)

[//]: # (title={Revisiting Data Augmentation in Deep Reinforcement Learning},)

[//]: # (author={Jianshu Hu and Yunpeng Jiang and Paul Weng},)

[//]: # (booktitle={The Twelfth International Conference on Learning Representations},)

[//]: # (year={2024},)

[//]: # (url={https://openreview.net/forum?id=EGQBpkIEuu})

[//]: # (})

[//]: # (```)

## Instructions

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate drqv2
```

Train the agent with original DrQv2:
```sh
python train.py task=quadruped_walk
```

Train the agent with SNAP:
```sh
python train.py task=quadruped_walk repeat_type=1 action_repeat=1 update_every_steps=4 nstep=6
```

Monitor results:
```sh
tensorboard --logdir exp_local
```

## License
The majority of this code is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.