#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_dyn_rew_model_tie_dyn_critic
seed=5

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup train_dynamics_model=true action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
