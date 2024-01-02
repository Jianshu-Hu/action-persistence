#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_action_repeat_2
seed=5

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup train_dynamics_model=1 action_repeat=2 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
