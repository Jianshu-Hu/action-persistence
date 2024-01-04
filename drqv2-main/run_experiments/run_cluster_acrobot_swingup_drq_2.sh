#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=fs_2_add_last_action_process_each_half_feature
seed=2

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup frame_stack=2 feature_dim=25 train_dynamics_model=1 action_repeat=2 save_model=false experiment=$tag seed=$seed num_train_frames=1000000
