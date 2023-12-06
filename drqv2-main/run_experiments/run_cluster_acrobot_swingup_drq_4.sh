#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_baseline_time_scale_4
seed=4

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup time_ssl=1 time_ssl_K=4 train_dynamics_model=true action_repeat=2 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
