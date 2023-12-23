#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_baseline_repeat_4_time_ssl_K_4_actor
seed=1

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup time_ssl_K=4 train_dynamics_model=1 action_repeat=4 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
