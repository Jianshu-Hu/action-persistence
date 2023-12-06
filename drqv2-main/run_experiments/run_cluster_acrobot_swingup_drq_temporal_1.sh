#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=2_load_4
seed=1
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup time_ssl=0 time_ssl_K=4 train_dynamics_model=true action_repeat=2 stddev_schedule=$std load_folder=repeat_4 load_model=ar_4_step_237500 experiment=$tag seed=$seed num_train_frames=250000
