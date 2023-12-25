#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=1_t4_load_2_t4
seed=1
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=hopper_hop time_ssl_K=4 train_dynamics_model=1 action_repeat=1 stddev_schedule=$std load_folder=t4_critic load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
