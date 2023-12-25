#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=2_load_4
seed=2
std=\'linear\(0.75,0.1,375000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard time_ssl_K=0 train_dynamics_model=1 action_repeat=2 stddev_schedule=$std load_folder=repeat_4 load_model=ar_4_step_125000 experiment=$tag seed=$seed num_train_frames=500000
