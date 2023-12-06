#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=baseline_1_load_2_tscale_4
seed=3
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run time_ssl=1 time_ssl_K=4 train_dynamics_model=true action_repeat=1 stddev_schedule=$std load_folder=tscale_4 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
