#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_baseline_repeat_2_ssl_scale_5
seed=5
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run time_ssl=1 time_ssl_K=9 train_dynamics_model=true action_repeat=2 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
