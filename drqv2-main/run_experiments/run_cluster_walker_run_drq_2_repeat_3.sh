#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_baseline_time_ssl_K_4_critic
seed=3
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run time_ssl_K=4 train_dynamics_model=1 action_repeat=2 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
