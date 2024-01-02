#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=load_decay
seed=2
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=quadruped_run train_dynamics_model=1 action_repeat=1 load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
