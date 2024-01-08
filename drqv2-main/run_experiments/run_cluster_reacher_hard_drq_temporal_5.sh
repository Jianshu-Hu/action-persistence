#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=pretrain_5e4
seed=5
std=\'linear\(0.5,0.1,500000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard pretrain_steps=50000 train_dynamics_model=1 action_repeat=1 stddev_schedule=$std load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
