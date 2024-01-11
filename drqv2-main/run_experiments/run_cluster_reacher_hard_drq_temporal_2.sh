#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=5e4_buffer_more_explore
seed=2
std=\'linear\(1.0,0.1,500000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard load_num_frames=50000 train_dynamics_model=1 action_repeat=1 stddev_schedule=$std load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
