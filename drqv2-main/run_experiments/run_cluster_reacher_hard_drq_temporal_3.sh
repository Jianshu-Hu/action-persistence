#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=1e5_buffer_up_4
seed=3
std=\'linear\(0.5,0.1,300000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard num_updates=4 load_num_frames=100000 train_dynamics_model=1 action_repeat=1 stddev_schedule=$std load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=300000
