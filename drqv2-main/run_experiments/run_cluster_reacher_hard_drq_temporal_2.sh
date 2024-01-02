#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=load_decay_longer
seed=2
std=\'linear\(0.5,0.1,500000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard train_dynamics_model=1 action_repeat=1 num_updates=1 stddev_schedule=$std load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=1500000
