#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_epsilon_greedy
seed=4

echo "start running $tag with seed $seed"
python train.py task=manipulation_reach_site epsilon_greedy=true epsilon_schedule=\'linear\(1.0,0.1,2500000\)\' repeat_type=0 update_every_steps=2 nstep=3 action_repeat=2 experiment=$tag seed=$seed num_train_frames=5000000
