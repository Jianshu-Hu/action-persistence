#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_epsilon_greedy
seed=1

echo "start running $tag with seed $seed"
python train.py task=humanoid_stand epsilon_zeta=false epsilon_greedy=true epsilon_schedule=\'linear\(1.0,0.1,5000000\)\' action_repeat=2 experiment=$tag seed=$seed num_train_frames=10000000
