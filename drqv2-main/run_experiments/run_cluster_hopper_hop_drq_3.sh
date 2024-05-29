#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_epsilon
seed=3

echo "start running $tag with seed $seed"
python train.py task=hopper_hop epsilon_greedy=true epsilon_zeta=false action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
