#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_baseline_repeat_2
seed=2

echo "start running $tag with seed $seed"
python train.py task=reacher_hard train_dynamics_model=1 save_model=true action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
