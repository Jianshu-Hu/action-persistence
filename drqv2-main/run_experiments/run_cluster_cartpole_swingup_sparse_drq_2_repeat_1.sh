#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=cartpole_swingup_sparse_baseline_repeat_2
seed=1
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=cartpole_swingup_sparse train_dynamics_model=true action_repeat=2 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
