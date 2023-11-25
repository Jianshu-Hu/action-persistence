#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reach_duplo_baseline_repeat_1_load_repeat_2_step_125000
seed=3
std=\'linear\(0.75,0.1,750000\)\'

echo "start running $tag with seed $seed"
python train.py task=reach_duplo train_dynamics_model=true action_repeat=1 stddev_schedule=$std load_model=action_repeat_2_step_125000 experiment=$tag seed=$seed num_train_frames=250000
