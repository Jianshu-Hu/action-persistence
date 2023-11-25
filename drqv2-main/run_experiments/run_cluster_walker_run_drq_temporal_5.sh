#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_baseline_repeat_1_load_repeat_2
seed=5
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run train_dynamics_model=true action_repeat=1 stddev_schedule=$std load_model=action_repeat_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
