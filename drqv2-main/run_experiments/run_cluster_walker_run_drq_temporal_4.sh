#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dyn_rew_model+extend_actor_critic_repeat_4_remove_all_expl
seed=4
std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run train_dynamics_model=true stddev_schedule=$std load_model=action_repeat_4_step_237500 experiment=$tag seed=$seed num_train_frames=1000000
