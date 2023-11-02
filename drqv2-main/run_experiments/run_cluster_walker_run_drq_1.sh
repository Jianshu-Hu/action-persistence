#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dyn_rew+load_repeat_4_remove_exploration_original_std
seed=1
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run train_dynamics_model=true num_expl_steps=0 load_model=action_repeat_4_step_237500 experiment=$tag seed=$seed num_train_frames=1000000
