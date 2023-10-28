#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dyn_rew_model_tie_dyn_critic_repeat_1_mimic_repeat_2_policy_decay_weight
seed=3

echo "start running $tag with seed $seed"
python train.py task=walker_run train_dynamics_model=true action_repeat=1 load_model=action_repeat_2_step_475000 experiment=$tag seed=$seed num_train_frames=1000000
