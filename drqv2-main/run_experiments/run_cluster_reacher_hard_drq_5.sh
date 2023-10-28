#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_dyn_rew_model_tie_dyn_critic_time_scale_2_mimic_repeat_4_decay_weight
seed=1

echo "start running $tag with seed $seed"
python train.py task=reacher_hard time_scale=2.0 train_dynamics_model=true load_model=action_repeat_4_step_237500 action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
