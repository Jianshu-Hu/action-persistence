#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dyn_rew_model_tie_dyn_critic_action_repeat_1
seed=2

echo "start running $tag with seed $seed"
python train.py task=walker_run train_dynamics_model=true action_repeat=1 save_model=true experiment=$tag seed=$seed num_train_frames=1000000
