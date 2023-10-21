#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dyn_rew_model_tie_dyn_critic_time_reflect_time_scale_2
seed=3

echo "start running $tag with seed $seed"
python train.py task=walker_run time_scale=2.0 time_reflection=true train_dynamics_model=true experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
