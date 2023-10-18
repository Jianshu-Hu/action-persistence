#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_dynamics_reward_model
seed=5

echo "start running $tag with seed $seed"
python train.py train_dynamics_model=true train_reward_model=true task=walker_run experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
