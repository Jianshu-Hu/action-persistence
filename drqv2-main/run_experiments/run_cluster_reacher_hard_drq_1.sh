#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_frame_stack_2_add_last_action_process_each_frame
seed=1

echo "start running $tag with seed $seed"
python train.py task=reacher_hard frame_stack=2 train_dynamics_model=1 save_model=false action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
