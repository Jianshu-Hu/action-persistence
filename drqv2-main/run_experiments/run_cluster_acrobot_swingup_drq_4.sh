#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_dyn_prior_K_5_reverse_last_two
seed=4

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup dyn_prior_K=5 state_dim=15 train_dynamics_model=1 action_repeat=2 save_model=false experiment=$tag seed=$seed num_train_frames=1000000
