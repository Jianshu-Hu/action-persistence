#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_baseline_dyn_prior_5_previous_reverse_state_20
seed=3

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup state_dim=20 time_ssl_K=0 dyn_prior_K=5 train_dynamics_model=1 action_repeat=2 save_model=false experiment=$tag seed=$seed num_train_frames=1000000
