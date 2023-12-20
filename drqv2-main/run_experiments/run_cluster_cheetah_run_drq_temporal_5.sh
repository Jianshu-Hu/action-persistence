#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=cheetah_run_baseline_dyn_prior_5_previous_reverse
seed=5
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=cheetah_run time_ssl=3 time_ssl_K=3 state_dim=9 train_dynamics_model=1 action_repeat=2 save_model=false experiment=$tag seed=$seed num_train_frames=1000000
