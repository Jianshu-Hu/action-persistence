#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_inv_dyn_reverse_loss_reversed_obs
seed=1

echo "start running $tag with seed $seed"
python train.py task=reacher_hard time_ssl=0 time_ssl_K=3 train_dynamics_model=2 save_model=false action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
