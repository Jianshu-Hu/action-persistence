#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/drqv2-main/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=walker_run_inv_dyn_model_reverse_loss_wo_reflect
seed=2
#std=\'linear\(0.5,0.1,250000\)\'

echo "start running $tag with seed $seed"
python train.py task=walker_run time_ssl=0 time_ssl_K=4 train_dynamics_model=2 action_repeat=2 save_model=false experiment=$tag seed=$seed num_train_frames=1000000
