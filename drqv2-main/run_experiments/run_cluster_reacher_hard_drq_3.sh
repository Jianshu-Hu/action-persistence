#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_simhash+actor_sim_proj
seed=3

echo "start running $tag with seed $seed"
python train.py task=reacher_hard repeat_type=1 temp_cluster=true update_every_steps=2 nstep=3 action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
