#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_repeat_4_train_same_batch_size
seed=2

echo "start running $tag with seed $seed"
python train.py task=manipulation_reach_site repeat_type=2 update_every_steps=2 nstep=8 action_repeat=2 experiment=$tag seed=$seed num_train_frames=3000000
