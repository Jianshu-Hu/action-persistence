#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_batch_unvisit_repeat_nstep6_upevery4
seed=4

echo "start running $tag with seed $seed"
python train.py task=humanoid_run transfer=true transfer_frames=500000 update_every_steps=4 nstep=6 action_repeat=1 experiment=$tag seed=$seed num_train_frames=10000000
