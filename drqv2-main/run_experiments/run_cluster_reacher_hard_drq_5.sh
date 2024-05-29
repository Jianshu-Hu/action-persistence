#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_repeat_4_update_every_1_nstep_2
seed=5

echo "start running $tag with seed $seed"
python train.py task=reacher_hard repeat_type=0 update_every_steps=1 nstep=2 action_repeat=4 experiment=$tag seed=$seed num_train_frames=1000000
