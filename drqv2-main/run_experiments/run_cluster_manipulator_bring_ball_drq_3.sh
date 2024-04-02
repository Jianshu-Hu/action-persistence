#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_simhash_repeat
seed=3

echo "start running $tag with seed $seed"
python train.py task=manipulator_bring_ball repeat_type=1 update_every_steps=2 nstep=3 action_repeat=2 experiment=$tag seed=$seed num_train_frames=5000000
