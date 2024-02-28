#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=drqv2_rnd_initial_20_loss_norm_obs_lr01
seed=1

echo "start running $tag with seed $seed"
python train.py task=reacher_hard repeat_type=3 update_every_steps=4 nstep=6 action_repeat=1 experiment=$tag seed=$seed num_train_frames=1000000
