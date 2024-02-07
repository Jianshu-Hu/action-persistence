#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/drqv2-main/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate drqv2

tag=repeat_unvisit
seed=5

echo "start running $tag with seed $seed"
python train.py task=cheetah_run transfer=true train_dynamics_model=1 action_repeat=1 experiment=$tag seed=$seed num_train_frames=1000000
