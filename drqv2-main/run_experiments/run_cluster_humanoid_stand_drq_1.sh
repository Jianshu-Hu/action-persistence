#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=humanoid_stand
seed=1

echo "start running $tag with seed $seed"
python train.py task=humanoid_stand experiment=$tag seed=$seed replay_buffer_num_workers=4