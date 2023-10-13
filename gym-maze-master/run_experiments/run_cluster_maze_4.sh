#!/bin/bash

cd /bigdata/users/jhu/temporal-invariance/gym-maze-master
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate T-INV

tag=sample_maze_5_5_with_time_inverse
seed=4

echo "start running $tag with seed $seed"
python Q-learning.py --env=maze-sample-5x5-v0 --time_inv=1 --seed=$seed
