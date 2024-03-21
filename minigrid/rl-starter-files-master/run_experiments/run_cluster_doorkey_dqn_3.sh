#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/minigrid/rl-starter-files-master/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate minigrid

tag=dqn_simhash_repeat
seed=3

echo "start running $tag with seed $seed"
python3 -m scripts.train --simhash_repeat --algo dqn --env MiniGrid-DoorKey-6x6-v0 --seed $seed --frames=100000
