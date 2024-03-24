#!/bin/bash

cd /bd_targaryen/users/jhu/temporal-invariance/minigrid/rl-starter-files-master/
source /bd_targaryen/users/jhu/anaconda3/bin/activate
conda activate minigrid

tag=dqn_simhash_repeat_target_network_c05
seed=2

echo "start running $tag with seed $seed"
python3 -m scripts.train --simhash_repeat --algo dqn --env MiniGrid-Empty-8x8-v0 --eval_freq=1000 --seed $seed --frames=50000
