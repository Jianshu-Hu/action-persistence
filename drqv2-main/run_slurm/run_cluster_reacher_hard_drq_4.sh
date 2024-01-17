#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -J drq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd ~/temporal-invariance/drqv2-main/
source ~/anaconda3/bin/activate
conda activate drqv2

tag=reacher_hard
seed=4

echo "start running $tag with seed $seed"
python train.py task=reacher_hard train_dynamics_model=1 save_model=true action_repeat=2 experiment=$tag seed=$seed num_train_frames=1000000
