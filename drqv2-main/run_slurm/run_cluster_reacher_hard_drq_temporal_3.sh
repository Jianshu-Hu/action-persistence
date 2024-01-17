#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -J drqv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd ~/temporal-invariance/drqv2-main/
source ~/anaconda3/bin/activate
conda activate drqv2

tag=load_pretrain_5e4
seed=3
std=\'linear\(0.5,0.1,500000\)\'

echo "start running $tag with seed $seed"
python train.py task=reacher_hard pretrain_steps=2500 train_dynamics_model=1 action_repeat=1 stddev_schedule=$std load_folder=repeat_2 load_model=ar_2_step_250000 experiment=$tag seed=$seed num_train_frames=500000
