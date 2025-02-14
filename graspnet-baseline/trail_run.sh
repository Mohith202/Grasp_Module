#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=36
#SBATCH --gres=gpu:4
#SBATCH -p ihub
#SBATCH --nodelist=gnode109
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/mohit.g/results/create_ft.txt

# Load Miniconda and activate the conda environment
source /home/mohit.g/ENTER/etc/profile.d/conda.sh
conda activate graspenv

# Confirm the environment activation
echo "Conda environment activated: $(conda info --envs | grep '*')"

CUDA_VISIBLE_DEVICES=3 python train.py --camera realsense --log_dir logs/log_rs --batch_size 1 --dataset_root /ssd_scratch/mohit.g/GraspNet
