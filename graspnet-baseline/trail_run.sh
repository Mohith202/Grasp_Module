#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=38
#SBATCH --gres=gpu:3
#SBATCH -p ihub
#SBATCH --nodelist=gnode097
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/mohit.g/Basline/grasptest/graspnet-baseline/create_ft.txt

# Load Miniconda and activate the conda environment
source /home/mohit.g/ENTER/etc/profile.d/conda.sh
conda activate graspenv

# Confirm the environment activation
echo "Conda environment activated: $(conda info --envs | grep '*')"

CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs_trail_attention/log_rs --batch_size 1 --dataset_root /ssd_scratch/mohit.g/ --num_point 20000 --num_view 300 --max_epoch 20 --learning_rate 0.001 --weight_decay 0 --bn_decay_step 2 --bn_decay_rate 0.5 --lr_decay_steps 8,12,16
