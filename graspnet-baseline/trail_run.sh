#!/bin/bash
#SBATCH -A hai
#SBATCH --mincpus=37
#SBATCH --gres=gpu:4
#SBATCH -p ihub
#SBATCH --nodelist=gnode109
#SBATCH --mem-per-cpu=3G
#SBATCH --time=90:00:00
#SBATCH --output=/home/mohit.g/Grasp_Module/graspnet-baseline/create_ft.txt

# Load Miniconda and activate the conda environment
source /home/mohit.g/ENTER/etc/profile.d/conda.sh
conda activate graspenv

# Confirm the environment activation
echo "Conda environment activated: $(conda info --envs | grep '*')"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --camera realsense --log_dir logs_V1/log --batch_size 4 --dataset_root /ssd_scratch/mohit.g/GraspNet --num_point 20000 --num_view 300 --max_epoch 14 --learning_rate 0.001 --weight_decay 0 --bn_decay_step 2 --bn_decay_rate 0.5 --lr_decay_steps 8,12,16