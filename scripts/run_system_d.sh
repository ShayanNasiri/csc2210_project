#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=results/system_d_%j.log

source scripts/setup_env.sh
python -m src.inference --system system_d --batch_size 64 --data_path data/dev_tokenized.pt
