#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH --output=results/download_data_%j.log

source scripts/setup_env.sh
python data/download_data.py
