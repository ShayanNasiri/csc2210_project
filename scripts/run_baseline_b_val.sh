#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=results/baseline_b_val_%j.log

source scripts/setup_env.sh
python -m src.inference --system baseline_b --batch_size 64 --data_path data/val_tokenized.pt --results_tag val_
