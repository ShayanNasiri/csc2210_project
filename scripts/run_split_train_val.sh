#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --output=results/split_train_val_%j.log

source scripts/setup_env.sh
python -m data.split_train_val --input_path data/msmarco_train.parquet --output_dir data --val_fraction 0.05
