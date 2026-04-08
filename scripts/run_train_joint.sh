#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH --output=results/train_joint_%j.log

source scripts/setup_env.sh
python -m src.train_joint --data_path data/msmarco_train_split.parquet --epochs 3 --batch_size 32
