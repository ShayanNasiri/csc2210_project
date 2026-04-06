#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH --output=results/train_joint_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate
python -m src.train_joint --data_path data/msmarco_train.parquet --epochs 3 --batch_size 64
