#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=results/baseline_b_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate
python -m src.inference --system baseline_b --batch_size 64 --data_path data/dev_tokenized.pt
