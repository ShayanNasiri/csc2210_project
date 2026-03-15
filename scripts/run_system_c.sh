#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=results/system_c_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate
python -m src.inference --system system_c --batch_size 64 --data_path data/dev_tokenized.pt
