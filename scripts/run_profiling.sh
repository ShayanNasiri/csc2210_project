#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 5:00:00
#SBATCH --output=results/profiling_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate

echo "=== Full latency sweep ==="
python -m src.inference --system full_sweep --data_path data/dev_tokenized.pt
