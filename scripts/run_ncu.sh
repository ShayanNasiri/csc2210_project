#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH --output=results/ncu_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate

ncu --set full -o results/nsight_compact_kernel python scripts/ncu_microbench.py
