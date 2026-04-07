#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=results/baseline_b_sanity_%j.log

source scripts/setup_env.sh

python -c "
from src.inference import run_baseline_b
r = run_baseline_b('data/dev_tokenized.pt', batch_size=64, thresholds=[0.0])
print('MRR@10:', r[0]['mrr10'])
print('Exit counts:', r[0]['exit_counts'])
print('Expected: MRR ~0.73, exit_counts=[0,0,0,0,0,62357]')
"
