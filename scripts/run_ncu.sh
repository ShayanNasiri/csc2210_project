#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH --output=results/ncu_%j.log

source scripts/setup_env.sh

ncu --set full -o results/nsight_compact_kernel python scripts/ncu_microbench.py
