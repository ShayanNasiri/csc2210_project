#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=results/pytest_%j.log

source scripts/setup_env.sh

python -m pytest tests/ -v --tb=short
