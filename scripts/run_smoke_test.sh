#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH --output=results/smoke_test_%j.log

source .venv/bin/activate
python scripts/smoke_test.py
