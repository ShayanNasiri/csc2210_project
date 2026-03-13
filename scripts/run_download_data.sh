#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH --output=results/download_data_%j.log

export IR_DATASETS_HOME=/tmp/ir_datasets_$USER

source .venv/bin/activate
python data/download_data.py
