#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH --output=results/smoke_test_%j.log

# Build venv if missing
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

python scripts/smoke_test.py
