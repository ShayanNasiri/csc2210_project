#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=results/system_f_smoke_%j.log

# Smoke test for the per-ramp threshold refactor:
# re-run System D alpha=0.5 val with float threshold and verify the
# refactored forward_compacted_early_exit reproduces the existing JSON.
# Tagged "smoke_" so the canonical val_alpha0.5_system_d_results.json
# is not overwritten.

source scripts/setup_env.sh

python -m src.inference \
    --system system_d \
    --batch_size 64 \
    --data_path data/val_tokenized.pt \
    --weights_path results/joint_alpha0.5_weights.pt \
    --results_tag "smoke_alpha0.5_"
