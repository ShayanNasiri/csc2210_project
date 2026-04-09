#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --array=0-3
#SBATCH --output=results/system_e_alpha0.5_val_%A_%a.log

source scripts/setup_env.sh

# Each array task evaluates one beta value at alpha=0.5.
# Intended to run with --dependency=aftercorr:<train_array_jobid> so that
# val task i starts as soon as train task i finishes.
BETAS=(0.1 0.5 1.0 2.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

weights="results/system_e_alpha0.5_beta${BETA}_weights.pt"

echo "=========================================="
echo "Evaluating System E with alpha=0.5 beta=$BETA (val, task $SLURM_ARRAY_TASK_ID)"
echo "=========================================="

python -m src.inference \
    --system system_e \
    --batch_size 64 \
    --data_path data/val_tokenized.pt \
    --weights_path "$weights" \
    --results_tag "val_alpha0.5_beta${BETA}_"
