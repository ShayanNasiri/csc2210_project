#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --array=0-48
#SBATCH --output=results/system_f_sweep_%A_%a.log

# System F: per-ramp entropy threshold grid sweep over System D alpha=0.5.
#
# Full grid: 7^5 = 16,807 configs over the log-spaced grid
#   [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
# applied independently to each of the 5 off-ramps.
#
# Partitioning: 49 array tasks (7 x 7), one per (t0, t1) pair. Each task
# evaluates the inner (t2, t3, t4) sub-grid (343 configs) and writes a CSV
# named results/sweep_results/system_f_t0=<t0>_t1=<t1>.csv.
#
# Tasks are independent — SLURM schedules them in parallel as rtx_4090 slots
# free up. No --dependency needed.

source scripts/setup_env.sh

echo "=========================================="
echo "System F per-ramp sweep — array task $SLURM_ARRAY_TASK_ID / 48"
echo "=========================================="

python -m src.inference \
    --system per_ramp_sweep \
    --batch_size 64 \
    --data_path data/val_tokenized.pt \
    --weights_path results/joint_alpha0.5_weights.pt \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks 49
