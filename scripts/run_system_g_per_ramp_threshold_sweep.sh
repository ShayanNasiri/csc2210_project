#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode7,gpunode13,gpunode33
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --array=0-48
#SBATCH --job-name=system_g_sweep
#SBATCH --output=results/system_g_sweep_%A_%a.log

# System G: per-ramp entropy threshold grid sweep over System E (alpha=1.0,
# beta=1.0) weights — `results/system_e_alpha1.0_beta1.0_weights.pt`.
#
# Same 7^5 = 16,807-config grid and 49-task (t0, t1) partitioning as System F.
# The only differences from System F are the weights, the output subdirectory,
# and the CSV filename prefix — chosen so System G outputs cannot collide
# with the already-committed System F sweep results.
#
# Output: results/system_g_sweep_results/system_g_e_alpha1.0_beta1.0_t0=<t0>_t1=<t1>.csv
#
# Selection criterion (post-sweep): max MRR@10 subject to mean batch latency
# <= 16.5 ms (same cap as System F). System E beta=1.0 runs at 14.37 ms
# single-threshold, so System G has ~2.1 ms of latency headroom that System
# F's 16.38 ms winner doesn't have.
#
# Tasks are independent — SLURM schedules them in parallel as rtx_4090 slots
# free up. No --dependency needed.
#
# Exclude list (verified 2026-04-11):
#   gpunode7, gpunode13 — historically RTX A4500 nodes (cuBLAS silently
#                          corrupts joint-trained weights; see
#                          memory/project_cluster_gpu_constraint.md).
#                          --gres=gpu:rtx_4090:1 already filters them, but
#                          explicit exclude is defense-in-depth.
#   gpunode33           — drained by admin for system disk failure 2026-04-10.
# gpunode4 and gpunode5 are NOT excluded — graduated to trusted rtx_4090 set
# 2026-04-10 via bit-equality proofs vs gpunode32/34.

source scripts/setup_env.sh

echo "=========================================="
echo "System G per-ramp sweep — array task $SLURM_ARRAY_TASK_ID / 48"
echo "Weights: results/system_e_alpha1.0_beta1.0_weights.pt"
echo "Output : results/system_g_sweep_results/"
echo "=========================================="

python -m src.inference \
    --system per_ramp_sweep \
    --batch_size 64 \
    --data_path data/val_tokenized.pt \
    --weights_path results/system_e_alpha1.0_beta1.0_weights.pt \
    --sweep_subdir system_g_sweep_results \
    --csv_prefix system_g_e_alpha1.0_beta1.0 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks 49
