#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode7,gpunode13,gpunode33
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:15:00
#SBATCH --array=0-195
#SBATCH --job-name=system_h_sweep
#SBATCH --output=results/system_h_sweep_%A_%a.log

# System H: patience-based early exit (PABEE) sweep over System E (alpha=1.0,
# beta=1.0) weights — same weights as System G.
#
# Sweeps patience P in {2, 3, 4, 5} over the same 7^5 = 16,807-config
# per-ramp threshold grid as System G. P=1 results already exist as the
# System G sweep — no need to re-run.
#
# 196 tasks = 4 patience values x 49 (t0, t1) cells.
# Task ID encoding:
#   PATIENCE = task_id / 49 + 2   (0-48 -> P=2, 49-97 -> P=3, etc.)
#   CELL_ID  = task_id % 49       (usual (t0, t1) cell index)
#
# Output: results/system_h_sweep_results/system_h_p{P}_t0=<t0>_t1=<t1>.csv
#
# Exclude list: gpunode7/13 (A4500, cuBLAS corruption), gpunode33 (drained).

export HF_HOME=/tmp/hf_cache_$USER
export IR_DATASETS_HOME=/tmp/ir_datasets_$USER

source scripts/setup_env.sh

PATIENCE=$(( SLURM_ARRAY_TASK_ID / 49 + 2 ))
CELL_ID=$(( SLURM_ARRAY_TASK_ID % 49 ))

echo "=========================================="
echo "System H patience sweep — array task $SLURM_ARRAY_TASK_ID / 195"
echo "Patience: $PATIENCE"
echo "Cell ID:  $CELL_ID / 48"
echo "Weights:  results/system_e_alpha1.0_beta1.0_weights.pt"
echo "Output:   results/system_h_sweep_results/"
echo "=========================================="

python -m src.inference \
    --system per_ramp_sweep \
    --batch_size 64 \
    --data_path data/val_tokenized.pt \
    --weights_path results/system_e_alpha1.0_beta1.0_weights.pt \
    --sweep_subdir system_h_sweep_results \
    --csv_prefix "system_h_p${PATIENCE}" \
    --patience $PATIENCE \
    --task_id $CELL_ID \
    --num_tasks 49
