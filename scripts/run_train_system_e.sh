#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13,gpunode1,gpunode28,gpunode6,gpunode11
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH --array=0-3
#SBATCH --output=results/train_system_e_%A_%a.log

source scripts/setup_env.sh

# Each array task trains one beta value
BETAS=(0.1 0.5 1.0 2.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training System E with beta=$BETA (task $SLURM_ARRAY_TASK_ID of ${#BETAS[@]})"
echo "=========================================="

python -m src.train_joint \
    --data_path data/msmarco_train_split.parquet \
    --epochs 3 \
    --batch_size 32 \
    --beta "$BETA" \
    --output_weights_name "system_e_beta${BETA}_weights.pt"
