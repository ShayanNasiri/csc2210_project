#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13,gpunode1,gpunode28,gpunode6,gpunode11
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH --array=0-1
#SBATCH --output=results/train_joint_alpha_%A_%a.log

source scripts/setup_env.sh

# Each array task trains one alpha value.
# alpha=1.0 already exists as results/joint_weights.pt — not re-trained here.
ALPHAS=(0.5 2.0)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training System D with alpha=$ALPHA (task $SLURM_ARRAY_TASK_ID of ${#ALPHAS[@]})"
echo "=========================================="

python -m src.train_joint \
    --data_path data/msmarco_train_split.parquet \
    --epochs 3 \
    --batch_size 32 \
    --alpha "$ALPHA" \
    --output_weights_name "joint_alpha${ALPHA}_weights.pt"
