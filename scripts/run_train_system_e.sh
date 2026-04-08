#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --output=results/train_system_e_%j.log

source scripts/setup_env.sh

# Sweep beta values for self-distillation KL loss
for beta in 0.1 0.5 1.0 2.0; do
    echo "=========================================="
    echo "Training System E with beta=$beta"
    echo "=========================================="
    python -m src.train_joint \
        --data_path data/msmarco_train_split.parquet \
        --epochs 3 \
        --batch_size 32 \
        --beta "$beta" \
        --output_weights_name "system_e_beta${beta}_weights.pt"
    echo ""
done

echo "All beta sweeps complete."
