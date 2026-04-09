#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --output=results/system_e_alpha0.5_val_%j.log

source scripts/setup_env.sh

# Evaluate all System E alpha=0.5 beta sweep weight files on validation set
for beta in 0.1 0.5 1.0 2.0; do
    weights="results/system_e_alpha0.5_beta${beta}_weights.pt"
    if [ -f "$weights" ]; then
        echo "=========================================="
        echo "Evaluating System E with alpha=0.5 beta=$beta (val)"
        echo "=========================================="
        python -m src.inference \
            --system system_e \
            --batch_size 64 \
            --data_path data/val_tokenized.pt \
            --weights_path "$weights" \
            --results_tag "val_alpha0.5_beta${beta}_"
        echo ""
    else
        echo "SKIP: $weights not found"
    fi
done

echo "All System E alpha=0.5 val evaluations complete."
