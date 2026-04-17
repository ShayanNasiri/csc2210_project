#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --output=results/system_e_test_%j.log

source scripts/setup_env.sh

# Evaluate all System E beta sweep weight files on test set
for beta in 0.1 0.5 1.0 2.0; do
    weights="results/system_e_alpha1.0_beta${beta}_weights.pt"
    if [ -f "$weights" ]; then
        echo "=========================================="
        echo "Evaluating System E with beta=$beta (test)"
        echo "=========================================="
        python -m src.inference \
            --system system_e \
            --batch_size 64 \
            --data_path data/dev_tokenized.pt \
            --weights_path "$weights" \
            --results_tag "test_beta${beta}_"
        echo ""
    else
        echo "SKIP: $weights not found"
    fi
done

echo "All System E test evaluations complete."
