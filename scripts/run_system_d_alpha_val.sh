#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode4,gpunode5,gpunode7,gpunode13,gpunode1,gpunode28,gpunode6,gpunode11
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --output=results/system_d_alpha_val_%j.log

source scripts/setup_env.sh

# Evaluate all System D alpha sweep weight files on validation set.
# alpha=1.0 uses the canonical joint_weights.pt; sweep variants live at
# results/joint_alpha${ALPHA}_weights.pt.
for alpha in 0.5 2.0; do
    weights="results/joint_alpha${alpha}_weights.pt"
    if [ -f "$weights" ]; then
        echo "=========================================="
        echo "Evaluating System D with alpha=$alpha (val)"
        echo "=========================================="
        python -m src.inference \
            --system system_d \
            --batch_size 64 \
            --data_path data/val_tokenized.pt \
            --weights_path "$weights" \
            --results_tag "val_alpha${alpha}_"
        echo ""
    else
        echo "SKIP: $weights not found"
    fi
done

echo "All System D alpha val evaluations complete."
