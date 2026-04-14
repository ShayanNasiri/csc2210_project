#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode7,gpunode13,gpunode33
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=results/final_eval_%j.log

# Unified test-set evaluation for the 8 reportable systems.
# Output: results/final_eval/test_final_<system>_results.json
#   Baselines B/C/D/E produce 6-element lists (internal entropy sweep).
#   F/G/H produce 1-element lists at their winning per-ramp threshold vectors.

source scripts/setup_env.sh

OUT_DIR=results/final_eval
mkdir -p "$OUT_DIR"

DATA=data/dev_tokenized.pt
BS=64
TAG=test_final_

ALPHA05_W=results/joint_alpha0.5_weights.pt
BETA10_W=results/system_e_beta1.0_weights.pt

echo "=== [1/8] Baseline A ==="
python -m src.inference --system baseline_a \
    --batch_size $BS --data_path $DATA \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [2/8] Baseline B ==="
python -m src.inference --system baseline_b \
    --batch_size $BS --data_path $DATA \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [3/8] System C ==="
python -m src.inference --system system_c \
    --batch_size $BS --data_path $DATA \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [4/8] System D (alpha=0.5) ==="
python -m src.inference --system system_d \
    --batch_size $BS --data_path $DATA \
    --weights_path "$ALPHA05_W" \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [5/8] System E (beta=1.0, alpha=1.0) ==="
python -m src.inference --system system_e \
    --batch_size $BS --data_path $DATA \
    --weights_path "$BETA10_W" \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [6/8] System F (alpha=0.5, per-ramp 0.001,0.01,0.3,0.5,0.5) ==="
python -m src.inference --system system_f \
    --batch_size $BS --data_path $DATA \
    --weights_path "$ALPHA05_W" \
    --thresholds 0.001,0.01,0.3,0.5,0.5 \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [7/8] System G (beta=1.0 alpha=1.0, per-ramp 0.001,0.01,0.1,0.5,0.003) ==="
python -m src.inference --system system_g \
    --batch_size $BS --data_path $DATA \
    --weights_path "$BETA10_W" \
    --thresholds 0.001,0.01,0.1,0.5,0.003 \
    --output_dir $OUT_DIR --results_tag $TAG

echo "=== [8/8] System H P=2 (beta=1.0 alpha=1.0, per-ramp 0.03,0.1,0.1,0.003,0.001) ==="
python -m src.inference --system system_h \
    --batch_size $BS --data_path $DATA \
    --weights_path "$BETA10_W" \
    --thresholds 0.03,0.1,0.1,0.003,0.001 \
    --patience 2 \
    --output_dir $OUT_DIR --results_tag $TAG

echo "All 8 final-eval runs complete. Results in $OUT_DIR/"
