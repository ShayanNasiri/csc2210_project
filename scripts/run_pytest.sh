#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=gpunode7,gpunode13,gpunode33
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=results/pytest_%j.log

# Exclude list (verified 2026-04-11):
#   gpunode7, gpunode13 — historically RTX A4500 nodes (cuBLAS silently
#                          corrupts joint-trained weights; see
#                          memory/project_cluster_gpu_constraint.md).
#                          --gres=gpu:rtx_4090:1 already filters them, but
#                          explicit exclude is defense-in-depth.
#   gpunode33           — drained by admin for system disk failure 2026-04-10.
# gpunode4 and gpunode5 dropped from the exclude list (graduated to trusted
# rtx_4090 set 2026-04-10 via bit-equality proofs).

source scripts/setup_env.sh

python -m pytest tests/ -v --tb=short
