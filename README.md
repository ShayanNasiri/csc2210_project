# Hardware-Aware Dynamic Batch Compaction for Early-Exit Cross-Encoders

**CSC2210 — GPU Systems Project, University of Toronto**

A Triton kernel that physically compacts batches during early-exit inference in a cross-encoder re-ranker. Exited documents are removed from the batch between transformer layers, eliminating the wasted compute of padding-style early exit. Evaluated on MS MARCO passage ranking (MRR@10) across eight systems spanning two baselines and six early-exit variants.

Full write-up: [`docs/Project_Report/`](docs/Project_Report/).

## Test-set results

MS MARCO passage ranking dev set, batch size 64, RTX 4090.

| System | Description | MRR@10 | Latency (ms) | Speedup | Δ MRR vs A |
|---|---|---:|---:|---:|---:|
| **Baseline A** | Full cross-encoder, no early exit | 0.7332 | 37.29 | 1.00× | — |
| Baseline B | Naive early exit (jagged batch) | 0.0758 | 36.17 | 1.03× | −89.7% |
| System C | Triton-compacted, frozen off-ramps | 0.0758 | 7.17 | 5.20× | −89.7% |
| System D | Joint backbone + off-ramps (α=0.5) | 0.3661 | 17.23 | 2.16× | −50.1% |
| System E | System D + self-distillation (β=1.0, α=1.0) | 0.3754 | 15.47 | 2.41× | −48.8% |
| System F | System D weights + per-ramp entropy thresholds | 0.4556 | 17.44 | 2.14× | −37.9% |
| **System G** | System E weights + per-ramp entropy thresholds | **0.5320** | 18.15 | 2.05× | **−27.4%** |
| System H | System G + patience P=2 | 0.4648 | 17.62 | 2.12× | −36.6% |

System G is the headline result: +42% MRR over System E's scalar-threshold inference using the same weights, preserving the 2× speedup from Triton compaction.

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA support
- PyTorch 2.5.1 (CUDA 12.4)
- Triton ≥ 3.1.0 (Linux only)

## Setup

```bash
git clone https://github.com/ShayanNasiri/csc2210_project.git
cd csc2210_project
bash setup_env.sh
source .venv/bin/activate
```

On Windows, PyTorch pulls a CPU-only wheel by default. Force the CUDA build:

```bash
pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## Reproducing the test-set results

### 1. Data

```bash
python data/download_data.py        # MS MARCO passage ranking
python data/split_train_val.py      # 5% validation carved from train
```

### 2. Train

```bash
# System C (frozen backbone)
python -m src.train_offramps --data_path data/msmarco_train_split.parquet

# System D (joint, α=0.5) — also used by System F
python -m src.train_joint --data_path data/msmarco_train_split.parquet --alpha 0.5

# System E (joint + distillation, α=1.0 β=1.0) — also used by G and H
python -m src.train_joint --data_path data/msmarco_train_split.parquet \
    --alpha 1.0 --beta 1.0 \
    --output_weights_name system_e_alpha1.0_beta1.0_weights.pt
```

### 3. Inference

The canonical end-to-end recipe for all eight systems is in [`scripts/run_final_eval.sh`](scripts/run_final_eval.sh). Example — System G (champion):

```bash
python -m src.inference --system system_g \
    --weights_path results/system_e_alpha1.0_beta1.0_weights.pt \
    --thresholds 0.001,0.01,0.1,0.5,0.003 \
    --batch_size 64 \
    --data_path data/dev_tokenized.pt \
    --output_dir results/test_set_results \
    --results_tag test_final_
```

### 4. Analyze per-ramp threshold sweeps

Raw sweep CSVs under `results/system_{f,g,h}_sweep_results/` are ranked by:

```bash
python scripts/analyze_system_f_sweep.py
python scripts/analyze_system_g_sweep.py
python scripts/analyze_system_h_sweep.py
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project layout

```
src/
  triton_compact.py   # Triton batch-compaction kernel
  inference.py        # Per-system inference drivers (A–H)
  train_offramps.py   # Frozen-backbone off-ramp training
  train_joint.py      # Joint + optional self-distillation training
  model.py            # EarlyExitCrossEncoder wrapper
  offramps.py         # Off-ramp classifier heads
  evaluate.py         # MRR@10 evaluation
scripts/              # Inference runners, sweep drivers, analyzers
tests/                # Pytest suite
data/                 # Data download + train/val split utilities
results/              # Inference JSONs and sweep CSVs (tracked)
  test_set_results/         # Test-set JSONs reported in the paper
  validation_set_results/   # Validation-set JSONs for hyperparameter tuning
docs/Project_Report/  # LaTeX source of the write-up
```
