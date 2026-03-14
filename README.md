# Hardware-Aware Dynamic Batch Compaction for Early-Exit Cross-Encoders

**CSC2210 — GPU Systems Project, University of Toronto**

This project builds a custom Triton kernel to physically compact batches during early-exit inference in a cross-encoder re-ranker. It solves the "jagged batch" problem where exited documents waste GPU compute as padding.

## Architecture

The project compares three systems:

- **Baseline A** — Standard `cross-encoder/ms-marco-MiniLM-L-6-v2` inference (no early exit)
- **Baseline B** — Naive early-exit with off-ramps; exited documents remain in the batch as padding
- **System C** — Triton-compacted early-exit; a custom kernel physically removes exited documents between layers

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA support
- Triton >= 3.2.0 (Linux only — required for Python 3.13 compatibility)

## Setup

### Local Development (Windows/Linux)

```bash
# Clone the repository
git clone https://github.com/ShayanNasiri/csc2210_project.git
cd csc2210_project

# Create virtual environment and install dependencies
bash setup_env.sh

# Activate the environment
source .venv/bin/activate        # Linux/macOS
source .venv/Scripts/activate    # Windows (Git Bash)
```

**Note (Windows):** PyTorch installs as CPU-only by default on Windows. To enable GPU support, reinstall with the CUDA index:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

### SLURM Cluster (UofT CSLab)

The home directory has limited disk quota. Pip cache can fill it — always use `--no-cache-dir`.

```bash
# SSH into the cluster login node
ssh <your-username>@cs.toronto.edu

# Clone the repo
git clone https://github.com/ShayanNasiri/csc2210_project.git
cd csc2210_project

# Create venv (clear pip cache first if you hit quota errors)
pip cache purge && rm -rf ~/.cache/pip
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Create results directory (needed for SLURM log output)
mkdir -p results
```

> **Note:** SLURM commands (`sbatch`, `srun`, `squeue`) must be run from `comps0-3.cs`, not `apps0`. SSH to a compute server first: `ssh comps0.cs`

## Running

### Smoke Test

Validates CUDA, Triton, model loading, and utility functions.

```bash
# Local
python scripts/smoke_test.py

# SLURM cluster (run from comps0.cs, not apps0)
# Batch (output to log file):
sbatch scripts/run_smoke_test.sh
# Monitor: squeue -u $USER
# Check output: cat results/smoke_test_*.log

# Interactive (output streams to terminal):
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 0:15:00 --pty bash scripts/run_smoke_test.sh
```

### Data Preparation (Phase 1)

Downloads MS MARCO passage ranking data and creates train/dev splits.

```bash
# Local (Windows needs UTF-8 mode)
set PYTHONUTF8=1
python data/download_data.py

# SLURM cluster (run from comps0.cs)
sbatch scripts/run_download_data.sh
# Check output: cat results/download_data_*.log

# Validate data files:
python -m pytest tests/test_data.py -v
```

### Baseline A — Standard Inference (Phase 2)

```bash
# Local
python -m src.inference --system baseline_a --batch_size 64

# SLURM
sbatch scripts/run_baseline_a.sh
```

### Off-Ramp Training (Phase 3)

```bash
# Local (quick test)
python -m src.train_offramps --data_path data/msmarco_train.parquet --epochs 1 --max_steps 100

# SLURM (full training)
sbatch scripts/run_train_offramps.sh
```

### Baseline B — Naive Early Exit (Phase 4)

```bash
# Local (single threshold, no GPU timing)
python -m src.inference --system baseline_b --batch_size 32

# SLURM (full threshold sweep)
sbatch scripts/run_baseline_b.sh

# SLURM (sanity check: threshold=0.0 should match Baseline A MRR)
sbatch scripts/run_baseline_b_sanity.sh
```

### System C — Triton-Compacted Early Exit (Phase 6)

```bash
# SLURM
sbatch scripts/run_system_c.sh
```

### Full Profiling Sweep (Phase 7)

```bash
# Latency sweep across all systems, batch sizes, and thresholds
sbatch scripts/run_profiling.sh

# Nsight profiling (isolated micro-benchmark)
sbatch scripts/run_ncu.sh
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
csc2210_project/
├── requirements.txt         # Python dependencies
├── setup_env.sh             # Environment setup script
├── src/
│   ├── __init__.py
│   ├── constants.py         # Project-wide constants and defaults
│   ├── utils.py             # Seeds, timing, device helpers
│   ├── inference_utils.py   # Shared data loading and BatchRunner
│   ├── model.py             # EarlyExitCrossEncoder wrapper
│   ├── offramps.py          # Off-ramp classifier heads
│   ├── triton_compact.py    # Triton batch-compaction kernel
│   ├── inference.py         # Inference drivers (A/B/C)
│   ├── train_offramps.py    # Off-ramp training loop
│   └── evaluate.py          # MRR@10 evaluation
├── scripts/
│   ├── smoke_test.py        # Cluster sanity check
│   ├── run_smoke_test.sh    # SLURM job for smoke test
│   ├── run_baseline_a.sh
│   ├── run_baseline_b.sh
│   ├── run_baseline_b_sanity.sh  # Sanity check: threshold=0.0
│   ├── run_system_c.sh
│   ├── run_train_offramps.sh
│   ├── run_profiling.sh
│   ├── ncu_microbench.py    # Isolated Nsight micro-benchmark
│   └── run_ncu.sh
├── data/
│   ├── download_data.py     # Data download and preprocessing
│   └── README.md            # Data provenance
├── tests/
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_baseline_a.py
│   ├── test_offramps.py
│   ├── test_baseline_b.py
│   ├── test_triton_kernel.py
│   └── test_system_c.py
├── results/                  # Generated at runtime (not in git)
└── notebooks/
    └── plot_pareto.ipynb     # Pareto trade-off plots
```

## Key Technical Details

| Component | Detail |
|-----------|--------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 layers, hidden=384) |
| Off-ramps | 5 linear classifiers after layers 1-5, output shape `(batch,)` |
| Exit criterion | Shannon entropy of sigmoid output < threshold |
| Dataset | MS MARCO passage ranking (train: ~9,971 queries/~977K pairs, dev: 744 queries/~62K pairs) |
| Metric | MRR@10, target ≤5% relative degradation from Baseline A |
| Target GPU | RTX 4090 (24GB VRAM) on UofT CSLab SLURM cluster |
| Local dev GPU | RTX 2060 |
