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
- Triton (Linux only — used on the SLURM cluster)

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

```bash
# SSH into the cluster
ssh <your-username>@cs.toronto.edu

# Clone and set up
git clone https://github.com/ShayanNasiri/csc2210_project.git
cd csc2210_project
bash setup_env.sh
mkdir -p results
```

## Running

### Smoke Test

Validates CUDA, Triton, model loading, and utility functions.

```bash
# Local
python scripts/smoke_test.py

# SLURM cluster
sbatch scripts/run_smoke_test.sh
# Monitor: squeue -u $USER
# Check output: cat results/smoke_test_*.log
```

### Data Preparation (Phase 1)

Downloads MS MARCO passage ranking data and creates train/dev splits.

```bash
python data/download_data.py
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
# SLURM
sbatch scripts/run_baseline_b.sh
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
├── IDEA.md                 # Project proposal
├── EXECUTIONPLAN.md         # 8-phase implementation plan
├── TODO.md                  # Development checklists
├── SLURM.md                 # Cluster usage guide
├── requirements.txt         # Python dependencies
├── setup_env.sh             # Environment setup script
├── src/
│   ├── __init__.py
│   ├── utils.py             # Seeds, timing, device helpers
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
│   ├── run_system_c.sh
│   ├── run_train_offramps.sh
│   ├── run_profiling.sh
│   ├── ncu_microbench.py    # Isolated Nsight micro-benchmark
│   └── run_ncu.sh
├── data/
│   ├── download_data.py     # Data download and preprocessing
│   └── README.md            # Data provenance
├── tests/
│   ├── test_offramps.py
│   ├── test_triton_kernel.py
│   └── test_inference.py
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
| Dataset | MS MARCO passage ranking (train: 10K queries/~1M pairs, dev: 1K queries/100K pairs) |
| Metric | MRR@10, target ≤5% relative degradation from Baseline A |
| Target GPU | RTX 4090 (24GB VRAM) on UofT CSLab SLURM cluster |
| Local dev GPU | RTX 2060 |
