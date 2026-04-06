#!/bin/bash
# Create and configure the Python virtual environment on the cluster.
# Sourced by SLURM job scripts — creates the venv if it doesn't exist,
# then activates it and ensures dependencies are installed.

export HF_HOME=/tmp/hf_cache_$USER
export IR_DATASETS_HOME=/tmp/ir_datasets_$USER

VENV_DIR="$HOME/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies if torch is missing (first-time setup)
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies (--no-cache-dir to save disk) ..."
    pip install --no-cache-dir -r requirements.txt
fi
