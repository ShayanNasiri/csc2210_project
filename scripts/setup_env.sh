#!/bin/bash
# Create and configure a temporary Python venv on the compute node.
# Sourced by SLURM job scripts. Builds in /tmp to avoid home quota issues
# and ensures the correct CUDA-enabled PyTorch is installed.

export HF_HOME=/tmp/hf_cache_$USER
export IR_DATASETS_HOME=/tmp/ir_datasets_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

VENV_DIR=/tmp/venv_$USER

# Rebuild venv if torch is missing or broken
if [ ! -d "$VENV_DIR" ] || ! "$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Creating virtual environment at $VENV_DIR ..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing PyTorch with CUDA 12.4 ..."
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    echo "Installing remaining dependencies ..."
    # Constraint file prevents pip from upgrading torch when installing deps
    echo "torch==2.5.1" > /tmp/torch_constraint_$USER.txt
    pip install --no-cache-dir -r requirements.txt -c /tmp/torch_constraint_$USER.txt
else
    source "$VENV_DIR/bin/activate"
fi

# Verify CUDA is accessible
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check PyTorch/driver compatibility'; print(f'GPU ready: {torch.cuda.get_device_name(0)}')"
