#!/bin/bash
# =============================================================
# AdaGuard HPC Environment Setup — Matilda Cluster
# Run this ONCE after first SSH login:
#   bash hpc/setup_env.sh
# =============================================================

set -e

echo "=== AdaGuard HPC Environment Setup ==="

# Project paths
PROJECT_DIR="/projects/secure-distributed-ml"
SCRATCH_DIR="/scratch/projects/secure-distributed-ml"
VENV_DIR="$PROJECT_DIR/venv"

# Create working directories
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR/data"
mkdir -p "$SCRATCH_DIR/logs"

echo "[1/4] Loading modules..."
module load Python/3.10.14
module load CUDA/12.4

echo "[2/4] Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
else
    echo "  Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib seaborn pyyaml streamlit plotly scipy

echo "[4/4] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import scipy
print(f'SciPy {scipy.__version__}')
print('All dependencies OK.')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate the env in future sessions with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Project space: $PROJECT_DIR"
echo "Scratch space: $SCRATCH_DIR"
echo "Results will be saved to: $SCRATCH_DIR/results"
