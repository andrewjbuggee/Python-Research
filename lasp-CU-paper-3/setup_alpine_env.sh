#!/bin/bash
# ============================================================
# Setup conda environment on Alpine
# ============================================================
# IMPORTANT: Run this from a compile node or interactive job,
# NOT from a login node.  The anaconda module is only available
# on compute/compile nodes.
#
#   Option A (compile node):
#     acompile
#     cd /projects/anbu8374/paper3
#     bash setup_alpine_env.sh
#
#   Option B (interactive GPU session):
#     sinteractive --partition=atesting_a100 --gres=gpu:1 \
#       --time=01:00:00 --account=ucb762_asc1 --qos=testing
#     cd /projects/anbu8374/paper3
#     bash setup_alpine_env.sh
#
# This creates the dropProfs_nn environment with CUDA-enabled
# PyTorch for the A100 GPUs.
# ============================================================

set -e  # exit on any error

echo "============================================"
echo "Setting up dropProfs_nn environment on Alpine"
echo "============================================"

# Check we're not on a login node
if hostname | grep -q "login"; then
    echo ""
    echo "WARNING: You appear to be on a login node ($(hostname))."
    echo "The anaconda module is only available on compute/compile nodes."
    echo ""
    echo "Please run one of:"
    echo "  acompile                          # get a compile node"
    echo "  sinteractive --partition=atesting_a100 --gres=gpu:1 \\"
    echo "    --time=01:00:00 --account=ucb762_asc1 --qos=testing"
    echo ""
    echo "Then re-run: bash setup_alpine_env.sh"
    exit 1
fi

# ── Load the anaconda module ───────────────────────────────────────────────
module load anaconda

echo "Conda loaded: $(which conda)"
echo "Conda version: $(conda --version)"

# ── Ensure .condarc is configured ──────────────────────────────────────────
# Store environments and package cache in /projects (not $HOME, which has
# a small quota on Alpine).
mkdir -p /projects/$USER/.conda_pkgs
mkdir -p /projects/$USER/software/anaconda/envs

CONDARC="$HOME/.condarc"
if [ ! -f "$CONDARC" ] || ! grep -q "pkgs_dirs" "$CONDARC" 2>/dev/null; then
    echo "Configuring .condarc..."
    cat > "$CONDARC" << EOF
pkgs_dirs:
  - /projects/$USER/.conda_pkgs
envs_dirs:
  - /projects/$USER/software/anaconda/envs
EOF
fi

# ── Create the environment ─────────────────────────────────────────────────
echo ""
echo "Creating conda environment: dropProfs_nn (Python 3.11)..."
conda create -n dropProfs_nn python=3.11 -y

# ── Activate it ────────────────────────────────────────────────────────────
conda activate dropProfs_nn

# Make sure pip is installed inside the env (CURC docs recommend this
# to avoid accidentally installing into the base environment)
conda install pip -y

# ── Install PyTorch with CUDA 12.1 support ─────────────────────────────────
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Install remaining dependencies ─────────────────────────────────────────
echo ""
echo "Installing remaining packages..."
pip install h5py numpy scipy matplotlib pyyaml

# ── Verify installation ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'CUDA version:    {torch.version.cuda}')

import h5py, numpy, scipy, matplotlib, yaml
print(f'h5py:       {h5py.__version__}')
print(f'numpy:      {numpy.__version__}')
print(f'scipy:      {scipy.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print('All packages installed successfully!')
"

echo ""
echo "============================================"
echo "Environment ready! Now submit jobs with:"
echo "  sbatch sweep_test_one.sh"
echo "============================================"
