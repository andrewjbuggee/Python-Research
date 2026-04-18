#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=atesting_a100               # Testing partition
#SBATCH --qos=testing
#SBATCH --mem=8G                                # Sweep 1 peaked at 1.5G; 8G is plenty
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pinn_test
#SBATCH --output=logs/sweep_test_%j.out
#SBATCH --error=logs/sweep_test_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ============================================================
# Single-run test on atesting_a100
# ============================================================
# Submit with:  sbatch sweep_test_one.sh
# Monitor with: squeue -u $USER
# ============================================================

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

# Load modules
# Load anaconda (only available on compute/compile nodes, not login nodes)
module load anaconda

# Activate conda environment
conda activate dropProfs_nn

# Navigate to project directory
cd /projects/anbu8374/Python-Research/lasp-CU-paper-3

# Create directories
mkdir -p logs sweep_results_test

# Verify HDF5 data file exists
H5_FILE="/scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_7-levels_17_April_2026.h5"
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: HDF5 file not found at $H5_FILE"
    echo "Upload it first with:"
    echo "  scp <local_path> anbu8374@login.rc.colorado.edu:$H5_FILE"
    exit 1
fi
echo "HDF5 file found: $(ls -lh $H5_FILE)"

# Verify Python imports work
echo ""
echo "Checking Python imports..."
python -c "
from models import DropletProfileNetwork, CombinedLoss, RetrievalConfig
from data import create_dataloaders
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
print('All imports OK')
" || { echo "ERROR: Python import check failed"; exit 1; }

# Run a single sweep config (run_000 from sweep #2)
echo ""
echo "Running sweep config run_000 (sweep 2)..."
python sweep_train.py --config-json sweep_configs_2/run_000.json

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "TEST PASSED — safe to submit the full sweep with:"
    echo "  sbatch sweep_alpine.sh"
else
    echo ""
    echo "TEST FAILED — check logs/sweep_test_${SLURM_JOB_ID}.err for details"
fi
