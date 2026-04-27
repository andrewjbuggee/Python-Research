#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=01:00:00                         # K=5 folds × ~10 min/fold + margin
#SBATCH --partition=atesting_a100               # Testing partition
#SBATCH --qos=testing
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4a
#SBATCH --job-name=nn_test_profOnly
#SBATCH --output=logs/sweep_test_profile_only_%j.out
#SBATCH --error=logs/sweep_test_profile_only_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ============================================================
# Single-config K-fold smoke test for the profile-only sweep
# ============================================================
# Runs run_000 from sweep_configs_profile_only/ — that one config
# trains n_folds independent models and reports test_mean_rmse ± std.
#
# Submit with:  sbatch sweep_test_one_profile_only.sh
# Monitor with: squeue -u $USER
#
# Before submitting:
#   1. python generate_sweep_profile_only.py  (locally)
#   2. Upload project + sweep_configs_profile_only/ to Alpine
#   3. Confirm the 50-evenZ-levels HDF5 is on /scratch/alpine
# ============================================================

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

# Load modules
module load anaconda
conda activate dropProfs_nn

# Navigate to the hyper_parameter_sweep directory.
# sweep_train_profile_only.py adjusts sys.path internally so it can find
# models_profile_only.py and data.py at the repo root.
cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep

# Create directories.  output_dir in the JSON config is
# sweep_results_profile_only — sweep_train_profile_only.py creates it.
mkdir -p logs sweep_results_profile_only

# Verify HDF5 data file exists (50-evenZ-levels target)
H5_FILE="/scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5"
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: HDF5 file not found at $H5_FILE"
    echo "Upload it first with:"
    echo "  scp <local_path> anbu8374@login.rc.colorado.edu:$H5_FILE"
    exit 1
fi
echo "HDF5 file found: $(ls -lh $H5_FILE)"

# Verify Python imports work.
# models_profile_only.py and data.py live at the repo root, one level up.
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
echo ""
echo "Checking Python imports (PYTHONPATH=$PYTHONPATH)..."
python -c "
from models_profile_only import ProfileOnlyNetwork, ProfileOnlyLoss
from models import RetrievalConfig
from data import create_kfold_dataloaders, resolve_h5_path
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
print('All imports OK')
" || { echo "ERROR: Python import check failed"; exit 1; }

# Run a single sweep config (run_000 from the profile-only sweep)
echo ""
echo "Running sweep config run_000 (profile-only, K-fold CV)..."
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"

python sweep_train_profile_only.py \
    --config-json sweep_configs_profile_only/run_000.json \
    --training-data-dir "$TRAINING_DATA_DIR"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "TEST PASSED — safe to submit the full sweep with:"
    echo "  sbatch sweep_alpine_profile_only.sh"
else
    echo ""
    echo "TEST FAILED — check logs/sweep_test_profile_only_${SLURM_JOB_ID}.err for details"
fi

exit $EXIT_CODE
