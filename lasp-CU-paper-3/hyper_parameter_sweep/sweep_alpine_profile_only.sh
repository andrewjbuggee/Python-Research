#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=02:30:00            # Smoke test of run_000 (a slow config: LR=1e-5,
                                   # dropout=0.38) finished in 54 min for K=5.
                                   # 2.5 h gives ~2.7× safety margin for any
                                   # outlier configs that hit early-stop later.
#SBATCH --partition=al40           # Alpine GPU partition (NVIDIA L40)
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU per task; one config = K trainings sequentially
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pinn_sweep_profOnly
#SBATCH --output=logs/sweep_profOnly_%A_%a.out
#SBATCH --error=logs/sweep_profOnly_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-149%6            # 150 configs, max 6 concurrent (al40 has 9 L40s,
                                   # CURC convention is to leave 1+ free).
                                   # At ~55 min/task observed, 6 concurrent ≈ 23 h
                                   # total wall time.  Bump %N if al40 is empty.

# ============================================================
# Profile-only Hyperparameter Sweep — SLURM Job Array
# ============================================================
#
# What this runs:
#   For each task in the array, sweep_train_profile_only.py loads
#   sweep_configs_profile_only/run_<id>.json and trains the
#   ProfileOnlyNetwork with K-fold CV (K read from the config's
#   `hyperparams.n_folds`).  Each task therefore = K independent
#   trainings + a final aggregate (mean ± std test RMSE).
#
# Before submitting:
#   1. Run `python generate_sweep_profile_only.py` locally
#      (defaults: 150 configs, K=5; pass --n-folds N to change).
#   2. Upload project to Alpine:
#      rsync -av --exclude='__pycache__' --exclude='.git' \
#        /Users/andrewbuggee/Documents/VS_CODE/Python-Research/lasp-CU-paper-3/ \
#        anbu8374@login.rc.colorado.edu:/projects/anbu8374/Python-Research/lasp-CU-paper-3/
#   3. Make sure the 50-evenZ-levels HDF5 is at:
#      /scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5
#
# Submit with:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
#   sbatch sweep_alpine_profile_only.sh
#
# Monitor with:
#   squeue -u $USER
#   sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
#
# After all tasks complete, results live in
#   sweep_results_profile_only/run_<id>/summary.json
# (one summary.json per config, with per-fold breakdown + aggregated metrics).
# ============================================================

echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

# Load modules
module load anaconda

# Activate conda environment
conda activate dropProfs_nn

# Navigate to the hyper_parameter_sweep directory.
# sweep_train_profile_only.py adjusts sys.path internally to find
# models_profile_only.py and data.py at the repo root.
cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep

# Create directories
mkdir -p logs sweep_results_profile_only

# Run this task's configuration.
# SLURM_ARRAY_TASK_ID is 0..149.
RUN_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="sweep_configs_profile_only/run_${RUN_ID}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    echo "Did you run generate_sweep_profile_only.py and upload the configs?"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo ""

# --training-data-dir overrides the directory portion of h5_path stored in
# the config JSON, so the same configs work regardless of where the HDF5 lives.
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"

python sweep_train_profile_only.py \
    --config-json "$CONFIG_FILE" \
    --training-data-dir "$TRAINING_DATA_DIR"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE   # propagate failure so SLURM marks the task as FAILED
