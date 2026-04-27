#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=02:30:00            # K=5 folds × ~10 min/fold + early-stop margin.
                                   # Same per-task budget as the HySICS sweep.
#SBATCH --partition=al40           # NVIDIA L40 GPU partition
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nn_top10_emit
#SBATCH --output=logs/top10_emit_%A_%a.out
#SBATCH --error=logs/top10_emit_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-9%6              # 10 tasks (top-10 configs), max 6 concurrent
                                   # (al40 has 9 L40s; CURC convention is to
                                   # leave at least 1 free).

# ============================================================
# Top-10 profile-only configs retrained on simulated EMIT spectra
# ============================================================
#
# This SLURM array re-runs the top-10 hyperparameter configs from the K=5
# HySICS sweep (sweep_results_profile_only/) but reads /reflectances_emit
# from the same HDF5 instead of /reflectances_hysics.  The EMIT spectra
# already have 2 % Gaussian noise baked in (vs HySICS's 0.3 %), so this
# tells you how much that 6.7× noise increase costs you in retrieval RMSE
# at otherwise-identical settings.
#
# Each task does its own K=5 K-fold CV, just like the original sweep, so
# the EMIT results are directly comparable to the HySICS numbers
# (same hyperparameters, same profile-aware splits, same n_test_profiles).
#
# Top-10 run IDs (from the HySICS sweep, ranked by test_mean_rmse_mean):
#     rank  run_id  mean RMSE (μm)  ± std
#       1     110         1.405     0.034
#       2      19         1.406     0.027
#       3     149         1.408     0.037
#       4       9         1.412     0.048
#       5      97         1.414     0.081
#       6      52         1.417     0.032
#       7       0         1.419     0.062
#       8      63         1.420     0.043
#       9     100         1.422     0.051
#      10      46         1.424     0.038
#
# Submit with:  sbatch sweep_alpine_profile_only_emit.sh
# Monitor with: squeue -u $USER
# ============================================================

set -u
# DO NOT set -e: if one config's training fails, others should still run.

# ── Map SLURM array task ID → top-10 run ID ─────────────────────────────────
TOP10_RUN_IDS=(110 19 149 9 97 52 0 63 100 46)
RUN_ID=${TOP10_RUN_IDS[$SLURM_ARRAY_TASK_ID]}

REPO_ROOT="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
H5_FILE="${TRAINING_DATA_DIR}combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5"
SWEEP_CONFIGS="sweep_configs_profile_only"          # relative to hyper_parameter_sweep/
OUTPUT_DIR_BASE="sweep_results_profile_only_emit"   # parallel to sweep_results_profile_only/

# ── Banner ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID  (one of 0..${#TOP10_RUN_IDS[@]} - 1)"
echo "Run ID:        $RUN_ID  (top-10 rank $((SLURM_ARRAY_TASK_ID + 1)))"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"
echo "Instrument:    EMIT  (2% noise; replaces HySICS 0.3% from original sweep)"
echo "HDF5 file:     $H5_FILE"
echo "Output dir:    ${REPO_ROOT}/hyper_parameter_sweep/${OUTPUT_DIR_BASE}/run_$(printf '%03d' $RUN_ID)/"
echo "============================================"

# ── Module + env ────────────────────────────────────────────────────────────
module load anaconda
conda activate dropProfs_nn

# Navigate to the sweep dir.  sweep_train_profile_only.py adjusts sys.path
# internally to import models_profile_only / data from the repo root.
cd "${REPO_ROOT}/hyper_parameter_sweep"
mkdir -p logs "$OUTPUT_DIR_BASE"

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: HDF5 file not found at $H5_FILE"
    exit 1
fi

CONFIG_FILE="${SWEEP_CONFIGS}/run_$(printf '%03d' $RUN_ID).json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    echo "  Did you upload sweep_configs_profile_only/ to Alpine?"
    exit 1
fi
echo "Config file:   $CONFIG_FILE"

# Quick check that the EMIT reflectance dataset is actually in the HDF5
# (the convert script writes both, but be paranoid — failure mode here is a
#  KeyError 30 seconds into training, after the GPU is already allocated).
python -c "
import h5py
with h5py.File('${H5_FILE}', 'r') as f:
    assert 'reflectances_emit' in f, 'HDF5 missing /reflectances_emit'
    print(f\"  /reflectances_emit shape: {f['reflectances_emit'].shape}\")
    print(f\"  /reflectances_uncertainty_emit shape: {f['reflectances_uncertainty_emit'].shape}\")
" || { echo "ERROR: EMIT reflectances missing from HDF5"; exit 1; }

# ── Run the K-fold trainer with --instrument emit and a separate output dir ─
echo
python sweep_train_profile_only.py \
    --config-json "$CONFIG_FILE" \
    --training-data-dir "$TRAINING_DATA_DIR" \
    --instrument emit \
    --output-dir "$OUTPUT_DIR_BASE"

EXIT_CODE=$?

echo
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "Run ID:        $RUN_ID  (rank $((SLURM_ARRAY_TASK_ID + 1)) of top 10)"
echo "Results in:    ${REPO_ROOT}/hyper_parameter_sweep/${OUTPUT_DIR_BASE}/run_$(printf '%03d' $RUN_ID)/"
echo "============================================"

exit $EXIT_CODE
