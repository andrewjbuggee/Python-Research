#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=02:00:00            # K=5 folds × ~5–7 min/fold for the 10-level
                                   # output.  2 h gives a comfortable margin.
#SBATCH --partition=al40           # NVIDIA L40 GPU partition
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nn_top10_10lvl
#SBATCH --output=logs/top10_10level_%A_%a.out
#SBATCH --error=logs/top10_10level_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-9%6              # 10 tasks (top-10 configs), max 6 concurrent

# ============================================================
# Top-10 10-level configs retrained as profile-only with K=5 CV  (HySICS)
# ============================================================
#
# Re-trains the top-10 hyperparameter configs from the original 10-level sweep
# (sweep_results_2/) using the new ProfileOnlyNetwork pipeline:
#   • profile-only model (no τ head, no τ NLL term)
#   • K=5 profile-aware K-fold CV (mean ± std headline metric)
#   • HySICS spectra (0.3 % noise) — same instrument as in the original sweep,
#     so RMSE shifts are attributable purely to the model architecture change
#     and the K-fold uncertainty quantification.
#
# Why this experiment matters
# ---------------------------
# Together with the 7-level run (sweep_alpine_profile_only_top10_7level.sh)
# and the existing 50-level K=5 sweep, this lets you compare retrieval RMSE
# at three different output resolutions trained from the same in-situ catalog.
# That isolates the effect of the low-pass-filter step (raw in-situ → 7/10/50
# evenly-spaced levels) on the model-recoverable information.
#
# NOTE on sweep numbering
# -----------------------
# The original 10-level sweep results live in sweep_results_2/ (NOT
# sweep_configs_2/, which is the 7-level configs).  The top-10 run IDs
# below are read from sweep_results_2/run_*/results.json sorted by mean_rmse.
# The original configs lived in sweep_configs/ — that's the directory this
# script reads from.
#
# Top-10 10-level run IDs (from sweep_results_2, ranked by test_metrics.mean_rmse):
#     rank  run_id  mean_rmse (μm)
#       1     095         1.019
#       2     085         1.023
#       3     023         1.024
#       4     063         1.033
#       5     032         1.035
#       6     088         1.043
#       7     034         1.059
#       8     099         1.062
#       9     003         1.063
#      10     050         1.066
#
# IMPORTANT: the original 10-level sweep used
#     combined_vocals_oracles_training_data_13_April_2026.h5
# but the most recent 10-level evenZ resampling lives in
#     combined_vocals_oracles_training_data_10-evenZ-levels_19_April_2026.h5
# This script uses the *evenZ* file because (a) it matches the resampling
# convention used in the 50-level K=5 sweep and (b) it's the one currently
# on Alpine.  The hyperparameters from the older sweep transfer cleanly —
# only the training-target grid changed.
#
# Submit with:  sbatch sweep_alpine_profile_only_top10_10level.sh
# Monitor with: squeue -u $USER
# ============================================================

set -u
# DO NOT set -e: if one config's training fails, others should still run.

# ── Map SLURM array task ID → top-10 run ID ─────────────────────────────────
TOP10_RUN_IDS=(95 85 23 63 32 88 34 99 3 50)
RUN_ID=${TOP10_RUN_IDS[$SLURM_ARRAY_TASK_ID]}

REPO_ROOT="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
H5_FILE="${TRAINING_DATA_DIR}combined_vocals_oracles_training_data_10-evenZ-levels_19_April_2026.h5"
SOURCE_SWEEP_CONFIGS="sweep_configs"                    # original 10-level configs
                                                        # (corresponds to sweep_results_2)
OUTPUT_DIR_BASE="sweep_results_profile_only_10level"    # parallel to other profile-only outputs

# ── Banner ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID  (one of 0..$((${#TOP10_RUN_IDS[@]} - 1)))"
echo "Run ID:        $RUN_ID  (top-10 rank $((SLURM_ARRAY_TASK_ID + 1)))"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"
echo "Profile grid:  10 levels (evenZ resampling)"
echo "Instrument:    HySICS  (0.3% noise)"
echo "HDF5 file:     $H5_FILE"
echo "Source sweep:  ${REPO_ROOT}/hyper_parameter_sweep/${SOURCE_SWEEP_CONFIGS}/run_$(printf '%03d' $RUN_ID).json"
echo "Output dir:    ${REPO_ROOT}/hyper_parameter_sweep/${OUTPUT_DIR_BASE}/run_$(printf '%03d' $RUN_ID)/"
echo "============================================"

# ── Module + env ────────────────────────────────────────────────────────────
module load anaconda
conda activate dropProfs_nn

cd "${REPO_ROOT}/hyper_parameter_sweep"
mkdir -p logs "$OUTPUT_DIR_BASE"

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: HDF5 file not found at $H5_FILE"
    echo "  If only the older 10-level file is on Alpine, edit H5_FILE above."
    exit 1
fi

CONFIG_FILE="${SOURCE_SWEEP_CONFIGS}/run_$(printf '%03d' $RUN_ID).json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: source config not found: $CONFIG_FILE"
    echo "  Did you upload sweep_configs/ (the original 10-level configs) to Alpine?"
    exit 1
fi
echo "Config file:   $CONFIG_FILE"

# Verify the HDF5 schema and profile shape match what the config expects (10 levels)
python -c "
import h5py, json
cfg = json.load(open('${CONFIG_FILE}'))
n_lw = len(cfg['hyperparams']['level_weights'])
with h5py.File('${H5_FILE}', 'r') as f:
    n_lev_h5 = f['profiles'].shape[1]
    for k in ('reflectances_hysics', 'sza', 'vza', 'saz', 'vaz'):
        assert k in f, f'HDF5 missing /{k}'
    assert n_lev_h5 == n_lw, (
        f'HDF5 has {n_lev_h5} profile levels but config has {n_lw} level_weights')
print(f'  HDF5 profile levels: {n_lev_h5}  (matches level_weights length)')
print(f'  HDF5 has all required datasets for profile-only HySICS training.')
" || { echo "ERROR: HDF5 / config compatibility check failed"; exit 1; }

# ── Run the K-fold trainer with the 10-level config + HySICS instrument ────
echo
python sweep_train_profile_only.py \
    --config-json "$CONFIG_FILE" \
    --h5-path "$H5_FILE" \
    --training-data-dir "$TRAINING_DATA_DIR" \
    --instrument hysics \
    --output-dir "$OUTPUT_DIR_BASE"

EXIT_CODE=$?

echo
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "Run ID:        $RUN_ID  (rank $((SLURM_ARRAY_TASK_ID + 1)) of top 10, 10-level)"
echo "Results in:    ${REPO_ROOT}/hyper_parameter_sweep/${OUTPUT_DIR_BASE}/run_$(printf '%03d' $RUN_ID)/"
echo "============================================"

exit $EXIT_CODE
