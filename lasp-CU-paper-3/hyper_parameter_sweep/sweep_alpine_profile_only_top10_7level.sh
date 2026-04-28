#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=02:00:00            # K=5 folds × ~5 min/fold for the 7-level
                                   # output (smaller than 50-level → faster).
                                   # 2 h gives a comfortable margin.
#SBATCH --partition=al40           # NVIDIA L40 GPU partition
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nn_top10_7lvl
#SBATCH --output=logs/top10_7level_%A_%a.out
#SBATCH --error=logs/top10_7level_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-9%6              # 10 tasks (top-10 configs), max 6 concurrent

# ============================================================
# Top-10 7-level configs retrained as profile-only with K=5 CV  (HySICS)
# ============================================================
#
# Re-trains the top-10 hyperparameter configs from the original 7-level sweep
# (sweep_results_3/) using the *new* ProfileOnlyNetwork pipeline:
#   • profile-only model (no τ head, no τ NLL term)
#   • K=5 profile-aware K-fold CV (mean ± std headline metric)
#   • HySICS spectra (0.3 % noise) — the same instrument used in the original
#     7-level sweep, so any RMSE shift here is attributable purely to (a) the
#     model architecture change and (b) the K-fold uncertainty quantification.
#
# Why this experiment matters
# ---------------------------
# The 7-level training data is a low-pass-filtered version of the in-situ
# profiles (sampling RMSE ≈ 1 μm from the resampling itself).  Comparing this
# run's mean test RMSE to the 50-level profile-only sweep's 1.405 μm baseline
# tells you how much of your retrieval error is *model-recoverable* vs how
# much is just sampling error baked into the training target.
#
# Top-10 7-level run IDs (from sweep_results_3, ranked by test_metrics.mean_rmse):
#     rank  run_id  mean_rmse (μm)
#       1     050         1.591
#       2     001         1.620
#       3     044         1.625
#       4     045         1.628
#       5     031         1.629
#       6     026         1.637
#       7     057         1.638
#       8     006         1.639
#       9     082         1.640
#      10     005         1.641
#
# Submit with:  sbatch sweep_alpine_profile_only_top10_7level.sh
# Monitor with: squeue -u $USER
# ============================================================

set -u
# DO NOT set -e: if one config's training fails, others should still run.

# ── Map SLURM array task ID → top-10 run ID ─────────────────────────────────
TOP10_RUN_IDS=(50 1 44 45 31 26 57 6 82 5)
RUN_ID=${TOP10_RUN_IDS[$SLURM_ARRAY_TASK_ID]}

REPO_ROOT="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
H5_FILE="${TRAINING_DATA_DIR}combined_vocals_oracles_training_data_7-levels_17_April_2026.h5"
SOURCE_SWEEP_CONFIGS="sweep_configs_2"                  # 7-level configs
                                                        # (corresponds to sweep_results_3)
OUTPUT_DIR_BASE="sweep_results_profile_only_7level"     # parallel to other profile-only outputs

# ── Banner ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID  (one of 0..$((${#TOP10_RUN_IDS[@]} - 1)))"
echo "Run ID:        $RUN_ID  (top-10 rank $((SLURM_ARRAY_TASK_ID + 1)))"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"
echo "Profile grid:  7 levels"
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
    exit 1
fi

CONFIG_FILE="${SOURCE_SWEEP_CONFIGS}/run_$(printf '%03d' $RUN_ID).json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: source config not found: $CONFIG_FILE"
    echo "  Did you upload sweep_configs_2/ (the 7-level configs) to Alpine?"
    exit 1
fi
echo "Config file:   $CONFIG_FILE"

# Verify the HDF5 schema and profile shape match what the config expects (7 levels)
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

# ── Run the K-fold trainer with the 7-level config + HySICS instrument ─────
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
echo "Run ID:        $RUN_ID  (rank $((SLURM_ARRAY_TASK_ID + 1)) of top 10, 7-level)"
echo "Results in:    ${REPO_ROOT}/hyper_parameter_sweep/${OUTPUT_DIR_BASE}/run_$(printf '%03d' $RUN_ID)/"
echo "============================================"

exit $EXIT_CODE
