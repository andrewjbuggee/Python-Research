#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=08:30:00                         # K=21 sequential + importance sweep
                                                # ~3.5h for run_110-style configs
                                                # (~10 min/fold);  bump if reusing a
                                                # config with a smaller LR (e.g. run_149
                                                # was ~27 min/fold).
#SBATCH --partition=al40                        # NVIDIA L40 GPU partition
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nn_kfold_fullcov
#SBATCH --output=logs/kfold_fullcov_%j.out
#SBATCH --error=logs/kfold_fullcov_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ============================================================
# Full-coverage K-fold standalone train + spectral feature-importance sweep
# ============================================================
#
# This script does TWO sequential things on one al40 GPU:
#
#   STEP 1: train_kfold_full_coverage_profile_only.py  (K folds back-to-back)
#           Trains the ProfileOnlyNetwork K=N_FOLDS times so that every one
#           of the 290 unique droplet profiles ends up in test EXACTLY ONCE.
#           Aggregates into a per-profile RMSE table + correlations against
#           tau_c, water vapor, adiabaticity, drizzle.
#
#   STEP 2: compute_spectral_feature_importance.py
#           For each fold's best checkpoint, runs mean-substitution
#           perturbation on each of the 636 spectral channels and produces
#           a "relative importance vs wavelength" plot averaged across folds.
#
# Step 2 only runs if Step 1 completed cleanly (RC == 0).
#
# Submit with:  sbatch run_kfold_full_coverage_alpine.sh
# Monitor with: squeue -u $USER
#
# Before submitting:
#   1. Run the chosen sweep config first (sweep_results_profile_only/run_<ID>/
#      must exist on Alpine — usually already there if you uploaded the sweep
#      results).
#   2. Confirm the 50-evenZ-levels HDF5 is on /scratch/alpine.
#   3. Edit RUN_ID below.
# ============================================================

set -u                            # error on undefined vars
# DO NOT set -e: STEP 2 should still attempt if STEP 1 partially failed.

# ── User-editable configuration ─────────────────────────────────────────────
RUN_ID=110                         # sweep config to reuse (run_<ID>)
N_FOLDS=21                         # 21 covers all 290 profiles with 13–14 per fold
N_VAL_PROFILES=14
N_SAMPLES_IMPORTANCE=5000          # subsample size for the perturbation sweep
                                   # (1000 ≈ 5 min/fold on L40; full ~7300 ≈ 35 min)

REPO_ROOT="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
H5_FILE="${TRAINING_DATA_DIR}combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5"
SWEEP_DIR="hyper_parameter_sweep/sweep_results_profile_only"   # relative to REPO_ROOT
RESULTS_PARENT="${REPO_ROOT}/standalone_results_profile_only_kfold"

# Date-stamped suite dir so re-runs of this script don't overwrite previous outputs.
SUITE_TAG="run$(printf "%03d" $RUN_ID)_K${N_FOLDS}_$(date +%Y%m%d_%H%M)"
SUITE_DIR="${RESULTS_PARENT}/${SUITE_TAG}"
mkdir -p "${SUITE_DIR}"

# ── Banner ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"
echo "Run ID (sweep config to reuse):  $RUN_ID"
echo "K folds (full-coverage):         $N_FOLDS"
echo "Importance subsample size:       $N_SAMPLES_IMPORTANCE samples/fold"
echo "Repo root:                       $REPO_ROOT"
echo "HDF5 file:                       $H5_FILE"
echo "Suite output dir:                $SUITE_DIR"
echo "============================================"

# ── Module + env ────────────────────────────────────────────────────────────
module load anaconda
conda activate dropProfs_nn

cd "$REPO_ROOT"
mkdir -p hyper_parameter_sweep/logs

# ── Pre-flight checks (fail fast before any GPU work) ───────────────────────
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: HDF5 file not found at $H5_FILE"
    echo "  Upload it first: scp <local_path> anbu8374@login.rc.colorado.edu:$H5_FILE"
    exit 1
fi

CFG_PATH="${REPO_ROOT}/${SWEEP_DIR}/run_$(printf "%03d" $RUN_ID)/config.json"
if [ ! -f "$CFG_PATH" ]; then
    echo "ERROR: sweep config not found at $CFG_PATH"
    echo "  Did you upload the sweep_results_profile_only/ tree to Alpine?"
    exit 1
fi
echo "Sweep config in:  $CFG_PATH"
echo "HDF5 in:          $(ls -lh $H5_FILE | awk '{print $5, $9}')"

echo
echo "Checking Python imports…"
python -c "
from models_profile_only import ProfileOnlyNetwork, ProfileOnlyLoss
from models              import RetrievalConfig
from data                import (LibRadtranDataset,
                                 create_rotating_kfold_splits,
                                 create_rotating_kfold_dataloaders,
                                 resolve_h5_path)
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print('  imports OK')
" || { echo "ERROR: Python import check failed"; exit 1; }

# ============================================================
# STEP 1: full-coverage K-fold training + per-profile aggregation
# ============================================================
echo
echo "============================================"
echo " STEP 1 — train_kfold_full_coverage_profile_only.py"
echo " Start: $(date)"
echo "============================================"

python train_kfold_full_coverage_profile_only.py \
    --run-id "$RUN_ID" \
    --h5-path "$H5_FILE" \
    --training-data-dir "$TRAINING_DATA_DIR" \
    --sweep-dir "$SWEEP_DIR" \
    --output-dir "$SUITE_DIR" \
    --n-folds "$N_FOLDS" \
    --n-val-profiles "$N_VAL_PROFILES"

RC_TRAIN=$?
if [ $RC_TRAIN -eq 0 ]; then
    echo "  >>> STEP 1 COMPLETED  (exit 0,  $(date))"
else
    echo "  >>> STEP 1 FAILED     (exit $RC_TRAIN, $(date))"
    echo "      STEP 2 will still attempt — if any fold_NN/ subdirs"
    echo "      contain best_model.pt, importance can run on those."
fi

# ============================================================
# STEP 2: per-wavelength feature importance
# ============================================================
# This step is intentionally NOT gated on $RC_TRAIN: if STEP 1 timed out
# part-way through K=21 but, say, 18 folds completed, we still want
# importance for those 18 folds.  The Python script auto-detects which
# fold_NN/ directories have a best_model.pt.

# Quick sanity: are there any fold_NN/best_model.pt files at all?
N_FOLDS_DONE=$(find "$SUITE_DIR" -maxdepth 2 -type f -name 'best_model.pt' \
                  -path '*/fold_*' | wc -l)
echo
echo "============================================"
echo " STEP 2 — compute_spectral_feature_importance.py"
echo " Start: $(date)"
echo " Folds with best_model.pt to analyze: $N_FOLDS_DONE / $N_FOLDS"
echo "============================================"

if [ "$N_FOLDS_DONE" -lt 1 ]; then
    echo "  No fold checkpoints found in $SUITE_DIR — skipping STEP 2."
    RC_IMPORT=2
else
    python compute_spectral_feature_importance.py \
        --output-dir "$SUITE_DIR" \
        --h5-path "$H5_FILE" \
        --training-data-dir "$TRAINING_DATA_DIR" \
        --n-folds "$N_FOLDS" \
        --n-val-profiles "$N_VAL_PROFILES" \
        --n-samples "$N_SAMPLES_IMPORTANCE"
    RC_IMPORT=$?
    if [ $RC_IMPORT -eq 0 ]; then
        echo "  >>> STEP 2 COMPLETED  (exit 0,  $(date))"
    else
        echo "  >>> STEP 2 FAILED     (exit $RC_IMPORT, $(date))"
    fi
fi

# ── Final summary ───────────────────────────────────────────────────────────
echo
echo "============================================"
echo " RUN COMPLETE"
echo "============================================"
echo " End time: $(date)"
echo " STEP 1 (K-fold train + aggregate)    : exit $RC_TRAIN"
echo " STEP 2 (spectral feature importance) : exit $RC_IMPORT"
echo
echo " Suite results in: $SUITE_DIR"
echo "   per-profile  : per_profile_summary.csv, per_profile_correlations.json"
echo "   importance   : spectral_feature_importance.csv"
echo "   figures/     : per_profile_rmse_distribution.png, rmse_vs_predictors.png,"
echo "                  per_level_uncertainty.png, per_profile_rmse_heatmap.png,"
echo "                  spectral_feature_importance.png,"
echo "                  spectral_feature_importance_absolute.png"
echo "============================================"

# Propagate worst exit code so SLURM marks the task FAILED if anything broke,
# without preventing the artifacts that DID get written from being usable.
if   [ $RC_TRAIN  -ne 0 ]; then exit $RC_TRAIN
elif [ $RC_IMPORT -ne 0 ]; then exit $RC_IMPORT
else                            exit 0
fi
