#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=01:00:00                         # atesting_a100 hard cap = 1h
#SBATCH --partition=atesting_a100               # Testing partition (low queue, 1h max)
#SBATCH --qos=testing
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=nn_extras_ablate
#SBATCH --output=logs/extras_ablate_%j.out
#SBATCH --error=logs/extras_ablate_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ============================================================
# Sequential ablation runs for the 643-input profile-only model
# ============================================================
#
# Runs three trainings of train_standalone_profile_only_extras.py back-to-back
# using the SAME hyperparameter config (RUN_ID below).  The three runs differ
# only in which of the three extra inputs (tau_c, wv_above_cloud, wv_in_cloud)
# is zeroed out:
#
#     run_a : all three extras active             (full 643-input model)
#     run_b : --zero-wv-in       (no within-cloud water vapor)
#     run_c : --zero-wv-above    (no above-cloud water vapor)
#
# Each Python invocation writes its own best_model.pt, config.json,
# history.json, results.json, loss_curves.png, and profiles_true_vs_pred.png
# the moment that run finishes — so if the 1-hour atesting_a100 limit cuts the
# job off mid-stream, the runs that DID complete keep their artifacts on disk.
# Subsequent ablations to add (e.g. --zero-tau-c, all-three-zero parity) can
# be added here following the same pattern.
#
# Submit with:  sbatch sweep_test_extras_profile_only.sh
# Monitor with: squeue -u $USER
# ============================================================

set -u                          # error on undefined vars
# DO NOT set -e: if one run fails we still want the next two to run.

# ── Run-time configuration (edit RUN_ID before sbatch'ing) ──────────────────
RUN_ID=149                       # which run from sweep_results_profile_only/
                                 # to borrow hyperparameters from

REPO_ROOT="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
H5_FILE="${TRAINING_DATA_DIR}combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5"
SWEEP_DIR="hyper_parameter_sweep/sweep_results_profile_only"   # relative to REPO_ROOT
RESULTS_PARENT="${REPO_ROOT}/standalone_results_profile_only_extras"

# Run-suite tag used to keep all three ablations grouped under one parent dir.
# Date-stamped so re-running this script doesn't overwrite previous results.
SUITE_TAG="run$(printf "%03d" $RUN_ID)_$(date +%Y%m%d_%H%M)"
SUITE_DIR="${RESULTS_PARENT}/${SUITE_TAG}"
mkdir -p "${SUITE_DIR}"

# ── Banner ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"
echo "Run ID (sweep config to reuse): $RUN_ID"
echo "Repo root:                      $REPO_ROOT"
echo "HDF5 file:                      $H5_FILE"
echo "Suite output dir:               $SUITE_DIR"
echo "============================================"

# ── Module + env ────────────────────────────────────────────────────────────
module load anaconda
conda activate dropProfs_nn

cd "$REPO_ROOT"
mkdir -p hyper_parameter_sweep/logs

# ── Pre-flight checks (fail fast before launching any training) ─────────────
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

# Quick import check — anything that's going to break should break here
echo
echo "Checking Python imports…"
python -c "
from models_profile_only_extras import ProfileOnlyNetworkExtras
from models_profile_only         import ProfileOnlyLoss
from models                      import RetrievalConfig
from data                        import create_dataloaders_extras, resolve_h5_path
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print('  imports OK')
" || { echo "ERROR: Python import check failed"; exit 1; }

# ── Helper: run one ablation, print banner, log status ──────────────────────
run_one() {
    local label="$1"        # short label for logs (e.g. "all_extras")
    local outdir="$2"       # full output dir for this run
    shift 2                 # remaining args = ablation flags

    echo
    echo "============================================"
    echo " RUN: $label"
    echo " Start: $(date)"
    echo " Output: $outdir"
    echo " Ablation flags: $@"
    echo "============================================"

    # Each call saves its own artifacts atomically (results.json is written
    # only after training completes), so a hard SLURM kill mid-run leaves
    # the previous runs' artifacts intact.
    python train_standalone_profile_only_extras.py \
        --run-id "$RUN_ID" \
        --h5-path "$H5_FILE" \
        --training-data-dir "$TRAINING_DATA_DIR" \
        --sweep-dir "$SWEEP_DIR" \
        --output-dir "$outdir" \
        "$@"

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  >>> $label COMPLETED  (exit 0,  $(date))"
    else
        echo "  >>> $label FAILED     (exit $rc, $(date)) — moving on"
    fi
    return $rc
}

# ── Run 1: all three extras active ──────────────────────────────────────────
run_one "all_extras" \
        "${SUITE_DIR}/run_a_all_extras"
RC_A=$?

# ── Run 2: zero water vapor WITHIN the cloud ────────────────────────────────
run_one "no_wv_in" \
        "${SUITE_DIR}/run_b_no_wv_in" \
        --zero-wv-in
RC_B=$?

# ── Run 3: zero water vapor ABOVE the cloud ─────────────────────────────────
run_one "no_wv_above" \
        "${SUITE_DIR}/run_c_no_wv_above" \
        --zero-wv-above
RC_C=$?

# ── Final summary ───────────────────────────────────────────────────────────
echo
echo "============================================"
echo " ABLATION SUITE COMPLETE"
echo "============================================"
echo " End time: $(date)"
echo " run_a_all_extras   : exit $RC_A"
echo " run_b_no_wv_in     : exit $RC_B"
echo " run_c_no_wv_above  : exit $RC_C"
echo
echo " Results in:  $SUITE_DIR"
echo "============================================"

# Exit 0 only if all three succeeded.  Otherwise propagate the worst exit code
# so SLURM marks the array task as FAILED for visibility, even though the
# successful runs' artifacts are still on disk.
if   [ $RC_A -ne 0 ]; then exit $RC_A
elif [ $RC_B -ne 0 ]; then exit $RC_B
elif [ $RC_C -ne 0 ]; then exit $RC_C
else                       exit 0
fi
