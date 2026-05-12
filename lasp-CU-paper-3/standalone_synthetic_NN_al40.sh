#!/bin/bash
# ============================================================
# Smoke-test the synthetic-cloud standalone trainer on Alpine's
# atesting_a100 node (single A100, fast queue, 1-hour cap).
#
# What this runs:
#   train_standalone_profile_only_synthetic.py — pulls one sweep config
#   (variant + run_id) and re-trains it on a chosen synthetic HDF5.
#
# Edit the four ALL-CAPS variables in the "User configuration" block,
# then submit with:
#     sbatch standalone_synthetic_atesting_a100.sh
#
# Wall-time note: with the new ~42 k-sample HDF5, a full 1500-epoch
# training takes less than 60 minutes on a single A100 — right at
# atesting_a100's 1-hour limit. By default we cap at 500 epochs (set
# via N_EPOCHS_OVERRIDE below) so the smoke test finishes inside the
# testing window. Bump N_EPOCHS_OVERRIDE to "" or 1500 once you want
# the full run on al40.
# ============================================================

# ----- SLURM directives -----------------------------------------------------
#SBATCH --account=ucb762_asc1
#SBATCH --partition=al40
#SBATCH --qos=normal              # al40 uses normal QOS; testing QOS is for atesting_* only
#SBATCH --time=20:00:00           # al40 wall budget — generous for 1500 epochs on ~42k samples
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G                 # plenty for ~42k samples + A100
#SBATCH --job-name=synth_solo_kfold
#SBATCH --output=logs/standalone_synth_kfold_%A.out
#SBATCH --error=logs/standalone_synth_kfold_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ----- User configuration ---------------------------------------------------
# Pick one of M0 / M1 / M2 and a run_id from that variant's sweep results.
# The trainer reads
#   hyper_parameter_sweep/sweep_results_profile_only_synthetic_<VARIANT>/run_<RUN_ID>/summary.json
# for hyperparameters and extras-flags.
VARIANT="M0"
RUN_ID="98"

# Synthetic HDF5 to train on. Update once your ~42k-sample file is finalized.
H5_PATH="/scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_7-levels_8_May_2026.h5"

# Cap epochs for the smoke test so a single config finishes inside the 1-hour
# testing window. Set to empty string to use the run config's full n_epochs.
N_EPOCHS_OVERRIDE="2500"

# K-fold mode for the trainer. 1 = single 80/10/10 split (current default).
# K > 1 = full-coverage K-fold; the trainer trains K models, each tested on a
# disjoint partition, and runs all aggregate diagnostic plots on the
# concatenated predictions covering the entire dataset. Wall time scales
# roughly linearly with K, so bump --time below if you increase this.
N_FOLDS="20"

# Output dir (per-variant + per-run, easy to identify alongside other tests).
H5_STEM=$(basename "$H5_PATH" .h5)
OUTPUT_DIR="/projects/anbu8374/Python-Research/lasp-CU-paper-3/standalone_results_profile_only_synthetic/${VARIANT}_run$(printf '%03d' ${RUN_ID})_${H5_STEM}_atest"

# ----- Banner ---------------------------------------------------------------
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "Variant:       $VARIANT"
echo "Run ID:        $RUN_ID"
echo "HDF5:          $H5_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "n_epochs cap:  ${N_EPOCHS_OVERRIDE:-<sweep config default>}"
echo "n_folds:       $N_FOLDS"
echo "============================================"

# ----- Environment ----------------------------------------------------------
module purge
module load anaconda
conda activate dropProfs_nn

cd /projects/anbu8374/Python-Research/lasp-CU-paper-3
mkdir -p logs

# Sanity-check that the sweep summary actually exists before we burn an A100
SUMMARY_PATH="hyper_parameter_sweep/sweep_results_profile_only_synthetic_${VARIANT}/run_$(printf '%03d' ${RUN_ID})/summary.json"
if [ ! -f "$SUMMARY_PATH" ]; then
    echo "ERROR: sweep summary not found at $SUMMARY_PATH"
    echo "Confirm VARIANT=$VARIANT and RUN_ID=$RUN_ID, or rsync sweep_results from local."
    exit 1
fi
if [ ! -f "$H5_PATH" ]; then
    echo "ERROR: HDF5 not found at $H5_PATH"
    exit 1
fi

# ----- Run ------------------------------------------------------------------
EXTRA_FLAGS=""
if [ -n "$N_EPOCHS_OVERRIDE" ]; then
    EXTRA_FLAGS="--n-epochs $N_EPOCHS_OVERRIDE"
fi

time python train_standalone_profile_only_synthetic.py \
    --variant     "$VARIANT" \
    --run-id      "$RUN_ID" \
    --h5-path     "$H5_PATH" \
    --output-dir  "$OUTPUT_DIR" \
    --n-folds     "$N_FOLDS" \
    $EXTRA_FLAGS

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
