#!/bin/bash
# ============================================================
# Standalone long re-train of ONE profile+τ (PT) sweep config on al40.
#
# Why this exists: the PT sweep capped n_epochs at 1500 and the top runs were
# all still improving at the cap (best epochs ~1430-1495, early stop never
# triggered).  This script re-trains a chosen config with a longer epoch
# budget using the SAME trainer, config JSON, seed and 80/10/10 split as the
# sweep — so the result is directly comparable to the sweep row it extends.
#
# Edit the ALL-CAPS variables in the "User configuration" block, then:
#     cd /projects/anbu8374/Python-Research/lasp-CU-paper-3
#     mkdir -p logs
#     sbatch shell_scripts/standalone_profile_tau_al40.sh
#
# Wall-time note: run_004 took ~35 min for 1500 epochs on al40; the slowest
# top-10 config (batch 128, 3x128) took ~43 min. 4500 epochs → ~1.8-2.2 h.
# The 04:00:00 budget below is ~2x margin. Early stopping will usually end
# the run well before the epoch cap.
# ============================================================

# ----- SLURM directives -----------------------------------------------------
#SBATCH --account=ucb762_asc1
#SBATCH --partition=al40
#SBATCH --qos=gpu-normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1           # al40 requires the GRES type spelled out
#SBATCH --mem=8G
#SBATCH --job-name=PT_solo_long
#SBATCH --output=logs/standalone_PT_long_%A.out
#SBATCH --error=logs/standalone_PT_long_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

# ----- User configuration ---------------------------------------------------
# Which PT sweep config to re-train (hyperparameters are read from the same
# JSON the sweep used).
RUN_ID="4"

# Epoch budget + early-stop patience for the long run. The sweep used
# 1500 / 150. Patience is doubled here so the slow late-training improvement
# phase (after several ReduceLROnPlateau halvings) isn't cut off prematurely;
# checkpointing is val-selected, so a generous budget can't hurt the final
# model — only the wall time.
N_EPOCHS="4500"
EARLY_STOP_PATIENCE="300"

# Synthetic HDF5 (same file the sweep trained on).
H5_PATH="/scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_7-levels_8_May_2026.h5"

REPO="/projects/anbu8374/Python-Research/lasp-CU-paper-3"
CONFIG_FILE="${REPO}/hyper_parameter_sweep/sweep_configs_profile_tau_synthetic/run_$(printf '%03d' ${RUN_ID}).json"

# Output parent dir; the trainer creates run_NNN/ underneath it, so results
# land in .../PT_long_<stem>_<epochs>ep/run_NNN/ — completely separate from
# the sweep's results and directly consumable by plot_run_figures_profile_tau.py
# and the feature-importance scripts.
H5_STEM=$(basename "$H5_PATH" .h5)
OUTPUT_DIR="${REPO}/standalone_results_profile_tau_synthetic/PT_long_${H5_STEM}_${N_EPOCHS}ep"

# ----- Banner ---------------------------------------------------------------
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "Run ID:        $RUN_ID  (PT sweep config)"
echo "Config:        $CONFIG_FILE"
echo "n_epochs:      $N_EPOCHS  (patience $EARLY_STOP_PATIENCE)"
echo "HDF5:          $H5_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "============================================"

# ----- Environment ----------------------------------------------------------
module purge
module load anaconda
conda activate /projects/anbu8374/software/anaconda/envs/dropProfs_nn

cd "$REPO/hyper_parameter_sweep"
mkdir -p "$REPO/logs"

# ----- Sanity checks before burning GPU time --------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: sweep config not found at $CONFIG_FILE"
    echo "Confirm RUN_ID=$RUN_ID and that sweep_configs_profile_tau_synthetic/ was uploaded."
    exit 1
fi
if [ ! -f "$H5_PATH" ]; then
    echo "ERROR: HDF5 not found at $H5_PATH"
    exit 1
fi

# ----- Run ------------------------------------------------------------------
time python sweep_train_profile_tau_synthetic.py \
    --config-json         "$CONFIG_FILE" \
    --h5-path             "$H5_PATH" \
    --output-dir          "$OUTPUT_DIR" \
    --n-epochs            "$N_EPOCHS" \
    --early-stop-patience "$EARLY_STOP_PATIENCE"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
