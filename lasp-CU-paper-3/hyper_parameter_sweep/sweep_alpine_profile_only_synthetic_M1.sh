#!/bin/bash
#SBATCH --account=ucb762_asc1
#SBATCH --nodes=1
#SBATCH --time=01:30:00            # Per-config wall budget. Single 80/10/10
                                   # split, ~7 min/config on al40 from the
                                   # smoke test extrapolation; 1.5 h is a
                                   # 12× safety margin.
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --mem=7G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pinn_synth_M1
#SBATCH --output=logs/sweep_synth_M1_%A_%a.out
#SBATCH --error=logs/sweep_synth_M1_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-99%6             # 100 configs, max 6 concurrent

# ============================================================
# Synthetic-cloud Profile-Only Hyperparameter Sweep — VARIANT M1
#   Inputs: 636 spectra + 4 geometry            (extras all zeroed)
#
# Before submitting:
#   1. Run `python generate_sweep_profile_only_synthetic.py` locally.
#   2. Upload to Alpine.
#   3. Make sure the synthetic HDF5 is at:
#      /scratch/alpine/anbu8374/neural_network_training_data/
#        synthetic_training_data_7-levels_8_May_2026.h5
#
# Submit with:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
#   sbatch sweep_alpine_profile_only_synthetic_M1.sh
# ============================================================

echo "============================================"
echo "Variant:       M1 (spectra + geometry + tau_c                  (1 active extra))"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

module load anaconda
conda activate dropProfs_nn

cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
mkdir -p logs sweep_results_profile_only_synthetic_M1

VARIANT=M1
RUN_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="sweep_configs_profile_only_synthetic_${VARIANT}/run_${RUN_ID}.json"
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
# Per-variant results dir, sibling of the configs dir (not nested inside it).
# The trainer creates run_NNN/ subdirs underneath whatever path we pass here.
OUTPUT_DIR="/projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep/sweep_results_profile_only_synthetic_${VARIANT}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    echo "Did you run generate_sweep_profile_only_synthetic.py and upload the configs?"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo ""

python sweep_train_profile_only_synthetic.py \
    --config-json "$CONFIG_FILE" \
    --training-data-dir "$TRAINING_DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
