#!/bin/bash
#SBATCH --account=ucb762_asc1
#SBATCH --nodes=1
#SBATCH --time=02:30:00            # Per-config wall budget. Slowest config in
                                   # the profile-only M0_rev2 sweep on the same
                                   # 42k-sample HDF5 took 3017 s (50 min); the
                                   # tau head adds ~nothing. 2.5 h is ~3x margin.
#SBATCH --partition=al40
#SBATCH --qos=gpu-normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=NN_synth_sweep_PT
#SBATCH --output=logs/sweep_synth_PT_%A_%a.out
#SBATCH --error=logs/sweep_synth_PT_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-99%6             # 100 configs, max 6 concurrent

# ============================================================
# Synthetic-cloud PROFILE + TAU Hyperparameter Sweep — VARIANT PT
#
#   Model : DropletProfileNetwork (models.py)
#   Inputs: 636 HySICS reflectances + 4 geometry angles  (640)
#   Output: 7-level r_e profile + per-level sigma
#           AND cloud optical thickness tau_c + sigma
#
#   Companion to sweep_alpine_profile_only_synthetic_M0.sh.  The shared
#   hyperparameter draws are bit-identical to the M0 configs (same seed, same
#   LHS stream), and the profile-aware split is the same, so run_NNN here and
#   run_NNN in sweep_results_profile_only_synthetic_M0_rev2 are a paired
#   comparison: what does adding the tau task do to profile retrieval?
#
#   Two extra swept axes, drawn from a separate RNG stream:
#     tau_weight    log-uniform in [0.02, 2.0]  — tau NLL vs profile NLL
#     tau_transform linear | log10 (50/50)      — tau normalization before NLL
#
# Before submitting:
#   1. Generate configs (locally or on a login node):
#        python generate_sweep_profile_tau_synthetic.py
#   2. Upload the repo to Alpine.
#   3. Make sure the synthetic HDF5 is at:
#      /scratch/alpine/anbu8374/neural_network_training_data/
#        synthetic_training_data_7-levels_8_May_2026.h5
#
# Submit with:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
#   sbatch sweep_alpine_profile_tau_synthetic.sh
#
# Aggregate afterwards (the summary.json keys match the profile-only sweep):
#   python compare_sweep_profile_only_synthetic.py \
#       --sweep-dir sweep_results_profile_tau_synthetic \
#                   sweep_results_profile_only_synthetic_M0_rev2
# ============================================================

echo "============================================"
echo "Variant:       PT (profile + tau_c)"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

module purge
module load anaconda
conda activate /projects/anbu8374/software/anaconda/envs/dropProfs_nn

cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
mkdir -p logs sweep_results_profile_tau_synthetic

VARIANT=PT
RUN_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="sweep_configs_profile_tau_synthetic/run_${RUN_ID}.json"

# Full path to the HDF5 on this machine.  Passed as --h5-path (not just the
# directory) so the file actually used is unambiguous in the logs and does not
# depend on the filename baked into the config.
H5_PATH="/scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_7-levels_8_May_2026.h5"

# Results dir, sibling of the configs dir (not nested inside it).
# The trainer creates run_NNN/ subdirs underneath whatever path we pass here.
OUTPUT_DIR="/projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep/sweep_results_profile_tau_synthetic"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    echo "Did you run generate_sweep_profile_tau_synthetic.py and upload the configs?"
    exit 1
fi

if [ ! -f "$H5_PATH" ]; then
    echo "ERROR: training HDF5 not found: $H5_PATH"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo "HDF5:   $H5_PATH"
echo ""

python sweep_train_profile_tau_synthetic.py \
    --config-json "$CONFIG_FILE" \
    --h5-path     "$H5_PATH" \
    --output-dir  "$OUTPUT_DIR"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
