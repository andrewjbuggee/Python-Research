#!/bin/bash
#SBATCH --account=ucb762_asc1
#SBATCH --nodes=1
#SBATCH --time=01:59:00            # run_004 trained in ~35 min; 2 h is ample.
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=NN_wv_core_ablation
#SBATCH --output=/projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation/logs/wv_core_ablation_%A_%a.out
#SBATCH --error=/projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation/logs/wv_core_ablation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-2                # 0=full(re-baseline) 1=wv_core 2=continuum_control

# ============================================================
# Water-vapor-core ablation retrain of paper-3 run_004 (variant M0).
#   Removes WV band cores (940/1140/1380/1900 nm), retrains, and compares
#   against a count-matched redundant-continuum control and the published
#   baseline. See train_wv_core_ablation.py / wv_band_mask.py.
#
# All scripts and outputs live in this folder:
#   /projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation/
# The shared modules (models.py, data.py, ...) stay one level up at the repo
# root; the scripts add that to sys.path themselves.
#
# Before submitting (run once, in an interactive/compile node with the env):
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation
#   mkdir -p logs                       # SLURM writes the .out/.err here at job start
#   module load anaconda
#   conda activate /projects/anbu8374/software/anaconda/envs/dropProfs_nn
#   python wv_band_mask.py --plot       # writes wv_core_ablation_masks.npz here
#
# Submit:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation
#   sbatch run_wv_core_ablation_alpine.sh
# ============================================================

echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

module load anaconda
conda activate /projects/anbu8374/software/anaconda/envs/dropProfs_nn

# Folder holding the ablation scripts + outputs (this directory).
ABLATION_DIR=/projects/anbu8374/Python-Research/lasp-CU-paper-3/wv_core_ablation
cd "$ABLATION_DIR"
mkdir -p logs

MASKS=(full wv_core continuum_control)
MASK=${MASKS[$SLURM_ARRAY_TASK_ID]}

# Full path to the synthetic HDF5 (8-May file). Passed via --h5-path, NOT
# --training-data-dir: resolve_h5_path() keeps the *filename* from the config,
# and run_004.json still names the older 5-May file, so a dir override would
# point at a non-existent path.
H5_PATH="/scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_7-levels_8_May_2026.h5"
MASKS_NPZ="$ABLATION_DIR/wv_core_ablation_masks.npz"

echo "Mask:     $MASK"
echo "HDF5:     $H5_PATH"
echo "Masks:    $MASKS_NPZ"
echo "Out dir:  $ABLATION_DIR"
echo ""

python train_wv_core_ablation.py \
    --mask "$MASK" \
    --h5-path "$H5_PATH" \
    --masks-npz "$MASKS_NPZ" \
    --output-dir "$ABLATION_DIR"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
