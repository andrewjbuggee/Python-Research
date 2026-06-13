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
#SBATCH --output=logs/wv_core_ablation_%A_%a.out
#SBATCH --error=logs/wv_core_ablation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-2                # 0=full(re-baseline) 1=wv_core 2=continuum_control

# ============================================================
# Water-vapor-core ablation retrain of paper-3 run_004 (variant M0).
#   Removes WV band cores (940/1140/1380/1900 nm), retrains, and compares
#   against a count-matched redundant-continuum control and the published
#   baseline. See train_wv_core_ablation.py / wv_band_mask.py.
#
# Before submitting:
#   1. Build the masks once (writes wv_core_ablation/wv_core_ablation_masks.npz):
#        python wv_band_mask.py --plot
#   2. Confirm the synthetic HDF5 is at:
#        /scratch/alpine/anbu8374/neural_network_training_data/
#          synthetic_training_data_7-levels_8_May_2026.h5
#
# Submit:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3
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

REPO=/projects/anbu8374/Python-Research/lasp-CU-paper-3
cd "$REPO"
mkdir -p logs wv_core_ablation

MASKS=(full wv_core continuum_control)
MASK=${MASKS[$SLURM_ARRAY_TASK_ID]}
TRAINING_DATA_DIR="/scratch/alpine/anbu8374/neural_network_training_data/"
MASKS_NPZ="$REPO/wv_core_ablation/wv_core_ablation_masks.npz"

echo "Mask: $MASK"
echo ""

python train_wv_core_ablation.py \
    --mask "$MASK" \
    --training-data-dir "$TRAINING_DATA_DIR" \
    --masks-npz "$MASKS_NPZ" \
    --output-dir "$REPO/wv_core_ablation"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
