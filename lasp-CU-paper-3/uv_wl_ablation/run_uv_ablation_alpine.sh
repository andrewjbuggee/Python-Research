#!/bin/bash
#SBATCH --account=ucb762_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00          # ~35 min/run on al40 (run_004 took 2094 s);
                                 # 2 h is a comfortable margin.
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=uv_ablation
#SBATCH --output=uv_wl_ablation/logs/uv_ablation_%A_%a.out
#SBATCH --error=uv_wl_ablation/logs/uv_ablation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-1              # task 0 = control (all 636); task 1 = ablation (<500 nm dropped)

# ============================================================
# UV retrain-ablation on Alpine (CUDA).  Retrains run_004's EXACT recipe twice:
#   task 0  --cutoff-nm 0     control : keep all 636 spectral channels
#   task 1  --cutoff-nm 500   ablation: drop the 49 channels < 500 nm (keep 587)
# Comparing the two isolates the effect of removing the UV/blue channels from
# train-to-train variance.  Both run in parallel as a 2-task array.
#
# Submit FROM THE REPO ROOT so the relative log/script paths resolve:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3
#   mkdir -p uv_wl_ablation/logs
#   sbatch uv_wl_ablation/run_uv_ablation_alpine.sh
#
# Requires on Alpine:
#   - the HDF5 at  $DATA_DIR/synthetic_training_data_7-levels_8_May_2026.h5
#   - run_004's summary.json at
#       hyper_parameter_sweep/sweep_results_profile_only_synthetic_M0_rev2/run_004/
#     (it holds the hyperparameters; it is .gitignored, so if you sync via git
#      rather than rsync, copy that one ~2 KB file over manually.)
# ============================================================

echo "============================================"
echo "Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "============================================"

module load anaconda
conda activate /projects/anbu8374/software/anaconda/envs/dropProfs_nn

DATA_DIR=/scratch/alpine/anbu8374/neural_network_training_data/

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    CUTOFF=0       # control
else
    CUTOFF=500     # ablation
fi

echo "Running with --cutoff-nm $CUTOFF"
python train_uv_ablation_run004.py \
    --cutoff-nm "$CUTOFF" \
    --training-data-dir "$DATA_DIR" \
    --device cuda
EXIT_CODE=$?

echo "============================================"
echo "End:        $(date)"
echo "Exit code:  $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
