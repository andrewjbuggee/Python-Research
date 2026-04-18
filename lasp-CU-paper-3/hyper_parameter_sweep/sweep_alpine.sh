#!/bin/bash
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=01:30:00            # Sweep 2: n_epochs=1000 + noise aug can fill ~50 min; 1.5h margin
#SBATCH --partition=aa100          # Alpine GPU partition (NVIDIA A100)
#SBATCH --qos=normal
#SBATCH --mem=8G                   # Sweep 1 peaked at 1.5G; 8G leaves ~5x headroom for larger data
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU per run (each run is small)
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pinn_sweep_2
#SBATCH --output=logs/sweep2_%A_%a.out
#SBATCH --error=logs/sweep2_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --array=0-99%16               # 100 runs (matches run_000.json to run_099.json)

# ============================================================
# Hyperparameter Sweep — SLURM Job Array
# ============================================================
#
# Before submitting:
#   1. Run `python generate_sweep_2.py` locally to create sweep_configs_2/
#   2. Upload project to Alpine:
#      rsync -av --exclude='__pycache__' --exclude='.git' \
#        /Users/andrewbuggee/Documents/VS_CODE/Python-Research/lasp-CU-paper-3/ \
#        anbu8374@login.rc.colorado.edu:/projects/anbu8374/Python-Research/lasp-CU-paper-3/
#   3. Make sure HDF5 data is at:
#      /scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_7-levels_17_April_2026.h5
#
# Submit with:
#   cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep
#   sbatch sweep_alpine.sh
#
# Monitor with:
#   squeue -u $USER
#   sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
#
# After all jobs complete, aggregate results:
#   python compare_sweep.py
# ============================================================

echo "============================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================"

# Load modules
# Load anaconda (only available on compute/compile nodes, not login nodes)
module load anaconda

# Activate conda environment
conda activate dropProfs_nn

# Navigate to the hyper_parameter_sweep directory.
# sweep_train.py, generate_sweep_2.py, sweep_configs_2/, and sweep_results_3/
# all live here.  models.py and data.py are at the parent (repo root);
# sweep_train.py's sys.path is set up to find them automatically.
cd /projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep

# Create directories
mkdir -p logs sweep_results_3

# Run this task's configuration
# SLURM_ARRAY_TASK_ID is 0, 1, 2, ..., 99
RUN_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="sweep_configs_2/run_${RUN_ID}.json"

echo "Config: $CONFIG_FILE"
echo ""

python sweep_train.py --config-json "$CONFIG_FILE"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE   # propagate failure so SLURM marks the task as FAILED
