#!/bin/bash
#SBATCH --job-name=pinn_train
#SBATCH --partition=aa100          # Alpine GPU partition (NVIDIA A100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.buggee@colorado.edu
#SBATCH --account=ucb-general       # Update with your allocation

# ============================================================
# SLURM Training Script for CU Boulder Alpine
# ============================================================
# Submit with:  sbatch scripts/train_alpine.sh
# Monitor with: squeue -u $USER
# ============================================================

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "============================================"

# Load modules (adjust versions to what's available on Alpine)
module purge
module load anaconda
module load cuda/12.1

# Activate your conda environment
# Create this once with: conda create -n paper3 python=3.11 pytorch torchvision
#                        pytorch-cuda=12.1 -c pytorch -c nvidia
#                        conda activate paper3
#                        pip install h5py pyyaml matplotlib scipy
conda activate paper3

# Navigate to project directory
cd /projects/$USER/paper3

# Create log directory if needed
mkdir -p logs checkpoints

# Run training
python train.py \
    --config configs/stage1_modis.yaml \
    --output-dir checkpoints

echo "============================================"
echo "End time:     $(date)"
echo "============================================"
