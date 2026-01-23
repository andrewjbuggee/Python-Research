#!/bin/bash
#SBATCH --job-name=combine_mie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/combine_alpha_%a.out
#SBATCH --error=logs/combine_alpha_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anbu8374@colorado.edu

# Job array for all alpha values
# Alpha values: 1-40, 45, 50, 55, 60, 80, 100, 125, 150
# We'll use array indices 1-48 and map them to the actual alpha values

# Load required modules
module purge

# Try to load Python 3 (Alpine has different versions available)
# Try common Python 3 module names
if module load python 2>/dev/null; then
    echo "Loaded default python module"
elif module load python/3.11.6 2>/dev/null; then
    echo "Loaded python/3.11.6"
elif module load python/3.10.10 2>/dev/null; then
    echo "Loaded python/3.10.10"
elif module load python/3.9.10 2>/dev/null; then
    echo "Loaded python/3.9.10"
elif module load anaconda 2>/dev/null; then
    echo "Loaded anaconda"
else
    echo "ERROR: Could not load Python module"
    echo "Available Python modules:"
    module spider python
    exit 1
fi

# Show Python version
python3 --version

# Install netCDF4 if not already installed (to user directory)
python3 -c "import netCDF4" 2>/dev/null || pip3 install --user netCDF4

# Create logs directory if it doesn't exist
mkdir -p logs

# Map SLURM_ARRAY_TASK_ID to actual alpha value
# Indices 1-40 map to alpha 1-40
# Index 41 -> alpha 45
# Index 42 -> alpha 50
# Index 43 -> alpha 55
# Index 44 -> alpha 60
# Index 45 -> alpha 80
# Index 46 -> alpha 100
# Index 47 -> alpha 125
# Index 48 -> alpha 150

if [ $SLURM_ARRAY_TASK_ID -le 40 ]; then
    ALPHA=$SLURM_ARRAY_TASK_ID
elif [ $SLURM_ARRAY_TASK_ID -eq 41 ]; then
    ALPHA=45
elif [ $SLURM_ARRAY_TASK_ID -eq 42 ]; then
    ALPHA=50
elif [ $SLURM_ARRAY_TASK_ID -eq 43 ]; then
    ALPHA=55
elif [ $SLURM_ARRAY_TASK_ID -eq 44 ]; then
    ALPHA=60
elif [ $SLURM_ARRAY_TASK_ID -eq 45 ]; then
    ALPHA=80
elif [ $SLURM_ARRAY_TASK_ID -eq 46 ]; then
    ALPHA=100
elif [ $SLURM_ARRAY_TASK_ID -eq 47 ]; then
    ALPHA=125
elif [ $SLURM_ARRAY_TASK_ID -eq 48 ]; then
    ALPHA=150
else
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Alpha value: $ALPHA"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Run Python script for this alpha value
python3 combine_alpha_netcdf.py $ALPHA

EXIT_CODE=$?

echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
