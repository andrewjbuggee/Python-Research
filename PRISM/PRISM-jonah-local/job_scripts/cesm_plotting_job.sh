#!/bin/bash -l
#PBS -N cesm_plotting
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=8GB:ngpus=0
#PBS -l walltime=0:59:59
#PBS -q casper
#PBS -j oe

# ---- PRISM paths -------------------------------------------------------------
# PBS runs a spooled copy of this script, so the repo location cannot be taken from
# the script's own path. PRISM_ROOT must point at the PRISM-jonah-local checkout:
# export it in ~/.bashrc, or submit with  qsub -v PRISM_ROOT=/path/to/PRISM-jonah-local
: "${PRISM_ROOT:?Set PRISM_ROOT to the PRISM-jonah-local checkout (see job_scripts header)}"
# Conda environment used to run the Python scripts (override with PRISM_CONDA_ENV)
PRISM_CONDA_ENV="${PRISM_CONDA_ENV:-/glade/work/$USER/conda-envs/prism}"
module load conda   # provides mamba on NCAR systems
# -------------------------------------------------------------------------------

# Kill all casper jobs with: qselect -u $USER | xargs qdel
mkdir -p "$PRISM_ROOT/job_logs" && cd "$PRISM_ROOT/job_logs"
mamba run -p "$PRISM_CONDA_ENV" \
    python "$PRISM_ROOT/J03_CESM_OLR_ASR_plots_multiensemble.py"

mamba run -p "$PRISM_CONDA_ENV" \
    python "$PRISM_ROOT/J04_CESM_EEI_timeseries_multiensemble.py"