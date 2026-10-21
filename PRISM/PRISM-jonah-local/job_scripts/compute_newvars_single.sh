#!/bin/bash -l
#PBS -N ohc_compute
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=0
#PBS -l walltime=0:40:00
#PBS -q casper
# PBS directives cannot expand variables, so -o/-e are not set here: logs are written to the
# directory qsub is run from. Submit from $PRISM_ROOT/job_logs, or add  qsub -o <dir>.

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

echo $PBS_JOBID

if [[ -z "$FILE_PATH" ]] || [[ -z "$SAVE_PATH" ]]; then
    echo "ERROR: missing variables FILE_PATH or SAVE_PATH"
    echo ""
    echo "Usage: qsub -v FILE_PATH=/path/to/input,SAVE_PATH=/path/to/output $0"
    exit 2
fi

echo "Job started at $(date)"
mamba run -p "$PRISM_CONDA_ENV" \
    python "$PRISM_ROOT/J14_CESM_compute_newvars_single.py" "$FILE_PATH" "$SAVE_PATH"

echo "Job completed successfully at $(date)"