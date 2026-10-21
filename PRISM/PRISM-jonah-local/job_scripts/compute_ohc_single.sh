#!/bin/bash -l
#PBS -N ohc_compute
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=0
#PBS -l walltime=0:20:00
#PBS -q casper
#PBS -j oe
# Project code (-A) is NOT set here: #PBS lines cannot vary by user. Submit with
#   job_scripts/submit_prism_job.sh <this script>   (adds -A for the current user)
# PBS directives cannot expand variables, so -o/-e are not set here: logs are written to the
# directory qsub is run from (submit_prism_job.sh runs it from $PRISM_ROOT/job_logs).

# ---- PRISM paths -------------------------------------------------------------
# PBS runs a spooled copy of this script, so the repo location cannot be taken from
# the script's own path. PRISM_ROOT must point at the PRISM-jonah-local checkout;
# submit_prism_job.sh passes it (or export it in ~/.bashrc).
: "${PRISM_ROOT:?Set PRISM_ROOT to the PRISM-jonah-local checkout (see job_scripts header)}"
# Conda environment used to run the Python scripts (override with PRISM_CONDA_ENV)
PRISM_CONDA_ENV="${PRISM_CONDA_ENV:-/glade/work/$USER/conda-envs/prism}"
module load conda   # provides mamba on NCAR systems
# -------------------------------------------------------------------------------

# Kill all casper jobs with: qselect -u $USER | xargs qdel

echo $PBS_JOBID

if [[ -z "$INPUT_FILE" ]] || [[ -z "$OUTPUT_FILE" ]]; then
    echo "ERROR: missing variables INPUT_FILE or OUTPUT_FILE"
    exit 2
fi

mamba run -p "$PRISM_CONDA_ENV" \
    python "$PRISM_ROOT/J10_OHC_calculation_single.py" "$INPUT_FILE" "$OUTPUT_FILE"
