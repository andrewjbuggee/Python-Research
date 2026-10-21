#!/bin/bash
# Submit a PRISM PBS job with the right project code and repo location for whoever runs it.
#
# Usage (from anywhere):
#   job_scripts/submit_prism_job.sh <job_script.sh> [extra qsub options]
# Example:
#   job_scripts/submit_prism_job.sh spatial_averaging_job.sh
#   job_scripts/submit_prism_job.sh control_background_job.sh -l walltime=02:00:00
#
# Why a wrapper: #PBS directives are read literally by qsub (no variables, no if/else), so
# a job script cannot pick its own project code. This script asks prism_paths.py for the
# code of the current user (PBS_ACCOUNT_CASPER / PBS_ACCOUNT, else the per-user default)
# and passes it on the command line with -A. It also passes PRISM_ROOT, because PBS runs
# a spooled copy of the job script that cannot find the repo by itself.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <job_script.sh> [extra qsub options]" >&2
    exit 2
fi
job_script=$(basename "$1")
shift

# This wrapper is run directly (not through PBS), so its own path locates the repo
PRISM_ROOT="${PRISM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# prism_paths.py uses only the standard library, so the system python3 is enough here
account=$(python3 "$PRISM_ROOT/prism_paths.py" --account)

# Job output files land in the directory qsub is run from
mkdir -p "$PRISM_ROOT/job_logs"
cd "$PRISM_ROOT/job_logs"

echo "Submitting $job_script with project $account (PRISM_ROOT=$PRISM_ROOT)"
# Command-line options must come before the script path
qsub -A "$account" -v "PRISM_ROOT=$PRISM_ROOT" "$@" "$PRISM_ROOT/job_scripts/$job_script"
