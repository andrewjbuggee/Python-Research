#!/usr/bin/env bash
# One-time setup ON CASPER (run on a login node, which has internet).
#
#   cd ~/Python-Research/ERA5/surface_energy_budget/aws_pipeline/casper
#   ./setup_casper.sh                       # uses NCAR's NPL environment
#   ERA5_ENV_MODE=own ./setup_casper.sh     # builds a private conda env instead
#
# Writes ~/.era5_casper_env, which every job script sources. Re-running is safe.
set -euo pipefail

NPL_VERSION="${NPL_VERSION:-npl-2025b}"     # pin: 'npl' floats to the newest
ENV_MODE="${ERA5_ENV_MODE:-npl}"            # npl | own
OWN_ENV_NAME="${OWN_ENV_NAME:-era5-seb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SEB_DIR="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$SEB_DIR/../.." && pwd)"

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

: "${SCRATCH:=/glade/derecho/scratch/$USER}"
: "${WORK:=/glade/work/$USER}"
[ -d "$SCRATCH" ] || { echo "no SCRATCH at $SCRATCH -- is this a Casper login node?"; exit 1; }

# --- 1. the ERA5 archive --------------------------------------------------
# GDEX renamed ds633.0 to d633000 in Sep 2026; try the current path first.
ERA5_ROOT=""
for cand in ${ERA5_GLADE_ROOT:-} \
            /gdex/data/d633000 \
            /glade/campaign/collections/gdex/data/d633000 \
            /glade/campaign/collections/rda/data/ds633.0 \
            /glade/collections/rda/data/ds633.0; do
  if [ -d "$cand/e5.oper.an.sfc" ]; then ERA5_ROOT="$cand"; break; fi
done
if [ -z "$ERA5_ROOT" ]; then
  log "WARNING: no ERA5 archive found. Look for it with:"
  log "    ls -d /gdex/data/* 2>/dev/null | head"
  log "    find /glade/campaign/collections -maxdepth 4 -name 'e5.oper.an.sfc' 2>/dev/null | head"
  log "  then re-run with ERA5_GLADE_ROOT=<path> ./setup_casper.sh"
  ERA5_ROOT="${ERA5_GLADE_ROOT:-}"
else
  log "ERA5 archive: $ERA5_ROOT"
fi

# --- 2. python environment ------------------------------------------------
module load conda 2>/dev/null || log "note: 'module load conda' failed; is this an NCAR system?"
if [ "$ENV_MODE" = "own" ]; then
  # By PREFIX under $WORK: a conda env is several GB and home is only 50 GB.
  OWN_ENV_PREFIX="$WORK/conda-envs/$OWN_ENV_NAME"
  if [ ! -d "$OWN_ENV_PREFIX" ]; then
    log "creating conda env at $OWN_ENV_PREFIX (several minutes)"
    conda env create --prefix "$OWN_ENV_PREFIX" -f "$HERE/environment_casper.yml"
  fi
  ENV_ACTIVATE="$OWN_ENV_PREFIX"
else
  ENV_ACTIVATE="$NPL_VERSION"
  log "using NCAR's $NPL_VERSION (set ERA5_ENV_MODE=own to build a private env)"
fi
conda activate "$ENV_ACTIVATE" || { log "could not activate $ENV_ACTIVATE"; exit 1; }
log "python: $(python -c 'import sys;print(sys.version.split()[0], sys.executable)')"

# --- 3. working directories ------------------------------------------------
CACHE_DIR="$SCRATCH/era5_cache"          # decoded chunk cache; purged with scratch
RESULTS_ROOT="$WORK/era5_results"        # figures + pickles; NOT purged
CARTOPY_DIR="$WORK/cartopy_data"         # map shapefiles, staged from a login node
mkdir -p "$CACHE_DIR" "$RESULTS_ROOT" "$CARTOPY_DIR" "$SCRATCH/temp" "$SCRATCH/mplconfig"

# --- 4. cartopy map data (compute nodes may have no internet) ---------------
log "staging cartopy map data into $CARTOPY_DIR"
CARTOPY_DATA_DIR="$CARTOPY_DIR" python "$HERE/prefetch_cartopy.py" || \
  log "WARNING: cartopy prefetch failed; the map figures may fail on a compute node"

# --- 5. the env file every job sources --------------------------------------
cat > "$HOME/.era5_casper_env" <<ENV
# written by setup_casper.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
export REPO_ROOT="$REPO_ROOT"
export SEB_DIR="$SEB_DIR"
export ERA5_CONDA_ENV="$ENV_ACTIVATE"
$([ -n "$ERA5_ROOT" ] && echo "export ERA5_GLADE_ROOT=\"$ERA5_ROOT\"")
export ERA5_SOURCE=glade              # fail loudly rather than silently using the network
export ERA5_CACHE="$CACHE_DIR"     # decoded regional chunks, reused across passes
export RESULTS_ROOT="$RESULTS_ROOT"
export CARTOPY_DATA_DIR="$CARTOPY_DIR"
export MPLCONFIGDIR="$SCRATCH/mplconfig"
export TMPDIR="$SCRATCH/temp"
export MPLBACKEND=Agg
# libhdf5 takes POSIX locks when opening a file. Read-only GPFS/NFS mounts often
# refuse them, and the failure reads as a corrupt-file error rather than a
# permissions one. The archive is read-only, so disabling them is safe.
export HDF5_USE_FILE_LOCKING=FALSE
ENV
log "wrote $HOME/.era5_casper_env"

# --- 6. prove it works ------------------------------------------------------
set +u; source "$HOME/.era5_casper_env"; set -u
log "running preflight"
python "$HERE/preflight.py" || log "preflight reported problems (see above)"
cat <<MSG

Next:
  1. From the LAPTOP, sync your working tree (carries the gitignored observation files):
       ERA5/surface_energy_budget/aws_pipeline/casper/sync_to_casper.sh <user>@casper.hpc.ucar.edu
  2. Submit a small test:
       ./submit.sh --project <PROJECT_CODE> --region barrow --years 2023 --stages lwph
  3. Then the real thing:
       ./submit.sh --project <PROJECT_CODE> --region arctic_circle --years 2014-2024 --all
MSG
