#!/usr/bin/env bash
# EC2 user-data / first-login bootstrap for the ERA5 SEB analysis (Ubuntu 24.04).
#
# What it does, idempotently:
#   1. installs micromamba under /opt/micromamba and creates the era5-seb env
#      from aws_pipeline/remote/environment.yml
#   2. clones (or pulls) the research repo into ~/Python-Research
#   3. mounts the instance's local NVMe (if any) or a big EBS volume at /data
#      and points the S3 chunk cache there (ERA5_S3_CACHE)
#   4. writes ~/.era5_env with everything run_job.sh needs
#
# Run as the login user (ubuntu) -- e.g. from launch_instance.sh's user-data
# via `sudo -u ubuntu bash bootstrap.sh` -- or by hand after ssh-ing in.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/andrewjbuggee/Python-Research.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="${REPO_DIR:-$HOME/Python-Research}"
ENV_NAME="era5-seb"
MAMBA_ROOT="/opt/micromamba"
DATA_ROOT="${DATA_ROOT:-/data}"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

# --- 1. micromamba -----------------------------------------------------------
# Unconditional and lock-tolerant: at first boot unattended-upgrades may hold
# the dpkg lock for minutes; a second run is a few seconds.
sudo apt-get -o DPkg::Lock::Timeout=600 update -y
sudo apt-get -o DPkg::Lock::Timeout=600 install -y git curl bzip2 rsync htop
if [ ! -x "$MAMBA_ROOT/bin/micromamba" ]; then
  log "installing micromamba"
  sudo mkdir -p "$MAMBA_ROOT/bin" && sudo chown -R "$USER" "$MAMBA_ROOT"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"

# --- 2. repo -----------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  log "updating $REPO_DIR"; git -C "$REPO_DIR" pull --ff-only
else
  log "cloning $REPO_URL ($REPO_BRANCH) -> $REPO_DIR"
  git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
fi
SEB_DIR="$REPO_DIR/ERA5/surface_energy_budget"

# --- 3. environment ----------------------------------------------------------
# environment.yml: from the clone if aws_pipeline/ is pushed, else the copy
# launch_instance.sh dropped next to this script.
ENV_YML="$SEB_DIR/aws_pipeline/remote/environment.yml"
[ -f "$ENV_YML" ] || ENV_YML="$(dirname "$0")/environment.yml"
[ -f "$ENV_YML" ] || { log "no environment.yml found (push aws_pipeline/ or run sync_code.sh), stopping"; exit 1; }
if ! micromamba env list --json | grep -q "/$ENV_NAME\""; then
  log "creating conda env $ENV_NAME from $ENV_YML (a few minutes)"
  micromamba create -y -n "$ENV_NAME" -f "$ENV_YML"
fi

# --- 4. scratch disk for the chunk cache --------------------------------------
# c7i/r7i have no local NVMe: use the root EBS volume (sized in launch_instance.sh).
# Instance types with local NVMe (c7id, i4i, ...) expose it as /dev/nvme1n1.
if [ ! -d "$DATA_ROOT" ]; then
  sudo mkdir -p "$DATA_ROOT"
  if lsblk -dn -o NAME,TYPE | grep -q "^nvme1n1 disk"; then
    log "formatting local NVMe for $DATA_ROOT"
    sudo mkfs.ext4 -q /dev/nvme1n1 && sudo mount /dev/nvme1n1 "$DATA_ROOT"
  fi
  sudo chown "$USER" "$DATA_ROOT"
fi
mkdir -p "$DATA_ROOT/era5_s3_cache" "$DATA_ROOT/results"

# --- 5. environment file used by run_job.sh / jupyter ---------------------------
cat > "$HOME/.era5_env" <<ENV
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export REPO_DIR="$REPO_DIR"
export SEB_DIR="$SEB_DIR"
export ERA5_S3_CACHE="$DATA_ROOT/era5_s3_cache"
export ERA5_S3_WORKERS="$(nproc)"
export RESULTS_ROOT="$DATA_ROOT/results"
export MPLBACKEND=Agg
ENV
log "done. Next: source ~/.era5_env && micromamba activate $ENV_NAME"
