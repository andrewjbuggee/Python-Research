#!/usr/bin/env bash
# Laptop: rsync THIS working tree of ERA5/surface_energy_budget onto the
# instance -- the "tweak locally, run remotely" loop, no commit needed.
#
#   ./sync_code.sh            # code + small inputs; excludes data/, figures/, logs/, caches
#   DRY=1 ./sync_code.sh      # show what would change
#
# Carries what a git clone cannot: uncommitted edits and the gitignored
# observation inputs (genie_arm_*.xlsx/.txt) the lwph figures read. Local
# ERA5 downloads (data/) are never sent; the instance reads the bucket.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/.last_instance" ] || { echo "no .last_instance here: run launch_instance.sh first"; exit 1; }
read -r INSTANCE_ID DNS KEY_FILE < "$HERE/.last_instance"
SRC="$(cd "$HERE/../.." && pwd)"                      # .../ERA5/surface_energy_budget
DEST="~/Python-Research/ERA5/surface_energy_budget"
FLAGS=(-avz --progress
       --exclude data/ --exclude figures/ --exclude logs/ --exclude __pycache__/
       --exclude '.ipynb_checkpoints/' --exclude 'aws_pipeline/results/'
       --exclude '*.nc' --exclude '*.png' --exclude '*.pdf' --exclude 'nohup.out')
[ "${DRY:-0}" = 1 ] && FLAGS+=(--dry-run)
ssh -i "$KEY_FILE" "ubuntu@$DNS" "mkdir -p $DEST"
rsync "${FLAGS[@]}" -e "ssh -i $KEY_FILE" "$SRC/" "ubuntu@$DNS:$DEST/"
echo "synced $SRC -> $DNS:$DEST"
