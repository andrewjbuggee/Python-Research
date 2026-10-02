#!/usr/bin/env bash
# LAPTOP side: rsync this working tree to Casper. The "edit locally, run
# remotely" loop -- no commit needed, and it carries the two things git cannot:
# uncommitted edits, and the gitignored ARM observation files (*.xlsx) that
# figures 2 and 3 read.
#
#   ./sync_to_casper.sh abuggee@casper.hpc.ucar.edu
#   DRY=1 ./sync_to_casper.sh abuggee@casper.hpc.ucar.edu     # preview
#
# Local ERA5 downloads (data/) are never sent: on Casper the archive is read
# straight off GLADE.
set -euo pipefail
DEST_HOST="${1:-}"
[ -n "$DEST_HOST" ] || { echo "usage: $0 <user>@casper.hpc.ucar.edu"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"          # .../Python-Research
DEST_ROOT="${DEST_ROOT:-Python-Research}"

FLAGS=(-avz --progress
       --exclude 'data/' --exclude 'figures/' --exclude 'logs/' --exclude '__pycache__/'
       --exclude '.git/' --exclude '.ipynb_checkpoints/' --exclude 'aws_pipeline/results/'
       --exclude '*.nc' --exclude '*.png' --exclude '*.pdf' --exclude 'nohup.out'
       --exclude '*.pkl')
[ "${DRY:-0}" = 1 ] && FLAGS+=(--dry-run)

ssh "$DEST_HOST" "mkdir -p $DEST_ROOT/ERA5 $DEST_ROOT/Presentations_and_papers"
rsync "${FLAGS[@]}" "$REPO_ROOT/ERA5/surface_energy_budget/" \
      "$DEST_HOST:$DEST_ROOT/ERA5/surface_energy_budget/"
rsync "${FLAGS[@]}" "$REPO_ROOT/Presentations_and_papers/Ocean_Visions/" \
      "$DEST_HOST:$DEST_ROOT/Presentations_and_papers/Ocean_Visions/"
cat <<MSG

Synced to $DEST_HOST:$DEST_ROOT
On Casper:
  cd $DEST_ROOT/ERA5/surface_energy_budget/aws_pipeline/casper
  ./setup_casper.sh          # first time only
  python preflight.py
MSG
