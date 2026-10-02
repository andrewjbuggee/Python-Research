#!/usr/bin/env bash
# LAPTOP side: bring results back.
#   ./fetch_from_casper.sh abuggee@casper.hpc.ucar.edu                 # newest run
#   ./fetch_from_casper.sh abuggee@casper.hpc.ucar.edu arctic_circle_2014-2024_2026...
set -euo pipefail
DEST_HOST="${1:-}"; RUN="${2:-}"
[ -n "$DEST_HOST" ] || { echo "usage: $0 <user>@casper.hpc.ucar.edu [run-directory]"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_ROOT="$(ssh "$DEST_HOST" 'source ~/.era5_casper_env && echo $RESULTS_ROOT')"
[ -n "$RUN" ] || RUN="$(ssh "$DEST_HOST" "ls -1t '$REMOTE_ROOT' | head -1")"
LOCAL="$HERE/../results/$RUN"
mkdir -p "$LOCAL"
rsync -avz --progress "$DEST_HOST:$REMOTE_ROOT/$RUN/" "$LOCAL/"
echo "-> $LOCAL"
