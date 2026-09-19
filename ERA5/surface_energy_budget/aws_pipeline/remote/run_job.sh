#!/usr/bin/env bash
# On the instance: run run_analysis.py detached, logging to the results dir.
#
#   ./run_job.sh --region beaufort_chukchi --years 2014-2024 --stages all
#   tail -f /data/results/latest.log
#
# Every argument is passed straight to run_analysis.py; --out defaults to
# $RESULTS_ROOT/<timestamp>. The job survives the ssh session ending (nohup).
set -euo pipefail
source "$HOME/.era5_env"
export MAMBA_ROOT_PREFIX
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
micromamba activate era5-seb

# The working tree may have been rsync'ed from the laptop (sync_code.sh), in
# which case a pull would conflict: only pull when the tree is clean.
if [ -z "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
  git -C "$REPO_DIR" pull --ff-only || echo "(git pull failed; running the checked-out code)"
else
  echo "(working tree has synced/local changes; not pulling)"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RESULTS_ROOT/$STAMP"
mkdir -p "$OUT"
ln -sfn "$OUT" "$RESULTS_ROOT/latest"
ln -sfn "$OUT/run.log" "$RESULTS_ROOT/latest.log"
cd "$SEB_DIR/aws_pipeline"
echo "job $STAMP: run_analysis.py $* --out $OUT" | tee "$OUT/run.log"
nohup python run_analysis.py "$@" --out "$OUT" >> "$OUT/run.log" 2>&1 &
echo "pid $! ; tail -f $OUT/run.log"
