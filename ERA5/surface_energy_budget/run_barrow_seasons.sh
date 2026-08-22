#!/bin/bash
# Sequential ERA5 downloads for Barrow, one season at a time.
# CDS only serves one request at a time, so these run strictly back-to-back.
#
# Usage:  mkdir -p logs && nohup ./run_barrow_seasons.sh > logs/runner.log 2>&1 &
# (the mkdir must happen in the calling shell -- the redirect is set up before
#  this script's own mkdir ever runs)

cd "$(dirname "$0")" || exit 1

LOGDIR="logs"
mkdir -p "$LOGDIR"

SEASONS=(
  "2011-08-01 2012-03-31"
  "2012-08-01 2013-03-31"
  "2013-08-01 2014-03-31"
)

for season in "${SEASONS[@]}"; do
  read -r start end <<< "$season"
  log="$LOGDIR/barrow_${start}_${end}.log"
  echo "=== starting $start to $end at $(date) ===" | tee -a "$log"

  python3 download_era5_seb.py \
    --region barrow \
    --chunk-days 15 \
    --jobs 1 \
    --start "$start" \
    --end "$end" \
    --storage local >> "$log" 2>&1

  rc=$?
  echo "=== finished $start to $end at $(date), exit code $rc ===" | tee -a "$log"
  # No `exit` on failure: the script is resumable, and a bad season should not
  # block the remaining ones from using the night.
done

echo "all seasons done at $(date)"
