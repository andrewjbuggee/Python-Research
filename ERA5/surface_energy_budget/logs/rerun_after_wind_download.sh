#!/bin/bash
# Waits for the wind download job (pid given as $1) to exit, then re-executes
# cloud_spatial_extent.ipynb so section 6 draws the cloud-level-wind figures.
# Written by Claude on 2026-09-16; safe to delete once it has run.
cd "$(dirname "$0")/.." || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== download job $1 exited at $(date); re-executing the notebook ==="
n_files=$(ls data/barrow_pressure_wind/*.nc 2>/dev/null | wc -l)
echo "wind files on disk: $n_files"
JUPYTER_CONFIG_DIR=$(mktemp -d) /opt/anaconda3/bin/jupyter nbconvert --to notebook \
    --execute --inplace --ExecutePreprocessor.timeout=3600 cloud_spatial_extent.ipynb
echo "=== notebook re-executed with exit $? at $(date) ==="
