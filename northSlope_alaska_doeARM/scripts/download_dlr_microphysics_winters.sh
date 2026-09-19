#!/usr/bin/env bash
# Bulk download for the "DLR vs cloud microphysics" notebook
# (dlr_cloud_microphysics_barrow.ipynb): two complete cold seasons at NSA C1,
# September through April, 2023/24 and 2024/25.
#
# Why these two seasons: microbasepi2 (the product the notebook was first
# scoped around) ends in March 2011, before thermocldphase begins (Nov 2011),
# so the phase and microphysics products never coexist. The successor MICROBASE
# (nsamicrobaseC1.c1) overlaps thermocldphase c0 from Oct 2020 on; 2023/24 and
# 2024/25 are the two most recent seasons with complete thermocldphase, QCRAD
# and MICROBASE coverage (2025/26 thermocldphase stops 2026-01-20 and QCRAD
# pyrgeometer 1 failed Dec 2025 - Feb 2026, see README).
#
# Datastreams (keys from arm_nsa/config.py), ~0.5 MB/day unless noted:
#   qcrad          QCRAD1LONG c2 (best-estimate LWD; c1 already on disk)
#   met            2-m T/RH, wind, pressure
#   mwr            MWRRET c2/c1 LWP + PWV (backup to thermocldphase's mwr_lwp_be)
#   gndirt         downward-looking IRT -> skin temperature (~1 MB/day)
#   cldtype        cloud-type classification, 1-min reflectivity, precip (~9 MB/day)
#   thermocldphase 30-s phase, layer boundaries, radar moments, MPL, sondes, LWP
#                  (~65 MB/day -> ~32 GB for the two seasons; the long pole)
# MICROBASE is NOT fetched here: at 670 MB/day it is reduced-and-deleted by
# scripts/reduce_microbase.py instead.
#
# Re-runnable: existing files are skipped. Run from the repo root:
#   nohup bash scripts/download_dlr_microphysics_winters.sh > data/download_logs/dl_winters.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
SEASONS=("2023-09-01 2024-04-30" "2024-09-01 2025-04-30")
SMALL="qcrad met mwr gndirt cldtype"
for s in "${SEASONS[@]}"; do
  set -- $s
  echo "=== $(date '+%F %T') small datastreams $1 .. $2 ==="
  python scripts/download_nsa_data.py --datastreams $SMALL --start "$1" --end "$2"
done
for s in "${SEASONS[@]}"; do
  set -- $s
  echo "=== $(date '+%F %T') thermocldphase $1 .. $2 ==="
  python scripts/download_nsa_data.py --datastreams thermocldphase --start "$1" --end "$2"
done
echo "=== $(date '+%F %T') all done ==="
