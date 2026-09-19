#!/usr/bin/env python3
"""Build per-day THERMOCLDPHASE column-feature files for the DLR study.

Wraps arm_nsa.column_features.build_column_files: every local
nsathermocldphaseC1.c0 day in the range is reduced to a (time,) file of
scalar column features under data/processed/dlr_columns/<lidar>/. Days
already reduced are skipped, so this can be re-run while the raw download
(scripts/download_dlr_microphysics_winters.sh) is still in progress.

    python scripts/build_dlr_columns.py --start 2023-09-01 --end 2025-04-30
    python scripts/build_dlr_columns.py --start 2023-09-01 --end 2025-04-30 --lidar hsrl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa.column_features import build_column_files  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--lidar", default="mplgr", choices=("mplgr", "hsrl"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    counts = build_column_files(a.start, a.end, lidar=a.lidar, verbose=not a.quiet)
    print(f"done={counts['done']} skipped={counts['skipped']} failed={counts['failed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
