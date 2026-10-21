"""
Make a small, laptop-sized copy of the J09 OHC output: OHC_global_mean only, no OHC maps.

J09 writes one *.pop.h.OHC.*.nc per POP TEMP file, holding
    OHC              (ohc_depth, time, nlat, nlon) [J m-2]  full maps   (~4 MB per month)
    OHC_global_mean  (time, ohc_depth)             [J m-2]  ocean-area-weighted global mean
plus attrs["ocean_area_m2"] (and "global_area_m2"). J04 only uses OHC_global_mean and
ocean_area_m2, so this script copies exactly those into a tree with the SAME sub-folders and
file names. J04/J15 find files by name (*<case>*.OHC.*nc), so the copy is a drop-in
replacement for spatial_OHC_data: on the laptop, put it at data/spatial_OHC_data and J04
uses it automatically (see USE_PLACEHOLDER_OHC in J04).

Run on Casper (reads the real OHC files; each file only has its small global-mean variable read):
    python Jxx_extract_ohc_globalmean.py
    python Jxx_extract_ohc_globalmean.py --cases ARISE-1.0 CESM2_WACCM_HIST   # subset
Defaults: input = prism_paths.SPATIAL_OHC_DIR, output = <WORK_ROOT>/spatial_OHC_globalmean.
Existing output files are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import os
from pathlib import Path

import xarray as xr

from prism_paths import SPATIAL_OHC_DIR, WORK_ROOT

# J09 CASE_CONFIG keys (= top-level folders of spatial_OHC_data) for the cases J04 plots.
# ARISE-1.0 has its own folder in the OHC tree (it sits under ARISE_SAI/ only in the atmosphere tree).
J04_OHC_CASES = [
    "CESM2_LME",
    "CESM2_WACCM_1850control",
    "CESM2_WACCM_HIST",
    "CESM2_WACCM_SSP2-4.5",
    "ARISE_SAI",
    "ARISE-1.0",
    "CESM2_WACCM_SSP2-4.5_MCB",
]

# Variables J04 needs; their coordinates (time, ohc_depth) come along automatically
KEEP_VARS = ["OHC_global_mean"]


def extract_one(src: Path, dst: Path) -> None:
    """Copy KEEP_VARS (with their coordinates and all file attributes) from src to dst."""
    # decode_times=False copies the raw time values and their units/calendar attributes unchanged,
    # so the small file decodes exactly like the original (no cftime round trip)
    with xr.open_dataset(src, decode_times=False, decode_timedelta=False) as ds:
        # Selecting by name keeps only coordinates whose dimensions belong to these variables, so
        # 2-D grid coordinates (nlat, nlon) of the OHC map are dropped with it. Dataset attrs
        # (ocean_area_m2, global_area_m2, description) are kept.
        small = ds[KEEP_VARS].load()
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first, so a crash never leaves a truncated file that a restart
    # would mistake for a finished one
    tmp = dst.with_name(dst.name + ".tmp")
    small.to_netcdf(tmp)
    os.replace(tmp, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=SPATIAL_OHC_DIR, help="J09 OHC tree (default: %(default)s)")
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "spatial_OHC_globalmean", help="output tree (default: %(default)s)"
    )
    parser.add_argument("--cases", nargs="+", default=J04_OHC_CASES, help="top-level case folders to copy")
    parser.add_argument("--overwrite", action="store_true", help="redo files that already exist in --out")
    args = parser.parse_args()

    n_done = n_skipped = 0
    failed = []
    for case in args.cases:
        case_dir = args.src / case
        if not case_dir.is_dir():
            print(f"WARNING: no folder {case_dir}; skipping")
            continue
        # rglob follows the J09 layout below the case folder (e.g. .../ocn/proc/tseries/month_1/)
        files = sorted(case_dir.rglob("*.OHC.*nc"))
        print(f"{case}: {len(files)} files")
        for src in files:
            dst = args.out / src.relative_to(args.src)
            if dst.exists() and not args.overwrite:
                n_skipped += 1
                continue
            try:
                extract_one(src, dst)
                n_done += 1
            except (OSError, KeyError, ValueError) as err:
                # KeyError: file lacks OHC_global_mean; OSError/ValueError: unreadable or corrupt file
                failed.append(f"{src}: {err!r}")

    total_mb = sum(f.stat().st_size for f in args.out.rglob("*.nc")) / 1e6 if args.out.is_dir() else 0.0
    print(f"\nwrote {n_done}, skipped {n_skipped} existing, failed {len(failed)}")
    print(f"output: {args.out}  ({total_mb:.1f} MB)")
    for line in failed:
        print("  FAILED", line)


if __name__ == "__main__":
    main()
