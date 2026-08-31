#!/usr/bin/env python3
"""Download only the ERA5 the SASSIE comparison is missing.

The R/V *Woldstad* ran 69.20-73.52 N, 165.96-144.89 W during the SASSIE fall
2022 campaign. Enclosing that track on the 0.25 deg ERA5 grid with one cell of
margin takes

    69.00-73.75 N, 166.25-144.50 W        (20 lat x 88 lon = 1760 cells)

and the ``barrow`` archive already under ``../data/barrow`` (70-80 N,
165-150 W) supplies the middle 976 of those cells. This script fetches the
other 784, as three tiles that abut the Barrow strip without overlapping it or
each other:

    sassie_gap_south   69.00-69.75 N   166.25-144.50 W     4 x 88 = 352 cells
    sassie_gap_west    70.00-73.75 N   166.25-165.25 W    16 x  5 =  80 cells
    sassie_gap_east    70.00-73.75 N   149.75-144.50 W    16 x 22 = 352 cells

    352 + 80 + 352 + 976 (barrow) = 1760.  Exact tiling, verified in
    ``era5_sassie_mosaic.check_tiling()``.

Downloading the whole box instead would re-fetch the 976 cells already on disk
and roughly triple the request.

Chunking is half-monthly to keep the run to **9 CDS requests** rather than the
75 that the default one-file-per-day chunking would submit. At 35 variables and
25 days the whole pull is on the order of 20 MB.

Everything here delegates to ``../download_era5_seb.py``: same CDS credentials,
same variable set, same ``avg_*`` name normalisation, same resume-in-date-space
and manifest behaviour. Re-running skips tiles already on disk.

    python download_era5_sassie_gap.py --dry-run     # plan only, no CDS contact
    python download_era5_sassie_gap.py               # do it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import download_era5_seb  # noqa: E402

# --- What to fetch ----------------------------------------------------------
# The shipboard MET record runs 2022-09-09T01:08Z to 2022-10-01T23:48Z. One day
# of padding each side covers the hour-ending accumulation windows at the edges.
START_DATE = "2022-09-08"
END_DATE = "2022-10-02"

GAP_REGIONS = ("sassie_gap_south", "sassie_gap_west", "sassie_gap_east")

# Matches the chunking of the existing barrow files (era5_seb_barrow_202209_01-15.nc),
# so the mosaic loader sees the same file granularity everywhere.
CHUNK_DAYS = 15

# Same 35-variable set as the barrow archive; the mosaic requires identical
# variables across tiles.
VAR_SET = "recommended"

OUT_DIR = HERE / "data"


def build_argv(region: str, dry_run: bool, overwrite: bool) -> list[str]:
    """Command line for one tile, as ``download_era5_seb.main`` expects it."""
    argv = [
        "--region", region,
        "--start", START_DATE,
        "--end", END_DATE,
        "--var-set", VAR_SET,
        "--chunk-days", str(CHUNK_DAYS),
        "--out-dir", str(OUT_DIR),
    ]
    if dry_run:
        argv.append("--dry-run")
    if overwrite:
        argv.append("--overwrite")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the ERA5 tiles the SASSIE comparison is missing.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each tile's plan without contacting the CDS.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download tiles whose files already exist.")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for i, region in enumerate(GAP_REGIONS, 1):
        print("=" * 72)
        print(f"[{i}/{len(GAP_REGIONS)}] {region}")
        print("=" * 72)
        status = download_era5_seb.main(build_argv(region, args.dry_run, args.overwrite))
        if status != 0:
            failures.append(region)
            print(f"!! {region} returned status {status}", file=sys.stderr)

    print("=" * 72)
    if failures:
        print(f"FAILED tiles: {', '.join(failures)}", file=sys.stderr)
        print("Re-run this script to retry them; finished tiles are skipped.",
              file=sys.stderr)
        return 1

    if not args.dry_run:
        print("All three gap tiles downloaded. Mosaic them with the Barrow strip via")
        print("    from era5_sassie_mosaic import load_sassie_box")
        print("    era5 = load_sassie_box('2022-09-08', '2022-10-02')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
