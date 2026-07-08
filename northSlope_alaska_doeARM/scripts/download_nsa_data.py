#!/usr/bin/env python3
"""Download ARM NSA data for the mixed-phase cloud pipeline.

Examples
--------
# One winter month of the light Hartig instruments (sonde, MWR, ceilometer):
python scripts/download_nsa_data.py --datastreams sonde mwr ceil \
    --start 2022-01-01 --end 2022-01-31

# Add the radar for a couple of days (KAZR files are LARGE -- see below):
python scripts/download_nsa_data.py --datastreams kazr \
    --start 2022-01-05 --end 2022-01-06

# The extension datastreams for radiation / surface temperature plots:
python scripts/download_nsa_data.py --datastreams met qcrad \
    --start 2022-01-01 --end 2022-01-31

# The Shupe-Turner phase product + QCRAD for the phase/radiation analysis
# (product coverage is approximately 2004-2019):
python scripts/download_nsa_data.py --datastreams shupeturn qcrad \
    --start 2015-01-01 --end 2015-03-31

Data volume guidance (approximate, per day):
    sonde  ~1-2 MB     mwr   ~5 MB      ceil      ~5-10 MB
    met    ~1 MB       qcrad ~5 MB      shupeturn ~10-50 MB
    kazr   ~0.5-2 GB  (!)
A full 12-winter KAZR archive is on the order of 1 TB: for that scale, use
ARM's co-located computing (Data Workbench / Cumulus cluster, see README)
instead of downloading locally, or download month-by-month and process with
build_sonde_library.py as you go.

Requires ARM Live credentials -- see arm_nsa/credentials.py docstring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402
from arm_nsa.download import download_datastream  # noqa: E402

HARTIG_CORE_KEYS = ["sonde", "mwr", "ceil", "kazr"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datastreams",
        nargs="+",
        default=["sonde", "mwr", "ceil"],
        help=(
            "Pipeline keys to download: "
            f"{sorted(config.DATASTREAMS)} or 'hartig' for the full Hartig26 "
            "set incl. radar. Default: sonde mwr ceil (the light ones)."
        ),
    )
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download files that already exist locally",
    )
    args = parser.parse_args()

    keys = args.datastreams
    if keys == ["hartig"]:
        keys = HARTIG_CORE_KEYS

    print(f"Data root: {config.DATA_ROOT}")
    total = 0
    for key in keys:
        print(f"\n=== {key} ===")
        paths = download_datastream(
            key, args.start, args.end, overwrite=args.overwrite
        )
        total += len(paths)
    print(f"\nDone. {total} file(s) present locally for {args.start}..{args.end}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
