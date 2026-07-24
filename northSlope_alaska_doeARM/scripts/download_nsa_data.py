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

# Store the data on an external drive instead of inside the repo (raw/ and
# processed/ are created under the given root; pass the same --data-root to the
# analysis scripts to read it back):
python scripts/download_nsa_data.py --datastreams qcrad \
    --start 2018-01-01 --end 2019-01-01 --data-root /Volumes/ARM_NSA/data

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
import os
import sys
from pathlib import Path

# Make the repo importable when running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402
from arm_nsa.download import download_datastream  # noqa: E402

HARTIG_CORE_KEYS = ["sonde", "mwr", "ceil", "kazr"]


def verify_writable_root(data_root: Path) -> None:
    """Refuse to download into an unmounted external-drive path.

    On macOS an external drive is at /Volumes/<name>; if it is NOT mounted,
    mkdir(parents=True) would silently build the raw/ tree on the internal disk
    under that path, and the data would be stranded (invisible) once the real
    drive mounts later. Require the mount point -- or, off /Volumes, the data
    root's parent -- to already exist so a typo or unmounted drive fails loudly.
    """
    parts = data_root.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        mount_point = Path(*parts[:3])  # /Volumes/<name>
        if not (os.path.ismount(mount_point) or mount_point.exists()):
            raise SystemExit(
                f"error: {mount_point} is not mounted -- refusing to write to "
                f"{data_root}.\n  Mount the drive (check its name with "
                f"`ls /Volumes`) and re-run."
            )
        return
    if not data_root.parent.exists():
        raise SystemExit(
            f"error: parent directory {data_root.parent} does not exist -- "
            f"refusing to create {data_root}.\n  Check the path, or that the "
            f"external drive is mounted."
        )


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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Root directory for downloaded data; raw/ and processed/ are "
            "created under it. Point this at an external drive to store data "
            "off the internal disk, e.g. --data-root /Volumes/ARM_NSA/data. "
            "Overrides the ARM_NSA_DATA_ROOT env var; default is <repo>/data."
        ),
    )
    args = parser.parse_args()

    keys = args.datastreams
    if keys == ["hartig"]:
        keys = HARTIG_CORE_KEYS

    if args.data_root is not None:
        root = config.set_data_root(args.data_root)
        verify_writable_root(root)

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
