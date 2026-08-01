#!/usr/bin/env python3
"""Build the sonde-coordinated multi-instrument library (Hartig26 Sect. 2).

For every radiosonde launch in the date range, averages MWR / ceilometer /
KAZR (and optionally MET / QCRAD) over the following hour and stores one
aligned record per sounding, together with the gridded sounding profiles and
saturated-layer statistics. Output goes to data/processed/ as netCDF.

Examples
--------
# Everything already downloaded for January 2022, radar included:
python scripts/build_sonde_library.py --start 2022-01-01 --end 2022-01-31

# While the (huge) radar download is still running, build without it:
python scripts/build_sonde_library.py --start 2022-01-01 --end 2022-01-31 \
    --skip-radar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa.cli import add_data_root_argument, apply_data_root  # noqa: E402
from arm_nsa.coordinate import build_library, save_library  # noqa: E402
from arm_nsa.phase import classify_phase, phase_occurrence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--skip-radar",
        action="store_true",
        help="build without KAZR (radar-derived variables become NaN)",
    )
    parser.add_argument(
        "--no-extensions",
        action="store_true",
        help="strict Hartig26 instrument set (skip MET / QCRAD columns)",
    )
    parser.add_argument(
        "--output", default=None, help="output filename (default: auto-named)"
    )
    add_data_root_argument(parser)
    args = parser.parse_args()
    apply_data_root(args.data_root)

    library = build_library(
        args.start,
        args.end,
        include_extensions=not args.no_extensions,
        skip_radar=args.skip_radar,
    )

    # Attach the phase classification so downstream plots get it for free.
    library["phase_code"] = classify_phase(library)
    stats = phase_occurrence(library["phase_code"])
    print("\nSky-state occurrence over classified soundings:")
    for name, value in stats.items():
        if name == "n_classified":
            print(f"  {name}: {value:.0f}")
        else:
            print(f"  {name}: {value:.1%}")

    path = save_library(library, args.output)
    print(f"\nLibrary written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
