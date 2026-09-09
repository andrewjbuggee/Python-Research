#!/usr/bin/env python3
"""Download one cold season of NSA surface/tower WIND data.

This is the narrow, wind-only companion to download_nsa_data.py. It exists so
the Taylor frozen-turbulence cloud-scale estimate (see the notebook
taylor_hypothesis_cloud_spatial_scale.ipynb) has a one-call way to fetch just
the meteorological datastreams that carry horizontal wind speed at Utqiagvik,
without pulling any of the radar/lidar/phase products.

Examples
--------
# The default: the 2025/26 cold season, 1 Oct 2025 - 31 Mar 2026 (~200 MB).
python scripts/download_nsa_wind_data.py

# A different cold season, named by the year it STARTS in.
python scripts/download_nsa_wind_data.py --season 2019

# An arbitrary range, and onto an external drive.
python scripts/download_nsa_wind_data.py --start 2020-10-01 --end 2021-03-31 \
    --data-root '/Volumes/My Passport/SCRIPPS/DOE_ARM/NSA/data'

Which datastreams, and why these
--------------------------------
`twr` -- nsatwrC1.b1, the "Forty Meter Tower: meteorological data, 2, 10, 20 &
40 m, 1-min avg" product -- is the one that matters, and on its own it covers
every level this analysis needs: it reports wind speed and direction on a
(time, height) grid at 2, 10, 20 and 40 m AGL, one record per minute.

`met` -- nsametC1.b1 -- is fetched alongside it as a cross-check and for the
station pressure/temperature. Note that MET is NOT an independent wind
measurement: its file header gives `wind_measurement_height: 10m` and its
`input_source` is the same METData collection the tower stream is ingested
from, so MET's wind speed is numerically identical to the tower's 10-m level.
It is included because it is ~0.4 MB/day and makes the identity checkable
rather than assumed.

TWO REQUESTED PRODUCTS ARE NOT AVAILABLE AT NSA
-----------------------------------------------
The ARM "Horizontal wind" measurement listing is site-agnostic; two of the
three instruments on it are not deployed at the NSA Central Facility. Queried
against the ARM Live archive on 2026-09-08, over 1998-2026 and every facility
code, the following returned zero files for every date range tried:

    30smos     Surface Meteorological Observation Station, 30-min averages
               (nsa30smosC1.b1, nsa30smosC2.b1, nsasmosC1.b1)
    mettiptwr  Ten Meter Tower: meteorological data, 2 & 6 m, 1-min avg
               (nsamettiptwrC1.b1, nsamettwr2hC1.b1)

Nothing is lost for this analysis: the 40-m tower already carries a 2-m level,
which is the near-surface wind those two products would have supplied, and it
does so at 1-min resolution rather than SMOS's 30-min averages.

Separately, note that the `mettwr` key registered in config.py is NOT the
40-m tower -- it is nsamettwrC1.b1, an early ingest that stops around 2003.
The modern tower stream is nsatwrC1.b1, registered here as `twr`.

Data volume: nsatwrC1.b1 is ~0.5 MB/day and nsametC1.b1 ~0.4 MB/day, so one
182-day cold season of both is roughly 165 MB. Re-runs are free -- files that
already exist locally are skipped.

Requires ARM Live credentials -- see arm_nsa/credentials.py docstring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Sequence

# Make the repo importable when running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402
from arm_nsa.cli import add_data_root_argument, apply_data_root  # noqa: E402
from arm_nsa.download import download_datastream  # noqa: E402

# Pipeline keys fetched by default. Order matters only for print output.
WIND_DATASTREAM_KEYS = ("twr", "met")

# Instruments on the ARM "Horizontal wind" listing that NSA C1 does not have;
# reported once at the end of a run so the absence is explicit rather than
# silently missing. See the module docstring for how this was checked.
UNAVAILABLE_AT_NSA = {
    "30smos": "Surface Meteorological Observation Station, 30-min averages",
    "mettiptwr": "Ten Meter Tower, 2 & 6 m, 1-min averages",
}


def cold_season_bounds(start_year: int) -> tuple[str, str]:
    """Return ("YYYY-10-01", "YYYY+1-03-31") for the cold season starting in `start_year`.

    The 1 Oct - 31 Mar convention matches the Barrow cloud-susceptibility work
    in this repo and spans the polar-night core at Utqiagvik.
    """
    start = f"{start_year}-{config.COLD_SEASON_START_MONTH_DAY}"
    end = f"{start_year + 1}-{config.COLD_SEASON_END_MONTH_DAY}"
    return start, end


def download_wind_season(
    start_date: str,
    end_date: str,
    keys: Sequence[str] = WIND_DATASTREAM_KEYS,
    data_root: "str | Path | None" = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """Download every wind-carrying NSA datastream for one date range.

    Parameters
    ----------
    start_date, end_date:
        Inclusive "YYYY-MM-DD" bounds.
    keys:
        Pipeline datastream keys; default ("twr", "met"). Raw ARM datastream
        names (anything containing a ".") are also accepted, as in
        download_datastream().
    data_root:
        Root of the data tree (the parent of raw/ and processed/). None keeps
        whatever config.DATA_ROOT currently is -- <repo>/data unless
        ARM_NSA_DATA_ROOT is set.
    overwrite:
        Re-download files that already exist locally. Default False, which
        makes the call cheap to repeat and safe to resume.

    Returns
    -------
    Local paths of every file present for the range, downloaded now or
    already on disk.

    Notes
    -----
    Nothing is read or decoded here -- this only puts files under
    <data_root>/raw/<arm_datastream>/. Read them back with
    arm_nsa.surface.read_tower_winds() / read_met().
    """
    if data_root is not None:
        # require_raw=False: this is an entry point that creates raw/ itself.
        apply_data_root(Path(data_root), require_raw=False, quiet=not verbose)

    if verbose:
        print(f"Data root: {config.DATA_ROOT}")
        print(f"Wind season: {start_date} .. {end_date}")

    paths: List[Path] = []
    for key in keys:
        if verbose:
            print(f"\n=== {key} ===")
        paths.extend(
            download_datastream(
                key, start_date, end_date, overwrite=overwrite, verbose=verbose
            )
        )
    return paths


def _default_season_year() -> int:
    """Most recent cold season that has already started, by today's date."""
    today = dt.date.today()
    return today.year if today.month >= 10 else today.year - 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="YYYY",
        help=(
            "Cold season named by the year it starts in: --season 2025 means "
            "2025-10-01..2026-03-31. Default: 2025. Mutually exclusive with "
            "--start/--end."
        ),
    )
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--datastreams",
        nargs="+",
        default=list(WIND_DATASTREAM_KEYS),
        help=(
            f"Pipeline keys to download. Default: {' '.join(WIND_DATASTREAM_KEYS)} "
            "('twr' is the 40-m tower and carries all four levels; 'met' is the "
            "10-m cross-check)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download files that already exist locally",
    )
    add_data_root_argument(parser)
    args = parser.parse_args()

    if args.season is not None and (args.start or args.end):
        parser.error("--season cannot be combined with --start/--end")
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be given together")

    if args.start:
        start_date, end_date = args.start, args.end
    else:
        season = args.season if args.season is not None else 2025
        start_date, end_date = cold_season_bounds(season)

    paths = download_wind_season(
        start_date,
        end_date,
        keys=args.datastreams,
        data_root=args.data_root,
        overwrite=args.overwrite,
    )

    print(f"\nDone. {len(paths)} file(s) present locally for {start_date}..{end_date}.")
    print(
        "\nNot fetched -- these ARM 'Horizontal wind' instruments are not "
        "deployed at NSA C1 (archive checked 2026-09-08):"
    )
    for code, what in UNAVAILABLE_AT_NSA.items():
        print(f"  {code:11s} {what}")
    print(
        "  The 40-m tower's 2-m level supplies the near-surface wind those "
        "would have provided."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
