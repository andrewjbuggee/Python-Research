#!/usr/bin/env python3
"""Download the ARM NSA C1 (Utqiagvik / Barrow) observations for a surface
energy budget analysis following Equation (1) of Sledd et al. (2025).

    Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy
    balance responses to radiative forcing in the central Arctic from MOSAiC and
    models. JGR Atmospheres, 130, e2024JD042578.
    https://doi.org/10.1029/2024JD042578

This is the observational twin of
``ERA5/surface_energy_budget/download_era5_seb.py``: same equation, same tiers
(--var-set core / recommended / extended), same resumable per-chunk downloads,
same manifest -- but the "chunks" are ARM's daily files and the source is the
ARM Live web service rather than the CDS. Read arm_nsa/seb.py for the
term-by-term account of what NSA measures, what it does not, and why each
datastream is in its tier.

    LWD - LWU + SWD - SWU - SWT - SH - LH + G = M        (Eq. 1)

    measured at Barrow ........ LWD, LWU, SWD, SWU   (QCRAD, 1-min)
    parameterized from obs .... SH, LH               (bulk_flux.py; inputs
                                                       GNDIRT, MET, tower)
    estimated ................. G                    (ground_flux.py)
    zero in polar night ....... SWT, M

Cloud state for the ERA5 comparison: LWP (MWRRET + MWR3C), cloud boundaries
(ARSCL), a 1-min reflectivity profile for IWP (CLDTYPE), cloud phase
(THERMOCLDPHASE) and temperature profiles (sondes) -- see --var-set.


OPTIONS
=======

Which period
------------
``--season YYYY``
    The cold season 1 Oct YYYY .. 31 Mar YYYY+1 (the convention already used
    by the Barrow cloud-susceptibility work). Default: 2025, i.e.
    2025-10-01 .. 2026-03-31.
``--start YYYY-MM-DD`` / ``--end YYYY-MM-DD``
    An arbitrary inclusive range instead of --season.

Which datastreams
-----------------
``--var-set {core,recommended,extended}``   (default: recommended)
    core (6 keys, ~4 MB/day, < 1 GB per season)
        qcrad     QCRAD1LONG broadband LW/SW up/down, QC'd, 1-min
        gndirt    ground-looking IRT skin temperature, 1-min
        met       2-m T/RH, 10-m wind, pressure, PWD precipitation, 1-min
        twr       40-m tower T/RH/wind at 2/10/20/40 m, 1-min
        mwr       MWRRET LWP and PWV (c2 where processed, else c1)
        arsclbnd  ARSCL cloud-layer base/top heights, 4-s
    recommended (11 keys, ~100 MB/day, ~15 GB per season)
        + mwr3c           3-channel MWR LWP/PWV, ~1-s (21 MB/day) -- the
                          continuous cross-check on MWRRET, which has gaps
                          and a bias episode this season
        + cldtype         cloud type + 1-min ARSCL reflectivity profile for a
                          Z-based IWP + precipitation (9 MB/day)
        + sonde           raw radiosonde launches, ~4/day (cloud temperature)
        + thermocldphase  pixel cloud phase + embedded sonde T (70 MB/day;
                          processed through 2026-01-20 at the time of writing)
        + sirs            merged SKYRAD+GNDRAD b1: the SECOND downwelling
                          pyrgeometer (its disagreement with the first is the
                          LWD uncertainty), case/dome temperatures, and the
                          un-QC'd fallback for dates QCRAD has not processed
    extended (16 keys, ~190 MB/day)
        + interpsonde     1-min interpolated sonde profiles (61 MB/day)
        + ceil            ceilometer cloud base + backscatter
        + mplcmask        MPL cloud mask + depolarization (phase proxy)
        + ecor_e10        MEASURED SH/LH -- at OLIKTOK POINT (E10), not Barrow
        + sebs_e10        MEASURED soil heat flux -- at OLIKTOK POINT (E10)

``--datastreams KEY [KEY ...]``
    Explicit pipeline keys (or raw ARM datastream names containing a "."),
    overriding --var-set. This is the only way to get the very large
    products: ``microbase`` (LWC/IWC profiles, 670 MB/day) and ``kazr``.

Where the files go
------------------
``--data-root DIR``
    Root of the data tree (parent of raw/ and processed/); default <repo>/data
    or $ARM_NSA_DATA_ROOT. Files land in raw/<arm_datastream>/ exactly as the
    other download scripts put them, so every reader in arm_nsa finds them.

Run behaviour
-------------
``--dry-run``
    Query the archive for the file count of every datastream in the plan,
    print the per-key coverage and size estimate, and write nothing. Run this
    first: it is how you find out that a VAP has not been processed yet.
``--overwrite``
    Re-download files that already exist locally. Without it existing files
    are skipped, which is what makes an interrupted run resumable.

Manifest
--------
Every non-dry run writes processed/seb_manifest_<start>_<end>.json recording
the tiers, keys, ARM datastream names, files in the archive vs. on disk,
first/last file dates, on-disk sizes, and the list of SEB instruments that do
not exist at C1 (seb.UNAVAILABLE_AT_C1) so the absence travels with the data.


EXAMPLES
========
See what the archive holds for the 2025/26 season, download nothing::

    python scripts/download_nsa_seb_data.py --dry-run

The core SEB terms only (minutes, < 1 GB)::

    python scripts/download_nsa_seb_data.py --var-set core

The recommended set for 2025/26 (default; ~15 GB)::

    python scripts/download_nsa_seb_data.py

The same onto an external drive::

    python scripts/download_nsa_seb_data.py --data-root '/Volumes/My Passport/SCRIPPS/DOE_ARM/NSA/data'

One month of MICROBASE (~20 GB) for an IWP cross-check::

    python scripts/download_nsa_seb_data.py --datastreams microbase \\
        --start 2025-12-01 --end 2025-12-31

Requires ARM Live credentials -- see arm_nsa/credentials.py docstring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Make the repo importable when running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config, seb  # noqa: E402
from arm_nsa.cli import add_data_root_argument, apply_data_root  # noqa: E402
from arm_nsa.download import download_datastream, local_files, query_files  # noqa: E402

DEFAULT_SEASON_START_YEAR = 2025
DEFAULT_VAR_SET = "recommended"
_FILE_DATE_RE = re.compile(r"\.(\d{8})\.\d{6}\.(nc|cdf)$")


def cold_season_bounds(start_year: int) -> tuple[str, str]:
    """("YYYY-10-01", "YYYY+1-03-31") for the cold season starting in `start_year`."""
    return (
        f"{start_year}-{config.COLD_SEASON_START_MONTH_DAY}",
        f"{start_year + 1}-{config.COLD_SEASON_END_MONTH_DAY}",
    )


def n_days_inclusive(start_date: str, end_date: str) -> int:
    s = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    e = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    if e < s:
        raise SystemExit(f"error: --end {end_date} precedes --start {start_date}")
    return (e - s).days + 1


def _dates_of(filenames: Sequence[str]) -> List[str]:
    out = []
    for f in filenames:
        m = _FILE_DATE_RE.search(f)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def arm_names_for(key: str) -> tuple[str, ...]:
    """Raw ARM datastream names behind a pipeline key (or the name itself)."""
    return (key,) if "." in key else config.get_spec(key).datastreams


def plan_table(
    keys: Sequence[str], start_date: str, end_date: str, query: bool
) -> List[Dict]:
    """One row per pipeline key: description, tier info, archive coverage, size."""
    n_days = n_days_inclusive(start_date, end_date)
    sizes = seb.estimate_size_gb(keys, n_days)
    rows: List[Dict] = []
    for key in keys:
        info = seb.SEB_STREAMS.get(key)
        row: Dict = {
            "key": key,
            "arm_datastreams": list(arm_names_for(key)),
            "term": info.term if info else "(not an SEB stream)",
            "site": info.site if info else ("E10" if "E10" in key else "C1"),
            "cadence": info.cadence if info else "",
            "mb_per_day": info.mb_per_day if info else None,
            "est_size_gb": None if sizes[key] != sizes[key] else round(sizes[key], 2),
            "coverage_note_2025_26": info.coverage_2025_26 if info else "",
        }
        if query:
            archive: Dict[str, Dict] = {}
            for name in row["arm_datastreams"]:
                files = query_files(name, start_date, end_date)
                dates = _dates_of(files)
                archive[name] = {
                    "n_files": len(files),
                    "first": dates[0] if dates else None,
                    "last": dates[-1] if dates else None,
                }
            row["archive"] = archive
        rows.append(row)
    return rows


def print_plan(
    rows: List[Dict], start_date: str, end_date: str, var_set_label: str
) -> None:
    n_days = n_days_inclusive(start_date, end_date)
    print("=" * 78)
    print("NSA C1 surface energy budget download (observations)")
    print("=" * 78)
    print(f"  Period       : {start_date} .. {end_date} ({n_days} days)")
    print(f"  Selection    : {var_set_label}")
    print(f"  Data root    : {config.DATA_ROOT}")
    total_gb = sum(r["est_size_gb"] or 0.0 for r in rows)
    print(
        f"  Size estimate: ~{total_gb:.1f} GB (measured per-day sizes x days; "
        "upper bound for lagging VAPs)"
    )
    print("-" * 78)
    for r in rows:
        site = "" if r["site"] == "C1" else f"  [{r['site']} -- NOT BARROW]"
        print(f"{r['key']:15s} {r['term']}{site}")
        print(
            f"{'':15s} {', '.join(r['arm_datastreams'])}; {r['cadence']}; "
            f"~{r['mb_per_day']} MB/day -> ~{r['est_size_gb']} GB"
        )
        if "archive" in r:
            for name, a in r["archive"].items():
                if a["n_files"]:
                    print(
                        f"{'':15s} archive {name}: {a['n_files']} files, "
                        f"{a['first']}..{a['last']}"
                    )
                else:
                    print(f"{'':15s} archive {name}: 0 files")
        if r["coverage_note_2025_26"]:
            print(f"{'':15s} note: {r['coverage_note_2025_26']}")
    print("-" * 78)


def print_unavailable() -> None:
    print(
        "\nNot fetched -- SEB instruments that do NOT exist at NSA C1 (ARM Live "
        "archive queried 2026-09-15, see arm_nsa/seb.py):"
    )
    for name, why in seb.UNAVAILABLE_AT_C1.items():
        print(f"  {name}\n      {why}")
    print(
        "  => SH and LH must come from the bulk parameterization "
        "(arm_nsa/bulk_flux.py); G from arm_nsa/ground_flux.py."
    )


def write_manifest(
    rows: List[Dict],
    start_date: str,
    end_date: str,
    var_set_label: str,
    downloaded: Dict[str, List[Path]],
) -> Path:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DATA_DIR / f"seb_manifest_{start_date}_{end_date}.json"
    # Inventory EVERY SEB stream on disk for the period, not just the keys of
    # this run, so successive runs (core now, sirs later, microbase for one
    # month) keep one complete, truthful manifest instead of overwriting each
    # other's view.
    on_disk: Dict[str, Dict] = {}
    inventory_keys = list(dict.fromkeys(list(seb.SEB_STREAMS) + list(downloaded)))
    for key in inventory_keys:
        for name in arm_names_for(key):
            d = config.raw_dir_for(name)
            files = [
                p for p in local_files(name) if _in_range(p.name, start_date, end_date)
            ]
            if not files:
                continue
            dates = _dates_of([p.name for p in files])
            on_disk[name] = {
                "key": key,
                "directory": str(d),
                "n_files": len(files),
                "first": dates[0] if dates else None,
                "last": dates[-1] if dates else None,
                "size_gb": round(sum(p.stat().st_size for p in files) / 1024**3, 3),
            }
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "site": "ARM NSA C1, Utqiagvik (Barrow), AK; E10 = Oliktok Point where noted",
        "reference": "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578, Eq. (1)",
        "period": {"start": start_date, "end": end_date},
        "selection_this_run": var_set_label,
        "data_root": str(config.DATA_ROOT),
        "datastreams_this_run": rows,
        "on_disk_all_seb_streams": on_disk,
        "unavailable_at_c1": seb.UNAVAILABLE_AT_C1,
        "variable_sets": {k: list(v) for k, v in seb.SEB_VARIABLE_SETS.items()},
    }
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _in_range(filename: str, start_date: str, end_date: str) -> bool:
    m = _FILE_DATE_RE.search(filename)
    if not m:
        return False
    d = m.group(1)
    return start_date.replace("-", "") <= d <= end_date.replace("-", "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g_time = parser.add_argument_group("time period")
    g_time.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="YYYY",
        help=f"cold season Oct 1 YYYY .. Mar 31 YYYY+1 (default {DEFAULT_SEASON_START_YEAR})",
    )
    g_time.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    g_time.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    g_vars = parser.add_argument_group("datastreams")
    g_vars.add_argument(
        "--var-set",
        choices=sorted(seb.SEB_VARIABLE_SETS),
        default=DEFAULT_VAR_SET,
        help=f"tier to download (default {DEFAULT_VAR_SET}); see the docstring",
    )
    g_vars.add_argument(
        "--datastreams",
        nargs="+",
        default=None,
        metavar="KEY",
        help="explicit pipeline keys or raw ARM names, overriding --var-set "
        "(e.g. microbase, kazr, nsaarsclkazr1kolliasC1.c0)",
    )
    g_run = parser.add_argument_group("run behaviour")
    g_run.add_argument(
        "--dry-run",
        action="store_true",
        help="query the archive and print the plan; download nothing",
    )
    g_run.add_argument(
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
        season = args.season if args.season is not None else DEFAULT_SEASON_START_YEAR
        start_date, end_date = cold_season_bounds(season)
    n_days_inclusive(start_date, end_date)  # validates ordering

    if args.datastreams:
        keys = list(args.datastreams)
        label = "explicit: " + " ".join(keys)
    else:
        keys = list(seb.seb_keys(args.var_set))
        label = f"--var-set {args.var_set} ({len(keys)} keys)"

    if args.data_root is not None:
        apply_data_root(args.data_root, require_raw=False, quiet=True)

    rows = plan_table(keys, start_date, end_date, query=True)
    print_plan(rows, start_date, end_date, label)

    if args.dry_run:
        print("\nDry run: nothing downloaded.")
        print_unavailable()
        return 0

    downloaded: Dict[str, List[Path]] = {}
    for key in keys:
        print(f"\n=== {key} ===")
        downloaded[key] = download_datastream(
            key, start_date, end_date, overwrite=args.overwrite
        )
    total = sum(len(v) for v in downloaded.values())
    manifest = write_manifest(rows, start_date, end_date, label, downloaded)
    print(f"\nDone. {total} file(s) present locally for {start_date}..{end_date}.")
    print(f"Manifest: {manifest}")
    print_unavailable()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
