#!/usr/bin/env python
"""Download everything the seasonal-averages check needs, then build the combined files.

ARM products (listed in SEASONAL_PRODUCTS, defined in config.yaml): ARM's
server extracts only the needed variables (and their qc_ companions) from
each daily file; downloads resume and files already present are skipped.
Time-series products are then merged into one netCDF per product in
<data folder>/processed/, which is what EPCAPE.analysis_tools.products.load_product reads.
Radiosonde products marked ``layout: per_launch`` are left as one file per
launch (see sources.read_per_launch).

UC San Diego Library files (AMS at Mt. Soledad, GCVI enhancement factors):
the files under ``ucsd_library:`` in config.yaml.

Examples (run from the EPCAPE folder)
  python comparisons/seasonal_averages/download_data.py              # everything
  python comparisons/seasonal_averages/download_data.py --dry-run    # what would be fetched
  python comparisons/seasonal_averages/download_data.py --only rain_ld_M1 rain_ld_S2
  python comparisons/seasonal_averages/download_data.py --skip-ucsd --no-combine

Approximate volume for the whole campaign (server-side subsets): KAZR-ARSCL
~0.5-0.7 GB, ceilometer ~0.2 GB, everything else < 0.1 GB each; UCSD files
~0.3 GB. The first file of each ARM datastream is fetched complete once to
read its variable list (up to ~160 MB for the disdrometer VAP files).

Credentials: ARM_USERNAME/ARM_TOKEN, or ~/.arm_credentials (see README).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# EPCAPE/ is itself the Python package: put its parent folder on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.download_data.armlive import ArmLiveError  # noqa: E402
from EPCAPE.download_data.combine import combine_product  # noqa: E402
from EPCAPE.download_data.config import active_machine, as_date, campaign_dates, get_product, load_config  # noqa: E402
from EPCAPE.analysis_tools.products import processed_path  # noqa: E402
from EPCAPE.download_data.sync import sync_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages.sources import (  # noqa: E402
    download_ucsd_library,
    product_layout,
)

# Ordered smallest download first, so the quick products are usable early.
# Spreadsheet rows each product serves (Cloud&MetQuantities sheet):
SEASONAL_PRODUCTS = [
    "pblh_thermo_M1",  # row 17  PBLH-Thermo (Zhang)
    "rain_ld_M1",  # row 22  rain, LD at M1
    "rain_ld_S2",  # row 23  rain, LD at Mt. Soledad (the sheet says "LDS1")
    "rain_vdis_M1",  # row 24  rain, VDIS
    "radflux_M1",  # rows 7-8 SW transmittance, LW down
    "lwp_mwrret1_M1",  # row 10  LWP
    "lwp_mwr_M1",  # row 10  LWP (MWRLOS, already used by the cloud-property comparison)
    "sondeparam_M1",  # row 20  LCL (SONDEPARAM)
    "pblh_sonde_M1",  # rows 17-20 sonde PBL heights and surface p/T/RH for LCL
    "cbh_ceil_M1",  # row 21  ceilometer cloud base
    "cloud_arscl_M1",  # row 15  KAZR cloud top
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", nargs="+", metavar="PRODUCT", help="just these ARM products")
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: campaign start)")
    parser.add_argument("--end", help="last day, inclusive (default: campaign end)")
    parser.add_argument("--skip-arm", action="store_true", help="do not download ARM products")
    parser.add_argument("--skip-ucsd", action="store_true", help="do not download UCSD Library files")
    parser.add_argument("--no-combine", action="store_true", help="download only; do not build combined files")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel ARM downloads (default: 1; ARM Live's subset service answered 502/503 to 4 parallel requests on 2026-10-06)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded, then stop")
    args = parser.parse_args(argv)

    cfg = load_config()
    machine = active_machine(cfg)
    print(f"Machine: {machine.name}   data folder: {machine.data_root}")
    c_start, c_end = campaign_dates(cfg)
    start = as_date(args.start) if args.start else c_start
    end = as_date(args.end) if args.end else c_end
    names = args.only or SEASONAL_PRODUCTS
    problems = {}

    if not args.skip_ucsd:
        try:
            download_ucsd_library(machine=machine, dry_run=args.dry_run)
        except (OSError, ValueError) as exc:  # requests errors are OSError subclasses
            problems["ucsd_library"] = str(exc)
            print(f"UCSD Library download failed: {exc}", file=sys.stderr)

    if not args.skip_arm:
        for name in names:
            print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
            try:
                product = get_product(name, cfg)
                result = sync_product(
                    product, start, end, machine=machine, workers=args.workers, dry_run=args.dry_run
                )
            except (ArmLiveError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
                problems[name] = str(exc)
                print(f"Error: {exc}", file=sys.stderr)
                continue
            if args.dry_run:
                continue
            print(
                f"{name}: {len(result.downloaded)} downloaded ({result.bytes_downloaded / 1e6:.1f} MB), "
                f"{len(result.skipped)} already present, {len(result.unavailable)} not available, "
                f"{len(result.failed)} failed"
            )
            if result.failed:
                problems[name] = f"{len(result.failed)} files failed (rerun to retry)"
            if args.no_combine or product_layout(name, cfg) == "per_launch":
                continue
            # Build (or rebuild after new downloads) the full-campaign combined file
            # that EPCAPE.analysis_tools.products.load_product reads.
            out = processed_path(name, machine, cfg)
            if out.is_file() and not result.downloaded:
                print(f"{out.name} is up to date")
                continue
            try:
                combine_product(product, c_start, c_end, machine=machine, out=out)
            except (ValueError, FileNotFoundError, OSError) as exc:
                problems[name] = f"combine failed: {exc}"
                print(f"Combine failed: {exc}", file=sys.stderr)

    if problems:
        print("\nProblems (rerun the same command to retry):")
        for name, why in problems.items():
            print(f"  {name}: {why}")
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
