#!/usr/bin/env python
"""Download EPCAPE data from the ARM Live web service.

Examples
  python download_arm.py cbh_ceil_M1          # a product from config.yaml: only its variables
  python download_arm.py cbh_ceil_M1 --full   # the same datastream as complete files
  python download_arm.py --datastream epcceilM1.b1 --start 2023-07-01 --end 2023-07-03
  python download_arm.py cbh_ceil_M1 --dry-run
  python download_arm.py --list               # products, active machine, data folder

Re-running the same command resumes: files already on disk are skipped.
Credentials: ARM_USERNAME/ARM_TOKEN, or ~/.arm_credentials (prompted on first run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The EPCAPE folder is itself the Python package, so its parent folder must be on
# sys.path for `import EPCAPE` to work when this script is run from inside it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EPCAPE.armlive import ArmLiveError
from EPCAPE.config import active_machine, as_date, campaign_dates, config_path, get_product, load_config
from EPCAPE.sync import sync_datastream, sync_product


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("product", nargs="?", help="product name from config.yaml")
    parser.add_argument("--datastream", help="download complete files of this datastream")
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: campaign start)")
    parser.add_argument("--end", help="last day, inclusive (default: campaign end)")
    parser.add_argument("--full", action="store_true", help="complete files for the product's datastream")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="download again even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded, then stop")
    parser.add_argument("--list", action="store_true", help="show configured products and the data folder")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        machine = active_machine(cfg)
        print(f"Machine: {machine.name}   data folder: {machine.data_root}")
        if args.list:
            print(f"Products in {config_path()}:")
            for name, p in (cfg.get("products") or {}).items():
                print(f"  {name:<16} {p.get('datastream', '?'):<16} {p.get('description', '')}")
            return 0
        if not args.product and not args.datastream:
            parser.error("give a product name or --datastream")

        c_start, c_end = campaign_dates(cfg)
        start = as_date(args.start) if args.start else c_start
        end = as_date(args.end) if args.end else c_end
        if end < start:
            parser.error("--end is before --start")
        common = dict(machine=machine, workers=args.workers, overwrite=args.overwrite, dry_run=args.dry_run)

        if args.datastream:
            result = sync_datastream(args.datastream, start, end, **common)
        else:
            product = get_product(args.product, cfg)
            if args.full:
                result = sync_datastream(product.datastream, start, end, **common)
            else:
                result = sync_product(product, start, end, **common)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
        return 130
    except (ArmLiveError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2

    if args.dry_run or result.source == "archive":
        return 0
    print(
        f"\nDone: {len(result.downloaded)} downloaded ({result.bytes_downloaded / 1e6:.1f} MB), "
        f"{len(result.skipped)} already present, {len(result.unavailable)} not available, "
        f"{len(result.failed)} failed."
    )
    if result.downloaded and result.full_file_bytes:
        mean = result.bytes_downloaded / len(result.downloaded)
        print(f"Subset files average {mean / 1e6:.2f} MB; one complete file is "
              f"{result.full_file_bytes / 1e6:.2f} MB.")
    if result.fallbacks:
        print(f"{len(result.fallbacks)} files could not be subset by ARM's server and were fetched "
              "whole and subset here (details in manifest.json).")
    if result.unavailable:
        print("Not available through ARM Live (order these through ARM Data Discovery):")
        for name in result.unavailable:
            print(f"  {name}")
    if result.failed:
        print("Failed (run the same command again to retry just these):")
        for name, why in sorted(result.failed.items()):
            print(f"  {name}: {why}")
    print(f"Files are in {result.directory}")
    if args.product and not args.full and not args.datastream:
        print(f"Next: python combine_product.py {args.product}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
