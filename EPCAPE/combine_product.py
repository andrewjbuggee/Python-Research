#!/usr/bin/env python
"""Merge a product's daily ARM files into one netCDF in <data folder>/processed/.

Examples
  python combine_product.py cbh_ceil_M1
  python combine_product.py cbh_ceil_M1 --start 2023-06-01 --end 2023-08-31
  python combine_product.py cbh_ceil_M1 --out ~/Desktop/cbh.nc

Input is taken from, in order of preference and most complete coverage: a
mounted ARM archive, the product's subset folder, or complete files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epcape.combine import combine_product
from epcape.config import active_machine, as_date, campaign_dates, get_product, load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("product", help="product name from config.yaml")
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: campaign start)")
    parser.add_argument("--end", help="last day, inclusive (default: campaign end)")
    parser.add_argument("--source", choices=["auto", "archive", "subset", "full"], default="auto",
                        help="which local files to read (default: auto)")
    parser.add_argument("--out", help="output file (default: <data folder>/processed/<product>_<start>_<end>.nc)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        machine = active_machine(cfg)
        print(f"Machine: {machine.name}   data folder: {machine.data_root}")
        product = get_product(args.product, cfg)
        c_start, c_end = campaign_dates(cfg)
        start = as_date(args.start) if args.start else c_start
        end = as_date(args.end) if args.end else c_end
        out = Path(args.out).expanduser() if args.out else None
        combine_product(product, start, end, machine=machine, source=args.source, out=out)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
