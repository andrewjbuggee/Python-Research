#!/usr/bin/env python
"""Download what the MWR vs GOES-18 LWP comparison needs beyond the seasonal-averages products.

  goes18_visst_grid_X1  epcvisstgridg18minnisX1.c1, COMPLETE daily files (~6-12 MB/day, ~3.5 GB for the
                        campaign). ARM Live's subset service took ~25 s per file for this datastream
                        (19 Oct 2026) against ~2-8 s for a complete file, so complete files are fetched,
                        then reduced to the 3 x 3 boxes around the Pier (goes18.reduce_pier_boxes ->
                        processed/goes18_visst_pier_X1_<start>_<end>.nc, 34 MB), then to the
                        Pier box alone (goes18.extract_pier_box -> processed/goes18_visst_pierbox_X1_*.nc, 10 MB;
                        the only file the notebook reads). The daily files can be deleted afterwards.
  rain_mwrret2_M1       rain_intensity carried in the MWRRET v2 files (MWR3C rain sensor), subset + combined
  precip_arscl_M1       precip_mean (rain gauge) carried in the KAZR-ARSCL files, subset + combined

The MWRRET v2 LWP, ceilometer and KAZR-ARSCL layer products come from
comparisons/seasonal_averages/download_data.py (lwp_mwrret2_M1, cbh_ceil_M1, cloud_arscl_M1).

Examples (run from the EPCAPE folder)
  python comparisons/lwp_mwr_vs_goes/download_data.py
  python comparisons/lwp_mwr_vs_goes/download_data.py --reduce-only     # files already downloaded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# EPCAPE/ is itself the Python package: put its parent folder on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import xarray as xr  # noqa: E402

from EPCAPE.analysis_tools.products import processed_path  # noqa: E402
from EPCAPE.download_data.armlive import ArmLiveError  # noqa: E402
from EPCAPE.download_data.combine import combine_product  # noqa: E402
from EPCAPE.download_data.config import active_machine, campaign_dates, get_product, load_config  # noqa: E402
from EPCAPE.download_data.sync import sync_datastream, sync_product  # noqa: E402
from EPCAPE.comparisons.lwp_mwr_vs_goes import goes18  # noqa: E402

SUBSET_PRODUCTS = ["rain_mwrret2_M1", "precip_arscl_M1"]  # small: server-side variable subsets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reduce-only", action="store_true", help="only rebuild the reduced GOES-18 file")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (default 2)")
    args = parser.parse_args(argv)

    cfg = load_config()
    machine = active_machine(cfg)
    start, end = campaign_dates(cfg)
    problems = {}
    if not args.reduce_only:
        # GOES-18 grid: complete files into data/arm/epcvisstgridg18minnisX1.c1/
        try:
            sync_datastream(get_product(goes18.PRODUCT, cfg).datastream, start, end, machine=machine,
                            workers=args.workers)
        except (ArmLiveError, RuntimeError, ValueError) as exc:
            problems[goes18.PRODUCT] = str(exc)
        # rain fields: subsets, then one combined file each (what load_product reads)
        for name in SUBSET_PRODUCTS:
            try:
                result = sync_product(get_product(name, cfg), start, end, machine=machine, workers=1)
                if result.downloaded or not processed_path(name, machine, cfg).is_file():
                    combine_product(get_product(name, cfg), start, end, machine=machine,
                                    out=processed_path(name, machine, cfg))
            except (ArmLiveError, RuntimeError, ValueError, OSError) as exc:
                problems[name] = str(exc)

    # The Pier's coordinates from the ceilometer file (the same ones the seasonal-averages notebook uses)
    with xr.open_dataset(processed_path("cbh_ceil_M1", machine, cfg)) as ceil:
        lat_deg, lon_deg = float(ceil["lat"].values.ravel()[0]), float(ceil["lon"].values.ravel()[0])
    try:
        goes18.reduce_pier_boxes(lat_deg, lon_deg, machine=machine)
        goes18.extract_pier_box(machine=machine)  # the Pier box alone: the file the notebook reads
    except (FileNotFoundError, ValueError, OSError) as exc:
        problems["reduce"] = str(exc)

    if problems:
        print("\nProblems (rerun to retry):")
        for k, v in problems.items():
            print(f"  {k}: {v}")
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
