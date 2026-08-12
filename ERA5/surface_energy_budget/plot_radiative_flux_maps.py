#!/usr/bin/env python3
"""Spatial maps of Arctic surface radiative fluxes over ocean, 1 row x 3 columns.

Panels, left to right: net radiative flux (LW_net + SW_net), net longwave flux,
net shortwave flux. Each panel is the time mean over a user-specified date
range. The radiative counterpart of ``plot_turbulent_flux_maps.py``, sharing its
masking, layout, and colour conventions.

Only ocean grid cells are shown; land and sea-ice-covered water are dropped
according to ``--mask`` and ``--max-siconc``, exactly as in the turbulent
script.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**. ERA5's
net radiative fluxes are archived positive-downward already, so unlike the
turbulent terms no sign flip is involved. In the Arctic the net longwave is
almost always negative (surface losing heat to the sky); positive net shortwave
appears only when the sun is up.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
``--data-root`` overrides both with an explicit path.

Examples
--------
Fall 2000 over the Barrow strip, from the external drive::

    python plot_radiative_flux_maps.py --storage external --region barrow \
        --start 2000-09-01 --end 2000-11-30 --mask all-ocean

Open ocean only (the default mask), loosened ice threshold::

    python plot_radiative_flux_maps.py --region barrow --max-siconc 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_turbulent_flux_maps import make_maps
from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    RADIATIVE_PANELS,
    add_data_source_args,
    build_ocean_mask,
    compute_radiative_fluxes,
    load_seb_data,
    parse_date,
    resolve_region_dir,
    warn_if_sparse,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--start", type=parse_date, default=None,
                        help="First time, YYYY-MM-DD[THH]. Default: start of record.")
    parser.add_argument("--end", type=parse_date, default=None,
                        help="Last time, inclusive, YYYY-MM-DD[THH]. Default: end of record.")
    parser.add_argument("--mask", choices=("open-ocean", "all-ocean"),
                        default="open-ocean",
                        help="open-ocean drops land and sea ice (default); "
                             "all-ocean drops only land.")
    parser.add_argument("--max-siconc", type=float, default=DEFAULT_MAX_SICONC,
                        help=f"Sea ice fraction below which a cell is open water "
                             f"(default {DEFAULT_MAX_SICONC}). Only used with --mask open-ocean.")
    parser.add_argument("--per-panel-scale", action="store_true",
                        help="Give each panel its own colour scale instead of one shared scale.")
    parser.add_argument("--projection", choices=("polar", "platecarree"), default="polar",
                        help="polar = North Polar Stereographic with coastlines (default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path. Default: figures/<region>_radiative_flux_maps.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 radiative flux maps")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, args.start, args.end, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    print(f"  Source     : {region_dir}")
    t0 = str(ds.valid_time.values[0])[:16].replace("T", " ")
    t1 = str(ds.valid_time.values[-1])[:16].replace("T", " ")
    n_times = ds.sizes["valid_time"]
    print(f"  Region     : {args.region}")
    print(f"  Time range : {t0} to {t1} UTC  ({n_times} time steps)")
    print(f"  Grid       : {ds.sizes['latitude']} lat x {ds.sizes['longitude']} lon")

    mask, report = build_ocean_mask(ds, args.mask, args.max_siconc)
    print(f"\n  Masking ({args.mask}):")
    print(report.describe())
    warn_if_sparse(report)

    fluxes = compute_radiative_fluxes(ds, mask)
    means = fluxes.mean(dim="valid_time", skipna=True).compute()

    print("\n  Time-mean flux over retained cells [W m-2, positive downward]:")
    for name, title, _ in RADIATIVE_PANELS:
        v = means[name].values
        f = v[np.isfinite(v)]
        if f.size:
            print(f"    {title:<22} mean={f.mean():>8.2f}  min={f.min():>8.2f}  "
                  f"max={f.max():>8.2f}  cells={f.size:,}")
        else:
            print(f"    {title:<22} no valid cells")

    time_label = f"time mean, {t0} to {t1} UTC ({n_times} steps)"
    if args.mask == "open-ocean":
        mask_label = f"open ocean only (sea ice fraction < {args.max_siconc:g})"
    else:
        mask_label = "all ocean, including sea ice"

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        output_path = fig_dir / f"{args.region}_radiative_flux_maps.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_maps(
        means,
        region=args.region,
        time_label=time_label,
        mask_label=mask_label,
        shared_scale=not args.per_panel_scale,
        projection=args.projection,
        output_path=output_path,
        dpi=args.dpi,
        panels=RADIATIVE_PANELS,
        quantity_noun="surface radiative fluxes",
        cbar_quantity="Radiative flux",
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
