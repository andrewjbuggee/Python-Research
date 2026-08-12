#!/usr/bin/env python3
"""Probability density functions of Arctic surface radiative fluxes, 1 row x 3 columns.

Panels, left to right: net radiative flux (LW_net + SW_net), net longwave flux,
net shortwave flux. The radiative counterpart of ``plot_turbulent_flux_pdfs.py``,
sharing its masking, area weighting, and layout.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**. No sign
flip is involved for the radiative terms.

Note on the shortwave panel: through polar night SW_net is exactly zero, so its
distribution collapses to a spike at 0 for winter date ranges. That is physics,
not a plotting bug.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
``--data-root`` overrides both with an explicit path.

Examples
--------
    python plot_radiative_flux_pdfs.py --storage external --region barrow \
        --start 2000-09-01 --end 2000-11-30 --mask all-ocean

    python plot_radiative_flux_pdfs.py --region barrow --no-area-weight --no-kde
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_turbulent_flux_pdfs import DEFAULT_BINS, DEFAULT_XLIM_PERCENTILE, make_pdfs
from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    RADIATIVE_PANELS,
    add_data_source_args,
    area_weights,
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
                        help=f"Open-water threshold (default {DEFAULT_MAX_SICONC}).")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--no-kde", action="store_true", help="Histogram only.")
    parser.add_argument("--no-area-weight", action="store_true",
                        help="Weight every grid cell equally instead of by cos(lat).")
    parser.add_argument("--shared-xlim", action="store_true",
                        help="Use one x range across all three panels.")
    parser.add_argument("--xlim-percentile", type=float, default=DEFAULT_XLIM_PERCENTILE,
                        help=f"Percentile trimmed from each end when setting the x range "
                             f"(default {DEFAULT_XLIM_PERCENTILE}).")
    parser.add_argument("--log-y", action="store_true",
                        help="Logarithmic density axis, for heavy tails.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: figures/<region>_radiative_flux_pdfs.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 radiative flux probability density functions")
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

    mask, report = build_ocean_mask(ds, args.mask, args.max_siconc)
    print(f"\n  Masking ({args.mask}):")
    print(report.describe())
    warn_if_sparse(report)

    fluxes = compute_radiative_fluxes(ds, mask).compute()

    if args.no_area_weight:
        w_full = None
        weight_label = "unweighted (per grid cell)"
    else:
        w_full = area_weights(ds).broadcast_like(fluxes["lw_net_W_m2"]).values
        weight_label = "area-weighted by cos(latitude)"

    samples: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for name, _, _ in RADIATIVE_PANELS:
        v = fluxes[name].values.ravel()
        finite = np.isfinite(v)
        samples[name] = v[finite]
        weights[name] = (
            np.ones(finite.sum()) if w_full is None else w_full.ravel()[finite]
        )

    print(f"\n  Weighting  : {weight_label}")

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        output_path = fig_dir / f"{args.region}_radiative_flux_pdfs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    if args.mask == "open-ocean":
        mask_label = f"open ocean only (sea ice fraction < {args.max_siconc:g})"
    else:
        mask_label = "all ocean, including sea ice"
    time_label = f"{t0} to {t1} UTC ({n_times} steps)"

    fig, stats_all = make_pdfs(
        samples, weights,
        region=args.region,
        time_label=time_label,
        mask_label=mask_label,
        weight_label=weight_label,
        bins=args.bins,
        show_kde=not args.no_kde,
        shared_xlim=args.shared_xlim,
        xlim_percentile=args.xlim_percentile,
        log_y=args.log_y,
        output_path=output_path,
        dpi=args.dpi,
        panels=RADIATIVE_PANELS,
        quantity_noun="surface radiative flux",
    )

    print("\n  Distribution statistics [W m-2, positive downward]:")
    header = f"    {'quantity':<22}{'mean':>9}{'median':>9}{'std':>9}{'P(>0)':>9}{'n':>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, title, _ in RADIATIVE_PANELS:
        if name in stats_all:
            s = stats_all[name]
            print(f"    {title:<22}{s['mean']:>9.2f}{s['median']:>9.2f}"
                  f"{s['std']:>9.2f}{100 * s['frac_positive']:>8.1f}%{s['n']:>12,}")
        else:
            print(f"    {title:<22}{'no data after masking':>47}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
