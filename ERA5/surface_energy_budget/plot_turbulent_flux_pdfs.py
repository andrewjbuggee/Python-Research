#!/usr/bin/env python3
"""Probability density functions of Arctic turbulent heat fluxes, 1 row x 3 columns.

Panels, left to right: net turbulent flux (SH + LH), sensible heat flux, latent
heat flux. Each panel pools every retained ocean cell-time in the selected date
range into one distribution.

Only ocean grid cells are used. ERA5 leaves sea ice cover undefined over land, so
land is dropped automatically; sea-ice-covered water is dropped according to
``--mask`` and ``--max-siconc``.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**, matching
``plot_turbulent_flux_maps.py``.

Area weighting
--------------
Grid cell area scales as cos(latitude), so a 70N cell covers about twice the area
of an 80N cell at 0.25 deg resolution. The densities are cos(lat)-weighted by
default so they describe the distribution over AREA rather than over grid cells.
Pass ``--no-area-weight`` for the raw per-cell distribution.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
``--data-root`` overrides both with an explicit path.

Examples
--------
    python plot_turbulent_flux_pdfs.py --region barrow \
        --start 2026-01-01 --end 2026-02-17 --mask all-ocean

    python plot_turbulent_flux_pdfs.py --storage external --region barrow \
        --start 2000-01-01 --end 2025-12-31 --mask all-ocean

    python plot_turbulent_flux_pdfs.py --region barrow --no-area-weight --no-kde
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    FLUX_PANELS,
    add_data_source_args,
    area_weights,
    build_ocean_mask,
    compute_turbulent_fluxes,
    load_seb_data,
    parse_date,
    resolve_region_dir,
    warn_if_sparse,
)

# One hue per panel, assigned in fixed order and never cycled. These are three
# distinct hues from a colourblind-safe qualitative set; each panel is a single
# distribution named by its own title, so colour carries no extra meaning here
# beyond keeping the panels visually distinct.
PANEL_COLORS = ("#4C72B0", "#C44E52", "#55A868")

DEFAULT_BINS = 80

# Percentile trimmed from each end when choosing the x range. These flux
# distributions are strongly left-skewed: coastal polynya events put a long
# negative tail on an otherwise sharply peaked bulk. Trimming at 0.5% leaves the
# central 90% of the data occupying only ~15% of the axis, which renders the
# bulk as an unreadable spike. At 2% it occupies about half the axis. The
# fraction of samples falling outside the axis is annotated on each panel, so
# nothing is hidden by the choice.
DEFAULT_XLIM_PERCENTILE = 2.0


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Weighted mean, standard deviation, median and percentiles."""
    total_w = weights.sum()
    mean = float(np.sum(values * weights) / total_w)
    var = float(np.sum(weights * (values - mean) ** 2) / total_w)

    order = np.argsort(values)
    v_sorted = values[order]
    cum_w = np.cumsum(weights[order]) / total_w

    def wq(q: float) -> float:
        return float(np.interp(q, cum_w, v_sorted))

    return {
        "mean": mean,
        "std": float(np.sqrt(var)),
        "median": wq(0.5),
        "p05": wq(0.05),
        "p95": wq(0.95),
        "frac_positive": float(weights[values > 0].sum() / total_w),
        "n": int(values.size),
    }


def make_pdfs(
    samples: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    region: str,
    time_label: str,
    mask_label: str,
    weight_label: str,
    bins: int = DEFAULT_BINS,
    show_kde: bool = True,
    shared_xlim: bool = False,
    xlim_percentile: float = DEFAULT_XLIM_PERCENTILE,
    log_y: bool = False,
    output_path: Path | None = None,
    dpi: int = 150,
    panels=FLUX_PANELS,
    quantity_noun: str = "turbulent heat flux",
):
    """Draw the 1xN PDF figure and return the matplotlib Figure.

    ``panels`` names the variables in ``samples``/``weights``; the turbulent set
    is the default and the radiative PDF script passes RADIATIVE_PANELS.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(15, 5.0), constrained_layout=True)

    p_lo, p_hi = xlim_percentile, 100.0 - xlim_percentile
    if shared_xlim:
        allv = np.concatenate([v for v in samples.values() if v.size])
        lo, hi = np.percentile(allv, [p_lo, p_hi])
        pad = 0.05 * (hi - lo)
        xlims = [(lo - pad, hi + pad)] * len(panels)

    stats_all = {}
    for i, (ax, (name, title, subtitle), color) in enumerate(
        zip(axes, panels, PANEL_COLORS)
    ):
        v = samples[name]
        w = weights[name]

        if v.size == 0:
            ax.text(0.5, 0.5, "no data after masking", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="0.4")
            ax.set_title(f"{title}\n{subtitle}", fontsize=11)
            continue

        if not shared_xlim:
            lo, hi = np.percentile(v, [p_lo, p_hi])
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            xlim = (lo - pad, hi + pad)
        else:
            xlim = xlims[i]

        # Density is computed over the FULL sample, then displayed on the
        # trimmed axis, so the curve keeps its true normalisation.
        edges = np.linspace(xlim[0], xlim[1], bins + 1)
        ax.hist(v, bins=edges, weights=w, density=True, color=color, alpha=0.55,
                edgecolor="none")

        w_total = w.sum()
        frac_below = float(w[v < xlim[0]].sum() / w_total)
        frac_above = float(w[v > xlim[1]].sum() / w_total)

        if show_kde and v.size > 2 and np.ptp(v) > 0:
            try:
                from scipy.stats import gaussian_kde
                # Subsample very large arrays: a KDE over millions of points is
                # slow and visually identical to one over a large random subset.
                if v.size > 200_000:
                    rng = np.random.default_rng(20260810)  # seed recorded
                    idx = rng.choice(v.size, 200_000, replace=False)
                    kde = gaussian_kde(v[idx], weights=w[idx])
                else:
                    kde = gaussian_kde(v, weights=w)
                xs = np.linspace(xlim[0], xlim[1], 400)
                ax.plot(xs, kde(xs), color=color, linewidth=2.0)
            except (ImportError, np.linalg.LinAlgError, ValueError) as exc:
                print(f"  KDE skipped for {title}: {exc}", file=sys.stderr)

        s = weighted_stats(v, w)
        stats_all[name] = s

        # Zero line: the sign change is the physically meaningful reference.
        ax.axvline(0.0, color="0.35", linewidth=1.0, linestyle="-", zorder=1)
        ax.axvline(s["mean"], color="0.15", linewidth=1.6, linestyle="--", zorder=5)

        ax.set_xlim(xlim)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Flux [W m$^{-2}$]  —  positive downward", fontsize=10)
        if i == 0:
            ax.set_ylabel("Probability density [ (W m$^{-2}$)$^{-1}$ ]", fontsize=10)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8)

        # Label the mean line directly rather than carrying a legend box: each
        # panel shows a single distribution that its own title already names.
        ax.annotate(
            f"mean {s['mean']:.1f}",
            xy=(s["mean"], 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, -12), textcoords="offset points",
            fontsize=8.5, color="0.15", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )

        txt = (
            f"mean    {s['mean']:>8.2f}\n"
            f"median  {s['median']:>8.2f}\n"
            f"std     {s['std']:>8.2f}\n"
            f"5-95%  {s['p05']:>7.1f},{s['p95']:>6.1f}\n"
            f"P(>0)   {100 * s['frac_positive']:>7.1f}%\n"
            f"n       {s['n']:>8,}\n"
            f"off-axis {100 * (frac_below + frac_above):>6.2f}%"
        )
        # Upper LEFT: these distributions are left-skewed with the mode near the
        # right of the axis, so the left corner is the empty one and the box
        # never lands on the mode or on the mean-line label.
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, family="monospace", color="0.2",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.8", alpha=0.92), zorder=10)

        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("0.8")
        ax.tick_params(labelsize=9, colors="0.35")

    fig.suptitle(
        f"ERA5 {quantity_noun} distributions over ocean — {region}\n"
        f"{time_label}   |   {mask_label}   |   {weight_label}",
        fontsize=13,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig, stats_all


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
                             f"(default {DEFAULT_XLIM_PERCENTILE}). Use 0.5 to show more "
                             f"of the tail at the cost of squashing the bulk.")
    parser.add_argument("--log-y", action="store_true",
                        help="Logarithmic density axis, which makes the heavy negative "
                             "tail from polynya events legible.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: figures/<region>_turbulent_flux_pdfs.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 turbulent flux probability density functions")
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
    print(f"  Time range : {t0} to {t1} UTC  ({n_times} hourly steps)")

    mask, report = build_ocean_mask(ds, args.mask, args.max_siconc)
    print(f"\n  Masking ({args.mask}):")
    print(report.describe())
    warn_if_sparse(report)

    fluxes = compute_turbulent_fluxes(ds, mask).compute()

    # Broadcast cos(lat) weights against the full (time, lat, lon) field so every
    # retained sample carries the weight of its own grid cell.
    if args.no_area_weight:
        w_full = None
        weight_label = "unweighted (per grid cell)"
    else:
        w_full = area_weights(ds).broadcast_like(fluxes["shf_W_m2"]).values
        weight_label = "area-weighted by cos(latitude)"

    samples: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for name, _, _ in FLUX_PANELS:
        v = fluxes[name].values.ravel()
        finite = np.isfinite(v)
        samples[name] = v[finite]
        weights[name] = (
            np.ones(finite.sum()) if w_full is None else w_full.ravel()[finite]
        )

    print(f"\n  Weighting  : {weight_label}")
    print("\n  Distribution statistics [W m-2, positive downward]:")
    header = f"    {'quantity':<22}{'mean':>9}{'median':>9}{'std':>9}{'P(>0)':>9}{'n':>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        output_path = fig_dir / f"{args.region}_turbulent_flux_pdfs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    if args.mask == "open-ocean":
        mask_label = f"open ocean only (sea ice fraction < {args.max_siconc:g})"
    else:
        mask_label = "all ocean, including sea ice"
    time_label = f"{t0} to {t1} UTC ({n_times} h)"

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
    )

    for name, title, _ in FLUX_PANELS:
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
