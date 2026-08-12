#!/usr/bin/env python3
"""Climatological time series of the net surface energy balance over open ocean
through the freeze-up season, with the region's ice-free fraction below it.

The net surface energy balance, in ERA5's positive-downward convention:

    SEB_net = msnlwrf + msnswrf + msshf + mslhf     [W m-2]

Positive means the ocean is gaining energy; the persistently negative values of
fall are what cool the mixed layer toward the freezing point and drive sea ice
growth. Over open ocean there is no subsurface conduction term to close -- the
residual goes into the water column.

What is plotted
---------------
Top panel: for every time-of-season step (e.g. 1 September 12:00), the values of
SEB_net from ALL years of data and ALL open-ocean grid cells at that step are
pooled, and the cos(lat)-weighted median is drawn with the 25th-75th percentile
range shaded. Only cells with sea ice fraction below ``--max-siconc`` enter, so
the curve tracks the energy balance of the water that is actually open. Steps
where fewer than ``--min-cells`` cells survive masking are left blank rather
than plotted from a handful of cells.

Bottom panel: the fraction of the region's ocean area that is ice free
(sea ice fraction < the same threshold), median and interquartile range across
years. This shows how much open water the top panel's statistics are standing
on, and when the region effectively closes for the winter.

As freeze-up progresses the two panels are deliberately coupled: the open-ocean
sample shrinks with the ice-free fraction, and the SEB curve going blank is
itself the signal that the region has frozen.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
``--data-root`` overrides both with an explicit path.

Examples
--------
Freeze-up season over the Barrow strip, from the external drive::

    python plot_fall_seb_timeseries.py --storage external --region barrow

Pool each calendar day's 24 hours instead of keeping hourly resolution::

    python plot_fall_seb_timeseries.py --storage external --region barrow \
        --group day

A different season window::

    python plot_fall_seb_timeseries.py --season-start 08-15 --season-end 11-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    add_data_source_args,
    area_weights,
    build_ocean_mask,
    compute_net_seb,
    load_seb_data,
    resolve_region_dir,
    weighted_quantiles,
)

# Non-leap reference year used only to give the time-of-season axis real
# calendar positions for matplotlib's month locator.
REF_YEAR = 2001

SEB_COLOR = "#4C72B0"
ICE_COLOR = "#2A7E78"


def parse_month_day(text: str) -> tuple[int, int]:
    """Parse ``MM-DD`` into a (month, day) pair."""
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(REF_YEAR, month, day)  # validates the combination
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a valid MM-DD month-day"
        ) from None
    return month, day


def season_group_stats(
    times: np.ndarray,
    seb_tyx: np.ndarray,
    open_mask_tyx: np.ndarray,
    ocean_mask_yx: np.ndarray,
    w_yx: np.ndarray,
    season: tuple[tuple[int, int], tuple[int, int]],
    group: str,
    min_cells: int,
) -> dict:
    """Pool every year at each time-of-season step and reduce to quantiles.

    Returns arrays indexed by group: plot positions (as datetime64 in the
    reference year), SEB quantiles [25, 50, 75], open-water sample counts, and
    the ice-free area fraction's [25, 50, 75] across years.
    """
    (m0, d0), (m1, d1) = season
    if (m0, d0) > (m1, d1):
        raise ValueError("Season windows that wrap the new year are not supported.")

    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    hours = (times.astype("datetime64[h]") - times.astype("datetime64[D]")).astype(int)

    # Month-day comparison via a single integer key (mm*100+dd), which orders
    # correctly for a season that does not wrap the new year.
    md = months * 100 + days
    in_season = (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)
    idx_season = np.nonzero(in_season)[0]
    if idx_season.size == 0:
        raise ValueError(
            f"No time steps fall inside the season "
            f"{m0:02d}-{d0:02d} .. {m1:02d}-{d1:02d}."
        )

    # Group key: time-of-season position, shared across years.
    if group == "hour":
        keys = md[idx_season] * 100 + hours[idx_season]
    else:  # day
        keys = md[idx_season]

    # Per-timestep ice-free area fraction of the ocean (cheap, vectorised).
    w_ocean_total = float(w_yx[ocean_mask_yx].sum())
    w_bcast = np.broadcast_to(w_yx, seb_tyx.shape)
    frac_t = (
        (open_mask_tyx * w_bcast).sum(axis=(1, 2)) / w_ocean_total
    )

    uniq = np.unique(keys)
    n_groups = uniq.size
    positions = np.empty(n_groups, dtype="datetime64[m]")
    seb_q = np.full((n_groups, 3), np.nan)
    frac_q = np.full((n_groups, 3), np.nan)
    n_cells = np.zeros(n_groups, dtype=int)
    n_years = np.zeros(n_groups, dtype=int)

    for i, key in enumerate(uniq):
        sel = idx_season[keys == key]
        if group == "hour":
            mo, dy, hr = key // 10000, (key // 100) % 100, key % 100
        else:
            mo, dy, hr = key // 100, key % 100, 0
        positions[i] = np.datetime64(f"{REF_YEAR}-{mo:02d}-{dy:02d}T{hr:02d}:00")

        vals = seb_tyx[sel]
        keep = open_mask_tyx[sel] & np.isfinite(vals)
        v = vals[keep]
        w = np.broadcast_to(w_yx, vals.shape)[keep]
        n_cells[i] = v.size
        n_years[i] = sel.size if group == "hour" else len(
            np.unique(times[sel].astype("datetime64[Y]"))
        )
        if v.size >= min_cells:
            seb_q[i] = weighted_quantiles(v, w, (0.25, 0.50, 0.75))
        frac_q[i] = np.nanquantile(frac_t[sel], [0.25, 0.50, 0.75])

    # Years that actually contribute to the season window. The record may hold
    # other months from other years (smoke tests, partial downloads); those must
    # not inflate the figure's "N years" label.
    season_years = sorted(
        int(y) + 1970
        for y in np.unique(times[idx_season].astype("datetime64[Y]").astype(int))
    )

    return {
        "positions": positions,
        "seb_q25_50_75_W_m2": seb_q,
        "icefree_q25_50_75": frac_q,
        "n_cells": n_cells,
        "n_years": n_years,
        "season_years": season_years,
    }


def make_figure(
    stats: dict,
    region: str,
    years_label: str,
    mask_label: str,
    group: str,
    min_cells: int,
    output_path: Path | None,
    dpi: int,
):
    """Two stacked panels on a shared time-of-season axis."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    pos = stats["positions"].astype("datetime64[m]").astype("O")
    seb = stats["seb_q25_50_75_W_m2"]
    ice = stats["icefree_q25_50_75"] * 100.0  # -> percent

    fig, (ax_seb, ax_ice) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True,
        height_ratios=[2.2, 1.0],
    )

    # --- Top: net SEB over open ocean ---------------------------------------
    ax_seb.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    ax_seb.fill_between(pos, seb[:, 0], seb[:, 2], color=SEB_COLOR, alpha=0.25,
                        linewidth=0, label="25th–75th percentile")
    ax_seb.plot(pos, seb[:, 1], color=SEB_COLOR, linewidth=1.6, label="median")
    ax_seb.set_ylabel("Net SEB [W m$^{-2}$]\npositive downward", fontsize=10)
    ax_seb.set_title(
        f"Net surface energy balance over OPEN ocean — {region}\n"
        f"{years_label}   |   {mask_label}",
        fontsize=12, pad=10,
    )
    ax_seb.legend(loc="lower left", fontsize=9, framealpha=0.9)

    blanked = int(np.isnan(seb[:, 1]).sum())
    if blanked:
        ax_seb.text(
            0.99, 0.03,
            f"{blanked} of {len(pos)} steps blank: fewer than {min_cells} "
            f"open-ocean cells",
            transform=ax_seb.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.35",
        )

    # --- Bottom: how much of the region is ice free -------------------------
    ax_ice.fill_between(pos, ice[:, 0], ice[:, 2], color=ICE_COLOR, alpha=0.25,
                        linewidth=0)
    ax_ice.plot(pos, ice[:, 1], color=ICE_COLOR, linewidth=1.6)
    ax_ice.set_ylabel("Ice-free ocean\narea [%]", fontsize=10)
    ax_ice.set_ylim(0, max(5.0, float(np.nanmax(ice[:, 2])) * 1.1))
    step_word = "hourly" if group == "hour" else "daily"
    ax_ice.set_xlabel(f"Time of season ({step_word} steps, pooled across years)",
                      fontsize=10)

    for ax in (ax_seb, ax_ice):
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("0.8")
        ax.tick_params(labelsize=9, colors="0.35")

    ax_ice.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ice.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_ice.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(9, 1),
                        metavar="MM-DD",
                        help="First day of the season window (default 09-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(12, 31),
                        metavar="MM-DD",
                        help="Last day of the season window, inclusive (default 12-31).")
    parser.add_argument("--max-siconc", type=float, default=DEFAULT_MAX_SICONC,
                        help=f"A cell is open water when sea ice fraction is below "
                             f"this (default {DEFAULT_MAX_SICONC}).")
    parser.add_argument("--group", choices=("hour", "day"), default="hour",
                        help="Pool across years at each hour of the season "
                             "(default), or pool each calendar day's steps "
                             "together for a smoother curve.")
    parser.add_argument("--min-cells", type=int, default=10,
                        help="Blank a step when fewer open-ocean samples than "
                             "this survive masking (default 10).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: figures/<region>_fall_seb_timeseries.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Freeze-up season net SEB time series")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    (m0, d0), (m1, d1) = args.season_start, args.season_end
    print(f"  Source     : {region_dir}")
    print(f"  Season     : {m0:02d}-{d0:02d} to {m1:02d}-{d1:02d}, "
          f"grouped by {args.group}")

    ds = ds.compute()
    times = ds["valid_time"].values

    open_mask, report = build_ocean_mask(ds, "open-ocean", args.max_siconc)
    print(f"\n  Masking (open-ocean):")
    print(report.describe())

    seb = compute_net_seb(ds)  # unmasked; masking is applied per group
    ocean_yx = ds["siconc"].notnull().any("valid_time").values
    w_yx = np.broadcast_to(
        area_weights(ds).values[:, None],
        (ds.sizes["latitude"], ds.sizes["longitude"]),
    )
    # Zero weight on permanent land so it can never enter the ocean total.
    w_yx = np.where(ocean_yx, w_yx, 0.0)

    try:
        stats = season_group_stats(
            times=times,
            seb_tyx=seb.values,
            open_mask_tyx=open_mask.values,
            ocean_mask_yx=ocean_yx,
            w_yx=w_yx,
            season=(args.season_start, args.season_end),
            group=args.group,
            min_cells=args.min_cells,
        )
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    yrs = stats["season_years"]
    n_steps = len(stats["positions"])
    n_blank = int(np.isnan(stats["seb_q25_50_75_W_m2"][:, 1]).sum())
    print(f"\n  Years in season : {yrs}")
    step_word = "hourly" if args.group == "hour" else "daily"
    print(f"  Season steps    : {n_steps} ({step_word}), "
          f"{n_blank} blanked below --min-cells {args.min_cells}")

    # Monthly digest of the median curve, for the terminal.
    seb_med = stats["seb_q25_50_75_W_m2"][:, 1]
    ice_med = stats["icefree_q25_50_75"][:, 1] * 100
    pos_months = stats["positions"].astype("datetime64[M]").astype(int) % 12 + 1
    print(f"\n  {'month':<8}{'median SEB [W m-2]':>20}{'ice-free area [%]':>20}")
    print("  " + "-" * 48)
    for mo in sorted(set(pos_months)):
        sel = pos_months == mo
        s = seb_med[sel]
        print(f"  {date(REF_YEAR, mo, 1):%b}"
              f"{np.nanmean(s) if np.isfinite(s).any() else float('nan'):>23.1f}"
              f"{np.nanmean(ice_med[sel]):>20.1f}")

    years_label = (
        f"{len(yrs)} year{'s' if len(yrs) != 1 else ''}: "
        f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1 else f"single year: {yrs[0]}"
    )
    mask_label = f"open ocean only (sea ice fraction < {args.max_siconc:g})"

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        output_path = fig_dir / f"{args.region}_fall_seb_timeseries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_figure(
        stats,
        region=args.region,
        years_label=years_label,
        mask_label=mask_label,
        group=args.group,
        min_cells=args.min_cells,
        output_path=output_path,
        dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
