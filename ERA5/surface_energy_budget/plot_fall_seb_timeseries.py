#!/usr/bin/env python3
"""Freeze-up season time series over open Arctic ocean, from August to December.

Stacked panels on one time-of-season axis, every one pooled across all years in
the record and restricted to grid cells that are still OPEN water
(``siconc < --max-siconc``):

    1. Net surface energy balance          median + 25th-75th percentile band
    2. Its four components                 net LW, net SW, sensible, latent
    3. Surface temperatures                SST and 2 m air temperature
    4. Ice-free fraction of the region     how much open water panel 1 stands on

plus an inset map showing where the average is taken.

The net surface energy balance, in ERA5's positive-downward convention:

    SEB_net = msnlwrf + msnswrf + msshf + mslhf     [W m-2]

All four terms share that convention, so the sum needs no sign flips. Positive
means the ocean is gaining energy. The season starts in August so the turn from
net warming to net cooling is visible, and ends 1 December because essentially
no open-ocean cells remain after early November.

Panel 3 is the diagnostic contrast worth watching: SST is pinned near the
freezing point of sea water once the mixed layer has given up its heat, while
2 m air temperature keeps falling with the synoptic airmass. The growing gap
between them is what drives the sensible heat flux in panel 2.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.

Examples
--------
    python plot_fall_seb_timeseries.py --storage external --region barrow

    python plot_fall_seb_timeseries.py --storage external --region barrow \
        --group day --season-start 08-01 --season-end 12-01
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

REF_YEAR = 2001  # non-leap reference year for the time-of-season axis

SEB_COLOR = "#4C72B0"
ICE_COLOR = "#2A7E78"

# Panel 2: the four SEB components, in ERA5 positive-downward form.
COMPONENTS = (
    ("msnlwrf", "net LW", "#C44E52"),
    ("msnswrf", "net SW", "#DD8452"),
    ("msshf", "sensible", "#4C72B0"),
    ("mslhf", "latent", "#55A868"),
)

# Panel 3: surface and near-surface temperature.
TEMPERATURES = (
    ("sst", "SST", "#1F77B4"),
    ("t2m", "2 m air", "#D62728"),
    ("skt", "skin", "#7F7F7F"),
)

FREEZING_SEAWATER_K = 271.35  # ~ -1.8 C, the salinity-depressed freezing point


def parse_month_day(text: str) -> tuple[int, int]:
    """Parse ``MM-DD`` into a (month, day) pair."""
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(REF_YEAR, month, day)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid MM-DD") from None
    return month, day


def season_stats(
    times: np.ndarray,
    fields: dict[str, np.ndarray],
    open_mask_tyx: np.ndarray,
    ocean_mask_yx: np.ndarray,
    w_yx: np.ndarray,
    season: tuple[tuple[int, int], tuple[int, int]],
    group: str,
    min_cells: int,
) -> dict:
    """Pool all years at each time-of-season step and reduce to quantiles.

    Every field in ``fields`` is reduced the same way, over the same open-water
    mask, so the panels are directly comparable: a difference between them is a
    difference in the physics, never in which cells were counted.
    """
    (m0, d0), (m1, d1) = season
    if (m0, d0) > (m1, d1):
        raise ValueError("Season windows that wrap the new year are not supported.")

    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    hours = (times.astype("datetime64[h]") - times.astype("datetime64[D]")).astype(int)

    md = months * 100 + days
    in_season = (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)
    idx_season = np.nonzero(in_season)[0]
    if idx_season.size == 0:
        raise ValueError(
            f"No time steps inside {m0:02d}-{d0:02d} .. {m1:02d}-{d1:02d}."
        )

    keys = (md[idx_season] * 100 + hours[idx_season]) if group == "hour" else md[idx_season]

    w_ocean_total = float(w_yx[ocean_mask_yx].sum())
    w_bcast = np.broadcast_to(w_yx, open_mask_tyx.shape)
    frac_t = (open_mask_tyx * w_bcast).sum(axis=(1, 2)) / w_ocean_total

    uniq = np.unique(keys)
    n = uniq.size
    positions = np.empty(n, dtype="datetime64[m]")
    quant = {name: np.full((n, 3), np.nan) for name in fields}
    frac_q = np.full((n, 3), np.nan)
    n_cells = np.zeros(n, dtype=int)

    for i, key in enumerate(uniq):
        sel = idx_season[keys == key]
        if group == "hour":
            mo, dy, hr = key // 10000, (key // 100) % 100, key % 100
        else:
            mo, dy, hr = key // 100, key % 100, 0
        positions[i] = np.datetime64(f"{REF_YEAR}-{mo:02d}-{dy:02d}T{hr:02d}:00")

        keep_base = open_mask_tyx[sel]
        w = np.broadcast_to(w_yx, keep_base.shape)
        for name, arr in fields.items():
            vals = arr[sel]
            keep = keep_base & np.isfinite(vals)
            v, wv = vals[keep], w[keep]
            if name == "net_seb":
                n_cells[i] = v.size
            if v.size >= min_cells:
                quant[name][i] = weighted_quantiles(v, wv, (0.25, 0.50, 0.75))
        frac_q[i] = np.nanquantile(frac_t[sel], [0.25, 0.50, 0.75])

    season_years = sorted(
        int(y) + 1970
        for y in np.unique(times[idx_season].astype("datetime64[Y]").astype(int))
    )
    return {
        "positions": positions,
        "quant": quant,
        "icefree_q25_50_75": frac_q,
        "n_cells": n_cells,
        "season_years": season_years,
    }


def _add_location_inset(fig, ax, lat, lon):
    """Small polar-stereographic inset showing the averaging domain.

    Anchored to the SEB panel's own axes rather than to figure coordinates, so
    it tracks the panel under constrained_layout instead of drifting across it.
    It sits in the right-hand quarter, which is empty because open water is gone
    by early November while the axis runs to 1 December.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.path as mpath
    except ImportError:
        return None

    central = float(np.mean([lon.min(), lon.max()]))
    inset = ax.inset_axes(
        [0.775, 0.06, 0.21, 0.62],
        projection=ccrs.NorthPolarStereo(central_longitude=central),
    )
    # Whole Arctic for context, with the analysis box drawn on it.
    inset.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 200)
    inset.set_boundary(
        mpath.Path(np.column_stack([0.5 + 0.5 * np.sin(theta),
                                    0.5 + 0.5 * np.cos(theta)])),
        transform=inset.transAxes,
    )
    inset.add_feature(cfeature.LAND, facecolor="0.85", zorder=1)
    inset.add_feature(cfeature.OCEAN, facecolor="#DCE9F2", zorder=0)
    inset.coastlines(resolution="110m", linewidth=0.4, color="0.4", zorder=2)
    inset.gridlines(linewidth=0.3, color="0.75", alpha=0.7, zorder=3)

    box_lon = [lon.min(), lon.max(), lon.max(), lon.min(), lon.min()]
    box_lat = [lat.min(), lat.min(), lat.max(), lat.max(), lat.min()]
    inset.plot(box_lon, box_lat, transform=ccrs.PlateCarree(),
               color="#C44E52", linewidth=1.8, zorder=5)
    inset.set_title("averaging domain", fontsize=8, pad=3, color="0.3")
    return inset


def make_figure(
    stats: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    region: str,
    years_label: str,
    mask_label: str,
    group: str,
    min_cells: int,
    output_path: Path | None,
    dpi: int,
):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    pos = stats["positions"].astype("datetime64[m]").astype("O")
    q = stats["quant"]
    ice = stats["icefree_q25_50_75"] * 100.0

    have_temps = [k for k, _, _ in TEMPERATURES if k in q and np.isfinite(q[k]).any()]
    n_panels = 4 if have_temps else 3
    heights = [2.0, 1.5, 1.3, 0.9][:n_panels] if have_temps else [2.0, 1.5, 0.9]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13.5, 3.0 * n_panels), sharex=True,
        constrained_layout=True, height_ratios=heights,
    )
    ax_seb, ax_comp = axes[0], axes[1]
    ax_temp = axes[2] if have_temps else None
    ax_ice = axes[-1]

    # --- 1. net SEB ---------------------------------------------------------
    seb = q["net_seb"]
    ax_seb.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    ax_seb.fill_between(pos, seb[:, 0], seb[:, 2], color=SEB_COLOR, alpha=0.25,
                        linewidth=0, label="25th–75th percentile")
    ax_seb.plot(pos, seb[:, 1], color=SEB_COLOR, linewidth=1.6, label="median")
    ax_seb.set_ylabel("Net SEB [W m$^{-2}$]\npositive downward", fontsize=10)
    ax_seb.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_seb.set_title(
        f"Freeze-up season over OPEN ocean — {region}\n"
        f"{years_label}   |   {mask_label}",
        fontsize=12, pad=10,
    )
    blanked = int(np.isnan(seb[:, 1]).sum())
    if blanked:
        # Left of centre: the right-hand quarter is reserved for the map inset.
        ax_seb.text(0.30, 0.03,
                    f"{blanked} of {len(pos)} steps blank: fewer than "
                    f"{min_cells} open-ocean cells",
                    transform=ax_seb.transAxes, ha="left", va="bottom",
                    fontsize=8.5, color="0.35")

    # --- 2. components ------------------------------------------------------
    # Medians only: four overlapping IQR bands would be unreadable, and the
    # point of this panel is which term drives the net, not their spread.
    ax_comp.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    for name, label, color in COMPONENTS:
        if name in q and np.isfinite(q[name]).any():
            ax_comp.plot(pos, q[name][:, 1], color=color, linewidth=1.4, label=label)
    ax_comp.set_ylabel("SEB components\n[W m$^{-2}$]", fontsize=10)
    ax_comp.legend(loc="lower left", fontsize=9, ncol=4, framealpha=0.9)

    # --- 3. temperatures ----------------------------------------------------
    if ax_temp is not None:
        for name, label, color in TEMPERATURES:
            if name in have_temps:
                ax_temp.plot(pos, q[name][:, 1] - 273.15, color=color,
                             linewidth=1.4, label=label)
        ax_temp.axhline(FREEZING_SEAWATER_K - 273.15, color="0.45",
                        linestyle=":", linewidth=1.2)
        ax_temp.text(0.005, FREEZING_SEAWATER_K - 273.15, " sea-water freezing",
                     transform=ax_temp.get_yaxis_transform(), va="bottom",
                     fontsize=8, color="0.45")
        ax_temp.set_ylabel("Temperature [°C]", fontsize=10)
        ax_temp.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)

    # --- 4. ice-free fraction ----------------------------------------------
    ax_ice.fill_between(pos, ice[:, 0], ice[:, 2], color=ICE_COLOR, alpha=0.25,
                        linewidth=0)
    ax_ice.plot(pos, ice[:, 1], color=ICE_COLOR, linewidth=1.6)
    ax_ice.set_ylabel("Ice-free ocean\narea [%]", fontsize=10)
    ax_ice.set_ylim(0, max(5.0, float(np.nanmax(ice[:, 2])) * 1.1))
    step_word = "hourly" if group == "hour" else "daily"
    ax_ice.set_xlabel(f"Time of season ({step_word} steps, pooled across years)",
                      fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("0.8")
        ax.tick_params(labelsize=9, colors="0.35")

    ax_ice.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ice.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_ice.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 15)))

    _add_location_inset(fig, ax_seb, lat, lon)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD",
                        help="First day of the season window (default 08-01, so the "
                             "turn from net warming to net cooling is visible).")
    parser.add_argument("--season-end", type=parse_month_day, default=(12, 1),
                        metavar="MM-DD",
                        help="Last day, inclusive (default 12-01; essentially no "
                             "open-ocean cells remain after early November).")
    parser.add_argument("--max-siconc", type=float, default=DEFAULT_MAX_SICONC,
                        help=f"A cell is open water below this sea ice fraction "
                             f"(default {DEFAULT_MAX_SICONC}).")
    parser.add_argument("--group", choices=("hour", "day"), default="hour",
                        help="Pool across years at each hour of the season (default) "
                             "or per calendar day for a smoother curve.")
    parser.add_argument("--min-cells", type=int, default=10,
                        help="Blank a step with fewer open-ocean samples than this "
                             "(default 10).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: figures/<region>_fall_seb_timeseries.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Freeze-up season time series")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    (m0, d0), (m1, d1) = args.season_start, args.season_end
    print(f"  Source     : {region_dir}")
    print(f"  Season     : {m0:02d}-{d0:02d} to {m1:02d}-{d1:02d}, by {args.group}")

    ds = ds.compute()
    times = ds["valid_time"].values

    open_mask, report = build_ocean_mask(ds, "open-ocean", args.max_siconc)
    print("\n  Masking (open-ocean):")
    print(report.describe())

    # Every quantity is reduced over the same open-water mask.
    fields = {"net_seb": compute_net_seb(ds).values}
    for name, _, _ in COMPONENTS:
        if name in ds:
            fields[name] = ds[name].values
    missing_temps = []
    for name, label, _ in TEMPERATURES:
        if name in ds and np.isfinite(ds[name].values).any():
            fields[name] = ds[name].values
        else:
            missing_temps.append(label)
    if missing_temps:
        print(f"\n  Note: no usable {', '.join(missing_temps)} in this record; "
              f"those lines are omitted.")
        if "sst" in ds:
            print("        (sst is present but entirely NaN -- it was added to the")
            print("         variable set after these files were downloaded.)")

    ocean_yx = ds["siconc"].notnull().any("valid_time").values
    w_yx = np.broadcast_to(
        area_weights(ds).values[:, None],
        (ds.sizes["latitude"], ds.sizes["longitude"]),
    )
    w_yx = np.where(ocean_yx, w_yx, 0.0)

    try:
        stats = season_stats(
            times=times, fields=fields, open_mask_tyx=open_mask.values,
            ocean_mask_yx=ocean_yx, w_yx=w_yx,
            season=(args.season_start, args.season_end),
            group=args.group, min_cells=args.min_cells,
        )
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    # Say which temperature lines survived. A variable can be present in the
    # dataset yet entirely NaN inside the season -- which is exactly what
    # happens to sst when it was added to the variable set after most of the
    # archive was downloaded, so open_mfdataset back-fills it with NaN.
    plotted = [lab for key, lab, _ in TEMPERATURES
               if key in stats["quant"] and np.isfinite(stats["quant"][key]).any()]
    dropped = [lab for key, lab, _ in TEMPERATURES if lab not in plotted]
    if dropped:
        print(f"\n  Temperature lines plotted: {', '.join(plotted) or 'none'}")
        print(f"  Not available in this season: {', '.join(dropped)}")
        if "SST" in dropped and "sst" in ds:
            print("    sst exists in the newer files but is NaN across this season;")
            print("    re-download the season with the current variable set to get it.")

    yrs = stats["season_years"]
    n_steps = len(stats["positions"])
    n_blank = int(np.isnan(stats["quant"]["net_seb"][:, 1]).sum())
    step_word = "hourly" if args.group == "hour" else "daily"
    print(f"\n  Years in season : {yrs}")
    print(f"  Season steps    : {n_steps} ({step_word}), {n_blank} blanked")

    # Monthly digest of the medians.
    pos_months = stats["positions"].astype("datetime64[M]").astype(int) % 12 + 1
    cols = ["net_seb"] + [c for c, _, _ in COMPONENTS if c in stats["quant"]]
    hdr = f"  {'month':<7}" + "".join(f"{c:>11}" for c in cols) + f"{'ice-free %':>12}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    ice_med = stats["icefree_q25_50_75"][:, 1] * 100
    import calendar as _cal
    for mo in sorted(set(pos_months)):
        sel = pos_months == mo
        row = f"  {_cal.month_abbr[mo]:<7}"
        for c in cols:
            v = stats["quant"][c][sel, 1]
            row += f"{np.nanmean(v):>11.1f}" if np.isfinite(v).any() else f"{'--':>11}"
        row += f"{np.nanmean(ice_med[sel]):>12.1f}"
        print(row)

    years_label = (f"{len(yrs)} years: {yrs[0]}–{yrs[-1]}" if len(yrs) > 1
                   else f"single year: {yrs[0]}")
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
        stats, ds["latitude"].values, ds["longitude"].values,
        region=args.region, years_label=years_label, mask_label=mask_label,
        group=args.group, min_cells=args.min_cells,
        output_path=output_path, dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
