#!/usr/bin/env python3
"""Freeze-up season time series of the surface energy budget, by sea ice regime.

For every step of the season, values are pooled across all requested years and
all grid cells in a sea ice regime, then reduced to a median and a 25th-75th
percentile band. One figure per regime, plus two figures comparing them.

FIGURES PRODUCED
================
    <region>_fall_seb_open_ocean.png       ice-free cells only
    <region>_fall_seb_marginal.png         marginal ice zone
    <region>_fall_seb_ice_covered.png      consolidated pack
    <region>_regime_comparison_seb.png     net SEB, all regimes on one axis
    <region>_regime_comparison_components.png
                                           three panels: radiative fluxes
                                           (net SW / net LW), turbulent fluxes
                                           (sensible / latent), and area share.
                                           Colour = regime, line style = term.

Each per-regime figure has up to four stacked panels: net SEB; the four SEB
components; skin / 2 m air / sea surface temperature; and the regime's share of
ocean area. Every line is a MEDIAN and every band the 25th-75th percentile.

SEA ICE REGIMES
===============
The three partition the ocean exactly -- every ocean cell-time falls in one, and
their area shares sum to 100% at every step:

    open ocean    siconc <  --max-siconc            (default 0.05)
    marginal      --max-siconc <= siconc <= --ice-min
    ice covered   siconc >  --ice-min               (default 0.95)

HOW "AREA" IS COMPUTED
======================
Area is **cos(latitude)-weighted, not a cell count**. ERA5 is on a regular
lat/lon grid, so every cell spans the same degrees but shrinks in ground area
toward the pole: a cell at 70 N covers about twice the area of one at 80 N.
Each cell therefore carries weight cos(lat), zero over land, and

    regime share = (weighted area of cells in the regime) / (total ocean area)

Counting cells instead would over-weight the northern, ice-covered end of this
domain by roughly a factor of two. The same weights are used for the medians and
percentile bands, so a cell's influence is always proportional to the area it
represents.

OPTIONS
=======

Data source
-----------
--storage {local,external}   Which disk to read (default local).
--data-root PATH             Explicit directory, overriding --storage.
--region NAME                Region subdirectory (default barrow).

Season and years
----------------
--season-start MM-DD         First day of the window (default 08-01).
--season-end MM-DD           Last day, inclusive (default 12-01). The window MAY
                             wrap the new year: 09-01 to 03-31 is one season.
--years SPEC                 '2019', '2019-2025', or '2000,2019-2020'. For a
                             wrapping window this is the year the season STARTS
                             in, so 2019 with 09-01..03-31 means Sep 2019 to
                             Mar 2020. Default: every year present.
--group {hour,day}           Pool by hour of season (default) or by calendar
                             day. Daily is smoother and drops the diurnal cycle.

Regimes
-------
--max-siconc F               Open-water bound (default 0.05).
--ice-min F                  Ice-covered bound (default 0.95).
--regimes LIST               'all' (default) or a subset of
                             open_ocean,marginal,ice_covered.
--min-cells N                Blank a step with fewer than N cells in the regime
                             (default 10); prevents a median being drawn from a
                             handful of pixels.

Appearance
----------
--no-component-bands         Medians only in the component and temperature
                             panels, without percentile shading.
--no-histogram               Suppress the pointer to plot_sea_ice_histogram.py.
--output PATH                Output file, honoured only when a single regime is
                             selected; otherwise names are generated per regime.
--dpi N                      Figure resolution (default 350).
--show                       Open interactive windows.

With more than one year the band carries interannual AND spatial spread; with a
single year it is spatial spread alone. The figure subtitle says which.

SIGN CONVENTION: native ERA5, **positive downward**, so a negative net SEB means
the surface is losing energy. All four terms share this convention, so

    SEB_net = msnlwrf + msnswrf + msshf + mslhf

needs no sign flips.

Sea ice concentration histograms now live in ``plot_sea_ice_histogram.py``.

Examples
--------
    python plot_fall_seb_timeseries.py --region barrow --years 2019 --group day

    python plot_fall_seb_timeseries.py --region barrow \
        --season-start 09-01 --season-end 03-31 --years 2019-2025 --group day
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from seb_analysis_common import (
    hour_of,
    season_calendar,
    season_slot_of,
    season_year_of,
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

# Sea ice regimes, each producing its own figure. The bounds are half-open on
# the low side so the three partition the ocean exactly: every ocean cell-time
# lands in one and only one regime, and the three area fractions sum to 1.
#   open ocean   siconc <  open_max        (default 0.05)
#   marginal     open_max <= siconc <= ice_min
#   ice covered  siconc >  ice_min         (default 0.95)
SHORT_LABELS = {
    "open_ocean": "Ice-free ocean",
    "marginal": "Marginal ice zone",
    "ice_covered": "Ice-covered ocean",
}

REGIMES = (
    ("open_ocean", "Open ocean (ice free)"),
    ("marginal", "Marginal ice zone"),
    ("ice_covered", "Ice covered"),
)

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


def parse_years(text: str) -> tuple[int, ...]:
    """Parse '2000,2001' or '2000-2002' (or a mix) into a sorted year tuple."""
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo:
                    raise ValueError
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{part!r} is not a year or an ascending YYYY-YYYY range"
            ) from None
    if not out:
        raise argparse.ArgumentTypeError("no years given")
    return tuple(sorted(out))


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
    # Position within the season comes from a slot lookup, not from arithmetic
    # on the date, which is what lets the window wrap the new year (09-01 to
    # 03-31 is one season, labelled by the year it starts in).
    slots = season_calendar(season[0], season[1])
    slot = season_slot_of(times, slots)
    hours = hour_of(times)

    idx_season = np.nonzero(slot >= 0)[0]
    if idx_season.size == 0:
        raise ValueError(
            f"No time steps inside {m0:02d}-{d0:02d} .. {m1:02d}-{d1:02d}."
        )

    keys = ((slot[idx_season] * 100 + hours[idx_season]) if group == "hour"
            else slot[idx_season])

    # AREA is cos(latitude)-weighted, not a raw cell count. ERA5 sits on a
    # regular lat/lon grid, so a cell spans the same degrees everywhere but
    # shrinks in ground area toward the pole: at 70 N a cell covers roughly
    # twice the area of one at 80 N. w_yx holds cos(lat) for every ocean cell
    # and zero over land, so
    #     frac_t = (area of cells in this regime) / (total ocean area)
    # is a genuine area fraction. Counting cells instead would over-weight the
    # northern, ice-covered end of this domain by about a factor of two.
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
        s_i = int(key // 100) if group == "hour" else int(key)
        hr = int(key % 100) if group == "hour" else 0
        mo, dy = slots[s_i]
        # Reference years match season_calendar's: 1999 -> 2000 for a wrapping
        # window so the post-wrap half lands in a LEAP year and the 29 February
        # slot is a valid date; a non-leap reference year raises here.
        wraps = season[1] < season[0]
        base = 1999 if wraps else 2000
        yr = base + (1 if (wraps and (mo, dy) < season[0]) else 0)
        positions[i] = np.datetime64(f"{yr}-{mo:02d}-{dy:02d}T{hr:02d}:00")

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

    # Season year = the calendar year the season STARTS in. For a wrapping
    # window, January belongs to the season that began the previous autumn, so
    # taking the raw calendar year would split one season across two labels.
    season_years = sorted(set(season_year_of(times[idx_season], season[0]).tolist()))
    return {
        "positions": positions,
        "quant": quant,
        "icefree_q25_50_75": frac_q,
        "n_cells": n_cells,
        "season_years": season_years,
    }


def _pick_inset_corner(ax, series_list) -> str:
    """Put the inset in whichever right-hand corner the curves avoid.

    A fixed corner covers the data whenever the regime changes the curve's shape
    -- the open-ocean SEB dives to the bottom right while the ice-covered one
    stays high. Compares how close the plotted values come to the top and bottom
    of the axis over the right-hand third and takes the freer one.
    """
    finite = [np.asarray(v, dtype=float) for v in series_list if v is not None]
    if not finite:
        return "upper right"
    stacked = np.concatenate([f[~np.isnan(f)] for f in finite if np.isfinite(f).any()]
                             or [np.array([0.0])])
    lo, hi = float(np.nanmin(stacked)), float(np.nanmax(stacked))
    span = (hi - lo) or 1.0
    # Only the right-hand third matters, since the inset lives on the right.
    right = [f[int(0.66 * len(f)):] for f in finite if len(f)]
    right = np.concatenate([r[~np.isnan(r)] for r in right if np.isfinite(r).any()]
                           or [np.array([0.0])])
    if right.size == 0:
        return "upper right"
    head_room = (hi - float(np.nanmax(right))) / span
    foot_room = (float(np.nanmin(right)) - lo) / span
    return "upper right" if head_room >= foot_room else "lower right"


def _add_location_inset(fig, ax, lat, lon, corner: str = "upper right"):
    """Small polar-stereographic inset showing the averaging domain.

    Anchored to the SEB panel's own axes rather than to figure coordinates, so
    it tracks the panel under constrained_layout instead of drifting across it.
    ``corner`` selects the upper or lower half of the right-hand edge; the caller
    picks whichever the curve leaves free, since a fixed position lands on top of
    the data for some regimes (the open-ocean SEB dives to the bottom right,
    while the ice-covered one stays near the top).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.path as mpath
    except ImportError:
        return None

    central = float(np.mean([lon.min(), lon.max()]))
    # y position follows the caller's choice of corner so the inset lands in
    # whichever half of the right-hand edge the curve is not occupying.
    y0 = 0.5 if corner == "upper right" else 0.04
    inset = ax.inset_axes(
        [0.79, y0, 0.20, 0.58],
        projection=ccrs.NorthPolarStereo(central_longitude=central),
    )
    # Whole Arctic for context, with the analysis box drawn on it.
    inset.set_extent([-180, 180, 66, 90], crs=ccrs.PlateCarree())
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
    regime_label: str = "open ocean",
    regime_short: str = "Ice-free ocean",
    show_bands: bool = True,
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
        f"Freeze-up season \u2014 {regime_label} \u2014 {region}\n"
        f"{years_label}   |   {mask_label}",
        fontsize=12, pad=10,
    )
    # --- 2. components ------------------------------------------------------
    # Every line is a MEDIAN, with its own 25th-75th band. Four overlapping
    # bands are busy, so they are drawn faint (alpha 0.13) and can be turned off
    # with --no-component-bands; the lines stay legible either way.
    ax_comp.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    for name, label, color in COMPONENTS:
        if name in q and np.isfinite(q[name]).any():
            if show_bands:
                ax_comp.fill_between(pos, q[name][:, 0], q[name][:, 2],
                                     color=color, alpha=0.13, linewidth=0)
            ax_comp.plot(pos, q[name][:, 1], color=color, linewidth=1.4,
                         label=f"{label} (median)")
    ax_comp.set_ylabel("SEB components\n[W m$^{-2}$]", fontsize=10)
    ax_comp.legend(loc="lower left", fontsize=8.5, ncol=4, framealpha=0.9,
                   title="shading = 25th–75th percentile" if show_bands else None,
                   title_fontsize=8)

    # --- 3. temperatures ----------------------------------------------------
    if ax_temp is not None:
        for name, label, color in TEMPERATURES:
            if name in have_temps:
                if show_bands:
                    ax_temp.fill_between(pos, q[name][:, 0] - 273.15,
                                         q[name][:, 2] - 273.15,
                                         color=color, alpha=0.13, linewidth=0)
                ax_temp.plot(pos, q[name][:, 1] - 273.15, color=color,
                             linewidth=1.4, label=f"{label} (median)")
        ax_temp.set_ylabel("Temperature [°C]", fontsize=10)
        ax_temp.legend(loc="lower left", fontsize=8.5, ncol=3, framealpha=0.9,
                       title="shading = 25th–75th percentile" if show_bands else None,
                       title_fontsize=8)

    # --- 4. ice-free fraction ----------------------------------------------
    ax_ice.fill_between(pos, ice[:, 0], ice[:, 2], color=ICE_COLOR, alpha=0.25,
                        linewidth=0, label="25th–75th percentile across years")
    ax_ice.plot(pos, ice[:, 1], color=ICE_COLOR, linewidth=1.6, label="median")
    if regime_short == "Ice-free ocean":
        ax_ice.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
    elif regime_short == "Marginal ice zone" or regime_short == "Ice-covered ocean":
        ax_ice.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    ax_ice.set_ylabel(f"{regime_short}\narea [%]", fontsize=10)
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

    # Place the inset away from the curve rather than at a fixed corner.
    corner = _pick_inset_corner(ax_seb, [seb[:, 1], seb[:, 2], seb[:, 0]])
    _add_location_inset(fig, ax_seb, lat, lon, corner=corner)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig


# color identifies the regime across every comparison figure, assigned in a
# fixed order so a regime keeps its color no matter which subset is plotted.
REGIME_COLORS = {
    "open_ocean": "#1B7FBD",   # blue   - open water
    "marginal": "#E08214",     # orange - transition
    "ice_covered": "#5E3C99",  # purple - consolidated pack
}


def _style_axes(axes):
    for ax in axes:
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("0.8")
        ax.tick_params(labelsize=9, colors="0.35")


def _area_panel(ax, pos, stats_by_regime, labels):
    """Bottom panel shared by both comparison figures: area share per regime.

    Each curve is the cos(latitude)-weighted fraction of the region's OCEAN area
    in that regime, so the three sum to 100% at every step -- which is worth
    checking by eye, since it is the guarantee that the regimes partition the
    ocean rather than overlapping.
    """
    for key, st in stats_by_regime.items():
        frac = st["icefree_q25_50_75"][:, 1] * 100.0
        ax.plot(pos, frac, color=REGIME_COLORS[key], linewidth=1.7,
                label=f"{labels[key]} (median)")
    ax.set_ylabel("Share of ocean\narea [%]", fontsize=10)
    ax.set_ylim(0, 100)
    #ax.legend(loc="upper left", fontsize=8.5, ncol=3, framealpha=0.9)


def make_regime_comparison(stats_by_regime, labels, region, years_label,
                           group, output_path, dpi):
    """Net SEB for all three regimes on one axis, with the area shares below."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    any_stats = next(iter(stats_by_regime.values()))
    pos = any_stats["positions"].astype("datetime64[m]").astype("O")

    fig, (ax, ax_area) = plt.subplots(
        2, 1, figsize=(13.5, 8.0), sharex=True, constrained_layout=True,
        height_ratios=[2.0, 1.0],
    )
    ax.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    for key, st in stats_by_regime.items():
        seb = st["quant"]["net_seb"]
        ax.fill_between(pos, seb[:, 0], seb[:, 2], color=REGIME_COLORS[key],
                        alpha=0.13, linewidth=0)
        ax.plot(pos, seb[:, 1], color=REGIME_COLORS[key], linewidth=1.8,
                label=f"{labels[key]} (median)")
    ax.set_ylabel("Net SEB [W m$^{-2}$]\npositive downward", fontsize=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9,
              title="shading = 25th–75th percentile", title_fontsize=8)
    ax.set_title(f"Net surface energy balance by sea ice regime — {region}\n"
                 f"{years_label}", fontsize=12, pad=10)

    _area_panel(ax_area, pos, stats_by_regime, labels)
    step_word = "hourly" if group == "hour" else "daily"
    ax_area.set_xlabel(f"Time of season ({step_word} steps, pooled across years)",
                       fontsize=10)
    _style_axes([ax, ax_area])
    ax_area.xaxis.set_major_locator(mdates.MonthLocator())
    ax_area.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  comparison -> {output_path}")
    return fig


# Panel groupings for the component comparison: radiative terms on one axis,
# turbulent on another. Two lines per regime per panel instead of four keeps the
# dash patterns distinguishable, which four never quite were.
RADIATIVE_TERMS = (("msnswrf", "net SW"), ("msnlwrf", "net LW"))
TURBULENT_TERMS = (("msshf", "sensible"), ("mslhf", "latent"))


def _draw_term_panel(ax, pos, stats_by_regime, terms, styles):
    """Plot one group of SEB terms for every regime on a single axis.

    Colour carries the regime and line style carries the term, so the two stay
    separable. A linear scale is used deliberately: these fluxes are signed and
    mostly NEGATIVE through the freeze-up season, so a log axis would silently
    discard every non-positive value -- which is nearly all of the data.
    """
    ax.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
    drawn = []
    for key, st in stats_by_regime.items():
        for (name, label), style in zip(terms, styles):
            q = st["quant"]
            if name in q and np.isfinite(q[name]).any():
                ax.plot(pos, q[name][:, 1], color=REGIME_COLORS[key],
                        linestyle=style, linewidth=1.5)
                if label not in drawn:
                    drawn.append(label)
    return drawn


def make_component_comparison(stats_by_regime, labels, region, years_label,
                              group, output_path, dpi):
    """Radiative and turbulent SEB terms by regime, on separate panels.

    Three stacked panels: net shortwave and net longwave; sensible and latent
    heat; and the regimes' shares of ocean area. Splitting radiative from
    turbulent keeps each panel to two line styles per regime, and lets the two
    families take their own y scales -- they differ by enough in August that a
    shared axis flattened the turbulent terms.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    any_stats = next(iter(stats_by_regime.values()))
    pos = any_stats["positions"].astype("datetime64[m]").astype("O")
    styles = ["-", "--"]

    fig, (ax_rad, ax_turb, ax_area) = plt.subplots(
        3, 1, figsize=(13.5, 10.5), sharex=True, constrained_layout=True,
        height_ratios=[1.6, 1.6, 1.0],
    )

    rad_drawn = _draw_term_panel(ax_rad, pos, stats_by_regime,
                                 RADIATIVE_TERMS, styles)
    ax_rad.set_ylabel("Radiative fluxes\n[W m$^{-2}$] (medians)", fontsize=10)
    ax_rad.set_title(f"SEB components by sea ice regime — {region}\n{years_label}",
                     fontsize=12, pad=10)

    turb_drawn = _draw_term_panel(ax_turb, pos, stats_by_regime,
                                  TURBULENT_TERMS, styles)
    ax_turb.set_ylabel("Turbulent fluxes\n[W m$^{-2}$] (medians)", fontsize=10)

    # Colour legend on the top panel, line-style legend on each panel, since the
    # style meaning differs between them.
    reg_handles = [Line2D([], [], color=REGIME_COLORS[k], lw=2, label=labels[k])
                   for k in stats_by_regime]
    leg = ax_rad.legend(handles=reg_handles, loc="upper right", fontsize=8.5,
                        ncol=3, framealpha=0.9, title="regime (colour)",
                        title_fontsize=8)
    ax_rad.add_artist(leg)
    for ax, drawn in ((ax_rad, rad_drawn), (ax_turb, turb_drawn)):
        handles = [Line2D([], [], color="0.3", lw=1.6, linestyle=st, label=lab)
                   for lab, st in zip(drawn, styles)]
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=8.5, ncol=2,
                      framealpha=0.9, title="term (line style)", title_fontsize=8)

    _area_panel(ax_area, pos, stats_by_regime, labels)
    step_word = "hourly" if group == "hour" else "daily"
    ax_area.set_xlabel(f"Time of season ({step_word} steps, pooled across years)",
                       fontsize=10)

    _style_axes([ax_rad, ax_turb, ax_area])
    ax_area.xaxis.set_major_locator(mdates.MonthLocator())
    ax_area.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  components -> {output_path}")
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
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Restrict to these years, e.g. '2000,2001' or "
                             "'2000-2002'. Default: every year present. With more "
                             "than one year the band is the 25th-75th percentile "
                             "across years AND grid cells; with one year it is the "
                             "spread across grid cells alone.")
    parser.add_argument("--ice-min", type=float, default=0.95, metavar="F",
                        help="Sea ice fraction above which a cell counts as ice "
                             "covered (default 0.95). Cells between --max-siconc "
                             "and this are the marginal ice zone.")
    parser.add_argument("--regimes", default="all",
                        help="Which regime figures to make: 'all' (default) or a "
                             "comma-separated subset of "
                             "open_ocean,marginal,ice_covered.")
    parser.add_argument("--no-component-bands", action="store_true",
                        help="Draw only medians in the component and temperature "
                             "panels, without their 25th-75th percentile shading.")
    parser.add_argument("--no-histogram", action="store_true",
                        help="Skip the sea-ice-concentration histogram figure.")
    parser.add_argument("--hist-linear", action="store_true",
                        help="Linear color scale on the histogram instead of log. "
                             "Log is the default because the concentration "
                             "distribution is strongly bimodal.")
    parser.add_argument("--hist-bins", type=int, default=20,
                        help="Sea ice concentration bins in the histogram (default 20).")
    parser.add_argument("--max-siconc", type=float, default=0.05,
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
    parser.add_argument("--dpi", type=int, default=350)
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

    # Year filter first, so every downstream reduction sees only the requested
    # years and the printed diagnostics describe what is actually plotted.
    if args.years is not None:
        # Filter on the SEASON year (the year the window starts in), not the
        # calendar year. For 09-01..03-31, "2019" has to keep Jan-Mar 2020;
        # filtering on calendar year would silently drop the second half of
        # every wrapping season.
        syr = season_year_of(ds["valid_time"].values, args.season_start)
        keep_y = xr.DataArray(np.isin(syr, list(args.years)),
                              dims="valid_time",
                              coords={"valid_time": ds["valid_time"]})
        n_before = ds.sizes["valid_time"]
        ds = ds.sel(valid_time=keep_y)
        if ds.sizes["valid_time"] == 0:
            print(f"  Error: no data for years {list(args.years)}.", file=sys.stderr)
            return 1
        print(f"  Years      : {list(args.years)} "
              f"({ds.sizes['valid_time']:,} of {n_before:,} steps kept)")

    ds = ds.compute()
    times = ds["valid_time"].values

    if not (0.0 <= args.max_siconc <= args.ice_min <= 1.0):
        print(f"  Error: need 0 <= --max-siconc ({args.max_siconc}) <= --ice-min "
              f"({args.ice_min}) <= 1.", file=sys.stderr)
        return 1

    # Three mutually exclusive, collectively exhaustive regimes over the ocean.
    si = ds["siconc"]
    is_ocean = si.notnull()
    regime_masks = {
        "open_ocean": is_ocean & (si < args.max_siconc),
        "marginal": is_ocean & (si >= args.max_siconc) & (si <= args.ice_min),
        "ice_covered": is_ocean & (si > args.ice_min),
    }
    n_ocean = int(is_ocean.sum())
    print(f"\n  Ocean cell-times: {n_ocean:,}")
    for key, label in REGIMES:
        n = int(regime_masks[key].sum())
        print(f"    {label:<24} {n:>12,}  ({100*n/n_ocean:5.2f}% of ocean)")
    total = sum(int(regime_masks[k].sum()) for k, _ in REGIMES)
    if total != n_ocean:
        print(f"    !! regimes sum to {total:,}, not {n_ocean:,} -- bounds overlap",
              file=sys.stderr)

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

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    wanted = ([k for k, _ in REGIMES] if args.regimes == "all"
              else [r.strip() for r in args.regimes.split(",") if r.strip()])
    unknown = [r for r in wanted if r not in dict(REGIMES)]
    if unknown:
        print(f"  Error: unknown regime(s) {unknown}. "
              f"Choose from {[k for k, _ in REGIMES]}.", file=sys.stderr)
        return 1

    import calendar as _cal
    labels = dict(REGIMES)
    bounds = {
        "open_ocean": f"sea ice fraction < {args.max_siconc:g}",
        "marginal": f"{args.max_siconc:g} \u2264 sea ice fraction \u2264 {args.ice_min:g}",
        "ice_covered": f"sea ice fraction > {args.ice_min:g}",
    }

    stats_by_regime = {}
    for regime in wanted:
        print("\n" + "-" * 72)
        print(f"  {labels[regime]}  ({bounds[regime]})")
        print("-" * 72)
        try:
            stats = season_stats(
                times=times, fields=fields,
                open_mask_tyx=regime_masks[regime].values,
                ocean_mask_yx=ocean_yx, w_yx=w_yx,
                season=(args.season_start, args.season_end),
                group=args.group, min_cells=args.min_cells,
            )
        except ValueError as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            return 1

        yrs = stats["season_years"]
        n_steps = len(stats["positions"])
        n_blank = int(np.isnan(stats["quant"]["net_seb"][:, 1]).sum())
        step_word = "hourly" if args.group == "hour" else "daily"
        print(f"  Years in season : {yrs}")
        print(f"  Season steps    : {n_steps} ({step_word}), {n_blank} blanked "
              f"(fewer than {args.min_cells} cells)")
        if n_blank == n_steps:
            print(f"  !! No step has {args.min_cells}+ cells in this regime; "
                  f"the figure will be empty.", file=sys.stderr)

        # Months in season order. positions carry a reference year that advances
        # at the wrap, so sorting the positions keeps Sep..Mar in the right order
        # instead of restarting at January.
        pos_dt = stats["positions"].astype("datetime64[M]")
        pos_months = pos_dt.astype(int) % 12 + 1
        month_order = list(dict.fromkeys(pos_months.tolist()))
        cols = ["net_seb"] + [c for c, _, _ in COMPONENTS if c in stats["quant"]]
        hdr = (f"  {'month':<7}" + "".join(f"{c:>11}" for c in cols)
               + f"{'regime %':>12}")
        print("\n" + hdr)
        print("  " + "-" * (len(hdr) - 2))
        frac_med = stats["icefree_q25_50_75"][:, 1] * 100
        for mo in month_order:
            sel = pos_months == mo
            row = f"  {_cal.month_abbr[mo]:<7}"
            for c in cols:
                v = stats["quant"][c][sel, 1]
                row += (f"{np.nanmean(v):>11.1f}" if np.isfinite(v).any()
                        else f"{'--':>11}")
            row += f"{np.nanmean(frac_med[sel]):>12.1f}"
            print(row)

        # With one year the band is spatial spread only; with several it also
        # carries interannual spread. Saying which avoids over-reading it.
        if len(yrs) > 1:
            years_label = (f"{len(yrs)} years: {yrs[0]}\u2013{yrs[-1]} ")
            # years_label = (f"{len(yrs)} years: {yrs[0]}\u2013{yrs[-1]} "
            #                f"(band = 25th\u201375th pct across years and cells)")
        else:
            years_label = (f"single year: {yrs[0]} ")
            # years_label = (f"single year: {yrs[0]} "
            #                f"(band = 25th\u201375th pct across grid cells)")

        out = (args.output if (args.output and len(wanted) == 1)
               else fig_dir / f"{args.region}_fall_seb_{regime}.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        make_figure(
            stats, ds["latitude"].values, ds["longitude"].values,
            region=args.region, years_label=years_label,
            mask_label=f"{labels[regime]} \u2014 {bounds[regime]}",
            group=args.group, min_cells=args.min_cells,
            output_path=Path(out), dpi=args.dpi,
            regime_label=labels[regime], regime_short=SHORT_LABELS[regime],
            show_bands=not args.no_component_bands,
        )
        stats_by_regime[regime] = stats

    if len(stats_by_regime) > 1:
        print("\n" + "-" * 72)
        print("  Regime comparison figures")
        print("-" * 72)
        make_regime_comparison(
            stats_by_regime, labels, args.region, years_label, args.group,
            fig_dir / f"{args.region}_regime_comparison_seb.png", args.dpi)
        make_component_comparison(
            stats_by_regime, labels, args.region, years_label, args.group,
            fig_dir / f"{args.region}_regime_comparison_components.png", args.dpi)

    if not args.no_histogram:
        print("\n  Sea ice concentration histograms are now a separate script:")
        print(f"    python plot_sea_ice_histogram.py --region {args.region} "
              f"--season-start {args.season_start[0]:02d}-{args.season_start[1]:02d} "
              f"--season-end {args.season_end[0]:02d}-{args.season_end[1]:02d}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
