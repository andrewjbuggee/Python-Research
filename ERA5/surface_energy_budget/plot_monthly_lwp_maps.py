#!/usr/bin/env python3
"""Monthly maps of cloud liquid water path, one panel per month.

A 2-row by 3-column grid of spatial maps: every grid cell of the region carries
its monthly-MEDIAN liquid water path (``tclw``, converted from ERA5's kg m-2 to
g m-2), averaged first within each season and then across the seasons
requested. Sea ice concentration is contoured on top, exactly as in
``plot_monthly_lwd_maps.py``, which this script otherwise mirrors line for
line -- see that script for the season-window, year-averaging and equal-season-
weighting conventions, all unchanged here.

Median, not mean
-----------------
LWP in cloudy Arctic scenes is heavily right-skewed and, in winter, piled up
near zero (see ``analyze_cloud_liquid_frequency.py``): the mean is dragged
around by a small number of thick-cloud hours, while the median tracks what a
typical hour actually looked like. Both the within-season-month reduction and
the across-season reduction use the median, so a thin winter month is not
inflated by a handful of large values the way a mean would be.

Threshold
---------
ERA5 carries trace liquid almost everywhere (see the ``analyze_cloud_liquid_
frequency.py`` docstring), so an unthresholded median is rarely a clean zero
even in midwinter. ``--lwp-threshold`` (default 0.03 g m-2) zeroes any hourly
LWP at or below it BEFORE the median is taken, so a month whose typical hour
carries only numerical-floor liquid reads as 0, not as a small positive value
with no physical meaning.

Colour scale
------------
LWP is a magnitude with no meaningful zero in its range other than "no
liquid", so it takes a sequential ramp running from white (zero) to dark blue,
built specifically for this figure rather than imported. Limits come from
robust percentiles pooled over all panels, as in ``plot_monthly_lwd_maps.py``;
``--per-month-scale`` gives each panel its own limits and colorbar instead.

Examples
--------
Six seasons averaged, the usual freeze-up view::

    python plot_monthly_lwp_maps.py --region barrow --years 2019-2024

A stricter liquid threshold and each panel on its own colour limits::

    python plot_monthly_lwp_maps.py --region barrow --years 2019-2024 \\
        --lwp-threshold 0.1 --per-month-scale
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

from ERA5.surface_energy_budget.plot_latitude_time_hovmoller_DLR import (
    COLOR_PERCENTILE,
    nanmean_quiet,
    parse_levels,
    parse_month_day,
    parse_years,
)
from plot_monthly_longwave_maps import (
    ARM_UTQIAGVIK_LAT,
    ARM_UTQIAGVIK_LON,
    draw_site_marker,
)
from plot_turbulent_flux_maps import _make_circular, domain_aspect
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_region_dir,
    season_calendar,
    season_slot_of,
    season_year_of,
)

LWP_VAR = "tclw"
LWP_LABEL = "Liquid water path"
LWP_UNITS = "g m$^{-2}$"

# White at zero liquid, running to a dark navy at the top of the scale. Built
# locally rather than taken from a stock colormap so the bottom is EXACTLY
# white, matching the "no liquid" reading a zeroed-out sub-threshold cell gets.
LWP_CMAP = LinearSegmentedColormap.from_list(
    "white_darkblue", ["#ffffff", "#08306b"]
)

# Liquid at or below this many g m-2 is treated as no liquid before the median
# is taken. Matches the near-zero trace-liquid floor discussed in
# analyze_cloud_liquid_frequency.py.
DEFAULT_LWP_THRESHOLD_G = 0.03

# Sea ice contour levels. 0.05 traces the outer edge of any ice at all and 0.95
# the edge of the consolidated pack, so together they bracket the marginal ice
# zone -- the band where the surface is neither open water nor solid ice.
DEFAULT_ICE_LEVELS = (0.05, 0.95)

# Aug to Jan inclusive: six calendar months, the 2x3 grid.
DEFAULT_SEASON_START = (8, 1)
DEFAULT_SEASON_END = (1, 31)

ICE_LINESTYLES = ("--", "-", ":", "-.")


def nanmedian_quiet(a, axis=None):
    """np.nanmedian without the "All-NaN slice" RuntimeWarning.

    An all-NaN slice is expected here, not a mistake: a season that never
    covered a given month, or a land cell with no sea ice, leaves that slice
    empty, and it should stay NaN silently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def months_of_window(
    slots: list[tuple[int, int]],
) -> tuple[list[int], dict[int, int], list[int]]:
    """Calendar months the season window touches, in season order.

    Returns ``(months, days_in_window, split_months)`` where ``days_in_window``
    counts the day-slots the window gives each month (fewer than the month's
    length wherever the window clips it) and ``split_months`` names any month the
    window enters twice, which only happens for a window that wraps onto its own
    start month.
    """
    months: list[int] = []
    days_in_window: dict[int, int] = {}
    first_i: dict[int, int] = {}
    last_i: dict[int, int] = {}
    for i, (m, _d) in enumerate(slots):
        if m not in days_in_window:
            months.append(m)
            first_i[m] = i
        days_in_window[m] = days_in_window.get(m, 0) + 1
        last_i[m] = i
    split = [m for m in months if last_i[m] - first_i[m] + 1 != days_in_window[m]]
    return months, days_in_window, split


def month_of(times: np.ndarray) -> np.ndarray:
    """Calendar month (1-12) of each timestamp."""
    return times.astype("datetime64[M]").astype(int) % 12 + 1


def season_coverage(
    slot: np.ndarray, syear: np.ndarray, n_slots: int
) -> dict[int, float]:
    """Fraction of the window's day-slots each season actually contains."""
    cov: dict[int, float] = {}
    in_window = slot >= 0
    for y in sorted(set(syear[in_window].tolist())):
        sel = in_window & (syear == y)
        cov[int(y)] = len(np.unique(slot[sel])) / n_slots
    return cov


def build_monthly_stack(
    ds,
    seasons: list[int],
    months: list[int],
    slot: np.ndarray,
    syear: np.ndarray,
    lwp_threshold_g: float,
) -> dict:
    """Median LWP and mean sea ice maps for every (season, month).

    Returns arrays shaped ``(season, month, latitude, longitude)`` plus the
    number of day-slots each pair contributed. LWP is thresholded (values at or
    below ``lwp_threshold_g`` zeroed) BEFORE the median is taken, so trace
    liquid at ERA5's numerical floor does not keep a dry month from reading as
    zero. Sea ice stays a mean -- only the plotted quantity's statistic changes.

    One season is loaded at a time. A six-season, six-month window is roughly
    26,000 hourly steps over the grid; pulling all of it at once to produce 36
    small maps costs about half a gigabyte for no benefit.
    """
    lat_deg = ds["latitude"].values
    lon_deg = ds["longitude"].values
    n_s, n_m = len(seasons), len(months)
    lwp_g_m2 = np.full((n_s, n_m, lat_deg.size, lon_deg.size), np.nan)
    ice_frac = np.full_like(lwp_g_m2, np.nan)
    n_days = np.zeros((n_s, n_m), dtype=int)

    in_window = slot >= 0
    times = ds["valid_time"].values
    mon = month_of(times)

    for s_i, s in enumerate(seasons):
        sel_s = in_window & (syear == s)
        idx_s = np.nonzero(sel_s)[0]
        band = ds[[LWP_VAR, "siconc"]].isel(valid_time=idx_s).load()
        lwp_t = band[LWP_VAR].values * 1000.0   # kg m-2 -> g m-2, (time, lat, lon)
        lwp_t = np.where(lwp_t > lwp_threshold_g, lwp_t, 0.0)
        lwp_t[np.isnan(band[LWP_VAR].values)] = np.nan  # keep genuine gaps as NaN
        ice_t = band["siconc"].values           # (time, lat, lon); NaN over land
        mon_s = mon[idx_s]
        slot_s = slot[idx_s]
        for m_i, m in enumerate(months):
            take = mon_s == m
            if not take.any():
                continue
            lwp_g_m2[s_i, m_i] = nanmedian_quiet(lwp_t[take], axis=0)
            ice_frac[s_i, m_i] = nanmean_quiet(ice_t[take], axis=0)
            n_days[s_i, m_i] = len(np.unique(slot_s[take]))
        del band, lwp_t, ice_t

    return {
        "lwp_g_m2": lwp_g_m2,
        "ice_frac": ice_frac,
        "n_days": n_days,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
    }


def label_site(ax, transform=None) -> None:
    """Name the ARM site, anchored so the text stays inside the panel.

    ``draw_site_marker``'s own label always sits to the upper right, which runs
    off the edge whenever the site is in the right-hand part of the domain. The
    side is chosen from the marker's position in AXES coordinates rather than
    from its longitude, because in a polar-stereographic panel the two are not
    the same thing. Call after ``set_extent``: the axes limits set the mapping.
    """
    if transform is None:
        x_data, y_data = ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT
    else:
        x_data, y_data = ax.projection.transform_point(
            ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT, transform)
    x_axes, _ = ax.transAxes.inverted().transform(
        ax.transData.transform((x_data, y_data)))

    if x_axes < 0.25:
        ha, dx = "left", 7
    elif x_axes > 0.75:
        ha, dx = "right", -7
    else:
        ha, dx = "center", 0
    kw = {} if transform is None else {"transform": transform}
    ax.annotate("Utqiaġvik (ARM)", xy=(ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT),
                xytext=(dx, 9), textcoords="offset points",
                ha=ha, va="bottom", fontsize=8, zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8), **kw)


def grid_shape(n_panels: int, ncols: int | None) -> tuple[int, int]:
    """Rows and columns for the panel grid; six months give the 2x3 default."""
    if ncols is None:
        ncols = min(3, n_panels)
    ncols = max(1, min(int(ncols), n_panels))
    return int(np.ceil(n_panels / ncols)), ncols


def color_limits(fields: list[np.ndarray]) -> tuple[float, float]:
    """Robust sequential limits, pooled over whichever panels are passed.

    The lower limit is pinned to 0 rather than a percentile: 0 is "no liquid",
    a meaningful floor for LWP, and letting a robust-low percentile float above
    it would make some genuinely dry cells look non-white.
    """
    finite = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
    if finite.size == 0:
        return 0.0, 1.0
    hi = float(np.percentile(finite, COLOR_PERCENTILE))
    if hi <= 0:
        hi = 1.0
    return 0.0, hi


def figure_size(
    nrows: int, ncols: int, aspect: float, is_circumpolar: bool
) -> tuple[float, float]:
    """Figure size in inches, from the domain's physical (not degree) aspect."""
    panel_h = 3.2 if not is_circumpolar else 3.0
    # A polar-stereographic lat/lon box is a wedge, so the axes is always wider
    # than the data it holds; the lower clamp buys back some of that margin.
    panel_w = float(np.clip(panel_h * (1.0 if is_circumpolar else aspect), 2.2, 4.4))
    return (ncols * panel_w + 2.6, nrows * panel_h + 2.6)


def make_month_grid(
    stack: dict,
    months: list[int],
    days_in_window: dict[int, int],
    region: str,
    mode_label: str,
    season_label: str,
    lwp_threshold_g: float,
    ice_levels: tuple[float, ...],
    contour_style: str = "contrast",
    per_month_scale: bool = False,
    show_site: bool = True,
    projection: str = "polar",
    ncols: int | None = None,
    output_path: Path | None = None,
    dpi: int = 150,
):
    """Draw the month grid of median LWP and return the Figure."""
    import matplotlib.pyplot as plt

    use_cartopy = projection == "polar"
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            print("  cartopy unavailable; using a plain lat/lon grid.", file=sys.stderr)
            use_cartopy = False

    lat_deg, lon_deg = stack["lat_deg"], stack["lon_deg"]
    # Aggregate the per-season maps: every season counts once, whatever its
    # length. Median, matching the within-season reduction, not a mean of
    # medians pretending to be a single median -- see the module docstring.
    lwp_m = nanmedian_quiet(stack["lwp_g_m2"], axis=0)   # (month, lat, lon)
    ice_m = nanmean_quiet(stack["ice_frac"], axis=0)
    n_seasons_used = np.sum(
        ~np.all(np.isnan(stack["lwp_g_m2"]), axis=(2, 3)), axis=0
    )  # per month

    is_circumpolar = (lon_deg.max() - lon_deg.min()) >= 350.0
    nrows, n_cols = grid_shape(len(months), ncols)

    proj_kw = {}
    if use_cartopy:
        central = 0.0 if is_circumpolar else float(np.mean([lon_deg.min(), lon_deg.max()]))
        proj_kw = {"projection": ccrs.NorthPolarStereo(central_longitude=central)}
        data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows, n_cols,
        figsize=figure_size(nrows, n_cols, domain_aspect(lat_deg, lon_deg),
                            is_circumpolar and use_cartopy),
        subplot_kw=proj_kw, squeeze=False, constrained_layout=True,
    )
    flat = axes.ravel()

    vmin, vmax = color_limits([lwp_m[i] for i in range(len(months))])

    mesh = None
    for i, m in enumerate(months):
        ax = flat[i]
        lo, hi = (color_limits([lwp_m[i]]) if per_month_scale else (vmin, vmax))
        plot_kw = dict(cmap=LWP_CMAP, vmin=lo, vmax=hi, shading="auto")
        if use_cartopy:
            plot_kw["transform"] = data_crs
        mesh = ax.pcolormesh(lon_deg, lat_deg, lwp_m[i], **plot_kw)

        ice_panel = ice_m[i]
        if np.isfinite(ice_panel).any():
            # Only levels the panel actually crosses are drawn. Passing 0.95 to
            # an ice-free August raises "no contour levels found" and returns an
            # empty set; filtering says the same thing without the noise.
            lo_i = float(np.nanmin(ice_panel))
            hi_i = float(np.nanmax(ice_panel))
            live = [v for v in ice_levels if lo_i < v < hi_i]
            if live:
                ckw = dict(colors="black", zorder=5,
                           linewidths=[1.0 + 0.4 * ice_levels.index(v) for v in live],
                           linestyles=[ICE_LINESTYLES[ice_levels.index(v) % 4]
                                       for v in live])
                if use_cartopy:
                    ckw["transform"] = data_crs
                cs = ax.contour(lon_deg, lat_deg, ice_panel, levels=live, **ckw)
                if contour_style == "contrast":
                    # A plain black line disappears against a dark-blue panel.
                    # Outlining it in white keeps it readable over any part of
                    # the ramp without giving the line a colour of its own,
                    # which would compete with the data for meaning.
                    halo = [patheffects.withStroke(linewidth=3.0, foreground="white")]
                    for coll in (cs.collections if hasattr(cs, "collections") else [cs]):
                        coll.set_path_effects(halo)

        if use_cartopy:
            if is_circumpolar:
                ax.set_extent([-180, 180, lat_deg.min(), 90], crs=data_crs)
                _make_circular(ax)
            else:
                ax.set_extent([lon_deg.min(), lon_deg.max(),
                               lat_deg.min(), lat_deg.max()], crs=data_crs)
            # Land under the data: LWP is defined over land too, so the field
            # shows there and the grey only fills what is outside it.
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
            # Mid grey, not a darker shade: the ramp runs to near-black navy in
            # the highest-LWP panels, where a dark coastline would disappear.
            ax.coastlines(resolution="50m", linewidth=0.6, color="0.6", zorder=3)
            ax.gridlines(linewidth=0.3, color="0.75", alpha=0.6, zorder=4)
            if show_site:
                draw_site_marker(ax, transform=data_crs)
                if i == 0:
                    label_site(ax, transform=data_crs)
        else:
            ax.tick_params(labelsize=7, colors="0.4")
            if show_site:
                draw_site_marker(ax)
                if i == 0:
                    label_site(ax)

        head = calendar.month_abbr[m]
        note = []
        full_days = calendar.monthrange(2000, m)[1]  # leap year: 29 Feb has a slot
        clipped = days_in_window[m] < full_days
        if clipped:
            note.append(f"{days_in_window[m]} of {full_days} days")
        n_y = int(n_seasons_used[i])
        if n_y != int(n_seasons_used.max()):
            note.append(f"{n_y} yr" + ("s" if n_y != 1 else ""))
        if note:
            head += "\n(" + ", ".join(note) + ")"
        ax.set_title(head, fontsize=11, pad=6,
                     color="0.45" if clipped else "black")

        if per_month_scale:
            # One bar PER PANEL, because with per-panel limits a single shared
            # bar shows only the last panel's range while appearing to describe
            # all six -- values would be read off it wrongly.
            cb_i = fig.colorbar(mesh, ax=ax, location="right", pad=0.02,
                                shrink=0.92, aspect=22)
            cb_i.ax.tick_params(labelsize=7)

    for ax in flat[len(months):]:
        ax.set_visible(False)

    if not per_month_scale:
        cb = fig.colorbar(mesh, ax=list(flat[: len(months)]), location="right",
                          pad=0.015, shrink=0.9, aspect=26)
        cb.set_label(f"{LWP_LABEL} [{LWP_UNITS}]", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    # A legend rather than inline contour labels: on panels this size a clabel
    # sits on top of the field it is meant to describe, and both levels mean the
    # same thing in every month, so naming them once is enough.
    if ice_levels:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color="black", linewidth=1.0 + 0.4 * i,
                   linestyle=ICE_LINESTYLES[i % 4],
                   label=f"sea ice {v:g}")
            for i, v in enumerate(ice_levels)
        ]
        # "outside lower center" is what makes constrained_layout reserve a
        # strip for the legend; a plain "lower center" is placed in figure
        # coordinates and lands on top of the bottom row of panels.
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=9.5)

    # Kept narrow deliberately: a single long line sets the saved figure's width
    # under bbox_inches="tight", stranding the panels in whitespace.
    fig.suptitle(
        f"ERA5 {LWP_LABEL.lower()} ({LWP_VAR}) — {region}\n"
        f"monthly median per grid cell [{LWP_UNITS}], threshold "
        f"{lwp_threshold_g:g} g m$^{{-2}}$, season {season_label}"
        + ("  (per-panel colour limits)" if per_month_scale else "") + "\n"
        f"{mode_label}",
        fontsize=12, linespacing=1.4,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--season-start", type=parse_month_day,
                        default=DEFAULT_SEASON_START, metavar="MM-DD",
                        help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day,
                        default=DEFAULT_SEASON_END, metavar="MM-DD",
                        help="Season end, inclusive (default 01-31, wrapping the "
                             "year). The window's calendar months become the panels, "
                             "so the default gives the 6-panel 2x3 grid.")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to average, by the year each season STARTS "
                             "in: '2019', '2019-2024', or '2000,2019-2020'. "
                             "Default: every season in the record that meets "
                             "--min-season-coverage. Naming years explicitly "
                             "overrides that filter, so a short season is still "
                             "included if you ask for it.")
    parser.add_argument("--lwp-threshold", type=float,
                        default=DEFAULT_LWP_THRESHOLD_G, metavar="G",
                        help="LWP in g m-2 at or below which an hour is treated as "
                             "no liquid before the median is taken (default "
                             f"{DEFAULT_LWP_THRESHOLD_G:g}). Guards against ERA5's "
                             "near-zero trace liquid making a dry month look "
                             "falsely non-zero.")
    parser.add_argument("--ice-levels", type=parse_levels, default=DEFAULT_ICE_LEVELS,
                        metavar="L1,L2",
                        help="Sea ice contour levels (default 0.05,0.95).")
    parser.add_argument("--contour-style", choices=("black", "contrast"),
                        default="contrast",
                        help="How the sea ice contours are drawn. 'contrast' "
                             "(default) outlines the black lines in white so they "
                             "stay readable over the dark end of the colour ramp; "
                             "'black' is a plain unoutlined line.")
    parser.add_argument("--per-month-scale", action="store_true",
                        help="Give each panel its own colour limits instead of one "
                             "scale for the figure. Reveals within-month structure "
                             "that the shared ramp crushes, but panels are then no "
                             "longer comparable to each other.")
    parser.add_argument("--ncols", type=int, default=None,
                        help="Columns in the panel grid (default 3, giving 2x3 for "
                             "the 6-month default window).")
    parser.add_argument("--min-season-coverage", type=float, default=0.6, metavar="F",
                        help="Exclude seasons covering less than this fraction of the "
                             "window from a climatology (default 0.6).")
    parser.add_argument("--no-site-marker", action="store_true",
                        help="Do not mark the DOE ARM Utqiagvik site.")
    parser.add_argument("--projection", choices=("polar", "platecarree"),
                        default="polar")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Monthly maps of cloud liquid water path")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = sorted({LWP_VAR, "siconc"} - set(ds.data_vars))
    if missing:
        print(f"  Error: dataset is missing {missing}.", file=sys.stderr)
        return 1

    slots = season_calendar(args.season_start, args.season_end)
    months, days_in_window, split = months_of_window(slots)
    wraps = args.season_end < args.season_start

    print(f"  Source     : {region_dir}")
    print(f"  Region     : {args.region}")
    print(f"  Grid       : {ds.sizes['latitude']} lat x {ds.sizes['longitude']} lon")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
          f"{args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)" if wraps else ""))
    print(f"  LWP threshold: {args.lwp_threshold:g} g m-2 (values at or below are "
          f"zeroed before the median is taken)")
    print(f"  Panels     : {len(months)} months — "
          + ", ".join(calendar.month_abbr[m] for m in months))
    clipped = [m for m in months if days_in_window[m] < calendar.monthrange(2000, m)[1]]
    if clipped:
        detail = ", ".join(
            f"{calendar.month_abbr[m]} {days_in_window[m]}/"
            f"{calendar.monthrange(2000, m)[1]}" for m in clipped)
        print(f"  !! Window clips {detail} day(s); those panels average only the "
              f"days inside the window and are labelled.")
    if split:
        print(f"  !! Window enters {[calendar.month_abbr[m] for m in split]} twice; "
              f"that panel merges both fragments of the month.", file=sys.stderr)

    times = ds["valid_time"].values
    slot = season_slot_of(times, slots)
    syear = season_year_of(times, args.season_start)
    if not (slot >= 0).any():
        print("  Error: no time steps fall inside the requested season window.",
              file=sys.stderr)
        return 1

    cov = season_coverage(slot, syear, len(slots))
    print(f"\n  Seasons found ({len(cov)}), coverage of the {len(slots)}-day window:")
    for y, f in cov.items():
        flag = "" if f >= args.min_season_coverage else "   (below --min-season-coverage)"
        print(f"    {y}/{y+1}: {f*100:5.1f}%{flag}")

    if args.years is not None:
        absent = [y for y in args.years if y not in cov]
        if absent:
            print(f"\n  Error: no data for season(s) {absent}. "
                  f"Available: {sorted(cov)}", file=sys.stderr)
            return 1
        seasons = list(args.years)
        # An explicit request is honoured as given: --min-season-coverage only
        # guards the automatic default, it does not silently drop a season the
        # user named. Short ones are flagged instead.
        thin = [y for y in seasons if cov[y] < args.min_season_coverage]
        if thin:
            print(f"\n  !! Requested season(s) {thin} cover less than "
                  f"{args.min_season_coverage:.0%} of the window and are included "
                  f"anyway because you named them.", file=sys.stderr)
    else:
        seasons = [y for y, f in cov.items() if f >= args.min_season_coverage]
        if not seasons:
            print(f"\n  Error: no season meets --min-season-coverage "
                  f"{args.min_season_coverage}.", file=sys.stderr)
            return 1

    print(f"\n  Using {len(seasons)} season(s): {seasons}")

    stack = build_monthly_stack(ds, seasons, months, slot, syear, args.lwp_threshold)

    # Days each season contributes to each panel. Equal season weighting means a
    # thin season counts as much as a full one, so how thin they are is printed.
    print("\n  Days contributing per season and month:")
    hdr = "    season   " + "".join(f"{calendar.month_abbr[m]:>7}" for m in months)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for s_i, s in enumerate(seasons):
        row = f"    {s}/{str(s+1)[-2:]}  "
        for m_i, m in enumerate(months):
            n = stack["n_days"][s_i, m_i]
            row += f"{n:>7}" if n else f"{'--':>7}"
        print(row)
    print("    " + "-" * (len(hdr) - 4))
    print("    window   " + "".join(f"{days_in_window[m]:>7}" for m in months))

    lwp_m = nanmedian_quiet(stack["lwp_g_m2"], axis=0)
    ice_m = nanmean_quiet(stack["ice_frac"], axis=0)
    print(f"\n  Domain-median {LWP_LABEL.lower()} [g m-2] and sea ice fraction:")
    for m_i, m in enumerate(months):
        f_lwp = lwp_m[m_i][np.isfinite(lwp_m[m_i])]
        f_ice = ice_m[m_i][np.isfinite(ice_m[m_i])]
        line = f"    {calendar.month_abbr[m]:<5} "
        if f_lwp.size:
            line += (f"LWP {np.median(f_lwp):7.3f}  "
                     f"(range {f_lwp.min():7.3f} to {f_lwp.max():7.3f})")
        else:
            line += "LWP      --"
        # Sea ice is undefined over land, so this is the ocean-cell mean.
        line += f"   sea ice {f_ice.mean():.3f}" if f_ice.size else "   sea ice   --"
        print(line)

    if len(seasons) == 1:
        mode_label = f"season {seasons[0]}/{seasons[0]+1}"
        tag = f"season{seasons[0]}"
    else:
        mode_label = (f"median of {len(seasons)} seasons: "
                      + ", ".join(f"{s}/{str(s+1)[-2:]}" for s in seasons))
        tag = f"median{seasons[0]}-{seasons[-1]}"
    season_label = (f"{args.season_start[0]:02d}-{args.season_start[1]:02d} to "
                    f"{args.season_end[0]:02d}-{args.season_end[1]:02d}")

    output_path = args.output
    if output_path is None:
        out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_permonth" if args.per_month_scale else ""
        output_path = out_dir / f"{args.region}_monthly_lwp_maps_{tag}{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_month_grid(
        stack, months, days_in_window,
        region=args.region, mode_label=mode_label, season_label=season_label,
        lwp_threshold_g=args.lwp_threshold,
        ice_levels=tuple(args.ice_levels), contour_style=args.contour_style,
        per_month_scale=args.per_month_scale, show_site=not args.no_site_marker,
        projection=args.projection, ncols=args.ncols,
        output_path=output_path, dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
