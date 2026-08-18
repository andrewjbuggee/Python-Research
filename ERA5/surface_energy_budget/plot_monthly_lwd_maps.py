#!/usr/bin/env python3
"""Monthly maps of downwelling longwave radiation, one panel per month.

A 2-row by 3-column grid of spatial maps: every grid cell of the region carries
its monthly-mean downwelling longwave flux (``msdwlwrf``), averaged first within
each season and then across the seasons requested. Sea ice concentration is
contoured on top, so the radiation field can be read against the ice edge and the
consolidated pack, exactly as in ``plot_latitude_time_hovmoller.py``. The
colormap is imported from that script rather than re-declared, so the two figures
cannot drift apart.

Relationship to plot_monthly_longwave_maps.py
---------------------------------------------
That script draws three quantities as rows (LW down, LW net, sea ice) across
calendar months of the whole record. This one draws a SINGLE quantity as a grid
of months, and selects its time window the way the Hovmoller does: by a season
that may wrap the new year, and by the seasons (years) to average over. Use that
one to compare quantities, this one to compare months of a chosen freeze-up
season at full spatial detail.

The season
----------
The default window, 1 August to 31 January, **wraps the new year**, so a season
is labelled by the calendar year it starts in: season 2019 means Aug 2019 to
Jan 2020. That window touches exactly six calendar months, which is what fills
the 2x3 grid. Any window may be given; the grid stays three columns wide and
grows as many rows as the months require (``--ncols`` overrides).

A month clipped by the window -- say April under ``--season-end 04-01`` -- is
averaged over only the days inside the window, and the panel says so. It is not
silently presented as a full April.

Averaging over years
--------------------
``--years`` selects which seasons are averaged, by the year each season STARTS
in: ``--years 2019`` is one season, ``--years 2019-2024`` averages six. The
default is every season in the record that meets ``--min-season-coverage``;
naming years explicitly overrides that filter, so a short season is still used
if you ask for it (with a warning), because an explicit request should be
honoured rather than silently dropped.

Each season is weighted EQUALLY: the mean map for a (season, month) pair is
formed first, and those maps are then averaged. A season contributing 20 days to
October therefore does not outweigh one contributing 31. Pooling every time step
instead would weight by sample count; the per-season table printed at run time
shows how uneven that would have been.

Colour scale
------------
Downwelling longwave is a magnitude with no meaningful zero in its range, so it
takes a sequential ramp (magma), with limits from robust percentiles pooled over
all panels -- a colour means the same thing in every month, which is the point of
the figure. ``--per-month-scale`` gives each panel its own limits and its
own colorbar instead, revealing within-month structure that the shared ramp
crushes (January sits at the dark end of an ~180-310 W m-2 scale) at the cost
of comparability between panels.

SIGN CONVENTION: native ERA5, **positive downward**. Downwelling longwave is
positive everywhere by construction.

Examples
--------
Climatology over every complete season in the record, Aug to Jan::

    python plot_monthly_lwd_maps.py --storage external --region barrow

Six seasons averaged, the usual freeze-up view::

    python plot_monthly_lwd_maps.py --region barrow --years 2019-2024

One season, and a nine-month window (grid becomes 3x3)::

    python plot_monthly_lwd_maps.py --region barrow --years 2019 \
        --season-start 08-01 --season-end 04-01

Different ice contours, and each panel on its own colour limits::

    python plot_monthly_lwd_maps.py --region barrow --years 2019-2024 \
        --ice-levels 0.15,0.8 --per-month-scale
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
from matplotlib import patheffects

# The colormap, robust-percentile setting and CLI parsers are imported from the
# Hovmoller script rather than copied, so "the same colormap as the Hovmoller"
# stays true if that script ever changes.
from ERA5.surface_energy_budget.plot_latitude_time_hovmoller_DLR import (
    COLOR_PERCENTILE,
    QUANTITIES,
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

LWD_VAR, LWD_LABEL, LWD_UNITS, _LWD_KIND, LWD_CMAP = QUANTITIES["lwd"]

# Sea ice contour levels. 0.05 traces the outer edge of any ice at all and 0.95
# the edge of the consolidated pack, so together they bracket the marginal ice
# zone -- the band where the surface is neither open water nor solid ice.
DEFAULT_ICE_LEVELS = (0.05, 0.95)

# Aug to Jan inclusive: six calendar months, the 2x3 grid.
DEFAULT_SEASON_START = (8, 1)
DEFAULT_SEASON_END = (1, 31)

ICE_LINESTYLES = ("--", "-", ":", "-.")


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
) -> dict:
    """Mean maps of downwelling longwave and sea ice for every (season, month).

    Returns arrays shaped ``(season, month, latitude, longitude)`` plus the
    number of day-slots each pair contributed.

    One season is loaded at a time. A six-season, six-month window is roughly
    26,000 hourly steps over the grid; pulling all of it at once to produce 36
    small mean maps costs about half a gigabyte for no benefit.
    """
    lat_deg = ds["latitude"].values
    lon_deg = ds["longitude"].values
    n_s, n_m = len(seasons), len(months)
    lwd_W_m2 = np.full((n_s, n_m, lat_deg.size, lon_deg.size), np.nan)
    ice_frac = np.full_like(lwd_W_m2, np.nan)
    n_days = np.zeros((n_s, n_m), dtype=int)

    in_window = slot >= 0
    times = ds["valid_time"].values
    mon = month_of(times)

    for s_i, s in enumerate(seasons):
        sel_s = in_window & (syear == s)
        idx_s = np.nonzero(sel_s)[0]
        band = ds[[LWD_VAR, "siconc"]].isel(valid_time=idx_s).load()
        lwd_t = band[LWD_VAR].values      # (time, lat, lon)
        ice_t = band["siconc"].values     # (time, lat, lon); NaN over land
        mon_s = mon[idx_s]
        slot_s = slot[idx_s]
        for m_i, m in enumerate(months):
            take = mon_s == m
            if not take.any():
                continue
            lwd_W_m2[s_i, m_i] = nanmean_quiet(lwd_t[take], axis=0)
            ice_frac[s_i, m_i] = nanmean_quiet(ice_t[take], axis=0)
            n_days[s_i, m_i] = len(np.unique(slot_s[take]))
        del band, lwd_t, ice_t

    return {
        "lwd_W_m2": lwd_W_m2,
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
    """Robust sequential limits, pooled over whichever panels are passed."""
    finite = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [100 - COLOR_PERCENTILE, COLOR_PERCENTILE])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


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
    ice_levels: tuple[float, ...],
    contour_style: str = "contrast",
    per_month_scale: bool = False,
    show_site: bool = True,
    projection: str = "polar",
    ncols: int | None = None,
    output_path: Path | None = None,
    dpi: int = 150,
):
    """Draw the month grid of downwelling longwave and return the Figure."""
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
    # Average the per-season maps: every season counts once, whatever its length.
    lwd_m = nanmean_quiet(stack["lwd_W_m2"], axis=0)   # (month, lat, lon)
    ice_m = nanmean_quiet(stack["ice_frac"], axis=0)
    n_seasons_used = np.sum(
        ~np.all(np.isnan(stack["lwd_W_m2"]), axis=(2, 3)), axis=0
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

    vmin, vmax = color_limits([lwd_m[i] for i in range(len(months))])

    mesh = None
    for i, m in enumerate(months):
        ax = flat[i]
        lo, hi = (color_limits([lwd_m[i]]) if per_month_scale else (vmin, vmax))
        plot_kw = dict(cmap=LWD_CMAP, vmin=lo, vmax=hi, shading="auto")
        if use_cartopy:
            plot_kw["transform"] = data_crs
        mesh = ax.pcolormesh(lon_deg, lat_deg, lwd_m[i], **plot_kw)

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
                    # A plain black line disappears at the dark end of magma.
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
            # Land under the data: downwelling longwave is defined over land, so
            # the field shows there and the grey only fills what is outside it.
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
            # Mid grey, not the darker 0.35 the sibling scripts use: magma runs
            # to near-black in the winter panels, where a dark coastline is
            # simply invisible.
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
        cb.set_label(f"{LWD_LABEL} [{LWD_UNITS}]", fontsize=10)
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
        f"ERA5 {LWD_LABEL.lower()} ({LWD_VAR}) — {region}\n"
        f"monthly mean per grid cell [{LWD_UNITS}], season {season_label}"
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
    parser.add_argument("--ice-levels", type=parse_levels, default=DEFAULT_ICE_LEVELS,
                        metavar="L1,L2",
                        help="Sea ice contour levels (default 0.05,0.95).")
    parser.add_argument("--contour-style", choices=("black", "contrast"),
                        default="contrast",
                        help="How the sea ice contours are drawn. 'contrast' "
                             "(default) outlines the black lines in white so they "
                             "stay readable over the dark end of magma; 'black' is "
                             "a plain unoutlined line.")
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
    print("Monthly maps of downwelling longwave radiation")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = sorted({LWD_VAR, "siconc"} - set(ds.data_vars))
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

    stack = build_monthly_stack(ds, seasons, months, slot, syear)

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

    lwd_m = nanmean_quiet(stack["lwd_W_m2"], axis=0)
    ice_m = nanmean_quiet(stack["ice_frac"], axis=0)
    print(f"\n  Domain-mean {LWD_LABEL.lower()} [W m-2] and sea ice fraction:")
    for m_i, m in enumerate(months):
        f_lwd = lwd_m[m_i][np.isfinite(lwd_m[m_i])]
        f_ice = ice_m[m_i][np.isfinite(ice_m[m_i])]
        line = f"    {calendar.month_abbr[m]:<5} "
        if f_lwd.size:
            line += (f"LW down {f_lwd.mean():7.2f}  "
                     f"(range {f_lwd.min():7.2f} to {f_lwd.max():7.2f})")
        else:
            line += "LW down      --"
        # Sea ice is undefined over land, so this is the ocean-cell mean.
        line += f"   sea ice {f_ice.mean():.3f}" if f_ice.size else "   sea ice   --"
        print(line)

    if len(seasons) == 1:
        mode_label = f"season {seasons[0]}/{seasons[0]+1}"
        tag = f"season{seasons[0]}"
    else:
        mode_label = (f"mean of {len(seasons)} seasons: "
                      + ", ".join(f"{s}/{str(s+1)[-2:]}" for s in seasons))
        tag = f"mean{seasons[0]}-{seasons[-1]}"
    season_label = (f"{args.season_start[0]:02d}-{args.season_start[1]:02d} to "
                    f"{args.season_end[0]:02d}-{args.season_end[1]:02d}")

    output_path = args.output
    if output_path is None:
        out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_permonth" if args.per_month_scale else ""
        output_path = out_dir / f"{args.region}_monthly_lwd_maps_{tag}{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_month_grid(
        stack, months, days_in_window,
        region=args.region, mode_label=mode_label, season_label=season_label,
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
