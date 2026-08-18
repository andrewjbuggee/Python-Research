#!/usr/bin/env python3
"""Latitude-time (Hovmoller) sections of surface longwave radiation.

Latitude on the y axis, time on the x axis, for a narrow meridional strip
running north from the DOE ARM site at Utqiagvik, Alaska. Sea ice concentration
is contoured on top, so the radiation field can be read against the advancing
ice edge.

Three quantities, selected with ``--quantity``:

    lwd         downwelling longwave              msdwlwrf     [W m-2]
    lwnet       net longwave                      msnlwrf      [W m-2]
    dlwd_dlat   meridional gradient of msdwlwrf                [W m-2 per deg lat]
    all         produces all three figures

The strip
---------
At each latitude the value is the mean of ``--n-lon-cells`` grid cells (3 by
default) centred on the ARM longitude, which smooths the point-to-point noise of
a single 0.25 deg column without smearing across the coast.

The y axis runs from ``--lat-south`` (default: the southern edge of the data,
70 N) to ``--lat-north`` (default: its northern edge, 80 N), so the section
includes the coastal land at the bottom. The DOE ARM site at Utqiagvik is marked
on the y axis, and a horizontal line shows the land edge -- the northernmost
latitude at which any cell in the strip is land.

On the spatial average: collapsing one dimension by averaging is standard
Hovmoller practice, not an embellishment. The canonical equatorial-Pacific
diagram averages roughly 2S-2N (or 5S-5N) before plotting longitude against
time. Three cells here is a deliberately conservative version of the same idea:
about 27 km at 71 N, enough to damp point-to-point noise without smearing the
coastal gradient.

The season
----------
The default window, 1 August to 1 April, **wraps the new year**, so a "season"
is labelled by the calendar year it starts in: season 2000 means Aug 2000 to
Apr 2001. Time is indexed by day-of-season rather than by date, which keeps the
axis identical across seasons of different length and lets several be averaged.
29 February gets its own slot; in non-leap seasons it is simply empty, and
all-empty columns are dropped rather than drawn as gaps.

Averaging over years
--------------------
``--years`` selects which seasons are averaged, by the year each season STARTS
in: ``--years 2019`` is one season, ``--years 2019-2025`` averages seven. The
default is every season in the record that meets ``--min-season-coverage``;
naming years explicitly overrides that filter, so a short season is still used
if you ask for it (with a warning), because an explicit request should be
honoured rather than silently dropped.

Downwelling and net longwave are averaged over time at each (latitude,
time-of-season) cell, so both panels describe the same locations and steps.

The derivative panel is the **time mean of the derivative, not the derivative of
the time mean**: d/dlatitude is evaluated at every individual time step, year,
and location first, and only then averaged. The two differ whenever the
meridional gradient co-varies with the flow, which is precisely the freeze-up
case this plot is for.

Averaging seasons of unequal coverage mixes well- and poorly-sampled columns, so
per-season completeness is always printed.

SIGN CONVENTION: native ERA5, **positive downward**. Net longwave is negative
wherever the surface radiates away more than it receives.

Examples
--------
Climatology of downwelling longwave, all complete seasons::

    python plot_latitude_time_hovmoller.py --storage external --region barrow

One season, net longwave::

    python plot_latitude_time_hovmoller.py --region barrow \
        --quantity lwnet --years 2000

Several seasons averaged together::

    python plot_latitude_time_hovmoller.py --region barrow \
        --quantity all --years 2019-2025

All three quantities, custom window and ice contours::

    python plot_latitude_time_hovmoller.py --storage external --region barrow \
        --quantity all --season-start 08-01 --season-end 04-01 \
        --ice-levels 0.15,0.5,0.9
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from matplotlib import patheffects

from plot_monthly_longwave_maps import ARM_UTQIAGVIK_LAT, ARM_UTQIAGVIK_LON
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_region_dir,
)

# quantity -> (source variable, label, units, scale kind, colormap)
QUANTITIES = {
    "lwd": ("msdwlwrf", "Downwelling longwave", "W m$^{-2}$", "sequential", "magma"),
    "lwnet": ("msnlwrf", "Net longwave", "W m$^{-2}$", "diverging", "RdBu_r"),
    "dlwd_dlat": ("msdwlwrf", "d(downwelling LW)/d(latitude)",
                  "W m$^{-2}$ per $^\\circ$lat", "diverging", "PuOr_r"),
}

DEFAULT_ICE_LEVELS = (0.1, 0.8)
COLOR_PERCENTILE = 98.0


def nanmean_quiet(a, axis=None):
    """np.nanmean without the "Mean of empty slice" RuntimeWarning.

    All-NaN slices are an expected, meaningful state here, not a mistake: land
    cells have no sea ice, and a day-of-season slot has no data in a season that
    did not cover it. Both should yield NaN silently. np.errstate does not help
    -- numpy raises this through the warnings module, not the FP error state --
    so the warning is filtered directly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def parse_years(text: str) -> tuple[int, ...]:
    """Parse '2019', '2019-2025', or '2000,2019-2020' into a sorted year tuple."""
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
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(2000, month, day)  # 2000 is a leap year, so 02-29 validates
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid MM-DD") from None
    return month, day


def parse_levels(text: str) -> tuple[float, ...]:
    try:
        vals = tuple(sorted(float(v) for v in text.split(",") if v.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of numbers"
        ) from None
    if not vals:
        raise argparse.ArgumentTypeError("at least one contour level is required")
    return vals


def season_calendar(
    start_md: tuple[int, int], end_md: tuple[int, int]
) -> list[tuple[int, int]]:
    """Ordered (month, day) slots from start to end, wrapping the new year.

    Built on a leap year so 29 February has a slot; seasons without it simply
    leave that column empty.
    """
    m0, d0 = start_md
    m1, d1 = end_md
    wraps = (m1, d1) < (m0, d0)
    # Reference years chosen so that whichever year contains February is a leap
    # year, giving 29 Feb a slot. For a wrapping window that is the SECOND year
    # (1999 -> 2000); for a non-wrapping one it is the only year (2000).
    if wraps:
        cur, stop = date(1999, m0, d0), date(2000, m1, d1)
    else:
        cur, stop = date(2000, m0, d0), date(2000, m1, d1)
    out = []
    while cur <= stop:
        out.append((cur.month, cur.day))
        cur += timedelta(days=1)
    return out


def season_year_of(times: np.ndarray, start_md: tuple[int, int]) -> np.ndarray:
    """Which season each timestamp belongs to, for a possibly wrapping window."""
    years = times.astype("datetime64[Y]").astype(int) + 1970
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    m0, d0 = start_md
    before_start = (months * 100 + days) < (m0 * 100 + d0)
    return np.where(before_start, years - 1, years)


def build_section(
    ds,
    quantity: str,
    lon_center: float,
    n_lon_cells: int,
    lat_south: float,
    lat_north: float,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
):
    """Reduce the dataset to (season, day-of-season, latitude) arrays.

    Returns a dict with the quantity field, the sea ice field, the latitude
    axis, the day-of-season labels, and the seasons found.
    """
    var, _, _, _, _ = QUANTITIES[quantity]

    lon = ds["longitude"].values
    j = int(np.abs(lon - lon_center).argmin())
    half = n_lon_cells // 2
    j0, j1 = max(0, j - half), min(lon.size, j + half + 1)

    lat_all = ds["latitude"].values
    keep_lat = (lat_all >= lat_south - 1e-9) & (lat_all <= lat_north + 1e-9)
    if not keep_lat.any():
        raise ValueError(f"No grid latitudes between {lat_south} and {lat_north}.")

    # Subset variables AND space while still lazy, then load. Calling .compute()
    # on the whole record first would pull ~9.5 GB (35 variables x the full grid)
    # to obtain a 3-variable, 3-column strip.
    band = (
        ds[[var, "siconc"]]
        .isel(longitude=slice(j0, j1), latitude=np.nonzero(keep_lat)[0])
        .load()
    )
    lat = band["latitude"].values

    # nanmean across the strip: over land siconc is NaN, so a coastal row
    # averages only its ocean cells rather than collapsing to NaN.
    field_t = nanmean_quiet(band[var].values, axis=2)     # (time, lat)
    ice_t = nanmean_quiet(band["siconc"].values, axis=2)  # (time, lat)

    # Land edge: northernmost latitude where ANY cell of the strip is land.
    land_any = np.isnan(band["siconc"].values).any(axis=(0, 2))
    land_edge = float(lat[land_any].max()) if land_any.any() else None

    if quantity == "dlwd_dlat":
        # The derivative is taken PER TIME STEP, before any averaging: field_t is
        # (time, lat) here, and every slot/season mean below is applied to the
        # result. So what is plotted is the TIME-MEAN OF THE DERIVATIVE, not the
        # derivative of the time-mean. The two differ whenever the meridional
        # gradient varies with the flow, which is exactly the freeze-up case.
        # (Averaging the 3 longitudes first is harmless: that is a linear
        # operation on a different axis, so it commutes with d/dlat.)
        order = np.argsort(lat)
        field_t = np.gradient(field_t[:, order], lat[order], axis=1)
        ice_t = ice_t[:, order]
        lat = lat[order]
    else:
        order = np.argsort(lat)
        field_t, ice_t, lat = field_t[:, order], ice_t[:, order], lat[order]

    times = ds["valid_time"].values
    slots = season_calendar(start_md, end_md)
    slot_index = {md: i for i, md in enumerate(slots)}
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    dos = np.array([slot_index.get((int(m), int(d)), -1)
                    for m, d in zip(months, days)])
    seasons = season_year_of(times, start_md)

    in_window = dos >= 0
    if not in_window.any():
        raise ValueError("No time steps fall inside the requested season window.")

    uniq_seasons = sorted(set(seasons[in_window].tolist()))
    n_slot, n_lat = len(slots), lat.size
    stack = np.full((len(uniq_seasons), n_slot, n_lat), np.nan)
    ice_stack = np.full_like(stack, np.nan)
    counts = np.zeros((len(uniq_seasons), n_slot), dtype=int)

    for s_i, s in enumerate(uniq_seasons):
        sel_s = in_window & (seasons == s)
        for slot in np.unique(dos[sel_s]):
            sel = sel_s & (dos == slot)
            stack[s_i, slot] = nanmean_quiet(field_t[sel], axis=0)
            ice_stack[s_i, slot] = nanmean_quiet(ice_t[sel], axis=0)
            counts[s_i, slot] = int(sel.sum())

    return {
        "field": stack, "ice": ice_stack, "counts": counts,
        "lat": lat, "slots": slots, "seasons": uniq_seasons,
        "land_edge": land_edge,
        "lon_used": lon[j0:j1],
    }


def month_ticks(slots: list[tuple[int, int]]) -> tuple[list[int], list[str]]:
    """Tick at the first of each month present, labelled by month abbreviation."""
    pos, lab = [], []
    for i, (m, d) in enumerate(slots):
        if d == 1:
            pos.append(i)
            lab.append(calendar.month_abbr[m])
    return pos, lab


def make_hovmoller(
    sec: dict,
    quantity: str,
    region: str,
    mode_label: str,
    ice_levels: tuple[float, ...],
    contour_style: str = "contrast",
    output_path: Path | None = None,
    dpi: int = 150,
):
    import matplotlib.pyplot as plt

    _, qlabel, units, kind, cmap = QUANTITIES[quantity]

    # How many seasons actually contribute to each column. Seasons rarely cover
    # the same part of the window, so this varies along x -- and where it jumps,
    # the climatology jumps with it. That is the source of any vertical striping,
    # so it is measured and reported rather than left to be misread as signal.
    n_contrib = np.sum(~np.all(np.isnan(sec["field"]), axis=2), axis=0)

    field = nanmean_quiet(sec["field"], axis=0)   # average over seasons
    ice = nanmean_quiet(sec["ice"], axis=0)
    lat, slots = sec["lat"], sec["slots"]

    # Drop slots with no data at all (e.g. 29 Feb in non-leap seasons).
    keep = ~np.all(np.isnan(field), axis=1)
    field, ice, n_contrib = field[keep], ice[keep], n_contrib[keep]
    kept_slots = [s for s, k in zip(slots, keep) if k]
    x = np.arange(len(kept_slots))

    finite = field[np.isfinite(field)]
    if kind == "diverging":
        v = float(np.percentile(np.abs(finite), COLOR_PERCENTILE)) if finite.size else 1.0
        vmin, vmax = -v, v
    else:
        vmin, vmax = (np.percentile(finite, [100 - COLOR_PERCENTILE, COLOR_PERCENTILE])
                      if finite.size else (0.0, 1.0))

    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    mesh = ax.pcolormesh(x, lat, field.T, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto")

    if np.isfinite(ice).any():
        cs = ax.contour(x, lat, ice.T, levels=list(ice_levels), colors="black",
                        linewidths=[1.0 + 0.5 * i for i in range(len(ice_levels))],
                        linestyles=["--", "-", ":", "-."][: len(ice_levels)],
                        zorder=5)
        labels = ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=9, inline=True)

        if contour_style == "contrast":
            # A plain black line disappears wherever the colormap goes dark --
            # the top of magma, the deep end of RdBu_r. Outlining each line in
            # white makes it readable on any background without changing the
            # line colour, which is preferable to picking a single "visible"
            # hue that then competes with the data for meaning.
            halo = [patheffects.withStroke(linewidth=3.2, foreground="white")]
            for coll in cs.collections if hasattr(cs, "collections") else [cs]:
                coll.set_path_effects(halo)
            for lab in labels:
                lab.set_path_effects(
                    [patheffects.withStroke(linewidth=2.6, foreground="white")]
                )

    if sec["land_edge"] is not None:
        ax.axhline(sec["land_edge"], color="#00A000", linewidth=2.2, zorder=6)
        ax.annotate("land edge", xy=(0.994, sec["land_edge"]),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points", ha="right",
                    fontsize=12, color="#006400", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

    # DOE ARM Utqiagvik, marked on the y axis. Drawn as a bullseye outside the
    # axes plus a short tick reaching in, so it is unmistakable against any of
    # the colour ramps without obscuring data.
    if lat.min() <= ARM_UTQIAGVIK_LAT <= lat.max():
        ytrans = ax.get_yaxis_transform()
        ax.plot([-0.012], [ARM_UTQIAGVIK_LAT], marker="o", markersize=11,
                clip_on=False, markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=2.0, transform=ytrans, zorder=10, linestyle="none")
        ax.plot([-0.012], [ARM_UTQIAGVIK_LAT], marker="o", markersize=4,
                clip_on=False, markerfacecolor="black", markeredgecolor="black",
                transform=ytrans, zorder=11, linestyle="none")
        ax.plot([0.0, 0.035], [ARM_UTQIAGVIK_LAT] * 2, color="black",
                linewidth=2.0, clip_on=False, transform=ytrans, zorder=10)
        ax.annotate(f"ARM Utqiaġvik\n{ARM_UTQIAGVIK_LAT:.2f}°N",
                    xy=(-0.012, ARM_UTQIAGVIK_LAT), xycoords=ytrans,
                    xytext=(-14, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, zorder=11)

    pos, lab = month_ticks(kept_slots)
    ax.set_xticks(pos)
    ax.set_xticklabels(lab)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(lat.min(), lat.max())
    ax.set_ylabel("Latitude [$^\\circ$N]", fontsize=10)
    ax.set_xlabel(
        f"Day of season ({kept_slots[0][0]:02d}-{kept_slots[0][1]:02d} to "
        f"{kept_slots[-1][0]:02d}-{kept_slots[-1][1]:02d})", fontsize=10)
    lon_used = sec["lon_used"]
    ax.set_title(
        f"{qlabel} — {region}, meridional strip north of Utqiaġvik\n"
        f"{mode_label}   |   {lon_used.size}-cell longitude mean "
        f"({lon_used.min():.2f} to {lon_used.max():.2f}°E)   |   "
        f"contours: sea ice {', '.join(f'{v:g}' for v in ice_levels)}",
        fontsize=11.5)
    if n_contrib.size and n_contrib.min() != n_contrib.max():
        # A single vague on-plot caption ("varies X-Y") reads as if the whole
        # figure were undersampled, when in practice it is almost always one
        # column: 29 Feb only exists in a season whose wrap year is a leap
        # year, so every OTHER season leaves that column empty. Naming that
        # column separately from any other thin day keeps a real gap
        # elsewhere from being mistaken for the same harmless artifact.
        n_used = sec["field"].shape[0]
        lo, hi = int(n_contrib.min()), int(n_contrib.max())
        thin = [(kept_slots[i], int(n_contrib[i])) for i in range(len(kept_slots))
                if n_contrib[i] < hi]
        leap_day = [n for (md, n) in thin if md == (2, 29)]
        other_thin = [(md, n) for (md, n) in thin if md != (2, 29)]
        print(f"\n  Season coverage varies by day of season: {lo}-{hi} of "
              f"{n_used} seasons contribute per column.")
        if leap_day:
            print(f"    29 Feb: only {leap_day[0]} season(s) wrap into a leap "
                  f"year and cover it -- expected, not a data gap.")
        if other_thin:
            print("    Other thin day(s): " + ", ".join(
                f"{calendar.month_abbr[m]} {d} ({n} of {n_used} seasons)"
                for (m, d), n in other_thin))
    ax.grid(alpha=0.18, linewidth=0.5, color="white")
    ax.set_axisbelow(False)

    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=26)
    cb.set_label(f"{qlabel} [{units}]", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"    figure -> {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--quantity", choices=(*QUANTITIES, "all"), default="lwd",
                        help="Which field to section (default lwd; 'all' makes all three).")
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD", help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(4, 1),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 04-01, wrapping the year).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to average, by the year each season STARTS "
                             "in: '2019', '2019-2025', or '2000,2019-2020'. "
                             "Default: every season in the record that meets "
                             "--min-season-coverage. Naming years explicitly "
                             "overrides that filter, so a short season is still "
                             "included if you ask for it.")
    parser.add_argument("--lat-south", type=float, default=None,
                        help="Southern edge (default: the southern edge of the data, "
                             "70 N for the barrow strip). The ARM site is marked on "
                             "the y axis wherever it falls.")
    parser.add_argument("--lat-north", type=float, default=None,
                        help="Northern edge (default: northern edge of the data).")
    parser.add_argument("--lon-center", type=float, default=ARM_UTQIAGVIK_LON,
                        help="Longitude the strip is centred on (default: ARM site).")
    parser.add_argument("--n-lon-cells", type=int, default=3,
                        help="Grid cells averaged across longitude (default 3).")
    parser.add_argument("--ice-levels", type=parse_levels, default=DEFAULT_ICE_LEVELS,
                        metavar="L1,L2",
                        help="Sea ice contour levels (default 0.1,0.8).")
    parser.add_argument("--contour-style", choices=("black", "contrast"),
                        default="contrast",
                        help="How the sea ice contours are drawn. 'contrast' "
                             "(default) outlines the black lines in white so they "
                             "stay readable over the dark end of a colormap; "
                             "'black' is a plain unoutlined line.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6, metavar="F",
                        help="Exclude seasons covering less than this fraction of the "
                             "window from a climatology (default 0.6).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Latitude-time sections of surface longwave")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    needed = {"msdwlwrf", "msnlwrf", "siconc"}
    missing = sorted(needed - set(ds.data_vars))
    if missing:
        print(f"  Error: dataset is missing {missing}.", file=sys.stderr)
        return 1

    lat_north = args.lat_north if args.lat_north is not None else float(
        ds["latitude"].max())
    lat_south = args.lat_south if args.lat_south is not None else float(
        ds["latitude"].min())

    print(f"  Source     : {region_dir}")
    print(f"  Strip      : {lat_south:.3f}N to {lat_north:.3f}N, "
          f"{args.n_lon_cells} cells about {args.lon_center:.3f}E")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
          f"{args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)"
             if args.season_end < args.season_start else ""))

    quantities = list(QUANTITIES) if args.quantity == "all" else [args.quantity]

    try:
        sec = build_section(ds, quantities[0], args.lon_center, args.n_lon_cells,
                            lat_south, lat_north,
                            args.season_start, args.season_end)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    n_slot = len(sec["slots"])
    print(f"\n  Seasons found ({len(sec['seasons'])}), coverage of the "
          f"{n_slot}-day window:")
    frac = {}
    for s_i, s in enumerate(sec["seasons"]):
        f = float((sec["counts"][s_i] > 0).sum()) / n_slot
        frac[s] = f
        print(f"    {s}/{s+1}: {f*100:5.1f}%" + ("" if f >= args.min_season_coverage
                                                 else "   (below --min-season-coverage)"))

    if args.years is not None:
        missing = [y for y in args.years if y not in sec["seasons"]]
        if missing:
            print(f"\n  Error: no data for season(s) {missing}. "
                  f"Available: {sec['seasons']}", file=sys.stderr)
            return 1
        # An explicit request is honoured as given: --min-season-coverage only
        # guards the automatic default, it does not silently drop a season the
        # user named. Short ones are flagged instead.
        keep_idx = [sec["seasons"].index(y) for y in args.years]
        thin = [y for y in args.years if frac[y] < args.min_season_coverage]
        if thin:
            print(f"\n  !! Requested season(s) {thin} cover less than "
                  f"{args.min_season_coverage:.0%} of the window and are included "
                  f"anyway because you named them.", file=sys.stderr)
    else:
        keep_idx = [i for i, s in enumerate(sec["seasons"])
                    if frac[s] >= args.min_season_coverage]
        if not keep_idx:
            print(f"\n  Error: no season meets --min-season-coverage "
                  f"{args.min_season_coverage}.", file=sys.stderr)
            return 1

    used = [sec["seasons"][i] for i in keep_idx]
    if len(used) == 1:
        mode_label = f"season {used[0]}/{used[0]+1}"
    else:
        mode_label = (f"mean of {len(used)} seasons: "
                      + ", ".join(f"{u}/{u+1}" for u in used))
    print(f"\n  Using {len(used)} season(s): {used}")

    if sec["land_edge"] is not None:
        print(f"  Land edge  : {sec['land_edge']:.2f}N "
              f"(northernmost latitude with any land in the strip)")

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    print()
    for q in quantities:
        s = (sec if q == quantities[0]
             else build_section(ds, q, args.lon_center, args.n_lon_cells,
                                lat_south, lat_north,
                                args.season_start, args.season_end))
        s = dict(s)
        s["field"] = s["field"][keep_idx]
        s["ice"] = s["ice"][keep_idx]

        mean_field = nanmean_quiet(s["field"], axis=0)
        finite = mean_field[np.isfinite(mean_field)]
        label = QUANTITIES[q][1]
        plain = (QUANTITIES[q][2].replace("$", "").replace("^{-2}", "-2")
                 .replace("^\\circ", "deg ").replace("\\", ""))
        print(f"  {label}: mean={finite.mean():.2f}, "
              f"range {finite.min():.2f} to {finite.max():.2f} [{plain}]")

        tag = (f"season{used[0]}" if len(used) == 1
               else f"mean{used[0]}-{used[-1]}")
        make_hovmoller(s, q, args.region, mode_label, tuple(args.ice_levels),
                       contour_style=args.contour_style,
                       output_path=out_dir / f"{args.region}_hovmoller_{q}_{tag}.png",
                       dpi=args.dpi)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
