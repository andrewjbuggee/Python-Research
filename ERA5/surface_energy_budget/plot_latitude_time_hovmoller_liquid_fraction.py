#!/usr/bin/env python3
"""Latitude-time (Hovmoller) section of the cloudy-scene liquid fraction.

Latitude on the y axis, time on the x axis, for a narrow meridional strip
running north from the DOE ARM site at Utqiagvik, Alaska -- identical in every
other respect to ``plot_latitude_time_hovmoller.py`` and its LWP sibling
``plot_latitude_time_hovmoller_lwp.py`` (same season-window, year-averaging,
strip-averaging and sea ice contour machinery, imported not copied). This one
plots the fraction of overcast cell-hours that also carry liquid, using the
exact definitions and ratio from ``analyze_cloud_liquid_frequency.py``:

    cloudy = tcc >= --min-cloud-fraction AND (LWP > --lwp-threshold
                                               OR IWP > --iwp-threshold)
    liquid = cloudy AND LWP > --lwp-threshold

    fraction = (cloudy cell-hours with liquid) / (cloudy cell-hours)

See that script's module docstring for what this fraction is and is not: a
conditional relative frequency over CELL-HOURS, not a per-cloud statistic, and
a census rather than a sample.

Mean, not median
-----------------
Unlike the LWP magnitude in the sibling Hovmoller script, this fraction is
plotted as a MEAN across seasons, as requested: for each (day-of-season slot,
latitude), the ratio above is formed once per season by pooling the 3-cell
longitude strip and every hour in that slot, and the per-season ratios are
then averaged. Averaging the ratios, not re-pooling raw counts across seasons,
keeps every season weighted equally regardless of its cell-hour count, the
same equal-season-weighting the rest of this codebase uses.

Threshold
---------
``--lwp-threshold`` (default 5 g m-2, unlike the near-zero default used
elsewhere) sets both the liquid test and half of the cloudy test, exactly as
in ``analyze_cloud_liquid_frequency.py``. A higher default here is deliberate:
near a microwave radiometer's detection floor (roughly 10-25 g m-2) is a more
useful cutoff for a fraction meant to say "cloud liquid that would actually be
retrievable," rather than the near-zero floor used to zero out ERA5's
numerical trace liquid in the two magnitude figures.

Colour scale
------------
A white (0%) to dark grey (100%) sequential ramp, fixed to the fraction's
natural 0-100% range rather than a robust percentile -- unlike LWP, this
quantity has a real, meaningful ceiling.

Examples
--------
Climatology, every complete season in the record::

    python plot_latitude_time_hovmoller_liquid_fraction.py --storage external --region barrow

One season::

    python plot_latitude_time_hovmoller_liquid_fraction.py --region barrow --years 2019

Several seasons, a stricter liquid threshold::

    python plot_latitude_time_hovmoller_liquid_fraction.py --region barrow \\
        --years 2019-2025 --lwp-threshold 10
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

from ERA5.surface_energy_budget.plot_latitude_time_hovmoller_DLR import (
    nanmean_quiet,
    parse_levels,
    parse_month_day,
    parse_years,
    season_calendar,
    season_year_of,
)
from plot_monthly_longwave_maps import ARM_UTQIAGVIK_LAT, ARM_UTQIAGVIK_LON
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_region_dir,
)

LIQUID_FRAC_LABEL = "Cloudy scenes with liquid"
LIQUID_FRAC_UNITS = "%"

# White at 0%, dark grey (not black) at 100%.
LIQUID_FRAC_CMAP = LinearSegmentedColormap.from_list(
    "white_darkgrey", ["#ffffff", "#333333"]
)

# LWP above which liquid counts as present, g m-2. Higher than the near-zero
# default used in the LWP magnitude scripts -- see the module docstring.
DEFAULT_LWP_THRESHOLD_G = 5.0

# IWP used in the cloudy test, kg m-2, and the cloud-cover floor for a scene to
# count as cloudy at all. Both match analyze_cloud_liquid_frequency.py's
# defaults.
DEFAULT_IWP_THRESHOLD_KG = 0.0
DEFAULT_MIN_CLOUD_FRACTION = 1.0

DEFAULT_ICE_LEVELS = (0.05, 0.95)


def build_section(
    ds,
    lon_center: float,
    n_lon_cells: int,
    lat_south: float,
    lat_north: float,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
    lwp_threshold_g: float,
    iwp_threshold_kg: float,
    min_cloud_fraction: float,
):
    """Reduce the dataset to (season, day-of-season, latitude) arrays.

    Returns a dict with the liquid FRACTION field (%, one ratio per season
    formed by pooling the longitude strip and every hour in a day-of-season
    slot -- see the module docstring), the sea ice field (mean), the latitude
    axis, the day-of-season labels, and the seasons found.
    """
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
    # to obtain a 4-variable, 3-column strip.
    band = (
        ds[["tcc", "tclw", "tciw", "siconc"]]
        .isel(longitude=slice(j0, j1), latitude=np.nonzero(keep_lat)[0])
        .load()
    )
    lat = band["latitude"].values

    tcc = band["tcc"].values
    tclw_g = band["tclw"].values * 1000.0    # kg m-2 -> g m-2
    tciw = band["tciw"].values               # kg m-2

    valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw)
    cloudy = valid & (tcc >= min_cloud_fraction) & (
        (tclw_g > lwp_threshold_g) | (tciw > iwp_threshold_kg))
    liquid = cloudy & (tclw_g > lwp_threshold_g)

    # Counts (not means) across the longitude strip: every cell in the strip
    # sits at the same latitude, so each carries equal weight and a straight
    # sum is what the ratio needs -- averaging per-cell fractions instead
    # would be undefined wherever a cell has zero cloudy hours.
    cloudy_t = cloudy.sum(axis=2).astype(float)   # (time, lat)
    liquid_t = liquid.sum(axis=2).astype(float)   # (time, lat)
    ice_t = nanmean_quiet(band["siconc"].values, axis=2)   # (time, lat)

    # Land edge: northernmost latitude where ANY cell of the strip is land.
    land_any = np.isnan(band["siconc"].values).any(axis=(0, 2))
    land_edge = float(lat[land_any].max()) if land_any.any() else None

    order = np.argsort(lat)
    cloudy_t = cloudy_t[:, order]
    liquid_t = liquid_t[:, order]
    ice_t = ice_t[:, order]
    lat = lat[order]

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
            c_sum = cloudy_t[sel].sum(axis=0)
            l_sum = liquid_t[sel].sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                stack[s_i, slot] = np.where(c_sum > 0, 100.0 * l_sum / c_sum, np.nan)
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
    region: str,
    mode_label: str,
    lwp_threshold_g: float,
    iwp_threshold_kg: float,
    min_cloud_fraction: float,
    ice_levels: tuple[float, ...],
    contour_style: str = "contrast",
    output_path: Path | None = None,
    dpi: int = 150,
):
    import matplotlib.pyplot as plt

    # How many seasons actually contribute to each column. Seasons rarely cover
    # the same part of the window, so this varies along x -- and where it jumps,
    # the climatology jumps with it. That is the source of any vertical striping,
    # so it is measured and reported rather than left to be misread as signal.
    n_contrib = np.sum(~np.all(np.isnan(sec["field"]), axis=2), axis=0)

    field = nanmean_quiet(sec["field"], axis=0)   # mean of the per-season ratios
    ice = nanmean_quiet(sec["ice"], axis=0)
    lat, slots = sec["lat"], sec["slots"]

    # Drop slots with no data at all (e.g. 29 Feb in non-leap seasons).
    keep = ~np.all(np.isnan(field), axis=1)
    field, ice, n_contrib = field[keep], ice[keep], n_contrib[keep]
    kept_slots = [s for s, k in zip(slots, keep) if k]
    x = np.arange(len(kept_slots))

    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    mesh = ax.pcolormesh(x, lat, field.T, cmap=LIQUID_FRAC_CMAP, vmin=0, vmax=100,
                         shading="auto")

    if np.isfinite(ice).any():
        cs = ax.contour(x, lat, ice.T, levels=list(ice_levels), colors="black",
                        linewidths=[1.0 + 0.5 * i for i in range(len(ice_levels))],
                        linestyles=["--", "-", ":", "-."][: len(ice_levels)],
                        zorder=5)
        labels = ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=9, inline=True)

        if contour_style == "contrast":
            # A plain black line disappears against the dark-grey end of the
            # ramp. Outlining each line in white keeps it readable on any
            # background without giving the line a colour of its own, which
            # would compete with the data for meaning.
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
    cf_label = ("total cloud cover = 1" if min_cloud_fraction >= 1.0
                else f"total cloud cover $\\geq$ {min_cloud_fraction:g}")
    ax.set_title(
        f"{LIQUID_FRAC_LABEL} — {region}, meridional strip north of Utqiaġvik\n"
        f"{mode_label}   |   {lon_used.size}-cell longitude mean "
        f"({lon_used.min():.2f} to {lon_used.max():.2f}°E)\n"
        f"liquid = LWP > {lwp_threshold_g:g} g m$^{{-2}}$   |   {cf_label}   |   "
        f"IWP > {iwp_threshold_kg:g} kg m$^{{-2}}$ (cloudy test)   |   "
        f"contours: sea ice {', '.join(f'{v:g}' for v in ice_levels)}",
        fontsize=11)
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
    ax.grid(alpha=0.18, linewidth=0.5, color="0.5")
    ax.set_axisbelow(False)

    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=26)
    cb.set_label(f"{LIQUID_FRAC_LABEL} [{LIQUID_FRAC_UNITS}]", fontsize=10)
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
    parser.add_argument("--lwp-threshold", type=float,
                        default=DEFAULT_LWP_THRESHOLD_G, metavar="G",
                        help="LWP in g m-2 above which liquid counts as present "
                             f"(default {DEFAULT_LWP_THRESHOLD_G:g}). Used for both "
                             "the liquid test and the cloudy test, exactly as in "
                             "analyze_cloud_liquid_frequency.py.")
    parser.add_argument("--iwp-threshold", type=float,
                        default=DEFAULT_IWP_THRESHOLD_KG, metavar="F",
                        help="IWP in kg m-2 used in the cloudy test (default "
                             f"{DEFAULT_IWP_THRESHOLD_KG:g}).")
    parser.add_argument("--min-cloud-fraction", type=float,
                        default=DEFAULT_MIN_CLOUD_FRACTION, metavar="F",
                        help="Total cloud cover at or above which a scene counts "
                             f"as cloudy (default {DEFAULT_MIN_CLOUD_FRACTION:g}, "
                             "fully overcast).")
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
                        help="Sea ice contour levels (default 0.05,0.95).")
    parser.add_argument("--contour-style", choices=("black", "contrast"),
                        default="contrast",
                        help="How the sea ice contours are drawn. 'contrast' "
                             "(default) outlines the black lines in white so they "
                             "stay readable over the dark end of the colour ramp; "
                             "'black' is a plain unoutlined line.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6, metavar="F",
                        help="Exclude seasons covering less than this fraction of the "
                             "window from a climatology (default 0.6).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Latitude-time section of the cloudy-scene liquid fraction")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = sorted({"tcc", "tclw", "tciw", "siconc"} - set(ds.data_vars))
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
    print(f"  Cloudy scene: tcc >= {args.min_cloud_fraction:g} AND "
          f"(LWP > {args.lwp_threshold:g} g m-2 or IWP > {args.iwp_threshold:g} kg m-2)")
    print(f"  Liquid      : LWP > {args.lwp_threshold:g} g m-2")

    try:
        sec = build_section(ds, args.lon_center, args.n_lon_cells,
                            lat_south, lat_north,
                            args.season_start, args.season_end,
                            args.lwp_threshold, args.iwp_threshold,
                            args.min_cloud_fraction)
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
        absent = [y for y in args.years if y not in sec["seasons"]]
        if absent:
            print(f"\n  Error: no data for season(s) {absent}. "
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

    sec = dict(sec)
    sec["field"] = sec["field"][keep_idx]
    sec["ice"] = sec["ice"][keep_idx]

    mean_field = nanmean_quiet(sec["field"], axis=0)
    finite = mean_field[np.isfinite(mean_field)]
    print(f"\n  {LIQUID_FRAC_LABEL}: mean={finite.mean():.1f}%, "
          f"range {finite.min():.1f}% to {finite.max():.1f}%")

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    make_hovmoller(sec, args.region, mode_label, args.lwp_threshold,
                   args.iwp_threshold, args.min_cloud_fraction,
                   tuple(args.ice_levels), contour_style=args.contour_style,
                   output_path=out_dir / f"{args.region}_hovmoller_liquid_fraction_{tag}.png",
                   dpi=args.dpi)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
