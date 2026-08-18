#!/usr/bin/env python3
"""Seasonal time series of a variable averaged within five surface classes.

Two stacked panels, sharing a day-of-season x axis that runs 08-01 to 03-31 by
default:

    top     the variable of interest, one line per surface class
    bottom  the % of the domain's area occupied by each class, same colours

The classification (land / coastal / open ocean / marginal ice zone / sea ice)
is done per grid cell per time step by ``surface_classification.py``; see that
module for the thresholds and for why land is detected from ``lsm`` rather than
from ``siconc == 0``.

Why the bottom panel is not decoration
======================================
The ice edge moves several degrees between years, so a class's line in the top
panel is a mean over a footprint that is itself changing size. Late in the
season "open ocean" may be 1% of the domain, and its mean is then a noisy
average over a handful of cells that says little about the region. The bottom
panel is what tells you when to believe the top one. For the same reason a
class's line is BLANKED wherever it holds less than ``--min-class-area`` percent
of the domain, rather than drawn as a spike.

Quantities (``--variable``, default ``all``, one figure each)
============================================================
    dlr               downwelling longwave, msdwlwrf              [W m-2]
    lwp               liquid water path, tclw                     [g m-2]
    iwp               ice water path, tciw                        [g m-2]
    liquid_fraction   cloudy scenes containing liquid             [%]

``lwp`` and ``iwp`` zero any hour at or below ``--trace-threshold`` before
averaging, the same guard against ERA5's near-zero trace condensate that
``plot_monthly_lwp_maps.py`` applies before its median.

``liquid_fraction`` uses the definition from
``plot_latitude_time_hovmoller_liquid_fraction.py``::

    cloudy = tcc >= --min-cloud-fraction AND (LWP > --lwp-threshold
                                              OR IWP > --iwp-threshold)
    liquid = cloudy AND LWP > --lwp-threshold
    fraction = area-weighted cloudy cell-hours with liquid / cloudy cell-hours

Note the two LWP thresholds are different knobs: ``--lwp-threshold`` (5 g m-2)
decides whether liquid is PRESENT, while ``--trace-threshold`` (0.03 g m-2) only
suppresses trace values before an LWP magnitude is averaged.

How the reduction works
=======================
Means are area-weighted by cos(latitude) and pooled over every cell-hour in a
day-of-season slot, then averaged across seasons with equal weight per season.
Pooling rather than averaging per-hour means keeps a slot with partial coverage
honest.

The mean, not the median, is used: it composes correctly across the streaming
blocks this script reads in, which a median does not. For LWP that matters --
the distribution is strongly skewed, so these means sit above the medians that
``plot_monthly_lwp_maps.py`` reports. The two are not directly comparable.

Examples
========
  # all four figures, every season with adequate coverage
  ./plot_surface_class_timeseries.py --region barrow

  # one season, one variable
  ./plot_surface_class_timeseries.py --region barrow --years 2021 --variable dlr

  # a 7-season climatology of the liquid fraction
  ./plot_surface_class_timeseries.py --region barrow --years 2019-2025 \
      --variable liquid_fraction
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
    season_calendar,
    season_slot_of,
    season_year_of,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    DEFAULT_BLOCK_HOURS,
    add_classification_args,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

# quantity -> (label, units, kind). "mean" quantities average a field; "ratio"
# quantities divide two weighted counts.
QUANTITIES: dict[str, tuple[str, str, str]] = {
    "dlr": ("Downwelling longwave", "W m$^{-2}$", "mean"),
    "lwp": ("Liquid water path", "g m$^{-2}$", "mean"),
    "iwp": ("Ice water path", "g m$^{-2}$", "mean"),
    "liquid_fraction": ("Cloudy scenes with liquid", "%", "ratio"),
}

# ERA5 source variable and unit scaling for each "mean" quantity.
MEAN_SOURCES: dict[str, tuple[str, float]] = {
    "dlr": ("msdwlwrf", 1.0),
    "lwp": ("tclw", 1000.0),   # kg m-2 -> g m-2
    "iwp": ("tciw", 1000.0),   # kg m-2 -> g m-2
}

# Matches plot_latitude_time_hovmoller_liquid_fraction.py.
DEFAULT_LWP_THRESHOLD_G = 5.0
DEFAULT_IWP_THRESHOLD_KG = 0.0
DEFAULT_MIN_CLOUD_FRACTION = 1.0

# Matches plot_monthly_lwp_maps.py's trace guard.
DEFAULT_TRACE_THRESHOLD_G = 0.03

# A class holding less than this share of the domain has its line blanked.
DEFAULT_MIN_CLASS_AREA_PCT = 0.5

REQUIRED_VARS = ("msdwlwrf", "tclw", "tciw", "tcc", "siconc")


def nanmean_quiet(a, axis=None):
    """np.nanmean without the "Mean of empty slice" RuntimeWarning.

    All-NaN slices are expected here, not a mistake: a class can be absent from
    the domain for a whole slot, and a season may not cover the full window.
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


def month_ticks(slots: list[tuple[int, int]]) -> tuple[list[int], list[str]]:
    """Tick at the first of each month present, labelled by month abbreviation."""
    pos, lab = [], []
    for i, (m, d) in enumerate(slots):
        if d == 1:
            pos.append(i)
            lab.append(calendar.month_abbr[m])
    return pos, lab


# ----------------------------------------------------------------------------
# Reduction
# ----------------------------------------------------------------------------
def season_layout(ds, args) -> dict:
    """Map every time step to a season and a day-of-season slot.

    Derived from the time coordinate ALONE, so it is cheap and, critically,
    available before any field is read. Season selection therefore happens
    first and only the chosen seasons are ever loaded.
    """
    times = ds["valid_time"].values
    slots = season_calendar(args.season_start, args.season_end)
    dos = season_slot_of(times, slots)
    seasons = season_year_of(times, args.season_start)

    in_window = dos >= 0
    if not in_window.any():
        raise ValueError("No time steps fall inside the requested season window.")
    uniq_seasons = sorted(set(seasons[in_window].tolist()))

    # Coverage of the window per season, again from the time axis only.
    counts = np.zeros((len(uniq_seasons), len(slots)), dtype=int)
    season_index = {s: i for i, s in enumerate(uniq_seasons)}
    s_idx = np.array([season_index.get(int(s), -1) for s in seasons])
    sel = in_window & (s_idx >= 0)
    np.add.at(counts, (s_idx[sel], dos[sel]), 1)

    return {
        "slots": slots, "dos": dos, "s_idx": s_idx,
        "in_window": in_window, "seasons": uniq_seasons, "counts": counts,
    }


def build_series(ds, lsm: np.ndarray, args, layout: dict,
                 wanted_idx: list[int]) -> dict:
    """Reduce the chosen seasons to (season, day-of-season, class) arrays.

    Every quantity is accumulated in ONE streaming pass, because the four
    figures share the same classification and the same masking; reading the
    archive four times to produce them would be the dominant cost.

    Only seasons in ``wanted_idx`` are read. Their slots keep their position in
    the full season list so the caller's indexing stays valid.
    """
    slots = layout["slots"]
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]

    wanted = np.zeros(len(uniq_seasons), dtype=bool)
    wanted[wanted_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    n_season, n_slot, n_class = len(uniq_seasons), len(slots), len(CLASS_ORDER)
    shape = (n_season, n_slot, n_class)

    w_class = np.zeros(shape)          # area weight per class
    w_domain = np.zeros((n_season, n_slot))   # total domain weight, all cells
    vw = {q: np.zeros(shape) for q in MEAN_SOURCES}       # sum(w * value)
    vw_w = {q: np.zeros(shape) for q in MEAN_SOURCES}     # sum(w) over finite values
    cloudy_w = np.zeros(shape)
    liquid_w = np.zeros(shape)

    lat_deg = ds["latitude"].values
    weights = area_weights_2d(lat_deg, ds.sizes["longitude"])
    w_domain_per_step = float(weights.sum())

    n_unclassified = 0

    for i0, block in iter_time_blocks(ds, list(REQUIRED_VARS), args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_t)
        keep = use_step[sl]

        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )
        n_unclassified += int((classes == -1).sum())

        tclw_g = block["tclw"].values * 1000.0
        tciw_g = block["tciw"].values * 1000.0
        tcc = block["tcc"].values

        # Trace guard for the magnitude quantities only; the liquid test below
        # uses its own, much higher, threshold.
        values = {
            "dlr": block["msdwlwrf"].values,
            "lwp": np.where(tclw_g > args.trace_threshold, tclw_g, 0.0),
            "iwp": np.where(tciw_g > args.trace_threshold, tciw_g, 0.0),
        }

        valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
        cloudy = valid & (tcc >= args.min_cloud_fraction) & (
            (tclw_g > args.lwp_threshold)
            | (tciw_g / 1000.0 > args.iwp_threshold)
        )
        liquid = cloudy & (tclw_g > args.lwp_threshold)

        si = s_idx[sl][keep]
        di = dos[sl][keep]
        np.add.at(w_domain, (si, di), w_domain_per_step)

        for name in CLASS_ORDER:
            code = CLASS_CODES[name]
            wc = (classes == code) * weights          # (n_t, lat, lon)
            np.add.at(w_class, (si, di, code), wc.sum(axis=(1, 2))[keep])
            np.add.at(cloudy_w, (si, di, code),
                      (wc * cloudy).sum(axis=(1, 2))[keep])
            np.add.at(liquid_w, (si, di, code),
                      (wc * liquid).sum(axis=(1, 2))[keep])
            for q, field in values.items():
                finite = np.isfinite(field)
                np.add.at(vw[q], (si, di, code),
                          (wc * finite * np.nan_to_num(field)).sum(axis=(1, 2))[keep])
                np.add.at(vw_w[q], (si, di, code),
                          (wc * finite).sum(axis=(1, 2))[keep])

    # --- per-season slot values ---------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        area_pct = np.where(w_domain[..., None] > 0,
                            100.0 * w_class / w_domain[..., None], np.nan)
        series = {
            q: np.where(vw_w[q] > 0, vw[q] / np.where(vw_w[q] > 0, vw_w[q], 1.0),
                        np.nan)
            for q in MEAN_SOURCES
        }
        series["liquid_fraction"] = np.where(
            cloudy_w > 0, 100.0 * liquid_w / np.where(cloudy_w > 0, cloudy_w, 1.0),
            np.nan,
        )

    return {
        "series": series,
        "area_pct": area_pct,
        "slots": slots,
        "seasons": uniq_seasons,
        "n_unclassified": n_unclassified,
    }


def collapse_seasons(sec: dict, keep_idx: list[int]) -> dict:
    """Average the kept seasons, equal weight each, into (slot, class) arrays."""
    out = {"slots": sec["slots"]}
    out["area_pct"] = nanmean_quiet(sec["area_pct"][keep_idx], axis=0)
    out["series"] = {
        q: nanmean_quiet(v[keep_idx], axis=0) for q, v in sec["series"].items()
    }
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_figure(
    col: dict,
    quantity: str,
    region: str,
    mode_label: str,
    args,
    output_path: Path,
    dpi: int,
):
    """Draw the two-panel figure for one quantity and save it."""
    import matplotlib.pyplot as plt

    label, units, _kind = QUANTITIES[quantity]
    slots = col["slots"]
    x = np.arange(len(slots))
    field = col["series"][quantity]      # (slot, class)
    area = col["area_pct"]               # (slot, class)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )

    # --- top: the variable, one line per class ------------------------------
    for name in CLASS_ORDER:
        code = CLASS_CODES[name]
        y = field[:, code].copy()
        # Blank the line where the class is too small for its mean to mean
        # anything -- see the module docstring.
        y[area[:, code] < args.min_class_area] = np.nan
        if np.all(np.isnan(y)):
            continue
        ax_top.plot(x, y, color=CLASS_COLORS[name], linewidth=1.8,
                    label=CLASS_LABELS[name], solid_capstyle="round")

    ax_top.set_ylabel(f"{label} [{units}]")
    ax_top.grid(alpha=0.25, linewidth=0.6)
    ax_top.legend(loc="upper right", ncol=len(CLASS_ORDER), frameon=True,
                  framealpha=0.9, fontsize=9, columnspacing=1.1,
                  handlelength=1.6)

    # --- bottom: area share --------------------------------------------------
    stack = np.nan_to_num(area.T)        # (class, slot)
    if args.area_style == "stacked":
        ax_bot.stackplot(
            x, *stack,
            colors=[CLASS_COLORS[n] for n in CLASS_ORDER],
            labels=[CLASS_LABELS[n] for n in CLASS_ORDER],
            edgecolor="none",
        )
        ax_bot.set_ylim(0, 100)
    else:
        for name in CLASS_ORDER:
            ax_bot.plot(x, area[:, CLASS_CODES[name]], color=CLASS_COLORS[name],
                        linewidth=1.6, label=CLASS_LABELS[name])
        ax_bot.set_ylim(0, None)
    ax_bot.set_ylabel("Area of region [%]")
    ax_bot.set_xlabel(
        f"Day of season ({args.season_start[0]:02d}-{args.season_start[1]:02d} "
        f"to {args.season_end[0]:02d}-{args.season_end[1]:02d})"
    )
    ax_bot.grid(alpha=0.25, linewidth=0.6, color="white" if
                args.area_style == "stacked" else "grey")

    pos, lab = month_ticks(slots)
    ax_bot.set_xticks(pos)
    ax_bot.set_xticklabels(lab)
    ax_bot.set_xlim(0, len(slots) - 1)

    blanked = ""
    if args.min_class_area > 0:
        blanked = (f"   |   line blanked below {args.min_class_area:g}% of area")
    if quantity == "liquid_fraction":
        detail = (f"liquid = LWP > {args.lwp_threshold:g} g m$^{{-2}}$   |   "
                  f"cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}")
    else:
        detail = (f"area-weighted mean   |   trace "
                  f"$\\leq$ {args.trace_threshold:g} g m$^{{-2}}$ zeroed"
                  if quantity in ("lwp", "iwp") else "area-weighted mean")

    fig.suptitle(
        f"{label} by surface class — {region}\n"
        f"{mode_label}   |   {detail}{blanked}",
        fontsize=13, y=0.965,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.075, right=0.985)
    fig.savefig(output_path, dpi=dpi)
    print(f"  Wrote {output_path}")
    return fig


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    parser.add_argument("--variable", nargs="+",
                        choices=sorted(QUANTITIES) + ["all"], default=["all"],
                        help="Which quantity to plot, one figure each "
                             "(default: all four).")
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD", help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 03-31, wrapping the "
                             "year).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to average, by the year each season STARTS "
                             "in: '2019', '2019-2025', or '2000,2019-2020'. "
                             "Default: every season meeting --min-season-coverage.")
    parser.add_argument("--lwp-threshold", type=float,
                        default=DEFAULT_LWP_THRESHOLD_G, metavar="G",
                        help="LWP in g m-2 above which liquid counts as PRESENT, "
                             f"for the liquid_fraction panel (default "
                             f"{DEFAULT_LWP_THRESHOLD_G:g}). Matches "
                             "plot_latitude_time_hovmoller_liquid_fraction.py.")
    parser.add_argument("--iwp-threshold", type=float,
                        default=DEFAULT_IWP_THRESHOLD_KG, metavar="F",
                        help="IWP in kg m-2 used in the cloudy test (default "
                             f"{DEFAULT_IWP_THRESHOLD_KG:g}).")
    parser.add_argument("--min-cloud-fraction", type=float,
                        default=DEFAULT_MIN_CLOUD_FRACTION, metavar="F",
                        help="Total cloud cover at or above which a scene counts "
                             f"as cloudy (default {DEFAULT_MIN_CLOUD_FRACTION:g}, "
                             "fully overcast).")
    parser.add_argument("--trace-threshold", type=float,
                        default=DEFAULT_TRACE_THRESHOLD_G, metavar="G",
                        help="LWP/IWP in g m-2 at or below which an hour is "
                             "treated as zero before the MAGNITUDE mean (default "
                             f"{DEFAULT_TRACE_THRESHOLD_G:g}). Separate from "
                             "--lwp-threshold; matches plot_monthly_lwp_maps.py.")
    parser.add_argument("--min-class-area", type=float,
                        default=DEFAULT_MIN_CLASS_AREA_PCT, metavar="PCT",
                        help="Blank a class's line in the top panel wherever it "
                             "holds less than this % of the domain area (default "
                             f"{DEFAULT_MIN_CLASS_AREA_PCT:g}). Set 0 to draw "
                             "everything.")
    parser.add_argument("--area-style", choices=("stacked", "lines"),
                        default="stacked",
                        help="Bottom panel as a stacked area (default) or as "
                             "one line per class.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6,
                        metavar="F",
                        help="Exclude seasons covering less than this fraction of "
                             "the window from a climatology (default 0.6).")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    quantities = (sorted(QUANTITIES) if "all" in args.variable
                  else list(dict.fromkeys(args.variable)))

    print("=" * 72)
    print("Surface-class time series")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
        lsm_da = load_land_sea_mask(
            args.region, resolve_data_root(args.storage, args.data_root),
            args.mask_grid,
        )
        lsm = align_lsm_to_grid(lsm_da, ds)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = sorted(set(REQUIRED_VARS) - set(ds.data_vars))
    if missing:
        print(f"  Error: dataset is missing {missing}. Re-download with "
              f"--var-set recommended or extended.", file=sys.stderr)
        return 1

    print(f"  Source     : {region_dir}")
    print(f"  Grid       : {ds.sizes['latitude']} x {ds.sizes['longitude']} cells, "
          f"{ds.sizes['valid_time']:,} time steps")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
          f"{args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)" if args.season_end < args.season_start else ""))
    print(f"  Quantities : {', '.join(quantities)}")
    print(f"  Classes    : lsm tol {args.lsm_tol:g} | open ocean < "
          f"{args.open_ocean_max_siconc:g} | pack ice > {args.sea_ice_min_siconc:g}")

    try:
        sec = build_series(ds, lsm, args)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    if sec["n_unclassified"]:
        print(f"  !! {sec['n_unclassified']:,} unclassified cell-times; run "
              f"surface_classification.py for the breakdown.", file=sys.stderr)

    n_slot = len(sec["slots"])
    print(f"\n  Seasons found ({len(sec['seasons'])}), coverage of the "
          f"{n_slot}-day window:")
    frac = {}
    for s_i, s in enumerate(sec["seasons"]):
        f = float((sec["counts"][s_i] > 0).sum()) / n_slot
        frac[s] = f
        print(f"    {s}/{s+1}: {f*100:5.1f}%"
              + ("" if f >= args.min_season_coverage else "   (below --min-season-coverage)"))

    if args.years is not None:
        absent = [y for y in args.years if y not in sec["seasons"]]
        if absent:
            print(f"\n  Error: no data for season(s) {absent}. "
                  f"Available: {sec['seasons']}", file=sys.stderr)
            return 1
        # An explicit request is honoured as given: --min-season-coverage only
        # guards the automatic default.
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
    mode_label = (f"season {used[0]}/{used[0]+1}" if len(used) == 1
                  else f"mean of {len(used)} seasons: "
                       + ", ".join(f"{u}/{u+1}" for u in used))
    print(f"\n  Using {len(used)} season(s): {used}")

    col = collapse_seasons(sec, keep_idx)

    print("\n  Mean area share over the season window:")
    for name in CLASS_ORDER:
        share = nanmean_quiet(col["area_pct"][:, CLASS_CODES[name]])
        print(f"    {CLASS_LABELS[name]:<20}{share:6.2f}%")

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    print()
    for quantity in quantities:
        make_figure(
            col, quantity, args.region, mode_label, args,
            out_dir / f"{args.region}_surfaceclass_{quantity}_{tag}.png",
            args.dpi,
        )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
