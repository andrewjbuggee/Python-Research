#!/usr/bin/env python3
"""Sea ice concentration distribution against time of season.

Produces three views of the same underlying 2-D histogram -- sea ice
concentration on the y axis, time of season on the x axis -- differing only in
how each column is normalised:

    counts        raw ocean cell-times per bin
    cell_fraction fraction of the region's OCEAN GRID CELLS in each bin
    area_fraction fraction of the region's OCEAN AREA in each bin

The two fraction views each sum to 1.0 down every column, so a column reads
directly as "where was the region's ice concentration on this day", with nothing
unaccounted for. That is what the raw-count view cannot show: counts on a log
colour scale make a bin holding 1% of the region look comparable to one holding
50%, which is misleading in exactly the way that matters during freeze-up.

``area_fraction`` is the physically meaningful one. ERA5 is on a regular
latitude/longitude grid, so a cell at 70 N covers about twice the ground area of
one at 80 N (area scales as cos(latitude)); counting cells therefore
over-weights the northern, ice-covered end of this domain. Every cell is weighted
by cos(latitude) before the fraction is formed.

OPTIONS
=======

Data source
-----------
--storage {local,external}   Which disk to read (default local).
--data-root PATH             Explicit directory, overriding --storage.
--region NAME                Region subdirectory (default barrow).

Season
------
--season-start MM-DD         First day of the window (default 08-01).
--season-end MM-DD           Last day, inclusive (default 12-01).
                             The window MAY wrap the new year: 09-01 to 03-31 is
                             valid and is treated as one season labelled by the
                             year it starts in.
--years SPEC                 Restrict to these years, e.g. '2019' or '2019-2025'
                             or '2000,2019-2020'. For a wrapping window the year
                             is the one the season STARTS in, so --years 2019
                             with 09-01..03-31 means Sep 2019 to Mar 2020.
--group {hour,day}           Pool by hour of season (default) or by calendar day.

Histogram
---------
--bins N                     Concentration bins between 0 and 1 (default 20).
--views LIST                 Which figures to make: 'all' (default) or a comma
                             separated subset of counts,cell_fraction,area_fraction.
--log-counts / --no-log-counts
                             Log colour scale on the COUNTS figure (default on;
                             the distribution is strongly bimodal). The fraction
                             figures always use a linear 0..max scale, because
                             the whole point of them is that the numbers are
                             comparable.

Output
------
--output-dir PATH            Where figures go (default ./figures).
--dpi N                      Figure resolution (default 150).
--show                       Open interactive windows.

Examples
--------
    python plot_sea_ice_histogram.py --region barrow --years 2019

    python plot_sea_ice_histogram.py --region barrow \
        --season-start 09-01 --season-end 03-31 --years 2019-2025 --group day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from seb_analysis_common import (
    add_data_source_args,
    area_weights,
    hour_of,
    load_seb_data,
    resolve_region_dir,
    season_calendar,
    season_slot_of,
    season_year_of,
)

# Reference years for the x axis only. They must match season_calendar's choice
# so that whichever year holds February is a LEAP year: the slot list contains a
# 29 February entry, and building that date in a non-leap year raises
# "Day out of range". For a wrapping window the season runs 1999 -> 2000.
REF_YEAR_START = 2000        # non-wrapping window: one leap year covers it
REF_YEAR_WRAP = (1999, 2000)  # wrapping window: second half is the leap year

VIEWS = {
    "counts": ("Ocean cell-times per bin", "counts"),
    "cell_fraction": ("Fraction of ocean grid cells", "cell fraction"),
    "area_fraction": ("Fraction of ocean area", "area fraction"),
}


def parse_month_day(text: str) -> tuple[int, int]:
    from datetime import date
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(2000, month, day)  # leap year, so 02-29 validates
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid MM-DD") from None
    return month, day


def parse_years(text: str) -> tuple[int, ...]:
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


def parse_views(text: str) -> tuple[str, ...]:
    if text.strip() == "all":
        return tuple(VIEWS)
    picked = tuple(v.strip() for v in text.split(",") if v.strip())
    bad = [v for v in picked if v not in VIEWS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown view(s) {bad}; choose from {list(VIEWS)}"
        )
    return picked


def build_histograms(ds, season, years, group, n_bins):
    """Accumulate the three normalisations in one pass over the season.

    Returns positions (as datetime64 in a reference year), bin edges, a dict of
    view -> 2-D array shaped (bin, time), and bookkeeping.
    """
    times = ds["valid_time"].values
    wraps = season[1] < season[0]
    slots = season_calendar(*season)
    slot = season_slot_of(times, slots)
    season_yr = season_year_of(times, season[0])

    in_window = slot >= 0
    if years is not None:
        in_window &= np.isin(season_yr, list(years))
    idx = np.nonzero(in_window)[0]
    if idx.size == 0:
        raise ValueError("No time steps inside the requested season and years.")

    keys = slot[idx] * 100 + hour_of(times)[idx] if group == "hour" else slot[idx]
    uniq = np.unique(keys)

    siconc = ds["siconc"].values
    ocean_yx = ds["siconc"].notnull().any("valid_time").values
    # cos(latitude) weights: on a regular lat/lon grid a 70 N cell covers about
    # twice the ground area of an 80 N one, so counting cells over-weights the
    # northern (ice-covered) end of this domain.
    w_yx = np.broadcast_to(area_weights(ds).values[:, None], ocean_yx.shape)
    w_ocean = w_yx[ocean_yx]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_t = uniq.size
    counts = np.zeros((n_bins, n_t))
    area = np.zeros((n_bins, n_t))
    positions = np.empty(n_t, dtype="datetime64[m]")
    n_valid = np.zeros(n_t, dtype=int)

    for i, key in enumerate(uniq):
        sel = idx[keys == key]
        s_i = int(key // 100) if group == "hour" else int(key)
        hr = int(key % 100) if group == "hour" else 0
        mo, dy = slots[s_i]
        if wraps:
            # Advance the year at the wrap, or the axis would fold back on itself.
            yr = REF_YEAR_WRAP[1] if (mo, dy) < season[0] else REF_YEAR_WRAP[0]
        else:
            yr = REF_YEAR_START
        positions[i] = np.datetime64(f"{yr}-{mo:02d}-{dy:02d}T{hr:02d}:00")

        vals = siconc[sel][:, ocean_yx]                 # (n_times, n_ocean)
        wts = np.broadcast_to(w_ocean, vals.shape)
        good = np.isfinite(vals)
        if not good.any():
            continue
        counts[:, i] = np.histogram(vals[good], bins=edges)[0]
        area[:, i] = np.histogram(vals[good], bins=edges, weights=wts[good])[0]
        n_valid[i] = int(good.sum())

    # Normalise per column so each fraction view sums to 1 where data exists.
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_frac = counts / np.where(counts.sum(0) > 0, counts.sum(0), np.nan)
        area_frac = area / np.where(area.sum(0) > 0, area.sum(0), np.nan)

    return {
        "positions": positions,
        "edges": edges,
        "views": {"counts": counts,
                  "cell_fraction": cell_frac,
                  "area_fraction": area_frac},
        "n_valid": n_valid,
        "seasons": sorted(set(season_yr[idx].tolist())),
        "slots": slots,
    }


def make_histogram_figure(hist, view, region, years_label, season_label,
                          log_counts, output_path, dpi):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    field = hist["views"][view]
    edges = hist["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    pos = hist["positions"].astype("datetime64[m]").astype("O")
    label, short = VIEWS[view]

    shown = np.where(np.isfinite(field) & (field > 0), field, np.nan)
    norm = None
    if view == "counts" and log_counts and np.isfinite(shown).any():
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(1.0, np.nanmin(shown)), vmax=np.nanmax(shown))

    fig, ax = plt.subplots(figsize=(13.5, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(pos, centers, shown, cmap="viridis", norm=norm,
                         shading="auto")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sea ice concentration", fontsize=10)
    ax.set_xlabel(f"Time of season ({season_label})", fontsize=10)
    ax.set_title(f"Sea ice concentration distribution — {region}\n"
                 f"{years_label}   |   {label}", fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.2, linewidth=0.5, color="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=26)
    cb.set_label(label + (" (log scale)" if norm is not None else ""), fontsize=10)

    if view != "counts":
        colsum = np.nansum(field, axis=0)
        colsum = colsum[colsum > 0]
        if colsum.size:
            ax.text(0.995, -0.20,
                    f"each column sums to {colsum.min():.3f}–{colsum.max():.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="0.4")

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"    {short:<14} -> {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD")
    parser.add_argument("--season-end", type=parse_month_day, default=(12, 1),
                        metavar="MM-DD",
                        help="Inclusive. May wrap the new year (e.g. 03-31).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC")
    parser.add_argument("--group", choices=("hour", "day"), default="hour")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--views", type=parse_views, default=tuple(VIEWS))
    parser.add_argument("--log-counts", dest="log_counts", action="store_true",
                        default=True)
    parser.add_argument("--no-log-counts", dest="log_counts", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 72)
    print("Sea ice concentration histograms")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    if "siconc" not in ds:
        print("  Error: dataset has no 'siconc'.", file=sys.stderr)
        return 1

    (m0, d0), (m1, d1) = args.season_start, args.season_end
    wraps = (m1, d1) < (m0, d0)
    print(f"  Source     : {region_dir}")
    print(f"  Season     : {m0:02d}-{d0:02d} to {m1:02d}-{d1:02d}"
          + ("  (wraps the new year)" if wraps else ""))
    if args.years:
        print(f"  Years      : {list(args.years)}"
              + ("  (year the season STARTS in)" if wraps else ""))

    ds = ds[["siconc"]].compute()

    try:
        hist = build_histograms(ds, (args.season_start, args.season_end),
                                args.years, args.group, args.bins)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    n_cols = hist["positions"].size
    print(f"  Seasons    : {hist['seasons']}")
    print(f"  Columns    : {n_cols} ({args.group}ly slots), "
          f"{int((hist['n_valid'] == 0).sum())} empty")
    print(f"  Bins       : {args.bins} between 0 and 1")

    # A quick check that the fraction views really do account for everything.
    for v in ("cell_fraction", "area_fraction"):
        col = np.nansum(hist["views"][v], axis=0)
        col = col[col > 0]
        if col.size:
            print(f"  {v:<14} column sums {col.min():.4f} to {col.max():.4f}")

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    yrs = hist["seasons"]
    years_label = (f"{len(yrs)} seasons: {yrs[0]}–{yrs[-1]}" if len(yrs) > 1
                   else f"season {yrs[0]}" + (f"/{yrs[0]+1}" if wraps else ""))
    season_label = (f"{'hourly' if args.group == 'hour' else 'daily'} steps, "
                    f"{m0:02d}-{d0:02d} to {m1:02d}-{d1:02d}")

    print()
    for view in args.views:
        make_histogram_figure(
            hist, view, args.region, years_label, season_label,
            log_counts=args.log_counts,
            output_path=out_dir / f"{args.region}_sea_ice_hist_{view}.png",
            dpi=args.dpi,
        )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
