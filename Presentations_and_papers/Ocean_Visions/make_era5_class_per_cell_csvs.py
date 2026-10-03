#!/usr/bin/env python3
"""Per-grid-cell hours and time fractions of cloud phase, by surface class.

WHAT IT WRITES
==============
For each of the five surface classes (Land, Coastal, OpenOcean,
MarginalSeaIce, PackIce) and each cell-hour table that
``make_era5_hour_count_csvs.py`` writes for them (per season, per month x 3
cloud types, per day), two normalised twins:

    ERA5_avgHoursPerCell_<resolution>_2014_2025_[<type>_]<CLASS>.csv
        hours of the cloud type at an AVERAGE grid cell of the class
    ERA5_fractionOfTime_<resolution>_2014_2025_[<type>_]<CLASS>.csv
        the same as a fraction of the hours the class existed

plus ``ERA5_hoursClassPresent_*`` (the hours each class existed at all) and
a readME, all in ``era5_hour_count_csv/class_normalized/``. UTQ is a single
cell and needs no normalisation, so it is not included.

THE TWO AVERAGES (choose with --method)
======================================
Let, for class c and hour t,

    N(t)  = valid grid cells in the class at hour t
    n(t)  = those cells that also pass the filters and hold the cloud type
    f(t)  = n(t) / N(t)        the class's occupancy in that hour, 0..1

and let T be the hours of the period (a day, month, season) in which the
class existed, N(t) > 0. Hours with N(t) = 0 contribute nothing.

``per_hour`` (default -- divide by the cell count at EACH time step):

    avg hours per cell = sum over t of f(t)
    fraction of time   = (1/T) * sum over t of f(t)

  Every hour counts equally, however many cells the class had. For a class
  whose cells never change (Land, Coastal) this is exactly total cell-hours
  divided by the number of cells.

``pooled`` (pool first, divide once -- the project's convention,
``plot_class_liquid_fractions.py`` and ``season_phase_hours``):

    fraction of time   = (sum over t of n(t)) / (sum over t of N(t))
    avg hours per cell = fraction * T

  Every CELL-HOUR counts equally, so an hour with 400 marginal-ice cells
  weighs 200 times one with 2. This is the count table divided by the
  ``normalization/`` table.

The two are identical when N(t) is constant, and differ when the class's size
and its occupancy co-vary -- e.g. a marginal ice zone that is large and
cloudy during freeze-up. ``--compare`` prints both for every class and
season so the size of the difference is measured rather than assumed.

Neither average is area-weighted: a grid cell is a grid cell (the notebook's
class figures weight by cos(latitude) instead).

Usage
-----
    python make_era5_class_per_cell_csvs.py                 # per_hour, ~2.5 min
    python make_era5_class_per_cell_csvs.py --method pooled
    python make_era5_class_per_cell_csvs.py --compare       # also print both
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The counting pass, the figure settings and the table conventions all come
# from the cell-hour script, so these files share its population exactly.
from make_era5_hour_count_csvs import (
    CATS,
    DEFAULT_OUT_DIR,
    FILE_TAG,
    MONTH_ABBR,
    PHASE_CATS,
    SEASON_MONTHS,
    SETTINGS,
    SLOT_NAMES,
    build_args,
    count_cell_hours,
    git_describe,
    season_label,
)

METHODS = ("per_hour", "pooled")
DEFAULT_METHOD = "per_hour"

# The five classes only: slot 0 is UTQ, one fixed cell.
CLASS_SLOTS: tuple[int, ...] = tuple(range(1, len(SLOT_NAMES)))
I_ALL = CATS.index("all_valid")


# ----------------------------------------------------------------------------
# Per-day sums: everything else is a sum of these
# ----------------------------------------------------------------------------
def daily_sums(C: dict) -> dict:
    """Additive per-day pieces of both averages, per class slot and category.

    Returns arrays shaped ``(n_day, n_slot, n_phase)`` (``sum_f``, ``sum_n``)
    and ``(n_day, n_slot)`` (``hours_present``, ``sum_N``). Because each is a
    plain sum over hours, the monthly and seasonal values are sums of the
    daily ones, and both methods can be finished from them at any resolution.
    """
    H = C["hourly"]                                   # (hour, slot, cat) int32
    N = H[:, :, I_ALL].astype(float)                  # (hour, slot) class size
    n = H[:, :, [CATS.index(c) for c in PHASE_CATS]].astype(float)
    present = N > 0                                   # (hour, slot)
    with np.errstate(divide="ignore", invalid="ignore"):
        # f(t) = n(t)/N(t); 0 where the class did not exist (it then adds
        # nothing to either sum, and T does not count that hour).
        f = np.where(present[..., None], n / np.where(present, N, 1.0)[..., None], 0.0)
    if (f > 1.0 + 1e-12).any():
        raise AssertionError("an hourly occupancy exceeds 1")

    day = C["hour_day"]
    n_day = C["dates"].size

    def by_day(a: np.ndarray) -> np.ndarray:
        out = np.zeros((n_day, *a.shape[1:]))
        np.add.at(out, day, a)
        return out

    return {"sum_f": by_day(f), "sum_n": by_day(n), "sum_N": by_day(N),
            "hours_present": by_day(present.astype(float))}


def finish(D: dict, sel: np.ndarray, slot: int, method: str) -> tuple:
    """(avg hours per cell (3,), fraction (3,), hours present) over days ``sel``."""
    T = D["hours_present"][sel, slot].sum()
    if T == 0:
        return np.zeros(len(PHASE_CATS)), np.full(len(PHASE_CATS), np.nan), 0.0
    if method == "per_hour":
        avg = D["sum_f"][sel, slot].sum(axis=0)
        return avg, avg / T, T
    frac = D["sum_n"][sel, slot].sum(axis=0) / D["sum_N"][sel, slot].sum()
    return frac * T, frac, T


# ----------------------------------------------------------------------------
# Tables, in the same layouts as the cell-hour files
# ----------------------------------------------------------------------------
def season_tables(C, D, slot, method):
    """Rows = seasons, columns = the three types; (avg, fraction)."""
    avg_rows, frac_rows = [], []
    for y in C["seasons"]:
        a, f, _T = finish(D, C["day_season"] == y, slot, method)
        avg_rows.append([season_label(y), *a])
        frac_rows.append([season_label(y), *f])
    cols = ["season", *PHASE_CATS]
    return pd.DataFrame(avg_rows, columns=cols), pd.DataFrame(frac_rows, columns=cols)


def month_tables(C, D, slot, method):
    """{type: (avg, fraction)}, rows = Oct..Mar, columns = seasons."""
    out = {}
    for k, cat in enumerate(PHASE_CATS):
        avg = {"month": [MONTH_ABBR[m].rstrip(".") for m in SEASON_MONTHS]}
        frac = dict(avg)
        for y in C["seasons"]:
            a_col, f_col = [], []
            for m in SEASON_MONTHS:
                sel = (C["day_season"] == y) & (C["day_month"] == m)
                a, f, _T = finish(D, sel, slot, method)
                a_col.append(a[k])
                f_col.append(f[k])
            avg[season_label(y)] = a_col
            frac[season_label(y)] = f_col
        out[cat] = (pd.DataFrame(avg), pd.DataFrame(frac))
    return out


def day_tables(C, D, slot, method):
    """Rows = days, first column the date; (avg, fraction)."""
    dt = pd.to_datetime(C["dates"])
    labels = [f"{d.day} {MONTH_ABBR[d.month]} {d.year}" for d in dt]
    T = D["hours_present"][:, slot]                          # (day,)
    if method == "per_hour":
        avg = D["sum_f"][:, slot]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(T[:, None] > 0, avg / np.where(T > 0, T, 1.0)[:, None], np.nan)
    else:
        sN = D["sum_N"][:, slot]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(sN[:, None] > 0,
                            D["sum_n"][:, slot] / np.where(sN > 0, sN, 1.0)[:, None], np.nan)
        avg = np.nan_to_num(frac) * T[:, None]
    a_df, f_df = pd.DataFrame({"date": labels}), pd.DataFrame({"date": labels})
    for k, c in enumerate(PHASE_CATS):
        a_df[c], f_df[c] = avg[:, k], frac[:, k]
    return a_df, f_df


def presence_tables(C, D):
    """Hours each class existed: per season (one file), month (per class), day."""
    hp = D["hours_present"]
    season = pd.DataFrame({"season": [season_label(y) for y in C["seasons"]]})
    for s in CLASS_SLOTS:
        season[SLOT_NAMES[s]] = [hp[C["day_season"] == y, s].sum() for y in C["seasons"]]
    month = {}
    for s in CLASS_SLOTS:
        m = {"month": [MONTH_ABBR[mm].rstrip(".") for mm in SEASON_MONTHS]}
        for y in C["seasons"]:
            m[season_label(y)] = [hp[(C["day_season"] == y) & (C["day_month"] == mm), s].sum()
                                  for mm in SEASON_MONTHS]
        month[SLOT_NAMES[s]] = pd.DataFrame(m)
    dt = pd.to_datetime(C["dates"])
    day = pd.DataFrame({"date": [f"{d.day} {MONTH_ABBR[d.month]} {d.year}" for d in dt]})
    for s in CLASS_SLOTS:
        day[SLOT_NAMES[s]] = hp[:, s]
    return season, month, day


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
def write_all(C: dict, D: dict, method: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    span = f"{C['seasons'][0]}_{C['seasons'][-1] + 1}"
    written: list[Path] = []

    def save(df: pd.DataFrame, name: str, fmt: str) -> None:
        path = out_dir / name
        df.to_csv(path, index=False, float_format=fmt)
        written.append(path)

    # Hours to 4 decimals (a few seconds); fractions to 6.
    for s in CLASS_SLOTS:
        cls = SLOT_NAMES[s]
        a, f = season_tables(C, D, s, method)
        save(a, f"ERA5_avgHoursPerCell_perSeason_{span}_{cls}.csv", "%.4f")
        save(f, f"ERA5_fractionOfTime_perSeason_{span}_{cls}.csv", "%.6f")
        for cat, (a, f) in month_tables(C, D, s, method).items():
            save(a, f"ERA5_avgHoursPerCell_perMonth_{span}_{FILE_TAG[cat]}_{cls}.csv", "%.4f")
            save(f, f"ERA5_fractionOfTime_perMonth_{span}_{FILE_TAG[cat]}_{cls}.csv", "%.6f")
        a, f = day_tables(C, D, s, method)
        save(a, f"ERA5_avgHoursPerCell_perDay_{span}_{cls}.csv", "%.4f")
        save(f, f"ERA5_fractionOfTime_perDay_{span}_{cls}.csv", "%.6f")

    season, month, day = presence_tables(C, D)
    save(season, f"ERA5_hoursClassPresent_perSeason_{span}.csv", "%.0f")
    for cls, df in month.items():
        save(df, f"ERA5_hoursClassPresent_perMonth_{span}_{cls}.csv", "%.0f")
    save(day, f"ERA5_hoursClassPresent_perDay_{span}.csv", "%.0f")
    return written


# ----------------------------------------------------------------------------
# Checks and the method comparison
# ----------------------------------------------------------------------------
def check(C: dict, D: dict) -> None:
    """Hard checks that must hold for any data."""
    H = C["hourly"]
    for s in CLASS_SLOTS:
        N = H[:, s, I_ALL]
        if N.max() == N[N > 0].min():
            # Fixed-size class: per_hour avg must equal total / N exactly.
            tot = H[:, s, [CATS.index(c) for c in PHASE_CATS]].sum(axis=0)
            got = D["sum_f"][:, s].sum(axis=0)
            if not np.allclose(got, tot / N.max(), rtol=0, atol=1e-6):
                raise AssertionError(f"{SLOT_NAMES[s]}: per-hour avg != total / N")
    # Per-hour and pooled fractions are both 0..1; seasonal avg <= T.
    for method in METHODS:
        for s in CLASS_SLOTS:
            for y in C["seasons"]:
                a, f, T = finish(D, C["day_season"] == y, s, method)
                if T and not ((f >= 0) & (f <= 1) & (a <= T + 1e-9)).all():
                    raise AssertionError(f"{method} {SLOT_NAMES[s]} {y}: out of range")


def print_class_sizes(C: dict) -> None:
    """How many cells each class holds per hour -- the noise in f(t)."""
    H = C["hourly"]
    print(f"\n  cells per hour (valid)   {'min':>6s} {'median':>7s} {'max':>6s} "
          f"{'hours absent':>13s} {'hours < 10 cells':>17s}")
    for s in CLASS_SLOTS:
        N = H[:, s, I_ALL]
        pos = N[N > 0]
        print(f"  {SLOT_NAMES[s]:<24s} {pos.min():6d} {int(np.median(pos)):7d} "
              f"{pos.max():6d} {int((N == 0).sum()):13,d} {int(((N > 0) & (N < 10)).sum()):17,d}")


def print_comparison(C: dict, D: dict) -> None:
    """Seasonal fraction of time, both methods, every class and type."""
    for k, cat in enumerate(PHASE_CATS):
        print(f"\n  {cat}: fraction of time per season, per_hour / pooled "
              f"(difference in percentage points)")
        print("  " + " " * 8 + "".join(f"{SLOT_NAMES[s]:>22s}" for s in CLASS_SLOTS))
        for y in C["seasons"]:
            row = []
            for s in CLASS_SLOTS:
                _a, f1, _ = finish(D, C["day_season"] == y, s, "per_hour")
                _a, f2, _ = finish(D, C["day_season"] == y, s, "pooled")
                row.append(f"{f1[k]:.3f}/{f2[k]:.3f} ({100 * (f1[k] - f2[k]):+5.1f})")
            print(f"  {season_label(y):<8s}" + "".join(f"{r:>22s}" for r in row))


# ----------------------------------------------------------------------------
# readME
# ----------------------------------------------------------------------------
def write_readme(C: dict, method: str, out_dir: Path) -> Path:
    span = f"{C['seasons'][0]}_{C['seasons'][-1] + 1}"
    H = C["hourly"]
    size_lines = []
    for s in CLASS_SLOTS:
        N = H[:, s, I_ALL]
        pos = N[N > 0]
        size_lines.append(f"  {SLOT_NAMES[s]:<16s} {pos.min():5d} - {pos.max():5d} cells "
                          f"(median {int(np.median(pos))}); absent in "
                          f"{int((N == 0).sum()):,} of {N.size:,} hours")
    defn = ("""  per_hour (used here): divide by the number of cells at EACH hour,
      then sum over the hours.

        avg hours per cell = sum_t f(t)
        fraction of time   = sum_t f(t) / T

      Every hour counts equally, however many cells the class held.
      For Land and Coastal (fixed cells) this is simply the total
      cell-hours divided by the number of cells.""" if method == "per_hour" else
            """  pooled (used here): pool the cell-hours first, divide once.

        fraction of time   = sum_t n(t) / sum_t N(t)
        avg hours per cell = fraction of time * T

      Every cell-hour counts equally. This equals the cell-hour table
      divided by the matching table in ../normalization/.""")
    text = f"""PER-GRID-CELL NORMALISATION OF THE SURFACE-CLASS TABLES
=======================================================

Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_class_per_cell_csvs.py
(repo {git_describe()}), method = {method}.

Companion to ../readME.txt, which describes the data, the filters and the
surface classes. Every file here is a normalised version of one of the
cell-hour tables in the folder above, for the five surface classes. UTQ is a
single grid cell, so its tables are already per cell and are not repeated.


WHY NORMALISE
-------------
The tables above count cell-hours summed over every cell of a class. That
total grows with the size of the class, and the size of the three ocean
classes changes every hour as the ice edge moves (their membership is decided
from that hour's sea ice concentration). Over these seasons the classes held:

{chr(10).join(size_lines)}


DEFINITIONS
-----------
For one class and hour t:

  N(t) = grid cells in the class at hour t with valid tcc, tclw and tciw
  n(t) = those cells that also pass every filter (cloudy, condensate,
         not precipitating) and hold the cloud type
  f(t) = n(t) / N(t), the share of the class with that cloud type at t

For a period (day, month or season), T = the number of hours in it during
which the class existed (N(t) > 0); hours without the class add nothing.

{defn}

"avg hours per cell" is the number of hours an average grid cell of the
class spent under that cloud type, counting only the T hours the class
existed. "fraction of time" is the same as a share of T (0 to 1); multiply
by the full length of the period to express it as hours of a cell that
stayed in the class throughout, which is how the Ocean Visions class figures
present it. Cells are not area-weighted (those figures weight by
cos(latitude)).

A fraction is blank (NaN) where the class did not exist at all in that
period; the matching average is 0.

The ice-only, liquid-only and liquid-containing definitions and the 12-hour
"no phase" note are as in ../readME.txt.


CAUTION: SMALL CLASSES
----------------------
When a class holds few cells, f(t) jumps in large steps (one cell of two is
f = 0.5), so daily values for the marginal ice zone and, in mid-winter,
open ocean are noisy. Monthly and seasonal values average this out.


FILES ({len(CLASS_SLOTS)} classes: Land, Coastal, OpenOcean, MarginalSeaIce, PackIce)
-----
Same layouts as the cell-hour tables:

  ERA5_avgHoursPerCell_perSeason_{span}_<CLASS>.csv
  ERA5_fractionOfTime_perSeason_{span}_<CLASS>.csv
      rows = seasons; columns = liquid_containing, liquid_only, ice_only

  ERA5_avgHoursPerCell_perMonth_{span}_<TYPE>_<CLASS>.csv
  ERA5_fractionOfTime_perMonth_{span}_<TYPE>_<CLASS>.csv
      rows = Oct ... Mar; columns = seasons;
      <TYPE> = liquidContaining | liquidOnly | iceOnly

  ERA5_avgHoursPerCell_perDay_{span}_<CLASS>.csv
  ERA5_fractionOfTime_perDay_{span}_<CLASS>.csv
      one row per day; columns = date, liquid_containing, liquid_only, ice_only

  ERA5_hoursClassPresent_perSeason_{span}.csv   T per season (column per class)
  ERA5_hoursClassPresent_perMonth_{span}_<CLASS>.csv   T per month
  ERA5_hoursClassPresent_perDay_{span}.csv      T per day (column per class)
"""
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                   help=f"averaging method (default {DEFAULT_METHOD}); see the docstring")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default era5_hour_count_csv/class_normalized, "
                        "or class_normalized_pooled for --method pooled)")
    p.add_argument("--compare", action="store_true",
                   help="print both methods' seasonal fractions side by side")
    opts = p.parse_args(argv)
    out_dir = opts.out_dir or (DEFAULT_OUT_DIR / ("class_normalized" if opts.method == "per_hour"
                                                  else "class_normalized_pooled"))

    args = build_args(SETTINGS)
    print("=" * 72)
    print(f"ERA5 per-grid-cell hours and time fractions by class ({opts.method})")
    print("=" * 72)
    C = count_cell_hours(args)
    D = daily_sums(C)
    check(C, D)
    print_class_sizes(C)
    if opts.compare:
        print_comparison(C, D)

    written = write_all(C, D, opts.method, out_dir)
    readme = write_readme(C, opts.method, out_dir)
    print(f"\n  -> {len(written)} CSV files + {readme.name} in {out_dir}")


if __name__ == "__main__":
    main()
