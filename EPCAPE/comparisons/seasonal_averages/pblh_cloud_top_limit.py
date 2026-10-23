#!/usr/bin/env python
"""PBLHTTHERMO boundary-layer height under low cloud: 3500-m vs 2000-m cloud-top limit (seasonal table row 15).

Why: row 15 of the seasonal-averages table (EPCAPE_SeasonalAveragesTable.xlsx, sheet Cloud&MetQuantities; row 17 in
the older sheet that check_seasonal_averages.ipynb was written against) gives the PBLHTTHERMO PBL height only at
times when KAZR-ARSCL sees a LOW cloud, defined as a lowest-layer cloud top <= 3500 m (the notebook's
LOW_CLOUD_MAX_M). A cloud whose top is 2-3.5 km is not obviously a boundary-layer cloud, so this script repeats the
notebook's calculation with the limit lowered to 2000 m and compares the two: which 10-min PBL values drop out, what
they look like, and how the seasonal mean, std, median and IQR change.

Data (EPCAPE products in config.yaml; campaign window 15 Feb 2023 - 14 Feb 2024, UTC):
  pblh_thermo_M1  PBLHTTHERMO VAP, epcpblhtthermoM1.c1: pbl_height (km above ground) from combined Raman-lidar and
                  TROPoe potential-temperature profiles; one value every 10 min (:00, :10, ...), no QC field, no data
                  11-31 Oct 2023. pbl_height takes only 39 distinct values (a stretched height grid: 10-m steps near
                  the surface, ~80 m at 700 m, ~160 m at 1.5 km), so medians and quartiles move only in jumps of
                  one grid level.
  cloud_arscl_M1  KAZR-ARSCL, epcarsclkazr1kolliasM1.c1: cloud_layer_top_height and cloud_layer_base_height
                  (time, layer), m above ground, 4-s samples, up to 10 hydrometeor layers (layer 0 = lowest). No QC
                  field for these variables.

Method: notebook Sections 4-5, run once for each cloud-top limit X (the same functions, from quantities.py):
  1. cth = Q.lowest_layer_top_m(arscl, max_top_m=X, window_min=0, cloud_sources=None): the top of the LOWEST
     ARSCL layer at every 4-s sample whose lowest layer exists, is above ground and has its top <= X
     (notebook settings AVG_PERIOD_CT = 0, CT_CLOUD_SOURCES = None).
  2. LOW_CLOUD_5MIN = cth.notna().resample("5min").max() -> True for every 5-min bin holding at least one such sample.
  3. pblh = Q.pblh_thermo_m(thermo): pbl_height in m, as delivered (no averaging).
  4. Each 10-min PBL value takes the flag of the nearest 5-min bin within 5 min, i.e. the bin [t, t + 5 min).
  5. Seasonal mean, std (ddof = 1), median and n of the PBL values whose bin is flagged (S.seasonal_stats).
  With X = 3500 m this reproduces row 15 exactly (printed check).

What lowering the limit does: a bin is flagged at limit X exactly when the LOWEST lowest-layer top among its 4-s
samples is <= X. So the 2000-m sample is a SUBSET of the 3500-m sample: the PBL values themselves never change, the
2000-m limit only removes the values whose bin's lowest cloud top lies between 2000 and 3500 m ("dropped" below).
Hence mean(2000) - mean(3500) = f_dropped x (mean(2000) - mean(dropped)), with f_dropped = n_dropped / n(3500)
(printed as a check). The same equivalence gives the flags for any other limit from one per-bin table, which is
how the sweep is computed; for the two main limits the per-bin flags are checked against the notebook's code.

Statistics
  n, hours        PBL values in the selection; hours = n x 10 min
  mean, std       over the 10-min values (std with ddof = 1, as in the notebook)
  median, q25, q75, iqr   numpy.percentile (linear interpolation; with 39 grid levels these sit on or between levels)
  bin top, bin base       median over the selection of the matched 5-min bin's lowest cloud top, and of the base of
                  the same layer (the 4-s sample with the lowest top in the bin): what kind of cloud was overhead
  95 % CI         day-block bootstrap (resample whole UTC days with replacement, the same days for every selection,
                  so differences are paired): 10-min values within a day are strongly autocorrelated, days much
                  less so. Multi-day weather persistence is ignored, so the intervals are, if anything, too narrow.
                  Block bootstrap: Wilks (2011), Statistical Methods in the Atmospheric Sciences, 3rd ed., Ch. 5.
                  The seed is printed and written to the About sheet.

Outputs (in pblh_cloud_top_limit/, next to this script; --out chooses another folder):
  pblh_cth_limit_stats.csv        statistics per selection and season: all times, top <= 3500 m (row 15),
                                  top <= 2000 m, and the dropped values (2000 < top <= 3500 m)
  pblh_cth_limit_difference.csv   per season: 2000 m minus 3500 m for n, mean (with CI), std, median, IQR,
                                  and the decomposition check
  pblh_cth_limit_sweep.csv        the same statistics for a range of limits (default 1000-3500 m every 250 m,
                                  and no limit)
  pblh_cth_limit.xlsx             the three tables (rounded) plus an About sheet with every setting
  pblh_cth_limit.png / .pdf       top: seasonal mean vs cloud-top limit with its 95 % CI; bottom: distributions
                                  (cumulative) of the 3500-m, 2000-m and dropped values

Examples (from the Python-Research folder)
  python EPCAPE/comparisons/seasonal_averages/pblh_cloud_top_limit.py
  python EPCAPE/comparisons/seasonal_averages/pblh_cloud_top_limit.py --new-m 2500 --n-boot 5000 --block-days 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402

from EPCAPE.analysis_tools.derived import git_revision  # noqa: E402
from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.plotting import GRID, INK, INK_2, use_style  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: the notebook's values (settings cell, Sections 4-5), and this comparison's
# ---------------------------------------------------------------------------
BASELINE_LIMIT_M = 3500.0  # notebook LOW_CLOUD_MAX_M: the limit behind table row 15
NEW_LIMIT_M = 2000.0  # the lower limit to compare against
AVG_PERIOD_CT_MIN = 0  # notebook AVG_PERIOD_CT: 0 = every 4-s ARSCL sample (no time averaging)
CT_CLOUD_SOURCES = None  # notebook CT_CLOUD_SOURCES: None = any ARSCL layer, whichever instrument detected it
FLAG_BIN = "5min"  # notebook LOW_CLOUD_5MIN: the bin on which "low cloud overhead" is decided
MATCH_TOLERANCE = pd.Timedelta("5min")  # notebook: each 10-min PBL value takes the nearest bin within 5 min
SWEEP_LIMITS_M = tuple(np.arange(1000.0, 3501.0, 250.0)) + (np.inf,)  # limits for the sweep table and figure
N_BOOT = 2000  # bootstrap resamples
BLOCK_DAYS = 1  # bootstrap block length in days (UTC days)
SEED = 20261023  # bootstrap RNG seed (recorded in the output)

# Row 15 of EPCAPE_SeasonalAveragesTable.xlsx (identical in EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx):
# "epcpblhtthermoM1.c1 (PBLHTTHERMO) + KAZR-ARSCL", columns E-I = EPCAPE, Spring, Summer, Fall, Winter.
TABLE_ROW15 = ("804 ± 411", "779 ± 290", "618 ± 234", "920 ± 472", "1114 ± 595")

# Figure colours: the table's limit in neutral grey, the new limit in one accent, the dropped values in a second
# (the purple and gold of the other seasonal-averages figures; text stays in INK greys)
COLOR_BASELINE = INK_2
COLOR_NEW = "#5b4bb7"
COLOR_DROPPED = "#c99700"


# ---------------------------------------------------------------------------
# Notebook Sections 4-5: low-cloud flag and PBL height
# ---------------------------------------------------------------------------
def low_cloud_flag_5min(arscl: xr.Dataset, max_top_m: float) -> pd.Series:
    """Notebook Section 4, steps 2 and 5: True for each 5-min bin holding >= 1 ARSCL sample with a lowest-layer top
    <= max_top_m (m above ground). Regular 5-min grid from the first to the last such sample."""
    cth, _ = Q.lowest_layer_top_m(arscl, max_top_m=max_top_m, window_min=AVG_PERIOD_CT_MIN,
                                  cloud_sources=CT_CLOUD_SOURCES)
    # with window_min = 0, cth holds only the low-cloud samples, so empty bins -> NaN -> False
    return cth.notna().resample(FLAG_BIN).max().astype("boolean").fillna(False).astype(bool)


def match_to_pblh(flag_5min: pd.Series, pblh_index: pd.DatetimeIndex) -> np.ndarray:
    """Notebook Section 5, step 3: each 10-min PBL time takes the flag of the nearest bin within 5 min. Both grids
    fall on whole 5-/10-minute marks, so that is the bin STARTING at the PBL time stamp; no match -> False."""
    return (flag_5min.reindex(pblh_index, method="nearest", tolerance=MATCH_TOLERANCE)
            .astype("boolean").fillna(False).astype(bool).to_numpy())


def lowest_cloud_by_bin(arscl: xr.Dataset) -> pd.DataFrame:
    """Per 5-min bin: the lowest lowest-layer cloud top (m) among the bin's 4-s samples, and that layer's base (m).

    Uses Q.lowest_layer_top_m with no height limit, so a sample counts exactly when it would count in the notebook
    (layer 0 exists, base and top >= 0, top > 0). A bin is then flagged at limit X exactly when its top_m <= X:
    this one table gives the notebook's flag for any limit (checked in main for the two main limits). The result
    is on a regular 5-min grid (NaN = no cloud layer in the bin), like the notebook's LOW_CLOUD_5MIN, so the
    nearest-bin match in match_to_pblh picks the same bin."""
    top_any_m, _ = Q.lowest_layer_top_m(arscl, max_top_m=np.inf, window_min=AVG_PERIOD_CT_MIN,
                                        cloud_sources=CT_CLOUD_SOURCES)
    # base of the lowest layer at the same 4-s time stamps (the layer whose top is in top_any_m)
    layer_dim = [d for d in arscl["cloud_layer_base_height"].dims if d != "time"][0]
    base0_m = arscl["cloud_layer_base_height"].isel({layer_dim: 0}).to_series()
    df = pd.DataFrame({"top_m": top_any_m.to_numpy(),
                       "base_m": base0_m.reindex(top_any_m.index).to_numpy(),
                       "bin": top_any_m.index.floor(FLAG_BIN)})  # same bins as resample("5min"): 5 min divides a day
    # the 4-s sample with the lowest top in each bin (row positions, so duplicate time stamps cannot confuse it)
    rows = df.groupby("bin")["top_m"].idxmin().to_numpy()
    bins = df.loc[rows].set_index("bin")[["top_m", "base_m"]]
    return bins.reindex(pd.date_range(bins.index[0], bins.index[-1], freq=FLAG_BIN))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def describe(values_m: np.ndarray, step_min: float) -> Dict[str, float]:
    """n, hours (n x step), mean, std (ddof = 1), median, 25th/75th percentiles and IQR of the finite values (m)."""
    v = np.asarray(values_m, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "hours": 0.0, **{k: np.nan for k in ("mean", "std", "median", "q25", "q75", "iqr")}}
    q25, median, q75 = np.percentile(v, [25, 50, 75])  # linear interpolation (numpy default)
    return {"n": int(v.size), "hours": v.size * step_min / 60.0, "mean": v.mean(),
            "std": v.std(ddof=1) if v.size > 1 else np.nan, "median": median, "q25": q25, "q75": q75,
            "iqr": q75 - q25}


def bootstrap_means(pblh_m: pd.Series, selections: Dict[str, np.ndarray], season: str, *, n_boot: int,
                    block_days: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Day-block bootstrap of the season's mean PBL height for every selection, with the SAME resampled blocks for
    all selections (so differences between selections are paired).

    Inputs
      pblh_m      all 10-min PBL values (m), NaN where missing
      selections  name -> boolean array over pblh_m (True = value in the selection)
      season      sheet season (seasonal.SEASONS)
    Returns name -> array (n_boot,) of resampled means (m). Blocks are block_days-long runs of UTC days that hold at
    least one finite PBL value in the season; a resample draws as many blocks as there are, with replacement."""
    in_season = S.season_mask(pblh_m.index, season) & np.isfinite(pblh_m.to_numpy())
    t = pblh_m.index[in_season]
    v = pblh_m.to_numpy(dtype=np.float64)[in_season]
    # block number of each value: whole UTC days since the campaign start, grouped block_days at a time
    start = pd.Timestamp(S.SEASONS["EPCAPE"][0][0])
    block = np.asarray((t.normalize() - start) // pd.Timedelta(days=block_days))
    _, inv = np.unique(block, return_inverse=True)  # inv: block position 0..K-1 of each value
    n_blocks = int(inv.max()) + 1
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))  # (n_boot, K) block positions
    out = {}
    for name, sel in selections.items():
        s = sel[in_season]
        # per-block sum and count of the selected values, then the ratio of resampled totals
        sums = np.bincount(inv, weights=np.where(s, v, 0.0), minlength=n_blocks)
        counts = np.bincount(inv, weights=s.astype(np.float64), minlength=n_blocks)
        with np.errstate(invalid="ignore", divide="ignore"):  # a resample with no selected value -> NaN
            out[name] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return out


def ci95(samples: np.ndarray) -> Tuple[float, float]:
    """2.5th and 97.5th percentiles of the finite bootstrap samples (percentile interval)."""
    s = samples[np.isfinite(samples)]
    return (np.nan, np.nan) if s.size == 0 else tuple(np.percentile(s, [2.5, 97.5]))


def limit_label(limit_m: float) -> str:
    return "no limit" if not np.isfinite(limit_m) else f"top ≤ {limit_m:g} m"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(sweep: pd.DataFrame, pblh_m: pd.Series, sel: Dict[str, np.ndarray], names: Dict[str, str],
         baseline_m: float, new_m: float, out_dir: Path) -> None:
    """Top row: seasonal mean vs cloud-top limit (95 % CI band), the two main limits marked. Bottom row: cumulative
    distributions of the PBL values at the two limits and of the dropped values."""
    use_style()
    fig, axes = plt.subplots(2, len(S.SEASON_ORDER), figsize=(14.5, 6.6), sharey="row")
    finite = sweep[np.isfinite(sweep["limit_m"])]
    for j, season in enumerate(S.SEASON_ORDER):
        # --- top: mean vs limit ----------------------------------------------------------------------------------
        ax = axes[0, j]
        d = finite[finite["season"] == season].sort_values("limit_m")
        ax.fill_between(d["limit_m"], d["mean_ci_lo"], d["mean_ci_hi"], color=GRID, lw=0,
                        label="mean: 95 % CI (day-block bootstrap)")
        ax.plot(d["limit_m"], d["mean"], color=INK, lw=1.5, marker="o", ms=3, label="mean vs cloud-top limit")
        # the two main limits in the colours of their distributions below (no legend entry of their own)
        for limit, color in ((baseline_m, COLOR_BASELINE), (new_m, COLOR_NEW)):
            row = d[d["limit_m"] == limit]
            if len(row):  # a main limit that is not on the sweep grid is simply not marked
                ax.plot(row["limit_m"], row["mean"], ls="none", marker="o", ms=8, mfc=color, mec="white", mew=1.5,
                        zorder=3)
        ax.set_title(season, loc="left", color=INK)
        ax.set_xlabel("cloud-top limit (m)")
        if j == 0:
            ax.set_ylabel("mean PBL height (m)")
        # --- bottom: cumulative distributions ------------------------------------------------------------------
        ax = axes[1, j]
        in_season = S.season_mask(pblh_m.index, season) & np.isfinite(pblh_m.to_numpy())
        for key, color, ls in (("baseline", COLOR_BASELINE, "-"), ("new", COLOR_NEW, "--"), ("dropped", COLOR_DROPPED, "-")):
            v = np.sort(pblh_m.to_numpy()[in_season & sel[key]])
            if v.size:
                ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=color, ls=ls, lw=1.5,
                        label=f"{names[key]}")
        n_base, n_drop = int((in_season & sel["baseline"]).sum()), int((in_season & sel["dropped"]).sum())
        ax.text(0.97, 0.04, f"dropped {n_drop:,} of {n_base:,}\n({100 * n_drop / max(n_base, 1):.1f} %)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xlabel("PBL height (m)")
        if j == 0:
            ax.set_ylabel("cumulative fraction")
        ax.set_xlim(0, None)
    # one legend for the whole figure, between the title and the panels: the band and line of the top row, then
    # the three samples (their colours also mark the two main limits in the top row)
    handles, labels = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.84, wspace=0.12, hspace=0.42)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.935), ncol=5, frameon=False,
               handlelength=2.2, columnspacing=1.4)
    fig.suptitle("PBLHTTHERMO PBL height while KAZR-ARSCL sees a low cloud: effect of the cloud-top limit "
                 "(10-min values; sheet seasons, UTC)", x=0.01, y=0.995, ha="left", color=INK, fontsize=11,
                 fontweight="semibold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pblh_cth_limit.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--baseline-m", type=float, default=BASELINE_LIMIT_M,
                        help="cloud-top limit of the table value (m above ground; default 3500 = row 15)")
    parser.add_argument("--new-m", type=float, default=NEW_LIMIT_M, help="cloud-top limit to compare (default 2000)")
    parser.add_argument("--sweep-m", type=float, nargs="+", default=None,
                        help="limits for the sweep (m); default 1000-3500 every 250 m; 'no limit' is always added")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap resamples (default 2000)")
    parser.add_argument("--block-days", type=int, default=BLOCK_DAYS, help="bootstrap block length in days (default 1)")
    parser.add_argument("--seed", type=int, default=SEED, help="bootstrap RNG seed")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "pblh_cloud_top_limit",
                        help="output folder (default: pblh_cloud_top_limit/ next to this script)")
    args = parser.parse_args()
    if args.new_m == args.baseline_m:
        parser.error("--new-m must differ from --baseline-m")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    hi_m, lo_m = max(args.baseline_m, args.new_m), min(args.baseline_m, args.new_m)
    sweep_limits = tuple(sorted(set(args.sweep_m or SWEEP_LIMITS_M) | {args.baseline_m, args.new_m, np.inf}))
    rng = np.random.default_rng(args.seed)
    print(f"Bootstrap: {args.n_boot} resamples of {args.block_days}-day blocks, seed {args.seed}")

    # --- 1. PBL height, as in notebook Section 5 (pbl_height km -> m, no averaging) ------------------------------
    pblh_m, _ = Q.pblh_thermo_m(load_product("pblh_thermo_M1"))
    step_min = float(pd.Series(pblh_m.index).diff().median() / pd.Timedelta(minutes=1))  # 10 min
    levels_m = np.unique(pblh_m.dropna().to_numpy())
    print(f"PBLHTTHERMO: {int(pblh_m.notna().sum()):,} values every {step_min:g} min; pbl_height takes "
          f"{levels_m.size} distinct values (grid step {np.diff(levels_m).min():g} m near the surface to "
          f"{np.diff(levels_m)[np.searchsorted(levels_m, 1500.0) - 1]:g} m at 1.5 km)")

    # --- 2. Low-cloud flags, notebook Section 4 code, at the two limits; per-bin lowest top for the sweep --------
    arscl = load_product("cloud_arscl_M1")
    flagged = {limit: match_to_pblh(low_cloud_flag_5min(arscl, limit), pblh_m.index)
               for limit in (args.baseline_m, args.new_m)}
    bins = lowest_cloud_by_bin(arscl)
    del arscl  # ~0.7 GB
    # the matched bin's lowest cloud top and its base, for every 10-min PBL value (NaN = no cloud layer)
    bin_top_m = bins["top_m"].reindex(pblh_m.index, method="nearest", tolerance=MATCH_TOLERANCE).to_numpy()
    bin_base_m = bins["base_m"].reindex(pblh_m.index, method="nearest", tolerance=MATCH_TOLERANCE).to_numpy()
    # check: "bin's lowest top <= limit" must reproduce the notebook's flags exactly, or the sweep would not be
    # the notebook's calculation
    for limit, flag in flagged.items():
        n_bad = int(((bin_top_m <= limit) != flag).sum())
        if n_bad:
            raise RuntimeError(f"per-bin lowest top differs from the notebook flag at {limit:g} m for {n_bad} values")
    print(f"Per-bin lowest cloud top reproduces the notebook's low-cloud flags at {args.baseline_m:g} and "
          f"{args.new_m:g} m (0 differences)")

    # --- 3. Selections ---------------------------------------------------------------------------------------------
    finite = np.isfinite(pblh_m.to_numpy())
    sel = {"all": finite,
           "baseline": finite & flagged[args.baseline_m],
           "new": finite & flagged[args.new_m],
           # kept only by the higher limit: lowest top in the bin between the two limits
           "dropped": finite & flagged[hi_m] & ~flagged[lo_m]}
    names = {"all": "all times (no cloud condition)",
             "baseline": f"cloud top ≤ {args.baseline_m:g} m" + (" (table row 15)" if args.baseline_m == 3500 else ""),
             "new": f"cloud top ≤ {args.new_m:g} m",
             "dropped": f"{'dropped' if args.new_m < args.baseline_m else 'added'}: {lo_m:g} < cloud top ≤ {hi_m:g} m"}
    if not np.array_equal(flagged[lo_m] & finite, (flagged[lo_m] & flagged[hi_m]) & finite):
        raise RuntimeError("the lower-limit sample is not a subset of the higher-limit sample")

    # --- 4. Statistics per selection and season (+ bootstrap of the means) --------------------------------------
    sweep_sel = {limit: finite & (bin_top_m <= limit) for limit in sweep_limits}
    stat_rows, sweep_rows, diff_rows = [], [], []
    for season in S.SEASON_ORDER:
        in_season = S.season_mask(pblh_m.index, season)
        boot = bootstrap_means(pblh_m, {**sel, **{f"limit {k}": v for k, v in sweep_sel.items()}}, season,
                               n_boot=args.n_boot, block_days=args.block_days, rng=rng)
        for key in ("all", "baseline", "new", "dropped"):
            m = in_season & sel[key]
            lo, hi = ci95(boot[key])
            cloud = {} if key == "all" else {"bin_top_median_m": np.nanmedian(bin_top_m[m]) if m.any() else np.nan,
                                             "bin_base_median_m": np.nanmedian(bin_base_m[m]) if m.any() else np.nan}
            stat_rows.append({"selection": names[key], "season": season,
                              **describe(pblh_m.to_numpy()[m], step_min), "mean_ci_lo": lo, "mean_ci_hi": hi,
                              **cloud})
        n_season_base = int((in_season & sel["baseline"]).sum())
        for limit in sweep_limits:
            m = in_season & sweep_sel[limit]
            lo, hi = ci95(boot[f"limit {limit}"])
            sweep_rows.append({"limit_m": limit, "limit": limit_label(limit), "season": season,
                               **describe(pblh_m.to_numpy()[m], step_min), "mean_ci_lo": lo, "mean_ci_hi": hi,
                               "n_pct_of_baseline": 100.0 * m.sum() / max(n_season_base, 1)})
        # differences new - baseline, with the paired bootstrap interval of the change in the mean
        b = describe(pblh_m.to_numpy()[in_season & sel["baseline"]], step_min)
        n = describe(pblh_m.to_numpy()[in_season & sel["new"]], step_min)
        dr = describe(pblh_m.to_numpy()[in_season & sel["dropped"]], step_min)
        d_lo, d_hi = ci95(boot["new"] - boot["baseline"])
        base_lo, base_hi = ci95(boot["baseline"])
        # decomposition: with H = higher-limit sample, L = lower-limit sample (a subset of H) and D = H minus L,
        # mean(H) - mean(L) = (n_D / n_H) x (mean(D) - mean(L)). Lowering the limit (new = L) gives
        # mean(new) - mean(baseline) = (n_D / n_baseline) x (mean(new) - mean(dropped)).
        hi_stats, lo_stats = (b, n) if args.baseline_m > args.new_m else (n, b)
        sign = -1.0 if args.baseline_m > args.new_m else 1.0  # new - baseline = -(H - L) when the limit is lowered
        decomposition = (sign * dr["n"] / hi_stats["n"] * (dr["mean"] - lo_stats["mean"])) if dr["n"] else 0.0
        diff_rows.append({
            "season": season, "n_baseline": b["n"], "n_new": n["n"], "n_change": n["n"] - b["n"],
            "n_change_pct": 100.0 * (n["n"] - b["n"]) / b["n"] if b["n"] else np.nan,
            "mean_baseline": b["mean"], "mean_new": n["mean"], "mean_change": n["mean"] - b["mean"],
            "mean_change_pct": 100.0 * (n["mean"] - b["mean"]) / b["mean"],
            "mean_change_ci_lo": d_lo, "mean_change_ci_hi": d_hi,
            "mean_baseline_ci_halfwidth": (base_hi - base_lo) / 2.0,
            "std_baseline": b["std"], "std_new": n["std"], "std_change": n["std"] - b["std"],
            "median_baseline": b["median"], "median_new": n["median"], "median_change": n["median"] - b["median"],
            "iqr_baseline": b["iqr"], "iqr_new": n["iqr"], "iqr_change": n["iqr"] - b["iqr"],
            "n_dropped": dr["n"], "mean_dropped": dr["mean"],
            "mean_change_from_decomposition": decomposition,
        })
    stats = pd.DataFrame(stat_rows)
    sweep = pd.DataFrame(sweep_rows)
    diff = pd.DataFrame(diff_rows).set_index("season")

    # --- 5. Print ---------------------------------------------------------------------------------------------------
    show = ["n", "hours", "mean", "std", "median", "q25", "q75", "iqr", "mean_ci_lo", "mean_ci_hi",
            "bin_top_median_m", "bin_base_median_m"]
    print("\nPBL height (m) per selection and season; bin top/base = median lowest cloud top and its base in the "
          "matched 5-min bin:")
    print(stats.set_index(["selection", "season"])[show].round(1).to_string())

    print(f"\nChange from {args.baseline_m:g} m to {args.new_m:g} m (new - baseline), m:")
    print(diff[["n_baseline", "n_new", "n_change", "n_change_pct", "mean_baseline", "mean_new", "mean_change",
                "mean_change_pct", "mean_change_ci_lo", "mean_change_ci_hi", "mean_baseline_ci_halfwidth"]]
          .round(1).to_string())
    print(diff[["std_baseline", "std_new", "std_change", "median_baseline", "median_new", "median_change",
                "iqr_baseline", "iqr_new", "iqr_change", "mean_dropped", "mean_change",
                "mean_change_from_decomposition"]].round(1).to_string())

    print("\nSeasonal mean ± std as the table writes it:")
    pm = {names[k]: [S.format_pm(*diff.loc[s, [f"mean_{c}", f"std_{c}"]]) for s in S.SEASON_ORDER]
          for k, c in (("baseline", "baseline"), ("new", "new"))}
    if args.baseline_m == 3500:
        pm["table row 15"] = list(TABLE_ROW15)
    print(pd.DataFrame(pm, index=S.SEASON_ORDER).T.to_string())
    if args.baseline_m == 3500:
        ok = pm[names["baseline"]] == list(TABLE_ROW15)
        print(f"Row 15 reproduced at {args.baseline_m:g} m: {'yes' if ok else 'NO'}")

    print("\nSweep: seasonal mean (m) and values kept (% of the baseline sample) per cloud-top limit:")
    wide = sweep.pivot(index="limit", columns="season", values="mean")[S.SEASON_ORDER]
    pct = sweep.pivot(index="limit", columns="season", values="n_pct_of_baseline")[S.SEASON_ORDER]
    order = [limit_label(x) for x in sweep_limits]
    print(pd.concat({"mean (m)": wide.loc[order], "n (% of baseline)": pct.loc[order]}, axis=1).round(1).to_string())

    # --- 6. Write ---------------------------------------------------------------------------------------------------
    stats.to_csv(args.out / "pblh_cth_limit_stats.csv", index=False)
    diff.to_csv(args.out / "pblh_cth_limit_difference.csv")
    sweep.to_csv(args.out / "pblh_cth_limit_sweep.csv", index=False)
    about = pd.DataFrame({"value": {
        "quantity": "PBLHTTHERMO pbl_height (epcpblhtthermoM1.c1), m above ground, 10-min values as delivered "
                    "(no QC field, no averaging)",
        "low-cloud condition": "KAZR-ARSCL (epcarsclkazr1kolliasM1.c1) top of the lowest layer <= limit in at least "
                               "one 4-s sample of the 5-min bin starting at the PBL time stamp (notebook Sections 4-5)",
        "baseline limit (m)": args.baseline_m,
        "new limit (m)": args.new_m,
        "ARSCL settings": f"AVG_PERIOD_CT = {AVG_PERIOD_CT_MIN}, CT_CLOUD_SOURCES = {CT_CLOUD_SOURCES}",
        "dropped": f"values whose bin's lowest cloud top is between {lo_m:g} and {hi_m:g} m",
        "bin top / base": "median over the selection of the matched bin's lowest cloud top, and of the base of the "
                          "same layer (4-s sample with the lowest top)",
        "seasons": "sheet seasons, UTC: Spring AMJ23, Summer JAS23, Fall OND23, Winter 15 Feb-31 Mar 2023 + "
                   "1 Jan-14 Feb 2024; EPCAPE = 15 Feb 2023-14 Feb 2024",
        "mean, std": "over the 10-min values; std ddof = 1",
        "median, q25, q75, iqr": "numpy.percentile, linear interpolation; pbl_height has only "
                                 f"{levels_m.size} distinct values, so these move in grid steps",
        "95 % CI": f"day-block bootstrap, {args.n_boot} resamples, {args.block_days}-day blocks (UTC), paired across "
                   "selections; percentile interval (Wilks 2011, Ch. 5)",
        "seed": args.seed,
        "sweep limits (m)": ", ".join(limit_label(x) for x in sweep_limits),
        "table row 15": " | ".join(TABLE_ROW15),
        "script": Path(__file__).name,
        "git revision": git_revision(),
    }})
    xlsx = args.out / "pblh_cth_limit.xlsx"
    with pd.ExcelWriter(xlsx) as xl:
        stats.round(1).to_excel(xl, sheet_name="Stats", index=False)
        diff.round(1).to_excel(xl, sheet_name="Difference")
        sweep.round(1).to_excel(xl, sheet_name="Sweep", index=False)
        about.to_excel(xl, sheet_name="About")
    plot(sweep, pblh_m, sel, names, args.baseline_m, args.new_m, args.out)
    print(f"\nWrote the CSV tables, {xlsx.name} and pblh_cth_limit.png/.pdf to {args.out}")


if __name__ == "__main__":
    main()
