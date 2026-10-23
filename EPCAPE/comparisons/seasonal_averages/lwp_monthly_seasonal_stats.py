#!/usr/bin/env python
"""Pier MWR liquid water path per month and per season: mean, median and IQR (seasonal-averages table row 10).

Why: row 10 of the seasonal-averages table (EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx, sheet
Cloud&MetQuantities) gives the MWRRET v2 LWP as mean ± std per season. LWP is strongly right-skewed (the std is about
as large as the mean), so the median and interquartile range describe a typical cloud better, and monthly values
show how much each season's statistic depends on which months had data. This script builds exactly the same
sample as Section 3 of check_seasonal_averages.ipynb (code block 3) and tabulates it by month and by season.

Data (EPCAPE products in config.yaml; campaign window 15 Feb 2023 - 14 Feb 2024, UTC):
  lwp_mwrret2_M1  MWRRET v2 (epcmwrret2turnM1.c1, 3-channel MWR, ~1.3-s samples): phys_lwp with qc_phys_lwp,
                  phys_qc_flag (0 = retrieval good), and the ceilometer fields cbh_detected (km) and
                  detection_status copied onto the MWR times. [default; table row 10]
  lwp_mwrret1_M1  MWRRET v1 (epcmwrret1liljclouM1.c2, 2-channel MWR, ~20-s samples; Turner et al. 2007,
                  IEEE TGRS 45, 3680): phys_lwp with qc_phys_lwp, and cloud_base_height with its QC field.
                  [--retrieval v1]
  rain_ld_M1      Pier Parsivel (epcldM1.b1) precip_rate (mm h-1, 1 min), QC = 0 samples only.

Filters and thresholds (identical to notebook Section 3; the functions are the same, from quantities.py):
  1. QC: qc_phys_lwp = 0 (ARM QC = 0 only); MWRRET v2 also requires phys_qc_flag = 0.
  2. Cloudy scenes only:
       v2  ceilometer detection_status 1-3 and cbh_detected > 0 (Q.cloudy_mwrret2)
       v1  qc_cloud_base_height = 0 and a valid cloud_base_height (Q.cloudy_mwrret1)
  3. LWP > LWP_MIN_GM2 per sample (v2 only, the notebook's LWP_MIN_GM2 = 0 g m-2, added 20 Oct 2026): drops zero
     and negative retrievals under a detected cloud base. --lwp-min-gm2 none removes the limit.
  4. Rain-free: no Parsivel rain (rate > 0) within ±LWP_RAIN_WINDOW_MIN = 30 min of the sample (a wet radome emits
     at 23.8/31.4 GHz and biases LWP high). --include-rain keeps them.
  5. No time averaging (the notebook's AVG_PERIOD_LWP = 0): every sample that passes 1-4 enters the statistics on
     its own. --avg-min N averages over N-min windows first, as the notebook would.

Statistics (one row per month, then one per season):
  months   calendar months in UTC; February 2023 (15-28 Feb) and February 2024 (1-14 Feb) are separate rows.
  seasons  the sheet's seasons (seasonal.SEASONS): Spring AMJ 2023, Summer JAS 2023, Fall OND 2023, Winter
           15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024, and EPCAPE = the whole campaign.
  n            number of samples (or averaging windows) that passed the filters
  hours        n x the median spacing of the retrieval's time stamps: roughly how much cloudy, rain-free time the
               samples cover (approximate: the MWR does not sample perfectly regularly)
  mean, std    sample mean and standard deviation (ddof = 1, as in the notebook and MATLAB's std)
  median, q25, q75, iqr   50th, 25th and 75th percentiles and IQR = q75 - q25 (numpy.percentile, linear
               interpolation between order statistics)
  mean_of_monthly_means   seasons only: the plain average of the season's monthly means, so every month counts
               equally however many samples it has (the half-months of February count as one month each). The
               "mean" column instead weights every sample equally, which is how row 10 was computed.
  For seasons, mean ± std over all samples reproduces the notebook (and row 10, see the printed check).

Outputs (in lwp_monthly/, next to this script; --out chooses another folder; <tag> names the retrieval and filters,
e.g. mwrret2_rain_free_30min_lwp_gt_0):
  lwp_monthly_seasonal_<tag>.csv    the table (all digits)
  lwp_monthly_seasonal_<tag>.xlsx   the same table, rounded to 0.1 g m-2, with a sheet "About" listing the settings

Examples (from the Python-Research folder)
  python EPCAPE/comparisons/seasonal_averages/lwp_monthly_seasonal_stats.py
  python EPCAPE/comparisons/seasonal_averages/lwp_monthly_seasonal_stats.py --lwp-min-gm2 none
  python EPCAPE/comparisons/seasonal_averages/lwp_monthly_seasonal_stats.py --retrieval v1 --include-rain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.analysis_tools.derived import git_revision  # noqa: E402
from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: the values of check_seasonal_averages.ipynb (settings cell and Section 3)
# ---------------------------------------------------------------------------
LWP_RAIN_WINDOW_MIN = 30.0  # rain-free = no Pier Parsivel rain within ±30 min
AVG_PERIOD_LWP_MIN = 0.0  # 0 = no time averaging: every sample that passes the filters counts
LWP_MIN_GM2 = 0.0  # MWRRET v2 only: keep samples with LWP > 0 g m-2 (Andrew, 20 Oct 2026); v1 has no limit

# The two retrievals of notebook Section 3 (same products, cloud masks and extra flags)
RETRIEVALS = {
    "v2": dict(label="MWRRET v2 (3-channel MWR, epcmwrret2turnM1.c1)", product="lwp_mwrret2_M1",
               cloudy=Q.cloudy_mwrret2, require_zero=("phys_qc_flag",), min_gm2=LWP_MIN_GM2),
    "v1": dict(label="MWRRET v1 (2-channel MWR, epcmwrret1liljclouM1.c2)", product="lwp_mwrret1_M1",
               cloudy=Q.cloudy_mwrret1, require_zero=(), min_gm2=None),
}

# Row 10 of EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx (MWRRET v2, rain-free ±30 min, no averaging window;
# "Andrew Buggee"), columns E-I = EPCAPE, Spring, Summer, Fall, Winter. Printed next to this script's seasonal
# mean ± std as a check only.
TABLE_ROW10 = ("75 ± 84", "104 ± 89", "61 ± 70", "44 ± 63", "58 ± 93")


# ---------------------------------------------------------------------------
# The sample: QC, cloudy scenes, LWP limit, rain screen (notebook Section 3, steps 1-5)
# ---------------------------------------------------------------------------
def lwp_sample_gm2(retrieval: str, *, include_rain: bool, min_gm2: Optional[float], rain_window_min: float,
                   avg_min: float) -> Tuple[pd.Series, Dict[str, object]]:
    """LWP (g m-2) of the samples that pass the notebook's filters, indexed by UTC time.

    Returns (lwp_gm2, info). info: label, the screening counts from Q.lwp_cloudy_gm2, the median spacing of the
    retrieval's time stamps (s), and the share of cloudy values removed by the rain screen."""
    spec = RETRIEVALS[retrieval]

    # 1. rain record: Pier Parsivel precipitation rate, QC = 0 samples only (1 min)
    ld_m1 = load_product("rain_ld_M1")
    pier_rain_mm_h = Q.to_series(Q.qc0_values(ld_m1, "precip_rate")[0])
    del ld_m1

    # 2. the retrieval: QC = 0 (+ phys_qc_flag = 0 for v2), cloudy samples, LWP > min_gm2, window average
    ds = load_product(spec["product"])
    # sampling interval of the retrieval itself (all samples, before any filter), for the "hours" column
    step_s = float(pd.Series(ds["time"].values).diff().median() / pd.Timedelta(seconds=1))
    cloudy = spec["cloudy"](ds)
    lwp_gm2, info = Q.lwp_cloudy_gm2(ds, "phys_lwp", cloudy, require_zero=spec["require_zero"],
                                     window_min=avg_min, min_gm2=min_gm2)
    del ds, cloudy  # v2 holds ~22 million samples
    lwp_gm2 = lwp_gm2.dropna()

    # 3. rain screen: drop values within ±rain_window_min of rain (each value covers [t, t + avg_min])
    wet = Q.near_rain(pier_rain_mm_h, lwp_gm2.index, window_min=rain_window_min, value_length_min=avg_min)
    info["near_rain_fraction"] = float(wet.mean()) if wet.size else np.nan
    if not include_rain:
        lwp_gm2 = lwp_gm2[~wet]

    # 4. the campaign window only (the sheet's EPCAPE season)
    lwp_gm2 = lwp_gm2[S.season_mask(lwp_gm2.index, "EPCAPE")]
    info.update(label=spec["label"], step_s=step_s if avg_min == 0 else 60.0 * avg_min)
    return lwp_gm2.rename("lwp_gm2"), info


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def describe(values_gm2: np.ndarray, step_s: float) -> Dict[str, float]:
    """n, approximate hours, mean, std (ddof = 1), median, 25th/75th percentiles and IQR of the finite values."""
    # phys_lwp is stored as float32; sums over ~10 million values need float64 (as seasonal.seasonal_stats uses)
    v = np.asarray(values_gm2, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "hours": 0.0, **{k: np.nan for k in ("mean", "std", "median", "q25", "q75", "iqr")}}
    q25, median, q75 = np.percentile(v, [25, 50, 75])  # linear interpolation (numpy default)
    return {"n": int(v.size), "hours": v.size * step_s / 3600.0, "mean": v.mean(),
            "std": v.std(ddof=1) if v.size > 1 else np.nan, "median": median, "q25": q25, "q75": q75,
            "iqr": q75 - q25}


def monthly_table(lwp_gm2: pd.Series, step_s: float) -> pd.DataFrame:
    """One row per calendar month (UTC); the two half-Februaries are separate months, labelled by their days."""
    month = lwp_gm2.index.to_period("M")
    rows = {}
    for period in pd.period_range("2023-02", "2024-02", freq="M"):
        sel = lwp_gm2[month == period]
        # label: "Feb 2023 (15-28)" for the partial months, "Mar 2023" otherwise
        first_day, last_day = max(period.start_time, pd.Timestamp(S.SEASONS["EPCAPE"][0][0])), \
            min(period.end_time.normalize(), pd.Timestamp(S.SEASONS["EPCAPE"][0][1]))
        partial = first_day.day != 1 or last_day.day != period.days_in_month
        label = period.strftime("%b %Y") + (f" ({first_day.day}-{last_day.day})" if partial else "")
        rows[label] = {"period": "month", "season": season_of(first_day), **describe(sel.to_numpy(), step_s)}
    return pd.DataFrame.from_dict(rows, orient="index")


def season_of(day: pd.Timestamp) -> str:
    """The sheet season holding `day` (the month's first day inside the campaign window; every month lies in
    exactly one season)."""
    for season in S.SEASON_ORDER[1:]:
        if S.season_mask([day], season)[0]:
            return season
    raise ValueError(f"{day:%d %b %Y} is in no season")


def seasonal_table(lwp_gm2: pd.Series, months: pd.DataFrame, step_s: float) -> pd.DataFrame:
    """One row per sheet season: statistics over all of its samples, plus the mean of its monthly means."""
    rows = {}
    for season in S.SEASON_ORDER:
        sel = lwp_gm2[S.season_mask(lwp_gm2.index, season)]
        in_season = months if season == "EPCAPE" else months[months["season"] == season]
        rows[season] = {"period": "season", "season": season, **describe(sel.to_numpy(), step_s),
                        "mean_of_monthly_means": in_season["mean"].mean(),
                        "months": len(in_season)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_min(text: str, retrieval: str) -> Optional[float]:
    """--lwp-min-gm2: a number, 'none' (no lower limit) or 'default' (the notebook's value for the retrieval)."""
    text = text.strip().lower()
    if text == "default":
        return RETRIEVALS[retrieval]["min_gm2"]
    return None if text == "none" else float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retrieval", choices=sorted(RETRIEVALS), default="v2",
                        help="v2 = MWRRET v2 (table row 10, default); v1 = MWRRET v1")
    parser.add_argument("--include-rain", action="store_true",
                        help="keep samples near rain (the notebook's 'including rain' line)")
    parser.add_argument("--lwp-min-gm2", default="default",
                        help="keep samples with LWP > this (g m-2), or 'none'; default = the notebook's "
                             "(0 for v2, none for v1)")
    parser.add_argument("--rain-window-min", type=float, default=LWP_RAIN_WINDOW_MIN,
                        help="rain screen half-width in minutes (default 30)")
    parser.add_argument("--avg-min", type=float, default=AVG_PERIOD_LWP_MIN,
                        help="averaging window in minutes before the statistics (default 0 = none)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "lwp_monthly",
                        help="output folder (default: lwp_monthly/ next to this script)")
    args = parser.parse_args()
    min_gm2 = parse_min(args.lwp_min_gm2, args.retrieval)  # depends on --retrieval, so resolved here
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 20)

    # --- 1. The sample, exactly as in notebook Section 3 ----------------------------------------------------------
    lwp_gm2, info = lwp_sample_gm2(args.retrieval, include_rain=args.include_rain, min_gm2=min_gm2,
                                   rain_window_min=args.rain_window_min, avg_min=args.avg_min)
    rain_text = "including rain" if args.include_rain else f"rain-free (±{args.rain_window_min:g} min)"
    min_text = "no lower LWP limit" if min_gm2 is None else f"LWP > {min_gm2:g} g m-2"
    avg_text = "no averaging window" if args.avg_min == 0 else f"{args.avg_min:g}-min means"
    print(f"{info['label']}: phys_lwp, QC = 0, cloudy scenes, {min_text}, {rain_text}, {avg_text}")
    for step in info["qc"]:  # how many samples each screening step kept
        print(f"  {step['variable']:<45s} kept {step['n_kept']:>12,d} of {step['n_finite']:>12,d}")
    print(f"  values near rain: {100 * info['near_rain_fraction']:.1f} %"
          + ("" if args.include_rain else " (removed)"))
    print(f"  in the table: {len(lwp_gm2):,} values; sampling interval {info['step_s']:.2f} s")

    # --- 2. Monthly and seasonal statistics ---------------------------------------------------------------------
    months = monthly_table(lwp_gm2, info["step_s"])
    seasons = seasonal_table(lwp_gm2, months, info["step_s"])
    table = pd.concat([months, seasons])
    table.index.name = "month / season"
    cols = ["period", "season", "n", "hours", "mean", "std", "median", "q25", "q75", "iqr",
            "mean_of_monthly_means", "months"]
    table = table[cols]
    print("\nLWP (g m-2), per month and per season:")
    print(table.drop(columns=["period"]).round(1).to_string())

    # --- 3. Check: seasonal mean ± std against row 10 of the table (MWRRET v2, rain-free, no window) -----------
    ours = [S.format_pm(m, s) for m, s in zip(seasons["mean"], seasons["std"])]
    print("\nSeasonal mean ± std, this script vs table row 10 (MWRRET v2, rain-free ±30 min, no averaging window):")
    print(pd.DataFrame({"this script": ours, "table row 10": TABLE_ROW10}, index=S.SEASON_ORDER).T.to_string())

    # --- 4. Write the table -------------------------------------------------------------------------------------
    tag = (f"mwrret{args.retrieval[-1]}_" + ("with_rain" if args.include_rain else f"rain_free_{args.rain_window_min:g}min")
           + ("" if min_gm2 is None else f"_lwp_gt_{min_gm2:g}") + ("" if args.avg_min == 0 else f"_avg{args.avg_min:g}min"))
    csv_path = args.out / f"lwp_monthly_seasonal_{tag}.csv"
    xlsx_path = args.out / f"lwp_monthly_seasonal_{tag}.xlsx"
    table.to_csv(csv_path)
    about = pd.DataFrame({"value": {
        "retrieval": info["label"],
        "QC": "qc_phys_lwp = 0" + (" and phys_qc_flag = 0" if args.retrieval == "v2" else ""),
        "cloudy scenes": ("ceilometer detection_status 1-3 and cbh_detected > 0" if args.retrieval == "v2"
                          else "qc_cloud_base_height = 0 and valid cloud_base_height"),
        "LWP limit": min_text,
        "rain": rain_text + " at the Pier Parsivel (LD M1, precip_rate QC = 0, rate > 0)",
        "averaging": avg_text,
        "months": "calendar months, UTC; Feb 2023 (15-28) and Feb 2024 (1-14) separate",
        "seasons": "sheet seasons: Spring AMJ23, Summer JAS23, Fall OND23, Winter 15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024",
        "n / hours": "values passing the filters / n x the median sampling interval (approximate time covered)",
        "mean, std": "over all values (ddof = 1); seasons reproduce notebook Section 3",
        "median, q25, q75, iqr": "numpy.percentile, linear interpolation; iqr = q75 - q25",
        "mean_of_monthly_means": "seasons: plain average of the season's monthly means (each month counts once)",
        "units": "g m-2",
        "script": Path(__file__).name,
        "git revision": git_revision(),
    }})
    with pd.ExcelWriter(xlsx_path) as xl:
        table.round({c: 1 for c in cols if c not in ("period", "season", "n", "months")}).to_excel(xl, sheet_name="LWP")
        about.to_excel(xl, sheet_name="About")
    print(f"\nWrote {csv_path.name} and {xlsx_path.name} to {args.out}")


if __name__ == "__main__":
    main()
