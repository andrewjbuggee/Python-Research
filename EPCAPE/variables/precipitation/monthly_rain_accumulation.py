#!/usr/bin/env python
"""Monthly and seasonal rain accumulation at EPCAPE from three disdrometers.

Rebuilds the numbers behind Dié Wang's monthly accumulation figure (panel b,
the bars) for the BAMS overview paper (Lynn Russell's request, 2026-10-12).
Adds what her code does not track: how many days each instrument actually
recorded in each month, so missing data can be noted next to each total.

Instruments (names follow Dié's figure)
  LD_M1    laser disdrometer, Scripps Pier       LDQUANTS VAP, epcldquantsM1.c1
  LD_S1    laser disdrometer, Mt. Soledad        LDQUANTS VAP, epcldquantsS2.c1
           (Dié calls it S1; ARM's facility code for Mt. Soledad is S2)
  VDIS_M1  2-D video disdrometer, Scripps Pier   VDISQUANTS VAP, epcvdisquantsM1.c1

Method
  1. rain_rate (mm h-1) at 1-min resolution. Both VAPs write a time stamp
     every minute and leave rain_rate NaN when no drops are seen, so NaN
     means "no rain" and is set to 0. Neither VAP has a qc_rain_rate field.
  2. Accumulation over a minute = rate (mm h-1) x (1/60) h. Monthly total =
     sum over the valid minutes of the month (equivalent to summing Dié's
     daily totals, which is what her panel (b) shows).
  3. Valid minutes, i.e. when the instrument was actually recording:
       LD_M1, LD_S1  a minute is valid when the raw Parsivel record
                     (epcldM1.b1 / epcldS2.b1, the instrument's own 1-min
                     output, qc_precip_rate == 0) has that minute AND the
                     LDQUANTS VAP has it. The VAP alone cannot show outages:
                     the raw record shows LD_M1 down 2023-10-15 .. 10-30.
       VDIS_M1       the VAP writes every minute even when the instrument
                     has stopped, so there is no record of uptime in the
                     data. Uses the outages given by the instrument mentor,
                     Zeen Zhu (BNL, email of 2026-08-12): instrument stopped
                     during the 2023-08-20..21 Hilary storm; no data
                     2023-10-15 .. 12-03; after 2023-10-15 VDIS under-reads
                     relative to both LDs and "should not be used".
                     --vdis-policy all keeps every VAP minute (Dié's
                     choice except that she zeroed all of Aug and Oct).
  4. A day "has data" when >= --min-day-coverage (default 0.9) of its 1440
     minutes are valid. Days missing = days in the period - days with data.
  5. Months are calendar months in UTC, clipped to the campaign window
     2023-02-15 .. 2024-02-14, so Feb 2023 (15-28) and Feb 2024 (1-14) are
     half months, reported separately as Lynn asked.

No gap filling: a total covers only the minutes recorded. Months with
missing days are therefore lower bounds; the coverage columns say by how
much the record is short.

Seasonal table (the BAMS table's columns; seasons as in
comparisons/seasonal_averages/seasonal.py): for each instrument
  total_mm          sum over the season's valid minutes
  days_with_data    / days_in_period
  daily_mean_mm_d ± daily_std_mm_d
                    mean ± sample std (ddof=1) of daily totals over days
                    with data, dry days included (= the season's rain per
                    day of record; times days in season = coverage-scaled
                    total)
  monthly_mean_mm ± monthly_std_mm
                    mean ± std (ddof=1) of the monthly totals. In winter the
                    two half Februaries are added together into one
                    28-day February first, so all months are whole. Only
                    months with >= --min-month-coverage (default 0.9) of
                    their days recorded enter; n_months says how many.
                    With n = 3 these stds are very uncertain.

Outputs (data/processed/derived/monthly_rain_accumulation/)
  monthly_rain_accumulation.csv  one row per month x instrument (long form)
  monthly_rain_accumulation_wide.csv  months as rows; mm and days per instrument
  seasonal_rain_accumulation.csv  one row per season x instrument
  daily_rain_accumulation.csv  daily totals and minute coverage
  monthly_rain_accumulation.png  bar chart with days recorded under each month

Examples (run from the EPCAPE folder)
  python variables/precipitation/monthly_rain_accumulation.py
  python variables/precipitation/monthly_rain_accumulation.py --vdis-policy all

Data: python download_data/download_arm.py rain_ldquants_M1 --workers 1
      (also rain_ldquants_S2, rain_vdis_M1, rain_ld_M1, rain_ld_S2), then
      python download_data/combine_product.py <product>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/variables/precipitation/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.analysis_tools.qc import has_qc, qc_is_zero  # noqa: E402
from EPCAPE.comparisons.seasonal_averages.seasonal import SEASONS, SEASON_ORDER  # noqa: E402
from EPCAPE.download_data.config import active_machine, campaign_dates, load_config  # noqa: E402

MIN_PER_DAY = 1440
MIN_PER_H = 60.0

# Instrument name -> (rain-rate VAP product, raw-record product or None).
# Product names are entries in config.yaml.
INSTRUMENTS: Dict[str, Tuple[str, object]] = {
    "LD_M1": ("rain_ldquants_M1", "rain_ld_M1"),
    "VDIS_M1": ("rain_vdis_M1", None),
    "LD_S1": ("rain_ldquants_S2", "rain_ld_S2"),
}
# Dié Wang's colours, so this figure reads like hers.
COLORS = {"LD_M1": "slategray", "VDIS_M1": "orange", "LD_S1": "royalblue"}

# VDIS periods not to use (inclusive UTC days), from Zeen Zhu's email of
# 2026-08-12 (ARM instrument mentor, BNL). The open-ended last period covers
# both the outage (to 2023-12-03) and the later under-reading.
VDIS_EXCLUDE_ZEEN: List[Tuple[str, str]] = [
    ("2023-08-20", "2023-08-21"),  # stopped during the Tropical Storm Hilary rain
    ("2023-10-15", "2024-02-14"),  # outage 10-15..12-03, then under-reads vs LDs
]

# NOAA, San Diego Lindbergh Field, 2023 Local Climatological Data (annual):
# March 2023 precipitation 3.97 in (Lynn's email of 2026-10-08;
# https://www.ncei.noaa.gov/pub/access/cebrequests/2023lcdannual/01202313SAN.pdf).
NOAA_SAN_MAR2023_MM = 3.97 * 25.4


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def minute_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every minute from `start` 00:00 to `end` 23:59 UTC (campaign window)."""
    return pd.date_range(start, end + pd.Timedelta(days=1), freq="1min", inclusive="left")


def vap_rate(product: str, grid: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """Rain rate (mm h-1) from a QUANTS VAP on the minute grid.

    Returns (rate, present). `present` is True where the VAP has a time stamp
    for that minute. NaN rain_rate at a present minute means no drops -> 0.
    Time stamps are floored to the minute (they are on the minute already;
    this only guards against sub-second offsets)."""
    ds = load_product(product)
    rate = ds["rain_rate"].to_series().astype(float)
    rate.index = rate.index.floor("1min")
    rate = rate[~rate.index.duplicated(keep="first")]
    present = pd.Series(grid.isin(rate.index), index=grid)
    rate = rate.fillna(0.0).reindex(grid)  # NaN now = minute absent from the VAP
    if (rate < 0).any():
        raise ValueError(f"{product}: negative rain_rate found")
    return rate, present


def raw_record_present(product: str, grid: pd.DatetimeIndex) -> pd.Series:
    """True for minutes the raw Parsivel record has with qc_precip_rate == 0.

    The raw LD datastream (epcld*.b1) is the instrument's own 1-min output,
    so a missing minute there means the instrument was not recording."""
    ds = load_product(product)
    ok = qc_is_zero(ds, "precip_rate").to_series() if has_qc(ds, "precip_rate") else None
    times = ds["time"].to_index().floor("1min")
    keep = np.ones(times.size, bool) if ok is None else ok.to_numpy()
    good = pd.Index(times[keep]).unique()
    return pd.Series(grid.isin(good), index=grid)


def instrument_minutes(name: str, grid: pd.DatetimeIndex, vdis_policy: str) -> pd.DataFrame:
    """Per-minute frame for one instrument: rate_mm_h (NaN where not valid), valid."""
    vap_product, raw_product = INSTRUMENTS[name]
    rate, valid = vap_rate(vap_product, grid)
    if raw_product is not None:
        valid &= raw_record_present(raw_product, grid)
    if name.startswith("VDIS") and vdis_policy == "zeen":
        for first, last in VDIS_EXCLUDE_ZEEN:
            valid &= ~((grid >= pd.Timestamp(first)) & (grid < pd.Timestamp(last) + pd.Timedelta(days=1)))
    return pd.DataFrame({"rate_mm_h": rate.where(valid), "valid": valid})


# ---------------------------------------------------------------------------
# daily and monthly accumulation
# ---------------------------------------------------------------------------
def daily_table(minutes: pd.DataFrame, min_day_coverage: float) -> pd.DataFrame:
    """Daily accumulation (mm) over valid minutes, minute coverage, has_data flag."""
    mm = (minutes["rate_mm_h"] / MIN_PER_H).resample("1D").sum(min_count=0)
    cov = minutes["valid"].resample("1D").sum() / MIN_PER_DAY
    out = pd.DataFrame({"accum_mm": mm, "coverage": cov})
    out["has_data"] = out["coverage"] >= min_day_coverage
    return out


def month_label(t: pd.Timestamp) -> str:
    return f"{t:%Y-%m}"


def monthly_table(daily: pd.DataFrame, minutes: pd.DataFrame, rain_day_mm: float) -> pd.DataFrame:
    """One row per calendar month (UTC) of the campaign window.

    accum_mm is summed over every valid minute (partial days included), so it
    is consistent with the minute coverage column."""
    month = daily.index.to_period("M")
    mm_min = (minutes["rate_mm_h"] / MIN_PER_H).groupby(minutes.index.to_period("M")).sum(min_count=0)
    valid_min = minutes["valid"].groupby(minutes.index.to_period("M")).sum()
    g = daily.groupby(month)
    out = pd.DataFrame(
        {
            "first_day": g.apply(lambda d: d.index.min().date()),
            "last_day": g.apply(lambda d: d.index.max().date()),
            "days_in_period": g.size(),
            "days_with_data": g["has_data"].sum().astype(int),
            # rain days among days with data (daily total >= rain_day_mm)
            "rain_days": g.apply(lambda d: int(((d["accum_mm"] >= rain_day_mm) & d["has_data"]).sum())),
            "accum_mm": mm_min,
            "minute_coverage": valid_min / (g.size() * MIN_PER_DAY),
        }
    )
    out["days_missing"] = out["days_in_period"] - out["days_with_data"]
    out["missing_dates"] = g.apply(lambda d: compress_dates(d.index[~d["has_data"]]))
    # A month with no valid minutes at all has no measurement, not 0 mm.
    out.loc[valid_min == 0, "accum_mm"] = np.nan
    out.index = [month_label(p.to_timestamp()) for p in out.index]
    out.index.name = "month"
    return out


def compress_dates(days: pd.DatetimeIndex) -> str:
    """'2023-10-15..10-30, 2023-11-02' style list of (runs of) missing days."""
    if len(days) == 0:
        return ""
    runs, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if d - prev == pd.Timedelta(days=1):
            prev = d
            continue
        runs.append((start, prev))
        start = prev = d
    runs.append((start, prev))
    return ", ".join(f"{a:%Y-%m-%d}" if a == b else f"{a:%Y-%m-%d}..{b:%m-%d}" for a, b in runs)


# ---------------------------------------------------------------------------
# seasons
# ---------------------------------------------------------------------------
def in_season(index: pd.DatetimeIndex, season: str) -> np.ndarray:
    mask = np.zeros(len(index), bool)
    for first, last in SEASONS[season]:
        mask |= (index >= pd.Timestamp(first)) & (index < pd.Timestamp(last) + pd.Timedelta(days=1))
    return mask


def seasonal_table(daily: pd.DataFrame, minutes: pd.DataFrame, min_month_coverage: float) -> pd.DataFrame:
    """Seasonal statistics for one instrument (see module docstring)."""
    rows = {}
    for season in SEASON_ORDER:
        d = daily[in_season(daily.index, season)]
        m = minutes[in_season(minutes.index, season)]
        recorded = d.loc[d["has_data"], "accum_mm"]

        # Monthly totals of whole months; Feb 2023 + Feb 2024 merged into one
        # 28-day February (only the campaign/winter seasons hold both halves).
        cal_month = d.index.month
        key = np.where(cal_month == 2, "Feb", d.index.strftime("%Y-%m"))
        per_month = d.groupby(key).agg(accum_mm=("accum_mm", "sum"), recorded=("has_data", "mean"),
                                       days=("has_data", "size"))
        per_month["complete"] = per_month["recorded"] >= min_month_coverage
        if season not in ("EPCAPE", "Winter"):
            per_month = per_month.drop(index="Feb", errors="ignore")
        # A merged February counts only when both halves are present.
        if "Feb" in per_month.index and per_month.loc["Feb", "days"] < 28:
            per_month = per_month.drop(index="Feb")
        whole = per_month.loc[per_month["complete"], "accum_mm"]

        rows[season] = {
            "total_mm": float((m["rate_mm_h"] / MIN_PER_H).sum()) if m["valid"].any() else np.nan,
            "days_in_period": int(len(d)),
            "days_with_data": int(d["has_data"].sum()),
            "minute_coverage": float(m["valid"].mean()),
            "daily_mean_mm_d": recorded.mean() if recorded.size else np.nan,
            "daily_std_mm_d": recorded.std(ddof=1) if recorded.size > 1 else np.nan,
            "monthly_mean_mm": whole.mean() if whole.size else np.nan,
            "monthly_std_mm": whole.std(ddof=1) if whole.size > 1 else np.nan,
            "n_months": int(whole.size),
            "months_used": ", ".join(whole.index),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "season"
    return out


# ---------------------------------------------------------------------------
# Dié's method, for checking against her figure
# ---------------------------------------------------------------------------
def die_monthly_mm(vap_product: str, zero_months: Tuple[int, ...] = ()) -> pd.Series:
    """Monthly totals computed as in Dié Wang's disdro_epc.ipynb.

    Her code: rain_rate / 60 summed per UTC day (NaN skipped), daily sums
    grouped by month. VDIS months 8 and 10 set to 0. Monthly total here = sum
    of those daily sums; no coverage screening."""
    rate = load_product(vap_product)["rain_rate"].to_series().astype(float)
    if zero_months:
        rate[rate.index.month.isin(zero_months)] = 0.0
    daily = (rate / MIN_PER_H).resample("1D").sum()
    out = daily.groupby(daily.index.to_period("M")).sum()
    out.index = [month_label(p.to_timestamp()) for p in out.index]
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def plot_monthly(monthly: Dict[str, pd.DataFrame], path: Path, vdis_policy: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(monthly)
    months = list(monthly[names[0]].index)
    x = np.arange(len(months))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for k, name in enumerate(names):
        m = monthly[name]
        off = (k - 1) * width
        full = m["days_missing"] == 0
        ax.bar(x[full] + off, m["accum_mm"][full], width, color=COLORS[name], edgecolor="k", lw=0.5, label=name)
        # Months with missing days: hatched, since the total is a lower bound.
        part = ~full & m["accum_mm"].notna()
        ax.bar(x[part] + off, m["accum_mm"][part], width, color=COLORS[name], edgecolor="k", lw=0.5,
               hatch="///", alpha=0.6)
        # Days recorded / days in period, one row per instrument under the axis
        for i, mon in enumerate(months):
            ax.text(x[i], -0.17 - 0.06 * k, f"{int(m.loc[mon, 'days_with_data'])}/{int(m.loc[mon, 'days_in_period'])}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30, ha="right")
    ax.set_ylabel("Monthly rainfall accumulation (mm)")
    ax.legend(loc="upper right", title="hatched = days missing")
    ax.set_title(f"EPCAPE monthly rain accumulation; numbers under axis = days recorded / days "
                 f"(VDIS policy: {vdis_policy})", fontsize=9, loc="left")
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vdis-policy", choices=["zeen", "all"], default="zeen",
                   help="zeen: drop VDIS 2023-08-20..21 and from 2023-10-15 (default); all: keep every VAP minute")
    p.add_argument("--min-day-coverage", type=float, default=0.9,
                   help="fraction of a day's minutes that must be valid for the day to count as recorded")
    p.add_argument("--min-month-coverage", type=float, default=0.9,
                   help="fraction of a month's days that must be recorded for it to enter the monthly mean +- std")
    p.add_argument("--rain-day-mm", type=float, default=0.1, help="daily total (mm) that makes a rain day")
    p.add_argument("--out", type=Path, default=None, help="output folder")
    args = p.parse_args(argv)

    cfg = load_config()
    c_start, c_end = campaign_dates(cfg)
    grid = minute_grid(pd.Timestamp(c_start), pd.Timestamp(c_end))
    out_dir = args.out or active_machine(cfg).processed_dir() / "derived" / "monthly_rain_accumulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly, seasonal, daily_all = {}, {}, {}
    for name in INSTRUMENTS:
        minutes = instrument_minutes(name, grid, args.vdis_policy)
        daily = daily_table(minutes, args.min_day_coverage)
        monthly[name] = monthly_table(daily, minutes, args.rain_day_mm)
        seasonal[name] = seasonal_table(daily, minutes, args.min_month_coverage)
        daily_all[name] = daily

    # ---- files -------------------------------------------------------------
    long = pd.concat(monthly, names=["instrument"]).reset_index()
    long.insert(2, "vdis_policy", args.vdis_policy)
    long.to_csv(out_dir / "monthly_rain_accumulation.csv", index=False, float_format="%.2f")

    wide = pd.concat(
        {n: m[["accum_mm", "days_with_data", "days_in_period"]] for n, m in monthly.items()}, axis=1
    )
    wide.columns = [f"{n}_{c}" for n, c in wide.columns]
    wide.to_csv(out_dir / "monthly_rain_accumulation_wide.csv", float_format="%.2f")

    seas = pd.concat(seasonal, names=["instrument"]).reset_index()
    seas.to_csv(out_dir / "seasonal_rain_accumulation.csv", index=False, float_format="%.3f")

    daily_long = pd.concat(daily_all, names=["instrument", "date"]).reset_index()
    daily_long.to_csv(out_dir / "daily_rain_accumulation.csv", index=False, float_format="%.3f")

    plot_monthly(monthly, out_dir / "monthly_rain_accumulation.png", args.vdis_policy)

    # ---- screen summary ------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\nMonthly accumulation (mm) and days recorded / days in period  [VDIS policy: {args.vdis_policy}]")
    show = pd.DataFrame(
        {n: [f"{a:6.1f}  {d:2d}/{t:2d}" if np.isfinite(a) else f"   n/a  {d:2d}/{t:2d}"
             for a, d, t in zip(m["accum_mm"], m["days_with_data"], m["days_in_period"])]
         for n, m in monthly.items()},
        index=monthly["LD_M1"].index,
    )
    print(show.to_string())

    print("\nMissing days (recorded < {:.0%} of minutes)".format(args.min_day_coverage))
    for n, m in monthly.items():
        for mon, row in m[m["days_missing"] > 0].iterrows():
            print(f"  {n:8s} {mon}: {row['days_missing']:2d} d  {row['missing_dates']}")

    print("\nSeasonal values")
    cols = ["total_mm", "days_with_data", "days_in_period", "daily_mean_mm_d", "daily_std_mm_d",
            "monthly_mean_mm", "monthly_std_mm", "n_months"]
    print(seas.set_index(["instrument", "season"])[cols].round(2).to_string())

    print("\nChecks")
    print(f"  NOAA San Diego airport, Mar 2023: {NOAA_SAN_MAR2023_MM:.1f} mm; "
          + ", ".join(f"{n} {monthly[n].loc['2023-03', 'accum_mm']:.1f}" for n in monthly))
    die = pd.DataFrame({
        "LD_M1": die_monthly_mm("rain_ldquants_M1"),
        "VDIS_M1": die_monthly_mm("rain_vdis_M1", zero_months=(8, 10)),
        "LD_S1": die_monthly_mm("rain_ldquants_S2"),
    })
    mine = pd.DataFrame({n: m["accum_mm"] for n, m in monthly.items()})
    both = pd.concat({"Dié method": die, "this script": mine}, axis=1).round(1)
    print("  Dié's method (no coverage screen, VDIS Aug+Oct zeroed) vs this script, mm:")
    print(both.to_string())
    die.to_csv(out_dir / "monthly_rain_accumulation_die_method.csv", float_format="%.2f")
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
