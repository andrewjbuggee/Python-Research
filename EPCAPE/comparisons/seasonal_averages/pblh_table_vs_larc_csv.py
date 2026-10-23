#!/usr/bin/env python
"""Seasonal PBL height: the table's row 15 calculation vs the LaRC CSV behind BAMS Fig. 4 panel A.

Why: row 15 of the seasonal-averages table (EPCAPE_SeasonalAveragesTable.xlsx, sheet Cloud&MetQuantities, "Boundary
Layer Height, epcpblhtthermoM1.c1 (PBLHTTHERMO) + KAZR-ARSCL") is computed in check_seasonal_averages.ipynb
Section 5 from ARM's PBLHTTHERMO VAP. BAMS Fig. 4 panel A (EPCAPE_BAMS_Fig4_AJB_v3_1.m) instead plots PBLH_THERMO
and PBLH_MPL from Seethala Chellappan's LaRC file, EPCAPE_CTH_CBH_PBLH_SChellappan_LaRC.csv, as monthly boxes with no
further filter. This script puts the seasonal statistics of both side by side and shows where they part.

Data
  Table row 15 (notebook Section 5; same code as pblh_cloud_top_limit.py)
      pblh_thermo_M1   ARM PBLHTTHERMO VAP, epcpblhtthermoM1.c1 (dod pblhtthermo-c1-1.0): pbl_height, one value
                       every 10 min, used as delivered (no QC field, no averaging)
      cloud_arscl_M1   KAZR-ARSCL: keep a 10-min value only when the 5-min bin starting at its time stamp holds a
                       4-s sample whose lowest-layer cloud top is <= 3500 m (row 15); also shown with 2000 m
  LaRC CSV (Seethala Chellappan, 14 Oct 2026 copy in data/processed/PBLH - Damao Zhang/; the file Fig. 4 reads)
      2,962 GOES-18 half-hourly times (time stamps hh:00:20 / hh:30:21, UTC; 15 Feb 2023 - 10 Feb 2024), each with
      GOES18_CTH, KAZR_CTH, CEIL_CBH, PBLH_THERMO and PBLH_MPL. Every row has all five, and every value in every
      column is < 2 km (max 1984-1991 m): the file holds only times when ALL of them are below 2 km.
      PBLH_THERMO and PBLH_MPL are reproduced EXACTLY (all 2,962 rows; checked below) as the mean of the three 10-min
      values of Damao Zhang's daily files (epcpblht_be_ml.YYYYMMDD.000000.nc, folder
      epcpblht_be_ml_daily_pblhtmpl_py; variables pblht_thermo and pblht_mpl, m) at the GOES time floored to
      10 min and the next two, i.e. the 10-min values stamped in [t, t + 30 min). The match also confirms that the
      CSV times are UTC.
      Damao's pblht_thermo is NOT the archived ARM c1 pbl_height: the two agree exactly at ~72 % of the shared 10-min
      times, and where they differ Damao's is lower by a median of ~40 m (printed below). Which processing is newer
      is not known here.
  Kavin's 1 Sep 2026 copy of the CSV (Annual Echo Plot folder; starts 1 Mar 2023, older PBLH_MPL processing): its
      PBLH_MPL column gives table row 16 and its PBLH_THERMO column gave the earlier "PBLH-Thermo" row (899 m).

Statistics: mean, std (ddof = 1), median, 25th/75th percentiles and n per sheet season (seasonal.SEASONS, UTC:
Spring AMJ23, Summer JAS23, Fall OND23, Winter 15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024, EPCAPE = 15 Feb 2023 -
14 Feb 2024). One sample = one 10-min value (row 15) or one GOES half-hour (CSV). Fig. 4 groups the same CSV values
by PST month and draws medians and quartiles; the seasonal means here use the table's UTC seasons.

Bridge rows (what separates row 15 from the CSV's PBLH_THERMO):
  "the same values as half-hour means, every half hour"   averaging only: row 15's 10-min values averaged over the
      CSV's half-hour windows [hh:00, hh:30), [hh:30, hh+1:00), every half hour holding one of them
  "ARM c1 averaged like the CSV, at the CSV times"   ARM pbl_height, mean of the 10-min values in [t, t + 30 min)
      at the 2,962 CSV times. Same sampling and averaging as the CSV, ARM's product: its difference from the CSV
      row is the product version (Damao's thermo vs ARM c1); its difference from row 15 is the sampling (GOES-18
      half-hours with every height < 2 km, vs every 10-min value under a KAZR top <= 3.5 km) and the 30-min averaging.

Outputs (in pblh_table_vs_larc/, next to this script; --out chooses another folder):
  pblh_table_vs_larc_seasonal.csv    every line of the printed tables: mean, std, median, q25, q75, n per season
  pblh_table_vs_larc_seasonal.xlsx   mean ± std, median (q25-q75) and n sheets, plus About

Example (from the Python-Research folder)
  python EPCAPE/comparisons/seasonal_averages/pblh_table_vs_larc_csv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.analysis_tools.derived import git_revision  # noqa: E402
from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.comparisons.seasonal_averages.pblh_cloud_top_limit import (  # noqa: E402
    low_cloud_flag_5min,
    match_to_pblh,
)
from EPCAPE.download_data.config import active_machine  # noqa: E402

# ---------------------------------------------------------------------------
# Inputs and settings
# ---------------------------------------------------------------------------
DAMAO_DIR = active_machine().processed_dir() / "PBLH - Damao Zhang"
LARC_CSV = DAMAO_DIR / "EPCAPE_CTH_CBH_PBLH_SChellappan_LaRC.csv"  # 14 Oct 2026 copy, read by Fig. 4 v3_1
DAMAO_DAILY_DIR = DAMAO_DIR / "epcpblht_be_ml_daily_pblhtmpl_py"  # Damao's 10-min files behind the CSV
KAVIN_LARC_CSV = Path("~/Documents/MATLAB/Matlab-Research/EPCAPE/Kavin_code_for_BAMS/Annual Echo Plot/"
                      "EPCAPE_CTH_CBH_PBLH_SChellappan_LaRC.csv").expanduser()  # Kavin's 1 Sep 2026 copy (optional)
CSV_WINDOW_STEPS = 3  # the CSV value = mean of the 10-min values at floor10(t), +10 and +20 min
LOW_CLOUD_LIMITS_M = (3500.0, 2000.0)  # row 15's KAZR-ARSCL cloud-top limit, and the lower one tested

# The sheet's own values (EPCAPE_SeasonalAveragesTable.xlsx, columns E-I = EPCAPE, Spring, Summer, Fall, Winter).
# Row 16's EPCAPE cell is typed "790 ±.274" in the sheet.
SHEET = {
    "Table row 15: PBLHTTHERMO + KAZR-ARSCL (sheet)": ("804 ± 411", "779 ± 290", "618 ± 234", "920 ± 472", "1114 ± 595"),
    "Table row 16: PBLH-MPL (sheet)": ("790 ± 274", "765 ± 255", "728 ± 248", "844 ± 313", "970 ± 288"),
    "Earlier sheet 'PBLH-Thermo (Zhang)' row (sheet)": ("899 ± 305", "886 ± 256", "764 ± 206", "1014 ± 343",
                                                        "1124 ± 442"),
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def read_larc(path: Path) -> pd.DataFrame:
    """The LaRC CSV, indexed by its (UTC) time stamps; heights in m."""
    df = pd.read_csv(path, parse_dates=["Time"]).set_index("Time").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate time stamps")
    return df


def read_damao_daily(folder: Path) -> pd.DataFrame:
    """Damao Zhang's daily PBL-height files: pblht_thermo and pblht_mpl (m), 10-min time stamps (UTC).

    Each time stamp's time_bounds span [t - 15 min, t + 15 min]. Missing values are written as -9999 km x 1000 =
    -9.999e6 m; every negative value is set to NaN."""
    parts = []
    for f in sorted(folder.glob("epcpblht_be_ml.*.nc")):
        # one file at a time: base_time differs between files, so open_mfdataset refuses to combine them
        with xr.open_dataset(f) as ds:
            parts.append(ds[["pblht_thermo", "pblht_mpl"]].to_dataframe())
    if not parts:
        raise FileNotFoundError(f"no epcpblht_be_ml.*.nc files in {folder}")
    df = pd.concat(parts).astype(np.float64)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.where(df >= 0)


def mean_like_larc(series_m: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    """At each CSV time t: mean of the 10-min values stamped floor10(t), +10 and +20 min (NaN values skipped), the
    recipe that reproduces the CSV's PBLH_THERMO and PBLH_MPL from Damao's files."""
    start = times.floor("10min")
    vals = np.column_stack([series_m.reindex(start + pd.Timedelta(minutes=10 * k)).to_numpy()
                            for k in range(CSV_WINDOW_STEPS)])  # (n_times, 3)
    n_ok = np.isfinite(vals).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):  # a window with no value -> NaN
        return np.where(n_ok > 0, np.nansum(vals, axis=1) / n_ok, np.nan)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def season_stats(values_m: pd.Series) -> pd.DataFrame:
    """mean, std (ddof = 1), median, q25, q75, n per sheet season (rows = S.SEASON_ORDER)."""
    v = pd.Series(values_m).astype(np.float64)
    rows = {}
    for season in S.SEASON_ORDER:
        x = v.to_numpy()[S.season_mask(v.index, season)]
        x = x[np.isfinite(x)]
        q25, median, q75 = np.percentile(x, [25, 50, 75]) if x.size else (np.nan,) * 3
        rows[season] = {"mean": x.mean() if x.size else np.nan, "std": x.std(ddof=1) if x.size > 1 else np.nan,
                        "median": median, "q25": q25, "q75": q75, "n": int(x.size)}
    return pd.DataFrame.from_dict(rows, orient="index")


def compare_versions(a_m: pd.Series, b_m: pd.Series) -> Dict[str, float]:
    """How two 10-min PBL-height records differ at their shared time stamps (b - a, m)."""
    j = pd.concat([a_m.rename("a"), b_m.rename("b")], axis=1).dropna()
    d = j["b"] - j["a"]
    same = np.isclose(d, 0.0, atol=0.01)
    return {"shared": len(j), "identical_pct": 100.0 * same.mean(), "median_diff_where_different_m": d[~same].median(),
            "mean_diff_all_m": d.mean(), "abs_diff_gt_100m_pct": 100.0 * (d.abs() > 100.0).mean()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "pblh_table_vs_larc",
                        help="output folder (default: pblh_table_vs_larc/ next to this script)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    lines: Dict[str, pd.DataFrame] = {}  # label -> season_stats table

    # --- 1. Table row 15: notebook Section 5 (ARM c1 pbl_height while KAZR-ARSCL sees a low cloud) --------------
    pblh_m, _ = Q.pblh_thermo_m(load_product("pblh_thermo_M1"))
    arscl = load_product("cloud_arscl_M1")
    for limit in LOW_CLOUD_LIMITS_M:
        cloudy = match_to_pblh(low_cloud_flag_5min(arscl, limit), pblh_m.index)
        tag = " = table row 15" if limit == 3500 else ""
        lines[f"ARM c1 pbl_height, 10-min values, KAZR lowest top <= {limit:g} m (notebook Sec. 5{tag})"] = \
            season_stats(pblh_m.where(cloudy))
        # averaging only: the same 10-min values as half-hour means (the CSV's [hh:00, hh:30) windows), every half
        # hour holding at least one of them; each half hour then weighs the same, however many values it holds
        lines[f"Bridge: the same values (KAZR top <= {limit:g} m) as half-hour means, every half hour"] = \
            season_stats(pblh_m.where(cloudy).resample("30min").mean().dropna())
    del arscl  # ~0.7 GB

    # --- 2. The LaRC CSV read by Fig. 4 (and Kavin's earlier copy, if present) -----------------------------------
    larc = read_larc(LARC_CSV)
    print(f"LaRC CSV {LARC_CSV.name} (14 Oct copy): {len(larc):,} rows, {larc.index.min()} to {larc.index.max()}; "
          f"rows with every column present: {int(larc.notna().all(axis=1).sum()):,}; column maxima (m): "
          + ", ".join(f"{c} {larc[c].max():.0f}" for c in larc.columns))
    lines["LaRC CSV (14 Oct, Fig. 4): PBLH_THERMO, every row"] = season_stats(larc["PBLH_THERMO"])
    lines["LaRC CSV (14 Oct, Fig. 4): PBLH_MPL, every row"] = season_stats(larc["PBLH_MPL"])
    if KAVIN_LARC_CSV.is_file():
        old = read_larc(KAVIN_LARC_CSV)
        print(f"Kavin's 1 Sep copy: {len(old):,} rows, {old.index.min()} to {old.index.max()}")
        lines["LaRC CSV (Kavin's 1 Sep copy): PBLH_THERMO, every row"] = season_stats(old["PBLH_THERMO"])
        lines["LaRC CSV (Kavin's 1 Sep copy): PBLH_MPL, every row"] = season_stats(old["PBLH_MPL"])
    else:
        print(f"Kavin's 1 Sep copy not found ({KAVIN_LARC_CSV}); its lines are skipped")

    # --- 3. Where the CSV's PBLH values come from: Damao's 10-min files, and how they differ from ARM c1 ---------
    damao = read_damao_daily(DAMAO_DAILY_DIR)
    print(f"\nDamao's daily files ({DAMAO_DAILY_DIR.name}): {len(damao):,} 10-min times, "
          f"{damao.index.min()} to {damao.index.max()}")
    for col, src in (("PBLH_THERMO", "pblht_thermo"), ("PBLH_MPL", "pblht_mpl")):
        rebuilt = mean_like_larc(damao[src], larc.index)
        exact = np.isclose(rebuilt, larc[col].to_numpy(), rtol=1e-4, atol=0.05)
        print(f"  CSV {col} = mean of Damao's {src} in [t, t + 30 min): {int(exact.sum()):,} of {len(larc):,} rows "
              f"reproduced exactly")
    v = compare_versions(pblh_m, damao["pblht_thermo"])
    print(f"  Damao's pblht_thermo vs ARM c1 pbl_height at {v['shared']:,} shared 10-min times: identical "
          f"{v['identical_pct']:.1f} %; where different, Damao - ARM median {v['median_diff_where_different_m']:.0f} m; "
          f"|difference| > 100 m at {v['abs_diff_gt_100m_pct']:.1f} %; mean difference over all shared times "
          f"{v['mean_diff_all_m']:.0f} m")

    # --- 4. Bridge: ARM c1 sampled and averaged like the CSV ----------------------------------------------------
    arm_like_csv = pd.Series(mean_like_larc(pblh_m, larc.index), index=larc.index)
    lines["Bridge: ARM c1 pbl_height averaged like the CSV, at the CSV times"] = season_stats(arm_like_csv)

    # --- 5. Tables ----------------------------------------------------------------------------------------------
    order = list(SHEET)[:1] + [k for k in lines if k.startswith("ARM c1")] + \
        [k for k in lines if k.startswith("Bridge")] + [k for k in lines if "PBLH_THERMO" in k] + \
        list(SHEET)[1:2] + [k for k in lines if "PBLH_MPL" in k] + list(SHEET)[2:]
    pm, med, n = {}, {}, {}
    for label in order:
        if label in SHEET:  # the sheet's text, as typed (corrected "±." in row 16)
            pm[label], med[label], n[label] = list(SHEET[label]), ["–"] * 5, ["–"] * 5
            continue
        st = lines[label]
        pm[label] = [S.format_pm(m, s) for m, s in zip(st["mean"], st["std"])]
        med[label] = [f"{m:.0f} [{a:.0f}–{b:.0f}]" for m, a, b in zip(st["median"], st["q25"], st["q75"])]
        n[label] = [f"{x:,}" for x in st["n"]]
    pm, med, n = (pd.DataFrame(d, index=S.SEASON_ORDER).T for d in (pm, med, n))
    print("\nPBL height (m), mean ± std per sheet season (UTC):")
    print(pm.to_string())
    print("\nMedian [25th-75th percentile] (m):")
    print(med.to_string())
    print("\nNumber of samples (10-min values for ARM c1 rows; GOES half-hours for the CSV and bridge rows):")
    print(n.to_string())

    # --- 6. Write -----------------------------------------------------------------------------------------------
    long = pd.concat({label: st for label, st in lines.items()}, names=["line", "season"]).reset_index()
    long.to_csv(args.out / "pblh_table_vs_larc_seasonal.csv", index=False)
    about = pd.DataFrame({"value": {
        "row 15 (ARM c1)": "epcpblhtthermoM1.c1 pbl_height, 10-min values as delivered, kept when the 5-min bin "
                           "starting at the time stamp holds a KAZR-ARSCL lowest-layer top <= the limit "
                           "(check_seasonal_averages.ipynb Section 5; pblh_cloud_top_limit.py)",
        "LaRC CSV": f"{LARC_CSV} ({len(larc):,} GOES-18 half-hours; every column < 2 km); every row used",
        "CSV PBLH values": f"mean of Damao Zhang's 10-min pblht_thermo / pblht_mpl ({DAMAO_DAILY_DIR.name}) at "
                           "floor10(t), +10, +20 min; reproduced exactly",
        "Damao thermo vs ARM c1": f"identical at {v['identical_pct']:.1f} % of {v['shared']:,} shared 10-min times; "
                                  f"mean difference {v['mean_diff_all_m']:.0f} m",
        "bridge row": "ARM c1 pbl_height averaged with the CSV recipe at the CSV times",
        "Kavin's copy": str(KAVIN_LARC_CSV) + ("" if KAVIN_LARC_CSV.is_file() else " (not found)"),
        "seasons": "sheet seasons, UTC (seasonal.SEASONS); CSV time stamps are UTC (exact match to Damao's files)",
        "statistics": "mean, std (ddof = 1), median, q25, q75 (numpy.percentile, linear), n",
        "script": Path(__file__).name,
        "git revision": git_revision(),
    }})
    xlsx = args.out / "pblh_table_vs_larc_seasonal.xlsx"
    with pd.ExcelWriter(xlsx) as xl:
        pm.to_excel(xl, sheet_name="mean ± std")
        med.to_excel(xl, sheet_name="median (IQR)")
        n.to_excel(xl, sheet_name="n")
        about.to_excel(xl, sheet_name="About")
    print(f"\nWrote pblh_table_vs_larc_seasonal.csv and {xlsx.name} to {args.out}")


if __name__ == "__main__":
    main()
