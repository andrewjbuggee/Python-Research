#!/usr/bin/env python
"""Seasonal cloud-top and cloud-base heights: the table and notebook vs the LaRC CSV behind BAMS Fig. 4 panel A.

Why: the seasonal-averages table (EPCAPE_SeasonalAveragesTable.xlsx, sheet Cloud&MetQuantities) gives Cloud Top
Height (row 13, "KAZR", Kavin) and Cloud Base Height (row 19, "Ceilometer", Kavin). check_seasonal_averages.ipynb
re-derives both from ARM data (Section 4, cloud top; Section 7, cloud base). BAMS Fig. 4 panel A instead plots
GOES18_CTH, KAZR_CTH and CEIL_CBH from Seethala Chellappan's LaRC file, EPCAPE_CTH_CBH_PBLH_SChellappan_LaRC.csv.
This script puts the seasonal statistics of all of them side by side, the companion of pblh_table_vs_larc_csv.py
(which does the two PBL-height columns of the same file).

Data
  Notebook calculations (the same functions, from quantities.py; settings AVG_PERIOD_CT = AVG_PERIOD_CBH = 0)
      cloud_arscl_M1  KAZR-ARSCL epcarsclkazr1kolliasM1.c1 (4 s, up to 10 hydrometeor layers, m above ground, no QC
                      field): Section 4 = top of the LOWEST layer, every 4-s sample with that top <= 3500 m
                      (Q.lowest_layer_top_m). Also shown with 2000 m, the CSV's height limit.
      cbh_ceil_M1     Pier ceilometer epcceilM1.b1 (16 s): Section 7 = first_cbh with qc_first_cbh = 0,
                      detection_status 1-3 and base <= 3500 m, every 16-s sample (Q.ceilometer_cbh_m). Also 2000 m.
  LaRC CSV (Seethala Chellappan; 14 Oct 2026 copy in data/processed/PBLH - Damao Zhang/, the file Fig. 4 reads)
      2,962 GOES-18 half-hourly times (hh:00:20 / hh:30:21 UTC, 15 Feb 2023 - 10 Feb 2024). Every row has all five
      columns, each < 2 km. GOES18_CTH is the satellite cloud-top height for the Pier (no ground-based equivalent
      here). The two ground-based columns were rebuilt from ARM data (checked below):
        CEIL_CBH  mean of the ceilometer first_cbh samples with qc_first_cbh = 0, detection_status 1-3 and base
                  < 2000 m in the half hour [hh:00, hh:30) or [hh:30, hh+1:00) that holds the GOES time:
                  98.6 % of rows exact.
        KAZR_CTH  same half hour; at each 4-s ARSCL sample, the HIGHEST layer top among the layers whose top is
                  < 2000 m; mean over the half hour: 82 % of rows exact, mean difference 0.1 m (the rest differ
                  by a few metres to ~100 m; perhaps a different ARSCL version, not checked). Note the definition:
                  the CSV's cloud top is the top of the highest layer below 2 km (a satellite-like view), the
                  notebook's is the top of the LOWEST layer.
  Kavin's 1 Sep 2026 copy of the CSV (Annual Echo Plot folder; starts 1 Mar 2023): its KAZR_CTH and CEIL_CBH give
      table rows 13 and 19 exactly in summer, fall and winter (sheet seasons, UTC); spring and the campaign value
      are within 4 m (cause not found: PST or local-time seasons, or MAM spring, do not reproduce them).

Bridge rows (what separates the notebook from the CSV, using ARM data only):
  "same samples (< 2 km) as half-hour means, every half hour"  averaging only: every half hour weighs the same,
                                                               however many cloudy samples it holds
  "lowest-layer top < 2 km, half-hour mean at the CSV times"   + the CSV's choice of half hours (cloud top only)
  "CSV recipe from ARM data, at the CSV times"                 the rebuilt KAZR_CTH / CEIL_CBH above
  So: notebook (every sample, <= 3.5 km) -> same < 2 km -> half-hour means -> only the CSV's half hours -> CSV.

Statistics: mean, std (ddof = 1), median, 25th/75th percentiles and n per sheet season (seasonal.SEASONS, UTC:
Spring AMJ23, Summer JAS23, Fall OND23, Winter 15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024, EPCAPE = 15 Feb 2023 -
14 Feb 2024). One sample = one 4-s ARSCL sample or 16-s ceilometer sample (notebook rows) or one GOES half-hour
(CSV and bridge rows). Fig. 4 groups the CSV values by PST month and shows medians and quartiles.

Outputs (in cloud_heights_vs_larc/, next to this script; --out chooses another folder):
  cloud_heights_vs_larc_seasonal.csv    every line: mean, std, median, q25, q75, n per season
  cloud_heights_vs_larc_seasonal.xlsx   mean ± std, median (q25-q75) and n sheets, plus About

Example (from the Python-Research folder)
  python EPCAPE/comparisons/seasonal_averages/cloud_heights_table_vs_larc_csv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
from EPCAPE.comparisons.seasonal_averages.pblh_table_vs_larc_csv import (  # noqa: E402
    KAVIN_LARC_CSV,
    LARC_CSV,
    read_larc,
    season_stats,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NOTEBOOK_LIMITS_M = (3500.0, 2000.0)  # notebook LOW_CLOUD_MAX_M (row 13/19 checks), and the CSV's 2-km limit
CSV_LIMIT_M = 2000.0  # every height in the CSV is < 2 km
CSV_WINDOW = "30min"  # CSV KAZR_CTH / CEIL_CBH = mean over the half hour [hh:00, hh:30) or [hh:30, hh+1:00)

# The sheet's own values (EPCAPE_SeasonalAveragesTable.xlsx, columns E-I = EPCAPE, Spring, Summer, Fall, Winter)
SHEET = {
    "Table row 13: Cloud Top Height, KAZR (sheet)": ("920 ± 380", "890 ± 330", "774 ± 353", "1057 ± 416",
                                                     "1192 ± 397"),
    "Table row 19: Cloud Base Height, Ceilometer (sheet)": ("661 ± 316", "619 ± 270", "554 ± 254", "836 ± 389",
                                                            "929 ± 362"),
}


# ---------------------------------------------------------------------------
# The CSV's recipes, from ARM data
# ---------------------------------------------------------------------------
def highest_top_below_m(arscl: xr.Dataset, limit_m: float) -> pd.Series:
    """At each 4-s ARSCL sample: the highest layer top among the layers whose top is > 0 and < limit_m (m above
    ground). Samples with no such layer are dropped. Layers above the limit are ignored, so a low deck under cirrus
    still gives the low deck's top (the CSV's KAZR_CTH recipe, reproduced at 82 % of rows)."""
    top = arscl["cloud_layer_top_height"]
    layer_dim = [d for d in top.dims if d != "time"][0]
    vals = top.transpose("time", layer_dim).values.astype(np.float64)  # (n_time, n_layer); -1 = empty slot
    keep = np.isfinite(vals) & (vals > 0) & (vals < limit_m)
    hi = np.where(keep, vals, -np.inf).max(axis=1)  # -inf where no layer qualifies
    s = pd.Series(hi, index=pd.DatetimeIndex(arscl["time"].values), name="highest_top_below_m")
    return s[np.isfinite(s)]


def half_hour_at(values_m: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    """Mean of values_m over the half hour that holds each time: [floor30(t), floor30(t) + 30 min). NaN if the half
    hour holds no value. resample() bins are labelled by their start, so they line up with floor30(t)."""
    means = values_m.resample(CSV_WINDOW).mean()
    return means.reindex(times.floor(CSV_WINDOW)).to_numpy()


def recipe_check(rebuilt_m: np.ndarray, csv_m: pd.Series) -> Dict[str, float]:
    """How well a rebuilt column matches the CSV's: share of rows equal to 1e-3 m, and the differences (m)."""
    d = rebuilt_m - csv_m.to_numpy()
    exact = np.isclose(rebuilt_m, csv_m.to_numpy(), rtol=1e-6, atol=1e-3)
    return {"exact_pct": 100.0 * exact.mean(), "mean_diff_m": float(np.nanmean(d)),
            "p05_diff_m": float(np.nanpercentile(d, 5)), "p95_diff_m": float(np.nanpercentile(d, 95)),
            "rows_without_data": int(np.isnan(rebuilt_m).sum())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cloud_heights_vs_larc",
                        help="output folder (default: cloud_heights_vs_larc/ next to this script)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
    groups: Dict[str, Dict[str, pd.DataFrame]] = {"Cloud top height": {}, "Cloud base height": {}}
    checks = {}

    larc = read_larc(LARC_CSV)
    kavin = read_larc(KAVIN_LARC_CSV) if KAVIN_LARC_CSV.is_file() else None
    print(f"LaRC CSV (14 Oct copy): {len(larc):,} rows, {larc.index.min()} to {larc.index.max()}"
          + ("" if kavin is not None else f"; Kavin's copy not found ({KAVIN_LARC_CSV})"))
    cth = groups["Cloud top height"]
    cbh = groups["Cloud base height"]

    # --- 1. Cloud top: notebook Section 4 (lowest-layer top), and the CSV recipe from ARSCL -----------------------
    arscl = load_product("cloud_arscl_M1")
    for limit in NOTEBOOK_LIMITS_M:
        top_m, _ = Q.lowest_layer_top_m(arscl, max_top_m=limit, window_min=0, cloud_sources=None)
        tag = " (notebook Sec. 4 = check of row 13)" if limit == 3500 else ""
        cth[f"ARSCL lowest-layer top <= {limit:g} m, every 4-s sample{tag}"] = season_stats(top_m)
    # bridge 1: the notebook's definition, sampled and averaged like the CSV (< 2 km, half hours at the CSV times)
    lowest_m, _ = Q.lowest_layer_top_m(arscl, max_top_m=np.inf, window_min=0, cloud_sources=None)
    lowest_m = lowest_m[lowest_m < CSV_LIMIT_M]
    # averaging only: the same samples as the 2-km line above, as half-hour means, EVERY half hour with a value
    cth["Bridge: same samples (< 2 km) as half-hour means, every half hour"] = \
        season_stats(lowest_m.resample(CSV_WINDOW).mean().dropna())
    cth["Bridge: ARSCL lowest-layer top < 2 km, half-hour mean at the CSV times"] = \
        season_stats(pd.Series(half_hour_at(lowest_m, larc.index), index=larc.index))
    # bridge 2: the CSV's own recipe (highest top below 2 km)
    kazr_rebuilt = half_hour_at(highest_top_below_m(arscl, CSV_LIMIT_M), larc.index)
    checks["KAZR_CTH"] = recipe_check(kazr_rebuilt, larc["KAZR_CTH"])
    cth["Bridge: CSV recipe from ARSCL (highest top below 2 km), at the CSV times"] = \
        season_stats(pd.Series(kazr_rebuilt, index=larc.index))
    del arscl, lowest_m  # ~0.7 GB
    cth["LaRC CSV (14 Oct, Fig. 4): KAZR_CTH"] = season_stats(larc["KAZR_CTH"])
    if kavin is not None:
        cth["LaRC CSV (Kavin's 1 Sep copy): KAZR_CTH"] = season_stats(kavin["KAZR_CTH"])
    cth["LaRC CSV (14 Oct, Fig. 4): GOES18_CTH"] = season_stats(larc["GOES18_CTH"])
    if kavin is not None:
        cth["LaRC CSV (Kavin's 1 Sep copy): GOES18_CTH"] = season_stats(kavin["GOES18_CTH"])

    # --- 2. Cloud base: notebook Section 7 (ceilometer first_cbh, QC = 0), and the CSV recipe -----------------------
    ceil = load_product("cbh_ceil_M1")
    for limit in NOTEBOOK_LIMITS_M:
        base_m, _ = Q.ceilometer_cbh_m(ceil, max_cbh_m=limit, window_min=0)
        tag = " (notebook Sec. 7 = check of row 19)" if limit == 3500 else ""
        cbh[f"Ceilometer first_cbh <= {limit:g} m, QC = 0, every 16-s sample{tag}"] = season_stats(base_m)
    # the CSV recipe: same QC and detection screen, base < 2 km (strictly), half-hour means at the CSV times
    base_all_m, _ = Q.ceilometer_cbh_m(ceil, max_cbh_m=np.inf, window_min=0)
    del ceil
    base_lt2_m = base_all_m[base_all_m < CSV_LIMIT_M]
    # averaging only: the same samples (< 2 km) as half-hour means, every half hour with a value
    cbh["Bridge: same samples (< 2 km) as half-hour means, every half hour"] = \
        season_stats(base_lt2_m.resample(CSV_WINDOW).mean().dropna())
    ceil_rebuilt = half_hour_at(base_lt2_m, larc.index)
    checks["CEIL_CBH"] = recipe_check(ceil_rebuilt, larc["CEIL_CBH"])
    cbh["Bridge: CSV recipe from the ceilometer (base < 2 km, half-hour mean), at the CSV times"] = \
        season_stats(pd.Series(ceil_rebuilt, index=larc.index))
    cbh["LaRC CSV (14 Oct, Fig. 4): CEIL_CBH"] = season_stats(larc["CEIL_CBH"])
    if kavin is not None:
        cbh["LaRC CSV (Kavin's 1 Sep copy): CEIL_CBH"] = season_stats(kavin["CEIL_CBH"])

    # --- 3. Print ------------------------------------------------------------------------------------------------
    print("\nCSV columns rebuilt from ARM data (rebuilt - CSV):")
    for col, c in checks.items():
        print(f"  {col:9s} exact at {c['exact_pct']:.1f} % of rows; mean difference {c['mean_diff_m']:.1f} m; "
              f"5th-95th percentile {c['p05_diff_m']:.1f} to {c['p95_diff_m']:.1f} m; rows with no ARM data "
              f"{c['rows_without_data']}")
    sheet_for = {"Cloud top height": list(SHEET)[0], "Cloud base height": list(SHEET)[1]}
    tables: Dict[str, List[pd.DataFrame]] = {}
    for group, lines in groups.items():
        pm = {sheet_for[group]: list(SHEET[sheet_for[group]])}
        med, n = {}, {}
        for label, st in lines.items():
            pm[label] = [S.format_pm(m, s) for m, s in zip(st["mean"], st["std"])]
            med[label] = [f"{m:.0f} [{a:.0f}–{b:.0f}]" for m, a, b in zip(st["median"], st["q25"], st["q75"])]
            n[label] = [f"{x:,}" for x in st["n"]]
        pm, med, n = (pd.DataFrame(d, index=S.SEASON_ORDER).T for d in (pm, med, n))
        tables[group] = [pm, med, n]
        print(f"\n{group} (m), mean ± std per sheet season (UTC):")
        print(pm.to_string())
        print(f"\n{group}: median [25th-75th percentile] (m):")
        print(med.to_string())
        print(f"\n{group}: number of samples (4-s / 16-s samples for the notebook rows; GOES half-hours otherwise):")
        print(n.to_string())

    # --- 4. Write ------------------------------------------------------------------------------------------------
    long = pd.concat({(g, label): st for g, lines in groups.items() for label, st in lines.items()},
                     names=["quantity", "line", "season"]).reset_index()
    long.to_csv(args.out / "cloud_heights_vs_larc_seasonal.csv", index=False)
    about = pd.DataFrame({"value": {
        "notebook cloud top": "KAZR-ARSCL epcarsclkazr1kolliasM1.c1, top of the lowest layer <= limit, every 4-s "
                              "sample (check_seasonal_averages.ipynb Section 4; no QC field)",
        "notebook cloud base": "ceilometer epcceilM1.b1 first_cbh, qc_first_cbh = 0, detection_status 1-3, base <= "
                               "limit, every 16-s sample (Section 7)",
        "LaRC CSV": f"{LARC_CSV} ({len(larc):,} GOES-18 half-hours; every column < 2 km); every row used",
        "CSV KAZR_CTH recipe": "half-hour [hh:00, hh:30) mean of the highest ARSCL layer top below 2 km: exact at "
                               f"{checks['KAZR_CTH']['exact_pct']:.1f} % of rows",
        "CSV CEIL_CBH recipe": "half-hour mean of ceilometer first_cbh (QC = 0, status 1-3) < 2 km: exact at "
                               f"{checks['CEIL_CBH']['exact_pct']:.1f} % of rows",
        "GOES18_CTH": "satellite cloud-top height from the CSV; no ground-based equivalent",
        "Kavin's copy": str(KAVIN_LARC_CSV) + ("" if kavin is not None else " (not found)"),
        "seasons": "sheet seasons, UTC (seasonal.SEASONS)",
        "statistics": "mean, std (ddof = 1), median, q25, q75 (numpy.percentile, linear), n",
        "script": Path(__file__).name,
        "git revision": git_revision(),
    }})
    xlsx = args.out / "cloud_heights_vs_larc_seasonal.xlsx"
    with pd.ExcelWriter(xlsx) as xl:
        for group, (pm, med, n) in tables.items():
            short = "CTH" if group.startswith("Cloud top") else "CBH"
            pm.to_excel(xl, sheet_name=f"{short} mean ± std")
            med.to_excel(xl, sheet_name=f"{short} median (IQR)")
            n.to_excel(xl, sheet_name=f"{short} n")
        about.to_excel(xl, sheet_name="About")
    print(f"\nWrote cloud_heights_vs_larc_seasonal.csv and {xlsx.name} to {args.out}")


if __name__ == "__main__":
    main()
