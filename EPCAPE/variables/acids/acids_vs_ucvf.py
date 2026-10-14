#!/usr/bin/env python
"""Glycolic and glyoxylic acid at Mt. Soledad matched to 3-hourly UCVF (BAMS Figure 8).

Andrew Buggee, 2026-10-14. Lynn Russell asked for both acids in Figure 8 with
the same IQR bands as the other species. Elavarasi Ravichandran (email of
2026-10-13) sent only their mean and standard deviation in UCVF bins ("the PI
table"). This script
  1. rebuilds the PI table from the raw data (check of the method), and
  2. writes the acids as 3-hour means lined up with the 3-hourly UCVF, which
     Figure 8 (EPCAPEUCVFplots_kp_AJB_v3_6) bins into medians and IQRs.

Inputs (all in <data folder>/processed/)
  Acid Data/EPCAPE_Soledad_CIMS_processed_Non_Cloud.csv
        Iodide HR-ToF-CIMS on Mt. Soledad, 30 Apr - 6 Jul 2023, 5-min means in
        ppbv, UTC, only when the inlet was NOT behind the CVI (no cloud).
        UC San Diego Library object bb1788023c; README bb1788023c_1_1.txt
        (same folder). Columns used: msd__5min__Glycolic_C2H4O3_NOCLOUDS_ppb
        and msd__5min__GLYOXILLIC_C2H2O3_NOCLOUDS_ppb (the file's spelling).
  UCVF_bb2743661p_29_1.nc
        Upwind cloud vertical fraction, every 3 h (UTC), averaged over the
        last 24 h along the back trajectory to Scripps Pier. Library object
        bb2743661p ("Aerosol Quantification Products", doi:10.6075/J0WM1DSC),
        component 29. Identical to KAVIN_TABLE.UCVF in kavin_UCVF.mat.
  AOS rBC coating thickness, rBC Mass, and Fthickly Number.xlsx, tab "Fthickly"
        Elavarasi's AOS data (Mt. Soledad, Apr - Jul 2023), "filtered for CNW
        and only for times when the CVI was off". Only its 5-min TIMES are
        used, as the CNW + CVI-off mask (see Method, step 2).
  Acid Data/Glycolic and Glyoxylic vs UCVF - April 30-July 6.csv
        The PI table: per UCVF bin, mean UCVF and the mean and standard
        deviation (ppt) of each acid. Used only for the check in step 4.
  Aerosol Quantification Products/bb2743661p_25_1_ARMTRAJ_backTrajectories.nc
        3-hourly back-trajectory clusters at Mt. Soledad and Scripps Pier
        (1 = Coastal North-Westerly, CNW; 2 = LA-Long Beach; 3 = Southerly;
        4 = Easterly; 5 = Marine Westerly; library README aerosol_readMe.pdf).
        Its "Time" variable has no units attribute; the values are Unix
        seconds (first value 2023-02-15 00:00 UTC, 3-h steps). Copied into the
        output for reference only (see Method, step 2).

Method (each step was checked against the PI table; step 4)
  1. Units: ppbv x 1000 = pptv (the PI table is in ppt). All values are
     kept, including the 1422 negative glyoxylic acid values (the background
     is subtracted, so values near zero scatter around it); removing them
     would bias the means high.
  2. CNW + CVI-off mask: a 5-min sample is kept if its time is one of the
     times in Elavarasi's "Fthickly" tab. This is the only mask that
     reproduces the PI table. Two other masks were tried and do not:
     every CVI-off sample (no CNW filter), and the ARMTRAJ cluster file's
     Soledad cluster 1 (CNW). 13 % of the tab's times fall in other ARMTRAJ
     clusters, so Elavarasi's CNW selection is not simply that file's
     cluster 1; this is an open question for her.
  3. UCVF of a 5-min sample = the 3-hourly UCVF at the start of the 3-hour
     window that contains it (the latest UCVF time at or before the sample).
     Elavarasi's AOS tabs use exactly this rule (checked on all 13692 rows).
  4. Check: bins of width 0.05 centred on 0, 0.05, ..., 1 (edges -0.025,
     0.025, ..., 1.025, left edge included). Per bin: mean UCVF of all masked
     5-min times inside the acid record, and mean and sample standard
     deviation (n - 1) of each acid over the masked samples that have a
     value. This reproduces every number of the PI table (written to
     pi_table_reproduction.csv).
  5. 3-hour means: for each UCVF time t, the mean of each acid over the
     masked 5-min samples in [t, t + 3 h), i.e. the same samples that step 3
     gives the UCVF of time t. One value per UCVF time, so each UCVF value
     counts once, as for the 3-hourly AMS species in Figure 8. Windows with
     fewer than --min-samples masked samples are left empty (default 1: any
     sample). The number of samples behind every mean is in the file.

Outputs (<data folder>/processed/derived/acids_vs_ucvf/)
  acids_3h_ucvf.csv           one row per 3-hour UCVF time in the acid record
  pi_table_reproduction.csv   the PI table next to the rebuilt one

Columns of acids_3h_ucvf.csv
  time_utc                    UCVF time = start of the 3-hour window (UTC)
  ucvf                        UCVF at time_utc (unitless, 0 to 1)
  armtraj_cluster_soledad     ARMTRAJ back-trajectory cluster at Mt. Soledad
                              (1 = CNW; for reference, not used as a filter)
  glycolic_acid_ppt           mean over the masked samples in the window
  glycolic_n_5min             number of those samples (0 to 36)
  glyoxylic_acid_ppt, glyoxylic_n_5min        the same for glyoxylic acid
  glycolic_acid_all_cvi_off_ppt, glycolic_n_all_cvi_off_5min,
  glyoxylic_acid_all_cvi_off_ppt, glyoxylic_n_all_cvi_off_5min
                              the same over every CVI-off sample (no CNW
                              filter), for comparison

Example (run from the Python-Research folder or the EPCAPE folder)
  python EPCAPE/variables/acids/acids_vs_ucvf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/variables/acids/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.download_data.config import active_machine, load_config  # noqa: E402

PPB_TO_PPT = 1000.0
WINDOW = pd.Timedelta("3h")  # UCVF time step and averaging window

CIMS_FILE = Path("Acid Data") / "EPCAPE_Soledad_CIMS_processed_Non_Cloud.csv"
PI_TABLE_FILE = Path("Acid Data") / "Glycolic and Glyoxylic vs UCVF - April 30-July 6.csv"
UCVF_FILE = Path("UCVF_bb2743661p_29_1.nc")
AOS_FILE = Path("AOS rBC coating thickness, rBC Mass, and Fthickly Number.xlsx")
AOS_MASK_SHEET = "Fthickly"
TRAJ_FILE = Path("Aerosol Quantification Products") / "bb2743661p_25_1_ARMTRAJ_backTrajectories.nc"

# CIMS column names (the file spells glyoxylic "GLYOXILLIC")
ACIDS = {
    "glycolic": "msd__5min__Glycolic_C2H4O3_NOCLOUDS_ppb",
    "glyoxylic": "msd__5min__GLYOXILLIC_C2H2O3_NOCLOUDS_ppb",
}

# PI-table bins: width 0.05, centred on 0, 0.05, ..., 1
PI_BIN_EDGES = np.round(np.arange(-0.025, 1.0251, 0.05), 3)  # 22 edges = 21 bins


def read_ucvf(path: Path) -> pd.Series:
    """3-hourly UCVF (unitless) indexed by UTC time; fill values become NaN."""
    with netCDF4.Dataset(path) as nc:
        hours = np.asarray(nc["DateTimeUTC"][:], dtype=float)  # hours since 2023-02-16 00:00
        ucvf = np.ma.filled(nc["UCVF"][:].astype(float), np.nan)
    time_utc = pd.Timestamp("2023-02-16") + pd.to_timedelta(hours, unit="h")
    return pd.Series(ucvf, index=time_utc, name="ucvf")


def read_acids_ppt(path: Path) -> pd.DataFrame:
    """5-min CVI-off CIMS glycolic and glyoxylic acid in pptv, indexed by UTC time."""
    raw = pd.read_csv(path)
    time_utc = pd.to_datetime(raw["Date_Time_UTC"], format="%d/%m/%Y %H:%M")
    acids_ppt = pd.DataFrame(
        {name: raw[col].astype(float).values * PPB_TO_PPT for name, col in ACIDS.items()},
        index=time_utc,
    )
    return acids_ppt.sort_index()


def read_cnw_mask_times(path: Path) -> pd.DatetimeIndex:
    """5-min times in Elavarasi's CNW + CVI-off AOS tab (used as the mask)."""
    tab = pd.read_excel(path, sheet_name=AOS_MASK_SHEET)
    return pd.DatetimeIndex(pd.to_datetime(tab.iloc[:, 0]))


def read_soledad_clusters(path: Path) -> pd.Series:
    """3-hourly ARMTRAJ back-trajectory cluster at Mt. Soledad (1 = CNW)."""
    with netCDF4.Dataset(path) as nc:
        seconds = np.asarray(nc["Time"][:], dtype=float)  # Unix seconds (no units attribute)
        cluster = np.asarray(nc["Soledad"][:], dtype=float)
    return pd.Series(cluster, index=pd.to_datetime(seconds, unit="s"), name="cluster")


def bin_like_pi_table(ucvf: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Mean UCVF, mean and sample std (n - 1) of VALUES in the PI-table bins (21 rows)."""
    df = pd.DataFrame({"ucvf": ucvf, "value": values})
    df["bin"] = pd.cut(df["ucvf"], PI_BIN_EDGES, right=False, labels=False)
    has_value = df.dropna(subset=["value"]).groupby("bin")["value"]
    out = pd.DataFrame(
        {
            "ucvf_mean": df.dropna(subset=["ucvf"]).groupby("bin")["ucvf"].mean(),
            "mean_ppt": has_value.mean(),
            "std_ppt": has_value.std(ddof=1),
            "n_5min": has_value.size(),
        }
    )
    return out.reindex(range(len(PI_BIN_EDGES) - 1))


def reproduce_pi_table(acids_ppt: pd.DataFrame, ucvf_5min: np.ndarray, pi_path: Path) -> pd.DataFrame:
    """Rebuild the PI table from masked 5-min samples and put it next to the original."""
    pi = pd.read_csv(pi_path).iloc[: len(PI_BIN_EDGES) - 1].astype(float)
    pi.columns = ["ucvf", "glycolic_mean", "glycolic_std", "glyoxylic_mean", "glyoxylic_std"]
    rebuilt = {name: bin_like_pi_table(ucvf_5min, acids_ppt[name].values) for name in ACIDS}
    table = pd.DataFrame({"bin_centre": np.round(np.arange(0, 1.0001, 0.05), 2)})
    table["pi_ucvf"] = pi["ucvf"]
    table["rebuilt_ucvf"] = rebuilt["glycolic"]["ucvf_mean"].values  # the same for both acids
    for name in ACIDS:
        for stat in ("mean", "std"):
            table[f"pi_{name}_{stat}_ppt"] = pi[f"{name}_{stat}"]
            table[f"rebuilt_{name}_{stat}_ppt"] = rebuilt[name][f"{stat}_ppt"].values
        table[f"{name}_n_5min"] = rebuilt[name]["n_5min"].values
    return table


def three_hour_means(acids_ppt: pd.DataFrame, keep: np.ndarray, min_samples: int) -> pd.DataFrame:
    """Mean and sample count of each acid in each 3-hour window [t, t + 3 h)."""
    window_start = acids_ppt.index.floor(WINDOW)
    out = {}
    for name in ACIDS:
        values = acids_ppt[name].where(keep)  # masked-out samples become NaN
        grouped = values.groupby(window_start)
        mean_ppt = grouped.mean()
        n_5min = grouped.count()
        out[f"{name}_acid_ppt"] = mean_ppt.where(n_5min >= min_samples)
        out[f"{name}_n_5min"] = n_5min
    return pd.DataFrame(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--processed", type=Path, default=None, help="<data folder>/processed (default: config.yaml)"
    )
    p.add_argument("--out", type=Path, default=None, help="output folder")
    p.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="masked 5-min samples a 3-hour window needs for its mean (default 1)",
    )
    args = p.parse_args(argv)

    processed = args.processed or active_machine(load_config()).processed_dir()
    out_dir = args.out or processed / "derived" / "acids_vs_ucvf"
    out_dir.mkdir(parents=True, exist_ok=True)

    ucvf = read_ucvf(processed / UCVF_FILE)
    acids_ppt = read_acids_ppt(processed / CIMS_FILE)
    mask_times = read_cnw_mask_times(processed / AOS_FILE)
    clusters = read_soledad_clusters(processed / TRAJ_FILE)

    # Step 2: CNW + CVI-off mask; step 3: UCVF at the start of each sample's 3-hour window
    keep = acids_ppt.index.isin(mask_times)
    ucvf_5min = ucvf.reindex(acids_ppt.index.floor(WINDOW)).values

    # Step 4: rebuild the PI table from the masked samples
    check = reproduce_pi_table(acids_ppt[keep], ucvf_5min[keep], processed / PI_TABLE_FILE)
    check.to_csv(out_dir / "pi_table_reproduction.csv", index=False, float_format="%.10g")
    worst = {
        col: np.nanmax(np.abs(check[f"pi_{col}"] - check[f"rebuilt_{col}"]))
        for col in ["ucvf"] + [f"{n}_{s}_ppt" for n in ACIDS for s in ("mean", "std")]
    }

    # Step 5: 3-hour means on the UCVF time grid of the acid record
    first = acids_ppt.index.min().floor(WINDOW)
    last = acids_ppt.index.max().floor(WINDOW)
    grid = pd.date_range(first, last, freq=WINDOW, name="time_utc")
    cnw = three_hour_means(acids_ppt, keep, args.min_samples).reindex(grid)
    every = three_hour_means(acids_ppt, np.ones(len(acids_ppt), bool), args.min_samples).reindex(grid)
    table = pd.DataFrame(index=grid)
    table["ucvf"] = ucvf.reindex(grid).values
    table["armtraj_cluster_soledad"] = clusters.reindex(grid).values
    for name in ACIDS:
        table[f"{name}_acid_ppt"] = cnw[f"{name}_acid_ppt"]
        table[f"{name}_n_5min"] = cnw[f"{name}_n_5min"].fillna(0).astype(int)
    for name in ACIDS:
        table[f"{name}_acid_all_cvi_off_ppt"] = every[f"{name}_acid_ppt"]
        table[f"{name}_n_all_cvi_off_5min"] = every[f"{name}_n_5min"].fillna(0).astype(int)
    table.to_csv(out_dir / "acids_3h_ucvf.csv", float_format="%.6g", date_format="%Y-%m-%d %H:%M")

    # Summary
    print(
        f"CIMS 5-min samples (CVI off): {len(acids_ppt)}, {acids_ppt.index.min()} to {acids_ppt.index.max()} UTC"
    )
    print(f"  kept by the CNW + CVI-off mask: {keep.sum()}")
    print("PI table rebuilt; largest absolute difference per column:")
    for col, d in worst.items():
        print(f"  {col:22s} {d:.1e}")
    print(f"3-hour windows: {len(grid)} ({grid[0]} to {grid[-1]} UTC)")
    for name in ACIDS:
        n = table[f"{name}_n_5min"]
        has = table[f"{name}_acid_ppt"].notna()
        print(
            f"  {name}: {has.sum()} windows with a mean; samples per window median {int(n[has].median())}, "
            f"min {n[has].min()}, max {n[has].max()}; windows with fewer than 6 samples: {(has & (n < 6)).sum()}"
        )
    print(f"Wrote {out_dir / 'acids_3h_ucvf.csv'}")
    print(f"Wrote {out_dir / 'pi_table_reproduction.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
