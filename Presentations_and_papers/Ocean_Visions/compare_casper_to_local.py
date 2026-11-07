#!/usr/bin/env python3
"""Compare a Casper run with the Barrow 2014-2024 files made on the laptop.

Run this on the laptop after bringing a Casper run home:

    python compare_casper_to_local.py <folder holding era5_hour_count_csv/>

It pairs every CSV under the Casper folder with the file of the same relative
path beside this script, aligns rows on the first column, and reports the
differences. It changes nothing.

WHAT TO EXPECT (see aws_pipeline/README.md)
===========================================
The two records are not bit-identical, so small differences are expected:

* tp. GDEX has no hourly tp, so the reader rebuilds it as tp [m] = mtpr * 3.6.
  The two fields are stored on different grids of values, so the 0.05 mm/hr test
  falls in a slightly different place: CDS tp has levels 0.000477 mm/hr apart with
  one at 0.05007, so it flags from about 0.0498, while mtpr has levels 0.000429
  apart around 0.0500. Casper therefore counts slightly MORE hours, nearly always in
  the same direction. Measured for Barrow, 11 seasons: 0.03-0.25% in seasonal
  totals. The precipAllowed tables have no tp test and should be identical.
  Per-day tables hold small numbers, so the same few cell-hours look large there,
  and most of the "values differ" count comes from them. Read the perSeason row.
* skt, t2m, d2m, sp, tcwv, istl1, istl2 can differ by one float32 unit.
* CloudTemp. The laptop holds pressure levels for only part of the record, so
  those columns are compared only on rows where both files have a value, and the
  rows filled only on Casper are counted separately.

A LARGE difference, or a table present on one side only, is what needs a look.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRODUCTS = ("era5_hour_count_csv", "era5_hourly_values_csv")
NETCDF_GLOB = "era5_utqiagvik_OV_figs1-2-4_filtered_*.nc"


def compare_csv(casper: Path, local: Path) -> dict:
    """Worst differences over the numeric columns both files share.

    "worst_pct" divides by the mean magnitude of the column in the local file,
    not by each value, so a difference of 1 h on a value of 2 h does not dominate.
    """
    a = pd.read_csv(casper)
    b = pd.read_csv(local)
    # Rows are identified by their text columns. Some tables repeat those (the
    # ice_vs_water table has several rows per metric), so a running count within
    # each group makes every key unique. Without it the merge would pair each row
    # with every repeat and report differences between identical files.
    text_cols = [c for c in a.columns if not pd.api.types.is_numeric_dtype(a[c])]
    key = text_cols + ["_row_in_group"]
    for df in (a, b):
        df["_row_in_group"] = df.groupby(text_cols, dropna=False).cumcount() if text_cols else df.index
    keys_a = pd.MultiIndex.from_frame(a[key].astype(str))
    keys_b = pd.MultiIndex.from_frame(b[key].astype(str))
    # An inner merge would hide a row present on one side, so count those first.
    only_casper = int((~keys_a.isin(keys_b)).sum())
    only_local = int((~keys_b.isin(keys_a)).sum())
    merged = a.merge(b, on=key, suffixes=("_c", "_l"))
    out = {"worst_abs": 0.0, "worst_pct": 0.0, "column": "", "n_diff": 0, "n_values": 0,
           "only_casper_rows": only_casper, "only_local_rows": only_local,
           "filled_casper_only": 0, "filled_local_only": 0}
    for col in a.columns:
        if col in key or col not in b.columns or not pd.api.types.is_numeric_dtype(b[col]):
            continue
        x = merged[f"{col}_c"].to_numpy(float)
        y = merged[f"{col}_l"].to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        out["filled_casper_only"] += int((np.isfinite(x) & ~np.isfinite(y)).sum())
        out["filled_local_only"] += int((np.isfinite(y) & ~np.isfinite(x)).sum())
        if not both.any():
            continue
        d = np.abs(x[both] - y[both])
        out["n_values"] += int(both.sum())
        out["n_diff"] += int((d > 0).sum())
        scale = float(np.mean(np.abs(y[both])))
        if d.max() > out["worst_abs"]:
            out["worst_abs"] = float(d.max())
            out["column"] = col
        if scale > 0:
            out["worst_pct"] = max(out["worst_pct"], 100.0 * float(d.max()) / scale)
    return out


def compare_tree(casper_root: Path, product: str) -> pd.DataFrame:
    """One row per CSV of a product; prints files found on one side only."""
    c_dir, l_dir = casper_root / product, HERE / product
    if not c_dir.is_dir():
        print(f"  {product}: not in the Casper folder")
        return pd.DataFrame()
    c_files = {p.relative_to(c_dir) for p in c_dir.rglob("*.csv")}
    l_files = {p.relative_to(l_dir) for p in l_dir.rglob("*.csv")}
    for label, extra in (("Casper only", c_files - l_files), ("laptop only", l_files - c_files)):
        if extra:
            print(f"  {product}: {len(extra)} CSV files on {label}, e.g. {sorted(extra)[0]}")
    rows = [{"file": str(rel), **compare_csv(c_dir / rel, l_dir / rel)}
            for rel in sorted(c_files & l_files)]
    return pd.DataFrame(rows)


def compare_netcdf(casper_root: Path) -> None:
    """Variable-by-variable differences in the UTQ netCDF, when both exist."""
    import xarray as xr

    found = sorted(casper_root.rglob(NETCDF_GLOB))
    if not found:
        print("\n  no UTQ netCDF in the Casper folder")
        return
    local = HERE / found[0].name
    if not local.exists():
        print(f"\n  no local {local.name} to compare with")
        return
    a, b = xr.open_dataset(found[0]), xr.open_dataset(local)
    t_only = np.setxor1d(a["valid_time"].values, b["valid_time"].values).size
    print(f"\n  UTQ netCDF: {a.sizes['valid_time']:,} hours on Casper, "
          f"{b.sizes['valid_time']:,} on the laptop, {t_only} hours on one side only")
    shared_t = np.intersect1d(a["valid_time"].values, b["valid_time"].values)
    missing = sorted(set(a.data_vars) ^ set(b.data_vars))
    if missing:
        print(f"    variables on one side only: {missing}")
    for v in sorted(set(a.data_vars) & set(b.data_vars)):
        x = a[v].sel(valid_time=shared_t).values.astype(float)
        y = b[v].sel(valid_time=shared_t).values.astype(float)
        d = np.abs(x - y)
        worst = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
        if worst > 0:
            print(f"    {v:<26s} worst |diff| {worst:.4g}   ({int((d > 0).sum()):,} hours differ)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("casper_root", type=Path,
                   help="the Casper run folder holding era5_hour_count_csv/ (searched below too)")
    p.add_argument("--worst", type=int, default=10, help="how many of the largest to list")
    opts = p.parse_args(argv)

    # submit.sh nests results one level down, in <domain>_<span>/.
    hits = sorted(opts.casper_root.rglob("era5_hour_count_csv"))
    if not hits:
        raise SystemExit(f"no era5_hour_count_csv folder under {opts.casper_root}")
    root = hits[0].parent
    print(f"Casper: {root}\nLaptop: {HERE}\n")

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    for product in PRODUCTS:
        t = compare_tree(root, product)
        if t.empty:
            continue
        exact = int((t["n_diff"] == 0).sum())
        print(f"{product}: {len(t)} files compared, {exact} with no differing value; "
              f"{int(t['n_diff'].sum()):,} of {int(t['n_values'].sum()):,} values differ")
        print(f"  rows on one side only: {int(t['only_casper_rows'].sum())} Casper, "
              f"{int(t['only_local_rows'].sum())} laptop; values filled only on Casper "
              f"{int(t['filled_casper_only'].sum()):,}, only on laptop "
              f"{int(t['filled_local_only'].sum()):,}")
        cols = ["file", "column", "worst_abs", "worst_pct", "n_diff"]
        # Per-day values are small, so split by resolution before reading percentages.
        by = t.assign(resolution=t["file"].str.extract(r"_(perSeason|perMonth|perDay|perHour)_",
                                                       expand=False).fillna("other"))
        print(by.groupby("resolution").agg(files=("file", "size"), values_differ=("n_diff", "sum"),
                                           values=("n_values", "sum"),
                                           worst_pct=("worst_pct", "max")).round(3).to_string())
        no_tp = t[t["file"].str.contains("precipAllowed")]
        if len(no_tp):
            print(f"  precipAllowed tables (no tp test): {int(no_tp['n_diff'].sum())} values differ")
        print(t.sort_values("worst_pct", ascending=False).head(opts.worst)[cols]
              .round(4).to_string(index=False))
        print()
    compare_netcdf(root)


if __name__ == "__main__":
    main()
