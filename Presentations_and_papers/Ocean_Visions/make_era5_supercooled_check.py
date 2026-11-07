#!/usr/bin/env python3
"""Is ERA5's wintertime cloud liquid water actually supercooled?

THE QUESTION
============
The Ocean Visions analysis counts "supercooled liquid-containing" clouds using
tclw, ERA5's total column cloud liquid water, which has no temperature test in it.
This script measures how much of that liquid sits below 0 deg C, and compares
three estimates of the liquid path:

    tclw      total column cloud liquid water          ANALYSIS (GRIB dataType an)
    L_cold    clwc integrated over levels with T < 0 C  ANALYSIS (t and clwc are an)
    tcslw     total column supercooled liquid water    FORECAST (GRIB dataType fc)

THE VERTICAL INTEGRAL
=====================
For one cell and hour, on the pressure levels l (hydrostatic balance):

    L_l     = q_l * dp_l / g * 1000            [g m-2]
    L_total = sum over l of L_l
    L_cold  = sum over l with T_l < 273.15 K of L_l
    f_cold  = L_cold / L_total

q_l is clwc [kg kg-1], and dp_l is the pressure thickness each level owns, clipped
at the surface so that a level below ground contributes nothing
(convert_specific_to_absolute.layer_thickness_pa). This is the integral
make_era5_hourly_values_csvs.py uses for cloud temperature.

TWO LIMITS OF THIS ESTIMATE
===========================
1. Levels are 25-50 hPa apart in the lower troposphere, and each level is called
   wholly cold or wholly warm. L_cold_m2, the same sum with T_l < 271.15 K
   (-2 deg C), brackets how much the answer leans on levels near freezing.
2. 23 pressure levels are not ERA5's 137 model levels, so L_total will not equal
   tclw. closure_ratio = sum(L_total) / sum(tclw) reports how close they are.

WHY tcslw IS REPORTED BY FORECAST LEAD
======================================
tclw and clwc are analysis fields; tcslw comes from the short forecasts started at
06 and 18 UTC. compare_supercooled_liquid.ipynb measured the effect at the ARM
cell: the RMS of (tcslw - tclw) grows from 3.5 g m-2 at step 1 to 10.3 g m-2 at
step 12. So a difference between tcslw and an analysis field mixes physics with
forecast drift. The lead table separates them: read lead 1-3 for the closest
comparison, and expect some disagreement at every lead. The forecast step of a
valid hour is

    lead = ((UTC hour - 7) mod 12) + 1         07 UTC -> 1, 18 UTC -> 12, 19 UTC -> 1

THE TWO POPULATIONS
===================
    liquid_containing   the hours in the Ocean Visions tables: overcast, cloud water
                        above the floors, not precipitating, IWP/CWP < 0.90
    any_liquid          every cell-hour with tclw above the 0.01 g m-2 floor, with no
                        other test, in case the filters hide warm liquid

WHERE THE DATA ARE, AND TWO WAYS TO RUN
=======================================
GDEX has t and clwc for every season but no tcslw. The laptop has tcslw for every
season but pressure levels for only part of the record. So the full comparison has
two steps:

    # 1. On Casper: temperature check for all seasons, plus a netCDF of L_total, L_cold
    ./submit.sh --region barrow --years 2014-2024 --csvs supercooled --cell-weighting uniform

    # 2. On the laptop, after downloading that netCDF: adds tcslw from the local files
    python make_era5_supercooled_check.py --fields-in <file.nc> --cell-weighting uniform

If local files hold both, one local run does everything:

    python make_era5_supercooled_check.py --years 2024

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules on sys.path.
import ov_config as ov

import netCDF4
import numpy as np
import pandas as pd

from make_era5_hour_count_csvs import OUT_PRODUCT, SLOT_CLASS, SLOT_NAMES
from make_era5_hourly_values_csvs import PL_BLOCK_HOURS, open_pressure_archive
from make_era5_utqiagvik_filtered_netcdf import SETTINGS, build_args, git_describe

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from cloud_classification import fraction_phase_masks  # noqa: E402
from convert_specific_to_absolute import G_M_S2, layer_thickness_pa  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
    site_in_grid,
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)
from turbulent_flux_response import CLOUD_T_MIN_LWP_G  # noqa: E402  g m-2

FREEZING_K = 273.15          # K, 0 deg C: the definition of supercooled used here
MARGIN_K = 271.15            # K, -2 deg C: brackets levels close to freezing
TCSLW_QUANTUM_G_M2 = 0.031   # g m-2, storage quantum of tcslw; see lwph at LIQUID_VARS

# Share-of-column thresholds on f_cold reported as shares of cell-hours.
F_COLD_LEVELS = (0.50, 0.90, 0.99)

POPULATIONS = ("liquid_containing", "any_liquid")
SLOTS = (*SLOT_NAMES, "AllCells")

# What is summed per (hour, population, slot). Every entry is weighted by the cell
# weight, so "sum_w" is the denominator of a mean. Units: g m-2 for paths, and
# (g m-2)^2 for the squares and cross products used for RMS and correlation.
STATS = ("sum_w", "n_cell_hours",
         "tclw", "L_total", "L_cold", "L_cold_m2",
         "n_eval", "f_ge_050", "f_ge_090", "f_ge_099", "f_all_cold",
         "tcslw", "tcslw2", "tclw2", "L_cold2", "L_total2",
         "tclw_x_tcslw", "L_cold_x_tcslw", "tclw_x_L_total", "tcslw_gt_tclw")
I = {name: k for k, name in enumerate(STATS)}


def forecast_lead_h(hour_utc: np.ndarray) -> np.ndarray:
    """Forecast step of a valid hour: forecasts start at 06 and 18 UTC; steps 1-12."""
    return (np.asarray(hour_utc) - 7) % 12 + 1


def liquid_paths_from_levels(pds, pos: np.ndarray, sp_pa: np.ndarray) -> tuple:
    """(L_total, L_cold, L_cold_m2), g m-2, each (n, lat, lon), from t and clwc.

    ``pos`` indexes the pressure archive's time axis and ``sp_pa`` is the surface
    pressure (Pa) of the same hours, (n, lat, lon).
    """
    lv_hpa = pds["pressure_level"].values.astype(float)
    order = np.argsort(lv_hpa)                               # ascending pressure
    p_pa = lv_hpa[order] * 100.0                             # hPa -> Pa
    blk = pds[["t", "clwc"]].isel(valid_time=pos).load()
    dims = [d for d in blk["t"].dims if d != "pressure_level"]

    def lev_last(name: str) -> np.ndarray:                   # (n, lat, lon, lev)
        return blk[name].transpose(*dims, "pressure_level").values.astype(
            np.float64)[..., order]

    t_k = lev_last("t")
    dp_pa = layer_thickness_pa(p_pa, sp_pa)                  # (n, lat, lon, lev)
    # Liquid path each level contributes: q * dp / g * 1000 g kg-1. Negative clwc
    # is a numerical artefact, so it is clipped, as in the cloud temperature code.
    path_g_m2 = np.clip(lev_last("clwc"), 0.0, None) * dp_pa / G_M_S2 * 1000.0
    return (path_g_m2.sum(axis=-1),
            (path_g_m2 * (t_k < FREEZING_K)).sum(axis=-1),
            (path_g_m2 * (t_k < MARGIN_K)).sum(axis=-1))


class FieldsWriter:
    """Hourly L_total and L_cold on the grid, for the tcslw step on the laptop."""

    def __init__(self, path: Path, times: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.nc = netCDF4.Dataset(path, "w", format="NETCDF4")
        self.nc.createDimension("valid_time", times.size)
        self.nc.createDimension("latitude", lat.size)
        self.nc.createDimension("longitude", lon.size)
        t = self.nc.createVariable("valid_time", "i8", ("valid_time",))
        t.units = "hours since 1970-01-01 00:00:00"
        t.calendar = "proleptic_gregorian"
        t[:] = times.astype("datetime64[h]").astype("int64")
        self.nc.createVariable("latitude", "f8", ("latitude",))[:] = lat
        self.nc.createVariable("longitude", "f8", ("longitude",))[:] = lon
        self.vars = {}
        for name, desc in (("lwp_pl_total_g_m2", "clwc integrated over pressure levels"),
                           ("lwp_pl_cold_g_m2", "the same over levels with T < 273.15 K")):
            # 0.01 g m-2 is the analysis floor, so finer digits only cost disk.
            v = self.nc.createVariable(
                name, "f4", ("valid_time", "latitude", "longitude"), zlib=True, complevel=4,
                least_significant_digit=2,
                chunksizes=(min(24, times.size), lat.size, lon.size))
            v.units, v.long_name = "g m-2", desc
            self.vars[name] = v
        self.nc.history = f"make_era5_supercooled_check.py, repo {git_describe()}"

    def write(self, rows: np.ndarray, total: np.ndarray, cold: np.ndarray) -> None:
        self.vars["lwp_pl_total_g_m2"][rows] = total
        self.vars["lwp_pl_cold_g_m2"][rows] = cold

    def close(self) -> None:
        self.nc.close()


def accumulate(args: argparse.Namespace, fields_in: Path | None,
               fields_out: Path | None) -> dict:
    """Per-(hour, population, slot) sums of every entry in STATS."""
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(args.region, lwph.resolve_data_root(args.storage, args.data_root),
                           args.mask_grid), ds)

    layout = season_layout(ds, args)
    keep_idx, used, _ = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    times = np.asarray(ds["valid_time"].values)
    row_of_step = np.cumsum(use_step) - 1
    n_hour = int(use_step.sum())

    site_mask, _site_lat, _site_lon = site_cell_mask(ds)
    has_site = site_in_grid(ds)
    lat_deg = ds["latitude"].values
    if args.cell_weighting == "area":
        cell_weight = area_weights_2d(lat_deg, ds.sizes["longitude"])     # (lat, lon)
    else:
        cell_weight = np.ones((ds.sizes["latitude"], ds.sizes["longitude"]))

    # --- where the level-integrated paths come from ---------------------------
    pds = fields_nc = None
    if fields_in is not None:
        fields_nc = netCDF4.Dataset(fields_in)
        t_pl = (fields_nc["valid_time"][:].filled().astype("datetime64[h]")
                .astype("datetime64[ns]"))
        if not (np.allclose(fields_nc["latitude"][:], lat_deg)
                and np.allclose(fields_nc["longitude"][:], ds["longitude"].values)):
            raise ValueError(f"{fields_in} is on a different grid from region {args.region!r}")
    else:
        pds = open_pressure_archive(args, ds)
        if pds is None:
            raise SystemExit(f"No pressure levels for {args.region!r} and no --fields-in.")
        t_pl = pds["valid_time"].values

    writer = None
    if fields_out is not None:
        writer = FieldsWriter(fields_out, times[use_step], lat_deg, ds["longitude"].values)

    has_tcslw = "tcslw" in ds.data_vars
    read_vars = ["tcc", "tclw", "tciw", "tp", "siconc", "sp"] + (["tcslw"] if has_tcslw else [])
    print(f"  tcslw in this archive: {has_tcslw}.  Paths from "
          f"{'--fields-in' if fields_nc is not None else 'pressure levels'}.", flush=True)

    # (hour, population, slot, stat)
    A = np.zeros((n_hour, len(POPULATIONS), len(SLOTS), len(STATS)))
    matched = np.zeros(n_hour, dtype=bool)

    print(f"  Streaming {n_hour:,} hours ...", flush=True)
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours, keep_mask=use_step):
        sl = slice(i0, i0 + block.sizes["valid_time"])
        keep = use_step[sl]
        if not keep.any():
            continue
        rows = row_of_step[sl][keep]                             # (t,)
        tk = times[sl][keep]
        # Only hours that have level data can be evaluated.
        pos = np.searchsorted(t_pl, tk)
        inside = pos < t_pl.size
        match = np.zeros(rows.size, dtype=bool)
        match[inside] = t_pl[pos[inside]] == tk[inside]
        hit = np.flatnonzero(match)
        if not hit.size:
            continue
        matched[rows[hit]] = True

        classes = classify_cells(lsm, block["siconc"].values, args.lsm_tol,
                                 args.open_ocean_max_siconc, args.sea_ice_min_siconc,
                                 args.land_max_siconc)[keep]
        tcc = block["tcc"].values[keep]                          # 0-1
        tclw_g_m2 = block["tclw"].values[keep] * 1000.0          # kg -> g m-2
        iwp_g_m2 = block["tciw"].values[keep] * 1000.0
        rate_mm_hr = block["tp"].values[keep] * 1000.0           # m h-1 -> mm h-1
        sp_pa = block["sp"].values[keep]
        tcslw_g_m2 = block["tcslw"].values[keep] * 1000.0 if has_tcslw else None

        # The same filter as make_era5_hour_count_csvs.count_cell_hours.
        valid = np.isfinite(tcc) & np.isfinite(tclw_g_m2) & np.isfinite(iwp_g_m2)
        f = fraction_phase_masks(tclw_g_m2, iwp_g_m2, args.liquid_fraction_min,
                                 args.ice_fraction_min, args.min_lwp, args.min_iwp)
        raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
        kept = (valid & (tcc >= args.min_cloud_fraction) & (f["cwp_g"] > 0.0) & ~raining)
        pop_masks = (kept & (f["liquid"] | f["mixed"]), valid & (tclw_g_m2 > args.min_lwp))

        for j0 in range(0, hit.size, PL_BLOCK_HOURS):
            h = hit[j0:j0 + PL_BLOCK_HOURS]                      # (n,) within this block
            if fields_nc is not None:
                total = fields_nc["lwp_pl_total_g_m2"][pos[h]].filled(np.nan).astype(float)
                cold = fields_nc["lwp_pl_cold_g_m2"][pos[h]].filled(np.nan).astype(float)
                cold_m2 = np.full(total.shape, np.nan)           # not stored; Casper tables hold it
            else:
                total, cold, cold_m2 = liquid_paths_from_levels(pds, pos[h], sp_pa[h])
            if writer is not None:
                writer.write(rows[h], total, cold)

            # f_cold is judged only where the column holds enough liquid on the
            # levels for a share to mean something.
            evaluable = total >= CLOUD_T_MIN_LWP_G
            with np.errstate(invalid="ignore", divide="ignore"):
                f_cold = np.where(evaluable, cold / total, np.nan)
            tclw_h = tclw_g_m2[h].astype(float)
            fields = {
                "sum_w": np.ones(total.shape),
                "tclw": tclw_h, "L_total": total, "L_cold": cold, "L_cold_m2": cold_m2,
                "n_eval": evaluable.astype(float),
                "f_ge_050": evaluable & (f_cold >= F_COLD_LEVELS[0]),
                "f_ge_090": evaluable & (f_cold >= F_COLD_LEVELS[1]),
                "f_ge_099": evaluable & (f_cold >= F_COLD_LEVELS[2]),
                # All of the column's liquid below 0 C, to float precision.
                "f_all_cold": evaluable & (total - cold <= 1e-9),
                "tclw2": tclw_h ** 2, "L_cold2": cold ** 2, "L_total2": total ** 2,
                "tclw_x_L_total": tclw_h * total,
            }
            if has_tcslw:
                sc = tcslw_g_m2[h].astype(float)
                fields.update(tcslw=sc, tcslw2=sc ** 2, tclw_x_tcslw=tclw_h * sc,
                              L_cold_x_tcslw=cold * sc,
                              tcslw_gt_tclw=sc > tclw_h + TCSLW_QUANTUM_G_M2)

            for pi, pop in enumerate(pop_masks):
                pop_h = pop[h]
                for si, slot in enumerate(SLOTS):
                    if slot == "AllCells":
                        m = pop_h
                    elif slot == "UTQ":
                        if not has_site:
                            continue
                        m = pop_h & site_mask
                    else:
                        m = pop_h & (classes[h] == CLASS_CODES[SLOT_CLASS[slot]])
                    wm = m * cell_weight                         # (n, lat, lon)
                    A[rows[h], pi, si, I["n_cell_hours"]] = m.sum(axis=(1, 2))
                    for name, v in fields.items():
                        A[rows[h], pi, si, I[name]] = (
                            np.where(wm > 0, wm * v, 0.0)).sum(axis=(1, 2))

    if writer is not None:
        writer.close()
    if fields_nc is not None:
        fields_nc.close()
    hour_time = pd.DatetimeIndex(times[use_step])
    print(f"  level data matched {int(matched.sum()):,} of {n_hour:,} hours", flush=True)
    return {"A": A, "time": hour_time, "matched": matched, "has_site": has_site,
            "has_tcslw": has_tcslw, "seasons": list(used),
            "season_of_hour": np.asarray(layout["seasons"])[s_idx[use_step]],
            "weighting": args.cell_weighting, "paths_from_fields": fields_nc is not None}


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def corr(sxy, sx, sy, sxx, syy, w) -> float:
    """Weighted Pearson correlation from summed moments; NaN if a variance is zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / w - sx * sy / w ** 2
        var_x, var_y = sxx / w - (sx / w) ** 2, syy / w - (sy / w) ** 2
        return float(cov / np.sqrt(var_x * var_y))


def summarise(s: np.ndarray) -> dict:
    """Derived quantities from STATS summed over a set of hours. Paths in g m-2."""
    g = {name: float(s[I[name]]) for name in STATS}
    w = g["sum_w"]
    if not w > 0:
        return {"n_cell_hours": int(g["n_cell_hours"])}

    def ratio(a: str, b: str) -> float:
        return g[a] / g[b] if g[b] > 0 else np.nan

    out = {
        "n_cell_hours": int(g["n_cell_hours"]),
        "mean_tclw_g_m2": g["tclw"] / w,
        "mean_L_total_g_m2": g["L_total"] / w,
        "mean_L_cold_g_m2": g["L_cold"] / w,
        # Share of the liquid MASS below 0 C (and below -2 C), pooled.
        "share_of_liquid_cold": ratio("L_cold", "L_total"),
        "share_of_liquid_cold_m2": ratio("L_cold_m2", "L_total"),
        # Share of cell-hours, among those with enough liquid to judge.
        "share_evaluable": g["n_eval"] / w,
        "share_hours_f_ge_050": ratio("f_ge_050", "n_eval"),
        "share_hours_f_ge_090": ratio("f_ge_090", "n_eval"),
        "share_hours_f_ge_099": ratio("f_ge_099", "n_eval"),
        "share_hours_all_cold": ratio("f_all_cold", "n_eval"),
        # How closely the 23 levels reproduce the model's own column.
        "closure_L_total_over_tclw": ratio("L_total", "tclw"),
        "corr_tclw_L_total": corr(g["tclw_x_L_total"], g["tclw"], g["L_total"],
                                  g["tclw2"], g["L_total2"], w),
    }
    if g["tcslw2"] > 0:
        out.update(
            mean_tcslw_g_m2=g["tcslw"] / w,
            tcslw_over_tclw=ratio("tcslw", "tclw"),
            tcslw_over_L_cold=ratio("tcslw", "L_cold"),
            rms_tcslw_minus_tclw_g_m2=np.sqrt(max(
                (g["tcslw2"] - 2 * g["tclw_x_tcslw"] + g["tclw2"]) / w, 0.0)),
            rms_tcslw_minus_L_cold_g_m2=np.sqrt(max(
                (g["tcslw2"] - 2 * g["L_cold_x_tcslw"] + g["L_cold2"]) / w, 0.0)),
            corr_tcslw_tclw=corr(g["tclw_x_tcslw"], g["tcslw"], g["tclw"],
                                 g["tcslw2"], g["tclw2"], w),
            corr_tcslw_L_cold=corr(g["L_cold_x_tcslw"], g["tcslw"], g["L_cold"],
                                   g["tcslw2"], g["L_cold2"], w),
            share_tcslw_exceeds_tclw=g["tcslw_gt_tclw"] / w,
        )
    return out


def table(R: dict, group_cols: dict, select: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (population, slot, group value), summed over the matched hours."""
    use = R["matched"] if select is None else R["matched"] & select
    keys = list(group_cols)
    frame = pd.DataFrame({k: v[use] for k, v in group_cols.items()})
    idx_of = np.flatnonzero(use)
    rows = []
    for group, sub in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        hours = idx_of[sub.index.to_numpy()]
        for pi, pop in enumerate(POPULATIONS):
            for si, slot in enumerate(SLOTS):
                if slot == "UTQ" and not R["has_site"]:
                    continue
                s = R["A"][hours, pi, si].sum(axis=0)
                rows.append({"population": pop, "slot": slot,
                             **dict(zip(keys, group)), **summarise(s)})
    return pd.DataFrame(rows)


def write_tables(R: dict, out_dir: Path, span: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = R["time"]
    season = np.array([f"{y}/{(y + 1) % 100:02d}" for y in R["season_of_hour"]])
    month = np.array(t.month)
    lead = forecast_lead_h(t.hour)
    everything = np.full(t.size, "all_seasons")
    tag = "fields" if R["paths_from_fields"] else "levels"
    written = []

    def save(df: pd.DataFrame, name: str) -> None:
        path = out_dir / ov.weighted_names(f"ERA5_supercooledCheck_{name}_{span}_{tag}.csv",
                                           R["weighting"])
        df.to_csv(path, index=False, float_format="%.5g")
        written.append(path)

    save(table(R, {"record": everything}), "perRecord")
    save(table(R, {"season": season}), "perSeason")
    save(table(R, {"month": month}).assign(
        month_name=lambda d: d["month"].map(lambda m: calendar.month_abbr[m])), "perMonth")
    save(table(R, {"season": season, "month": month}), "perSeasonMonth")
    # Lead only matters against tcslw; written either way, so one layout serves both.
    save(table(R, {"forecast_lead_h": lead}), "perForecastLead")
    return written


def print_summary(R: dict) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 40)
    everything = np.full(R["time"].size, "all")
    df = table(R, {"record": everything}).drop(columns="record")
    cols = ["population", "slot", "n_cell_hours", "share_of_liquid_cold",
            "share_of_liquid_cold_m2", "share_hours_f_ge_090", "share_hours_f_ge_099",
            "share_hours_all_cold", "closure_L_total_over_tclw", "corr_tclw_L_total"]
    print("\n  How much of the liquid is below 0 C, whole record:")
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))

    month = np.array(R["time"].month)
    m = table(R, {"month": month})
    m = m[(m["population"] == "liquid_containing") & (m["slot"] == "AllCells")]
    print("\n  liquid-containing hours, all cells, by month:")
    print(m[["month", "n_cell_hours", "share_of_liquid_cold", "share_hours_f_ge_099"]]
          .round(4).to_string(index=False))

    if R["has_tcslw"]:
        lead = forecast_lead_h(R["time"].hour)
        d = table(R, {"forecast_lead_h": lead})
        d = d[(d["population"] == "liquid_containing") & (d["slot"] == "AllCells")]
        keep = ["forecast_lead_h", "mean_tclw_g_m2", "mean_L_cold_g_m2", "mean_tcslw_g_m2",
                "rms_tcslw_minus_tclw_g_m2", "rms_tcslw_minus_L_cold_g_m2",
                "corr_tcslw_tclw", "corr_tcslw_L_cold", "share_tcslw_exceeds_tclw"]
        print("\n  tclw (analysis), L_cold (analysis), tcslw (forecast), by forecast lead:")
        print(d[keep].round(3).to_string(index=False))


def write_readme(R: dict, out_dir: Path, run: ov.RunConfig) -> Path:
    text = f"""IS ERA5'S WINTERTIME CLOUD LIQUID SUPERCOOLED?
==============================================

Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_supercooled_check.py (repo {git_describe()}).
Domain {run.domain.name}: {run.domain.describe()}. Seasons {R['seasons'][0]}/{(R['seasons'][0] + 1) % 100:02d} to {R['seasons'][-1]}/{(R['seasons'][-1] + 1) % 100:02d}.
Level data matched {int(R['matched'].sum()):,} of {R['matched'].size:,} hours; tables use only the matched hours.
Cell weighting: {R['weighting']}. tcslw included: {R['has_tcslw']}.
File names end in _levels (paths integrated here) or _fields (paths read from a
netCDF written on Casper, stored to 0.01 g m-2; L_cold_m2 is then blank).

The method and its limits are in the script's docstring.

COLUMNS
  population               liquid_containing (the hours in the OV tables) or any_liquid
  slot                     UTQ, the five surface classes, or AllCells
  n_cell_hours             plain count of cell-hours
  mean_*_g_m2              weighted mean path over those cell-hours
  share_of_liquid_cold     sum(L_cold) / sum(L_total): share of liquid MASS below 0 C
  share_of_liquid_cold_m2  the same below -2 C, to bracket levels near freezing
  share_evaluable          share of cell-hours with L_total >= {CLOUD_T_MIN_LWP_G:g} g m-2 on the levels
  share_hours_f_ge_NNN     of those, share with f_cold >= 0.50, 0.90, 0.99
  share_hours_all_cold     of those, share with no liquid on any level at or above 0 C
  closure_L_total_over_tclw sum(L_total) / sum(tclw): levels against the model's column
  corr_tclw_L_total        correlation across cell-hours
  mean_tcslw_g_m2 and the columns containing tcslw appear only where tcslw is available.

COMPARING WITH tcslw
tclw and L_cold are analysis fields and tcslw is a forecast field. Use the
perForecastLead table and lead 1-3 for the closest comparison. See the docstring.
"""
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default <hour-count folder>/supercooled_check)")
    p.add_argument("--fields-out", action="store_true",
                   help="also write hourly L_total and L_cold to netCDF, for the tcslw step")
    p.add_argument("--fields-in", type=Path, default=None,
                   help="read paths from this netCDF instead of pressure levels")
    ov.add_weighting_arg(p)
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    out_dir = opts.out_dir or (run.out_dir(OUT_PRODUCT)
                               / f"supercooled_check{ov.weight_suffix(opts.cell_weighting)}")
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": opts.cell_weighting})
    span = ov.span_tag(run.years)
    fields_out = (out_dir / f"ERA5_liquidPathFromLevels_{run.domain.name}_{span}.nc"
                  if opts.fields_out else None)

    print("=" * 72)
    print("Is ERA5's cloud liquid water below 0 deg C?")
    print(run.report())
    print(f"  cell weighting {opts.cell_weighting}")
    print("=" * 72)
    R = accumulate(args, opts.fields_in, fields_out)
    written = write_tables(R, out_dir, span)
    readme = write_readme(R, out_dir, run)
    print_summary(R)
    print(f"\n  -> {len(written)} CSV files + {readme.name} in {out_dir}")
    if fields_out is not None:
        print(f"  -> {fields_out}  ({fields_out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
