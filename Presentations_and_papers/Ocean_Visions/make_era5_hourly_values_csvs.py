#!/usr/bin/env python3
"""Hourly ERA5 LWP, cloud temperature and DLR by cloud type and surface class.

WHAT IT WRITES
==============
For Genie's monthly box plots (part 2 of her results), one row per hour of
the 11-season record (1 Oct - 31 Mar, 2014/15 .. 2024/25; 48,120 hours) and
one column per sky state, laid out as her templates
(``~/Downloads/ERA5_*_perHour_2014_2025_OpenOcean.csv``):

    ERA5_LWP_perHour_2014_2025_<CLASS>.csv
        date, liquid_containing, liquid_only, ice_only                  [g m-2]
    ERA5_CloudTemp_perHour_2014_2025_<CLASS>.csv
        date, liquid_containing, liquid_only, ice_only                  [deg C]
    ERA5_DLR_perHour_2014_2025_<CLASS>.csv
        date, liquid_containing, liquid_only, ice_only, clear_sky,
        clear_sky_calc_liquid_containing, clear_sky_calc_liquid_only,
        clear_sky_calc_ice_only                                         [W m-2]

    <CLASS> in UTQ, Land, OpenOcean, PackIce

plus ``cell_counts/ERA5_nCells_perHour_2014_2025_<CLASS>.csv`` (how many grid
cells stand behind every value) and a readME.

WHICH HOURS
===========
The sky states are those of the earlier collaborator files and figure 7,
built from the netCDF script's SETTINGS (imported, not copied):

    liquid containing / liquid only / ice only
        tcc >= 0.95, cloud water above the 0.01 g m-2 floors,
        tp < 0.05 mm hr-1, then the phase from the floored LWP/CWP
        (liquid containing IWP/CWP < 0.90; liquid only LWP/CWP >= 0.90;
        ice only IWP/CWP >= 0.90)
    clear sky
        LWP and IWP both <= 0.01 g m-2, tcc < 0.05 (plot_dlr_by_phase's
        DEFAULT_CLEAR_TCC_MAX), tp < 0.05 mm hr-1

ONE VALUE PER HOUR PER CLASS
============================
UTQ is one grid cell: its value is that cell's, or blank when the cell is not
in the state. For a surface class the value is the plain (unweighted) MEAN
over every cell of the class that is in the state in that hour, blank when
none is. The cell counts are written alongside so a cell-hour-weighted
statistic can be rebuilt from the hourly means if wanted.

THE QUANTITIES
==============
LWP        1000 * tclw (g m-2): total column cloud liquid water, unfloored.
DLR        msdwlwrf (W m-2): mean surface downward longwave over the hour
           ending at the time stamp.
clear_sky_calc_<state>
           msdwlwrfcs (W m-2): ERA5's own clear-sky downward longwave --
           the same radiation calculation with the clouds removed -- over the
           hours of that state.
CloudTemp  from the PRESSURE-LEVEL archive (temperature t, cloud liquid
           clwc and ice ciwc on 23 levels), as in
           turbulent_flux_response.cloud_temperature_field:

               T_liq = sum_l (clwc_l dp_l T_l) / sum_l (clwc_l dp_l)

           the liquid-water-weighted mean temperature of the column, used for
           liquid containing and liquid only; ice only uses the same formula
           with ciwc (the temperature of the ice). dp_l is the pressure
           thickness of level l, clipped at the surface pressure
           (convert_specific_to_absolute.layer_thickness_pa). A column whose
           weighting path is below 1e-3 g m-2 gets no value.

           Hours without pressure-level data are blank. The local archive
           (data/barrow_pressure) covers 2024/25 within this record; run with
           --storage glade (on NCAR Casper) or --storage s3 to fill the other
           seasons from NCAR's ERA5 copy (whole-globe chunks per hour, ~3 TB
           read for 11 seasons -- practical on GLADE, not on a laptop).

Usage
-----
    python make_era5_hourly_values_csvs.py                    # local data, ~2 min
    python make_era5_hourly_values_csvs.py --storage glade    # all seasons, on Casper
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Figure settings and the module path setup live in the netCDF script.
from make_era5_utqiagvik_filtered_netcdf import HERE, SETTINGS, build_args, git_describe

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from cloud_classification import fraction_phase_masks  # noqa: E402
from convert_specific_to_absolute import G_M_S2, layer_thickness_pa  # noqa: E402
from plot_dlr_by_phase import DEFAULT_CLEAR_TCC_MAX  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
)
from seb_analysis_common import (  # noqa: E402
    available_regions,
    load_seb_data,
    resolve_region_dir,
)
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    align_lsm_to_grid,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)
from turbulent_flux_response import CLOUD_T_MIN_LWP_G  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_OUT_DIR = HERE / "era5_hourly_values_csv"

# Output slots: UTQ (one cell) and the three classes Genie asked for.
SLOTS: tuple[str, ...] = ("UTQ", "Land", "OpenOcean", "PackIce")
SLOT_CLASS = {"Land": "land", "OpenOcean": "open_ocean", "PackIce": "sea_ice"}

STATES: tuple[str, ...] = ("liquid_containing", "liquid_only", "ice_only", "clear_sky")
CLOUD_STATES = STATES[:3]

# (quantity, state) -> output column, per file, in template order.
FILES = {
    "LWP": [("lwp", s, s) for s in CLOUD_STATES],
    "CloudTemp": [("tcld", s, s) for s in CLOUD_STATES],
    "DLR": ([("dlr", s, s) for s in STATES]
            + [("dlr_cs", s, f"clear_sky_calc_{s}") for s in CLOUD_STATES]),
}
QUANTS = sorted({(q, s) for cols in FILES.values() for q, s, _ in cols})

# Pressure-level hours loaded at once: 23 levels x 2,501 cells x 3 fields.
PL_BLOCK_HOURS = 48


# ----------------------------------------------------------------------------
# Cloud temperature from the pressure-level archive
# ----------------------------------------------------------------------------
def open_pressure_archive(args: argparse.Namespace, ds):
    """The pressure-level dataset for the region, or None if there is none.

    Same lookup as ``turbulent_flux_response.cloud_temperature_field``.
    """
    data_root = lwph.resolve_data_root(args.storage, args.data_root)
    region_pl = f"{args.region}_pressure"
    if region_pl not in available_regions(data_root):
        print(f"  !! no pressure-level archive {region_pl}: CloudTemp will be blank")
        return None
    pds = load_seb_data(region_pl, None, None, data_root)
    missing = sorted({"t", "clwc", "ciwc"} - set(pds.data_vars))
    if missing:
        raise KeyError(f"pressure archive is missing {missing}")
    # Same 0.25 deg grid in the same order, or the cell-by-cell pairing below
    # would silently mix columns.
    for c in ("latitude", "longitude"):
        if not np.array_equal(pds[c].values, ds[c].values):
            raise ValueError(f"pressure and single-level {c} axes differ")
    return pds


def weighted_cloud_temperature(pds, pos: np.ndarray, sp_pa: np.ndarray) -> tuple:
    """Liquid- and ice-weighted column temperature (deg C) for some hours.

    ``pos`` indexes the pressure archive's time axis; ``sp_pa`` is the
    surface pressure (Pa) of the same hours, ``(n, lat, lon)``. Returns
    ``(t_liq_C, t_ice_C)``, each ``(n, lat, lon)``, NaN where the weighting
    path is under CLOUD_T_MIN_LWP_G.
    """
    lv = pds["pressure_level"].values.astype(float)
    order = np.argsort(lv)                                   # ascending pressure
    p_pa = lv[order] * 100.0                                 # hPa -> Pa
    blk = pds[["t", "clwc", "ciwc"]].isel(valid_time=pos).load()
    dims = [d for d in blk["t"].dims if d != "pressure_level"]

    def lev_last(name: str) -> np.ndarray:                   # (n, lat, lon, lev)
        return blk[name].transpose(*dims, "pressure_level").values.astype(
            np.float64)[..., order]

    t_k = lev_last("t")
    dp_pa = layer_thickness_pa(p_pa, sp_pa)                  # (n, lat, lon, lev)
    out = []
    for species in ("clwc", "ciwc"):
        # Water path each level contributes, g m-2: q * dp / g * 1000.
        wp_l = np.clip(lev_last(species), 0.0, None) * dp_pa / G_M_S2 * 1000.0
        denom = wp_l.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_mean = (wp_l * t_k).sum(axis=-1) / denom
        t_mean[~(denom >= CLOUD_T_MIN_LWP_G)] = np.nan
        out.append(t_mean - 273.15)                          # K -> deg C
    return out[0], out[1]


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def accumulate(args: argparse.Namespace) -> dict:
    """Per-hour sums and counts of every (quantity, state) for every slot."""
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(args.region,
                           lwph.resolve_data_root(args.storage, args.data_root),
                           args.mask_grid), ds)

    layout = season_layout(ds, args)
    keep_idx, used, _ = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    times = np.asarray(ds["valid_time"].values)
    row_of_step = np.cumsum(use_step) - 1
    n_hour, n_slot = int(use_step.sum()), len(SLOTS)

    site_mask, site_lat, site_lon = site_cell_mask(ds)
    pds = open_pressure_archive(args, ds)
    t_pl = pds["valid_time"].values if pds is not None else np.array([], "M8[ns]")

    # sums[(q, s)] and counts[(q, s)]: (hour, slot). n_state[s]: cells in state.
    sums = {k: np.zeros((n_hour, n_slot)) for k in QUANTS}
    counts = {k: np.zeros((n_hour, n_slot), dtype=np.int32) for k in QUANTS}
    n_state = {s: np.zeros((n_hour, n_slot), dtype=np.int32) for s in STATES}
    n_pl_hours = 0
    pl_matched = np.zeros(n_hour, dtype=bool)        # hour has pressure-level data

    read_vars = ["tcc", args.liquid_var, "tciw", "tp", "siconc",
                 "msdwlwrf", "msdwlwrfcs", "sp"]
    print(f"  Streaming {n_hour:,} hours ...", flush=True)
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        sl = slice(i0, i0 + block.sizes["valid_time"])
        keep = use_step[sl]
        if not keep.any():
            continue
        rows = row_of_step[sl][keep]                             # (t,)
        n_t = rows.size

        # --- spatial slots ---------------------------------------------------
        classes = classify_cells(lsm, block["siconc"].values, args.lsm_tol,
                                 args.open_ocean_max_siconc,
                                 args.sea_ice_min_siconc,
                                 args.land_max_siconc)[keep]
        slot_masks = [np.broadcast_to(site_mask, classes.shape)]
        slot_masks += [classes == CLASS_CODES[SLOT_CLASS[n]] for n in SLOTS[1:]]

        # --- sky states, exactly as the earlier collaborator files ------------
        tcc = block["tcc"].values[keep]                          # 0-1
        lwp_g_m2 = block[args.liquid_var].values[keep] * 1000.0  # kg -> g m-2
        iwp_g_m2 = block["tciw"].values[keep] * 1000.0
        rate_mm_hr = block["tp"].values[keep] * 1000.0           # m h-1 -> mm h-1
        valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
        raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
        f = fraction_phase_masks(lwp_g_m2, iwp_g_m2,
                                 args.liquid_fraction_min, args.ice_fraction_min,
                                 args.min_lwp, args.min_iwp)
        has_cloud = valid & (f["cwp_g"] > 0.0)                   # a path above its floor
        kept = valid & (tcc >= args.min_cloud_fraction) & has_cloud & ~raining
        state = {
            "liquid_containing": kept & (f["liquid"] | f["mixed"]),
            "liquid_only": kept & f["liquid"],
            "ice_only": kept & f["ice"],
            # figure 7's clear sky: both paths at or below the floors
            "clear_sky": valid & ~has_cloud & (tcc < DEFAULT_CLEAR_TCC_MAX) & ~raining,
        }

        # --- quantities ------------------------------------------------------
        t_liq = np.full(tcc.shape, np.nan)
        t_ice = np.full(tcc.shape, np.nan)
        if t_pl.size:
            tk = times[sl][keep]
            pos = np.searchsorted(t_pl, tk)
            inside = pos < t_pl.size
            match = np.zeros(n_t, dtype=bool)
            match[inside] = t_pl[pos[inside]] == tk[inside]
            hit = np.flatnonzero(match)
            n_pl_hours += hit.size
            pl_matched[rows[hit]] = True
            sp_pa = block["sp"].values[keep]
            for j0 in range(0, hit.size, PL_BLOCK_HOURS):
                h = hit[j0:j0 + PL_BLOCK_HOURS]
                t_liq[h], t_ice[h] = weighted_cloud_temperature(pds, pos[h], sp_pa[h])
        value = {
            "lwp": lwp_g_m2,
            "dlr": block["msdwlwrf"].values[keep],
            "dlr_cs": block["msdwlwrfcs"].values[keep],
        }

        # --- sums and counts per (hour, slot) ---------------------------------
        for si, sm in enumerate(slot_masks):
            for s in STATES:
                n_state[s][rows, si] = (state[s] & sm).sum(axis=(1, 2))
            for q, s in QUANTS:
                v = (t_ice if s == "ice_only" else t_liq) if q == "tcld" else value[q]
                sel = state[s] & sm & np.isfinite(v)
                sums[(q, s)][rows, si] = np.where(sel, v, 0.0).sum(axis=(1, 2))
                counts[(q, s)][rows, si] = sel.sum(axis=(1, 2))

    print(f"  pressure-level data matched {n_pl_hours:,} of {n_hour:,} hours")
    return {"sums": sums, "counts": counts, "n_state": n_state,
            "time": times[use_step], "seasons": list(used),
            "season_of_hour": np.asarray(layout["seasons"])[s_idx[use_step]],
            "n_pl_hours": n_pl_hours, "pl_matched": pl_matched, "site_lat": site_lat, "site_lon": site_lon}


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def hourly_mean(R: dict, q: str, s: str, slot: int) -> np.ndarray:
    """Mean over the slot's cells in the state; NaN where there are none."""
    n = R["counts"][(q, s)][:, slot]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, R["sums"][(q, s)][:, slot] / np.where(n > 0, n, 1), np.nan)


def write_all(R: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell_counts").mkdir(exist_ok=True)
    span = f"{R['seasons'][0]}_{R['seasons'][-1] + 1}"
    dates = pd.to_datetime(R["time"]).strftime("%Y-%m-%d %H:%M:%S")
    written = []
    for si, name in enumerate(SLOTS):
        for fname, cols in FILES.items():
            df = pd.DataFrame({"date": dates})
            for q, s, col in cols:
                df[col] = hourly_mean(R, q, s, si)
            path = out_dir / f"ERA5_{fname}_perHour_{span}_{name}.csv"
            df.to_csv(path, index=False, float_format="%.2f")   # template: 2 dp
            written.append(path)
        cnt = pd.DataFrame({"date": dates})
        for s in STATES:
            cnt[s] = R["n_state"][s][:, si]
        for s in CLOUD_STATES:
            cnt[f"cloudTemp_{s}"] = R["counts"][("tcld", s)][:, si]
        path = out_dir / "cell_counts" / f"ERA5_nCells_perHour_{span}_{name}.csv"
        cnt.to_csv(path, index=False)
        written.append(path)
    return written


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_against_earlier_files(R: dict) -> None:
    """UTQ against the netCDF file, class state counts against the hour CSVs."""
    import xarray as xr

    nc = HERE / "era5_utqiagvik_OV_figs1-2-4_filtered_2014-2025.nc"
    if nc.exists():
        d = xr.open_dataset(nc)
        t_nc = d["valid_time"].values
        row = np.searchsorted(R["time"], t_nc)
        for s, flag in (("liquid_containing", "is_liquid_containing"),
                        ("liquid_only", "is_liquid_only"), ("ice_only", "is_ice_only")):
            mine = R["n_state"][s][:, 0] == 1
            ref = np.zeros_like(mine)
            ref[row[d[flag].values == 1]] = True
            assert np.array_equal(mine, ref), f"UTQ {s} hours differ from the netCDF"
        lc = d["is_liquid_containing"].values == 1
        for q, var, scale in (("lwp", "tclw", 1000.0), ("dlr", "msdwlwrf", 1.0),
                              ("dlr_cs", "msdwlwrfcs", 1.0)):
            got = hourly_mean(R, q, "liquid_containing", 0)[row[lc]]
            ref = d[var].values[lc] * scale
            assert np.allclose(got, ref, rtol=1e-6, atol=1e-6), f"UTQ {q} differs"
        print("  UTQ states and values match the netCDF file exactly")

    hc = HERE / "era5_hour_count_csv"
    for name in SLOTS[1:]:
        f = hc / f"ERA5_totalHours_perSeason_2014_2025_{name}.csv"
        if not f.exists():
            continue
        ref = pd.read_csv(f)
        si = SLOTS.index(name)
        for k, y in enumerate(R["seasons"]):
            sel = R["season_of_hour"] == y
            for s in CLOUD_STATES:
                got = int(R["n_state"][s][sel, si].sum())
                assert got == int(ref[s].iloc[k]), f"{name} {y} {s}: {got} vs {ref[s].iloc[k]}"
    print("  class cell-hours match era5_hour_count_csv exactly")


# ----------------------------------------------------------------------------
# readME
# ----------------------------------------------------------------------------
def write_readme(R: dict, args: argparse.Namespace, out_dir: Path) -> Path:
    span = f"{R['seasons'][0]}_{R['seasons'][-1] + 1}"
    # Share of each season's hours with pressure-level data.
    cov = []
    for y in R["seasons"]:
        sel = R["season_of_hour"] == y
        frac = R["pl_matched"][sel].mean()
        if frac > 0:
            cov.append(f"{y}/{(y + 1) % 100:02d} ({100 * frac:.0f}% of its hours)")
    n_clear = R["n_state"]["clear_sky"][:, 0].sum()
    text = f"""HOURLY ERA5 LWP, CLOUD TEMPERATURE AND DLR BY SKY STATE AND SURFACE CLASS
=========================================================================

Andrew Buggee, Scripps Institution of Oceanography, UC San Diego
Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_hourly_values_csvs.py (repo {git_describe()})

Companion to ../era5_hour_count_csv/readME.txt, which describes the ERA5
data, the filters, the surface classes and the plot colours. The hours and
cloud types here are the same ones counted there.


1. LAYOUT
---------
One row per hour, every hour of 1 Oct - 31 Mar, {R['seasons'][0]}/{(R['seasons'][0] + 1) % 100:02d}
through {R['seasons'][-1]}/{(R['seasons'][-1] + 1) % 100:02d} ({len(R['time']):,} hours, UTC; the time stamp is the
end of the hour for the fluxes). A cell is blank when no grid cell of the class
was in that sky state in that hour.

  ERA5_LWP_perHour_{span}_<CLASS>.csv        g m-2
      date, liquid_containing, liquid_only, ice_only
  ERA5_CloudTemp_perHour_{span}_<CLASS>.csv  deg C
      date, liquid_containing, liquid_only, ice_only
  ERA5_DLR_perHour_{span}_<CLASS>.csv        W m-2
      date, liquid_containing, liquid_only, ice_only, clear_sky,
      clear_sky_calc_liquid_containing, clear_sky_calc_liquid_only,
      clear_sky_calc_ice_only

  <CLASS> = UTQ | Land | OpenOcean | PackIce

  cell_counts/ERA5_nCells_perHour_{span}_<CLASS>.csv
      number of grid cells in each sky state in each hour (the cells behind
      each value), and the number of those with a cloud temperature.


2. SKY STATES
-------------
  liquid containing  tcc >= {args.min_cloud_fraction:g}, LWP or IWP above {args.min_lwp:g} g m-2,
                     tp < {args.precip_rate_max:g} mm hr-1, IWP/CWP < {args.ice_fraction_min:.2f}
  liquid only        as above with LWP/CWP >= {args.liquid_fraction_min:.2f} (a subset of liquid containing)
  ice only           as above with IWP/CWP >= {args.ice_fraction_min:.2f}
  clear sky          LWP <= {args.min_lwp:g} and IWP <= {args.min_iwp:g} g m-2, tcc < {DEFAULT_CLEAR_TCC_MAX:g},
                     tp < {args.precip_rate_max:g} mm hr-1

CWP = LWP + IWP after setting each path at or below its floor to zero.
Clear sky is rare in ERA5 at UTQ: {int(n_clear):,} of {len(R['time']):,} hours.


3. QUANTITIES
-------------
  LWP        1000 x tclw, total column cloud liquid water (g m-2).
  DLR        msdwlwrf, surface downward longwave, mean over the hour (W m-2).
  clear_sky_calc_<state>
             msdwlwrfcs, ERA5's clear-sky surface downward longwave: the same
             radiation calculation with the clouds removed, for the hours of
             that state (W m-2). DLR - clear_sky_calc is the longwave cloud
             radiative effect at the surface.
  CloudTemp  from ERA5 pressure-level data (temperature, cloud liquid water
             content clwc and cloud ice water content ciwc on 23 levels,
             1000-200 hPa): the condensate-weighted mean temperature of the
             column,

                 T = sum_l (q_l dp_l T_l) / sum_l (q_l dp_l)

             with q = clwc (liquid-weighted) for liquid containing and liquid
             only, and q = ciwc (ice-weighted) for ice only; dp_l is the
             pressure thickness of level l, clipped at the surface. It is the
             temperature of the cloud's liquid (or ice), weighted toward where
             most of it is -- not cloud-base or cloud-top temperature. A column
             with less than {CLOUD_T_MIN_LWP_G:g} g m-2 of the weighting species on the
             pressure levels has no value.

             COVERAGE in this version: pressure-level data were available for
             {'; '.join(cov) or 'no hours'}.
             CloudTemp is blank for every other hour.


4. ONE VALUE PER HOUR
---------------------
UTQ is the single ERA5 grid cell containing the ARM site ({R['site_lat']:.2f} N, {abs(R['site_lon']):.2f} W):
its value is that cell's. For Land, OpenOcean and PackIce the value is the
unweighted mean over every grid cell of the class in that sky state in that
hour. A box plot of these hourly values gives each HOUR equal weight; to give
each grid-cell-hour equal weight instead, weight each value by its count in
cell_counts/. OpenOcean has no cells at all in most mid-winter hours, so its
Dec-Mar columns are mostly blank.
"""
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--storage", default=SETTINGS["storage"],
                   help="local (default), or glade / s3 / aws to read NCAR's ERA5 "
                        "copy (fills CloudTemp for every season)")
    opts = p.parse_args(argv)

    args = build_args({**SETTINGS, "storage": opts.storage})
    print("=" * 72)
    print("Hourly ERA5 LWP / cloud temperature / DLR by sky state and class")
    print("=" * 72)
    R = accumulate(args)
    check_against_earlier_files(R)
    written = write_all(R, opts.out_dir)
    readme = write_readme(R, args, opts.out_dir)
    print(f"\n  -> {len(written)} CSV files + {readme.name} in {opts.out_dir}")


if __name__ == "__main__":
    main()
