#!/usr/bin/env python3
"""Build an hourly (or 10-min) surface-energy-budget dataset from the NSA C1 observations.

This is the processing step that follows scripts/download_nsa_seb_data.py and
produces the observational file that an ERA5 grid cell (hourly, from
ERA5/surface_energy_budget/download_era5_seb.py + seb_terms.py) is compared
against. Variable names and sign conventions follow seb_terms.py:
radiative net fluxes positive downward, SH and LH positive UPWARD, G positive
toward the surface (Sledd et al. 2025, Eq. 1).

What goes in, at what cadence, and what comes out (--average 1h | 10min)
-------------------------------------------------
1-min   QCRAD (LW/SW up/down), SIRS (second LWD pyrgeometer), GNDIRT (skin
        brightness T), MET (2-m T/RH, 10-m wind, p)
        -> seb.compute_seb_terms()      Eq. (1) radiative terms, skin T
        -> bulk_flux.bulk_fluxes()      SH, LH (positive upward)
        -> hourly means of every term (and the number of valid minutes)
hourly  ground_flux: G as the Eq. (5) residual, and G from the skin-
        temperature history (thermal inertia, Wang & Bras 1999)
20 s / 1 s   MWRRET and MWR3C LWP, PWV -> hourly means, merged with provenance
4 s     ARSCL boundaries -> hourly cloud fraction, lowest base, highest top
1 min   CLDTYPE reflectivity -> hourly IWP (IWC = a Z^b), precipitation rate
~6 h    raw sondes -> temperature at the hourly-mean lowest cloud base

Cloud PHASE (THERMOCLDPHASE, 70 MB/day) is deliberately not folded in here:
it exists only through 2026-01-20, and the per-hour phase statistics you want
depend on the question (lowest layer? whole column? liquid-containing
fraction?). Use arm_nsa.readers.read_timeseries("thermocldphase", ...) with
config.THERMO_* codes, or shupe_turner.py-style scene rules, on top of this
file; cloud_water.iwp_from_reflectivity accepts the phase mask.

Every optional input that is missing on disk is skipped with a message, so
the script runs on the `core` tier alone (radiative terms, bulk fluxes, G,
LWP, cloud boundaries) and adds IWP / precipitation / cloud-base temperature
when `cldtype` and `sonde` are present.

Examples
--------
python scripts/build_nsa_seb_hourly.py                       # 2025/26 season
python scripts/build_nsa_seb_hourly.py --season 2024 --z0 1e-3
python scripts/build_nsa_seb_hourly.py --start 2025-12-01 --end 2025-12-31 \\
    --out data/processed/nsa_seb_hourly_dec2025.nc
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import (
    bulk_flux,
    cloud_water,
    config,
    ground_flux,
    mwr,
    seb,
    sonde,
    surface,
)  # noqa: E402
from arm_nsa.cli import add_data_root_argument, apply_data_root  # noqa: E402

DEFAULT_SEASON_START_YEAR = 2025
# Averaging window of the product. "1h" matches ERA5; "10min" matches the
# 10-min observational resolution of Sledd et al. (2025). Set by --average.
AVERAGE = "1h"
AVERAGE_CHOICES = ("1h", "10min")
# Fraction of the 60 one-minute samples an hour needs before its mean is kept;
# below this the hour is NaN so a single surviving minute cannot pose as an
# hourly mean.
MIN_VALID_FRACTION = 0.5
# Sonde launch must be within this many hours of the hour it is applied to.
SONDE_MAX_AGE_H = 3.0


def cold_season_bounds(start_year: int) -> tuple[str, str]:
    return (
        f"{start_year}-{config.COLD_SEASON_START_MONTH_DAY}",
        f"{start_year + 1}-{config.COLD_SEASON_END_MONTH_DAY}",
    )


def hourly_mean(
    da: xr.DataArray, min_fraction: float = MIN_VALID_FRACTION
) -> xr.DataArray:
    """Hourly mean that is NaN unless at least `min_fraction` of samples are valid."""
    grp = da.resample(time=AVERAGE)
    mean = grp.mean(skipna=True)
    count = np.isfinite(da).resample(time=AVERAGE).sum()
    total = xr.ones_like(da, dtype=float).resample(time=AVERAGE).sum()
    frac = count / total.where(total > 0)
    out = mean.where(frac >= min_fraction)
    out.attrs = dict(da.attrs)
    return out


def _fill_gaps_linear(da: xr.DataArray) -> xr.DataArray:
    """Linearly bridge NaN gaps (constant at the ends) with plain numpy.

    The thermal-inertia integral needs a gap-free series; hours that were NaN
    in the input are masked back out of its OUTPUT by the caller, so the fill
    only influences the history seen by later hours. Written with np.interp
    rather than xarray.interpolate_na to avoid the optional bottleneck
    dependency that the latter needs for gap-length limits.
    """
    v = np.asarray(da.values, dtype=float)
    good = np.isfinite(v)
    if good.sum() < 2:
        raise ValueError("fewer than two valid skin-temperature hours")
    idx = np.arange(v.size)
    filled = np.interp(idx, idx[good], v[good])
    return da.copy(data=filled)


def _month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-month sub-ranges (inclusive strings)."""
    import calendar

    s = dt.date.fromisoformat(start_date)
    e = dt.date.fromisoformat(end_date)
    out = []
    cur = s
    while cur <= e:
        last = dt.date(cur.year, cur.month, calendar.monthrange(cur.year, cur.month)[1])
        stop = min(last, e)
        out.append((cur.isoformat(), stop.isoformat()))
        cur = stop + dt.timedelta(days=1)
    return out


def _try(label: str, fn, *args, **kwargs):
    """Run an optional reader; report and return None when its files are absent."""
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as err:
        print(f"  [skip] {label}: {str(err).splitlines()[0]}")
        return None


def build(
    start_date: str, end_date: str, z0_m: float, verbose: bool = False
) -> xr.Dataset:
    print(f"Reading 1-min surface streams {start_date}..{end_date} ...")
    qcrad = surface.read_qcrad(start_date, end_date, verbose=verbose)
    met = surface.read_met(start_date, end_date, verbose=verbose)
    gndirt = _try(
        "gndirt (skin T from LWU instead)", seb.read_gndirt, start_date, end_date
    )
    sirs = _try(
        "sirs (LWD = QCRAD pyrgeometer 1 only)", seb.read_sirs, start_date, end_date
    )

    # --- Eq. (1) radiative terms and skin temperature at 1 min --------------
    terms = seb.compute_seb_terms(qcrad, gndirt=gndirt, met=met, sirs=sirs)

    # --- bulk turbulent fluxes at 1 min -----------------------------------
    print("Bulk fluxes (MOST, Grachev 2007 stable, Andreas 1987 z_T) ...")
    fl = bulk_flux.bulk_fluxes(
        t_skin_k=terms["t_skin_K"],
        t_air_k=met["temp_2m_c"] + 273.15,
        rh_air_pct=met["rh_2m_pct"],
        wspd_m_s=met["wspd_m_s"],
        p_pa=met["pres_kpa"] * 1.0e3,
        z_u_m=10.0,
        z_t_m=2.0,
        z0_m=z0_m,
    )
    terms = seb.compute_seb_terms(
        qcrad,
        gndirt=gndirt,
        met=met,
        sirs=sirs,
        sh_up_w_m2=fl["sh_up_W_m2"],
        lh_up_w_m2=fl["lh_up_W_m2"],
    )

    # --- hourly means -----------------------------------------------------
    print("Hourly means ...")
    out: Dict[str, xr.DataArray] = {}
    for name in terms.data_vars:
        if name in ("lwd_source", "t_skin_source"):
            # Mode is meaningless for a flag; keep the worst (largest) code.
            out[name] = (
                terms[name].resample(time=AVERAGE).max().fillna(0).astype("int8")
            )
            out[name].attrs = dict(terms[name].attrs)
            continue
        out[name] = hourly_mean(terms[name])
    for name in (
        "ustar_m_s",
        "zeta",
        "c_d",
        "c_h",
        "c_e",
        "rho_kg_m3",
        "q_air_kg_kg",
        "q_sfc_kg_kg",
    ):
        out[name] = hourly_mean(fl[name])
    out["n_valid_lwd"] = (
        np.isfinite(terms["lwd_W_m2"]).resample(time=AVERAGE).sum().astype("int16")
    )
    out["n_valid_sh"] = (
        np.isfinite(terms["sh_up_W_m2"]).resample(time=AVERAGE).sum().astype("int16")
    )
    out["n_valid_lwd"].attrs = {"long_name": "1-min LWD samples in the hour"}
    out["n_valid_sh"].attrs = {"long_name": "1-min bulk SH samples in the hour"}
    out["wind_speed_10m_m_s"] = hourly_mean(met["wspd_m_s"])
    out["rh_2m_pct"] = hourly_mean(met["rh_2m_pct"])
    out["p_sfc_Pa"] = hourly_mean(met["pres_kpa"] * 1.0e3)
    out["p_sfc_Pa"].attrs = {"units": "Pa", "long_name": "station pressure"}

    # --- G estimates at hourly resolution ------------------------------------
    print("Ground heat flux estimates ...")
    out["g_residual_W_m2"] = ground_flux.ground_flux_residual(out["na_flux_W_m2"])
    ts_h = out["t_skin_K"]
    g_ti = ground_flux.ground_flux_thermal_inertia(_fill_gaps_linear(ts_h))
    out["g_thermal_inertia_W_m2"] = g_ti.where(np.isfinite(ts_h))

    # --- LWP / PWV ----------------------------------------------------------
    print("Liquid water path ...")
    hourly_time = out["lwd_W_m2"]["time"]
    m = _try("mwr (MWRRET LWP/PWV)", mwr.read_mwr, start_date, end_date)
    m3 = _try("mwr3c (3-channel MWR LWP)", seb.read_mwr3c, start_date, end_date)
    lwp_primary = (
        hourly_mean(m["lwp_g_m2"], 0.25).reindex(time=hourly_time)
        if m is not None
        else None
    )
    lwp_secondary = (
        hourly_mean(m3["lwp_g_m2"], 0.25).reindex(time=hourly_time)
        if m3 is not None
        else None
    )
    if lwp_primary is not None and lwp_secondary is not None:
        merged = cloud_water.merge_lwp(lwp_primary, lwp_secondary, time=hourly_time)
        out["lwp_g_m2"] = merged["lwp_g_m2"]
        out["lwp_source"] = merged["lwp_source"]
        out["lwp_mwrret_g_m2"] = lwp_primary
        out["lwp_mwr3c_g_m2"] = lwp_secondary
    elif lwp_primary is not None:
        out["lwp_g_m2"] = lwp_primary
    elif lwp_secondary is not None:
        out["lwp_g_m2"] = lwp_secondary
    if m is not None:
        out["pwv_cm"] = hourly_mean(m["pwv_cm"], 0.25).reindex(time=hourly_time)

    # --- cloud boundaries -----------------------------------------------------
    print("Cloud boundaries ...")
    b = _try(
        "arsclbnd (cloud boundaries)", seb.read_arscl_boundaries, start_date, end_date
    )
    lowest_base_hourly: Optional[xr.DataArray] = None
    if b is not None:
        valid = b["sky_flag"] < 3
        cloudy = (b["sky_flag"] == 0).where(valid)
        out["cloud_fraction"] = hourly_mean(cloudy.astype(float), 0.25).reindex(
            time=hourly_time
        )
        out["cloud_fraction"].attrs = {
            "units": "1",
            "long_name": "fraction of 4-s ARSCL samples with a cloud base",
        }
        lowest_base_hourly = hourly_mean(b["cloud_base_best_estimate_m"], 0.1).reindex(
            time=hourly_time
        )
        out["cloud_base_lowest_m"] = lowest_base_hourly
        out["cloud_top_highest_m"] = hourly_mean(b["cloud_top_max_m"], 0.1).reindex(
            time=hourly_time
        )
        out["cloud_base_lowest_m"].attrs = {
            "units": "m",
            "long_name": "hourly mean lowest cloud base AGL (cloudy samples)",
        }
        out["cloud_top_highest_m"].attrs = {
            "units": "m",
            "long_name": "hourly mean highest cloud top AGL (cloudy samples)",
        }

    # --- IWP and precipitation from CLDTYPE -----------------------------------
    # Processed one calendar month at a time: the 1-min x 596-gate reflectivity
    # is ~1.3 GB per month in memory, so a whole season at once is not safe.
    print("Ice water path and precipitation ...")
    iwp_pieces, precip_pieces = [], []
    for m_start, m_end in _month_ranges(start_date, end_date):
        c = _try(
            f"cldtype {m_start[:7]} (IWP, precip)", seb.read_cldtype, m_start, m_end
        )
        if c is None:
            continue
        iwp = cloud_water.iwp_from_reflectivity(c["reflectivity_dbz"])
        iwp_pieces.append(hourly_mean(iwp, 0.25))
        precip_pieces.append(hourly_mean(c["precip_rate_mm_min"], 0.25) * 60.0)
        del c, iwp
    if iwp_pieces:
        iwp_h = xr.concat(iwp_pieces, dim="time").sortby("time")
        _, idx = np.unique(iwp_h["time"].values, return_index=True)
        out["iwp_g_m2"] = iwp_h.isel(time=idx).reindex(time=hourly_time)
        out["iwp_g_m2"].attrs = {
            "units": "g m-2",
            "long_name": "ice water path from CLDTYPE reflectivity (IWC = a Z^b), no phase mask",
            "a": config.IWC_PREFACTOR_A,
            "b": config.IWC_EXPONENT_B,
            "uncertainty": "factor ~2 (Z-IWC relation); see cloud_water.py",
        }
        pr_h = xr.concat(precip_pieces, dim="time").sortby("time")
        out["precip_rate_mm_h"] = pr_h.isel(time=idx).reindex(time=hourly_time)
        out["precip_rate_mm_h"].attrs = {
            "units": "mm/h",
            "long_name": "radar-derived precipitation rate (CLDTYPE)",
        }

    # --- cloud-base temperature from the nearest sonde -------------------------
    print("Cloud-base temperature from sondes ...")
    s = _try("sonde (cloud-base temperature)", sonde.read_sondes, start_date, end_date)
    if s is not None and lowest_base_hourly is not None:
        prof = s["tdry_c"].rename({"launch_time": "time"})
        prof = prof.reindex(
            time=hourly_time,
            method="nearest",
            tolerance=np.timedelta64(int(SONDE_MAX_AGE_H * 60), "m"),
        )
        cbh_msl = lowest_base_hourly + config.SITE_ALT_M
        t_cb = prof.interp(height=cbh_msl).drop_vars("height", errors="ignore") + 273.15
        out["t_cloud_base_K"] = t_cb
        out["t_cloud_base_K"].attrs = {
            "units": "K",
            "long_name": f"temperature at the hourly-mean lowest cloud base (nearest sonde within {SONDE_MAX_AGE_H:g} h)",
        }

    ds = xr.Dataset(out)
    ds.attrs.update(
        {
            "title": "ARM NSA C1 (Utqiagvik) hourly surface energy budget from observations",
            "reference": "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578, Eq. (1)",
            "convention": (
                "radiative net fluxes positive downward; SH, LH positive upward; "
                "G positive toward the surface"
            ),
            "period": f"{start_date} .. {end_date}",
            "averaging_window": AVERAGE,
            "bulk_flux": f"MOST, stable=grachev2007, scalar roughness=andreas1987, z0={z0_m} m, z_u=10 m, z_t=2 m",
            "lwd": (
                "two-pyrgeometer best estimate (seb.best_estimate_lwdn)"
                if sirs is not None
                else "QCRAD pyrgeometer 1"
            ),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "site_lat_deg": config.SITE_LAT_DEG,
            "site_lon_deg": config.SITE_LON_DEG,
        }
    )
    return ds


def monthly_summary(ds: xr.Dataset) -> None:
    cols = [
        c
        for c in (
            "lwd_W_m2",
            "lwu_W_m2",
            "lw_net_W_m2",
            "swn_W_m2",
            "sh_up_W_m2",
            "lh_up_W_m2",
            "na_flux_W_m2",
            "g_thermal_inertia_W_m2",
            "t_skin_K",
            "t_2m_K",
            "lwp_g_m2",
            "iwp_g_m2",
            "cloud_fraction",
            "t_cloud_base_K",
        )
        if c in ds
    ]
    print("\nMonthly means (hours with data):")
    print("  month " + " ".join(f"{c[:14]:>14s}" for c in cols))
    for mo, g in ds.groupby("time.month"):
        vals = []
        for c in cols:
            v = g[c].values.astype(float)
            vals.append(
                f"{np.nanmean(v):14.2f}" if np.isfinite(v).any() else f"{'nan':>14s}"
            )
        print(f"  {mo:5d} " + " ".join(vals))
    n_h = ds.sizes["time"]
    print(
        f"\n{n_h} hours; LWD valid {100*np.isfinite(ds['lwd_W_m2'].values).mean():.1f}%, "
        f"SH valid {100*np.isfinite(ds['sh_up_W_m2'].values).mean():.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, default=None, metavar="YYYY")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--z0",
        type=float,
        default=bulk_flux.DEFAULT_Z0_M,
        help="aerodynamic roughness length [m]",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output netCDF path (default data/processed/nsa_seb_hourly_<start>_<end>.nc)",
    )
    parser.add_argument(
        "--average",
        choices=AVERAGE_CHOICES,
        default="1h",
        help="averaging window: 1h (ERA5 cadence, default) or 10min (Sledd et al. 2025 cadence)",
    )

    parser.add_argument("--verbose", action="store_true")
    add_data_root_argument(parser)
    args = parser.parse_args()
    if args.season is not None and (args.start or args.end):
        parser.error("--season cannot be combined with --start/--end")
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be given together")
    if args.start:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = cold_season_bounds(
            args.season if args.season is not None else DEFAULT_SEASON_START_YEAR
        )
    if args.data_root is not None:
        apply_data_root(args.data_root)

    global AVERAGE
    AVERAGE = args.average
    ds = build(start_date, end_date, z0_m=args.z0, verbose=args.verbose)
    label = {"1h": "hourly", "10min": "10min"}[args.average]
    out_path = (
        Path(args.out)
        if args.out
        else config.PROCESSED_DATA_DIR / f"nsa_seb_{label}_{start_date}_{end_date}.nc"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ds.data_vars
        if ds[v].dtype.kind == "f"
    }
    ds.to_netcdf(out_path, encoding=encoding)
    print(
        f"\nWritten: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(ds.data_vars)} variables)"
    )
    monthly_summary(ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
