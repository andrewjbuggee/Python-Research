"""Synthetic daily files for the three cloud-property datastreams.

They follow the variable names, units and QC conventions in:
  MFRSRCLDOD  DOE/SC-ARM-TR-047, Table 2      (epcmfrsrcldod1minM1.c1)
  SPHOTCOD    DOE/SC-ARM-TR-317, Table 4      (epcsphotcod2chiuM1.c1)
  MWRLOS      DOE/SC-ARM-TR-016, Tables 1-6   (epcmwrlosM1.b1)

They are NOT real data. They exist so the loading, filtering, collocation
and plotting code can be tested before (or without) the real files. The
synthetic "truth" has known differences built in, so a test can check that
the pipeline recovers them:
  * SPHOT tau is biased +15% relative to MFRSR (cf. Sookdar et al. 2025)
  * MFRSR r_e comes from the MWR LWP (r_e = 3 LWP / (2 rho_w tau)), or the
    8 um default where the MWR LWP < 20 g m-2
  * the cloud deck breaks up in the afternoon (cloud fraction < 0.9)
  * a 40-minute rain/wet-window event corrupts the MWR on one day

Usage:  python tests/synthetic_cloud_vaps.py <data_root> [--start 2023-07-01] [--days 6] [--seed 7]
Files go to <data_root>/arm/<datastream>/, where epcape.combine looks for
complete files.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np

EPOCH = dt.datetime(1970, 1, 1)
MISSING = -9999.0

QC_BITS_STANDARD = {
    "bit_1_description": "Value is equal to missing_value.",
    "bit_1_assessment": "Bad",
    "bit_2_description": "Value is less than the valid_min.",
    "bit_2_assessment": "Bad",
    "bit_3_description": "Value is greater than the valid_max.",
    "bit_3_assessment": "Bad",
    "bit_4_description": "Difference between current and previous values exceeds valid_delta.",
    "bit_4_assessment": "Indeterminate",
    "flag_method": "bit",
}


def _sun(hours_utc: np.ndarray) -> np.ndarray:
    """Crude cos(solar zenith angle) for La Jolla in July: day from 13 to 03 UTC."""
    h = (hours_utc - 13.0) % 24.0
    return np.where(h < 14.0, 0.92 * np.sin(np.pi * h / 14.0), 0.0)


def _truth(hours_utc: np.ndarray, rng: np.random.Generator, day_index: int):
    """Smooth 'true' cloud fields: optical depth, r_e (um), LWP (g m-2), cloud fraction."""
    # AR(1) red noise for realistic temporal structure
    n = hours_utc.size
    eps = rng.normal(size=n)
    red = np.empty(n)
    red[0] = eps[0]
    phi = 0.995
    for i in range(1, n):
        red[i] = phi * red[i - 1] + np.sqrt(1 - phi**2) * eps[i]
    h_local = (hours_utc - 7.0) % 24.0  # PDT
    tau = np.clip(18.0 + 8.0 * np.cos(2 * np.pi * (h_local - 6) / 24) + 5.0 * red + 2 * day_index, 1.0, None)
    r_e = np.clip(10.0 + 1.5 * np.sin(2 * np.pi * h_local / 24 + day_index) + 0.8 * red, 4.0, None)
    lwp = (2.0 / 3.0) * tau * r_e  # g m-2 (rho_w = 1e6 g m-3, r_e in um)
    # deck breaks up 13-17 local time
    cf = np.where((h_local > 13) & (h_local < 17), 0.55 + 0.2 * np.sin(np.pi * (h_local - 13) / 4), 1.0)
    return tau, r_e, lwp, cf


def _var(nc, name, dtype, dims, data, **attrs):
    fill = attrs.pop("_fill", None)
    v = nc.createVariable(name, dtype, dims, fill_value=fill)
    v.setncatts(attrs)
    v[...] = data
    return v


def _time_vars(nc, start: dt.datetime, offsets: np.ndarray):
    midnight = dt.datetime(start.year, start.month, start.day)
    _var(
        nc,
        "base_time",
        "i4",
        (),
        int((start - EPOCH).total_seconds()),
        long_name="Base time in Epoch",
        units="seconds since 1970-1-1 0:00:00 0:00",
    )
    _var(
        nc,
        "time_offset",
        "f8",
        ("time",),
        offsets,
        long_name="Time offset from base_time",
        units=f"seconds since {start:%Y-%m-%d %H:%M:%S} 0:00",
    )
    _var(
        nc,
        "time",
        "f8",
        ("time",),
        (start - midnight).total_seconds() + offsets,
        long_name="Time offset from midnight",
        units=f"seconds since {start:%Y-%m-%d} 00:00:00 0:00",
        standard_name="time",
    )


def _location(nc):
    _var(nc, "lat", "f4", (), 32.867, long_name="North latitude", units="degree_N")
    _var(nc, "lon", "f4", (), -117.257, long_name="East longitude", units="degree_E")
    _var(nc, "alt", "f4", (), 10.0, long_name="Altitude above mean sea level", units="m")


def _qc(values: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    q = np.zeros(values.shape, dtype="i4")
    q |= np.where(values == MISSING, 1, 0).astype("i4")
    if vmin is not None:
        q |= np.where((values != MISSING) & (values < vmin), 2, 0).astype("i4")
    if vmax is not None:
        q |= np.where((values != MISSING) & (values > vmax), 4, 0).astype("i4")
    return q


def write_day(
    root: Path, day: dt.date, day_index: int, seed: int, rain: bool = False, sphot: bool = True
) -> None:
    rng = np.random.default_rng(seed + 1000 * day_index)
    start = dt.datetime.combine(day, dt.time())

    # ---- MWR LOS: 20 s, all day ------------------------------------------
    off_w = np.arange(0, 86400, 20.0)
    h_w = off_w / 3600.0
    tau_w, re_w, lwp_true_w, _ = _truth(h_w, rng, day_index)
    lwp_mwr = lwp_true_w + rng.normal(0, 15, h_w.size)  # g m-2, ~15 g m-2 noise
    tb23 = 25 + 0.03 * lwp_true_w + rng.normal(0, 0.3, h_w.size)
    tb31 = 15 + 0.05 * lwp_true_w + rng.normal(0, 0.3, h_w.size)
    wet = np.zeros(h_w.size, dtype="i4")
    if rain:
        r = (h_w > 20.0) & (h_w < 20.0 + 40 / 60)
        wet[r] = 1
        lwp_mwr[r] = 1500 + rng.normal(0, 200, r.sum())
        tb31[r] = 130
        tb23[r] = 110
    lwp_mwr[100:110] = -120.0  # below valid_min = -3 x RMS (-0.0092 cm = -92 g m-2): trips that QC test
    liq_cm = (lwp_mwr / 1e4).astype("f4")
    liq_cm[5:8] = MISSING
    ds = root / "arm" / "epcmwrlosM1.b1"
    ds.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(ds / f"epcmwrlosM1.b1.{day:%Y%m%d}.000000.cdf", "w", format="NETCDF3_CLASSIC") as nc:
        nc.setncatts(
            {
                "datastream": "epcmwrlosM1.b1",
                "site_id": "epc",
                "facility_id": "M1",
                "liquid_retrieval_rms_accuracy": "0.003083",
                "history": "synthetic test file",
            }
        )
        nc.createDimension("time", None)
        _time_vars(nc, start, off_w)
        _var(
            nc,
            "liq",
            "f4",
            ("time",),
            liq_cm,
            long_name="Total liquid water along LOS path",
            units="cm",
            missing_value=np.float32(MISSING),
        )
        _var(
            nc,
            "qc_liq",
            "i4",
            ("time",),
            _qc(liq_cm, vmin=-3 * 0.003083, vmax=1.0),
            long_name="Quality check results on field: Total liquid water along LOS path",
            units="1",
            **QC_BITS_STANDARD,
        )
        vap = (2.2 + 0.2 * np.sin(h_w / 3)).astype("f4")
        _var(
            nc,
            "vap",
            "f4",
            ("time",),
            vap,
            long_name="Total water vapor along LOS path",
            units="cm",
            missing_value=np.float32(MISSING),
        )
        _var(
            nc,
            "qc_vap",
            "i4",
            ("time",),
            _qc(vap, vmin=0),
            long_name="Quality check results on field: vap",
            units="1",
            **QC_BITS_STANDARD,
        )
        _var(nc, "tbsky23", "f4", ("time",), tb23, long_name="23.8 GHz sky brightness temperature", units="K")
        _var(nc, "tbsky31", "f4", ("time",), tb31, long_name="31.4 GHz sky brightness temperature", units="K")
        _var(
            nc,
            "wet_window",
            "i4",
            ("time",),
            wet,
            long_name="Water on Teflon window (1=WET, 0=DRY)",
            units="1",
        )
        _var(
            nc,
            "actel",
            "f4",
            ("time",),
            np.full(h_w.size, 90.0),
            long_name="Actual elevation angle",
            units="deg",
        )
        _location(nc)

    # ---- MFRSRCLDOD: 20 s, daytime retrievals --------------------------------
    off_m = np.arange(0, 86400, 20.0)
    h_m = off_m / 3600.0
    mu0 = _sun(h_m)
    tau_t, re_t, lwp_t, cf = _truth(h_m, rng, day_index)
    day_m = mu0 > 0.05
    # broken cloud: the hemispheric diffuse field no longer matches the zenith column
    tau_m = tau_t * (1 + rng.normal(0, 0.04, h_m.size)) * np.where(cf < 0.9, 0.6, 1.0)
    lwp_mwr_m = np.interp(off_m, off_w, lwp_mwr)
    from_mwr = lwp_mwr_m >= 20.0
    re_m = np.where(from_mwr, 1.5 * lwp_mwr_m / np.maximum(tau_m, 1e-3), 8.0)
    lwp_m_mm = np.where(from_mwr, lwp_mwr_m, (2.0 / 3.0) * tau_m * 8.0) / 1000.0
    tau_m = np.where(day_m, tau_m, MISSING).astype("f4")
    re_m = np.where(day_m, re_m, MISSING).astype("f4")
    lwp_m_mm = np.where(day_m, lwp_m_mm, MISSING).astype("f4")
    tau_avg = tau_m.copy()  # simple stand-in for the 5-min running mean
    ds = root / "arm" / "epcmfrsrcldod1minM1.c1"
    ds.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(
        ds / f"epcmfrsrcldod1minM1.c1.{day:%Y%m%d}.000000.nc", "w", format="NETCDF3_CLASSIC"
    ) as nc:
        nc.setncatts(
            {
                "datastream": "epcmfrsrcldod1minM1.c1",
                "site_id": "epc",
                "facility_id": "M1",
                "history": "synthetic test file",
            }
        )
        nc.createDimension("time", None)
        _time_vars(nc, start, off_m)
        for name, data, units, ln in [
            ("optical_depth_instantaneous", tau_m, "unitless", "Cloud Optical Depth (instantaneous)"),
            (
                "optical_depth_average",
                tau_avg,
                "unitless",
                "Five-Minute Running Average of Cloud Optical Depth",
            ),
            ("effective_radius_instantaneous", re_m, "microns", "Effective Radius (Instantaneous)"),
            ("effective_radius_average", re_m, "microns", "Five-Minute Running Average of Effective Radius"),
        ]:
            _var(
                nc, name, "f4", ("time",), data, long_name=ln, units=units, missing_value=np.float32(MISSING)
            )
            _var(
                nc,
                f"qc_{name}",
                "i4",
                ("time",),
                _qc(data, vmin=0),
                long_name=f"Quality check results on field: {ln}",
                units="1",
                **QC_BITS_STANDARD,
            )
        _var(
            nc,
            "cldtaui_toterror",
            "f4",
            ("time",),
            np.where(day_m, 0.05 * np.abs(tau_m), MISSING),
            long_name="Instantaneous Cloud Tau Total Uncertainty",
            units="unitless",
            missing_value=np.float32(MISSING),
        )
        _var(
            nc,
            "reffi_toterror",
            "f4",
            ("time",),
            np.where(day_m, 0.15 * np.abs(re_m), MISSING),
            long_name="Instantaneous Effective Radius Total Error",
            units="microns",
            missing_value=np.float32(MISSING),
        )
        _var(
            nc,
            "cosine_solar_zenith_angle",
            "f4",
            ("time",),
            mu0,
            long_name="Cosine Solar Zenith Angle",
            units="unitless",
        )
        _var(
            nc,
            "lwp",
            "f4",
            ("time",),
            lwp_m_mm,
            units="mm",
            missing_value=np.float32(MISSING),
            long_name="Total liquid water along LOS path, from MWR or MFRSR with assumed effective radius",
        )
        _var(
            nc,
            "source_lwp",  # name and flag style as in the real EPCAPE files (checked 2026-10-06)
            "i4",
            ("time",),
            np.where(from_mwr, 8, 2),
            long_name="Source for variable: Total liquid water along LOS path",
            units="1",
            flag_method="integer",
            flag_0_description="no_source_available",
            flag_2_description="none: lwp is derived from mfrsr.b1 using the formula "
            "(2/3) * default_re * optical_depth_instantaneous",
            flag_4_description="mwrret1liljclou.c2:phys_lwp",
            flag_8_description="mwrlos.b1:liq",
        )
        _var(
            nc,
            "cloudfraction",
            "f4",
            ("time",),
            np.where(day_m, cf, MISSING),
            units="unitless",
            missing_value=np.float32(MISSING),
            long_name="Estimated Average Fractional Sky Cover over the Hemispheric Dome (cf)",
        )
        _var(
            nc,
            "cloudbasebestestimate",
            "f4",
            ("time",),
            450 + 50 * np.sin(h_m),
            units="m AGL",
            long_name="LASER Cloud Base Height Best Estimate",
        )
        _var(
            nc,
            "ir_temp",
            "f4",
            ("time",),
            np.full(h_m.size, 286.0),
            units="K",
            long_name="IR Brightness Temperature",
        )
        _var(
            nc,
            "surface_albedo",
            "f4",
            ("time",),
            np.full(h_m.size, 0.036),
            units="unitless",
            long_name="Surface Albedo",
        )
        _location(nc)

    if not sphot:
        return
    # ---- SPHOTCOD: one retrieval per 10 min while cloud covers the sun -------
    off_s = np.arange(13.5 * 3600, 26 * 3600, 600.0)
    off_s = off_s[off_s < 86400]
    off_s = off_s + rng.uniform(0, 120, off_s.size)
    h_s = off_s / 3600.0
    mu_s = _sun(h_s)
    tau_ts, re_ts, _, cf_s = _truth(h_s, rng, day_index)
    # recompute truth on the MFRSR grid and interpolate so SPHOT sees the same cloud
    tau_ts = np.interp(off_s, off_m, tau_t)
    re_ts = np.interp(off_s, off_m, re_t)
    cf_s = np.interp(off_s, off_m, cf)
    sun_blocked = rng.random(off_s.size) < np.where(cf_s >= 0.9, 1.0, cf_s)
    keep = (mu_s > 0.1) & sun_blocked
    off_s, h_s, mu_s, tau_ts, re_ts = off_s[keep], h_s[keep], mu_s[keep], tau_ts[keep], re_ts[keep]
    tau_s = (tau_ts * 1.15 * (1 + rng.normal(0, 0.08, off_s.size))).astype("f4")
    re_s = (re_ts + rng.normal(0, 2.5, off_s.size)).astype("f4")
    lwp_s = ((2.0 / 3.0) * tau_s * re_s).astype("f4")
    flag = np.where(rng.random(off_s.size) < 0.1, 1, 0).astype("i4")
    nsol = np.where(flag == 0, rng.integers(3, 6, off_s.size), 0).astype("i4")
    ds = root / "arm" / "epcsphotcod2chiuM1.c1"
    ds.mkdir(parents=True, exist_ok=True)
    first = start + dt.timedelta(seconds=float(off_s[0])) if off_s.size else start
    with netCDF4.Dataset(
        ds / f"epcsphotcod2chiuM1.c1.{day:%Y%m%d}.{first:%H%M%S}.nc", "w", format="NETCDF4"
    ) as nc:
        nc.setncatts(
            {
                "datastream": "epcsphotcod2chiuM1.c1",
                "site_id": "epc",
                "facility_id": "M1",
                "history": "synthetic test file",
            }
        )
        nc.createDimension("time", None)
        nc.createDimension("modis_channel", 7)
        offsets = off_s - (first - start).total_seconds()
        _time_vars(nc, first, offsets)
        _var(
            nc,
            "modis_channel",
            "i4",
            ("modis_channel",),
            np.arange(1, 8),
            long_name="Coordinate variable for modis_channel",
        )
        _var(
            nc,
            "modis_wavelength",
            "f4",
            ("modis_channel",),
            [645, 858, 469, 555, 1240, 1640, 2130],
            units="nm",
            long_name="Central wavelength of modis_channel",
        )
        alb = np.tile(np.array([0.05, 0.12, 0.04, 0.05, 0.10, 0.07, 0.04], "f4"), (off_s.size, 1))
        _var(
            nc,
            "modis_white_sky_albedo",
            "f4",
            ("time", "modis_channel"),
            alb,
            units="unitless",
            long_name="Area average of white sky albedo for modis_channel",
        )
        _var(
            nc,
            "solar_zenith_angle",
            "f4",
            ("time",),
            np.degrees(np.arccos(mu_s)),
            units="degree",
            long_name="Solar zenith angle",
        )
        for name, data, units, ln in [
            ("cloud_optical_depth", tau_s, "unitless", "Cloud optical depth"),
            (
                "cloud_optical_depth_std",
                0.08 * tau_s,
                "unitless",
                "Standard deviation of cloud optical depth",
            ),
            ("effective_radius", re_s, "um", "Effective radius"),
            (
                "effective_radius_std",
                np.full(off_s.size, 2.0, "f4"),
                "um",
                "Standard deviation of effective radius",
            ),
            ("liquid_water_path", lwp_s, "g/m2", "Liquid water path"),
            ("liquid_water_path_std", 0.2 * lwp_s, "g/m2", "Standard deviation of liquid water path"),
        ]:
            _var(nc, name, "f4", ("time",), data, long_name=ln, units=units, _fill=np.float32(MISSING))
        _var(nc, "number_of_solutions", "i4", ("time",), nsol, long_name="Number of Solutions", units="count")
        _var(nc, "retrieval_flag", "i4", ("time",), flag, long_name="Quality check results", units="unitless")
        _location(nc)


def write_campaign(root: Path, start: dt.date, days: int, seed: int = 7) -> None:
    for i in range(days):
        day = start + dt.timedelta(days=i)
        write_day(Path(root), day, i, seed, rain=(i == 1), sphot=(i != 3))  # day 4: no SPHOT file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root")
    p.add_argument("--start", default="2023-07-01")
    p.add_argument("--days", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    write_campaign(Path(a.root), dt.date.fromisoformat(a.start), a.days, a.seed)
    print(f"Synthetic files written under {a.root}/arm/ (seed {a.seed})")
