"""Surface energy budget (SEB) from ARM NSA C1 (Utqiagvik / Barrow) observations.

This is the observational counterpart of
``ERA5/surface_energy_budget/download_era5_seb.py`` + ``seb_terms.py``: the
same Sledd et al. (2025) Equation (1), the same sign conventions, and the same
output variable names, so that an ERA5 grid cell and the NSA C1 instruments can
be compared term by term.

    Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy
    balance responses to radiative forcing in the central Arctic from MOSAiC and
    models. JGR Atmospheres, 130, e2024JD042578.
    https://doi.org/10.1029/2024JD042578

Equation (1), turbulent fluxes POSITIVE UPWARD (away from the surface), G
positive when it delivers energy to the surface from below:

    LWD - LWU + SWD - SWU - SWT - SH - LH + G = M

Equation (5), the "net atmospheric flux" used for model evaluation:

    LWD - LWU + SWD - SWU - SH - LH = M - SWT - G = NA

What NSA C1 measures, term by term (archive checked 2026-09-15)
================================================================
LWD, LWU, SWD, SWU
    Directly measured, 1-min, by the SKYRAD/GNDRAD radiometer stands and
    quality-controlled in the QCRAD1LONG VAP (Long & Shi 2008). This is the
    only part of Eq. (1) that is a genuine measurement at Barrow. Two
    co-located downwelling pyrgeometers give an in-situ LWD uncertainty.
SH, LH
    NOT measured at C1. Every eddy-correlation datastream (ecorsf, 30ecor,
    30qcecor, ecor) was queried at every NSA facility over 1998-2026: they
    exist only at E10, Oliktok Point (~250 km ESE). At Barrow the turbulent
    fluxes must be PARAMETERIZED from measured T_skin (GNDIRT, or inverted
    from LWU), T/RH/wind at 2-40 m (MET + 40-m tower) and pressure -- see
    bulk_flux.py. The E10 ECOR is registered ("ecor_e10") to test that
    parameterization on the same tundra type in the same season.
G
    NOT measured at C1. No SEBS (soil heat-flux plates), no STAMP (soil T /
    moisture profile), no snow-depth or snow-temperature datastream exists
    for C1; SEBS exists only at E10 ("sebs_e10"). See ground_flux.py for the
    three ways to estimate it (residual, snow conduction, thermal inertia).
SWT
    Shortwave transmitted below the surface; zero in polar night and
    negligible under winter snow. Not measured; treated as zero here.
M
    Melt energy; identically zero for a frozen tundra surface in Oct-Mar.

Cloud state (the ERA5 tclw / tciw / cloud temperature / phase analogues)
    LWP    MWRRET ("mwr", physical retrieval, the standard) and the 3-channel
           MWR ("mwr3c", regression, continuous) -- see the registry notes
           for why both are needed in 2025/26.
    IWP    Not retrieved routinely at 2-3 MB/day. Either integrate the
           Z-IWC relation over the 1-min ARSCL reflectivity profile carried by
           CLDTYPE (cloud_water.py) or download MICROBASE (670 MB/day).
    T_cld  ARSCL layer boundaries ("arsclbnd") placed in the sonde profile
           (THERMOCLDPHASE 'sonde_temp' through 2026-01-20, then raw sondes
           or INTERPSONDE).
    Phase  THERMOCLDPHASE ("thermocldphase") -- processed through 2026-01-20
           at the time of writing; nothing later exists yet in the archive.

Reliability ranking used for the tiers below (most to least reliable)
    1. QCRAD broadband fluxes -- calibrated radiometers, ventilated, QC'd.
    2. MET / tower T, RH, p, wind -- conventional sensors, 1-min.
    3. GNDIRT skin temperature -- brightness temperature; needs emissivity and
       reflected-sky correction (skin_temperature_from_irt) before use.
    4. ARSCL cloud boundaries -- radar/lidar detection, robust in winter.
    5. MWRRET LWP -- +-25 g/m^2 theoretical, but with bias episodes and gaps
       this season; MWR3C as the cross-check.
    6. THERMOCLDPHASE phase -- a classification, not a measurement.
    7. Z-based IWP -- factor-of-2 class uncertainty (Z-IWC relations).
    8. Bulk SH/LH -- parameterization; roughness length is a free parameter.
    9. G estimates -- weakest of all; every route needs unmeasured snow
       properties or is the residual of everything else.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import xarray as xr

from . import config
from .readers import _read_one_file, files_in_range, read_timeseries

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Stefan-Boltzmann constant [W m-2 K-4] (CODATA 2018).
SIGMA_SB_W_M2_K4 = 5.670374419e-8

# Broadband longwave emissivity of snow. 0.985 is the value Sledd et al. (2025)
# use to invert LWU for skin temperature over snow and sea ice, and the value
# the ERA5 seb_terms.py uses, so it is kept here for comparability. Laboratory
# and field spectra of snow in the 8-14 um window fall in ~0.98-0.995 depending
# on grain size and viewing angle (e.g. Hori et al. 2006, Remote Sens. Environ.
# 100, 486-502), so treat +-0.01 as the plausible range: at 250 K, d(T_s)/d(eps)
# ~ -T_s/(4 eps) ~ -63 K per unit emissivity, i.e. ~0.6 K per 0.01.
SNOW_EMISSIVITY = 0.985

# Mounting heights at NSA C1, read from the file metadata on 2026-09-15:
# QCRAD labels its field "Upwelling (10 meter) Longwave Hemispheric Irradiance"
# and nsagndirtC1.b1 carries sensor_height = "10m".
LWUP_SENSOR_HEIGHT_M = 10.0
IRT_SENSOR_HEIGHT_M = 10.0


# ---------------------------------------------------------------------------
# SEB datastream tiers and sizing
# ---------------------------------------------------------------------------


class SebStream(NamedTuple):
    """Sizing and coverage metadata for one SEB pipeline key.

    Attributes
    ----------
    key : config.DATASTREAMS key.
    term : Which Eq. (1) term or cloud-state quantity it serves.
    mb_per_day : Measured from one 2025-12-15 file (or the local archive) on
        2026-09-15; used for the --dry-run size estimate.
    cadence : Native time resolution.
    site : "C1" (Barrow) or "E10" (Oliktok Point) -- never mix on one axis.
    coverage_2025_26 : What the ARM Live archive held for 2025-10-01 ..
        2026-03-31 when queried on 2026-09-15. VAPs are processed with a lag
        of weeks to months, so re-query before assuming a gap is permanent.
    """

    key: str
    term: str
    mb_per_day: float
    cadence: str
    site: str
    coverage_2025_26: str


SEB_STREAMS: Dict[str, SebStream] = {
    # --- core: the measured Eq. (1) terms and the bulk-flux inputs ----------
    "qcrad": SebStream(
        "qcrad",
        "LWD, LWU, SWD, SWU (measured)",
        0.46,
        "1 min",
        "C1",
        "c1: complete, 182/182 days; c2 ends 2024-07-17 (reprocessing lag)",
    ),
    "gndirt": SebStream(
        "gndirt",
        "T_skin (IR brightness) for bulk SH/LH",
        0.14,
        "1 min",
        "C1",
        "complete, 183 files",
    ),
    "met": SebStream(
        "met",
        "T_2m, RH, p, U_10m, PWD precip for bulk SH/LH",
        0.38,
        "1 min",
        "C1",
        "complete, 182/182 days (already local)",
    ),
    "twr": SebStream(
        "twr",
        "T, RH, U at 2/10/20/40 m for bulk SH/LH (profile method)",
        0.51,
        "1 min",
        "C1",
        "complete, 182/182 days (already local)",
    ),
    "mwr": SebStream(
        "mwr",
        "LWP, PWV (MWRRET physical retrieval)",
        0.4,
        "20 s",
        "C1",
        "c2 through 2025-12-31 (90 files), c1 165 files to 2026-03-31; "
        "no retrieval 2025-12-15/16, no file 2025-12-14 and 2026-02-15, "
        "median LWP -47 g/m^2 on 2025-11-15 (bias episode)",
    ),
    "arsclbnd": SebStream(
        "arsclbnd",
        "cloud base/top per layer -> cloud temperature",
        2.34,
        "4 s",
        "C1",
        "c0 complete 182/182 days; c1 Oct-Dec 2025 (92 files)",
    ),
    # --- recommended additions: cloud state --------------------------------
    "mwr3c": SebStream(
        "mwr3c",
        "LWP, PWV (3-channel regression), IR sky T",
        21.5,
        "1 s",
        "C1",
        "complete, 184 files (continuous through the MWRRET gaps)",
    ),
    "cldtype": SebStream(
        "cldtype",
        "cloud type, 1-min ARSCL reflectivity profile -> IWP, precip",
        9.1,
        "1 min",
        "C1",
        "181/182 days",
    ),
    "sonde": SebStream(
        "sonde",
        "T/RH/wind profiles (raw launches) -> cloud temperature",
        1.2,
        "~4 launches/day",
        "C1",
        "734 files (~4/day)",
    ),
    "thermocldphase": SebStream(
        "thermocldphase",
        "cloud phase mask + embedded sonde T profile",
        70.0,
        "30 s",
        "C1",
        "c0: 2025-10-02 .. 2026-01-20 (111 files); nothing later " "processed yet",
    ),
    # --- extended: alternatives and the Oliktok reference fluxes ------------
    "interpsonde": SebStream(
        "interpsonde",
        "1-min interpolated T/RH/wind profiles",
        61.3,
        "1 min",
        "C1",
        "181 files (2025-10-02 ..)",
    ),
    "sirs": SebStream(
        "sirs",
        "second LWD pyrgeometer (= LWD uncertainty), case/dome T, QCRAD fallback",
        0.61,
        "1 min",
        "C1",
        "complete. QCRAD c1 LWD is pyrgeometer 1 verbatim; on 2025-12-15 "
        "pyrgeometer 2 read 4.6 W/m^2 lower on average (up to 12 W/m^2)",
    ),
    "ceil": SebStream(
        "ceil",
        "ceilometer cloud base + backscatter",
        6.3,
        "16 s",
        "C1",
        "complete, 182 files",
    ),
    "mplcmask": SebStream(
        "mplcmask",
        "MPL cloud mask + depolarization ratio (phase proxy)",
        22.8,
        "30 s",
        "C1",
        "nsa30smplcmask1zwangC1.c1: 2025-10-01 .. 2026-01-20",
    ),
    "ecor_e10": SebStream(
        "ecor_e10",
        "MEASURED SH, LH, u*, L at OLIKTOK POINT (not Barrow)",
        0.09,
        "30 min",
        "E10",
        "complete",
    ),
    "sebs_e10": SebStream(
        "sebs_e10",
        "MEASURED soil heat flux, soil T/moisture at OLIKTOK POINT",
        0.04,
        "30 min",
        "E10",
        "complete",
    ),
    # --- by explicit key only (too large for any tier) ----------------------
    "microbase": SebStream(
        "microbase",
        "retrieved LWC/IWC profiles -> LWP/IWP",
        670.0,
        "4 s",
        "C1",
        "2025-10-02 .. 2025-12-31 (91 files, ~61 GB)",
    ),
    "skyrad": SebStream(
        "skyrad",
        "raw downwelling fluxes (QCRAD input)",
        0.43,
        "1 min",
        "C1",
        "complete",
    ),
    "gndrad": SebStream(
        "gndrad",
        "raw upwelling fluxes (QCRAD input)",
        0.18,
        "1 min",
        "C1",
        "complete",
    ),
}

# Download tiers, mirroring --var-set {core, recommended, extended} of the ERA5
# script. Order is the download order.
SEB_VARIABLE_SETS: Dict[str, Tuple[str, ...]] = {
    # ~4 MB/day: everything Eq. (1) can be built from at Barrow, plus LWP and
    # cloud boundaries. One cold season is < 1 GB.
    "core": ("qcrad", "gndirt", "met", "twr", "mwr", "arsclbnd"),
    # ~100 MB/day (dominated by THERMOCLDPHASE at 70 MB/day while it lasts):
    # adds the second LWP source, the reflectivity profile for IWP, raw sondes
    # for cloud temperature, the phase mask, and the second downwelling
    # pyrgeometer (SIRS) whose disagreement with the first is the in-situ LWD
    # uncertainty. One cold season ~15 GB.
    "recommended": (
        "qcrad",
        "gndirt",
        "met",
        "twr",
        "mwr",
        "arsclbnd",
        "mwr3c",
        "cldtype",
        "sonde",
        "thermocldphase",
        "sirs",
    ),
    # ~190 MB/day: adds the 1-min interpolated sonde, ceilometer, MPL
    # depolarization, and the Oliktok reference fluxes.
    "extended": (
        "qcrad",
        "gndirt",
        "met",
        "twr",
        "mwr",
        "arsclbnd",
        "mwr3c",
        "cldtype",
        "sonde",
        "thermocldphase",
        "sirs",
        "interpsonde",
        "ceil",
        "mplcmask",
        "ecor_e10",
        "sebs_e10",
    ),
}

# Datastreams that the ARM "measurement" listings suggest for the SEB terms but
# that do NOT exist at NSA C1. Each was queried through the ARM Live API on
# 2026-09-15 over 1998-01-01 .. 2026-09-15 (and, for the flux instruments, at
# facility codes C1, C2, E10-E14 and M1). Kept here so the absence is recorded
# with its evidence rather than rediscovered.
UNAVAILABLE_AT_C1: Dict[str, str] = {
    "ecorsf / 30ecor / 30qcecor / ecor": (
        "eddy-correlation SH and LH: 0 files ever at C1 or C2; exists only at "
        "E10 Oliktok Point (nsaecorsfE10.b1 2024-10 .. present; nsa30ecorE10.b1 "
        "and nsa30qcecorE10.s1 2011-2024)"
    ),
    "sebs": (
        "soil heat-flux plates, soil T/moisture, net radiometer: 0 files at C1; "
        "only nsasebsE10.b1 (2011-09 .. present)"
    ),
    "stamp": "soil temperature/moisture profiles: 0 files at any NSA facility",
    "snow depth (any datastream name tried)": (
        "no snow-depth product; the PWD in nsametC1.b1 reports cumulative snow "
        "as a precipitation amount (pwd_cumul_snow), not a depth"
    ),
    "armbeatm / armbecldrad": (
        "ARM Best Estimate hourly products: yearly files 1998/2001 .. 2023-01-01 "
        "only; nothing for 2024 onward"
    ),
    "radflux1long": (
        "RADFLUXANAL clear-sky fits and cloud fraction: c1 ends 2025-08-15, c2 "
        "ends 2024-07-10 -- not yet processed for the 2025/26 season"
    ),
    "mwrret2turn": "MWRRET v2 (Turner): 0 files at NSA",
    "aerioe1turn": "AERIoe thermodynamic/cloud retrieval: 0 files at NSA",
    "microbasekaplus": (
        "0 files; the NSA MICROBASE product is nsamicrobaseC1.c1 "
        "(2011-11 .. 2025-12-31)"
    ),
    "pwd / wbpluvio2 / ld (stand-alone precipitation)": (
        "0 files at C1; present-weather-detector fields are embedded in " "nsametC1.b1"
    ),
    "tsiskycover": "daylight-only; last file 2024-10-30",
    "irt (nsairtC1.b1)": "ended 2025-10-29; superseded by nsagndirtC1.b1",
    "skyrad20s / gndrad20s .a0": "HTTP 403 -- raw a0 level is not served",
}


def seb_keys(var_set: str) -> Tuple[str, ...]:
    """Return the pipeline keys of a named SEB tier."""
    try:
        return SEB_VARIABLE_SETS[var_set]
    except KeyError as err:
        raise ValueError(
            f"Unknown SEB variable set {var_set!r}; choose from "
            f"{sorted(SEB_VARIABLE_SETS)}."
        ) from err


def estimate_size_gb(keys: Iterable[str], n_days: int) -> Dict[str, float]:
    """Per-key download size estimate [GB] for `n_days` of data.

    Uses the per-day sizes measured from real files (SEB_STREAMS). Keys that
    are not SEB streams (e.g. 'kazr') are reported as NaN rather than guessed.
    THERMOCLDPHASE and other lagging VAPs are sized as if complete, so the
    estimate is an upper bound for those.
    """
    out: Dict[str, float] = {}
    for key in keys:
        info = SEB_STREAMS.get(key)
        out[key] = float("nan") if info is None else info.mb_per_day * n_days / 1024.0
    return out


# ---------------------------------------------------------------------------
# Readers for the SEB-specific datastreams
# ---------------------------------------------------------------------------


def read_gndirt(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the 1-min ground-looking IRT brightness temperature [K].

    Returns
    -------
    Dataset over time with
        t_skin_ir_k      surface IR brightness temperature [K]
        t_skin_ir_std_k  its 1-min standard deviation [K]

    This is a BRIGHTNESS temperature. Use skin_temperature_from_irt() to
    convert it to a thermodynamic skin temperature with an emissivity and the
    reflected downwelling longwave.
    """
    ds = read_timeseries(
        "gndirt", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    ds["t_skin_ir_k"].attrs.setdefault("units", "K")
    ds["t_skin_ir_k"].attrs[
        "long_name"
    ] = "surface infrared brightness temperature (GNDIRT, 10-m mount)"
    return ds


def read_sirs(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the merged SKYRAD+GNDRAD 60-s fluxes (un-QC'd QCRAD fallback).

    Same canonical flux names as surface.read_qcrad() (swdn_w_m2, lwdn_w_m2,
    swup_w_m2, lwup_w_m2) plus the second downwelling pyrgeometer
    (lwdn2_w_m2) and the pyrgeometer case/dome temperatures. Only the ingest-
    level min/max/delta QC applies here; the Long & Shi (2008) climatological
    and cross-instrument tests of QCRAD do not.
    """
    ds = read_timeseries(
        "sirs", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    for name in ("swdn_w_m2", "lwdn_w_m2", "lwdn2_w_m2", "swup_w_m2", "lwup_w_m2"):
        ds[name].attrs.setdefault("units", "W/m^2")
    ds["lwdn_w_m2"].attrs["long_name"] = "downwelling longwave, pyrgeometer 1 (SIRS b1)"
    ds["lwdn2_w_m2"].attrs[
        "long_name"
    ] = "downwelling longwave, pyrgeometer 2 (SIRS b1)"
    return ds


def read_mwr3c(
    start_date: str,
    end_date: str,
    average_s: Optional[float] = 60.0,
    zenith_only: bool = True,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the 3-channel MWR LWP/PWV, converted to g/m^2 and cm.

    Parameters
    ----------
    average_s:
        Bin-average each daily file to this interval [s] before concatenating
        (default 60 s). The native cadence is ~1 s (~67,000 samples/day), so
        a season at full rate is ~12 M samples; averaging per file keeps the
        memory footprint small. None keeps the native samples.
    zenith_only:
        Drop samples where the elevation angle is not within 1 deg of 90 (the
        instrument occasionally scans). In the 2025/26 sample files every
        sample was at zenith, so this normally removes nothing.
    apply_qc_flags:
        There is NO qc_lwp / qc_pwv in this product. QC is applied to the
        brightness temperatures instead (missing / fail_min / fail_max /
        questionable calibration), and LWP/PWV are masked wherever any of the
        three channels is masked.

    Returns
    -------
    Dataset over time with
        lwp_g_m2, lwp_err_g_m2   LWP and its regression RMSE [g/m^2]
        pwv_cm                   precipitable water vapour [cm]
        tb_23_k, tb_31_k, tb_90_k  sky brightness temperatures [K]
        ir_sky_temp_k            10.5-um zenith IR temperature [K]
    """
    spec = config.get_spec("mwr3c")
    paths = files_in_range("mwr3c", start_date, end_date)
    if not paths:
        raise FileNotFoundError(
            f"No local mwr3c files in {start_date}..{end_date} under "
            f"{config.RAW_DATA_DIR}. Download with scripts/download_nsa_seb_data.py."
        )
    pieces: List[xr.Dataset] = []
    for path in paths:
        if verbose:
            print(f"reading {path.name}")
        # Read WITHOUT the generic QC pass: lwp has no qc companion and the
        # generic reader would silently apply none. The Tb QC is applied by
        # hand below and propagated to the retrievals.
        ds = _read_one_file(path, dict(spec.variables), apply_qc_flags, False)
        if zenith_only and "elevation_deg" in ds:
            at_zenith = np.abs(ds["elevation_deg"].values - 90.0) < 1.0
            ds = ds.isel(time=np.where(at_zenith)[0])
        if apply_qc_flags:
            bad = np.zeros(ds.sizes["time"], dtype=bool)
            for tb in ("tb_23_k", "tb_31_k", "tb_90_k"):
                bad |= ~np.isfinite(ds[tb].values)
            for name in ("lwp_g_m2", "lwp_err_g_m2", "pwv_cm"):
                ds[name] = ds[name].where(~xr.DataArray(bad, dims="time"))
        # mm of liquid water -> g/m^2 (1 mm = 1 kg/m^2 = 1000 g/m^2).
        for name in ("lwp_g_m2", "lwp_err_g_m2"):
            units = str(ds[name].attrs.get("units", "")).strip().lower()
            if units == "mm":
                attrs = dict(ds[name].attrs)
                ds[name] = ds[name] * 1.0e3
                ds[name].attrs = attrs
                ds[name].attrs["units"] = "g/m^2"
                ds[name].attrs["units_note"] = "converted from mm (x1000)"
        if average_s is not None:
            ds = ds.drop_vars("elevation_deg", errors="ignore")
            ds = ds.resample(time=f"{int(average_s)}s").mean()
        pieces.append(ds)
    out = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts").sortby("time")
    _, idx = np.unique(out["time"].values, return_index=True)
    if idx.size != out.sizes["time"]:
        out = out.isel(time=idx)
    out = out.sel(time=slice(start_date, f"{end_date}T23:59:59"))
    out["lwp_g_m2"].attrs["long_name"] = (
        "liquid water path, 3-channel MWR regression retrieval (no clear-sky "
        "bias correction)"
    )
    out.attrs["averaging_interval_s"] = average_s if average_s else "native (~1 s)"
    return out


def _flag_to_nan(arr: xr.DataArray, flag_values: Iterable[float]) -> xr.DataArray:
    """Replace ARSCL's in-band flag values (-1 clear, -2 possible clear) by NaN."""
    out = arr
    for v in flag_values:
        out = out.where(out != v)
    out.attrs = dict(arr.attrs)
    return out


def read_arscl_boundaries(
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> xr.Dataset:
    """Read ARSCL cloud-layer boundaries with clear sky as NaN + a flag.

    ARSCL encodes "clear sky" as -1 and "possible clear sky" (no MPL, obscured
    ceilometer, no cloud detected) as -2 INSIDE the height fields. Left alone,
    those sentinels would enter any height statistic as negative altitudes, so
    they are moved out into `sky_flag`:
        0  cloud detected (heights valid)
        1  clear sky
        2  possible clear sky
        3  no data (heights NaN for another reason)

    Returns
    -------
    Dataset over (time, layer) / time with
        cloud_layer_base_m, cloud_layer_top_m   [m AGL], NaN where no layer
        cloud_base_best_estimate_m              ceilometer/MPL base [m AGL]
        cloud_top_max_m                         highest layer top [m AGL]
        n_layers                                number of detected layers
        sky_flag                                see above
        instrument_availability_flag            bit mask: 1 KAZR 2 MPL 4 ceil
                                                8 MWR 16 rain gauge
    No qc_ variables exist in this product; nothing is masked beyond the
    sentinel handling.
    """
    ds = read_timeseries(
        "arsclbnd", start_date, end_date, apply_qc_flags=False, verbose=verbose
    )
    cbbe = ds["cloud_base_best_estimate_m"]
    sky_flag = xr.where(cbbe == -1.0, 1, 0)
    sky_flag = xr.where(cbbe == -2.0, 2, sky_flag)
    sky_flag = xr.where(~np.isfinite(cbbe), 3, sky_flag)
    ds["cloud_base_best_estimate_m"] = _flag_to_nan(cbbe, (-1.0, -2.0))
    ds["cloud_layer_base_m"] = _flag_to_nan(ds["cloud_layer_base_m"], (-1.0, -2.0))
    ds["cloud_layer_top_m"] = _flag_to_nan(ds["cloud_layer_top_m"], (-1.0, -2.0))
    ds["sky_flag"] = sky_flag.astype("int8")
    ds["sky_flag"].attrs = {
        "long_name": "sky state from cloud_base_best_estimate",
        "flag_values": "0 1 2 3",
        "flag_meanings": "cloud clear_sky possible_clear_sky no_data",
    }
    ds["n_layers"] = np.isfinite(ds["cloud_layer_base_m"]).sum("layer").astype("int8")
    ds["n_layers"].attrs = {"long_name": "number of ARSCL hydrometeor layers"}
    ds["cloud_top_max_m"] = ds["cloud_layer_top_m"].max("layer", skipna=True)
    ds["cloud_top_max_m"].attrs = {"units": "m", "long_name": "highest layer top AGL"}
    for name in (
        "cloud_layer_base_m",
        "cloud_layer_top_m",
        "cloud_base_best_estimate_m",
    ):
        ds[name].attrs.setdefault("units", "m")
    return ds


def read_cldtype(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
    include_reflectivity: bool = True,
) -> xr.Dataset:
    """Read CLDTYPE: 1-min cloud type, layer boundaries, reflectivity, precip.

    Parameters
    ----------
    include_reflectivity:
        The (time, height) reflectivity profile is 596 gates x 1440 min per
        day; set False to read only the per-layer and scalar fields when the
        IWP estimate is not needed.

    Returns
    -------
    Dataset with
        cloud_type (time, layer)   1 low_cloud 2 congestus 3 deep_convection
                                   4 altocumulus 5 altostratus
                                   6 cirrostratus/anvil 7 cirrus  (NaN: none)
        cloud_layer_base_m, cloud_layer_top_m (time, layer)  [m AGL]
        cloud_base_best_estimate_m (time)
        reflectivity_dbz (time, height)   ARSCL best-estimate Z [dBZ], NaN
                                          where no hydrometeor detected
        precip_rate_mm_min (time)         surface precipitation rate
        cloud_source_flag (time, height)  which instrument saw the cloud
    The layer heights use -1 for "no layer"; converted to NaN here.
    """
    canonical = list(config.get_spec("cldtype").variables)
    if not include_reflectivity:
        canonical = [
            c for c in canonical if c not in ("reflectivity_dbz", "cloud_source_flag")
        ]
    ds = read_timeseries(
        "cldtype",
        start_date,
        end_date,
        canonical_vars=canonical,
        apply_qc_flags=apply_qc_flags,
        verbose=verbose,
    )
    for name in (
        "cloud_layer_base_m",
        "cloud_layer_top_m",
        "cloud_base_best_estimate_m",
    ):
        ds[name] = _flag_to_nan(ds[name], (-1.0, -2.0))
        ds[name].attrs.setdefault("units", "m")
    ds["cloud_type"].attrs["flag_meanings"] = (
        "1 low_cloud 2 congestus 3 deep_convection 4 altocumulus 5 altostratus "
        "6 cirrostratus/anvil 7 cirrus"
    )
    return ds


def read_ecor_e10(
    start_date: str,
    end_date: str,
    max_quality_flag: int = 1,
    max_abs_flux_w_m2: float = 150.0,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the OLIKTOK POINT (NSA E10) eddy-correlation fluxes.

    Parameters
    ----------
    max_quality_flag:
        EddyPro flag threshold (Mauder & Foken 2004 scheme): 0 = best quality,
        1 = acceptable, 2 = discard. Fluxes with flag > max_quality_flag are
        set to NaN (default keeps 0 and 1).
    max_abs_flux_w_m2:
        Plausibility screen applied AFTER the flags. The EddyPro flags do not
        catch a rimed/iced sonic anemometer: on 2025-12-15 the flag-0 records
        reported SH of several hundred W/m^2 in polar night. Over winter
        tundra |SH| and |LH| rarely exceed ~50 W/m^2, so 150 W/m^2 is a loose
        bound; tighten it for the cold season. None disables the screen.

    Returns
    -------
    Dataset over time with sh_w_m2, lh_w_m2 (POSITIVE UPWARD), ustar_m_s,
    obukhov_length_m, wspd_m_s, t_air_k, q_air_kg_kg, and the raw flags.
    `site` attribute = "NSA E10 Oliktok Point" -- this is NOT Barrow.
    """
    ds = read_timeseries(
        "ecor_e10", start_date, end_date, apply_qc_flags=True, verbose=verbose
    )
    ds["sh_w_m2"] = ds["sh_w_m2"].where(ds["sh_flag"] <= max_quality_flag)
    ds["lh_w_m2"] = ds["lh_w_m2"].where(ds["lh_flag"] <= max_quality_flag)
    if max_abs_flux_w_m2 is not None:
        for name in ("sh_w_m2", "lh_w_m2"):
            ds[name] = ds[name].where(np.abs(ds[name]) <= max_abs_flux_w_m2)
        ds.attrs["plausibility_screen_w_m2"] = max_abs_flux_w_m2
    for name in ("sh_w_m2", "lh_w_m2"):
        ds[name].attrs["units"] = "W/m^2"
        ds[name].attrs["sign_convention"] = "positive upward (surface to atmosphere)"
    ds.attrs["site"] = "NSA E10 Oliktok Point (~250 km ESE of Utqiagvik) -- not Barrow"
    return ds


def read_sebs_e10(
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the OLIKTOK POINT (NSA E10) soil heat flux and soil state."""
    ds = read_timeseries(
        "sebs_e10", start_date, end_date, apply_qc_flags=True, verbose=verbose
    )
    ds.attrs["site"] = "NSA E10 Oliktok Point (~250 km ESE of Utqiagvik) -- not Barrow"
    # Sign convention read from the file: standard_name is
    # 'upward_heat_flux_at_ground_level_in_soil' and the file's own
    # surface_energy_balance equals net_radiation + G, so G is positive
    # UPWARD -- identical to the Sledd et al. (2025) Eq. (1) convention and
    # to ground_flux.py. No sign change is needed.
    ds["soil_heat_flux_w_m2"].attrs[
        "sign_convention"
    ] = "positive upward (toward the surface); same as Sledd et al. 2025 G"
    return ds


# ---------------------------------------------------------------------------
# Skin temperature from the radiometers
# ---------------------------------------------------------------------------


def skin_temperature_from_lwup(
    lwup_w_m2: xr.DataArray,
    lwdn_w_m2: xr.DataArray,
    emissivity: float = SNOW_EMISSIVITY,
) -> xr.DataArray:
    """Invert the upwelling longwave flux for the surface skin temperature.

    The upwelling pyrgeometer sees emitted plus reflected radiation,

        LWU = eps * sigma * T_s^4 + (1 - eps) * LWD,

    so  T_s = [ (LWU - (1 - eps) LWD) / (eps sigma) ]^(1/4).  This is the
    inversion Sledd et al. (2025, Sect. 2.1) apply to the MOSAiC radiometers
    with eps = 0.985, and the same expression the ERA5 seb_terms.py evaluates
    as t_skin_from_lwu_K, so the two pipelines are directly comparable.

    The pyrgeometer at NSA is mounted at 10 m and integrates a footprint of
    order 10 m radius, so this is an area-average skin temperature.
    """
    t_s = (
        (lwup_w_m2 - (1.0 - emissivity) * lwdn_w_m2) / (emissivity * SIGMA_SB_W_M2_K4)
    ) ** 0.25
    t_s.attrs = {
        "units": "K",
        "long_name": f"skin temperature inverted from LWU, emissivity {emissivity}",
    }
    return t_s


def skin_temperature_from_irt(
    t_brightness_k: xr.DataArray,
    lwdn_w_m2: xr.DataArray,
    emissivity: float = SNOW_EMISSIVITY,
) -> xr.DataArray:
    """Correct an IRT brightness temperature for emissivity and reflected sky.

    The IRT reports the temperature of a blackbody that would produce the
    measured in-band radiance. Treating the instrument band as broadband,

        sigma T_b^4 = eps sigma T_s^4 + (1 - eps) L_sky,

    with L_sky approximated by the broadband LWD. The reflected term is small
    (1 - eps ~ 0.015 of ~200 W/m^2 = ~3 W/m^2, i.e. ~1 K at 250 K), but the
    broadband proxy is a known approximation: the sky is much less emissive
    in the 8-14 um window than broadband, so this slightly OVER-estimates the
    reflected contribution under clear skies. For a rigorous conversion use
    the AERI window radiance; for SEB purposes the +-1 K this introduces is
    below the emissivity uncertainty itself.
    """
    t_s = (
        (SIGMA_SB_W_M2_K4 * t_brightness_k**4 - (1.0 - emissivity) * lwdn_w_m2)
        / (emissivity * SIGMA_SB_W_M2_K4)
    ) ** 0.25
    t_s.attrs = {
        "units": "K",
        "long_name": f"skin temperature from IRT brightness T, emissivity {emissivity}",
    }
    return t_s


# ---------------------------------------------------------------------------
# Best-estimate downwelling longwave from the two pyrgeometers
# ---------------------------------------------------------------------------

# Physical plausibility window for LWD relative to the 2-m air temperature,
# in the spirit of the QCRAD "LWdn to Ta" tests (Long & Shi 2008): LWD cannot
# much exceed blackbody emission at the air temperature (overcast, near-
# isothermal sky), and cannot fall below a fraction of it (clear, dry sky).
# The exact QCRAD constants are site-configurable and are not reproduced
# here; these are deliberately loose so they only reject impossible values.
LWDN_TA_UPPER_MARGIN_W_M2 = 25.0
LWDN_TA_LOWER_FRACTION = 0.5
LWDN_AGREEMENT_W_M2 = 10.0


def best_estimate_lwdn(
    sirs: xr.Dataset,
    t_air_k: Optional[xr.DataArray] = None,
    agreement_w_m2: float = LWDN_AGREEMENT_W_M2,
    prefer: int = 2,
) -> xr.Dataset:
    """Combine SKYRAD pyrgeometers 1 and 2 into one flagged LWD series.

    Why this exists
        QCRAD c1 reports pyrgeometer 1 verbatim, and in the 2025/26 season
        pyrgeometer 1 failed for 44 of the 90 days of Dec-Feb (values up to
        500 W/m^2 in polar night, 25% of Dec-Feb samples above
        sigma T_a^4 + 25), while pyrgeometer 2 stayed physically plausible
        (0.4% above that bound). QCRAD's own tests removed only the extreme
        part. The QCRAD c2 "best estimate" that would normally arbitrate
        does not yet exist for these dates.

    Rule (per sample)
        1. Each sensor is "plausible" if finite and, when `t_air_k` is given,
           within [LWDN_TA_LOWER_FRACTION * sigma T_a^4,
                   sigma T_a^4 + LWDN_TA_UPPER_MARGIN_W_M2].
        2. Both plausible and |LWD1 - LWD2| <= agreement_w_m2 -> their mean
           (source 1). Averaging two independent pyrgeometers halves the
           random error.
        3. Only one plausible -> that one (source 2 = pyrg1, 3 = pyrg2).
        4. Both plausible but disagreeing -> the `prefer`red sensor, flagged
           (source 4). Default prefer=2 because of the evidence above; set
           prefer=1 for seasons where pyrgeometer 2 is the suspect one.
        5. Neither -> NaN (source 0).

    Returns
    -------
    Dataset with lwdn_be_w_m2, lwdn_source (0-4 as above) and
    lwdn_spread_w_m2 = LWD1 - LWD2 (the in-situ measurement uncertainty).
    """
    lw1 = sirs["lwdn_w_m2"]
    lw2 = sirs["lwdn2_w_m2"]
    plaus1 = np.isfinite(lw1)
    plaus2 = np.isfinite(lw2)
    if t_air_k is not None:
        ta = t_air_k.reindex(
            time=sirs["time"], method="nearest", tolerance=np.timedelta64(90, "s")
        )
        bb = SIGMA_SB_W_M2_K4 * ta**4
        upper = bb + LWDN_TA_UPPER_MARGIN_W_M2
        lower = LWDN_TA_LOWER_FRACTION * bb
        have_ta = np.isfinite(ta)
        plaus1 = plaus1 & (~have_ta | ((lw1 <= upper) & (lw1 >= lower)))
        plaus2 = plaus2 & (~have_ta | ((lw2 <= upper) & (lw2 >= lower)))
    agree = plaus1 & plaus2 & (np.abs(lw1 - lw2) <= agreement_w_m2)
    only1 = plaus1 & ~plaus2
    only2 = plaus2 & ~plaus1
    disagree = plaus1 & plaus2 & ~agree
    preferred = lw2 if prefer == 2 else lw1

    be = xr.full_like(lw1, np.nan)
    be = be.where(~agree, 0.5 * (lw1 + lw2))
    be = be.where(~only1, lw1)
    be = be.where(~only2, lw2)
    be = be.where(~disagree, preferred)
    source = xr.zeros_like(lw1, dtype="int8")
    source = (
        source.where(~agree, 1)
        .where(~only1, 2)
        .where(~only2, 3)
        .where(~disagree, 4)
        .astype("int8")
    )

    out = xr.Dataset(
        {"lwdn_be_w_m2": be, "lwdn_source": source, "lwdn_spread_w_m2": lw1 - lw2}
    )
    out["lwdn_be_w_m2"].attrs = {
        "units": "W/m^2",
        "long_name": "downwelling longwave, best estimate of two pyrgeometers",
        "rule": f"mean if within {agreement_w_m2} W/m^2, else the plausible one, "
        f"else pyrgeometer {prefer} (flagged)",
    }
    out["lwdn_source"].attrs = {
        "flag_values": "0 1 2 3 4",
        "flag_meanings": "none mean_of_both pyrg1_only pyrg2_only disagree_used_preferred",
    }
    out["lwdn_spread_w_m2"].attrs = {
        "units": "W/m^2",
        "long_name": "pyrgeometer 1 minus 2",
    }
    return out


# ---------------------------------------------------------------------------
# Equation (1) terms, ERA5-compatible names
# ---------------------------------------------------------------------------


def compute_seb_terms(
    qcrad: xr.Dataset,
    gndirt: Optional[xr.Dataset] = None,
    met: Optional[xr.Dataset] = None,
    sirs: Optional[xr.Dataset] = None,
    sh_up_w_m2: Optional[xr.DataArray] = None,
    lh_up_w_m2: Optional[xr.DataArray] = None,
    g_up_w_m2: Optional[xr.DataArray] = None,
    emissivity: float = SNOW_EMISSIVITY,
) -> xr.Dataset:
    """Map NSA observations onto the Sledd et al. (2025) Eq. (1) terms.

    Output names and sign conventions are IDENTICAL to
    ERA5/surface_energy_budget/seb_terms.compute_seb_terms(), so
    ``obs["lwd_W_m2"] - era5["lwd_W_m2"]`` is the comparison.

    Parameters
    ----------
    qcrad:
        surface.read_qcrad() output (swdn/lwdn/swup/lwup in W/m^2).
    sirs:
        seb.read_sirs() output, optional but STRONGLY recommended for
        2025/26: LWD is then the two-pyrgeometer best estimate from
        best_estimate_lwdn() (using `met` 2-m temperature for the
        plausibility bounds when given) instead of QCRAD's pyrgeometer 1,
        and lwd_source / lwd_spread_W_m2 are carried along. The QCRAD QC
        mask is still honoured where QCRAD flagged its own value bad AND the
        best estimate came from pyrgeometer 1.
    gndirt:
        seb.read_gndirt() output, optional. If given, t_skin_K is the IRT-
        derived skin temperature; otherwise t_skin_K is inverted from LWU.
        t_skin_from_lwu_K is always provided.
    met:
        surface.read_met() output, optional; supplies dt_skin_minus_2m_K and
        wind_speed_10m_m_s (the MET wind is at 10 m per the file header).
    sh_up_w_m2, lh_up_w_m2:
        Turbulent fluxes, POSITIVE UPWARD, e.g. from bulk_flux.bulk_fluxes().
        Optional; without them na_flux_W_m2 cannot be formed and is omitted.
    g_up_w_m2:
        Ground heat flux, positive toward the surface from below (the Eq. (1)
        sign). Optional; enables residual_W_m2 = M (should be ~0 in winter).

    Returns
    -------
    Dataset of Eq. (1) terms with units/long_name attributes. Time axis is
    that of `qcrad`; the other inputs are aligned by nearest-neighbour
    reindexing with a 90 s tolerance (all are 1-min streams).
    """
    t = qcrad["time"]

    def _align(da: xr.DataArray) -> xr.DataArray:
        return da.reindex(time=t, method="nearest", tolerance=np.timedelta64(90, "s"))

    lwd = qcrad["lwdn_w_m2"]
    lwd_extra: Dict[str, xr.DataArray] = {}
    if sirs is not None:
        t_air_k = None if met is None else met["temp_2m_c"] + 273.15
        be = best_estimate_lwdn(sirs, t_air_k=t_air_k)
        lwd_be = _align(be["lwdn_be_w_m2"])
        source = _align(be["lwdn_source"].astype(float)).fillna(0).astype("int8")
        # Where QCRAD rejected its own (pyrgeometer 1) value and the best
        # estimate leaned on pyrgeometer 1 alone, keep QCRAD's verdict.
        qcrad_bad = ~np.isfinite(lwd)
        lwd_be = lwd_be.where(~(qcrad_bad & (source == 2)))
        lwd = lwd_be
        lwd_extra["lwd_source"] = source
        lwd_extra["lwd_spread_W_m2"] = _align(be["lwdn_spread_w_m2"])
    lwu = qcrad["lwup_w_m2"]
    swd = qcrad["swdn_w_m2"]
    swu = qcrad["swup_w_m2"]
    lw_net = lwd - lwu
    sw_net = swd - swu

    out: Dict[str, xr.DataArray] = {
        "lwd_W_m2": lwd,
        **lwd_extra,
        "lwu_W_m2": lwu,
        "lw_net_W_m2": lw_net,
        "swd_W_m2": swd,
        "swu_W_m2": swu,
        "swn_W_m2": sw_net,
        # Miller et al. (2017) radiative forcing, Sledd Sect. 3.2.
        "forcing_W_m2": lwd + sw_net,
        "t_skin_from_lwu_K": skin_temperature_from_lwup(lwu, lwd, emissivity),
    }

    # Skin temperature: the corrected IRT where it exists, the LWU inversion
    # where it does not, with per-sample provenance (1 = IRT, 2 = LWU). The
    # fallback keeps the bulk fluxes defined through IRT outages (~8% of the
    # 2025/26 season) at the cost of the LWU frost-bias caveat in the README.
    t_skin_lwu = out["t_skin_from_lwu_K"]
    if gndirt is not None:
        t_skin_irt = skin_temperature_from_irt(
            _align(gndirt["t_skin_ir_k"]), lwd, emissivity
        )
        have_irt = np.isfinite(t_skin_irt)
        out["t_skin_K"] = t_skin_irt.where(have_irt, t_skin_lwu)
        out["t_skin_K"].attrs[
            "source"
        ] = "GNDIRT (corrected) where available, else LWU inversion"
        out["t_skin_source"] = xr.where(
            have_irt, 1, xr.where(np.isfinite(t_skin_lwu), 2, 0)
        ).astype("int8")
    else:
        out["t_skin_K"] = t_skin_lwu.copy()
        out["t_skin_K"].attrs["source"] = "inverted from LWU (no IRT supplied)"
        out["t_skin_source"] = xr.where(np.isfinite(t_skin_lwu), 2, 0).astype("int8")

    if met is not None:
        t2m_k = _align(met["temp_2m_c"]) + 273.15
        out["t_2m_K"] = t2m_k
        out["dt_skin_minus_2m_K"] = out["t_skin_K"] - t2m_k
        if "wspd_m_s" in met:
            out["wind_speed_10m_m_s"] = _align(met["wspd_m_s"])

    if sh_up_w_m2 is not None:
        out["sh_up_W_m2"] = _align(sh_up_w_m2)
    if lh_up_w_m2 is not None:
        out["lh_up_W_m2"] = _align(lh_up_w_m2)
    if sh_up_w_m2 is not None and lh_up_w_m2 is not None:
        # Eq. (5): NA = LWD - LWU + SWD - SWU - SH - LH = M - SWT - G
        out["na_flux_W_m2"] = lw_net + sw_net - out["sh_up_W_m2"] - out["lh_up_W_m2"]
        if g_up_w_m2 is not None:
            out["g_up_W_m2"] = _align(g_up_w_m2)
            # Eq. (1) with SWT = 0: the residual is M, which for a frozen
            # tundra surface in Oct-Mar must be ~0 -- anything else is the sum
            # of the errors in every term (and is the usual way G is "found").
            out["residual_W_m2"] = out["na_flux_W_m2"] + out["g_up_W_m2"]

    result = xr.Dataset(out)
    long_names = {
        "lwd_W_m2": "Downwelling longwave radiative flux (LWD)"
        + (
            ", two-pyrgeometer best estimate"
            if sirs is not None
            else ", QCRAD pyrgeometer 1"
        ),
        "lwd_source": "LWD source: 0 none 1 mean 2 pyrg1 3 pyrg2 4 disagree->preferred",
        "lwd_spread_W_m2": "LWD pyrgeometer 1 minus 2 (in-situ uncertainty)",
        "lwu_W_m2": "Upwelling longwave radiative flux (LWU), QCRAD, 10-m mount",
        "lw_net_W_m2": "Net longwave, LWD - LWU, positive downward",
        "swd_W_m2": "Downwelling shortwave radiative flux (SWD), QCRAD",
        "swu_W_m2": "Upwelling shortwave radiative flux (SWU), QCRAD",
        "swn_W_m2": "Net shortwave, SWN = SWD - SWU, positive downward",
        "forcing_W_m2": "Radiative forcing, LWD + SWN (Miller et al. framework)",
        "t_skin_from_lwu_K": f"Skin temperature inverted from LWU, emissivity {emissivity}",
        "t_skin_K": "Skin temperature (see 'source' attribute and t_skin_source)",
        "t_skin_source": "skin temperature source: 0 none 1 IRT (corrected) 2 LWU inversion",
        "t_2m_K": "2 m air temperature (MET)",
        "dt_skin_minus_2m_K": "Skin minus 2 m temperature (near-surface stability)",
        "wind_speed_10m_m_s": "10 m wind speed (MET)",
        "sh_up_W_m2": "Sensible heat flux, positive UPWARD (Sledd SH)",
        "lh_up_W_m2": "Latent heat flux, positive UPWARD (Sledd LH)",
        "g_up_W_m2": "Ground heat flux, positive toward the surface (Sledd G)",
        "na_flux_W_m2": "Net atmospheric flux, Eq. (5) = M - SWT - G",
        "residual_W_m2": "Eq. (1) residual with SWT = 0 (= M; ~0 for frozen tundra)",
    }
    for name, da in result.data_vars.items():
        da.attrs["long_name"] = long_names.get(name, name)
        da.attrs["units"] = _units_from_name(name)
    result.attrs["convention"] = (
        "Turbulent fluxes positive UPWARD (Sledd et al. 2025 Eq. 1). Radiative "
        "net fluxes positive DOWNWARD. G positive toward the surface."
    )
    result.attrs["reference"] = "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578"
    result.attrs["site"] = "ARM NSA C1, Utqiagvik (Barrow), AK"
    return result


def _units_from_name(name: str) -> str:
    """Recover units from the trailing unit token of a variable name."""
    if name.endswith("_W_m2"):
        return "W m-2"
    if name.endswith("_K"):
        return "K"
    if name.endswith("_m_s"):
        return "m s-1"
    return "1"
