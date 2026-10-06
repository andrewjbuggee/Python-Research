"""MFRSRCLDOD: cloud optical depth and effective radius from the MFRSR.

Datastream  epcmfrsrcldod1minM1.c1 (config product ``cod_mfrsr_M1``)
Reference   Turner, D. D., C. Lo, Q. Min, D. Zhang, and K. Gaustad (2021),
            Cloud Optical Properties from the Multifilter Shadowband Radiometer
            (MFRSRCLDOD): An ARM Value-Added Product, DOE/SC-ARM-TR-047.
Algorithm   Min, Q., and L. C. Harrison (1996), GRL 23(13), 1641-1644,
            https://doi.org/10.1029/96GL01488

What the VAP does (TR-047 sections 1 and 4)
-------------------------------------------
* Measures the total (hemispheric) transmittance at 415 nm, T = I / I0. I0
  comes from Langley regressions on clear days within +/- 3 months, so the
  absolute calibration cancels.
* Inverts T for optical depth tau with a plane-parallel (1-D) radiative
  transfer model. At 415 nm gas absorption is negligible and the surface
  albedo is small and steady (0.036 assumed for snow-free land, +/- 0.01).
* If the MWR liquid water path is available and >= 20 g m-2 (the MWR
  retrieval uncertainty), it iterates for the effective radius using the
  vertically homogeneous relation
      LWP = (2/3) * rho_w * tau * r_e          (Stephens 1978, JAS 35, 2111)
  i.e. r_e is a column-mean radius forced to be consistent with the MWR.
  Otherwise it assumes r_e = 8.0 um. tau depends only weakly on r_e at 415 nm.
* Reports 20 s "instantaneous" and 5-min running-average values, with 1-sigma
  uncertainties propagated from I, I0, LWP and surface albedo.
  (The "1min" in the datastream name means "Min's version 1 VAP", not
  1-minute data; TR-047 section 3.)

Validity, from TR-047 sections 5 and 6
--------------------------------------
Only for overcast, all-liquid cloud with tau > ~7 over a snow-free surface.
TR-047 selected valid samples with: cloud fraction > 90%, IR sky brightness
temperature > 268 K, cloud base < 4 km, tau > 7, r_e > 0 and r_e != 8.00 um
(8.00 um means no MWR LWP was used). Those are the defaults in
``Criteria`` below.

Consequence for comparisons: when r_e came from the MWR, MFRSR r_e and the
MFRSR LWP are NOT independent of the MWR. Compare MFRSR r_e with the
sunphotometer, not with the MWR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import xarray as xr

from epcape import filters, qc, units
from epcape.products import load_product

PRODUCT = "cod_mfrsr_M1"
R_E_DEFAULT_UM = 8.0  # value the VAP assigns when no MWR LWP is used (TR-047 sec. 1)


def load(start=None, end=None, **kwargs) -> xr.Dataset:
    """Raw combined product for [start, end] (see epcape.products.load_product)."""
    return load_product(PRODUCT, start, end, **kwargs)


def standardize(raw: xr.Dataset) -> xr.Dataset:
    """Tidy dataset with explicit units in the variable names, plus QC masks.

    Variables (absent ancillary fields are simply left out):
        tau, tau_avg5min, tau_unc          cloud optical depth (unitless) and 1-sigma
        r_e_um, r_e_avg5min_um, r_e_unc_um effective radius (um)
        r_e_from_mwr                       True where r_e used MWR LWP (r_e != 8.00 um)
        lwp_gm2, lwp_source                LWP used/derived by the VAP (g m-2)
        cloud_fraction                     shortwave sky-cover fraction (0-1)
        cbh_m                              cloud-base height (m above ground)
        ir_temp_K                          IR sky brightness temperature (K)
        sza_deg                            solar zenith angle (deg)
        surface_albedo                     415 nm albedo assumed by the VAP
        qc_bad_tau, qc_bad_r_e             ARM QC failures (bit tests assessed "Bad")
    """
    out = xr.Dataset(coords={"time": raw["time"]})
    get = lambda n: filters.find_var(raw, n)  # noqa: E731  (case-insensitive lookup)

    out["tau"] = get("optical_depth_instantaneous")
    out["tau_avg5min"] = get("optical_depth_average")
    out["r_e_um"] = units.radius_to_um(get("effective_radius_instantaneous"))
    out["r_e_avg5min_um"] = units.radius_to_um(get("effective_radius_average"))
    if get("cldtaui_toterror") is not None:
        out["tau_unc"] = get("cldtaui_toterror")
    if get("reffi_toterror") is not None:
        out["r_e_unc_um"] = units.radius_to_um(get("reffi_toterror"))

    # r_e equal to the 8.00 um default means the VAP had no usable MWR LWP.
    # A tolerance covers float32 storage of 8.0.
    r_e = out["r_e_um"]
    out["r_e_from_mwr"] = np.isfinite(r_e) & (np.abs(r_e - R_E_DEFAULT_UM) > 1e-3)

    lwp = get("lwp")
    if lwp is not None:
        out["lwp_gm2"] = units.water_path_to_gm2(lwp)
    if get("lwp_source") is not None:
        out["lwp_source"] = get("lwp_source")
    if get("cloudfraction") is not None:
        out["cloud_fraction"] = get("cloudfraction")
    if get("cloudbasebestestimate") is not None:
        out["cbh_m"] = get("cloudbasebestestimate")
    if get("ir_temp") is not None:
        out["ir_temp_K"] = get("ir_temp")
    if get("surface_albedo") is not None:
        out["surface_albedo"] = get("surface_albedo")
    mu0 = get("cosine_solar_zenith_angle")
    if mu0 is not None:
        # clip guards against |mu0| marginally > 1 from rounding
        out["sza_deg"] = xr.apply_ufunc(lambda m: np.degrees(np.arccos(np.clip(m, -1, 1))), mu0)
        out["sza_deg"].attrs = {"long_name": "Solar zenith angle", "units": "degree"}

    out["qc_bad_tau"] = qc.bad_mask(raw, "optical_depth_instantaneous")
    out["qc_bad_r_e"] = qc.bad_mask(raw, "effective_radius_instantaneous")
    out.attrs = {
        "product": PRODUCT,
        "instrument": "MFRSR",
        "source_datastream": raw.attrs.get("source_datastream", "epcmfrsrcldod1minM1.c1"),
    }
    return out


@dataclass
class Criteria:
    """Selection thresholds. Defaults reproduce TR-047 section 5.

    Set a threshold to None to skip that test."""

    use_qc: bool = True
    tau_min: Optional[float] = 7.0  # retrieval valid only for tau > ~7 (TR-047 sec. 1)
    cloud_fraction_min: Optional[float] = 0.9  # overcast (plane-parallel assumption)
    ir_temp_min_K: Optional[float] = 268.0  # warm cloud base -> liquid (TR-047 sec. 5)
    cbh_max_m: Optional[float] = 4000.0  # low cloud (TR-047 sec. 5)
    sza_max_deg: Optional[float] = None  # not a VAP criterion; set to match other products
    require_mwr_r_e: bool = True  # for r_e only: drop the 8.00 um default

    def as_dict(self) -> dict:
        return asdict(self)


def tau_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """Ordered named tests for a valid optical depth (True = passes).

    A test whose ancillary variable is missing from the file is recorded as
    None (skipped) so the funnel table shows it was not applied."""
    t = std["tau"]
    crit: filters.Criteria = {"finite tau": np.isfinite(t)}
    crit["QC (instantaneous tau)"] = ~std["qc_bad_tau"] if c.use_qc else None
    if c.tau_min is not None:
        crit[f"tau > {c.tau_min:g}"] = t > c.tau_min
    if c.cloud_fraction_min is not None:
        crit[f"cloud fraction > {c.cloud_fraction_min:g}"] = (
            std["cloud_fraction"] > c.cloud_fraction_min if "cloud_fraction" in std else None
        )
    if c.ir_temp_min_K is not None:
        crit[f"IR sky T > {c.ir_temp_min_K:g} K"] = (
            std["ir_temp_K"] > c.ir_temp_min_K if "ir_temp_K" in std else None
        )
    if c.cbh_max_m is not None:
        crit[f"cloud base < {c.cbh_max_m:g} m"] = std["cbh_m"] < c.cbh_max_m if "cbh_m" in std else None
    if c.sza_max_deg is not None:
        crit[f"SZA < {c.sza_max_deg:g} deg"] = std["sza_deg"] < c.sza_max_deg if "sza_deg" in std else None
    return crit


def r_e_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """tau tests plus the effective-radius tests (r_e > 0, r_e from MWR, QC)."""
    crit = dict(tau_criteria(std, c))
    r = std["r_e_um"]
    crit["finite r_e > 0"] = np.isfinite(r) & (r > 0)
    crit["QC (instantaneous r_e)"] = ~std["qc_bad_r_e"] if c.use_qc else None
    if c.require_mwr_r_e:
        crit["r_e from MWR LWP (not 8.00 um default)"] = std["r_e_from_mwr"]
    return crit
