"""SPHOTCOD: cloud optical depth, effective radius and LWP from the Cimel
sunphotometer in "cloud mode".

Datastream  epcsphotcod2chiuM1.c1 (config product ``cod_sphot_M1``)
Reference   Ma, L. L., S. E. Giangrande, J. D. Rausch, D. Wang, and C. Chiu (2025),
            Three-Channel Sunphotometer Cloud Mode Value-Added Product Report,
            DOE/SC-ARM-TR-317.
Algorithm   Chiu, J. C., et al. (2012), ACP 12, 10313-10329,
            https://doi.org/10.5194/acp-12-10313-2012
Evaluation  Sookdar, K., et al. (2025), EGUsphere,
            https://doi.org/10.5194/egusphere-2025-694

What the VAP does (TR-317 section 2)
------------------------------------
* In cloud mode the Cimel points at the zenith (1.2 deg field of view) while
  cloud blocks the sun, and measures radiance at 440, 870 and 1640 nm. The
  cycle takes < 5 min and repeats every 5-15 min when the instrument's
  schedule allows, so samples are sparse and only exist when the sun is
  obscured.
* tau and r_e are retrieved together against a DISORT look-up table. The
  surface albedo comes from MODIS white-sky albedo (MCD43A2/A3, 500 m, 16-day
  window). The 1640 nm channel adds sensitivity to droplet size through
  liquid absorption.
* Uncertainty: radiances and albedo are perturbed by 5-10%, 40 times, with
  the 5 best look-up-table solutions kept each time. The reported values are
  the mean and the standard error of the 40 repetitions.
* LWP = (2/3) * rho_w * tau * r_e (TR-317 eq. 1), with a vertically uniform
  liquid water content assumed (Stephens 1978).

Known behaviour (Sookdar et al. 2025, as summarised in TR-317 section 6)
------------------------------------------------------------------------
tau correlates with the MFRSR at r ~ 0.81, with the photometer biased high.
r_e correlates poorly (r < 0.1) with radar/radiometer (MICROBASE) estimates;
standard deviation ~ 3 um. LWP is roughly unbiased in non-drizzling cases
(error ~ 50 g m-2, r ~ 0.7 against MWR retrievals).

Site-specific assumption to check at the Scripps Pier: the 500 m MODIS
albedo pixel mixes ocean, beach and land. That makes the assumed albedo
uncertain, and the 870/1640 nm retrieval is sensitive to albedo contrast.
This is my inference from the retrieval design, not a documented EPCAPE issue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from epcape import filters, qc, units
from epcape.products import load_product

PRODUCT = "cod_sphot_M1"


def load(start=None, end=None, **kwargs) -> xr.Dataset:
    """Raw combined product for [start, end] (see epcape.products.load_product)."""
    return load_product(PRODUCT, start, end, **kwargs)


def standardize(raw: xr.Dataset) -> xr.Dataset:
    """Tidy dataset with explicit units in the variable names.

    Variables (absent ancillary fields are left out):
        tau, tau_std                   optical depth and its perturbation standard error
        r_e_um, r_e_std_um             effective radius (um)
        lwp_gm2, lwp_std_gm2           liquid water path (g m-2)
        n_solutions                    number of look-up-table solutions
        retrieval_flag                 VAP quality flag (meaning: see flag attributes)
        sza_deg                        solar zenith angle (deg)
        modis_albedo                   MODIS white-sky albedo used (time, modis_channel)
        qc_bad_tau                     ARM QC failure on tau (if a qc_ field exists)
    """
    out = xr.Dataset(coords={"time": raw["time"]})
    get = lambda n: filters.find_var(raw, n)  # noqa: E731

    out["tau"] = get("cloud_optical_depth")
    out["r_e_um"] = units.radius_to_um(get("effective_radius"))
    out["lwp_gm2"] = units.water_path_to_gm2(get("liquid_water_path"))
    if get("cloud_optical_depth_std") is not None:
        out["tau_std"] = get("cloud_optical_depth_std")
    if get("effective_radius_std") is not None:
        out["r_e_std_um"] = units.radius_to_um(get("effective_radius_std"))
    if get("liquid_water_path_std") is not None:
        out["lwp_std_gm2"] = units.water_path_to_gm2(get("liquid_water_path_std"))
    if get("number_of_solutions") is not None:
        out["n_solutions"] = get("number_of_solutions")
    if get("retrieval_flag") is not None:
        out["retrieval_flag"] = get("retrieval_flag")
    if get("solar_zenith_angle") is not None:
        out["sza_deg"] = get("solar_zenith_angle")
    if get("modis_white_sky_albedo") is not None:
        out["modis_albedo"] = get("modis_white_sky_albedo")
    if get("modis_wavelength") is not None:
        out = out.assign_coords(modis_wavelength_nm=get("modis_wavelength"))

    out["qc_bad_tau"] = qc.bad_mask(raw, "cloud_optical_depth")
    out.attrs = {
        "product": PRODUCT,
        "instrument": "SPHOT",
        "source_datastream": raw.attrs.get("source_datastream", "epcsphotcod2chiuM1.c1"),
    }
    return out


@dataclass
class Criteria:
    """Selection thresholds for the sunphotometer retrievals.

    ``good_retrieval_flags``: TR-317 does not tabulate the meanings of
    retrieval_flag. 0 = good is an assumption. Print
    ``epcape.qc.flag_meanings(std['retrieval_flag'])`` on real data and adjust."""

    use_qc: bool = True
    good_retrieval_flags: Optional[Sequence[int]] = (0,)
    n_solutions_min: Optional[int] = 1
    tau_min: Optional[float] = None
    sza_max_deg: Optional[float] = None
    max_rel_tau_std: Optional[float] = None  # e.g. 0.5 drops retrievals with std > 50% of tau

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["good_retrieval_flags"] is not None:
            d["good_retrieval_flags"] = list(d["good_retrieval_flags"])
        return d


def tau_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """Ordered named tests for a valid optical depth (True = passes)."""
    t = std["tau"]
    crit: filters.Criteria = {"finite tau > 0": np.isfinite(t) & (t > 0)}
    crit["QC (tau)"] = ~std["qc_bad_tau"] if c.use_qc else None
    if c.good_retrieval_flags is not None:
        crit[f"retrieval_flag in {tuple(c.good_retrieval_flags)}"] = (
            std["retrieval_flag"].isin(list(c.good_retrieval_flags)) if "retrieval_flag" in std else None
        )
    if c.n_solutions_min is not None:
        crit[f"n_solutions >= {c.n_solutions_min}"] = (
            std["n_solutions"] >= c.n_solutions_min if "n_solutions" in std else None
        )
    if c.tau_min is not None:
        crit[f"tau > {c.tau_min:g}"] = t > c.tau_min
    if c.sza_max_deg is not None:
        crit[f"SZA < {c.sza_max_deg:g} deg"] = std["sza_deg"] < c.sza_max_deg if "sza_deg" in std else None
    if c.max_rel_tau_std is not None:
        crit[f"tau std / tau < {c.max_rel_tau_std:g}"] = (
            (std["tau_std"] / t) < c.max_rel_tau_std if "tau_std" in std else None
        )
    return crit


def r_e_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """tau tests plus a finite, positive effective radius."""
    crit = dict(tau_criteria(std, c))
    r = std["r_e_um"]
    crit["finite r_e > 0"] = np.isfinite(r) & (r > 0)
    return crit


def lwp_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """r_e tests plus a finite LWP (LWP is computed from tau and r_e)."""
    crit = dict(r_e_criteria(std, c))
    crit["finite LWP"] = np.isfinite(std["lwp_gm2"])
    return crit
