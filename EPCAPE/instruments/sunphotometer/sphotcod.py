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
albedo pixel mixes ocean, beach and land. In the EPCAPE files the assumed
white-sky albedo at 858 nm is typically 0.19-0.23, a land/vegetation value
(open ocean is roughly 0.03-0.05 there), with occasional ocean-like days
(e.g. 0.038 in May 2023). The 870/1640 nm retrieval is sensitive to that
albedo. Whether it biases tau and r_e here is an inference from the
retrieval design, not a documented EPCAPE result; the notebook tests it.

EPCAPE file structure (checked 2026-10-06; differs from TR-317 Table 4):
* tau, r_e, LWP, their std, number_of_solutions and retrieval_flag have a
  second dimension `gain` (see GAIN_* below); `standardize` selects one.
* retrieval_flag is a bit-packed QC field with 7 tests, all "Bad"; it is not
  a simple "0 = good" code.
* modis_white_sky_albedo has one value set per daily file (no time
  dimension); the combine step repeats it along time so it is kept per day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from EPCAPE.analysis_tools import filters, qc, units
from EPCAPE.analysis_tools.products import load_product

PRODUCT = "cod_sphot_M1"

# The retrieval is run three times, on radiances from the Cimel's two gain
# settings and on their mean. Meanings come from the `gain` coordinate's
# flag_meanings in the EPCAPE files (checked 2026-10-06):
#   0 = A (aureole gain), 1 = K (sky gain), 2 = mean of A and K.
GAIN_A, GAIN_K, GAIN_MEAN = 0, 1, 2
GAIN_LABELS = {GAIN_A: "A (aureole gain)", GAIN_K: "K (sky gain)", GAIN_MEAN: "mean of A and K"}
DEFAULT_GAIN = GAIN_MEAN

# retrieval_flag is a bit-packed ARM QC field (flag_method = "bit"); every test
# is assessed "Bad". Tests 2-5 compare radiance differences between 440, 675,
# 870 and 1020 nm against the spectral signature expected for cloud; test 2's
# description explicitly refers to a vegetated surface. Test 1 = sun vs sky
# collimator mismatch > 20%, test 6 = fewer than 15 look-up-table solutions,
# test 7 = missing data. Read the full text with
# EPCAPE.analysis_tools.qc.bit_breakdown(raw["retrieval_flag"]).
SPECTRAL_SIGNATURE_BITS = (2, 3, 4, 5)


def load(start=None, end=None, **kwargs) -> xr.Dataset:
    """Raw combined product for [start, end] (see EPCAPE.analysis_tools.products.load_product)."""
    return load_product(PRODUCT, start, end, **kwargs)


def standardize(raw: xr.Dataset, gain: Optional[int] = DEFAULT_GAIN) -> xr.Dataset:
    """Tidy dataset with explicit units in the variable names, for one gain.

    Parameters
    ----------
    raw : combined SPHOTCOD product (``load``)
    gain : 0 = A, 1 = K, 2 = mean of A and K (default). None keeps the
        `gain` dimension (every variable that has it stays 2-D).

    Variables (absent ancillary fields are left out):
        tau, tau_std                   optical depth and its perturbation standard error
        r_e_um, r_e_std_um             effective radius (um)
        lwp_gm2, lwp_std_gm2           liquid water path (g m-2)
        n_solutions                    number of look-up-table solutions
        retrieval_flag                 bit-packed QC field (decoded by the criteria functions)
        sza_deg                        solar zenith angle (deg)
        modis_albedo                   MODIS white-sky albedo the retrieval assumed (time, modis_channel)
        qc_bad_tau                     ARM QC failure on tau, if a qc_cloud_optical_depth field exists
    """
    if gain is not None and "gain" in raw.dims:
        raw = raw.sel(gain=gain)
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
        "gain": "all" if gain is None else GAIN_LABELS.get(int(gain), str(gain)),
        "source_datastream": raw.attrs.get("source_datastream", "epcsphotcod2chiuM1.c1"),
    }
    return out


@dataclass
class Criteria:
    """Selection thresholds for the sunphotometer retrievals.

    use_retrieval_flag : drop samples whose retrieval_flag fails any test
        assessed "Bad" (all of them, in the EPCAPE files).
    ignore_flag_bits : test numbers to leave out of that decision, e.g.
        ``SPECTRAL_SIGNATURE_BITS`` to keep retrievals that fail only the
        spectral-signature tests (see the notebook's sensitivity section).
    n_solutions_min : extra threshold on number_of_solutions. Off by default
        because flag test 6 already rejects < 15 solutions.
    """

    use_qc: bool = True
    use_retrieval_flag: bool = True
    ignore_flag_bits: Sequence[int] = ()
    n_solutions_min: Optional[int] = None
    tau_min: Optional[float] = None
    sza_max_deg: Optional[float] = None
    max_rel_tau_std: Optional[float] = None  # e.g. 0.5 drops retrievals with std > 50% of tau

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ignore_flag_bits"] = list(d["ignore_flag_bits"])
        return d


def flag_bad(std: xr.Dataset, ignore_bits: Sequence[int] = ()) -> xr.DataArray:
    """True where retrieval_flag fails a "Bad" test (ignoring `ignore_bits`)."""
    return qc.bad_from_flag(std["retrieval_flag"], ignore_bits=ignore_bits)


def tau_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """Ordered named tests for a valid optical depth (True = passes)."""
    t = std["tau"]
    crit: filters.Criteria = {"finite tau > 0": np.isfinite(t) & (t > 0)}
    crit["QC (tau)"] = ~std["qc_bad_tau"] if c.use_qc else None
    if c.use_retrieval_flag:
        label = "retrieval_flag: no Bad test failed"
        if c.ignore_flag_bits:
            label += f" (tests {', '.join(map(str, c.ignore_flag_bits))} ignored)"
        crit[label] = ~flag_bad(std, c.ignore_flag_bits) if "retrieval_flag" in std else None
    if c.n_solutions_min is not None:
        crit[f"n_solutions >= {c.n_solutions_min}"] = (
            std["n_solutions"] >= c.n_solutions_min if "n_solutions" in std else None
        )
    if c.tau_min is not None:
        crit[f"tau > {c.tau_min:g}"] = t > c.tau_min
    if c.sza_max_deg is not None:
        crit[f"SZA < {c.sza_max_deg:g} deg"] = (
            std["sza_deg"] < c.sza_max_deg if filters.available(std, "sza_deg") is not None else None
        )
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


def albedo_at(std: xr.Dataset, wavelength_nm: float = 858.0) -> Optional[xr.DataArray]:
    """MODIS white-sky albedo the retrieval assumed, at the MODIS channel nearest
    `wavelength_nm` (858 nm is the band next to the 870 nm radiance channel).
    None if the file carries no albedo."""
    if "modis_albedo" not in std or "modis_wavelength_nm" not in std.coords:
        return None
    wl = np.asarray(std["modis_wavelength_nm"].values, dtype=float)
    i = int(np.nanargmin(np.abs(wl - wavelength_nm)))
    out = (
        std["modis_albedo"]
        .isel(modis_channel=i)
        .drop_vars(["modis_channel", "modis_wavelength_nm"], errors="ignore")
    )
    out.attrs = {"long_name": f"MODIS white-sky albedo at {wl[i]:g} nm assumed by SPHOTCOD", "units": "1"}
    return out
