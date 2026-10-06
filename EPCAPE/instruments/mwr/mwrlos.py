"""MWRLOS: liquid water path and precipitable water vapour from the
2-channel (23.8 / 31.4 GHz) microwave radiometer.

Datastream  epcmwrlosM1.b1 (config product ``lwp_mwr_M1``)
Reference   Morris, V. R. (2019), Microwave Radiometer (MWR) Instrument
            Handbook, DOE/SC-ARM-TR-016.

Physics (TR-016 sections 5.1.1 and 7.2)
---------------------------------------
Cloud liquid emits a continuum that grows with frequency, so it dominates
the 31.4 GHz channel. Water vapour dominates 23.8 GHz, which sits on the
"hinge point" of the 22.2 GHz line, where emission does not depend on
pressure. ``liq`` and ``vap`` come from a statistical (linear-regression)
retrieval on the two channels' optical depths. The coefficients are tuned to
a climatological mean radiating temperature that updates only monthly.

Measurement characteristics relevant to comparisons
---------------------------------------------------
* Line of sight (LOS): ``liq`` is the path along the beam. It equals the
  vertical LWP only when the radiometer points at zenith (``actel`` = 90).
* Field of view 4.5-5.9 deg (TR-016 Table 8); about 50 m across at 500 m
  cloud base.
* LWP noise is ~3 g m-2, but the retrieval's residual RMS is ~30 g m-2
  (TR-016 FAQ). Values within +/- 30 g m-2 of zero can be clear sky, and
  small negative values are physically zero, not bad data.
* Works day and night, unlike the two solar instruments.

Quality rules used here (all from TR-016)
-----------------------------------------
* ARM QC bits, except the "minimum" test on liq. The handbook says negative
  LWP within the retrieval RMS is usable (FAQ: "Can we use the data when there
  are long periods of qcmin flags for liquid?").
* wet_window == 1: the heater is on because the rain/dew sensor triggered
  (rain, drizzle, dew). Liquid on the Teflon window makes LWP spuriously large.
* Sky brightness temperature > 100 K in either channel: precipitation or a
  wet window (Westwater rule; also used by MFRSRCLDOD, TR-047 section 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import xarray as xr

from epcape import filters, qc, units
from epcape.products import load_product

PRODUCT = "lwp_mwr_M1"
MIN_TEST_PATTERNS = ("valid_min", "minimum")  # QC test descriptions of the lower-limit check


def load(start=None, end=None, **kwargs) -> xr.Dataset:
    """Raw combined product for [start, end] (see epcape.products.load_product)."""
    return load_product(PRODUCT, start, end, **kwargs)


def standardize(raw: xr.Dataset) -> xr.Dataset:
    """Tidy dataset with explicit units in the variable names.

    Variables (absent ancillary fields are left out):
        lwp_gm2          liquid water path along the line of sight (g m-2)
        pwv_cm           water vapour path along the line of sight (cm)
        tb23_K, tb31_K   sky brightness temperatures (K)
        wet_window       1 = window heater on (rain/dew sensor triggered)
        sky_ir_temp_K    IR sky brightness temperature (K)
        elevation_deg    pointing elevation (deg; 90 = zenith)
        qc_bad_lwp       ARM QC failure on liq (minimum test ignored, see module doc)
        qc_bad_pwv       ARM QC failure on vap
    """
    out = xr.Dataset(coords={"time": raw["time"]})
    get = lambda n: filters.find_var(raw, n)  # noqa: E731

    out["lwp_gm2"] = units.water_path_to_gm2(get("liq"))
    out["pwv_cm"] = get("vap")
    for src, dst in [
        ("tbsky23", "tb23_K"),
        ("tbsky31", "tb31_K"),
        ("wet_window", "wet_window"),
        ("sky_ir_temp", "sky_ir_temp_K"),
        ("actel", "elevation_deg"),
    ]:
        if get(src) is not None:
            out[dst] = get(src)
    liq_name = get("liq").name
    # ARM's standard wording for this test is "Value is less than the valid_min";
    # older files say "minimum". Both are matched.
    out["qc_bad_lwp"] = qc.bad_mask(raw, liq_name, ignore_tests_matching=MIN_TEST_PATTERNS)
    out["qc_bad_pwv"] = qc.bad_mask(raw, get("vap").name)
    out.attrs = {
        "product": PRODUCT,
        "instrument": "MWR",
        "source_datastream": raw.attrs.get("source_datastream", "epcmwrlosM1.b1"),
    }
    return out


@dataclass
class Criteria:
    """Selection thresholds for MWR LWP. Defaults follow TR-016."""

    use_qc: bool = True
    reject_wet_window: bool = True
    tb_max_K: Optional[float] = 100.0  # rain / wet-window rule (TR-016 FAQ)
    zenith_tolerance_deg: Optional[float] = 1.0  # keep only zenith-pointing samples
    lwp_min_gm2: Optional[float] = None  # e.g. 30 to keep only clearly cloudy samples

    def as_dict(self) -> dict:
        return asdict(self)


def lwp_criteria(std: xr.Dataset, c: Criteria = Criteria()) -> filters.Criteria:
    """Ordered named tests for a usable LWP (True = passes)."""
    lwp = std["lwp_gm2"]
    crit: filters.Criteria = {"finite LWP": np.isfinite(lwp)}
    crit["QC (liq, minimum test ignored)"] = ~std["qc_bad_lwp"] if c.use_qc else None
    if c.reject_wet_window:
        # NaN wet_window (missing) is treated as dry: the Tb test below still
        # catches rain. (wet_window != 1) is True for NaN.
        crit["window dry (wet_window != 1)"] = (std["wet_window"] != 1) if "wet_window" in std else None
    if c.tb_max_K is not None:
        if "tb23_K" in std and "tb31_K" in std:
            crit[f"Tb23, Tb31 < {c.tb_max_K:g} K"] = (std["tb23_K"] < c.tb_max_K) & (
                std["tb31_K"] < c.tb_max_K
            )
        else:
            crit[f"Tb23, Tb31 < {c.tb_max_K:g} K"] = None
    if c.zenith_tolerance_deg is not None:
        crit[f"zenith pointing (|elev - 90| <= {c.zenith_tolerance_deg:g} deg)"] = (
            np.abs(std["elevation_deg"] - 90) <= c.zenith_tolerance_deg if "elevation_deg" in std else None
        )
    if c.lwp_min_gm2 is not None:
        crit[f"LWP > {c.lwp_min_gm2:g} g m-2"] = lwp > c.lwp_min_gm2
    return crit
