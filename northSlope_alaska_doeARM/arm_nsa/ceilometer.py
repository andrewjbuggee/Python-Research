"""Vaisala CL31 ceilometer: lowest cloud base height.

Hartig26 Table 1 uses the standard proprietary Vaisala cloud-base detection
from nsaceilC1.b1 (15-s resolution, 10-m height resolution). Only the first
(lowest) cloud base height is used in the pipeline.

Detection-status semantics for the CL31 ingest (`detection_status`):
    0  no significant backscatter (clear as far as the ceilometer can see)
    1  one cloud base detected
    2  two cloud bases detected
    3  three cloud bases detected
    4  full obscuration, no cloud base determined (e.g. fog, heavy snow)
    5  some obscuration but determined to be transparent
first_cbh is meaningful for status 1-3. Status 4 means the beam was
extinguished without a base determination -- NOT clear sky; those times get
NaN cloud base but detected=True here. Blowing snow at NSA commonly produces
status 4 in winter, so treat low "cloud bases" in storms with suspicion.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .readers import read_timeseries


def read_ceilometer(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read first cloud base height for a date range.

    Returns
    -------
    Dataset over time with:
        first_cbh_m       lowest detected cloud base [m AGL]; NaN when no
                          base was determined (clear OR fully obscured)
        detection_status  raw instrument status code (see module docstring)
        cloud_detected    boolean, True for status 1-4 (hydrometeors present)
    """
    ds = read_timeseries(
        "ceil",
        start_date,
        end_date,
        apply_qc_flags=apply_qc_flags,
        verbose=verbose,
    )
    cbh = ds["first_cbh_m"]
    # Negative/zero cloud bases are fill or nonsense; mask them.
    ds["first_cbh_m"] = cbh.where(cbh > 0)
    ds["first_cbh_m"].attrs.setdefault("units", "m")
    ds["first_cbh_m"].attrs["long_name"] = "lowest cloud base height above ground"

    status = ds["detection_status"]
    # Status is numeric in most file versions but a character/byte array in
    # some; normalize to float with NaN for anything non-numeric.
    raw = np.asarray(status.values).ravel()
    if status.dtype.kind in "iuf":
        numeric = raw.astype(float)
    else:
        numeric = np.array(
            [
                float(s) if str(s).strip().lstrip("b'\"").rstrip("'\"").isdigit()
                else np.nan
                for s in (x.decode() if isinstance(x, bytes) else x for x in raw)
            ]
        )
    status_num = xr.DataArray(
        numeric.reshape(status.shape), dims=status.dims, coords=status.coords
    )
    ds["cloud_detected"] = (status_num >= 1) & (status_num <= 4)
    ds["cloud_detected"].attrs["long_name"] = (
        "hydrometeors detected (incl. full obscuration, status 4)"
    )
    return ds
