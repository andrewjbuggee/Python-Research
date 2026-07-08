"""Handling of ARM embedded quality-control (QC) flags.

Every ARM b1/c1-level variable `x` may ship with a companion integer variable
`qc_x` whose bits encode the outcome of automated QC tests (missing value,
below minimum, above maximum, failed delta check, instrument-specific tests).
The meaning and severity of each bit is stored in netCDF attributes, in one of
two conventions used across ARM's history:

    new style:  qc_x.attrs["flag_masks"]       = [1, 2, 4, 8, ...]
                qc_x.attrs["flag_assessments"] = ["Bad", "Bad", "Indeterminate", ...]

    old style:  qc_x.attrs["bit_1_description"] = "...", and
                qc_x.attrs["bit_1_assessment"]  = "Bad" / "Indeterminate", etc.

The functions here decode either convention and build boolean masks. Default
behavior masks values whose QC flags include at least one "Bad" test;
"Indeterminate" values are kept (set include_indeterminate=True to drop them
too). If a variable has no qc_ companion, nothing is masked.

Reference: ARM Data Quality documentation, https://www.arm.gov/guidance/datause
(see "Data Quality" and the QC bit conventions in the netCDF file headers).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import xarray as xr

# Values ARM uses to mark missing data when nothing was measured. Applied in
# addition to any _FillValue/missing_value attribute decoding done by xarray.
ARM_MISSING_SENTINELS = (-9999.0, -7999.0, 9999.0)


def _bit_severities(qc_var: xr.DataArray) -> List[Tuple[int, str]]:
    """Return [(bitmask, assessment), ...] parsed from qc-variable attrs."""
    attrs = qc_var.attrs
    out: List[Tuple[int, str]] = []

    masks = attrs.get("flag_masks")
    assessments = attrs.get("flag_assessments")
    if masks is not None and assessments is not None:
        masks_list = np.atleast_1d(masks).astype("int64").tolist()
        assess_list = [str(a) for a in np.atleast_1d(assessments)]
        out = list(zip(masks_list, assess_list))
        if out:
            return out

    # Old-style attributes: bit_1_assessment, bit_2_assessment, ...
    bit = 1
    while f"bit_{bit}_description" in attrs or f"bit_{bit}_assessment" in attrs:
        assessment = str(attrs.get(f"bit_{bit}_assessment", "Bad"))
        out.append((1 << (bit - 1), assessment))
        bit += 1
    return out


def qc_mask(
    ds: xr.Dataset, var_name: str, include_indeterminate: bool = False
) -> xr.DataArray:
    """Boolean mask, True where `var_name` should be discarded per its QC.

    Parameters
    ----------
    ds:
        Dataset containing `var_name` and (optionally) `qc_<var_name>`.
    var_name:
        Name of the data variable inside `ds` (the file-native name).
    include_indeterminate:
        When True, also mask "Indeterminate" flags (stricter). Default masks
        only "Bad".

    Returns
    -------
    Boolean DataArray with the same shape as the variable. All False when no
    QC companion exists.
    """
    data = ds[var_name]
    qc_name = f"qc_{var_name}"
    mask = xr.zeros_like(data, dtype=bool)

    if qc_name in ds:
        qc_var = ds[qc_name]
        qc_int = qc_var.fillna(0).astype("int64")
        severities = _bit_severities(qc_var)
        if severities:
            for bitmask, assessment in severities:
                is_bad = assessment.strip().lower() == "bad"
                if is_bad or (include_indeterminate and not is_bad):
                    mask = mask | ((qc_int & bitmask) != 0)
        else:
            # No decodable bit metadata: treat ANY nonzero flag as suspect.
            # Conservative, but a qc variable without documentation is rare.
            mask = mask | (qc_int != 0)

    # Also catch ARM's numeric missing sentinels that survive decoding.
    for sentinel in ARM_MISSING_SENTINELS:
        mask = mask | (data == sentinel)
    return mask


def apply_qc(
    ds: xr.Dataset, var_name: str, include_indeterminate: bool = False
) -> xr.DataArray:
    """Return `var_name` with QC-flagged values replaced by NaN.

    The returned DataArray keeps the original attributes (units etc.).
    """
    data = ds[var_name]
    mask = qc_mask(ds, var_name, include_indeterminate=include_indeterminate)
    cleaned = data.where(~mask)
    cleaned.attrs = dict(data.attrs)
    return cleaned
