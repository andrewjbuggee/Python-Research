"""Decode ARM quality-control (QC) fields into boolean "bad sample" masks.

ARM pairs most variables X with an integer field qc_X. Two encodings occur:

1. Bit-packed (the ARM standard; ``flag_method = "bit"``). Each failed test
   sets one bit. Test N uses bit value 2**(N-1). Its meaning and severity are
   in the attributes ``bit_N_description`` and ``bit_N_assessment``, whose
   value is "Bad" or "Indeterminate". Newer files may instead carry the lists
   ``flag_masks``, ``flag_meanings`` and ``flag_assessments``.
   Reference: ARM Standards for Data Products, "Quality control fields"
   (https://www.arm.gov/guidance/datause/formatting-and-qc); the MWR handbook
   (DOE/SC-ARM-TR-016, Table 5) lists the same bits for mwrlos.b1:
   1 = missing, 2 = below minimum, 4 = above maximum, 8 = failed delta check.
2. Integer state (``flag_values`` + ``flag_meanings``): the value names one
   state. Here 0 is taken as good and anything else as bad, unless an
   assessment list says otherwise.

ACT (Atmospheric data Community Toolkit, ``act.qc``) implements the same
logic. A small local version is used here because ACT is not in this
project's environment, and these few rules are easy to audit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _as_list(value) -> List:
    """Attribute value -> list: space/comma separated strings and arrays both accepted."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(",", " ").split()
    return list(np.atleast_1d(value))


def qc_tests(qc: xr.DataArray) -> Dict[int, Tuple[str, str]]:
    """Bit value -> (description, assessment) for a bit-packed QC field.

    Returns an empty dict if the field does not describe its bits."""
    attrs = qc.attrs
    tests: Dict[int, Tuple[str, str]] = {}
    # Style 1: bit_N_description / bit_N_assessment
    for key, text in attrs.items():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "bit" and parts[2] == "description" and parts[1].isdigit():
            n = int(parts[1])
            assessment = str(attrs.get(f"bit_{n}_assessment", "Bad"))
            tests[2 ** (n - 1)] = (str(text), assessment)
    # Style 2: flag_masks / flag_meanings / flag_assessments
    masks = _as_list(attrs.get("flag_masks"))
    if masks and not tests:
        meanings = _as_list(attrs.get("flag_meanings"))
        assessments = _as_list(attrs.get("flag_assessments"))
        for i, m in enumerate(masks):
            desc = meanings[i].replace("_", " ") if i < len(meanings) else f"test with mask {m}"
            assess = assessments[i] if i < len(assessments) else "Bad"
            tests[int(m)] = (desc, str(assess))
    return tests


def bad_mask(
    ds: xr.Dataset,
    var: str,
    *,
    indeterminate_is_bad: bool = False,
    ignore_tests_matching: Iterable[str] = (),
) -> xr.DataArray:
    """True where `var` failed QC.

    Parameters
    ----------
    ds : dataset holding `var` and (ideally) ``qc_<var>``
    var : the data variable whose QC companion is decoded
    indeterminate_is_bad : count tests assessed "Indeterminate" as failures.
        False (default) keeps those samples, as ARM recommends for most uses.
    ignore_tests_matching : case-insensitive substrings; tests whose
        description contains one of them are ignored. Example: the MWR
        handbook says LWP "below minimum" (slightly negative) values within
        the retrieval RMS are usable, so the mwrlos filter passes ("minimum",).

    Rules
    -----
    * No ``qc_<var>`` field -> nothing is flagged (only NaNs in `var` itself
      will be removed by the caller's finite-value check).
    * A missing QC value (NaN after decoding the fill value) counts as bad.
    * Bit-packed field with described tests -> bad if any counted test's bit
      is set.
    * Bit-packed/integer field without descriptions -> bad if nonzero.
    """
    qc_name = f"qc_{var}"
    if qc_name not in ds:
        return xr.zeros_like(ds[var], dtype=bool)
    out = bad_from_flag(
        ds[qc_name], indeterminate_is_bad=indeterminate_is_bad, ignore_tests_matching=ignore_tests_matching
    )
    return out.rename(f"bad_{var}")


def bad_from_flag(
    flag: xr.DataArray,
    *,
    indeterminate_is_bad: bool = False,
    ignore_tests_matching: Iterable[str] = (),
    ignore_bits: Iterable[int] = (),
) -> xr.DataArray:
    """True where a bit-packed QC/flag field marks the sample as failed.

    Same rules as ``bad_mask``, but for any flag variable whatever its name
    (e.g. SPHOTCOD's ``retrieval_flag``, which is not called qc_<var>).

    ignore_bits : test numbers N (as in the ``bit_N_description`` attributes,
        so bit N has value 2**(N-1)) to leave out of the decision.
    """
    values = np.asarray(flag.values, dtype=float)
    missing = ~np.isfinite(values)
    ints = np.where(missing, 0, values).astype(np.int64)

    ignore = [s.lower() for s in ignore_tests_matching]
    skip_values = {2 ** (int(n) - 1) for n in ignore_bits}
    tests = qc_tests(flag)
    if tests:
        counted = 0
        for bit, (desc, assessment) in tests.items():
            if bit in skip_values or any(s in desc.lower() for s in ignore):
                continue
            if assessment.strip().lower() == "indeterminate" and not indeterminate_is_bad:
                continue
            counted |= bit
        bad = (ints & counted) != 0
    else:
        # undescribed field: any nonzero value is a failure, except ignored bits
        mask = ~0
        for v in skip_values:
            mask &= ~v
        bad = (ints & mask) != 0
    return xr.DataArray(bad | missing, coords=flag.coords, dims=flag.dims, name=f"bad_{flag.name}")


def bit_breakdown(flag: xr.DataArray) -> pd.DataFrame:
    """Table of each test in a bit-packed flag: bit number, value, assessment,
    percentage of samples failing it, percentage failing ONLY it, and its
    description. Missing flag values are excluded from the percentages."""
    values = np.asarray(flag.values, dtype=float).ravel()
    ints = values[np.isfinite(values)].astype(np.int64)
    n = max(ints.size, 1)
    rows = []
    for bit, (desc, assessment) in sorted(qc_tests(flag).items()):
        rows.append(
            {
                "bit": int(np.log2(bit)) + 1,
                "value": bit,
                "assessment": assessment,
                "failing_pct": 100 * ((ints & bit) != 0).sum() / n,
                "only_this_pct": 100 * (ints == bit).sum() / n,
                "description": desc,
            }
        )
    rows.append(
        {
            "bit": "-",
            "value": 0,
            "assessment": "",
            "failing_pct": np.nan,
            "only_this_pct": 100 * (ints == 0).sum() / n,
            "description": "passed every test",
        }
    )
    return pd.DataFrame(rows)


def has_qc(ds: xr.Dataset, var: str) -> bool:
    """True if the dataset carries a ``qc_<var>`` companion for `var`."""
    return f"qc_{var}" in ds


def qc_is_zero(ds: xr.Dataset, var: str) -> xr.DataArray:
    """True where ``qc_<var>`` is exactly 0: the strictest ARM screen.

    A value of 0 means the sample passed every test, both those assessed
    "Bad" and those assessed "Indeterminate". ``bad_mask`` by default keeps
    Indeterminate samples; this does not. A missing QC value (NaN after
    decoding the fill value, e.g. a day whose file lacked the QC field)
    counts as not-zero, so that sample is dropped.

    If there is no ``qc_<var>`` field, every sample passes: there is
    nothing to screen on. Check ``has_qc`` first and report that case.
    """
    qc_name = f"qc_{var}"
    if qc_name not in ds:
        return xr.ones_like(ds[var], dtype=bool).rename(f"qc0_{var}")
    values = np.asarray(ds[qc_name].values, dtype=float)
    ok = np.isfinite(values) & (values == 0)
    return xr.DataArray(ok, coords=ds[qc_name].coords, dims=ds[qc_name].dims, name=f"qc0_{var}")


def describe_qc(ds: xr.Dataset, var: str) -> str:
    """Readable summary of what qc_<var> tests and how often each test fails."""
    qc_name = f"qc_{var}"
    if qc_name not in ds:
        return f"{var}: no QC field"
    qc = ds[qc_name]
    values = np.asarray(qc.values, dtype=float)
    finite = np.isfinite(values)
    ints = values[finite].astype(np.int64)
    n = values.size
    lines = [f"{qc_name}: {n:,} samples, {100 * (~finite).mean():.1f}% QC value missing"]
    tests = qc_tests(qc)
    if not tests:
        lines.append(f"  no test descriptions; nonzero in {100 * (ints != 0).sum() / max(n, 1):.1f}%")
    for bit, (desc, assessment) in sorted(tests.items()):
        frac = 100 * ((ints & bit) != 0).sum() / max(n, 1)
        lines.append(f"  bit {bit:>4} [{assessment:<13}] {frac:5.1f}%  {desc}")
    return "\n".join(lines)


def flag_meanings(da: xr.DataArray) -> Optional[Dict[int, str]]:
    """Value -> meaning for an integer-state flag (flag_values/flag_meanings or
    flag_N_description attributes), or None if the field does not say."""
    out: Dict[int, str] = {}
    values = _as_list(da.attrs.get("flag_values"))
    meanings = _as_list(da.attrs.get("flag_meanings"))
    for v, m in zip(values, meanings):
        try:
            out[int(float(v))] = str(m).replace("_", " ")
        except (TypeError, ValueError):
            continue
    for key, text in da.attrs.items():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "flag" and parts[2] == "description" and parts[1].isdigit():
            out[int(parts[1])] = str(text)
    return out or None
