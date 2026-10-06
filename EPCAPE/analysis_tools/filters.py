"""Helpers shared by the instrument filters.

Every instrument module builds an ordered dict of named criteria
``{name: DataArray[bool] (True = sample passes) or None (criterion skipped)}``.
Keeping the criteria separate, rather than one combined mask, lets the
notebook show how many samples each step removes (the "filter funnel").
That is often where two products' sampling differences become visible.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

Criteria = Dict[str, Optional[xr.DataArray]]


def find_var(ds: xr.Dataset, name: str) -> Optional[xr.DataArray]:
    """ds[name] matched exactly or case-insensitively; None if absent."""
    if name in ds:
        return ds[name]
    lower = {k.lower(): k for k in ds.variables}
    key = lower.get(name.lower())
    return ds[key] if key is not None else None


def available(ds: xr.Dataset, name: str) -> Optional[xr.DataArray]:
    """ds[name] if it exists and has at least one finite value, else None.

    Criteria built on an ancillary variable use this, so that a variable the
    files never fill (e.g. MFRSRCLDOD ir_temp at EPCAPE, 0% finite) makes the
    test *skipped* rather than failing every sample (NaN > x is False)."""
    if name not in ds:
        return None
    da = ds[name]
    if da.dtype.kind == "f" and not np.isfinite(da.values).any():
        return None
    return da


def combine(criteria: Criteria) -> xr.DataArray:
    """Logical AND of all criteria that were applied (skipped ones ignored)."""
    applied = [m for m in criteria.values() if m is not None]
    if not applied:
        raise ValueError("No criteria were applied.")
    out = applied[0].copy()
    for m in applied[1:]:
        out = out & m
    return out


def funnel(criteria: Criteria, label: str = "") -> pd.DataFrame:
    """Samples remaining after each criterion is applied in order.

    Columns: criterion, remaining (count), remaining_pct (of all samples),
    removed_here (count removed by this step, given the previous steps)."""
    rows = []
    running = None
    n_total = None
    for name, mask in criteria.items():
        if mask is None:
            rows.append(
                {
                    "criterion": f"{name} (skipped: variable missing or all-NaN)",
                    "remaining": np.nan,
                    "remaining_pct": np.nan,
                    "removed_here": 0,
                }
            )
            continue
        values = np.asarray(mask.values, dtype=bool)
        if n_total is None:
            n_total = values.size
        before = int(running.sum()) if running is not None else n_total
        running = values if running is None else (running & values)
        after = int(running.sum())
        rows.append(
            {
                "criterion": name,
                "remaining": after,
                "remaining_pct": 100 * after / max(n_total, 1),
                "removed_here": before - after,
            }
        )
    df = pd.DataFrame(rows)
    if label:
        df.insert(0, "product", label)
    return df


def apply(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Values where mask is True, NaN elsewhere (attributes kept)."""
    out = da.where(mask)
    out.attrs = dict(da.attrs)
    return out


def daily_counts(mask: xr.DataArray) -> pd.Series:
    """Number of passing samples per UTC day."""
    s = pd.Series(np.asarray(mask.values, dtype=bool), index=pd.DatetimeIndex(mask["time"].values))
    return s.resample("1D").sum()
