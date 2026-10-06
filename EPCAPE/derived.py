"""Write derived (filtered, collocated) datasets for MATLAB and later reuse.

Derived products are the part of the workflow that cannot be re-downloaded,
so each file records how it was made: the git commit of this repository,
the analysis settings (as global attributes), and the processed source
files. Output follows the same conventions as ``epcape.combine``:

    time            float64 seconds since 1970-01-01 UTC
                    (MATLAB: datetime(t, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC'))
    float variables NaN for missing
    boolean masks   stored as int8 0/1 (MATLAB reads them as int8; use logical())

Files go to <data folder>/processed/derived/<name>.nc.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

from .combine import TIME_UNITS
from .config import REPO_ROOT, Machine, active_machine


def git_revision(repo: Path = REPO_ROOT) -> str:
    """Current commit hash (+ '-dirty' if there are uncommitted changes), or 'unknown'."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_derived(
    ds: xr.Dataset,
    name: str,
    *,
    settings: Optional[Dict[str, object]] = None,
    machine: Optional[Machine] = None,
    out: Optional[Path] = None,
) -> Path:
    """Write `ds` to processed/derived/<name>.nc in MATLAB-friendly form; return the path."""
    machine = machine or active_machine()
    out = Path(out) if out else machine.processed_dir() / "derived" / f"{name}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = ds.copy()
    # Booleans -> int8 so MATLAB and every netCDF reader handle them.
    for v in list(ds.data_vars):
        if ds[v].dtype == bool:
            attrs = dict(ds[v].attrs)
            ds[v] = ds[v].astype(np.int8)
            ds[v].attrs = {
                **attrs,
                "flag_values": np.array([0, 1], dtype=np.int8),
                "flag_meanings": "false true",
            }

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    attrs = dict(ds.attrs)
    attrs.update(
        history=f"{now} UTC: written by epcape.derived.save_derived",
        epcape_git_revision=git_revision(),
    )
    for key, value in (settings or {}).items():
        # netCDF attributes must be str/number/array; None and bools become strings
        if value is None or isinstance(value, bool):
            value = str(value)
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        attrs[f"setting_{key}"] = value
    ds.attrs = attrs

    encoding = {}
    for name_, var in ds.variables.items():
        if name_ == "time" or np.issubdtype(var.dtype, np.datetime64):
            encoding[name_] = {"units": TIME_UNITS, "calendar": "standard", "dtype": "float64"}
        elif var.dtype.kind == "f":
            encoding[name_] = {"_FillValue": np.nan, "zlib": True, "complevel": 4}
        elif var.dtype.kind in "iu":
            encoding[name_] = {"_FillValue": None, "zlib": True, "complevel": 4}

    tmp = out.with_name(out.name + ".part")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out
