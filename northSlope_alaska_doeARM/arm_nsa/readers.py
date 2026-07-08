"""Generic readers that turn raw ARM netCDF files into tidy xarray objects.

Responsibilities handled here, once, for every instrument:

* Locating local files for a pipeline datastream key (all product eras).
* Cheap date filtering using the YYYYMMDD stamp in ARM filenames, so a
  one-week analysis never opens twelve years of files.
* Resolving canonical variable names against the per-file candidates listed in
  config.DATASTREAMS (product renames are absorbed there).
* Applying ARM embedded QC flags (see qc.py) during the read.
* Concatenating along time, de-duplicating, and sorting.

Radiosonde files are one-profile-per-file and are handled separately in
sonde.py; this module covers the continuous time-series streams (KAZR, MWR,
ceilometer, MET, QCRAD).
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr

from . import config
from .download import local_files
from .qc import apply_qc

# ARM filenames look like <datastream>.YYYYMMDD.HHMMSS.<nc|cdf>
_FILE_DATE_RE = re.compile(r"\.(\d{8})\.(\d{6})\.(nc|cdf)$")


def file_timestamp(path: Path) -> Optional[dt.datetime]:
    """Parse the UTC timestamp embedded in an ARM filename (None if absent)."""
    match = _FILE_DATE_RE.search(path.name)
    if match is None:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def files_in_range(key: str, start_date: str, end_date: str) -> List[Path]:
    """Local files for `key` whose filename dates fall in [start, end].

    A one-day pad is applied at the front because an ARM daily file stamped
    2022-01-31 can contain records running into 2022-02-01 00:00.
    """
    start = dt.datetime.fromisoformat(start_date) - dt.timedelta(days=1)
    end = dt.datetime.fromisoformat(end_date) + dt.timedelta(days=1)
    selected: List[Path] = []
    for path in local_files(key):
        stamp = file_timestamp(path)
        if stamp is None or (start <= stamp <= end):
            selected.append(path)
    return selected


def resolve_variable(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    """Return the first candidate variable name present in the dataset.

    Raises a KeyError that lists the dataset's actual variables when nothing
    matches -- extend the candidate tuple in config.py if that happens with a
    new product version.
    """
    for name in candidates:
        if name in ds.variables:
            return name
    available = ", ".join(sorted(map(str, ds.data_vars)))
    raise KeyError(
        f"None of the candidate variables {tuple(candidates)} found. "
        f"Variables in file: {available}"
    )


def _read_one_file(
    path: Path,
    variable_map: Dict[str, Sequence[str]],
    apply_qc_flags: bool,
    include_indeterminate: bool,
) -> xr.Dataset:
    """Read one ARM file, returning canonicalized, QC-masked variables."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        out_vars: Dict[str, xr.DataArray] = {}
        for canonical, candidates in variable_map.items():
            native = resolve_variable(ds, candidates)
            if apply_qc_flags:
                arr = apply_qc(ds, native, include_indeterminate=include_indeterminate)
            else:
                arr = ds[native]
            arr.attrs.setdefault("source_variable", native)
            out_vars[canonical] = arr
        out = xr.Dataset(out_vars)
        out.attrs["source_file"] = path.name
        return out.load()  # pull into memory so the file handle can close


def read_timeseries(
    key: str,
    start_date: str,
    end_date: str,
    canonical_vars: Optional[Iterable[str]] = None,
    apply_qc_flags: bool = True,
    include_indeterminate: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    """Read a continuous ARM time-series datastream into one xarray Dataset.

    Parameters
    ----------
    key:
        Pipeline datastream key: "mwr", "ceil", "met", "qcrad" (or "kazr",
        though radar.read_kazr adds the required vertical-grid handling).
    start_date, end_date:
        Inclusive "YYYY-MM-DD" bounds on the time coordinate.
    canonical_vars:
        Subset of the spec's canonical variable names to read; default all.
    apply_qc_flags:
        Replace values failing "Bad" ARM QC tests with NaN (see qc.py).
    include_indeterminate:
        Also NaN-out "Indeterminate" QC values (stricter).

    Returns
    -------
    Dataset indexed by `time`, variables under canonical names (units carried
    in attrs from the source files, plus `source_variable` bookkeeping).

    Raises
    ------
    FileNotFoundError when no local files exist for the period -- run the
    download step first (scripts/download_nsa_data.py).
    """
    spec = config.get_spec(key)
    variable_map = dict(spec.variables)
    if key == "kazr":
        # range is a coordinate, not a time-series variable; radar.py handles it
        variable_map.pop("range_m", None)
    if canonical_vars is not None:
        variable_map = {k: variable_map[k] for k in canonical_vars}

    paths = files_in_range(key, start_date, end_date)
    if not paths:
        raise FileNotFoundError(
            f"No local files for '{key}' in {start_date}..{end_date} under "
            f"{config.RAW_DATA_DIR}. Download them first, e.g.:\n"
            f"  python scripts/download_nsa_data.py --datastreams {key} "
            f"--start {start_date} --end {end_date}"
        )

    pieces: List[xr.Dataset] = []
    for path in paths:
        if verbose:
            print(f"reading {path.name}")
        pieces.append(
            _read_one_file(path, variable_map, apply_qc_flags, include_indeterminate)
        )

    combined = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    combined = combined.sortby("time")
    # Duplicate timestamps can appear at file boundaries or after reprocessing;
    # keep the first occurrence.
    _, unique_idx = np.unique(combined["time"].values, return_index=True)
    if unique_idx.size != combined.sizes["time"]:
        combined = combined.isel(time=unique_idx)
    return combined.sel(time=slice(start_date, f"{end_date}T23:59:59"))


def restrict_to_winter(ds: xr.Dataset) -> xr.Dataset:
    """Keep only extended-winter (Nov-Mar) samples, per Hartig26 Sect. 2."""
    months = ds["time"].dt.month
    keep = xr.zeros_like(months, dtype=bool)
    for m in config.WINTER_MONTHS:
        keep = keep | (months == m)
    return ds.isel(time=np.where(keep.values)[0])
