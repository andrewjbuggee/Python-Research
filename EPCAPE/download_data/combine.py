"""Merge daily ARM files into one netCDF that MATLAB and Python read directly.

Output conventions:
  time            float64 seconds since 1970-01-01 UTC (POSIX time)
  float variables missing values are NaN
  integer flags   original integer type; missing values are the ARM fill (-9999)
Variable attributes (units, long_name, QC bit descriptions) are kept.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

from .arm_files import format_runs, list_local, missing_days
from .config import Machine, Product, active_machine
from .sync import MANIFEST, netcdf_variables, resolve_variables

TIME_UNITS = "seconds since 1970-01-01"  # UTC; MATLAB: datetime(t,'ConvertFrom','posixtime')
PER_FILE_TIME = ("base_time", "time_offset")  # replaced by a continuous time axis
PER_FILE_GLOBALS = ("history", "input_source", "input_datastreams")


def combine_files(
    files: Sequence[Path],
    variables: Sequence[str],
    out_path: Path,
    *,
    title: str = "",
    extra_attrs: Optional[Dict[str, object]] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Concatenate `variables` from `files` along time and write one netCDF-4 file."""
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("No input files to combine.")
    wanted = [v for v in variables if v not in PER_FILE_TIME and v != "time"]

    pieces: List[xr.Dataset] = []
    source_encoding: Dict[str, dict] = {}
    absent: Dict[str, int] = {}
    first_attrs: Dict[str, object] = {}
    for i, path in enumerate(files):
        with xr.open_dataset(path) as ds:
            ds = _with_time(ds, path)
            keep = [v for v in wanted if v in ds.variables]
            for v in wanted:
                if v not in ds.variables:
                    absent[v] = absent.get(v, 0) + 1
            for v in keep:
                source_encoding.setdefault(v, dict(ds[v].encoding))
            if i == 0:
                first_attrs = dict(ds.attrs)
            piece = ds[keep]
            if "time" not in piece.coords:
                piece = piece.assign_coords(time=ds["time"])
            pieces.append(piece.load())
    for v, n in absent.items():
        log(f"  note: {v} is missing from {n} of {len(files)} files; those times are filled as missing")

    _check_static_coords(pieces, files)
    pieces = _fill_absent(pieces, wanted)
    pieces, varying = _expand_varying_static(pieces, wanted)
    for v in varying:
        log(
            f"  note: {v} has no time dimension but differs between files; "
            "it is repeated along time so every file's value is kept"
        )
    combined = xr.concat(
        pieces,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="override",
    )
    combined = combined.sortby("time")
    t = combined["time"].values
    duplicate = np.concatenate([[False], t[1:] == t[:-1]])
    if duplicate.any():
        combined = combined.isel(time=~duplicate)
        log(f"  dropped {int(duplicate.sum())} duplicate time stamps")

    time_attrs = {k: v for k, v in combined["time"].attrs.items() if k not in ("bounds", "units", "calendar")}
    time_attrs.update(long_name="Time (UTC)", standard_name="time")
    combined["time"].attrs = time_attrs

    attrs = {k: v for k, v in first_attrs.items() if k not in PER_FILE_GLOBALS and not k.startswith("_")}
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    times = combined["time"].values
    attrs.update(
        {
            "title": title or attrs.get("title", ""),
            "source_file_count": len(files),
            "source_first_file": files[0].name,
            "source_last_file": files[-1].name,
            "time_coverage_start": str(times[0])[:19] + "Z",
            "time_coverage_end": str(times[-1])[:19] + "Z",
            "history": f"{now} UTC: combined from {len(files)} ARM files by the epcape tools",
            "attribute_note": "Global attributes not added by the epcape tools come from the first source file.",
        }
    )
    for key, value in (extra_attrs or {}).items():
        if value is not None:
            attrs[key] = value
    combined.attrs = attrs

    encoding = {"time": {"units": TIME_UNITS, "calendar": "standard", "dtype": "float64"}}
    for name, var in combined.variables.items():
        if name == "time":
            continue
        var.attrs.pop("_FillValue", None)
        var.attrs.pop("missing_value", None)
        if name in combined.coords:
            encoding[name] = {"_FillValue": None}
        else:
            encoding[name] = _encoding_for(var, source_encoding.get(name, {}))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    try:
        combined.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out_path


def combine_product(
    product: Product,
    start: dt.date,
    end: dt.date,
    *,
    machine: Optional[Machine] = None,
    source: str = "auto",
    out: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Combine a product's files for [start, end] into data/processed/."""
    machine = machine or active_machine()
    candidates = {
        "archive": machine.archive_dir(product.datastream),
        "subset": machine.subset_dir(product.name),
        "full": machine.full_dir(product.datastream),
    }
    order = ["archive", "subset", "full"] if source == "auto" else [source]
    found = []  # (n_files, -preference, source, directory, files); most files wins
    for rank, key in enumerate(order):
        directory = candidates.get(key)
        if directory is None:
            continue
        files = list_local(directory, start, end, product.datastream)
        if files:
            found.append((len(files), -rank, key, directory, files))
    if not found:
        raise FileNotFoundError(
            f"No {product.datastream} files dated {start} to {end} under {machine.data_root}.\n"
            f"Download them first:  python download_data/download_arm.py {product.name}"
        )
    _, _, key, directory, files = max(found, key=lambda f: (f[0], f[1]))
    log(f"Combining {len(files)} files from {directory} ({key}).")

    available = {}  # union over files: a variable absent from some files is still kept
    for f in files:
        for name, dims in netcdf_variables(f).items():
            available.setdefault(name, dims)
    variables = resolve_variables(product.variables, available, product.optional_variables)
    manifest = {}
    if (directory / MANIFEST).is_file():
        try:
            manifest = json.loads((directory / MANIFEST).read_text())
        except ValueError:
            pass
    out = Path(out) if out else machine.processed_dir() / f"{product.name}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    combine_files(
        files,
        variables,
        out,
        title=f"EPCAPE: {product.description}" if product.description else "",
        extra_attrs={
            "source_datastream": product.datastream,
            "epcape_product": product.name,
            "citation": manifest.get("citation"),
        },
        log=log,
    )
    summarize(out, start, end, log=log)
    return out


def summarize(
    path: Path,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    log: Callable[[str], None] = print,
) -> None:
    """Print coverage and simple statistics as a sanity check."""
    path = Path(path)
    with xr.open_dataset(path) as ds:
        t = ds["time"].values
        log(f"Wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        log(f"  {t.size:,} samples from {str(t[0])[:19]} to {str(t[-1])[:19]} UTC")
        if t.size > 1:
            step = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0
            log(f"  median time step {np.median(step):.1f} s")
        if start and end:
            days = {d.astype(object) for d in np.unique(t.astype("datetime64[D]"))}
            gaps = missing_days(days, start, end)
            n_missing = sum((b - a).days + 1 for a, b in gaps)
            if gaps:
                log(f"  days with no data: {n_missing} ({format_runs(gaps)})")
            else:
                log(f"  every day from {start} to {end} has data")
        for name, da in ds.data_vars.items():
            if "time" not in da.dims or name.startswith("qc_") or da.dtype.kind not in "fiu":
                continue
            values = np.asarray(da.values, dtype=float)
            valid = np.isfinite(values)
            line = f"  {name:<22} {100 * valid.mean():5.1f}% of samples non-missing"
            if valid.any() and da.attrs.get("units") == "m":
                p5, p50, p95 = np.nanpercentile(values, [5, 50, 95])
                line += f"; median {p50:.0f} m (5th-95th percentile {p5:.0f}-{p95:.0f} m)"
            log(line)
        if "detection_status" in ds:
            codes = ds["detection_status"].values
            codes = codes[np.isfinite(codes)].astype(int)
            if codes.size:
                values, counts = np.unique(codes, return_counts=True)
                desc = _flag_descriptions(ds["detection_status"].attrs)
                log("  detection_status:")
                for value, count in zip(values, counts):
                    log(f"    {value}  {100 * count / codes.size:5.1f}%  {desc.get(int(value), '')}")


# -- helpers ---------------------------------------------------------------
def _with_time(ds: xr.Dataset, path: Path) -> xr.Dataset:
    """Make sure `time` holds absolute datetimes (rebuilt from time_offset if needed)."""
    if "time" in ds.variables and np.issubdtype(ds["time"].dtype, np.datetime64):
        return ds
    if "time_offset" in ds.variables and np.issubdtype(ds["time_offset"].dtype, np.datetime64):
        return ds.assign_coords(time=("time", ds["time_offset"].values))
    raise ValueError(f"{path.name}: no decodable time variable")


def _check_static_coords(pieces: Sequence[xr.Dataset], files: Sequence[Path]) -> None:
    ref = pieces[0]
    for piece, path in zip(pieces[1:], files[1:]):
        for name, coord in piece.coords.items():
            if "time" in coord.dims or name not in ref.coords:
                continue
            if not coord.equals(ref.coords[name]):
                raise ValueError(
                    f"Coordinate {name!r} in {path.name} differs from {files[0].name}; "
                    "these files cannot be stacked in time without regridding."
                )


def _expand_varying_static(pieces: List[xr.Dataset], wanted: Sequence[str]):
    """Give a time dimension to data variables that lack one but differ between files.

    Example: SPHOTCOD's modis_white_sky_albedo(modis_channel) is one value set
    per daily file, updated as MODIS albedo changes. Concatenating along time
    would otherwise keep only the first file's values (compat="override").
    Each file's values are repeated for every sample of that file, so a
    sample's value is the one its own file used. Variables that are identical
    in every file (lat, lon, alt, wavelengths) are left as they are.
    Returns the pieces and the names of the variables that were expanded."""
    varying = []
    for v in wanted:
        ref = pieces[0][v] if v in pieces[0].data_vars else None
        if ref is None or "time" in ref.dims:
            continue
        for piece in pieces[1:]:
            other = piece[v] if v in piece.data_vars else None
            if (
                other is None
                or other.shape != ref.shape
                or not np.array_equal(
                    np.asarray(other.values), np.asarray(ref.values), equal_nan=ref.dtype.kind == "f"
                )
            ):
                varying.append(v)
                break
    if not varying:
        return pieces, varying
    out = []
    for piece in pieces:
        additions = {}
        for v in varying:
            da = piece[v]
            additions[v] = da.expand_dims(time=piece["time"].values).transpose("time", *da.dims)
            additions[v].attrs = dict(da.attrs)
        out.append(piece.assign(additions))
    return out, varying


def _fill_absent(pieces: List[xr.Dataset], wanted: Sequence[str]) -> List[xr.Dataset]:
    """Give every piece every variable; missing ones become all-missing arrays."""
    templates: Dict[str, xr.DataArray] = {}
    for piece in pieces:
        for v in wanted:
            if v in piece.variables and v not in templates:
                templates[v] = piece[v]
    out = []
    for piece in pieces:
        additions = {}
        for v, tpl in templates.items():
            if v in piece.variables:
                continue
            if "time" not in tpl.dims:
                additions[v] = tpl
                continue
            shape = tuple(piece.sizes["time"] if d == "time" else tpl.sizes[d] for d in tpl.dims)
            dtype = tpl.dtype if tpl.dtype.kind == "f" else np.float64
            additions[v] = xr.DataArray(
                np.full(shape, np.nan, dtype=dtype), dims=tpl.dims, attrs=dict(tpl.attrs)
            )
        out.append(piece.assign(additions) if additions else piece)
    return out


def _encoding_for(var: xr.Variable, source: dict) -> dict:
    dtype = np.dtype(source.get("dtype", var.dtype))
    fill = source.get("_FillValue", source.get("missing_value"))
    if fill is not None and np.size(fill) >= 1:
        fill = np.asarray(fill).ravel()[0].item()
    else:
        fill = None
    if dtype.kind in "iu":
        has_nan = var.dtype.kind == "f" and bool(np.isnan(var.values).any())
        if fill is None and has_nan:
            fill = -9999 if dtype.kind == "i" else int(np.iinfo(dtype).max)
        enc = {"dtype": dtype, "zlib": True, "complevel": 4, "_FillValue": None}
        if fill is not None:
            enc["_FillValue"] = dtype.type(fill)
            enc["missing_value"] = dtype.type(fill)
        return enc
    if dtype.kind == "f":
        return {"dtype": dtype, "zlib": True, "complevel": 4, "_FillValue": dtype.type(np.nan)}
    return {}


def _flag_descriptions(attrs: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    values = attrs.get("flag_values")
    meanings = attrs.get("flag_meanings")
    if values is not None and isinstance(meanings, str):
        if isinstance(values, str):
            values = values.replace(",", " ").split()
        try:
            for v, m in zip(np.atleast_1d(values), meanings.split()):
                out[int(v)] = m.replace("_", " ")
        except (TypeError, ValueError):
            pass
    for key, text in attrs.items():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "flag" and parts[2] == "description" and parts[1].isdigit():
            out[int(parts[1])] = str(text)
    return out
