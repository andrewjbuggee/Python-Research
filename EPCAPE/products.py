"""Load a configured product as one time-continuous xarray Dataset.

This is the entry point the analysis code uses. It never reads daily ARM files
directly: it reads the combined file that ``combine_product.py`` writes to
``<data folder>/processed/`` and builds that file first if it does not exist.
Where the daily files come from (a mounted ARM archive on JupyterHub/Cumulus,
server-side subsets, or complete downloads) is decided by ``config.yaml`` and
``epcape.combine``, so the same notebook runs unchanged on every machine.

The combined file always spans the full campaign; a shorter [start, end]
window is cut from it in memory. One cached file per product keeps the
processed folder simple and makes reruns with different dates instant.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import xarray as xr

from .combine import combine_product
from .config import Machine, active_machine, as_date, campaign_dates, get_product, load_config


def processed_path(name: str, machine: Optional[Machine] = None, cfg: Optional[dict] = None) -> Path:
    """Path of the full-campaign combined file for product `name`.

    Matches the default output name of ``combine_product.py <name>``, so a file
    built from the command line is found here and vice versa."""
    cfg = cfg if cfg is not None else load_config()
    machine = machine or active_machine(cfg)
    c_start, c_end = campaign_dates(cfg)
    return machine.processed_dir() / f"{name}_{c_start:%Y%m%d}_{c_end:%Y%m%d}.nc"


def load_product(
    name: str,
    start=None,
    end=None,
    *,
    rebuild: bool = False,
    machine: Optional[Machine] = None,
    log: Callable[[str], None] = print,
) -> xr.Dataset:
    """Product `name` from config.yaml for days [start, end] (inclusive), in memory.

    Parameters
    ----------
    name : product name in config.yaml, e.g. "cod_mfrsr_M1"
    start, end : first and last day (date, datetime or "YYYY-MM-DD"); default
        is the campaign window
    rebuild : recombine the daily files even if a combined file exists (use
        after downloading more days or changing the product's variable list)

    Missing float values are NaN. Integer QC fields that were missing in the
    source are decoded to NaN as well (xarray turns them into floats), which
    ``epcape.qc`` treats as "failed".
    """
    cfg = load_config()
    machine = machine or active_machine(cfg)
    c_start, c_end = campaign_dates(cfg)
    start = as_date(start) if start else c_start
    end = as_date(end) if end else c_end

    path = processed_path(name, machine, cfg)
    if rebuild or not path.is_file():
        log(f"{path.name} not found (or rebuild requested); combining the daily files once ...")
        combine_product(get_product(name, cfg), c_start, c_end, machine=machine, out=path, log=log)

    # Select [start 00:00, end+1 day 00:00) so `end` is inclusive for whole days.
    t0 = np.datetime64(dt.datetime.combine(start, dt.time()))
    t1 = np.datetime64(dt.datetime.combine(end + dt.timedelta(days=1), dt.time()))
    with xr.open_dataset(path) as ds:
        out = ds.sel(time=slice(t0, t1 - np.timedelta64(1, "ns"))).load()
    out.attrs["epcape_processed_file"] = str(path)
    return out


def source_file_attrs(name: str, machine: Optional[Machine] = None) -> dict:
    """Global attributes of the first local daily file of product `name`.

    The combined file drops per-file attributes such as ``input_datastreams``,
    which records which upstream datastreams a VAP used (for MFRSRCLDOD: which
    MWR product supplied the LWP). This reads them from one source file.
    Returns {} if no daily file is available on this machine."""
    import netCDF4

    from .arm_files import list_local

    cfg = load_config()
    machine = machine or active_machine(cfg)
    product = get_product(name, cfg)
    c_start, c_end = campaign_dates(cfg)
    for directory in (
        machine.archive_dir(product.datastream),
        machine.subset_dir(product.name),
        machine.full_dir(product.datastream),
    ):
        if directory is None:
            continue
        files = list_local(directory, c_start, c_end, product.datastream)
        if files:
            with netCDF4.Dataset(str(files[0])) as nc:
                attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
            attrs["_read_from"] = str(files[0])
            return attrs
    return {}
