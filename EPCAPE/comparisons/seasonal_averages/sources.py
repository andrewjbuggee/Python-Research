"""Data access for the seasonal-averages check.

Three kinds of input:

1. ARM time-series products (RADFLUX, MWR, ARSCL, ceilometer, disdrometers):
   downloaded with ``EPCAPE.sync.sync_product`` and read through
   ``EPCAPE.products.load_product`` like every other EPCAPE analysis.

2. ARM radiosonde products whose files each hold ONE launch with scalar
   results (PBLHTSONDE: one PBL height per launch; SONDEPARAM: one LCL per
   launch). ``config.yaml`` marks them ``layout: per_launch``. Combining them
   along time would keep only the first launch's scalars, so
   ``read_per_launch`` opens the files one by one and returns one row per
   launch instead.

3. Russell-group products in the UC San Diego Library Digital Collections
   (AMS at Mt. Soledad, GCVI enhancement factors). ``download_ucsd_library``
   fetches the files listed under ``ucsd_library:`` in ``config.yaml``;
   ``load_ams_soledad`` and ``load_gcvi`` read them.

Everything machine-specific (the data folder, a mounted ARM archive) comes
from ``config.yaml`` through ``EPCAPE.config.active_machine``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

from EPCAPE.arm_files import list_local
from EPCAPE.config import Machine, active_machine, campaign_dates, get_product, load_config

UCSD_OBJECT_URL = "https://library.ucsd.edu/dc/object"
# Identify the script honestly; the library serves file and JSON endpoints to it.
USER_AGENT = "epcape-data-tools/1.0 (python-requests)"
UCSD_MANIFEST = "ucsd_manifest.json"


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------
def product_layout(name: str, cfg: Optional[dict] = None) -> str:
    """'per_launch' for radiosonde products holding one launch per file, else 'time_series'."""
    cfg = cfg if cfg is not None else load_config()
    entry = (cfg.get("products") or {}).get(name) or {}
    return str(entry.get("layout", "time_series"))


def product_files(name: str, machine: Optional[Machine] = None) -> List[Path]:
    """Daily (or per-launch) files of product `name` on this machine, sorted by time.

    Same directory choice as ``EPCAPE.combine.combine_product``: a mounted
    ARM archive, the server-side subsets, or complete downloads, whichever
    holds the most files for the campaign window."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    product = get_product(name, cfg)
    start, end = campaign_dates(cfg)
    candidates = [
        machine.archive_dir(product.datastream),
        machine.subset_dir(product.name),
        machine.full_dir(product.datastream),
    ]
    best: List[Path] = []
    for directory in candidates:
        if directory is None:
            continue
        files = list_local(directory, start, end, product.datastream)
        if len(files) > len(best):
            best = files
    return best


# ---------------------------------------------------------------------------
# per-launch radiosonde products
# ---------------------------------------------------------------------------
def read_per_launch(
    name: str,
    reducer: Callable[[xr.Dataset], Dict[str, float]],
    *,
    machine: Optional[Machine] = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """One row per radiosonde launch: ``reducer(ds)`` applied to each file.

    The index is the launch time (UTC): the first ``time`` value in the
    file. A file that cannot be read is skipped and counted, not fatal,
    because one corrupt launch should not stop a season's statistics.

    Parameters
    ----------
    name : product name in config.yaml with ``layout: per_launch``
    reducer : function Dataset -> {column: value}; it should apply any QC
        screening itself and return NaN for values that fail
    """
    files = product_files(name, machine)
    if not files:
        raise FileNotFoundError(
            f"No files for product {name!r}. Download them first:\n"
            f"  python comparisons/seasonal_averages/download_data.py --only {name}"
        )
    rows, index, failed = [], [], 0
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                launch = pd.Timestamp(ds["time"].values[0])
                row = reducer(ds)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failed += 1
            log(f"  skipped {path.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        index.append(launch)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="launch_time")).sort_index()
    log(f"{name}: {len(out)} launches read from {files[0].parent}" + (f" ({failed} skipped)" if failed else ""))
    return out


# ---------------------------------------------------------------------------
# UC San Diego Library Digital Collections
# ---------------------------------------------------------------------------
def ucsd_collections(cfg: Optional[dict] = None) -> Dict[str, dict]:
    """The ``ucsd_library:`` entries of config.yaml."""
    cfg = cfg if cfg is not None else load_config()
    return dict(cfg.get("ucsd_library") or {})


def ucsd_file_path(filename: str, machine: Optional[Machine] = None, cfg: Optional[dict] = None) -> Path:
    """Local path of a configured UCSD Library file, by its configured filename."""
    cfg = cfg if cfg is not None else load_config()
    machine = machine or active_machine(cfg)
    for entry in ucsd_collections(cfg).values():
        for f in entry.get("files") or []:
            if f["filename"] == filename:
                path = machine.processed_dir() / entry["folder"] / filename
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{path} is missing. Download it with\n"
                        "  python comparisons/seasonal_averages/download_data.py --skip-arm"
                    )
                return path
    raise KeyError(f"{filename!r} is not listed under ucsd_library: in config.yaml")


def _ucsd_sizes(session: requests.Session, object_id: str) -> Dict[int, int]:
    """Component number -> size in bytes of its primary file, from the object's JSON record.

    The record lists each component's files as JSON strings under
    ``component_<n>_files_tesim``; the primary file has id ``1.<ext>``
    (derivative thumbnails have ids 2.jpg, 3.jpg, ...)."""
    r = session.get(f"{UCSD_OBJECT_URL}/{object_id}.json", timeout=60)
    r.raise_for_status()
    record = r.json()
    sizes: Dict[int, int] = {}
    for key, value in record.items():
        parts = key.split("_")
        if len(parts) >= 3 and parts[0] == "component" and parts[1].isdigit() and key.endswith("_files_tesim"):
            for item in value:
                info = json.loads(item)
                if str(info.get("id", "")).startswith("1.") and str(info.get("size", "")).isdigit():
                    sizes[int(parts[1])] = int(info["size"])
    return sizes


def download_ucsd_library(
    *,
    machine: Optional[Machine] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> List[Path]:
    """Fetch every file listed under ``ucsd_library:`` in config.yaml.

    A file already on disk with the size the library publishes is skipped.
    Each transfer goes to ``<name>.part`` first and is renamed only after its
    size matches, so an interrupted run never leaves a truncated file behind.
    A ``ucsd_manifest.json`` in each folder records the source URL, DOI,
    size and download time of every file. Returns the local paths."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    paths: List[Path] = []
    for key, entry in ucsd_collections(cfg).items():
        object_id = entry["object"]
        folder = machine.processed_dir() / entry["folder"]
        log(f"UCSD Library {object_id} ({entry.get('title', key)}) -> {folder}")
        try:
            sizes = _ucsd_sizes(session, object_id)
        except (requests.RequestException, ValueError) as exc:
            log(f"  could not read the object record ({exc}); sizes will not be checked")
            sizes = {}
        manifest_path = folder / UCSD_MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        manifest.update(object=object_id, title=entry.get("title"), doi=entry.get("doi"))
        files_record = manifest.setdefault("files", {})
        for f in entry.get("files") or []:
            comp, ext, filename = int(f["component"]), f["ext"], f["filename"]
            url = f"{UCSD_OBJECT_URL}/{object_id}/_{comp}_1.{ext}"
            dest = folder / filename
            expected = sizes.get(comp)
            paths.append(dest)
            if dest.is_file() and not overwrite and (expected is None or dest.stat().st_size == expected):
                log(f"  {filename}: present ({dest.stat().st_size / 1e6:.1f} MB)")
                continue
            size_txt = f"{expected / 1e6:.1f} MB" if expected else "size unknown"
            if dry_run:
                log(f"  {filename}: would download {url} ({size_txt})")
                continue
            log(f"  {filename}: downloading {url} ({size_txt}) ...")
            folder.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            try:
                with session.get(url, stream=True, timeout=(30, 600)) as r:
                    r.raise_for_status()
                    nbytes = 0
                    with open(tmp, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            out.write(chunk)
                            nbytes += len(chunk)
                if expected is not None and nbytes != expected:
                    raise IOError(f"received {nbytes} bytes, the library lists {expected}")
                os.replace(tmp, dest)
            finally:
                if tmp.exists():
                    tmp.unlink()
            files_record[filename] = {
                "url": url,
                "bytes": nbytes,
                "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
            log(f"  {filename}: {nbytes / 1e6:.1f} MB")
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return paths


# ---------------------------------------------------------------------------
# readers for the UCSD files
# ---------------------------------------------------------------------------
AMS_SPECIES = ("organics", "sulfate", "nitrate", "ammonium", "chloride")


def load_ams_soledad(machine: Optional[Machine] = None) -> pd.DataFrame:
    """HR-ToF-AMS NR-PM1 mass concentrations at Mt. Soledad (ug m-3), UTC index.

    Per the object README: CE-corrected with a monthly collection efficiency,
    flagged data already removed, timestamps = START of the 2-min V-mode
    sample. No QC field is distributed with this file. Ambient (isokinetic
    inlet) and GCVI (cloud-residual) periods are both in the file and are
    not labelled; ``quantities.gcvi_ams_residuals`` separates them using
    the GCVI enhancement-factor record."""
    path = ucsd_file_path("AMS_CE_corrected_msptof.nc", machine)
    with xr.open_dataset(path) as ds:
        df = ds[list(AMS_SPECIES)].to_dataframe()
    df.index = pd.DatetimeIndex(df.index, name="time")
    df.columns = [f"{c}_ugm3" for c in df.columns]
    return df.sort_index()


def load_gcvi(machine: Optional[Machine] = None) -> pd.DataFrame:
    """GCVI (ground-based counterflow virtual impactor) record at Mt. Soledad, 1 s, UTC.

    Columns (units from the file's header block):
        EF       enhancement factor (corrected; Shingler et al. 2012, AMT 5, 1259, Eq. 2)
        cut_um   CVI cut size (um)
        cntflow_lpm, airspeed_ms
    The file holds only times the GCVI was logging; EF <= 0 or non-finite
    marks a non-sampling state (see ``quantities.gcvi_segments``)."""
    path = ucsd_file_path("bb2743661p_36_1_CVI_EF.csv", machine)
    df = pd.read_csv(path, comment="#")
    df.columns = [c.replace("msd__cvi_v1__", "") for c in df.columns]
    df["time"] = pd.to_datetime(df.pop("datetime"))
    df = df.set_index("time").sort_index()
    df = df.rename(
        columns={
            "EF_cnttemp_round": "EF",
            "cutsize": "cut_um",
            "cntflow_calc": "cntflow_lpm",
            "airspeed": "airspeed_ms",
        }
    )
    # The file has a few +inf values (division by a zero flow); treat them as missing.
    return df.replace([np.inf, -np.inf], np.nan)


def _datenum_to_datetime(datenum) -> pd.DatetimeIndex:
    """MATLAB datenum (days since year 0, so 719529 = 1970-01-01) -> UTC timestamps,
    rounded to the second (the files store hourly steps as decimal fractions)."""
    seconds = (np.asarray(datenum, dtype=float) - 719529.0) * 86400.0
    return pd.to_datetime(seconds, unit="s").round("s")


def load_low_cloud_periods(machine: Optional[Machine] = None) -> pd.Series:
    """Hourly EPCAPE "Low Cloud Period" flag at Scripps Pier (True/False), UTC.

    Object README: single-layer cloud with base and top < 3 km, coupled to the
    marine boundary layer (cloud base within 150 m of the LCL). Time in the
    file is a MATLAB datenum."""
    path = ucsd_file_path("bb2743661p_30_1_lowCloudPeriods.txt", machine)
    df = pd.read_csv(path, sep="\t")
    flag = df["LowCloudPeriod"].astype(float) == 1
    return pd.Series(flag.values, index=_datenum_to_datetime(df["time"]), name="low_cloud_period")
