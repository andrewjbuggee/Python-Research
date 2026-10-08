"""Data access for the seasonal-averages check.

Three kinds of input:

1. ARM time-series products (RADFLUX, MWR, ARSCL, ceilometer, disdrometers):
   downloaded with ``EPCAPE.download_data.sync.sync_product`` and read through
   ``EPCAPE.analysis_tools.products.load_product`` like every other EPCAPE analysis.

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
from ``config.yaml`` through ``EPCAPE.download_data.config.active_machine``.
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

from EPCAPE.download_data.arm_files import list_local
from EPCAPE.download_data.config import Machine, active_machine, campaign_dates, get_product, load_config

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

    Same directory choice as ``EPCAPE.download_data.combine.combine_product``: a mounted
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


AMS_AMBIENT_FILE = "AMS/AMS_ambient.nc"  # under <data folder>/processed/; from the Russell group (2026-10-07)


def load_ams_ambient(machine: Optional[Machine] = None) -> pd.DataFrame:
    """AMS_ambient.nc: HR-ToF-AMS NR-PM1 at Mt. Soledad, ambient-inlet samples only (ug m-3), UTC.

    The file says only "EPCAPE AMS measurements of size-resolved submicron aerosol
    composition". Compared with AMS_CE_corrected_msptof.nc (notebook Section 10d):
      * its time stamps are a subset of that file's: the GCVI (cloud-residual)
        periods are left out, so it holds ambient samples only;
      * its values equal that file's times a constant that is fixed within each
        month (0.4-0.9) and the same for every species. That constant is the monthly
        collection efficiency (CE), so this file is NOT CE-corrected; dividing by the
        CE gives the corrected values.
    Columns <species>_ugm3 for AMS_SPECIES. The size-resolved variables and the
    run-number and flow-rate coordinates are not read."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    path = machine.processed_dir() / AMS_AMBIENT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"AMS ambient file missing: {path}")
    with xr.open_dataset(path) as ds:
        df = pd.DataFrame({f"{s}_ugm3": ds[s].values for s in AMS_SPECIES},
                          index=pd.DatetimeIndex(ds["time"].values, name="time"))
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


# ---------------------------------------------------------------------------
# ARSCL cloud_source_flag, reduced to the code at the lowest cloud top
# ---------------------------------------------------------------------------
# cloud_source_flag is a (time, height) field: ~28 MB per day as ARM Live
# serves it (uncompressed), ~10 GB for the campaign, far too large to combine
# along time. reduce_arscl_top_source() reads the daily files one at a time,
# keeps only the 1-D "which instrument saw the lowest layer's top" code
# (quantities.arscl_cloud_top_source), and writes one small campaign file.
TOP_SOURCE_PRODUCT = "cloud_source_arscl_M1"


def top_source_path(machine: Optional[Machine] = None) -> Path:
    """Campaign file of the per-sample ARSCL cloud-top source code."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    start, end = campaign_dates(cfg)
    return machine.processed_dir() / f"cloud_arscl_topsource_M1_{start:%Y%m%d}_{end:%Y%m%d}.nc"


def reduce_arscl_top_source(
    *, machine: Optional[Machine] = None, delete_daily: bool = False, log: Callable[[str], None] = print
) -> Path:
    """Reduce every daily cloud_source_arscl_M1 file to the code at its lowest cloud top.

    Writes <data folder>/processed/cloud_arscl_topsource_M1_<start>_<end>.nc with
    ``cloud_top_source`` (int16 on time; -1 = no layer). With delete_daily=True
    the ~28 MB daily files are removed after the campaign file is written."""
    from EPCAPE.comparisons.seasonal_averages.quantities import ARSCL_SOURCE_CODES, arscl_cloud_top_source

    files = product_files(TOP_SOURCE_PRODUCT, machine)
    if not files:
        raise FileNotFoundError(
            f"No {TOP_SOURCE_PRODUCT} files. Download them with\n"
            f"  python comparisons/seasonal_averages/download_data.py --only {TOP_SOURCE_PRODUCT}"
        )
    pieces = []
    for i, path in enumerate(files, 1):
        with xr.open_dataset(path) as ds:
            pieces.append(arscl_cloud_top_source(ds).load())
        if i % 50 == 0:
            log(f"  reduced {i}/{len(files)} files")
    out = xr.concat(pieces, dim="time").sortby("time").to_dataset()
    out["cloud_top_source"].attrs = {
        "long_name": "ARSCL cloud_source_flag at the range gate holding the lowest layer top",
        "units": "1",
        "flag_values": np.array([-1] + list(ARSCL_SOURCE_CODES), dtype=np.int16),
        "flag_meanings": "no_layer " + " ".join(v.replace(",", "").replace(" ", "_") for v in ARSCL_SOURCE_CODES.values()),
        "source": f"{TOP_SOURCE_PRODUCT} (epcarsclkazr1kolliasM1.c1): cloud_source_flag, cloud_layer_top_height",
    }
    path = top_source_path(machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path, encoding={"cloud_top_source": {"zlib": True, "complevel": 4},
                                  "time": {"units": "seconds since 1970-01-01", "dtype": "float64"}})
    log(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {out.sizes['time']:,} samples)")
    if delete_daily:
        for f in files:
            f.unlink()
        log(f"Deleted {len(files)} daily {TOP_SOURCE_PRODUCT} files")
    return path


def load_arscl_top_source(machine: Optional[Machine] = None) -> xr.DataArray:
    """The per-sample ARSCL cloud-top source code (see reduce_arscl_top_source)."""
    path = top_source_path(machine)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Create it with\n"
            f"  python comparisons/seasonal_averages/download_data.py --only {TOP_SOURCE_PRODUCT}"
        )
    with xr.open_dataset(path) as ds:
        return ds["cloud_top_source"].load()


# ---------------------------------------------------------------------------
# FM-120 fog monitor (Mt. Soledad), daily CSV files from the PIs
# ---------------------------------------------------------------------------
# Column names in the files -> names used here (units in the name).
FM120_COLUMNS = {
    "T Ambient (C)": "t_ambient_c",  # temperature inside the instrument
    "PAS (m/s)": "pas_ms",  # measured pump air speed (applied value 12 m/s)
    "Number Conc (#/cm^3)": "nd_cm3",  # total droplet number concentration, 2-50 um
    "LWC (g/m^3)": "lwc_gm3",  # liquid water content
    "MVD (um)": "mvd_um",  # median volume diameter
    "ED (um)": "ed_um",  # equivalent (effective) diameter
    "CVI Cutoff Sum > 9µm": "nd_ge9um_cm3",  # sum of bins 7-30 (>= ~8.5-9 um, the GCVI cut size)
}

# Edges of the 30 FM-120 size bins (um), from readme_FM120.md ("Lower/Upper Bin Limit"):
# 1-um bins from 2 to 14 um, then 2-um bins to 50 um. binNN_cm3 counts droplets in
# [FM120_BIN_EDGES_UM[NN-1], FM120_BIN_EDGES_UM[NN]).
FM120_BIN_EDGES_UM = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                               32, 34, 36, 38, 40, 42, 44, 46, 48, 50], dtype=float)


def fm120_folder(machine: Optional[Machine] = None, cfg: Optional[dict] = None) -> Path:
    """Folder holding the daily FM-120 CSV files (config.yaml: fm120.folder under processed/)."""
    cfg = cfg if cfg is not None else load_config()
    machine = machine or active_machine(cfg)
    entry = cfg.get("fm120") or {}
    if "folder" not in entry:
        raise KeyError("config.yaml has no fm120: folder entry")
    return machine.processed_dir() / entry["folder"]


def load_fm120(machine: Optional[Machine] = None) -> pd.DataFrame:
    """All FM-120 5-min averages, one row per 5-min interval with data, UTC index.

    File layout (readme_FM120.md): one CSV per day named YYYY-MM-DD.csv, first
    column = interval start time (UTC), 5-min averages of 1-s data from 00:00 to
    23:55. Some files end with a row for 00:00 of the NEXT day, which the README
    says to ignore (that interval is in the next day's file), so only rows dated
    on the file's own day are kept. Rows exist only while the instrument ran.
    One blank LWC cell (2023-11-14) is read as missing, and files with no rows
    (2024-01-02.csv) are skipped. A column repeated within one file
    (2023-03-19) keeps its first copy. No QC field exists.

    Columns: FM120_COLUMNS (renamed with units) plus bin01_cm3 ... bin30_cm3
    (dN per size bin, # cm-3; bin limits in the README)."""
    folder = fm120_folder(machine)
    files = sorted(folder.glob("????-??-??.csv"))
    if not files:
        raise FileNotFoundError(f"No FM-120 daily files (YYYY-MM-DD.csv) in {folder}")
    parts = []
    for path in files:
        day = pd.Timestamp(path.stem)
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        # headers differ only in a trailing space on the bin names ("Bin 1 - 3 µm " in 122 files):
        # strip it so every file's columns line up
        raw.columns = raw.columns.str.strip()
        # a repeated column header (2023-03-19 has "CVI Cutoff Sum > 9µm" twice) comes in as
        # "<name>.1"; keep the first copy (the two differ by <= 0.01 cm-3)
        raw = raw[[c for c in raw.columns if not (c.endswith(".1") and c[:-2] in raw.columns)]]
        raw = raw.apply(pd.to_numeric, errors="coerce")  # blank cells -> NaN
        own_day = raw[(raw.index >= day) & (raw.index < day + pd.Timedelta(days=1))]
        if not own_day.empty:  # e.g. 2024-01-02.csv has a header only (and a different one)
            parts.append(own_day)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "time"
    # rename: the fixed columns, then "Bin k - X µm " -> bin{k:02d}_cm3
    names = dict(FM120_COLUMNS)
    for col in df.columns:
        if col.startswith("Bin "):
            names[col] = f"bin{int(col.split()[1]):02d}_cm3"
    return df.rename(columns=names)


def load_visibility_msd(machine: Optional[Machine] = None, *, separate: bool = False):
    """Mt. Soledad visibility (m), ~1-s samples, UTC index, as stored in the files.

    Files: config.yaml fm120.visibility_files, in <data folder>/processed/, each with
    columns "datetime (GMT)" and "visibility (m)". The files overlap in time; repeated
    time stamps keep their first value. Values are NOT cleaned here: the sensor
    reports at most 48,270 m (30 statute miles), and the record has a few spikes far
    above that and a few zeros, which quantities.visibility_on_intervals removes.
    The files do not say which instrument measured this; the "msd" prefix matches
    the Mt. Soledad GCVI record (msd__cvi_v1).

    separate=True returns a list with one Series per file (in config order, each
    sorted by time, nothing removed), as Kavin's code reads them (kavin.py)."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    names = (cfg.get("fm120") or {}).get("visibility_files") or []
    if not names:
        raise KeyError("config.yaml has no fm120: visibility_files list")
    parts = []
    for name in names:
        path = machine.processed_dir() / name
        if not path.is_file():
            raise FileNotFoundError(f"Visibility file missing: {path}")
        df = pd.read_csv(path, dtype={"visibility (m)": "float64"})
        df.index = pd.to_datetime(df["datetime (GMT)"], format="%Y-%m-%d %H:%M:%S")
        df.index.name = "time"
        # stable sort keeps any repeated time stamps in file order
        parts.append(df["visibility (m)"].rename("visibility_m").sort_index(kind="stable"))
    if separate:
        return parts
    vis = pd.concat(parts).sort_index()
    vis = vis[~vis.index.duplicated(keep="first")]
    vis.index.name = "time"
    return vis.rename("visibility_m")
