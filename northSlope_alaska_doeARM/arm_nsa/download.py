"""Programmatic download of ARM data via the ARM Live Data Web Service.

ARM's recommended programmatic access path is the ARM Live REST API
(https://adc.arm.gov/armlive/):

    query:    https://adc.arm.gov/armlive/query?user=U:TOKEN&ds=DATASTREAM
                  &start=YYYY-MM-DD&end=YYYY-MM-DD&wt=json
    download: https://adc.arm.gov/armlive/saveData?user=U:TOKEN&file=FILENAME

This module implements both with only the Python standard library (urllib), so
the pipeline has no hard dependency on `act-atmos`. If you prefer ACT, the
equivalent one-liner is:

    import act
    act.discovery.download_arm_data(username, token, "nsaceilC1.b1",
                                    "2022-01-01", "2022-01-31")

and the files it downloads are interchangeable with the ones this module
fetches (same archive, same files).

Design notes
------------
* Downloads are per-file and atomic (write to <name>.part, then rename), so an
  interrupted job never leaves a truncated .nc file that would poison xarray.
* Existing files are skipped by default, making the download scripts safely
  re-runnable / resumable over long date ranges.
* Multi-name datastreams (KAZR era renames, QCRAD c2/c1) are handled by
  querying every name in the spec and downloading whatever the archive holds
  for the requested period.
* Transient HTTP failures are retried with exponential backoff.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import config
from .credentials import ArmCredentials, get_credentials

ARM_LIVE_BASE_URL = "https://adc.arm.gov/armlive"
_CHUNK_SIZE_BYTES = 1024 * 1024
_MAX_RETRIES = 4
_RETRY_BASE_DELAY_S = 5.0


def _http_get(url: str, timeout_s: float = 120.0) -> bytes:
    """GET a URL with retries; return the raw response body."""
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            # HTTP 4xx (bad credentials / bad request) will not fix itself;
            # bail out immediately instead of retrying.
            if isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500:
                break
            time.sleep(_RETRY_BASE_DELAY_S * 2**attempt)
    raise RuntimeError(f"ARM Live request failed after retries: {url}\n  -> {last_err}")


def query_files(
    datastream: str,
    start_date: str,
    end_date: str,
    credentials: Optional[ArmCredentials] = None,
) -> List[str]:
    """List archive filenames for one ARM datastream and date range.

    Parameters
    ----------
    datastream:
        Full ARM datastream name, e.g. "nsamwrret1liljclouC1.c2".
    start_date, end_date:
        Inclusive date strings, "YYYY-MM-DD".
    credentials:
        ARM Live credentials; looked up automatically when omitted.

    Returns
    -------
    Sorted list of filenames (e.g. "nsaceilC1.b1.20220101.000008.nc"). Empty
    when the archive has nothing for that datastream/date range -- which is
    normal for the "wrong" era of a renamed product like KAZR.
    """
    creds = credentials or get_credentials()
    params = urllib.parse.urlencode(
        {
            "user": creds.as_query_value(),
            "ds": datastream,
            "start": start_date,
            "end": end_date,
            "wt": "json",
        }
    )
    body = _http_get(f"{ARM_LIVE_BASE_URL}/query?{params}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(
            "ARM Live returned a non-JSON response, which usually means the "
            "username:token pair was rejected. Check your credentials "
            "(see arm_nsa/credentials.py docstring). Response began with: "
            f"{body[:200]!r}"
        )
    # The service replies {"status": "success", "files": [...]} but be tolerant
    # of a bare list, and of null when nothing matches.
    files = payload.get("files") if isinstance(payload, dict) else payload
    return sorted(files or [])


def download_file(
    filename: str,
    out_dir: Path,
    credentials: Optional[ArmCredentials] = None,
    overwrite: bool = False,
) -> Path:
    """Download one archive file into out_dir; returns the local path.

    Skips the download when the file already exists (unless overwrite=True).
    Writes are atomic via a .part temp file.
    """
    creds = credentials or get_credentials()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / filename
    if local_path.exists() and not overwrite:
        return local_path

    params = urllib.parse.urlencode(
        {"user": creds.as_query_value(), "file": filename}
    )
    url = f"{ARM_LIVE_BASE_URL}/saveData?{params}"

    part_path = local_path.with_suffix(local_path.suffix + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=600.0) as resp, open(
                part_path, "wb"
            ) as fh:
                while True:
                    chunk = resp.read(_CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
            part_path.rename(local_path)
            return local_path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            part_path.unlink(missing_ok=True)
            if isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500:
                break
            time.sleep(_RETRY_BASE_DELAY_S * 2**attempt)
    raise RuntimeError(
        f"Failed to download {filename} after retries.\n  -> {last_err}"
    )


def download_datastream(
    key: str,
    start_date: str,
    end_date: str,
    credentials: Optional[ArmCredentials] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """Download every file for one pipeline datastream key over a date range.

    Parameters
    ----------
    key:
        Pipeline key from config.DATASTREAMS ("sonde", "kazr", "mwr", "ceil",
        "met", "qcrad") OR a raw ARM datastream name (contains a ".").
    start_date, end_date:
        Inclusive "YYYY-MM-DD" strings.
    overwrite:
        Re-download files that already exist locally.

    Returns
    -------
    List of local file paths (downloaded now or already present).

    Notes
    -----
    Raw files land in data/raw/<arm_datastream_name>/, one directory per ARM
    datastream name, so the three KAZR eras stay separate on disk exactly as
    they are in the archive. Readers glob across all directories of a spec.
    """
    creds = credentials or get_credentials()
    if "." in key:  # raw ARM datastream name passed directly
        arm_names: Sequence[str] = (key,)
    else:
        arm_names = config.get_spec(key).datastreams

    local_paths: List[Path] = []
    for arm_name in arm_names:
        filenames = query_files(arm_name, start_date, end_date, creds)
        if verbose:
            print(f"{arm_name}: {len(filenames)} file(s) in archive for "
                  f"{start_date}..{end_date}")
        out_dir = config.raw_dir_for(arm_name)
        for i, filename in enumerate(filenames, start=1):
            path = download_file(filename, out_dir, creds, overwrite=overwrite)
            local_paths.append(path)
            if verbose and (i % 25 == 0 or i == len(filenames)):
                print(f"  [{i}/{len(filenames)}] {filename}")
    return local_paths


def local_files(key: str) -> List[Path]:
    """All locally downloaded files for a pipeline key, across product eras."""
    if "." in key:
        arm_names: Iterable[str] = (key,)
    else:
        arm_names = config.get_spec(key).datastreams
    paths: List[Path] = []
    for arm_name in arm_names:
        d = config.raw_dir_for(arm_name)
        if d.is_dir():
            # ARM ships .nc (newer) and .cdf (older) netCDF files.
            paths.extend(sorted(d.glob(f"{arm_name}.*.nc")))
            paths.extend(sorted(d.glob(f"{arm_name}.*.cdf")))
    return sorted(paths)
