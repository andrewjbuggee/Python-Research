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

import errno
import http.client
import json
import shutil
import ssl
import sys
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

# Socket timeout for bulk file downloads. This is a STALL timeout, not a cap on
# total transfer time: Python applies it to each individual socket read, so a
# large file that keeps making progress is never cut off, while a connection
# that goes silent raises TimeoutError after this long and drops into the retry
# loop in download_file(). Kept well below the 600 s used previously because no
# legitimate 1 MB chunk takes two minutes, and because the failure this guards
# against -- a laptop sleeping mid-download -- leaves a half-open TCP
# connection that never sends a RST. Without a timeout the read simply blocks
# forever, which is exactly what an overnight run was observed to do.
_STALL_TIMEOUT_S = 120.0

# Faults that mean "the network misbehaved", i.e. worth retrying. A long
# sequential job over thousands of files WILL hit these: the server closes a
# keep-alive connection, an SSL session is torn down mid-stream, a response is
# truncated. Enumerated deliberately rather than caught as a blanket OSError,
# because URLError, ConnectionError, TimeoutError and ssl.SSLError are all
# OSError subclasses and so is a full disk -- and retrying a full disk four
# times just buries the real cause under a misleading "failed after retries".
#   urllib.error.URLError      DNS / connect failures (HTTPError subclasses it)
#   http.client.HTTPException  IncompleteRead, BadStatusLine, RemoteDisconnected
#                              -- note these are NOT OSError subclasses
#   ConnectionError            BrokenPipeError, ConnectionResetError, aborts
#   TimeoutError               the _STALL_TIMEOUT_S read timeout above
#   ssl.SSLError               SSLEOFError and friends, mid-stream TLS aborts
_TRANSIENT_NETWORK_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
    ssl.SSLError,
)


def _warn_if_insufficient_space(out_dir: Path, n_missing: int) -> None:
    """Warn before a bulk download that looks unlikely to fit on disk.

    The per-file size is estimated from files already in `out_dir`, so the
    estimate self-calibrates per datastream (THERMOCLDPHASE files are ~70 MB,
    QCRAD ~0.5 MB) and costs no extra network calls. Stays silent when there is
    too little local data to measure. This only warns -- a partial download is
    still useful, and the caller may be deliberately filling the disk.
    """
    if n_missing <= 0:
        return
    local = [p for p in out_dir.glob("*") if p.suffix in (".nc", ".cdf")]
    if len(local) < 5:
        return
    sample = local[-50:]
    mean_bytes = sum(p.stat().st_size for p in sample) / len(sample)
    needed = mean_bytes * n_missing
    free = shutil.disk_usage(out_dir).free
    gib = 1024**3
    if needed <= free * 0.95:
        return
    affordable = int(free * 0.95 / mean_bytes) if mean_bytes else 0
    print(
        f"\nWARNING: not enough free disk space to finish this datastream.\n"
        f"  {n_missing} file(s) still to fetch x ~{mean_bytes / 1e6:.0f} MB "
        f"each = ~{needed / gib:.0f} GiB needed\n"
        f"  free space on this volume:            ~{free / gib:.0f} GiB\n"
        f"  the download will stop after roughly {affordable} more file(s), "
        f"when the disk fills.\n"
        f"  Consider --data-root to send this datastream to an external drive, "
        f"or narrow --start/--end.\n",
        file=sys.stderr,
    )


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
            with urllib.request.urlopen(url, timeout=_STALL_TIMEOUT_S) as resp, open(
                part_path, "wb"
            ) as fh:
                while True:
                    chunk = resp.read(_CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
            part_path.rename(local_path)
            return local_path
        except _TRANSIENT_NETWORK_ERRORS as err:
            last_err = err
            part_path.unlink(missing_ok=True)
            if isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500:
                break
            # Say so rather than stalling silently: a long run that hits a bad
            # patch of network should show why it slowed down, not look hung.
            if attempt + 1 < _MAX_RETRIES:
                print(
                    f"  retrying {filename} "
                    f"(attempt {attempt + 2}/{_MAX_RETRIES}): {err}",
                    file=sys.stderr,
                )
                time.sleep(_RETRY_BASE_DELAY_S * 2**attempt)
        except OSError as err:
            # Reached only for OSErrors that are NOT in the transient tuple
            # above, i.e. local filesystem faults rather than network ones.
            # Retrying cannot clear a full disk or a permissions problem, so
            # fail immediately and name the real cause.
            part_path.unlink(missing_ok=True)
            if err.errno == errno.ENOSPC:
                raise RuntimeError(
                    f"Out of disk space while writing {part_path}.\n"
                    f"  Free space, or re-run with --data-root pointing at a "
                    f"drive with room. Files already downloaded are complete "
                    f"and will be skipped when you resume."
                ) from err
            raise
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
    total_in_archive = 0
    for arm_name in arm_names:
        filenames = query_files(arm_name, start_date, end_date, creds)
        total_in_archive += len(filenames)
        if verbose:
            print(f"{arm_name}: {len(filenames)} file(s) in archive for "
                  f"{start_date}..{end_date}")
        out_dir = config.raw_dir_for(arm_name)
        if filenames:
            out_dir.mkdir(parents=True, exist_ok=True)
            _warn_if_insufficient_space(
                out_dir,
                sum(1 for f in filenames if not (out_dir / f).exists()),
            )
        for i, filename in enumerate(filenames, start=1):
            path = download_file(filename, out_dir, creds, overwrite=overwrite)
            local_paths.append(path)
            if verbose and (i % 25 == 0 or i == len(filenames)):
                print(f"  [{i}/{len(filenames)}] {filename}")

    # A spec with several names (KAZR eras, QCRAD c2/c1) legitimately gets 0
    # from the "wrong" era, so only shout when EVERY name came back empty.
    if total_in_archive == 0:
        _warn_no_files(key, arm_names, start_date, end_date)
    return local_paths


def _warn_no_files(
    key: str,
    arm_names: Sequence[str],
    start_date: str,
    end_date: str,
) -> None:
    """Loudly flag a datastream that the ARM Live API served zero files for.

    A zero-file result is NOT an HTTP error: the ARM Live query endpoint returns
    a normal "success" response with an empty file list both when (a) the date
    range is outside the product's coverage, and (b) the datastream is not served
    by the real-time web service at all -- notably PI-contributed products such
    as the Shupe-Turner cloud microphysics ("nsamicrobase2shupeturnC1.c1"), which
    per ARM's docs "have to go through the regular ordering process." Either way
    the loop downloads nothing, and the quiet "0 file(s)" line is easy to miss --
    so make it impossible to overlook here.
    """
    names = ", ".join(arm_names)
    print(
        f"\nWARNING: datastream key '{key}' ({names}) returned 0 files from the "
        f"ARM Live API for {start_date}..{end_date}; nothing was downloaded.\n"
        "  This is a 'success' response with an empty file list, not a crash. "
        "Likely causes:\n"
        "    - the date range is outside the product's archive coverage; or\n"
        "    - the datastream is not served by the ARM Live web service. Many\n"
        "      PI-contributed products (including the Shupe-Turner microphysics\n"
        "      used for cloud phase) must instead be ORDERED via ARM Data\n"
        "      Discovery (https://adc.arm.gov/discovery) or requested from\n"
        "      adc@arm.gov -- they cannot be fetched through this downloader.\n"
        "  Verify the datastream name and its coverage before re-running.",
        file=sys.stderr,
    )


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
