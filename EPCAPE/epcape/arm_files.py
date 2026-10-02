"""ARM file names and local file discovery.

ARM names files <datastream>.<YYYYMMDD>.<hhmmss>.<ext>, e.g.
epcceilM1.b1.20230215.000002.nc, where the date/time is the first sample.
"""
from __future__ import annotations

import datetime as dt
import re
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# The netCDF-C and HDF5 libraries are not thread-safe; every direct netCDF4 call
# made from download threads must hold this lock.
NETCDF_LOCK = threading.RLock()

_STAMP = re.compile(r"\.(\d{8})\.(\d{6})\.")
NETCDF_EXTENSIONS = (".nc", ".cdf")
# netCDF-3 classic / 64-bit offset / 64-bit data, and HDF5 (netCDF-4)
NETCDF_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")


def file_datetime(name) -> Optional[dt.datetime]:
    m = _STAMP.search(Path(str(name)).name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def file_date(name) -> Optional[dt.date]:
    stamp = file_datetime(name)
    return stamp.date() if stamp else None


def in_window(name, start: dt.date, end: dt.date) -> bool:
    """True if the file's date falls in [start, end]. Undated names are kept."""
    d = file_date(name)
    return d is None or start <= d <= end


def looks_like_netcdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8).startswith(NETCDF_MAGIC)
    except OSError:
        return False


def list_local(directory: Path, start: dt.date, end: dt.date, datastream: Optional[str] = None) -> List[Path]:
    """netCDF files in `directory` whose date is within [start, end], sorted by time."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix in NETCDF_EXTENSIONS
        and (datastream is None or p.name.startswith(datastream + "."))
        and in_window(p.name, start, end)
    ]
    return sorted(files, key=lambda p: (file_datetime(p.name) or dt.datetime.min, p.name))


def missing_days(dates: Iterable[dt.date], start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    """Runs of consecutive days in [start, end] that have no data, as (first, last) pairs."""
    have = set(dates)
    runs: List[Tuple[dt.date, dt.date]] = []
    day = start
    while day <= end:
        if day not in have:
            first = day
            while day + dt.timedelta(days=1) <= end and (day + dt.timedelta(days=1)) not in have:
                day += dt.timedelta(days=1)
            runs.append((first, day))
        day += dt.timedelta(days=1)
    return runs


def format_runs(runs: Sequence[Tuple[dt.date, dt.date]], limit: int = 12) -> str:
    parts = [str(a) if a == b else f"{a} to {b}" for a, b in runs[:limit]]
    if len(runs) > limit:
        parts.append(f"... ({len(runs) - limit} more)")
    return ", ".join(parts)
