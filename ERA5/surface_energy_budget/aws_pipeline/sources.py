"""Where the ERA5 archive is read from: the AWS mirror, or NCAR's GLADE disk.

Both hold the SAME archive with the SAME directory and file naming --
``<group>/<YYYYMM>/<group>.<table>_<num>_<short>.ll025<grid>.<t0>_<t1>.nc`` --
so ``era5_s3`` needs only three primitives to work against either:

    list_month_entries(group, year, month) -> [{"name": key, "size": bytes}]
    open_h5(key)                           -> something h5py.File accepts
    read_ranges(key, [(offset, nbytes)])   -> [bytes]

:class:`S3Source` fetches byte ranges over HTTPS from ``s3://nsf-ncar-era5``
(anonymous, no account). :class:`GladeSource` reads the same bytes from NCAR's
``/glade/campaign/collections/rda/data/ds633.0`` with ``os.pread``. Everything
above this layer -- the chunk decode, the lazy xarray backend, the windowing,
the analysis adapter -- is identical, which is the point: the figure modules
cannot tell the difference, and neither can the on-disk chunk cache (it is
keyed by file BASENAME and size, so a cache built on the laptop is still valid
on Casper and vice versa).

Choosing a source
-----------------
``ERA5_SOURCE`` selects it: ``auto`` (default), ``glade`` or ``s3``/``aws``.
``auto`` uses GLADE when its root is present and readable -- true on Casper,
false on a laptop -- and S3 otherwise. So the same notebook, with the same
``storage="aws"``, reads over the network at home and off local disk on Casper
without edits. Every run prints which source it used; nothing is implicit.

``ERA5_GLADE_ROOT`` overrides the GLADE path if NCAR moves it. The candidates
below are tried in order, and :func:`describe_sources` reports what was found,
which is the first thing to run on a new machine.

Why not read GLADE with plain h5py?
-----------------------------------
h5py serialises every libhdf5 call -- including the gzip inflate of a chunk --
behind one global lock, so a 36-core Casper node would decompress on one core.
Reading the chunks directly and inflating them with ``zlib`` in a thread pool
(what ``era5_s3`` does) releases the GIL and scales. On the laptop that was
measured at 9.4 s against 78 s for the same eight flux initialisations; on
Casper it is the difference between one core and as many as you reserved.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# AWS mirror
# ----------------------------------------------------------------------------
BUCKET = "nsf-ncar-era5"
AWS_REGION = "us-west-2"

# fsspec block size and cache policy for the h5py METADATA reads only (the data
# path uses explicit byte ranges). Measured best on a laptop: 1 MiB "bytes".
FS_BLOCK_SIZE = 2 ** 20
FS_CACHE_TYPE = "bytes"

# ----------------------------------------------------------------------------
# NCAR GLADE
# ----------------------------------------------------------------------------
# NSF NCAR ERA5, 908 TB, 1940-01 .. present. The dataset was renamed when the
# Research Data Archive relaunched as the Geoscience Data Exchange (GDEX) in
# September 2026: **ds633.0 is now d633000**, and rda.ucar.edu 301-redirects to
# gdex.ucar.edu. The directory layout BELOW the root is unchanged, which is why
# only the root is in question here.
#
# Candidates in priority order; the first that exists and is readable wins, and
# ERA5_GLADE_ROOT overrides the lot. Several are listed because the collection
# has moved more than once (/gpfs/fs1 -> /glade/collections -> campaign -> gdex)
# and because a stale CISL page still documents the old rda path.
DSID = "d633000"          # formerly ds633.0
GLADE_ROOT_CANDIDATES = (
    "/gdex/data/d633000",                               # current, NCAR's own examples use this
    "/glade/campaign/collections/gdex/data/d633000",    # campaign-storage equivalent
    "/glade/campaign/collections/rda/data/ds633.0",     # pre-GDEX name, may still exist
    "/glade/collections/rda/data/ds633.0",              # older still
    "/gpfs/fs1/collections/rda/data/ds633.0",           # oldest
)


class ArchiveSource:
    """Interface the reader needs. Implementations below."""

    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def cache_tag(self) -> str:
        """Short id for the listing cache. Must differ when the KEYS differ."""
        return self.name

    def month_prefix(self, group: str, year: int, month: int) -> str:
        raise NotImplementedError

    def list_month_entries(self, group: str, year: int, month: int) -> list[dict]:
        """``[{"name": <key>, "size": <bytes>}, ...]``; empty when absent."""
        raise NotImplementedError

    def open_h5(self, key: str):
        """Anything ``h5py.File(..., "r")`` accepts: a path or a file object."""
        raise NotImplementedError

    def read_ranges(self, key: str, locs: list[tuple[int, int]]) -> list[bytes]:
        """``[(byte_offset, nbytes), ...]`` -> the bytes, in the same order."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------
# S3
# ----------------------------------------------------------------------------
class S3Source(ArchiveSource):
    """The public AWS mirror, read anonymously over HTTPS."""

    name = "s3"

    def __init__(self, bucket: str = BUCKET):
        self.bucket = bucket
        self._fs = None
        self._lock = threading.Lock()

    def filesystem(self):
        with self._lock:
            if self._fs is None:
                import fsspec

                self._fs = fsspec.filesystem("s3", anon=True, default_block_size=FS_BLOCK_SIZE)
            return self._fs

    def available(self) -> bool:
        try:
            import fsspec  # noqa: F401
            import s3fs  # noqa: F401
        except ImportError:
            return False
        return True

    def describe(self) -> str:
        return f"s3://{self.bucket} (NSF NCAR ERA5 on AWS Open Data, anonymous)"

    def month_prefix(self, group: str, year: int, month: int) -> str:
        return f"{self.bucket}/{group}/{year:04d}{month:02d}/"

    def list_month_entries(self, group: str, year: int, month: int) -> list[dict]:
        try:
            raw = self.filesystem().ls(self.month_prefix(group, year, month), detail=True)
        except FileNotFoundError:
            return []
        return [{"name": r["name"], "size": int(r.get("size") or 0)} for r in raw]

    def open_h5(self, key: str):
        return self.filesystem().open(key, block_size=FS_BLOCK_SIZE, cache_type=FS_CACHE_TYPE)

    def read_ranges(self, key: str, locs: list[tuple[int, int]]) -> list[bytes]:
        fs = self.filesystem()
        blobs = fs.cat_ranges([key] * len(locs), [a for a, _ in locs],
                              [a + n for a, n in locs], on_error="return")
        blobs = list(blobs)
        bad = [b for b in blobs if isinstance(b, BaseException)]
        if bad:
            raise bad[0]
        short = [i for i, (b, (_, n)) in enumerate(zip(blobs, locs)) if len(b) != n]
        if short:
            raise OSError(f"{len(short)} of {len(locs)} ranges came back the wrong length "
                          f"from {key}")
        return blobs


# ----------------------------------------------------------------------------
# GLADE
# ----------------------------------------------------------------------------
class GladeSource(ArchiveSource):
    """ERA5 (d633000, formerly ds633.0) on NCAR's GLADE filesystem.

    Byte ranges are served by ``os.pread``, which takes the offset as an
    argument rather than moving a shared file position, so one open descriptor
    per file is safe to use from every reader thread at once.
    """

    name = "glade"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self._fds: "OrderedDict[str, int]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_fds = 64

    #: A group directory that must exist under the root for it to BE the
    #: archive. An automount stub, an empty mount, or a GDEX root holding other
    #: datasets would otherwise shadow the real one and list nothing.
    MARKER = "e5.oper.an.sfc"

    def available(self) -> bool:
        try:
            return ((self.root / self.MARKER).is_dir()
                    and os.access(self.root / self.MARKER, os.R_OK))
        except OSError:
            return False

    def cache_tag(self) -> str:
        # Two different GLADE roots hold different keys, so they must not share
        # a listing cache.
        return "glade-" + hashlib.sha1(str(self.root).encode()).hexdigest()[:10]

    def describe(self) -> str:
        return f"{self.root} (NSF NCAR GDEX {DSID} ERA5, on GLADE)"

    def month_prefix(self, group: str, year: int, month: int) -> str:
        return str(self.root / group / f"{year:04d}{month:02d}")

    def list_month_entries(self, group: str, year: int, month: int) -> list[dict]:
        d = self.root / group / f"{year:04d}{month:02d}"
        try:
            with os.scandir(d) as it:
                return sorted(
                    ({"name": e.path, "size": int(e.stat().st_size)}
                     for e in it if e.is_file() and e.name.endswith(".nc")),
                    key=lambda r: r["name"],
                )
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return []

    def open_h5(self, key: str):
        return key                      # h5py opens a local path directly

    def _fd_locked(self, key: str) -> int:
        """Cached read-only descriptor for ``key``. Caller holds ``self._lock``."""
        fd = self._fds.get(key)
        if fd is not None:
            self._fds.move_to_end(key)
            return fd
        while len(self._fds) >= self._max_fds:
            _, old = self._fds.popitem(last=False)
            try:
                os.close(old)
            except OSError:
                pass
        fd = os.open(key, os.O_RDONLY)
        self._fds[key] = fd
        return fd

    def read_ranges(self, key: str, locs: list[tuple[int, int]]) -> list[bytes]:
        # Duplicate the cached descriptor under the lock and read through the
        # copy. Without this, another thread evicting this key from the LRU
        # would close the fd mid-read; the number is then immediately reused by
        # the next os.open, and the read would silently return another FILE's
        # bytes. A dup costs ~1 us against a ~100 ms chunk read.
        with self._lock:
            fd = os.dup(self._fd_locked(key))
        try:
            out = []
            for offset, nbytes in locs:
                buf = os.pread(fd, nbytes, offset)
                if len(buf) != nbytes:
                    # Short read: a truncated or concurrently replaced file.
                    raise OSError(f"{key}: read {len(buf)} of {nbytes} bytes at offset {offset}")
                out.append(buf)
            return out
        finally:
            os.close(fd)

    def close(self) -> None:
        with self._lock:
            while self._fds:
                _, fd = self._fds.popitem()
                try:
                    os.close(fd)
                except OSError:
                    pass


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
_SOURCE: ArchiveSource | None = None
_SOURCE_LOCK = threading.Lock()


def glade_root(strict_env: bool = True) -> Path | None:
    """The configured or first existing GLADE root, or None if there is none.

    ``ERA5_GLADE_ROOT`` is AUTHORITATIVE: if it is set but does not hold the
    archive, this raises rather than quietly falling through to a built-in
    candidate. Falling through is how a job ends up reading the frozen
    pre-GDEX copy and reporting numbers from the wrong archive.
    """
    env = os.environ.get("ERA5_GLADE_ROOT")
    if env:
        if GladeSource(env).available():
            return Path(env)
        if strict_env:
            raise FileNotFoundError(
                f"ERA5_GLADE_ROOT={env!r} does not hold the ERA5 archive "
                f"(no {GladeSource.MARKER}/ under it). Unset it to search the "
                f"built-in candidates, or point it at the real root.")
        return None
    for c in GLADE_ROOT_CANDIDATES:
        if GladeSource(c).available():
            return Path(c)
    return None


def resolve_source(spec: str | None = None) -> ArchiveSource:
    """Build the source named by ``spec`` (or ``ERA5_SOURCE``, or auto)."""
    spec = (spec or os.environ.get("ERA5_SOURCE", "auto")).lower()
    if spec in ("glade", "ncar", "casper"):
        root = glade_root()
        if root is None:
            raise FileNotFoundError(
                "ERA5_SOURCE=glade but no ds633.0 archive was found. Tried "
                + ", ".join(((os.environ["ERA5_GLADE_ROOT"],) if os.environ.get("ERA5_GLADE_ROOT")
                             else ()) + GLADE_ROOT_CANDIDATES)
                + ". Set ERA5_GLADE_ROOT to the real path (on Casper: "
                  "`ls -d /gdex/data/d633000`).")
        return GladeSource(root)
    if spec in ("s3", "aws"):
        return S3Source()
    if spec != "auto":
        raise ValueError(f"ERA5_SOURCE must be auto, glade or s3; got {spec!r}")
    root = glade_root()
    return GladeSource(root) if root is not None else S3Source()


def source() -> ArchiveSource:
    """The process-wide source, resolved once on first use."""
    global _SOURCE
    with _SOURCE_LOCK:
        if _SOURCE is None:
            _SOURCE = resolve_source()
        return _SOURCE


def set_source(spec: str | ArchiveSource | None) -> ArchiveSource:
    """Pin the source for this process. ``None`` re-resolves from the env."""
    global _SOURCE
    with _SOURCE_LOCK:
        if _SOURCE is not None:
            _SOURCE.close()
        _SOURCE = spec if isinstance(spec, ArchiveSource) else (
            None if spec is None else resolve_source(spec))
        if _SOURCE is None:
            _SOURCE = resolve_source()
        return _SOURCE


def describe_sources() -> str:
    """What is reachable from this machine -- run this first on a new system.

    Never raises: this is the function you call precisely when the archive
    cannot be found, so it reports the failure as a line instead.
    """
    lines = [f"ERA5_SOURCE={os.environ.get('ERA5_SOURCE', 'auto')}"]
    env = os.environ.get("ERA5_GLADE_ROOT")
    if env:
        lines.append(f"ERA5_GLADE_ROOT={env}")
    for c in ((env,) if env else ()) + GLADE_ROOT_CANDIDATES:
        ok = GladeSource(c).available()
        lines.append(f"  {'FOUND  ' if ok else 'absent '} {c}"
                     + ("" if ok else f"   (no {GladeSource.MARKER}/ under it)"))
    s3 = S3Source()
    lines.append(f"  {'ready  ' if s3.available() else 'no s3fs'} s3://{BUCKET}")
    try:
        lines.append(f"  -> using: {source().describe()}")
    except Exception as exc:  # noqa: BLE001 - reporting the failure IS the job
        lines.append(f"  -> NO SOURCE: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


__all__ = [
    "ArchiveSource", "S3Source", "GladeSource", "BUCKET", "AWS_REGION",
    "GLADE_ROOT_CANDIDATES", "DSID", "glade_root", "resolve_source", "source",
    "set_source", "describe_sources", "FS_BLOCK_SIZE", "FS_CACHE_TYPE",
]
