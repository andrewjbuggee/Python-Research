"""Download ARM data, resuming where a previous run stopped.

Two modes:
  sync_product    a named product from config.yaml: ARM's server extracts only
                  the product's variables from each file
  sync_datastream complete files of any datastream

On a machine whose config lists a mounted ARM archive containing the
datastream, nothing is downloaded; the files are read in place.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .arm_files import NETCDF_LOCK, list_local, looks_like_netcdf
from .armlive import ArmLiveClient, ArmLiveError, FileNotAvailable, citation
from .config import Machine, Product, active_machine
from .credentials import get_credentials

# Always carried along when present, so every subset file is self-describing.
SUPPORT_VARIABLES = ("base_time", "time_offset", "time", "lat", "lon", "alt")
MANIFEST = "manifest.json"


@dataclass
class SyncResult:
    directory: Path
    files: List[str]
    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    fallbacks: Dict[str, str] = field(default_factory=dict)  # subset locally, with the reason
    bytes_downloaded: int = 0
    variables: Optional[List[str]] = None
    full_file_bytes: Optional[int] = None  # size of one complete file, for comparison
    source: str = "armlive"                # or "archive"

    @property
    def ok(self) -> bool:
        return not self.failed


def netcdf_variables(path: Path) -> Dict[str, Tuple[str, ...]]:
    """Variable name -> dimension names, read from a file's header."""
    import netCDF4

    with NETCDF_LOCK, netCDF4.Dataset(str(path)) as nc:
        return {name: tuple(var.dimensions) for name, var in nc.variables.items()}


def match_variables(requested: Sequence[str], available: Dict[str, Tuple[str, ...]],
                    *, required: bool = True) -> List[str]:
    """The names in `available` that `requested` refers to.

    An exact match wins; otherwise a unique case-insensitive match is accepted,
    because ARM documentation tables sometimes capitalise names differently from
    the files (e.g. "Lwp" in a table for the variable "lwp"). With required=True a
    name with no match raises ValueError; with required=False it is skipped."""
    by_lower: Dict[str, List[str]] = {}
    for name in available:
        by_lower.setdefault(name.lower(), []).append(name)
    matched: List[str] = []
    missing: List[str] = []
    for name in requested:
        if name in available:
            matched.append(name)
        elif len(by_lower.get(name.lower(), [])) == 1:
            matched.append(by_lower[name.lower()][0])
        else:
            missing.append(name)
    if missing and required:
        raise ValueError(
            f"Not in this datastream: {', '.join(missing)}.\n"
            f"Available variables: {', '.join(sorted(available))}"
        )
    return matched


def resolve_variables(requested: Sequence[str], available: Dict[str, Tuple[str, ...]],
                      optional: Sequence[str] = ()) -> List[str]:
    """The requested variables plus, where they exist: `optional` variables,
    time/location support variables, each variable's qc_ companion, and
    coordinate variables. A requested variable that does not exist is an error;
    an optional one that does not exist is skipped."""
    names = match_variables(requested, available, required=True)
    names += [n for n in match_variables(optional, available, required=False) if n not in names]
    chosen: List[str] = []

    def add(name: str) -> None:
        if name in available and name not in chosen:
            chosen.append(name)

    for name in SUPPORT_VARIABLES:
        add(name)
    for name in names:
        add(name)
        add(f"qc_{name}")
        for dim in available[name]:
            add(dim)  # coordinate variable, e.g. range for backscatter
    return chosen


def sync_product(
    product: Product,
    start: dt.date,
    end: dt.date,
    *,
    machine: Optional[Machine] = None,
    client: Optional[ArmLiveClient] = None,
    workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> SyncResult:
    machine = machine or active_machine()
    in_place = _archive_result(machine, product.datastream, start, end, log)
    if in_place:
        return in_place

    client = client or ArmLiveClient(*get_credentials(), log=log)
    names = client.query(product.datastream, start, end)
    log(f"ARM Live lists {len(names)} {product.datastream} files dated {start} to {end}.")
    dest = machine.subset_dir(product.name)
    if not names or dry_run:
        return _dry_result(dest, names, log)

    # Read the variable list from one complete file before asking the server
    # for hundreds of subsets, so a misspelled variable fails once, up front.
    probe, full_bytes = _probe(client, machine, product.datastream, names, log)
    probe_vars = netcdf_variables(probe)
    variables = resolve_variables(product.variables, probe_vars, product.optional_variables)
    core = match_variables(product.variables, probe_vars)  # names as spelled in the files
    log(f"Variables requested from ARM's server: {', '.join(variables)}")

    manifest = _load_manifest(dest)
    previous = manifest.get("variables")
    if previous and previous != variables and not overwrite and list_local(dest, start, end):
        raise RuntimeError(
            f"{dest} holds files downloaded with a different variable list:\n  {previous}\n"
            f"Re-run with --overwrite to replace them, or give the new selection its own product name."
        )
    manifest.update(datastream=product.datastream, product=product.name, variables=variables)
    manifest["citation"] = citation(product.datastream, start, end) or manifest.get("citation")

    result = _download_all(client, names, dest, variables, overwrite, workers, manifest, start, end, log,
                           core=core, full_dir=machine.full_dir(product.datastream))
    result.full_file_bytes = full_bytes
    return result


def sync_datastream(
    datastream: str,
    start: dt.date,
    end: dt.date,
    *,
    machine: Optional[Machine] = None,
    client: Optional[ArmLiveClient] = None,
    workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> SyncResult:
    machine = machine or active_machine()
    in_place = _archive_result(machine, datastream, start, end, log)
    if in_place:
        return in_place

    client = client or ArmLiveClient(*get_credentials(), log=log)
    names = client.query(datastream, start, end)
    log(f"ARM Live lists {len(names)} {datastream} files dated {start} to {end}.")
    dest = machine.full_dir(datastream)
    if not names or dry_run:
        return _dry_result(dest, names, log)

    manifest = _load_manifest(dest)
    manifest.update(datastream=datastream, product=None, variables=None)
    manifest["citation"] = citation(datastream, start, end) or manifest.get("citation")
    return _download_all(client, names, dest, None, overwrite, workers, manifest, start, end, log)


# -- helpers ---------------------------------------------------------------
def _archive_result(machine, datastream, start, end, log) -> Optional[SyncResult]:
    archive = machine.archive_dir(datastream)
    if archive is None:
        return None
    files = [p.name for p in list_local(archive, start, end, datastream)]
    log(f"ARM archive found at {archive}: {len(files)} files are read in place; nothing to download.")
    return SyncResult(directory=archive, files=files, skipped=list(files), source="archive")


def _dry_result(dest: Path, names: List[str], log) -> SyncResult:
    present = [n for n in names if looks_like_netcdf(dest / n)]
    if names:
        log(f"  first: {names[0]}\n  last:  {names[-1]}")
        log(f"  {len(present)} of them are already in {dest}")
    else:
        log("Nothing to download. Check the datastream name and dates.")
    return SyncResult(directory=dest, files=names, skipped=present)


def _probe(client, machine, datastream, names, log) -> Tuple[Path, int]:
    """A complete copy of one file (downloading it if needed) to read the variable list."""
    full_dir = machine.full_dir(datastream)
    for name in names[:5]:
        path = full_dir / name
        if not looks_like_netcdf(path):
            log(f"Reading the variable list from one complete file ({name}) ...")
            try:
                client.download(name, path)
            except FileNotAvailable as exc:
                log(f"  {exc}; trying the next file")
                continue
        return path, path.stat().st_size
    raise ArmLiveError(f"Could not download any of the first files of {datastream} to read its variables.")


def _subset_locally(src: Path, dest: Path, variables: Sequence[str]) -> List[str]:
    """Copy the listed variables that exist in `src` into `dest`, values untouched.
    Returns the variables `src` does not have."""
    import netCDF4

    tmp = dest.with_name(dest.name + ".part")
    with NETCDF_LOCK, netCDF4.Dataset(str(src)) as s:
        s.set_auto_maskandscale(False)
        present = [v for v in variables if v in s.variables]
        with netCDF4.Dataset(str(tmp), "w", format=s.data_model) as d:
            d.set_auto_maskandscale(False)
            d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
            for dim in dict.fromkeys(dim for v in present for dim in s.variables[v].dimensions):
                sd = s.dimensions[dim]
                d.createDimension(dim, None if sd.isunlimited() else len(sd))
            for v in present:
                sv = s.variables[v]
                fill = sv.getncattr("_FillValue") if "_FillValue" in sv.ncattrs() else None
                dv = d.createVariable(v, sv.datatype, sv.dimensions, fill_value=fill)
                dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs() if k != "_FillValue"})
                dv[...] = sv[...]
    os.replace(tmp, dest)
    return [v for v in variables if v not in present]


def _lacking(path: Path, core: Sequence[str]) -> List[str]:
    """Essential variables absent from a downloaded subset: the product's own
    variables and a time axis. qc_ and location variables are optional."""
    present = netcdf_variables(path)
    lacking = [v for v in core if v not in present]
    if "time" not in present and "time_offset" not in present:
        lacking.append("time")
    return lacking


def _download_all(client, names, dest, variables, overwrite, workers, manifest, start, end, log,
                  core: Sequence[str] = (), full_dir: Optional[Path] = None) -> SyncResult:
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.part"):
        stale.unlink()
    files_record = manifest.setdefault("files", {})
    result = SyncResult(directory=dest, files=list(names), variables=variables)

    todo = []
    for name in names:
        path = dest / name
        if not overwrite and looks_like_netcdf(path):
            result.skipped.append(name)
            files_record.setdefault(name, {"bytes": path.stat().st_size})
        else:
            todo.append(name)
    if result.skipped:
        log(f"{len(result.skipped)} files are already here and will be skipped.")
    if not todo:
        _save_manifest(dest, manifest, start, end, result)
        return result

    def fetch(name: str, allow_fallback: bool = True) -> dict:
        path = dest / name
        if not variables:
            return client.download(name, path)
        try:
            info = client.download(name, path, variables=variables)
            lacking = _lacking(path, core)
            if not lacking:
                return info
            path.unlink()
            reason = f"the server's subset lacked {', '.join(lacking)}"
        except FileNotAvailable:
            raise
        except ArmLiveError as exc:
            reason = client.redact(exc)
            if reason.startswith(name + ": "):
                reason = reason[len(name) + 2:]
        if not allow_fallback or full_dir is None:
            raise ArmLiveError(
                f"{name}: ARM's server could not extract the variables ({reason}). "
                "If this persists, download complete files with --full."
            )
        # This file differs (e.g. a variable added mid-campaign): fetch it whole, subset here.
        full = full_dir / name
        nbytes = 0
        if not looks_like_netcdf(full):
            nbytes = client.download(name, full)["bytes"]
        missing = _subset_locally(full, path, variables)
        note = f"subset locally because {reason}"
        if missing:
            note += f"; file has no {', '.join(missing)}"
        return {"bytes": nbytes, "note": note}

    total, done = len(todo), 0

    def record(name: str, info: Optional[dict] = None, error: Optional[BaseException] = None) -> None:
        nonlocal done
        done += 1
        if error is None:
            result.downloaded.append(name)
            result.bytes_downloaded += info["bytes"]
            files_record[name] = {
                "bytes": info["bytes"],
                "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
            if info.get("note"):
                files_record[name]["note"] = info["note"]
                result.fallbacks[name] = info["note"]
                log(f"[{done:>4}/{total}] {name}  {info['bytes'] / 1e6:6.2f} MB  ({info['note']})")
            else:
                log(f"[{done:>4}/{total}] {name}  {info['bytes'] / 1e6:6.2f} MB")
        elif isinstance(error, FileNotAvailable):
            result.unavailable.append(name)
            log(f"[{done:>4}/{total}] {name}  NOT AVAILABLE through ARM Live")
        else:
            result.failed[name] = client.redact(error)
            log(f"[{done:>4}/{total}] {name}  FAILED: {client.redact(error)}")

    # The first file alone: a systematic problem (bad token, a broken service)
    # then stops the run once instead of repeating for every file. If it is the
    # file the variable list was read from, the server must be able to subset it,
    # so no fallback: that keeps a broken subset service from silently turning
    # into complete downloads of every file.
    try:
        record(todo[0], fetch(todo[0], allow_fallback=todo[0] != names[0]))
    except FileNotAvailable as exc:
        record(todo[0], error=exc)
    except Exception:
        _save_manifest(dest, manifest, start, end, result)
        raise

    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futures = {pool.submit(fetch, name): name for name in todo[1:]}
        for future in as_completed(futures):
            name = futures[future]
            try:
                record(name, future.result())
            except Exception as exc:  # recorded per file; the run continues
                record(name, error=exc)
            if done % 25 == 0:
                _save_manifest(dest, manifest, start, end, result, final=False)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        _save_manifest(dest, manifest, start, end, result)
        raise
    pool.shutdown(wait=True)
    _save_manifest(dest, manifest, start, end, result)
    return result


def _load_manifest(dest: Path) -> dict:
    path = dest / MANIFEST
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {}


def _save_manifest(dest: Path, manifest: dict, start, end, result: SyncResult, final: bool = True) -> None:
    if final:
        manifest.setdefault("runs", []).append({
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "start": str(start),
            "end": str(end),
            "listed": len(result.files),
            "downloaded": len(result.downloaded),
            "skipped": len(result.skipped),
            "unavailable": result.unavailable,
            "failed": sorted(result.failed),
        })
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
    os.replace(tmp, dest / MANIFEST)
