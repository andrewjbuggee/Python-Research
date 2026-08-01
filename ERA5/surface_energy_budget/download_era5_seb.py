#!/usr/bin/env python3
"""Download hourly ERA5 single-level data for an Arctic surface energy budget (SEB)
analysis following Equation (1) of Sledd et al. (2025).

    Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy
    balance responses to radiative forcing in the central Arctic from MOSAiC and
    models. JGR Atmospheres, 130, e2024JD042578.
    https://doi.org/10.1029/2024JD042578

Data source
-----------
CDS dataset ``reanalysis-era5-single-levels`` (hourly, 0.25 deg native grid).
Requires the ``cdsapi`` package and a ``~/.cdsapirc`` holding

    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>

Note the current CDS endpoint has no ``/v2`` suffix and the key is the bare
token, not the old ``UID:KEY`` pair.

Examples
--------
Estimate the download size without contacting the CDS::

    python download_era5_seb.py --dry-run

Small smoke test (Barrow strip, one day, to the local disk)::

    python download_era5_seb.py --region barrow --start 2026-01-01 --end 2026-01-01

The default run: first week of January 2026, north of the Arctic Circle,
recommended variable set, written to the external drive::

    python download_era5_seb.py --storage external

Ad hoc box, given inline as North West South East::

    python download_era5_seb.py --region custom --area 82 -60 72 -10
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from era5_seb_variables import (
    VARIABLE_SETS,
    Region,
    cds_variable_list,
    get_region,
    normalise_names,
    region_names,
)

# ----------------------------------------------------------------------------
# Storage roots -- EDIT THESE TO TASTE
# ----------------------------------------------------------------------------
# External drive. The volume is case-insensitive, so "Scripps" also resolves;
# the on-disk casing is used here to match the sibling SCRIPPS/DOE_ARM tree.
EXTERNAL_ROOT = Path("/Volumes/My Passport/SCRIPPS/ERA5/surface_energy_budget")

# Local machine. Sits beside this script. The repository .gitignore already
# excludes *.nc, so downloads here will not be committed.
LOCAL_ROOT = Path(__file__).resolve().parent / "data"

STORAGE_ROOTS: dict[str, Path] = {"external": EXTERNAL_ROOT, "local": LOCAL_ROOT}

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DATASET = "reanalysis-era5-single-levels"
DEFAULT_START = date(2026, 1, 1)
DEFAULT_END = date(2026, 1, 7)  # inclusive -- first week of January 2026
DEFAULT_REGION = "arctic_circle"
DEFAULT_VAR_SET = "recommended"

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
NATIVE_GRID_DEG = 0.25


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def daterange(start_date: date, end_date: date) -> list[date]:
    """Every calendar day from ``start_date`` to ``end_date``, inclusive."""
    if end_date < start_date:
        raise ValueError(f"end date {end_date} precedes start date {start_date}")
    n_days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(n_days)]


def chunk_by_month(days: list[date]) -> list[tuple[int, int, list[date]]]:
    """Group a day list into ``(year, month, days_in_that_month)`` tuples."""
    grouped: dict[tuple[int, int], list[date]] = {}
    for d in days:
        grouped.setdefault((d.year, d.month), []).append(d)
    return [(y, m, ds) for (y, m), ds in sorted(grouped.items())]


def estimate_size_gb(
    region: Region, n_hours: int, n_vars: int, grid_deg: float = NATIVE_GRID_DEG
) -> float:
    """Uncompressed upper bound on the download size, in GB.

    Assumes float32 storage on a regular lat/lon grid. Real netCDF output is
    typically 2-4x smaller because CDS applies compression, so treat this as a
    ceiling rather than an expectation.
    """
    n_lat = round((region.north_deg - region.south_deg) / grid_deg) + 1
    spans_globe = region.west_deg <= -180.0 and region.east_deg >= 180.0
    if spans_globe:
        n_lon = round(360.0 / grid_deg)  # wraps, so no +1
    else:
        n_lon = round((region.east_deg - region.west_deg) / grid_deg) + 1
    n_bytes = n_lat * n_lon * n_hours * n_vars * 4
    return n_bytes / 1024**3


def check_volume_mounted(root: Path) -> None:
    """Raise if ``root`` sits on a /Volumes mount that is not currently mounted.

    Guards against silently creating a directory tree on the boot drive at the
    external path when the drive is unplugged, which would then fill it.
    """
    parts = root.parts
    if len(parts) < 3 or parts[1] != "Volumes":
        return  # not an external volume path; nothing to check
    volume = Path(parts[0], parts[1], parts[2])
    if not volume.is_dir():
        raise FileNotFoundError(
            f"External volume {volume} is not mounted. Connect the drive, rerun "
            f"with --storage local, or pass --out-dir explicitly."
        )


def resolve_output_root(storage: str, out_dir: str | None) -> Path:
    """Pick the output directory. Does not create it."""
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()
    return STORAGE_ROOTS[storage]


def verify_credentials() -> bool:
    """Check that ``~/.cdsapirc`` exists and points at the current CDS endpoint."""
    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.exists():
        print(f"  [x] No CDS credentials at {rc_path}")
        print("      Register at https://cds.climate.copernicus.eu/, then create")
        print("      ~/.cdsapirc containing:")
        print("        url: https://cds.climate.copernicus.eu/api")
        print("        key: <your personal access token>")
        return False

    text = rc_path.read_text()
    print(f"  [ok] Found CDS credentials at {rc_path}")
    if "/api/v2" in text:
        print("  [!] Warning: that file still uses the retired /api/v2 endpoint.")
        print("      Change the url line to https://cds.climate.copernicus.eu/api")
    return True


# ----------------------------------------------------------------------------
# Request construction and download
# ----------------------------------------------------------------------------
def build_request(
    variables: list[str],
    year: int,
    month: int,
    days: list[date],
    area: list[float],
    grid_deg: float | None = None,
) -> dict:
    """Assemble one CDS request covering the given days within a single month."""
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d.day:02d}" for d in days],
        "time": ALL_HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    if grid_deg is not None:
        request["grid"] = [grid_deg, grid_deg]
    return request


def consolidate_to_netcdf(raw_path: Path, final_path: Path) -> None:
    """Turn whatever the CDS actually returned into one tidy netCDF file.

    The CDS splits a request into separate netCDF files by GRIB ``stepType`` and
    returns them zipped whenever a request mixes types, regardless of
    ``download_format: unarchived``. Our variable list always mixes instantaneous
    fields (skin temperature, cloud cover), time-mean fluxes, and accumulations
    (total precipitation), so a zip of three files is the normal result.

    This merges those streams onto their shared (valid_time, latitude, longitude)
    grid and renames the CDS ``avg_*`` fields to canonical ERA5 short names, so
    downstream code sees one file with the names the registry advertises. A plain
    netCDF response is passed through the same normalisation.
    """
    import xarray as xr

    if not zipfile.is_zipfile(raw_path):
        ds = xr.open_dataset(raw_path)
        try:
            out = normalise_names(ds).load()
        finally:
            ds.close()
        out.to_netcdf(final_path, encoding=_compression_encoding(out))
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(raw_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not members:
                raise RuntimeError(f"{raw_path.name} contains no .nc members")
            zf.extractall(tmp_dir, members)

        opened = [xr.open_dataset(Path(tmp_dir) / m) for m in members]
        try:
            # join="exact" so a coordinate mismatch between streams fails loudly
            # rather than silently broadcasting into a larger array.
            # compat is pinned rather than left default: the streams carry
            # disjoint data variables but shared coordinates, and "no_conflicts"
            # verifies those coordinates agree instead of letting one win.
            merged = xr.merge(
                [normalise_names(d) for d in opened],
                join="exact",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            ).load()
        finally:
            for d in opened:
                d.close()

        merged.attrs["cds_stream_files"] = ", ".join(sorted(members))
        merged.to_netcdf(final_path, encoding=_compression_encoding(merged))


def _compression_encoding(ds, complevel: int = 4) -> dict:
    """zlib encoding for every float field, to keep the merged file compact."""
    return {
        name: {"zlib": True, "complevel": complevel}
        for name, da in ds.data_vars.items()
        if da.dtype.kind == "f"
    }


def download_one(
    client,
    request: dict,
    output_path: Path,
    max_retries: int = 4,
) -> tuple[bool, float]:
    """Retrieve one CDS request with exponential backoff, then consolidate it.

    Returns ``(success, size_mb)``. The raw CDS payload lands on a scratch path
    and the merged netCDF is only moved into place once it is complete, so an
    interrupted run never leaves a partial file that a later resume would mistake
    for a finished one.
    """
    raw_path = output_path.with_suffix(".raw.part")
    merged_path = output_path.with_suffix(".merged.part")

    def _cleanup() -> None:
        raw_path.unlink(missing_ok=True)
        merged_path.unlink(missing_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(DATASET, request).download(str(raw_path))
            consolidate_to_netcdf(raw_path, merged_path)
            merged_path.replace(output_path)
            raw_path.unlink(missing_ok=True)
            return True, output_path.stat().st_size / 1024**2
        except KeyboardInterrupt:
            _cleanup()
            raise
        except Exception as exc:  # cdsapi raises bare Exception on HTTP errors
            print(f"      attempt {attempt}/{max_retries} failed: {exc}")
            _cleanup()
            if attempt < max_retries:
                wait_s = 30 * 2 ** (attempt - 1)
                print(f"      retrying in {wait_s} s ...")
                time.sleep(wait_s)
            else:
                print("      giving up on this chunk")
                return False, 0.0
    return False, 0.0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage",
        choices=sorted(STORAGE_ROOTS),
        default="local",
        help="Write to the external drive or the local machine (default: local).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Explicit output directory, overriding --storage.",
    )
    parser.add_argument(
        "--region",
        choices=region_names(),
        default=DEFAULT_REGION,
        help=f"Named spatial region (default: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=None,
        help="Explicit lat/lon box in degrees, overriding --region.",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_START,
        help=f"First day, YYYY-MM-DD (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_END,
        help=f"Last day, inclusive, YYYY-MM-DD (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--var-set",
        choices=sorted(VARIABLE_SETS),
        default=DEFAULT_VAR_SET,
        help=(
            "core = the 17 SEB/cloud variables; recommended = core plus the "
            "terms needed to close Eq. (1) and mask to sea ice; extended = "
            f"recommended plus diagnostics (default: {DEFAULT_VAR_SET})."
        ),
    )
    parser.add_argument(
        "--chunk",
        choices=("day", "month"),
        default="day",
        help="One netCDF file per day or per month (default: day).",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=None,
        help="Regrid to this resolution in degrees (default: native 0.25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download chunks whose output file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and size estimate without contacting the CDS.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Resolve the spatial domain -----------------------------------------
    if args.area is not None:
        north_deg, west_deg, south_deg, east_deg = args.area
        if north_deg <= south_deg:
            raise SystemExit(
                f"--area NORTH ({north_deg}) must exceed SOUTH ({south_deg})."
            )
        region = Region(north_deg, west_deg, south_deg, east_deg, "explicit --area")
        region_label = "custom_area"
    else:
        region = get_region(args.region)
        region_label = args.region

    variables = cds_variable_list(args.var_set)
    days = daterange(args.start, args.end)
    n_hours = 24 * len(days)

    # --- Plan ----------------------------------------------------------------
    print("=" * 72)
    print("ERA5 surface energy budget download")
    print("=" * 72)
    print(f"  Dataset      : {DATASET}")
    print(f"  Region       : {region_label} -- {region.description}")
    print(f"  Area [N W S E]: {region.as_area()}")
    print(f"  Period       : {args.start} to {args.end} ({len(days)} days, hourly)")
    print(f"  Variable set : {args.var_set} ({len(variables)} variables)")
    print(f"  Grid         : {args.grid or NATIVE_GRID_DEG} deg")
    print(f"  Chunking     : one file per {args.chunk}")
    est_gb = estimate_size_gb(region, n_hours, len(variables), args.grid or NATIVE_GRID_DEG)
    print(f"  Size estimate: ~{est_gb:.2f} GB uncompressed (netCDF will be smaller)")

    output_root = resolve_output_root(args.storage, args.out_dir)
    run_dir = output_root / region_label
    print(f"  Output       : {run_dir}")
    print("-" * 72)

    # --- Build the chunk list ------------------------------------------------
    if args.chunk == "day":
        chunks = [
            (
                f"era5_seb_{region_label}_{d:%Y%m%d}.nc",
                d.year,
                d.month,
                [d],
            )
            for d in days
        ]
    else:
        chunks = [
            (
                f"era5_seb_{region_label}_{y:04d}{m:02d}"
                f"_{ds[0].day:02d}-{ds[-1].day:02d}.nc",
                y,
                m,
                ds,
            )
            for y, m, ds in chunk_by_month(days)
        ]

    if args.dry_run:
        print("Dry run -- the following chunks would be requested:")
        for filename, year, month, chunk_days in chunks:
            day_span = f"{chunk_days[0].day:02d}-{chunk_days[-1].day:02d}"
            print(f"  {filename}  ({year}-{month:02d} days {day_span}, 24 h each)")
        print()
        print("Variables:")
        for v in VARIABLE_SETS[args.var_set]:
            print(f"  {v.short_name:<11} {v.units:<16} {v.cds_name}  [{v.role}]")
        print()
        try:
            check_volume_mounted(run_dir)
        except FileNotFoundError as exc:
            print(f"Note: {exc}")
        print("No data was downloaded. Drop --dry-run to execute.")
        return 0

    # --- Credentials and client ---------------------------------------------
    check_volume_mounted(run_dir)
    print("Checking CDS credentials ...")
    if not verify_credentials():
        return 1
    try:
        import cdsapi
    except ImportError:
        print("  [x] cdsapi is not installed. Run: pip install cdsapi")
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    print()

    # --- Download loop -------------------------------------------------------
    manifest_entries: list[dict] = []
    n_done = n_skipped = n_failed = 0
    total_mb = 0.0
    t_start = time.monotonic()

    for i, (filename, year, month, chunk_days) in enumerate(chunks, start=1):
        output_path = run_dir / filename
        day_span = f"{chunk_days[0].day:02d}-{chunk_days[-1].day:02d}"
        print(f"[{i}/{len(chunks)}] {filename}  ({year}-{month:02d} day {day_span})")

        if output_path.exists() and not args.overwrite:
            size_mb = output_path.stat().st_size / 1024**2
            print(f"      already present ({size_mb:.1f} MB) -- skipping")
            n_skipped += 1
            manifest_entries.append(
                {"file": filename, "status": "skipped_existing", "size_mb": round(size_mb, 2)}
            )
            continue

        request = build_request(
            variables, year, month, chunk_days, region.as_area(), args.grid
        )
        ok, size_mb = download_one(client, request, output_path)

        if ok:
            print(f"      done ({size_mb:.1f} MB)")
            n_done += 1
            total_mb += size_mb
            manifest_entries.append(
                {"file": filename, "status": "downloaded", "size_mb": round(size_mb, 2)}
            )
        else:
            n_failed += 1
            manifest_entries.append({"file": filename, "status": "failed", "size_mb": 0.0})

    elapsed_min = (time.monotonic() - t_start) / 60.0

    # --- Manifest ------------------------------------------------------------
    # Records exactly what was requested so a run can be reproduced or audited
    # later without re-deriving the arguments.
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "region": region_label,
        "region_description": region.description,
        "area_north_west_south_east_deg": region.as_area(),
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "hours_utc": ALL_HOURS,
        "variable_set": args.var_set,
        "variables": [
            {"cds_name": v.cds_name, "short_name": v.short_name, "units": v.units, "role": v.role}
            for v in VARIABLE_SETS[args.var_set]
        ],
        "grid_deg": args.grid or NATIVE_GRID_DEG,
        "chunking": args.chunk,
        "sign_convention": (
            "ERA5 surface fluxes are positive DOWNWARD. Sledd et al. (2025) Eq. (1) "
            "defines SH and LH positive UPWARD, so SH = -msshf and LH = -mslhf. "
            "See seb_terms.py."
        ),
        "reference": "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578",
        "files": manifest_entries,
    }
    manifest_path = run_dir / f"manifest_{args.start:%Y%m%d}_{args.end:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- Summary -------------------------------------------------------------
    print()
    print("=" * 72)
    print(
        f"Downloaded {n_done}, skipped {n_skipped}, failed {n_failed} "
        f"of {len(chunks)} chunks in {elapsed_min:.1f} min"
    )
    print(f"Total new data: {total_mb / 1024:.2f} GB")
    print(f"Output dir    : {run_dir}")
    print(f"Manifest      : {manifest_path}")
    print("=" * 72)

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
