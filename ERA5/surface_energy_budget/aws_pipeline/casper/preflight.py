#!/usr/bin/env python3
"""Run this FIRST on Casper. Checks everything the pipeline needs, changes nothing.

    python preflight.py              # checks + a tiny real read off GLADE
    python preflight.py --no-read    # checks only, no I/O against the archive

It answers, in order: is the ERA5 archive visible and where; does this python
have the packages; do the analysis modules import; does a one-day regional read
off GLADE work and does it look like ERA5; where can the run write; and is
cartopy's map data staged (compute nodes may have no internet). Anything that
fails prints what to do about it.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEB_DIR = HERE.parent.parent                     # .../ERA5/surface_energy_budget
for p in (str(SEB_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

OK, WARN, BAD = "  ok   ", " WARN  ", " FAIL  "
_failures: list[str] = []
_warnings: list[str] = []


def say(status: str, label: str, detail: str = "", fix: str = "") -> None:
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    if status == BAD:
        _failures.append(label)
        if fix:
            print(f"         -> {fix}")
    elif status == WARN:
        _warnings.append(label)
        if fix:
            print(f"         -> {fix}")


def check_platform() -> None:
    print("=" * 78)
    print(f"preflight  {time.strftime('%Y-%m-%d %H:%M:%S')}  {platform.node()}")
    print("=" * 78)
    say(OK, "python", f"{platform.python_version()} at {sys.executable}")
    for var in ("PBS_JOBID", "PBS_O_QUEUE", "NCAR_HOST", "CONDA_DEFAULT_ENV", "SCRATCH", "WORK"):
        if os.environ.get(var):
            say(OK, f"env {var}", os.environ[var])


def check_packages() -> None:
    import importlib

    needed = ["numpy", "xarray", "h5py", "matplotlib", "pandas", "scipy", "netCDF4"]
    optional = ["cartopy", "s3fs", "fsspec", "dask", "openpyxl", "papermill", "nbconvert", "sklearn"]
    for mod in needed:
        try:
            m = importlib.import_module(mod)
            say(OK, f"import {mod}", getattr(m, "__version__", "?"))
        except ImportError as exc:
            say(BAD, f"import {mod}", str(exc),
                "module load conda && conda activate npl-2025b   (or see setup_casper.sh)")
    for mod in optional:
        try:
            m = importlib.import_module(mod)
            say(OK, f"import {mod}", getattr(m, "__version__", "?") + " (optional)")
        except ImportError:
            note = {"cartopy": "the map figures (5, 6, thumbnail) will fail without it",
                    "s3fs": "only needed to fall back to the AWS mirror",
                    "papermill": "only needed for the notebook runner",
                    "openpyxl": "only needed for the .xlsx observation source"}.get(mod, "")
            say(WARN, f"import {mod}", f"missing -- {note}" if note else "missing")


def check_numpy_vs_laptop() -> None:
    """NPL ships numpy 1.x; the laptop stack is 2.x. Flag it, do not fail on it."""
    import numpy as np

    major = int(np.__version__.split(".")[0])
    if major < 2:
        say(WARN, "numpy major version", f"{np.__version__} (laptop runs 2.x)",
            "no numpy-2-only call is used by these modules, but run the smoke test "
            "below before trusting a long job")


def check_archive(do_read: bool) -> None:
    from aws_pipeline import sources

    print("-" * 78)
    print(sources.describe_sources())      # never raises; reports failure as a line
    print("-" * 78)
    try:
        src = sources.source()
    except Exception as exc:  # noqa: BLE001 - diagnosing this IS the job
        say(BAD, "archive source", f"{type(exc).__name__}: {exc}",
            "find the archive with `ls -d /gdex/data/d633000` and set ERA5_GLADE_ROOT, "
            "or unset ERA5_SOURCE to fall back to the AWS mirror")
        return
    if src.name != "glade":
        say(WARN, "archive source", f"using {src.name}, not GLADE",
            "on Casper the ERA5 archive should be local. Find it with "
            "`ls -d /gdex/data/d633000` and export ERA5_GLADE_ROOT=<that path>")
    else:
        say(OK, "archive source", src.describe())

    from aws_pipeline import era5_s3

    try:
        files = era5_s3.list_month("e5.oper.an.sfc", 2024, 10)
        ci = [f for f in files if f.short == "ci"]
        say(OK if ci else BAD, "listing e5.oper.an.sfc/202410",
            f"{len(files)} files, sea-ice cover present: {bool(ci)}",
            "the archive root is wrong, or this month is missing")
    except Exception as exc:  # noqa: BLE001 - surfacing any failure is the point
        say(BAD, "listing e5.oper.an.sfc/202410", f"{type(exc).__name__}: {exc}")
        return

    if not do_read:
        return
    try:
        t0 = time.time()
        ds = era5_s3.open_dataset(80, -165, 70, -150, [("2024-10-01", "2024-10-01")],
                                  variables=["siconc", "tcc", "msdwlwrf"],
                                  month_align=False, verbose=False)
        sub = ds.load()
        dt = time.time() - t0
        import numpy as np

        sic, tcc, lwd = sub.siconc.values, sub.tcc.values, sub.msdwlwrf.values
        sane = (np.nanmin(sic) >= 0 and np.nanmax(sic) <= 1
                and np.nanmin(tcc) >= 0 and np.nanmax(tcc) <= 1
                and 80 < np.nanmean(lwd) < 400 and np.isnan(sic).any())
        say(OK if sane else BAD, "read 24 h x 3 variables (Barrow)",
            f"{dt:.1f} s | siconc {np.nanmin(sic):.2f}-{np.nanmax(sic):.2f} "
            f"({100 * np.isnan(sic).mean():.0f}% NaN = land) | "
            f"tcc {np.nanmin(tcc):.2f}-{np.nanmax(tcc):.2f} | "
            f"LWD mean {np.nanmean(lwd):.0f} W m-2",
            "values are out of range -- do not trust this archive copy")
        lsm = era5_s3.open_land_sea_mask(80, -165, 70, -150)
        frac = float((lsm.values > 0.5).mean())
        say(OK if 0.0 < frac < 0.5 else WARN, "land-sea mask", f"{100 * frac:.1f}% land over Barrow")
        era5_s3.shutdown_pool()
    except Exception as exc:  # noqa: BLE001
        say(BAD, "test read", f"{type(exc).__name__}: {exc}")


def check_modules() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    for mod in ("plot_lwp_histogram_by_surface_class", "cloud_spatial_extent",
                "map_liquid_hours", "plot_dlr_by_phase", "turbulent_flux_response",
                "cloud_level_wind", "seb_analysis_common"):
        try:
            __import__(mod)
            say(OK, f"analysis module {mod}")
        except Exception as exc:  # noqa: BLE001
            say(BAD, f"analysis module {mod}", f"{type(exc).__name__}: {exc}")


def check_obs_files() -> None:
    """The figure-1/2/3 observation inputs, some of which git cannot carry."""
    for name, why in (("genie_arm_seasonal_hours.txt", "figure 1"),
                      ("genie_arm_monthly_hours_noprecip.csv", "figure 2 (csv source)"),
                      ("genie_arm_cloud_durations.txt", "figure 3"),
                      ("genie_arm_retained_fraction.txt", "figure 3"),
                      ("genie_arm_monthly_hours.xlsx", "figure 2 (xlsx source)")):
        p = SEB_DIR / name
        say(OK if p.exists() else WARN, f"observation file {name}",
            f"{p.stat().st_size / 1e3:.0f} kB" if p.exists() else f"missing -- {why} will be skipped",
            "" if p.exists() else "run casper/sync_to_casper.sh from the laptop (git cannot "
                                  "carry the gitignored .xlsx)")


def check_writable() -> None:
    for var, default in (("SCRATCH", f"/glade/derecho/scratch/{os.environ.get('USER', '')}"),
                         ("WORK", f"/glade/work/{os.environ.get('USER', '')}")):
        root = Path(os.environ.get(var) or default)
        if not root.is_dir():
            say(WARN, f"{var} space", f"{root} not found")
            continue
        try:
            probe = root / ".era5_preflight_probe"
            probe.write_text("ok")
            probe.unlink()
            free = shutil.disk_usage(root).free / 1e12
            say(OK, f"{var} space", f"{root} writable, {free:.1f} TB free on the filesystem")
        except OSError as exc:
            say(BAD, f"{var} space", f"{root}: {exc}")
    cache = os.environ.get("ERA5_CACHE")
    say(OK if cache else WARN, "ERA5_CACHE", cache or "unset",
        "" if cache else "point it at scratch so repeated passes do not re-decode: "
                         "export ERA5_CACHE=$SCRATCH/era5_cache")


def check_cartopy_data() -> None:
    try:
        import cartopy
    except ImportError:
        return
    d = os.environ.get("CARTOPY_DATA_DIR")
    pre = cartopy.config.get("pre_existing_data_dir") or cartopy.config.get("data_dir")
    shp = Path(str(pre)) / "shapefiles" / "natural_earth" / "physical"
    have = shp.is_dir() and any(shp.glob("*coastline*"))
    say(OK if have else WARN, "cartopy map data",
        f"{pre} ({'staged' if have else 'not staged'})",
        "" if have else "compute nodes may have no internet; stage it once on a LOGIN node: "
                        "python casper/prefetch_cartopy.py, then export CARTOPY_DATA_DIR")
    if d:
        say(OK, "CARTOPY_DATA_DIR", d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-read", action="store_true", help="skip the test read of the archive")
    args = ap.parse_args(argv)

    check_platform()
    check_packages()
    check_numpy_vs_laptop()
    check_modules()
    check_archive(not args.no_read)
    check_obs_files()
    check_writable()
    check_cartopy_data()

    print("=" * 78)
    if _failures:
        print(f"{len(_failures)} FAILURE(S): " + ", ".join(_failures))
    if _warnings:
        print(f"{len(_warnings)} warning(s): " + ", ".join(_warnings))
    if not _failures:
        print("ready: the pipeline can run here"
              + (" (see the warnings above first)" if _warnings else ""))
    print("=" * 78)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
