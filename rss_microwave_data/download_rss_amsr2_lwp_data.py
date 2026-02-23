"""
download_RSS_AMSR2_LWP.py
=========================
Downloads RSS GCOM-W1 AMSR2 Daily Environmental Suite (Version 8.2) netCDF files
from the Remote Sensing Systems HTTPS server for each unique date found in the
EMIT coincident data directory.

Data source:
    https://data.remss.com/amsr2/ocean/L3/v08.2/daily/{YYYY}/{MM}/
    Wentz et al. (2021), doi:10.56236/RSS-bq

File contents (each .nc file contains ascending + descending passes):
    - sst        : Sea Surface Temperature (°C)
    - wspd_lf    : Wind Speed Low Frequency (m/s)
    - wspd_mf    : Wind Speed Medium Frequency (m/s)
    - vapor      : Columnar Water Vapor (mm)
    - cloud      : Columnar Cloud Liquid Water (mm)  <-- what you want
    - rain       : Rain Rate (mm/hr)
    - time       : UTC observation time

Grid: 0.25° x 0.25°, 1440 x 720 (global), lon [0.125°E to 359.875°E], lat [-89.875° to 89.875°]
Ascending pass  = daytime  (~13:30 local)
Descending pass = nighttime (~01:30 local)

Usage:
    python download_RSS_AMSR2_LWP.py

    The script will:
    1. Scan EMIT_DIR subdirectory names and extract unique acquisition dates.
    2. For each date, download the corresponding RSS AMSR2 daily netCDF file
       into OUTPUT_DIR/RSS_AMSR2_LWP/.
    3. Skip files that already exist (safe to re-run).
    4. Print a summary of successes, skips, and failures.

Dependencies:
    pip install requests tqdm
    (numpy, netCDF4, or xarray are NOT required just to download)
"""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Directory containing your coincident EMIT / Aqua data files
EMIT_DIR = Path(
    "/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/"
    "Hyperspectral_Cloud_Retrievals/Batch_Scripts/Paper-2/"
    "coincident_EMIT_Aqua_data/southEast_pacific"
)

# Where to save the RSS AMSR2 files (a subfolder is created automatically)
OUTPUT_DIR = EMIT_DIR / "RSS_AMSR2_LWP"

# RSS base URL for AMSR2 v8.2 daily files
BASE_URL = "https://data.remss.com/amsr2/ocean/L3/v08.2/daily"

# Version string used in filenames
VERSION = "v08.2"

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_dates_from_emit_dir(emit_dir: Path) -> list[datetime]:
    """
    Scan the EMIT directory for subdirectories and extract unique acquisition dates.

    Subdirectory names follow the pattern:
        YYYY_M_D_THHMMSS[_N]
    e.g. 2023_9_16_T191106_2  →  2023-09-16
         2023_10_4_T143022    →  2023-10-04

    The optional trailing _N (dataset index) is ignored.
    """
    # Matches: YYYY_M_D_THHMMSS  with an optional trailing _<digits>
    dir_pattern = re.compile(r'^(\d{4})_(\d{1,2})_(\d{1,2})_T\d{6}(?:_\d+)?$')

    dates = set()

    if not emit_dir.exists():
        print(f"[ERROR] EMIT directory not found:\n  {emit_dir}")
        sys.exit(1)

    subdirs = [p for p in emit_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[WARNING] No subdirectories found in:\n  {emit_dir}")

    for d in subdirs:
        m = dir_pattern.match(d.name)
        if m:
            try:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dates.add(datetime(year, month, day))
            except ValueError:
                pass  # not a real date

    return sorted(dates)


def build_url(date: datetime) -> str:
    """
    Build the HTTPS URL for the RSS AMSR2 daily netCDF file for a given date.

    URL pattern:
        https://data.remss.com/amsr2/ocean/L3/v08.2/daily/YYYY/
            RSS_AMSR2_ocean_L3_daily_YYYY-MM-DD_v08.2.nc

    Example:
        https://data.remss.com/amsr2/ocean/L3/v08.2/daily/2023/
            RSS_AMSR2_ocean_L3_daily_2023-09-16_v08.2.nc
    """
    yyyy = date.strftime("%Y")
    date_str = date.strftime("%Y-%m-%d")
    filename = f"RSS_AMSR2_ocean_L3_daily_{date_str}_{VERSION}.nc"
    return f"{BASE_URL}/{yyyy}/{filename}", filename


def download_file(url: str, dest_path: Path, timeout: int = 120) -> bool:
    """
    Download a single file from `url` to `dest_path`.
    Returns True on success, False on failure.
    Shows a simple progress indicator.
    """
    try:
        import requests
    except ImportError:
        print("[ERROR] 'requests' is not installed. Run:  pip install requests")
        sys.exit(1)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                print(f"  [MISSING]  No file on server for this date (404)")
                return False
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB chunks

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 100 * downloaded / total
                            print(f"\r  {pct:5.1f}%  {downloaded/1e6:.1f} / {total/1e6:.1f} MB",
                                  end="", flush=True)
            print()  # newline after progress
            return True

    except Exception as e:
        print(f"\n  [ERROR]  {e}")
        if dest_path.exists():
            dest_path.unlink()  # remove partial file
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  RSS AMSR2 v8.2 Daily LWP Downloader")
    print("=" * 65)
    print(f"  EMIT directory : {EMIT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    # 1. Find dates
    dates = extract_dates_from_emit_dir(EMIT_DIR)
    if not dates:
        print("[ERROR] Could not find any valid dates in EMIT filenames.")
        print("  Check that EMIT_DIR is correct and files follow standard naming.")
        sys.exit(1)

    print(f"Found {len(dates)} unique acquisition date(s):")
    for d in dates:
        print(f"  {d.strftime('%Y-%m-%d')}")
    print()

    # 2. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Download
    n_ok = n_skip = n_fail = 0

    for date in dates:
        url, filename = build_url(date)
        dest_path = OUTPUT_DIR / filename

        print(f"[{date.strftime('%Y-%m-%d')}]  {filename}")

        if dest_path.exists():
            print(f"  [SKIP]  File already exists ({dest_path.stat().st_size / 1e6:.1f} MB)")
            n_skip += 1
            continue

        print(f"  Downloading from:\n    {url}")
        success = download_file(url, dest_path)

        if success:
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  [OK]  Saved ({size_mb:.1f} MB) -> {dest_path.name}")
            n_ok += 1
        else:
            n_fail += 1

        print()

    # 4. Summary
    print("=" * 65)
    print(f"  Done.  Downloaded: {n_ok}  |  Skipped: {n_skip}  |  Failed: {n_fail}")
    print(f"  Files saved to: {OUTPUT_DIR}")
    print("=" * 65)

    # 5. Remind about reading the data
    print("""
How to read the cloud LWP from the downloaded files in Python
─────────────────────────────────────────────────────────────
import xarray as xr
import numpy as np

ds = xr.open_dataset("f32_20230815_v8.2.nc")

# Ascending pass (daytime, ~13:30 local) cloud LWP in mm
lwp_asc  = ds["cloud"].sel(pass_type="ascending")

# Descending pass (nighttime, ~01:30 local)
lwp_desc = ds["cloud"].sel(pass_type="descending")

# Grid coordinates
lat = ds["lat"].values   # shape (720,),  -89.875 to +89.875
lon = ds["lon"].values   # shape (1440,),  0.125 to 359.875

# NaN where flagged (land, sea-ice, rain, no data)
# Valid cloud range: approx -0.05 to 2.45 mm

# Southeast Pacific subset example (~VOCALS-REx region):
se_pac = ds["cloud"].sel(lat=slice(-35, -5), lon=slice(270, 290))

How to read in MATLAB
─────────────────────
  filename = 'f32_20230815_v8.2.nc';
  lwp  = ncread(filename, 'cloud');   % [lon x lat x pass], single
  lat  = ncread(filename, 'lat');
  lon  = ncread(filename, 'lon');
  % pass dim 1 = ascending, pass dim 2 = descending (check ncinfo)
  % Fill/missing values are typically NaN or a large fill value – check
  % the variable attributes with ncinfo(filename) or ncdisp(filename).

Citation
────────
Wentz, F.J., T. Meissner, C. Gentemann, K.A. Hilburn, J. Scott, 2021:
  RSS GCOM-W1 AMSR2 Daily Environmental Suite on 0.25 deg grid, Version 8.2,
  Remote Sensing Systems, Santa Rosa, CA.
  Available at www.remss.com. https://doi.org/10.56236/RSS-bq
""")


if __name__ == "__main__":
    main()