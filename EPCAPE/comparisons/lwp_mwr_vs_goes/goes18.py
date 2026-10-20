"""GOES-18 SatCORPS (VISST) gridded cloud products at the Scripps Pier (epcvisstgridg18minnisX1.c1).

The ARM datastream holds one file per day with 48 half-hourly images on a 0.5 deg x 0.5 deg grid of
30 latitude x 36 longitude boxes (28.25-42.75 N, 129.75-112.25 W, box centres; checked on the 2023-07-15
file, 19 Oct 2026). Box averages are over the satellite pixels in each box (file attribute ``Title``:
"Gridded cloud products derived from pixel level data"). The ARM grid is therefore coarser than the
0.2 deg x 0.2 deg box Seethala Chellappan describes using (email of 19 Oct 2026); her file is not this one.

Extra dimensions (global attributes of the files):
  level     1 total, 2 low (0-2 km), 3 mid (>2-<6 km), 4 high (>6 km)       e.g. cloud_percentage_level
  cld_type  1 total clouds, 2 ice, 3 water, 4 supercooled water            e.g. cloud_percentage, optical_depth_linear
  cld_phase 1 ice, 2 water, 3 supercooled water                            e.g. water_path, particle_size
  scn_type  1 total, 2 clear
(1-based in the attributes; 0-based indices below.) Cloud products are -9999 (NaN after decoding) when a
box holds no cloudy pixels of that kind ("Flag_comment" attribute).

Because load_product cannot combine the whole grid along time in reasonable memory, ``reduce_pier_boxes``
keeps only the 3 x 3 boxes centred on the Pier box and writes one campaign file to processed/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from EPCAPE.download_data.config import Machine, active_machine, campaign_dates, load_config
from EPCAPE.comparisons.seasonal_averages.sources import product_files

PRODUCT = "goes18_visst_grid_X1"  # config.yaml product (datastream epcvisstgridg18minnisX1.c1)

# 0-based indices along the extra dimensions (the files' attributes number them from 1)
LEVEL = {"total": 0, "low": 1, "mid": 2, "high": 3}  # level: low = 0-2 km, mid = >2-<6 km, high = >6 km
CLD_TYPE = {"total": 0, "ice": 1, "water": 2, "supercooled": 3}  # cld_type
CLD_PHASE = {"ice": 0, "water": 1, "supercooled": 2}  # cld_phase

# Variables kept by the reduction (all exist in the complete files; checked 19 Oct 2026)
KEEP_VARIABLES = [
    "cloud_percentage",  # (time, lat, lon, cld_type), %
    "cloud_percentage_level",  # (time, lat, lon, level), %
    "water_path",  # (time, lat, lon, cld_phase), g m-2: mean over the box's cloudy pixels of that phase
    "water_path_sd",  # (time, lat, lon, cld_phase), g m-2
    "optical_depth_linear",  # (time, lat, lon, cld_type), unitless
    "optical_depth_log",  # (time, lat, lon, cld_type), unitless
    "cloud_height_top",  # (time, lat, lon, cld_type), km
    "cloud_height_top_level",  # (time, lat, lon, level), km
    "cloud_height_base",  # (time, lat, lon, cld_type), km
    "particle_size",  # (time, lat, lon, cld_phase), um (water: effective radius)
    "solar_zenith_angle",  # (time, lat, lon), deg ("For internal use only!" in the attributes)
    "scan_time",  # (time, lat, lon), hours UTC: the time the scan line reached the box
]


def pier_box_index(lat: np.ndarray, lon: np.ndarray, site_lat_deg: float, site_lon_deg: float) -> tuple:
    """(i, j) of the grid box whose centre is nearest the site; raises if the site is outside the grid.

    lat, lon: box centres (deg N, deg E). A box spans +- half the grid spacing around its centre, so the
    nearest centre is the box that contains the site."""
    i = int(np.abs(lat - site_lat_deg).argmin())
    j = int(np.abs(lon - site_lon_deg).argmin())
    half_dlat_deg = 0.5 * float(np.median(np.abs(np.diff(lat))))
    half_dlon_deg = 0.5 * float(np.median(np.abs(np.diff(lon))))
    if abs(lat[i] - site_lat_deg) > half_dlat_deg or abs(lon[j] - site_lon_deg) > half_dlon_deg:
        raise ValueError(f"site ({site_lat_deg}, {site_lon_deg}) lies outside the grid")
    return i, j


def pier_path(machine: Optional[Machine] = None) -> Path:
    """processed/goes18_visst_pier_X1_<start>_<end>.nc (the output of ``reduce_pier_boxes``)."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    start, end = campaign_dates(cfg)
    return machine.processed_dir() / f"goes18_visst_pier_X1_{start:%Y%m%d}_{end:%Y%m%d}.nc"


def reduce_pier_boxes(
    site_lat_deg: float,
    site_lon_deg: float,
    *,
    half_width: int = 1,
    machine: Optional[Machine] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Cut the (2 half_width + 1)^2 boxes centred on the site's box out of every daily file; write one file.

    The site's box is at index (half_width, half_width) of the new lat/lon dimensions. Every daily file
    must have the same grid as the first one (checked). Output: ``pier_path(machine)``, with the
    variables of KEEP_VARIABLES (decoded: scale factors applied, -9999 -> NaN) plus
    ``scan_datetime`` = image day + scan_time (UTC; the time the scan reached each box)."""
    files = product_files(PRODUCT, machine)
    if not files:
        raise FileNotFoundError(f"No {PRODUCT} files. Download them with\n"
                                f"  python comparisons/lwp_mwr_vs_goes/download_data.py")
    pieces = []
    ref_lat = ref_lon = None
    for k, path in enumerate(files, 1):
        with xr.open_dataset(path) as ds:
            lat, lon = ds["latitude"].values, ds["longitude"].values
            if ref_lat is None:
                ref_lat, ref_lon = lat, lon
                i, j = pier_box_index(lat, lon, site_lat_deg, site_lon_deg)
                if not (half_width <= i < len(lat) - half_width and half_width <= j < len(lon) - half_width):
                    raise ValueError("the site's box is too close to the grid edge for this half_width")
                log(f"Site box: centre {lat[i]:.2f} N, {lon[j]:.2f} E (index {i}, {j}); "
                    f"{2 * half_width + 1} x {2 * half_width + 1} boxes kept")
            elif not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):
                raise ValueError(f"{path.name}: grid differs from the first file's")
            sub = ds[[v for v in KEEP_VARIABLES if v in ds]].isel(
                lat=slice(i - half_width, i + half_width + 1), lon=slice(j - half_width, j + half_width + 1))
            sub = sub.assign_coords(latitude=("lat", lat[i - half_width:i + half_width + 1]),
                                    longitude=("lon", lon[j - half_width:j + half_width + 1]))
            # absolute scan time: the image's UTC day + scan_time (hours). The scan of a box lies a few
            # minutes after the nominal image time (e.g. image 15:00:21, scan 15:01:48 at the Pier box).
            day = pd.DatetimeIndex(sub["time"].values).normalize()
            scan_h = sub["scan_time"].values  # (time, lat, lon), hours
            scan_dt = day.values[:, None, None] + (scan_h * 3600e9).astype("timedelta64[ns]")
            # a scan just after midnight belonging to an image just before it (hours < 1 for a 23:30 image)
            late = (pd.DatetimeIndex(sub["time"].values).hour >= 23)[:, None, None] & (scan_h < 1)
            scan_dt = np.where(late, scan_dt + np.timedelta64(1, "D"), scan_dt)
            sub["scan_datetime"] = (("time", "lat", "lon"), np.where(np.isfinite(scan_h), scan_dt,
                                                                     np.datetime64("NaT")))
            pieces.append(sub.load())
        if k % 50 == 0:
            log(f"  reduced {k}/{len(files)} files")
    out = xr.concat(pieces, dim="time").sortby("time")
    out = out.isel(time=~pd.DatetimeIndex(out["time"].values).duplicated())
    out.attrs = {
        "title": "GOES-18 SatCORPS (VISST) gridded cloud products, 0.5 deg boxes around the Scripps Pier",
        "source": f"{PRODUCT} = epcvisstgridg18minnisX1.c1 ({len(files)} daily files)",
        "site_box_index": f"lat={half_width}, lon={half_width} (the box containing {site_lat_deg:.4f} N, "
                          f"{site_lon_deg:.4f} E)",
        "dimensions": "level 0 total, 1 low (0-2 km), 2 mid (>2-<6 km), 3 high (>6 km); cld_type 0 total, "
                      "1 ice, 2 water, 3 supercooled; cld_phase 0 ice, 1 water, 2 supercooled (0-based)",
    }
    path = pier_path(machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars if v != "scan_datetime"}
    enc["time"] = {"units": "seconds since 1970-01-01", "dtype": "float64"}
    enc["scan_datetime"] = {"units": "seconds since 1970-01-01", "dtype": "float64"}
    out.to_netcdf(path, encoding=enc)
    log(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {out.sizes['time']:,} images)")
    return path


def load_pier(machine: Optional[Machine] = None) -> xr.Dataset:
    """The reduced file of ``reduce_pier_boxes``, in memory."""
    path = pier_path(machine)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Create it with\n"
                                f"  python comparisons/lwp_mwr_vs_goes/download_data.py")
    with xr.open_dataset(path) as ds:
        return ds.load()


# What the MWR vs GOES-18 LWP comparison reads from the Pier box (box_table); everything else is dropped by
# extract_pier_box. cld_type keeps "total" and "water" so GOES_CLOUD_KIND can still choose either.
COMPARISON_VARIABLES = ["water_path", "cloud_percentage_level", "optical_depth_linear", "cloud_height_top",
                        "particle_size", "solar_zenith_angle", "scan_datetime"]


def pier_box_path(machine: Optional[Machine] = None) -> Path:
    """processed/goes18_visst_pierbox_X1_<start>_<end>.nc (the output of ``extract_pier_box``)."""
    cfg = load_config()
    machine = machine or active_machine(cfg)
    start, end = campaign_dates(cfg)
    return machine.processed_dir() / f"goes18_visst_pierbox_X1_{start:%Y%m%d}_{end:%Y%m%d}.nc"


def extract_pier_box(machine: Optional[Machine] = None, log: Callable[[str], None] = print) -> Path:
    """The Pier box alone, with only COMPARISON_VARIABLES, from the 3 x 3 file of ``reduce_pier_boxes``.

    This is the only GOES-18 file the comparison notebook needs, so the ~3.5 GB of daily grid files can be
    deleted once it exists (re-download them with download_data.py to rebuild it, or to use other boxes).
    The lat and lon dimensions are kept with length 1, so ``box_table(ds, i=0, j=0)`` reads it."""
    with xr.open_dataset(pier_path(machine)) as ds:
        k = int(str(ds.attrs["site_box_index"]).split("lat=")[1].split(",")[0])  # the Pier box's index (1)
        out = ds[COMPARISON_VARIABLES].isel(lat=[k], lon=[k]).load()
        out.attrs = dict(ds.attrs)
    out.attrs["title"] = "GOES-18 SatCORPS (VISST) gridded cloud products, the 0.5 deg box containing the Scripps Pier"
    out.attrs["site_box_index"] = "lat=0, lon=0 (the only box)"
    out.attrs["variables_kept"] = ("only what comparisons/lwp_mwr_vs_goes reads (goes18.COMPARISON_VARIABLES); "
                                   f"cut from {pier_path(machine).name}")
    path = pier_box_path(machine)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars if v != "scan_datetime"}
    enc["time"] = {"units": "seconds since 1970-01-01", "dtype": "float64"}
    enc["scan_datetime"] = {"units": "seconds since 1970-01-01", "dtype": "float64"}
    out.to_netcdf(path, encoding=enc)
    log(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {out.sizes['time']:,} images)")
    return path


def load_pier_box(machine: Optional[Machine] = None) -> xr.Dataset:
    """The file of ``extract_pier_box``, in memory (read it with ``box_table(ds, i=0, j=0)``)."""
    path = pier_box_path(machine)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Create it with\n"
                                f"  python comparisons/lwp_mwr_vs_goes/download_data.py")
    with xr.open_dataset(path) as ds:
        return ds.load()


def box_table(ds: xr.Dataset, *, i: int = 1, j: int = 1, cloud_kind: str = "total") -> pd.DataFrame:
    """The constraint quantities of one grid box, one row per image (index = nominal image time, UTC).

    Columns (units in the names):
      lwp_gm2         water_path, water phase: mean LWP over the box's water-cloud pixels (g m-2)
      lcf             low cloud fraction (0-1) = cloud_percentage_level, level "low" (0-2 km) / 100
      tcf             total cloud fraction (0-1) = cloud_percentage_level, level "total" / 100
      cod             optical_depth_linear of `cloud_kind` clouds ("total" or "water")
      cth_km          cloud_height_top of `cloud_kind` clouds (km)
      sza_deg         solar_zenith_angle of the box (deg)
      re_um           particle_size, water phase (effective radius, um)
      scan_time       when the scan reached the box (UTC)
    A box with no cloud has lcf = tcf = 0 and NaN cloud properties."""
    box = ds.isel(lat=i, lon=j)
    k = CLD_TYPE[cloud_kind]
    return pd.DataFrame({
        "lwp_gm2": box["water_path"].isel(cld_phase=CLD_PHASE["water"]).values,
        "lcf": box["cloud_percentage_level"].isel(level=LEVEL["low"]).values / 100.0,
        "tcf": box["cloud_percentage_level"].isel(level=LEVEL["total"]).values / 100.0,
        "cod": box["optical_depth_linear"].isel(cld_type=k).values,
        "cth_km": box["cloud_height_top"].isel(cld_type=k).values,
        "sza_deg": box["solar_zenith_angle"].values,
        "re_um": box["particle_size"].isel(cld_phase=CLD_PHASE["water"]).values,
        "scan_time": pd.DatetimeIndex(box["scan_datetime"].values),
    }, index=pd.DatetimeIndex(box["time"].values, name="image_time"))
