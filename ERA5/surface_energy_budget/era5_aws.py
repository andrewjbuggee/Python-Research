"""Read ERA5 directly from the NSF NCAR public S3 bucket, no download required.

    https://registry.opendata.aws/nsf-ncar-era5/
    bucket s3://nsf-ncar-era5   region us-west-2   anonymous access

The point of this module is to return a Dataset that is INDISTINGUISHABLE from
one written by ``download_era5_seb.py``, so every existing analysis function --
``build_ocean_mask``, ``compute_turbulent_fluxes``, ``compute_radiative_fluxes``,
``compute_net_seb``, ``make_maps``, ``make_pdfs`` -- works on it unchanged.

Four differences from the CDS have to be reconciled to make that true:

1. **Short names differ.** The bucket uses original ECMWF GRIB short names, so
   sea ice cover is ``ci`` not ``siconc``, 2 m temperature is ``2t`` not ``t2m``.
   NCAR_VARS below maps every canonical name used in this project.

2. **Longitude runs 0..359.75**, not -180..180. A Barrow request for -165..-150
   has to become 195..210, and the result is converted back so that the region
   definitions and map code keep working.

3. **Time is stored two different ways.** Analysis files carry a plain hourly
   ``time`` axis. Mean-flux files carry a 2-D ``(forecast_initial_time,
   forecast_hour)`` grid that must be flattened; valid time is the sum of the
   two.

4. **One variable per file**, split monthly (analysis) or half-monthly (flux),
   rather than all variables in one file per chunk.

Cost model, measured against the live bucket
--------------------------------------------
Opening a file costs roughly 12-15 s of HDF5 header reading over HTTP, and that
dominates: the array reads themselves take well under a second for a small
region. Because there is one file per variable, a request costs about
``15 s x n_variables x n_periods``. That makes this excellent for exploratory
work over weeks-to-months, and poor for bulk multi-decade extraction, where the
CDS's server-side subsetting still wins despite its queue.

The mean-flux files are also chunked ``(1, 12, 721, 1440)`` -- one chunk spans
the entire globe -- so a small spatial subset still inflates whole global
chunks. Analysis files are tiled ``(27, 139, 279)`` and subset far better.

Not available here: ``tcslw`` (total column supercooled liquid water) is absent
from the bucket's surface groups. ``tp`` is not archived as an accumulation;
use ``mtpr`` (mean total precipitation rate, kg m-2 s-1) instead.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

BUCKET = "nsf-ncar-era5"
EPOCH = np.datetime64("1900-01-01T00:00:00")

# canonical project name -> (bucket group, NCAR/GRIB short name)
# Verified against a live listing of the bucket for 2000-09.
NCAR_VARS: dict[str, tuple[str, str]] = {
    # --- time-mean surface fluxes (forecast, half-month files) --------------
    "msshf": ("e5.oper.fc.sfc.meanflux", "msshf"),
    "mslhf": ("e5.oper.fc.sfc.meanflux", "mslhf"),
    "msnlwrf": ("e5.oper.fc.sfc.meanflux", "msnlwrf"),
    "msnswrf": ("e5.oper.fc.sfc.meanflux", "msnswrf"),
    "msdwlwrf": ("e5.oper.fc.sfc.meanflux", "msdwlwrf"),
    "msdwswrf": ("e5.oper.fc.sfc.meanflux", "msdwswrf"),
    "msnlwrfcs": ("e5.oper.fc.sfc.meanflux", "msnlwrfcs"),
    "msnswrfcs": ("e5.oper.fc.sfc.meanflux", "msnswrfcs"),
    "msdwlwrfcs": ("e5.oper.fc.sfc.meanflux", "msdwlwrfcs"),
    "msdwswrfcs": ("e5.oper.fc.sfc.meanflux", "msdwswrfcs"),
    "mtpr": ("e5.oper.fc.sfc.meanflux", "mtpr"),  # stands in for CDS "tp"
    # --- surface analysis (monthly files) -----------------------------------
    "siconc": ("e5.oper.an.sfc", "ci"),
    "skt": ("e5.oper.an.sfc", "skt"),
    "t2m": ("e5.oper.an.sfc", "2t"),
    "d2m": ("e5.oper.an.sfc", "2d"),
    "u10": ("e5.oper.an.sfc", "10u"),
    "v10": ("e5.oper.an.sfc", "10v"),
    "sp": ("e5.oper.an.sfc", "sp"),
    "tcc": ("e5.oper.an.sfc", "tcc"),
    "lcc": ("e5.oper.an.sfc", "lcc"),
    "mcc": ("e5.oper.an.sfc", "mcc"),
    "hcc": ("e5.oper.an.sfc", "hcc"),
    "tciw": ("e5.oper.an.sfc", "tciw"),
    "tclw": ("e5.oper.an.sfc", "tclw"),
    "tcsw": ("e5.oper.an.sfc", "tcsw"),
    "tcrw": ("e5.oper.an.sfc", "tcrw"),
    "tcwv": ("e5.oper.an.sfc", "tcwv"),
    "fal": ("e5.oper.an.sfc", "fal"),
    "sst": ("e5.oper.an.sfc", "sstk"),
    "istl1": ("e5.oper.an.sfc", "istl1"),
    "istl2": ("e5.oper.an.sfc", "istl2"),
    "istl3": ("e5.oper.an.sfc", "istl3"),
    "istl4": ("e5.oper.an.sfc", "istl4"),
    # --- instantaneous forecast surface -------------------------------------
    "cbh": ("e5.oper.fc.sfc.instan", "cbh"),
}

# Present in the CDS variable sets but genuinely absent from this bucket.
UNAVAILABLE_ON_AWS = {"tcslw", "tp"}

# The minimum needed to run the SEB analysis end to end.
SEB_MINIMAL = (
    "msshf", "mslhf", "msnlwrf", "msnswrf", "siconc",
)
SEB_STANDARD = SEB_MINIMAL + (
    "msdwlwrf", "msdwswrf", "skt", "t2m", "u10", "v10", "fal", "tcc", "tcwv",
)


def _filesystem():
    """Anonymous s3fs handle. Imported lazily so the module loads without s3fs."""
    import fsspec

    return fsspec.filesystem("s3", anon=True)


@lru_cache(maxsize=512)
def _month_listing(group: str, year: int, month: int) -> tuple[tuple[str, str], ...]:
    """``((ncar_short_name, full_key), ...)`` for one group and month.

    The date-range suffix in each filename varies by group (monthly for
    analysis, half-monthly for fluxes), so the keys are discovered by listing
    rather than constructed by guesswork. Cached because a listing is reused by
    every variable in the same group and month.
    """
    fs = _filesystem()
    prefix = f"{BUCKET}/{group}/{year:04d}{month:02d}/"
    try:
        keys = fs.ls(prefix, detail=False)
    except FileNotFoundError:
        return ()
    out = []
    for k in keys:
        m = re.search(r"\.\d+_\d+_(\w+)\.ll025sc\.", k)
        if m:
            out.append((m.group(1), k))
    return tuple(sorted(out))


def _keys_for(canonical: str, months: list[tuple[int, int]]) -> list[str]:
    """Every S3 key holding ``canonical`` across the requested months."""
    if canonical in UNAVAILABLE_ON_AWS:
        raise KeyError(
            f"{canonical!r} is not in the NCAR bucket. "
            f"{'Use mtpr instead of tp.' if canonical == 'tp' else ''}"
        )
    if canonical not in NCAR_VARS:
        raise KeyError(f"No AWS mapping for {canonical!r}.")
    group, ncar = NCAR_VARS[canonical]
    found: list[str] = []
    for y, m in months:
        found += [k for name, k in _month_listing(group, y, m) if name == ncar]
    if not found:
        raise FileNotFoundError(
            f"No files for {canonical!r} ({group}/{ncar}) in {months}."
        )
    return sorted(found)


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _apply_cf(values: np.ndarray, attrs) -> np.ndarray:
    """Apply CF ``_FillValue`` / ``scale_factor`` / ``add_offset`` by hand.

    netCDF4 and xarray do this automatically; **raw h5py does not**. These files
    store land as ``_FillValue = 9.999e+20`` rather than NaN, so skipping this
    step silently turns every land cell into a finite value. Downstream that is
    not a cosmetic problem: ``build_ocean_mask`` uses "``siconc`` is not NaN" as
    its land test, so unmasked fill values make land look like ocean and pull the
    Alaskan North Slope into an "all-ocean" average.
    """
    out = values.astype("float32", copy=True)

    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is not None:
        fill = float(np.asarray(fill).ravel()[0])
        # Exact equality is how the sentinel is written, but compare with a
        # relative tolerance so a float32/float64 round-trip cannot miss it.
        out[np.isclose(out, fill, rtol=1e-6, atol=0.0)] = np.nan

    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None:
        out = out * float(np.asarray(scale).ravel()[0])
    if offset is not None:
        out = out + float(np.asarray(offset).ravel()[0])
    return out


def _read_one(
    key: str, lat_slice, lon_idx, lo_ns=None, hi_ns=None
) -> tuple[np.ndarray, np.ndarray]:
    """Read one file's window. Returns ``(values[time,lat,lon], times)``.

    Handles both storage layouts: a plain hourly ``time`` axis, and the
    ``(forecast_initial_time, forecast_hour)`` grid used by the flux files,
    whose valid time is the sum of the two coordinates.

    ``lo_ns``/``hi_ns`` restrict the read along time BEFORE any array is pulled.
    That matters enormously for the flux files: their HDF5 chunks are
    ``(1, 12, 721, 1440)``, one whole globe per initialisation time, so reading
    all 30 initialisations to then discard 28 of them would transfer the entire
    635 MB file to obtain two days of a small region.
    """
    import h5py

    fs = _filesystem()
    with h5py.File(fs.open(key), "r") as h:
        data_name = next(
            n for n in h
            if getattr(h[n], "shape", None) and len(h[n].shape) >= 3
        )
        d = h[data_name]

        if "forecast_initial_time" in h:
            init = np.asarray(h["forecast_initial_time"][:], dtype="int64")
            fhour = np.asarray(h["forecast_hour"][:], dtype="int64")
            valid = (init[:, None] + fhour[None, :])  # (n_init, n_fhour)
            t_all = (EPOCH + valid.astype("timedelta64[h]")).astype("datetime64[ns]")

            keep_init = np.ones(init.size, dtype=bool)
            if lo_ns is not None:
                # Keep an initialisation if ANY of its forecast hours is wanted.
                keep_init = ((t_all >= lo_ns) & (t_all <= hi_ns)).any(axis=1)
            idx = np.nonzero(keep_init)[0]
            if idx.size == 0:
                return (np.empty((0, 0, 0), dtype="float32"),
                        np.empty(0, dtype="datetime64[ns]"))
            # Contiguous slice keeps HDF5 reading whole chunks in order.
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
            block = _apply_cf(d[i0:i1, :, lat_slice, :][..., lon_idx], d.attrs)
            values = block.reshape(-1, block.shape[2], block.shape[3])
            times = t_all[i0:i1].reshape(-1)
        else:
            hours = np.asarray(h["time"][:], dtype="int64")
            t_all = (EPOCH + hours.astype("timedelta64[h]")).astype("datetime64[ns]")
            idx = np.nonzero(
                np.ones(t_all.size, dtype=bool) if lo_ns is None
                else ((t_all >= lo_ns) & (t_all <= hi_ns))
            )[0]
            if idx.size == 0:
                return (np.empty((0, 0, 0), dtype="float32"),
                        np.empty(0, dtype="datetime64[ns]"))
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
            values = _apply_cf(d[i0:i1, lat_slice, :][..., lon_idx], d.attrs)
            times = t_all[i0:i1]

    return values, times


def load_region(
    north_deg: float,
    west_deg: float,
    south_deg: float,
    east_deg: float,
    start: date | datetime | str,
    end: date | datetime | str,
    variables=SEB_STANDARD,
    verbose: bool = True,
) -> "xr.Dataset":
    """Load a lat/lon window straight from S3 as a canonical-name Dataset.

    Parameters mirror the CDS ``area`` ordering used elsewhere in this project
    (North, West, South, East, longitude on [-180, 180]).

    The returned Dataset has dims ``(valid_time, latitude, longitude)``,
    latitude descending and longitude ascending on [-180, 180], exactly like a
    file from ``download_era5_seb.py`` -- so it can be handed directly to the
    analysis functions in ``seb_analysis_common``.
    """
    import h5py
    import xarray as xr

    start = _as_date(start)
    end = _as_date(end)
    months = _months_between(start, end)
    fs = _filesystem()

    # Grid indices come from any one file; the grid is identical across the archive.
    probe_key = _keys_for(variables[0], months[:1])[0]
    with h5py.File(fs.open(probe_key), "r") as h:
        lat_all = np.asarray(h["latitude"][:], dtype="float64")   # 90 -> -90
        lon_all = np.asarray(h["longitude"][:], dtype="float64")  # 0 -> 359.75

    lat_hits = np.nonzero((lat_all <= north_deg) & (lat_all >= south_deg))[0]
    if lat_hits.size == 0:
        raise ValueError(f"No grid latitudes between {south_deg} and {north_deg}.")
    lat_slice = slice(int(lat_hits[0]), int(lat_hits[-1]) + 1)
    lat_sel = lat_all[lat_slice]

    # -180..180 request -> 0..360 storage, keeping the result sorted ascending
    # in the -180..180 frame so the map code sees the convention it expects.
    lon_0360 = np.where(lon_all > 180.0, lon_all - 360.0, lon_all)
    lon_hits = np.nonzero((lon_0360 >= west_deg) & (lon_0360 <= east_deg))[0]
    if lon_hits.size == 0:
        raise ValueError(f"No grid longitudes between {west_deg} and {east_deg}.")
    order = np.argsort(lon_0360[lon_hits])
    lon_idx = lon_hits[order]
    lon_sel = lon_0360[lon_idx]

    if verbose:
        n_files = sum(len(_keys_for(v, months)) for v in variables)
        print(f"  window   : {lat_sel.size} lat x {lon_sel.size} lon "
              f"({north_deg}..{south_deg}N, {west_deg}..{east_deg}E)")
        print(f"  months   : {len(months)}   variables: {len(variables)}")
        print(f"  files    : {n_files}  (~{n_files * 14 / 60:.1f} min of open overhead)")

    # Requested window, inclusive of the final day's last hour. Pushed down into
    # the reader so only the needed chunks ever leave S3.
    lo = np.datetime64(start, "ns")
    hi = np.datetime64(end, "ns") + np.timedelta64(1, "D") - np.timedelta64(1, "ns")

    data_vars = {}
    ref_times = None
    for i, canonical in enumerate(variables, 1):
        keys = _keys_for(canonical, months)
        chunks, times = [], []
        for k in keys:
            v, t = _read_one(k, lat_slice, lon_idx, lo, hi)
            if t.size:
                chunks.append(v)
                times.append(t)
        if not chunks:
            raise ValueError(f"No {canonical!r} data inside {start}..{end}.")
        values = np.concatenate(chunks, axis=0)
        tvals = np.concatenate(times)
        order_t = np.argsort(tvals)
        values, tvals = values[order_t], tvals[order_t]

        keep = (tvals >= lo) & (tvals <= hi)
        values, tvals = values[keep], tvals[keep]

        if ref_times is None:
            ref_times = tvals
        elif not np.array_equal(tvals, ref_times):
            # Analysis and forecast streams can disagree at the edges of the
            # request; align on the intersection rather than silently padding.
            common = np.intersect1d(ref_times, tvals)
            sel_new = np.isin(tvals, common)
            values, tvals = values[sel_new], tvals[sel_new]
            for name in list(data_vars):
                arr, old_t = data_vars[name]
                data_vars[name] = (arr[np.isin(old_t, common)], common)
            ref_times = common

        data_vars[canonical] = (values, tvals)
        if verbose:
            print(f"    [{i}/{len(variables)}] {canonical:<11} "
                  f"{values.shape[0]:>5} steps from {len(keys)} file(s)")

    ds = xr.Dataset(
        {
            name: (("valid_time", "latitude", "longitude"), arr.astype("float32"))
            for name, (arr, _) in data_vars.items()
        },
        coords={
            "valid_time": ref_times,
            "latitude": lat_sel,
            "longitude": lon_sel,
        },
    )
    ds.attrs["source"] = f"s3://{BUCKET} (NSF NCAR ERA5, anonymous)"
    ds.attrs["convention"] = "ERA5: surface fluxes positive downward"
    return ds


def _as_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x), "%Y-%m-%d").date()


def load_named_region(region_name: str, start, end, variables=SEB_STANDARD,
                      verbose: bool = True) -> "xr.Dataset":
    """``load_region`` for one of the project's named regions."""
    from era5_seb_variables import get_region

    r = get_region(region_name)
    return load_region(
        r.north_deg, r.west_deg, r.south_deg, r.east_deg,
        start, end, variables=variables, verbose=verbose,
    )
