#!/usr/bin/env python3
"""Turn ERA5 specific water contents into absolute ones.

Reads the pressure-level archive written by ``download_era5_pressure.py`` and
produces, for every condensate species:

    <x>_path      kg m-2   mass of the species in the layer each level owns
    <x>_column    kg m-2   the layer paths summed -- the column path
    <x>_density   kg m-3   mass per unit volume at the level
    <x>_incloud   kg kg-1  in-cloud specific content, where cloud fraction > 0

THE TWO CONVERSIONS
===================
ERA5 gives SPECIFIC content: kg of species per kg of total moist air, where
moist air is dry air + vapour + cloud liquid + cloud ice + rain + snow. It is
also a GRID-BOX MEAN, already including the clear part of the box.

Mass path, per layer. Hydrostatic balance puts dp/g kilograms of air above unit
area between two pressure surfaces, so

    W_x = q_x * dp / g                                       [kg m-2]

This needs nothing but the specific content and the pressure coordinate -- no
temperature, no humidity. It is the same integral ERA5 uses to form its own
single-level tclw/tciw, which is what makes ``--validate`` a real test.

Density. From p = rho*T*(q_d*R_d + q_v*R_v) with q_d = 1 - q_v - q_c,

    T_v   = T * (1 + 0.608*q_v - q_c),   q_c = q_l + q_i + q_r + q_s
    rho   = p / (R_d * T_v)
    rho_x = q_x * rho                                        [kg m-3]

The condensate term is SUBTRACTED, not added: suspended water is mass carrying
no partial pressure, so it makes the parcel denser. The form usually quoted,
T*(1 + w/eps)/(1 + w + w_c), is for MIXING ratios and is the wrong one here.

In-cloud content. q_x is a grid-box mean, so the value inside the cloudy part is
q_x / cc. A ground instrument sees the in-cloud value; comparing it against the
grid-box mean compares two different quantities, and in an Arctic winter with
layer cloud fractions near 0.5 that is a factor of two.

LAYER EDGES, AND THE GROUND
===========================
Each level owns the layer between the midpoints to its neighbours. The bottom
edge of the lowest layer, and every edge below it, is clipped to the surface
pressure, so a level sitting under the terrain gets dp = 0 and contributes
nothing. This matters: ERA5 fills below-ground levels by extrapolation, and
integrating them would add water that is inside the rock.

Surface pressure comes from the SINGLE-LEVEL archive (``sp``), which the
existing download already contains. Without it, ``--assume-sp`` falls back to a
fixed value, which is fine over open ocean and wrong over the North Slope.

VALIDATION
==========
``--validate`` integrates clwc and ciwc and compares the result against the
single-level tclw and tciw for the same times and cells. Those two are computed
by ECMWF from the model's own levels, not from the 23 pressure levels here, so
they will not agree exactly -- the pressure-level integral misses structure
between levels and below 1000 hPa. Agreement to a few percent means the
conversion is right; a factor of two or a sign error means it is not.

USAGE
=====
    # convert one season, writing alongside the input
    python convert_specific_to_absolute.py --region barrow

    # check the integral against ERA5's own column values first
    python convert_specific_to_absolute.py --region barrow --validate

    # a single file, to a chosen place
    python convert_specific_to_absolute.py \\
        --files data/barrow_pressure/era5_pl_barrow_202508_01-02.nc \\
        --out-dir /tmp/converted
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np

from download_era5_seb import STORAGE_ROOTS, days_covered_by_file

# Physical constants. g is ERA5's own value, the one its geopotential uses.
G_M_S2 = 9.80665
R_DRY = 287.0597       # J kg-1 K-1
EPS = 0.621981         # R_d / R_v
VIRTUAL_COEF = (1.0 - EPS) / EPS      # 0.6078

# short name -> (output prefix, human label)
SPECIES = {
    "clwc": ("lwc", "cloud liquid"),
    "ciwc": ("iwc", "cloud ice"),
    "crwc": ("rwc", "rain"),
    "cswc": ("swc", "snow"),
}

# Single-level counterparts, for --validate.
COLUMN_TRUTH = {"clwc": "tclw", "ciwc": "tciw", "crwc": "tcrw", "cswc": "tcsw"}


# ----------------------------------------------------------------------------
# Layer geometry
# ----------------------------------------------------------------------------
def layer_thickness_pa(levels_pa: np.ndarray, sp_pa: np.ndarray) -> np.ndarray:
    """Pressure thickness each level owns, clipped at the surface.

    ``levels_pa`` is 1-D and ASCENDING in pressure (top of atmosphere first).
    ``sp_pa`` broadcasts against the output's non-level dimensions. Returns an
    array shaped ``sp_pa.shape + (n_levels,)`` with the level axis LAST.

    Edges sit at the midpoints between neighbouring levels. The top edge is
    half a spacing above the topmost level, the bottom edge half a spacing
    below the lowest, and every edge is then clipped to at most the surface
    pressure. A level entirely below ground therefore gets thickness zero
    rather than contributing extrapolated water inside the terrain.
    """
    p = np.asarray(levels_pa, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("need at least two pressure levels, ascending")
    if not np.all(np.diff(p) > 0):
        raise ValueError("levels_pa must be strictly ascending in pressure")

    edges = np.empty(p.size + 1)
    edges[1:-1] = 0.5 * (p[:-1] + p[1:])
    edges[0] = max(0.0, p[0] - 0.5 * (p[1] - p[0]))
    edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])

    sp = np.asarray(sp_pa, dtype=float)[..., None]        # (..., 1)
    clipped = np.minimum(edges, sp)                       # (..., n+1)
    dp = np.diff(clipped, axis=-1)
    return np.clip(dp, 0.0, None)


def moist_density(p_pa: np.ndarray, t_k: np.ndarray, q_v: np.ndarray,
                  q_cond: np.ndarray) -> np.ndarray:
    """Density of moist air including condensate loading, kg m-3."""
    t_v = t_k * (1.0 + VIRTUAL_COEF * q_v - q_cond)
    return p_pa / (R_DRY * t_v)


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------
def convert_dataset(ds, sp_da=None, assume_sp_pa: float | None = None):
    """Add absolute water-content variables to a pressure-level dataset.

    Returns a NEW dataset holding only the derived fields plus the coordinates,
    so it can be written without carrying the (much larger) inputs along.
    """
    import xarray as xr

    level_name = _level_coord(ds)
    levels_hpa = ds[level_name].values.astype(float)
    order = np.argsort(levels_hpa)                 # ascending pressure
    lv_sorted = levels_hpa[order]
    p_pa_1d = lv_sorted * 100.0

    dims = [d for d in ds["t"].dims if d != level_name] if "t" in ds else None
    if dims is None:
        sample = next(ds[v] for v in ds.data_vars if level_name in ds[v].dims)
        dims = [d for d in sample.dims if d != level_name]

    # Surface pressure, broadcast to the non-level dims.
    if sp_da is not None:
        sp = sp_da.transpose(*dims).values.astype(float)
    elif assume_sp_pa is not None:
        shape = tuple(ds.sizes[d] for d in dims)
        sp = np.full(shape, float(assume_sp_pa))
    else:
        raise ValueError("need surface pressure: pass sp_da or assume_sp_pa")

    dp = layer_thickness_pa(p_pa_1d, sp)           # (..., n_levels), level last

    out = {}
    present = [k for k in SPECIES if k in ds]
    if not present:
        raise KeyError(f"none of {sorted(SPECIES)} present in the dataset")

    # Total condensate, for the virtual-temperature loading term.
    q_cond = None
    for k in present:
        arr = _level_last(ds[k], level_name, dims)[..., order]
        q_cond = arr if q_cond is None else q_cond + arr

    rho = None
    if "t" in ds and "q" in ds:
        t = _level_last(ds["t"], level_name, dims)[..., order]
        qv = _level_last(ds["q"], level_name, dims)[..., order]
        rho = moist_density(p_pa_1d, t, qv, q_cond)

    cc = None
    if "cc" in ds:
        cc = _level_last(ds["cc"], level_name, dims)[..., order]

    coords = {d: ds[d] for d in dims if d in ds.coords}
    coords[level_name] = lv_sorted
    out_dims = tuple(dims) + (level_name,)

    for k in present:
        prefix, _label = SPECIES[k]
        q = _level_last(ds[k], level_name, dims)[..., order]

        path = q * dp / G_M_S2                            # kg m-2 per layer
        out[f"{prefix}_path"] = xr.DataArray(
            path, dims=out_dims, coords=coords,
            attrs={"units": "kg m-2", "long_name":
                   f"{SPECIES[k][1]} mass path of this layer",
                   "note": "q * dp / g; dp clipped at the surface pressure"})

        out[f"{prefix}_column"] = xr.DataArray(
            np.nansum(path, axis=-1), dims=tuple(dims),
            coords={d: coords[d] for d in dims if d in coords},
            attrs={"units": "kg m-2",
                   "long_name": f"{SPECIES[k][1]} column path"})

        if rho is not None:
            out[f"{prefix}_density"] = xr.DataArray(
                q * rho, dims=out_dims, coords=coords,
                attrs={"units": "kg m-3",
                       "long_name": f"{SPECIES[k][1]} water content",
                       "note": "q * p / (R_d * T_v), T_v including "
                               "condensate loading"})

        if cc is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                incloud = np.where(cc > 0.0, q / np.where(cc > 0.0, cc, 1.0),
                                   np.nan)
            out[f"{prefix}_incloud"] = xr.DataArray(
                incloud, dims=out_dims, coords=coords,
                attrs={"units": "kg kg-1",
                       "long_name": f"in-cloud specific {SPECIES[k][1]} content",
                       "note": "grid-box mean divided by fraction_of_cloud_cover"})

    out["layer_dp"] = xr.DataArray(
        dp, dims=out_dims, coords=coords,
        attrs={"units": "Pa", "long_name": "pressure thickness of this layer",
               "note": "zero where the level sits below the surface"})
    if rho is not None:
        out["rho_moist"] = xr.DataArray(
            rho, dims=out_dims, coords=coords,
            attrs={"units": "kg m-3", "long_name": "moist air density"})

    result = xr.Dataset(out)
    result.attrs["conversion"] = (
        "absolute water contents from ERA5 specific contents; "
        "see convert_specific_to_absolute.py"
    )
    return result


def _level_coord(ds) -> str:
    for name in ("pressure_level", "level", "isobaricInhPa", "plev"):
        if name in ds.dims:
            return name
    raise KeyError(f"no pressure-level dimension found in {list(ds.dims)}")


def _level_last(da, level_name: str, dims: list[str]) -> np.ndarray:
    """Values with the level axis moved to the end, in the caller's dim order."""
    return da.transpose(*dims, level_name).values.astype(float)


# ----------------------------------------------------------------------------
# Validation against the single-level columns
# ----------------------------------------------------------------------------
def validate(pl_ds, converted, single_ds) -> int:
    """Compare integrated columns against ERA5's own tclw/tciw/tcrw/tcsw."""
    print("\n" + "-" * 78)
    print("  VALIDATION: pressure-level integral vs the single-level column")
    print("-" * 78)
    print("  ERA5 forms tclw/tciw on its own MODEL levels, which are finer than")
    print("  the pressure levels downloaded here and extend below 1000 hPa, so")
    print("  exact agreement is not expected. A few percent means the")
    print("  conversion is right; a factor of two means it is not.\n")

    times = np.intersect1d(converted["valid_time"].values,
                           single_ds["valid_time"].values)
    if times.size == 0:
        print("  No overlapping times; cannot validate.", file=sys.stderr)
        return 1
    a = converted.sel(valid_time=times)
    b = single_ds.sel(valid_time=times)

    print(f"  {'species':<14}{'integrated':>14}{'ERA5 column':>14}"
          f"{'ratio':>9}{'bias':>12}")
    worst = 0.0
    for short, (prefix, label) in SPECIES.items():
        col = f"{prefix}_column"
        truth = COLUMN_TRUTH[short]
        if col not in a or truth not in b:
            continue
        x = a[col].values.ravel()
        y = b[truth].values.ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        mx, my = float(np.mean(x[ok])), float(np.mean(y[ok]))
        ratio = mx / my if my else np.nan
        print(f"  {label:<14}{1000*mx:>12.3f} g{1000*my:>12.3f} g"
              f"{ratio:>9.3f}{1000*(mx-my):>10.3f} g")
        if np.isfinite(ratio):
            worst = max(worst, abs(ratio - 1.0))

    print(f"\n  worst departure from 1.0: {100*worst:.1f}%")
    if worst > 0.5:
        print("  !! That is too large to be discretisation. Check the level")
        print("     ordering, the dp clipping, and the surface-pressure units.",
              file=sys.stderr)
        return 1
    print("  Consistent with vertical discretisation, not a unit error.")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--storage", choices=sorted(STORAGE_ROOTS), default="external")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--region", default="barrow")
    p.add_argument("--files", nargs="+", default=None,
                   help="Explicit input files, overriding --region.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write. Default: <input dir>/absolute.")
    p.add_argument("--single-dir", type=Path, default=None,
                   help="Single-level archive for surface pressure and for "
                        "--validate. Default: the region's single-level dir.")
    p.add_argument("--assume-sp", type=float, default=None, metavar="PA",
                   help="Fixed surface pressure in Pa when the single-level "
                        "archive is unavailable. Wrong over terrain.")
    p.add_argument("--validate", action="store_true",
                   help="Compare the integrated columns against ERA5's own, "
                        "then exit without writing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files (for a quick check).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import xarray as xr

    root = args.data_root or STORAGE_ROOTS[args.storage]
    pl_dir = Path(root) / f"{args.region}_pressure"
    single_dir = args.single_dir or (Path(root) / args.region)

    if args.files:
        files = sorted(args.files)
    else:
        files = sorted(glob.glob(str(pl_dir / "*.nc")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"  Error: no pressure-level files in {pl_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (Path(files[0]).parent / "absolute")
    print("=" * 78)
    print("Specific -> absolute water content")
    print("=" * 78)
    print(f"  Input      : {len(files)} file(s) from {Path(files[0]).parent}")
    print(f"  Single lvl : {single_dir}")
    print(f"  Output     : {out_dir}")

    # Only the single-level files whose days overlap the inputs. Opening the
    # whole archive to fetch one variable took longer than the conversion
    # itself: 591 files is minutes of HDF5 header reads for a field that spans
    # a couple of days.
    want_days: set = set()
    for f in files:
        want_days |= days_covered_by_file(Path(f))
    sp_files = sorted(
        g for g in glob.glob(str(Path(single_dir) / "*.nc"))
        if not want_days or (days_covered_by_file(Path(g)) & want_days)
    )
    if want_days:
        print(f"  Surface p  : {len(sp_files)} single-level file(s) overlap "
              f"{len(want_days)} day(s)")
    single = None
    if sp_files:
        single = xr.open_mfdataset(sp_files, combine="by_coords", join="outer",
                                   compat="no_conflicts")
        if "sp" in single:
            # sp for the conversion, plus the column truths --validate needs.
            # Subsetting to sp alone silently emptied the validation table.
            keep = ["sp"] + [v for v in COLUMN_TRUTH.values() if v in single]
            single = single[keep].load()
        if "sp" not in single:
            print("  !! single-level archive has no 'sp'; falling back to "
                  "--assume-sp", file=sys.stderr)
            single = None
    elif args.assume_sp is None:
        print(f"  Error: no single-level files in {single_dir} and no "
              f"--assume-sp given.", file=sys.stderr)
        return 1

    if not args.validate:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for i, f in enumerate(files, 1):
        dest = out_dir / (Path(f).stem + "_absolute.nc")
        if dest.exists() and not args.overwrite and not args.validate:
            continue
        ds = xr.open_dataset(f).load()

        sp_da = None
        if single is not None:
            try:
                sp_da = single["sp"].sel(valid_time=ds["valid_time"]).load()
            except KeyError:
                sp_da = None
        if sp_da is None and args.assume_sp is None:
            print(f"  !! {Path(f).name}: no surface pressure for these times; "
                  f"skipped. Use --assume-sp to force one.", file=sys.stderr)
            ds.close()
            continue

        conv = convert_dataset(ds, sp_da=sp_da, assume_sp_pa=args.assume_sp)

        if args.validate:
            if single is None:
                print("  Error: --validate needs the single-level archive.",
                      file=sys.stderr)
                return 1
            rc = validate(ds, conv, single)
            print("=" * 78)
            return rc

        enc = {v: {"zlib": True, "complevel": 4} for v in conv.data_vars}
        conv.to_netcdf(dest, encoding=enc)
        ds.close()
        n_done += 1
        print(f"  [{i}/{len(files)}] {dest.name}  "
              f"{dest.stat().st_size/1024**2:,.1f} MB")

    print(f"\n  Wrote {n_done} file(s).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
