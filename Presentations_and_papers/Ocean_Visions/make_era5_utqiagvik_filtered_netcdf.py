#!/usr/bin/env python3
"""Write the filtered ERA5 record behind Ocean Visions figures 1, 2 and 4 to netCDF.

WHAT THIS FILE IS
=================
Figures 1, 2 and 4 of ``ocean_visions_figures.ipynb`` compare ERA5 against the
DOE ARM North Slope of Alaska observations at Utqiagvik. All three read the
single ERA5 grid cell that holds the ARM central facility, and all three use
the precipitation-filtered run (``A_precip`` for figures 1-2 -- figure 2 pairs
it with Genie's precipitation-filtered monthly counts, ``MONTHLY_OBS_SOURCE =
"noprecip"`` -- and ``E`` for figure 4, built with the same settings).

This script writes every hour of that cell that survives the figures' filters,
with EVERY variable in the raw single-level archive at that cell, plus the
derived quantities the classification used and a phase label per hour:

    1. season window      1 Oct - 31 Mar, seasons 2014/15 .. 2024/25 (11)
    2. overcast           tcc >= 0.95
    3. condensate         LWP > 0.01 g m-2 or IWP > 0.01 g m-2 (CWP > 0 after
                          each species is floored at its own minimum)
    4. not precipitating  tp < 0.05 mm hr-1

and then, within the kept hours, by the liquid share of the floored cloud
water path, f_liq = LWP / (LWP + IWP):

    liquid containing     IWP/CWP < 0.90    <=>  f_liq >  0.10
    liquid only (subset)  LWP/CWP >= 0.90   <=>  f_liq >= 0.90
    ice only              IWP/CWP >= 0.90   <=>  f_liq <= 0.10

Liquid containing = liquid only + mixed phase; liquid containing and ice only
partition the kept hours exactly.

NOTHING HERE IS RE-DERIVED
==========================
The archive loader, the site-cell picker, the season calendar and the phase
classifier are imported from ``ERA5/surface_energy_budget`` -- the same
functions ``lwph.prepare`` and ``fit_cloud_thresholds.extract_site_series``
call -- and the thresholds are copied from the notebook's settings cell
(``COMMON``, ``PRECIP``, ``ICE_FRACTION_MIN_COMPARISON``). ``--verify``
re-runs the figures' own pipeline and checks, hour by hour, that this file's
liquid-containing and ice-only hours are the figures' hours.

ONE CONVENTION TO KNOW
======================
Figures 1-2 fold the overcast hours that hold NO condensate above the floors
into the ice-only bar, so that liquid + ice sums to the overcast total
(``fit_cloud_thresholds._phase_hit``, ``want="ice"``). Step 3 above excludes
those hours from this file, which is the filter as described for the
collaborator; the script prints how many there were so the bar heights can be
reconciled. Figure 4 (event durations) uses liquid-containing hours only and
is unaffected.

Usage
-----
    python make_era5_utqiagvik_filtered_netcdf.py
    python make_era5_utqiagvik_filtered_netcdf.py --verify      # + exact check, ~10 min
    python make_era5_utqiagvik_filtered_netcdf.py --out /path/to/file.nc
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------
# Make the analysis modules importable
# ----------------------------------------------------------------------------
# Same anchoring as the notebook's setup cell: walk up to the repository root
# (the directory holding .git) rather than trusting the working directory, so
# the script runs from anywhere.
def repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.git``."""
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError(f"no .git directory above {start}")


HERE = Path(__file__).resolve().parent
REPO_ROOT = repo_root(HERE)
SEB_DIR = REPO_ROOT / "ERA5" / "surface_energy_budget"
if str(SEB_DIR) not in sys.path:
    sys.path.insert(0, str(SEB_DIR))

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402  args, loader
from cloud_classification import fraction_phase_masks  # noqa: E402  phase rule
from plot_surface_class_timeseries import (  # noqa: E402
    SITE_LAT,          # deg N, ARM NSA central facility
    SITE_LON,          # deg E (negative = W)
    season_layout,     # time step -> (season, day-of-season slot)
    select_seasons,    # which seasons the run keeps
    site_cell_mask,    # nearest ERA5 cell to the facility
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402

# open_mfdataset warns about join/compat defaults changing; load_seb_data pins
# both explicitly, so the warning is noise (same filter as the notebook).
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------
# The settings of figures 1, 2 and 4
# ----------------------------------------------------------------------------
# Copied from ocean_visions_figures.ipynb, "Shared settings" cell: COMMON,
# PRECIP and ICE_FRACTION_MIN_COMPARISON (the ARM-cell value, 0.90). Figure 4's
# extent run E uses exactly these too (cse.prepare_extent(**COMMON, **PRECIP,
# ice_fraction_min=ICE_FRACTION_MIN_COMPARISON)). If the notebook changes,
# change these to match -- --verify will fail loudly if they drift apart.
SETTINGS = dict(
    region="barrow",
    years=tuple(range(2014, 2025)),   # season START years: 2014/15 .. 2024/25
    season_start=(10, 1),             # (month, day), inclusive
    season_end=(3, 31),               # (month, day), inclusive; wraps the year
    phase_mode="fraction",
    liquid_fraction_min=0.90,         # LWP/CWP >= this: liquid only
    ice_fraction_min=0.90,            # IWP/CWP >= this: ice only
    min_lwp=0.01,                     # g m-2; liquid at or below counts as absent
    min_iwp=0.01,                     # g m-2; ice at or below counts as absent
    min_cloud_fraction=0.95,          # overcast gate: tcc >= this
    lsm_tol=0.01,                     # unused at one cell; kept for --verify parity
    storage="local",                  # data/ beside the modules
    no_precip=True,                   # precipitation filter on
    precip_var="rate",                # filter on tp (hourly accumulation)
    precip_rate_max=0.05,             # mm hr-1; an hour AT or above is dropped
)

DEFAULT_OUT = HERE / "era5_utqiagvik_OV_figs1-2-4_filtered_2014-2025.nc"

# Integer phase label written per hour. Values are CF flag_values; the
# meanings string below is the matching flag_meanings attribute.
PHASE_LIQUID_ONLY = 1
PHASE_MIXED = 2
PHASE_ICE_ONLY = 3


def build_args(overrides: dict) -> argparse.Namespace:
    """The module's own argument namespace with the figure settings applied.

    Starting from ``lwph.parse_args([])`` means every option this script does
    not name (block size, liquid variable = tclw, ...) takes the same default
    the figures took.
    """
    args = lwph.parse_args([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)
    return args


# ----------------------------------------------------------------------------
# Read the ARM cell
# ----------------------------------------------------------------------------
def read_site_cell(args: argparse.Namespace) -> tuple[xr.Dataset, dict]:
    """Every in-window hour of the selected seasons at the ARM grid cell.

    Returns the dataset (all archive variables, dims ``(valid_time,)``) and a
    dict of bookkeeping: the season start year of each hour and the cell
    centre.
    """
    region_dir = resolve_region_dir(args)
    # Opens lazily; nothing is read until .load() below.
    ds = load_seb_data(args.region, None, None, region_dir.parent)

    # Season bookkeeping from the time axis alone -- the same objects the
    # figures used, so this file and the figures agree on which hours exist.
    layout = season_layout(ds, args)
    keep_idx, used, _label = select_seasons(layout, args)

    # Hours inside the Oct-Mar window AND in one of the selected seasons.
    # Identical to use_step in fit_cloud_thresholds.extract_site_series.
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    # Nearest-neighbour ERA5 cell to the ARM facility (exactly one True).
    site_mask, cell_lat_deg, cell_lon_deg = site_cell_mask(ds)
    i_lat, j_lon = (int(v) for v in np.argwhere(site_mask)[0])

    # Point selection first, then load: only the one cell's column of each
    # chunk is kept in memory.
    t_sel = np.flatnonzero(use_step)
    print(f"  Reading {t_sel.size:,} hours x {len(ds.data_vars)} variables at "
          f"cell ({cell_lat_deg:.2f} N, {cell_lon_deg:.2f} E) ...", flush=True)
    cell = ds.isel(latitude=i_lat, longitude=j_lon, valid_time=t_sel).load()

    # Season START year of each kept hour (2014 = the 2014/15 season).
    season_year = np.asarray(layout["seasons"])[s_idx[t_sel]]
    return cell, {
        "season_year": season_year.astype(np.int32),
        "seasons_used": list(used),
        "cell_lat_deg": cell_lat_deg,
        "cell_lon_deg": cell_lon_deg,
    }


# ----------------------------------------------------------------------------
# Classify, exactly as the figures do
# ----------------------------------------------------------------------------
def classify(cell: xr.Dataset, args: argparse.Namespace) -> dict:
    """Per-hour masks and derived paths for the filters and the phase split.

    Mirrors ``fit_cloud_thresholds.extract_site_series`` + ``_phase_hit``:
    LWP and IWP are each floored at their own minimum (a species at or below
    its floor contributes nothing to CWP), then the phase is the floored
    share via ``cloud_classification.fraction_phase_masks``.
    """
    tcc = cell["tcc"].values                             # (time,) 0-1
    lwp_g_m2 = cell[args.liquid_var].values * 1000.0     # kg m-2 -> g m-2
    iwp_g_m2 = cell["tciw"].values * 1000.0              # kg m-2 -> g m-2
    # tp is the accumulation over the hour ENDING at valid_time, in metres of
    # water; 1 m h-1 = 1000 mm h-1.
    precip_rate_mm_hr = cell["tp"].values * 1000.0

    valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)

    # Liquid only / mixed / ice only, plus the floored CWP and liquid share.
    f = fraction_phase_masks(
        lwp_g_m2, iwp_g_m2,
        args.liquid_fraction_min, args.ice_fraction_min,
        args.min_lwp, args.min_iwp,
    )
    has_condensate = valid & (f["cwp_g"] > 0.0)

    # The floored paths, as the classifier saw them.
    with np.errstate(invalid="ignore"):
        lwp_eff_g_m2 = np.where(valid & (lwp_g_m2 > args.min_lwp), lwp_g_m2, 0.0)
        iwp_eff_g_m2 = np.where(valid & (iwp_g_m2 > args.min_iwp), iwp_g_m2, 0.0)

    overcast = valid & (tcc >= args.min_cloud_fraction)
    raining = np.isfinite(precip_rate_mm_hr) & (precip_rate_mm_hr >= args.precip_rate_max)

    keep = overcast & has_condensate & ~raining
    liquid_containing = keep & ~f["ice"]                 # liquid only + mixed
    return {
        "valid": valid,
        "overcast": overcast,
        "raining": raining,
        "has_condensate": has_condensate,
        "keep": keep,
        "liquid_only": keep & f["liquid"],
        "mixed": keep & f["mixed"],
        "ice_only": keep & f["ice"],
        "liquid_containing": liquid_containing,
        "lwp_eff_g_m2": lwp_eff_g_m2,
        "iwp_eff_g_m2": iwp_eff_g_m2,
        "cwp_g_m2": f["cwp_g"],
        "liquid_fraction": f["liquid_fraction"],
        "precip_rate_mm_hr": precip_rate_mm_hr,
        # Overcast, dry, but no condensate above the floors: figures 1-2 count
        # these as ice only; this file excludes them.
        "overcast_no_condensate": overcast & ~raining & valid & ~has_condensate,
    }


# ----------------------------------------------------------------------------
# Assemble the output
# ----------------------------------------------------------------------------
def git_describe() -> str:
    """Short commit hash of the repository, with '-dirty' if uncommitted."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_output(cell: xr.Dataset, book: dict, m: dict,
                 args: argparse.Namespace) -> xr.Dataset:
    """Kept hours only: every archive variable plus derived fields and labels."""
    k = m["keep"]

    # --- 1. Every raw archive variable, unchanged, at the kept hours. --------
    # Drop the scalar GRIB coordinate 'number' (ensemble member, always 0)
    # and the per-cell lat/lon coords, which are re-attached as scalars below.
    out = cell.isel(valid_time=np.flatnonzero(k))
    out = out.drop_vars([c for c in ("number", "latitude", "longitude")
                         if c in out.coords])
    for v in out.data_vars:
        # Strip on-disk encodings inherited from the source files (chunk
        # shapes for a 3-D grid, packing) so the 1-D output encodes cleanly.
        out[v].encoding = {}
        out[v].attrs["source"] = "ERA5 single levels, unmodified value at the ARM grid cell"

    # --- 2. Season bookkeeping. ---------------------------------------------
    out["season_start_year"] = ("valid_time", book["season_year"][k])
    out["season_start_year"].attrs = {
        "long_name": "start year of the Oct-Mar cold season",
        "comment": "2014 = the 2014/15 season (1 Oct 2014 - 31 Mar 2015)",
        "units": "1",
    }

    # --- 3. Derived quantities the classification used. ---------------------
    derived = {
        "lwp_floored_g_m2": (m["lwp_eff_g_m2"], {
            "long_name": "cloud liquid water path, floored",
            "units": "g m-2",
            "comment": (f"1000 * {args.liquid_var}; set to 0 where <= "
                        f"{args.min_lwp:g} g m-2 (liquid treated as absent)"),
        }),
        "iwp_floored_g_m2": (m["iwp_eff_g_m2"], {
            "long_name": "cloud ice water path, floored",
            "units": "g m-2",
            "comment": f"1000 * tciw; set to 0 where <= {args.min_iwp:g} g m-2",
        }),
        "cwp_g_m2": (m["cwp_g_m2"], {
            "long_name": "total condensed (cloud) water path",
            "units": "g m-2",
            "comment": "lwp_floored_g_m2 + iwp_floored_g_m2",
        }),
        "liquid_fraction": (m["liquid_fraction"], {
            "long_name": "liquid share of the cloud water path, LWP/CWP",
            "units": "1",
            "comment": "from the floored paths; ice share = 1 - liquid_fraction",
        }),
        "precip_rate_mm_hr": (m["precip_rate_mm_hr"], {
            "long_name": "total precipitation rate",
            "units": "mm hr-1",
            "comment": "1000 * tp; tp is the accumulation over the hour ending at valid_time",
        }),
    }
    for name, (arr, attrs) in derived.items():
        out[name] = ("valid_time", np.asarray(arr[k], dtype=np.float32))
        out[name].attrs = attrs

    # --- 4. Phase labels. ---------------------------------------------------
    phase = np.zeros(k.sum(), dtype=np.int8)
    phase[m["liquid_only"][k]] = PHASE_LIQUID_ONLY
    phase[m["mixed"][k]] = PHASE_MIXED
    phase[m["ice_only"][k]] = PHASE_ICE_ONLY
    if (phase == 0).any():
        raise AssertionError("a kept hour fell in no phase category")
    out["phase"] = ("valid_time", phase)
    out["phase"].attrs = {
        "long_name": "cloud phase from the liquid share of the cloud water path",
        "flag_values": np.array([PHASE_LIQUID_ONLY, PHASE_MIXED, PHASE_ICE_ONLY],
                                dtype=np.int8),
        "flag_meanings": "liquid_only mixed_phase ice_only",
        "comment": (f"liquid_only: LWP/CWP >= {args.liquid_fraction_min:g}; "
                    f"ice_only: IWP/CWP >= {args.ice_fraction_min:g}; "
                    f"mixed_phase: everything between. "
                    f"Liquid containing = liquid_only OR mixed_phase."),
    }
    # Convenience booleans (stored as int8 0/1; netCDF has no bool type).
    for name, mask, desc in (
        ("is_liquid_containing", m["liquid_containing"],
         f"liquid containing: IWP/CWP < {args.ice_fraction_min:g} "
         f"(liquid only + mixed phase); the red bars/boxes of figures 1-2 and "
         f"the events of figure 4"),
        ("is_liquid_only", m["liquid_only"],
         f"liquid only: LWP/CWP >= {args.liquid_fraction_min:g}; a subset of "
         f"liquid containing"),
        ("is_ice_only", m["ice_only"],
         f"ice only: IWP/CWP >= {args.ice_fraction_min:g}; the blue bars of "
         f"figure 1"),
    ):
        out[name] = ("valid_time", mask[k].astype(np.int8))
        out[name].attrs = {"long_name": desc, "flag_values": np.array([0, 1], dtype=np.int8),
                           "flag_meanings": "false true"}

    # --- 5. Where this cell is. ---------------------------------------------
    out = out.assign_coords(
        latitude=np.float32(book["cell_lat_deg"]),
        longitude=np.float32(book["cell_lon_deg"]),
    )
    out["latitude"].attrs = {"units": "degrees_north",
                             "long_name": "latitude of the ERA5 grid-cell centre"}
    out["longitude"].attrs = {"units": "degrees_east",
                              "long_name": "longitude of the ERA5 grid-cell centre"}
    out["valid_time"].attrs.update({
        "long_name": "ERA5 valid time (UTC)",
        "comment": ("instantaneous fields are at valid_time; tp and the "
                    "time-mean fluxes (ms*) are over the hour ending at valid_time"),
    })

    # --- 6. Provenance and the exact filter, in the global attributes. ------
    s0, s1 = book["seasons_used"][0], book["seasons_used"][-1]
    out.attrs = {
        "title": "ERA5 hourly single-level data at the ERA5 grid cell containing "
                 "Utqiagvik, Alaska (DOE ARM NSA), filtered for overcast, "
                 "non-precipitating, cloud-water-bearing cold-season hours",
        "source": "ERA5 hourly data on single levels (Hersbach et al. 2020, "
                  "doi:10.1002/qj.3803), via the Copernicus Climate Data Store",
        "site": (f"ARM NSA central facility {SITE_LAT:.3f} N, {SITE_LON:.3f} E; "
                 f"nearest ERA5 0.25-deg cell centre {book['cell_lat_deg']:.2f} N, "
                 f"{book['cell_lon_deg']:.2f} E"),
        "seasons": f"{s0}/{s0 + 1} - {s1}/{s1 + 1} ({len(book['seasons_used'])} seasons)",
        "season_window": (f"{args.season_start[0]:02d}-{args.season_start[1]:02d} to "
                          f"{args.season_end[0]:02d}-{args.season_end[1]:02d}, inclusive"),
        "filter_1_overcast": f"tcc >= {args.min_cloud_fraction:g}",
        "filter_2_condensate": (f"LWP > {args.min_lwp:g} g m-2 or IWP > {args.min_iwp:g} "
                                f"g m-2 (each species floored at its own minimum; "
                                f"CWP of the floored paths > 0)"),
        "filter_3_no_precipitation": f"tp < {args.precip_rate_max:g} mm hr-1",
        "phase_liquid_containing": f"IWP/CWP < {args.ice_fraction_min:g}",
        "phase_liquid_only": f"LWP/CWP >= {args.liquid_fraction_min:g}",
        "phase_ice_only": f"IWP/CWP >= {args.ice_fraction_min:g}",
        "liquid_variable": f"{args.liquid_var} (total column cloud liquid water)",
        "n_hours_kept": int(k.sum()),
        "n_hours_in_window": int(m["valid"].size),
        "note_overcast_no_condensate": (
            f"{int(m['overcast_no_condensate'].sum())} overcast, non-precipitating "
            f"hours with no condensate above the floors are excluded here; the "
            f"seasonal ice-only bars of the source figures count them as ice only"),
        "note_gaps": ("hours missing from the source archive are simply absent; "
                      "check valid_time for gaps before computing durations"),
        "history": (f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: "
                    f"{Path(__file__).name} (repo {git_describe()})"),
        "contact": "Andrew Buggee, Scripps Institution of Oceanography, UC San Diego",
    }
    return out


def print_summary(out: xr.Dataset, m: dict) -> None:
    """Hours removed at each step and kept per season and phase."""
    n_win = m["valid"].size
    n_ovc = int(m["overcast"].sum())
    n_cond = int((m["overcast"] & m["has_condensate"]).sum())
    n_keep = int(m["keep"].sum())
    print(f"\n  in-window hours            {n_win:7,d}")
    print(f"  overcast (tcc gate)        {n_ovc:7,d}")
    print(f"   + condensate above floors {n_cond:7,d}")
    print(f"   + not precipitating       {n_keep:7,d}   <- written")
    print(f"  (overcast, dry, no condensate -- excluded, counted as ice only in "
          f"figs 1-2: {int(m['overcast_no_condensate'].sum())})")

    seasons = np.unique(out["season_start_year"].values)
    print(f"\n  {'season':>9s} {'kept':>6s} {'liq-cont':>9s} {'liq-only':>9s} "
          f"{'mixed':>6s} {'ice-only':>9s}")
    for s in seasons:
        sel = out["season_start_year"].values == s
        ph = out["phase"].values[sel]
        print(f"  {s}/{(s + 1) % 100:02d} {sel.sum():6d} "
              f"{int(out['is_liquid_containing'].values[sel].sum()):9d} "
              f"{int((ph == PHASE_LIQUID_ONLY).sum()):9d} "
              f"{int((ph == PHASE_MIXED).sum()):6d} "
              f"{int((ph == PHASE_ICE_ONLY).sum()):9d}")


# ----------------------------------------------------------------------------
# Optional: exact check against the figures' own pipeline
# ----------------------------------------------------------------------------
def verify_against_figures(out: xr.Dataset, args: argparse.Namespace) -> None:
    """Hour-by-hour comparison with ``lwph.prepare`` + ``extract_site_series``.

    Runs the figures' pipeline with the same settings (several minutes: it
    streams the whole domain) and requires the SAME timestamps in each
    category. Figure 1/2's ice-only population includes the overcast hours
    with no condensate, so the comparison restricts it to hours that hold
    cloud water, which is this file's definition.
    """
    import fit_cloud_thresholds as fit

    print("\n  --verify: running lwph.prepare with the figure settings ...")
    A = lwph.prepare(args=args)
    S = fit.extract_site_series(A)
    tcc0 = float(args.min_cloud_fraction)
    ifm0 = float(args.ice_fraction_min)

    liq_ref = fit._phase_hit(S, tcc0, ifm0, True, "liquid")
    ice_ref = fit._phase_hit(S, tcc0, ifm0, True, "ice") & S["has_cloud"]

    t = out["valid_time"].values
    ok = True
    for label, ref, mine in (
        ("liquid containing", liq_ref, out["is_liquid_containing"].values == 1),
        ("ice only", ice_ref, out["is_ice_only"].values == 1),
    ):
        a = np.sort(S["time"][ref])
        b = np.sort(t[mine])
        same = a.size == b.size and bool(np.all(a == b))
        print(f"    {label:18s} figures {a.size:6,d} h | file {b.size:6,d} h | "
              f"{'IDENTICAL' if same else 'MISMATCH'}")
        ok &= same
    if not ok:
        raise AssertionError("the file does not reproduce the figures' hours")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output netCDF path (default {DEFAULT_OUT.name} beside this script)")
    p.add_argument("--verify", action="store_true",
                   help="also run the figures' pipeline and require identical hours (~10 min)")
    opts = p.parse_args(argv)

    args = build_args(SETTINGS)
    print("=" * 72)
    print("ERA5 at the Utqiagvik cell, filtered as Ocean Visions figures 1, 2, 4")
    print("=" * 72)

    cell, book = read_site_cell(args)
    m = classify(cell, args)
    out = build_output(cell, book, m, args)
    print_summary(out, m)

    if opts.verify:
        verify_against_figures(out, args)

    # zlib level 4 on every variable: a ~50k-hour 1-D record compresses to a
    # few MB. float32 is kept as in the archive.
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    enc["valid_time"] = {"units": "hours since 1970-01-01 00:00:00",
                         "calendar": "proleptic_gregorian", "dtype": "int64"}
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(opts.out, encoding=enc, format="NETCDF4")
    size_mb = opts.out.stat().st_size / 1e6
    print(f"\n  -> {opts.out}  ({size_mb:.1f} MB, {out.sizes['valid_time']:,} hours, "
          f"{len(out.data_vars)} variables)")


if __name__ == "__main__":
    main()
