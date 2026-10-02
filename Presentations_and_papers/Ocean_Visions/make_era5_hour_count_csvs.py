#!/usr/bin/env python3
"""CSV tables of ERA5 cloud cell-hours behind Ocean Visions figures 1, 2 and 4.

WHAT IT WRITES
==============
Counts of ERA5 cell-hours by cloud phase, for the ARM grid cell at Utqiagvik
(UTQ) and for each of the five surface classes of the Barrow domain, at three
time resolutions:

    per season   rows = 11 seasons,  cols = liquid containing / liquid only / ice only
                 ERA5_totalHours_perSeason_2014_2025_<class>.csv           (6 files)
    per month    rows = Oct..Mar,    cols = the 11 seasons; one file per phase
                 ERA5_totalHours_perMonth_2014_2025_<phase>_<class>.csv    (18 files)
    per day      rows = every day,   cols = date + the three phases
                 ERA5_totalHours_perDay_2014_2025_<class>.csv              (6 files)

    <class> in UTQ, Land, Coastal, OpenOcean, MarginalSeaIce, PackIce
    <phase> in liquidContaining, liquidOnly, iceOnly

plus, in ``normalization/``, the matching totals of ALL valid cell-hours in
each class (before any cloud filter), so a count can be turned into a
fraction of the class's area-time; and ``readME.txt``.

THE FILTER
==========
Identical to ``make_era5_utqiagvik_filtered_netcdf.py``, whose ``SETTINGS``
are imported rather than copied: overcast (tcc >= 0.95), cloud water above
the 0.01 g m-2 floors, not precipitating (tp < 0.05 mm hr-1), 1 Oct - 31 Mar
of 2014/15 .. 2024/25. Phase from the floored liquid share LWP/CWP:
liquid containing IWP/CWP < 0.90, liquid only LWP/CWP >= 0.90, ice only
IWP/CWP >= 0.90.

COUNTS, NOT WEIGHTED MEANS
==========================
Every entry is a plain count of (cell, hour) pairs: at UTQ that is hours; for
a class it is summed over every cell that was in that class in that hour.
Two consequences worth stating in the readME:

* The class figures in the notebook show cos(latitude)-weighted occupancy
  scaled to hours per season -- "hours at a typical cell of the class" --
  not totals. Totals here grow with the class's size, which for the ocean
  classes changes hour by hour as the ice edge moves (siconc is hourly).
  The normalization/ files are the denominators for turning a total into an
  (unweighted) occupancy.
* A cell's class is decided hour by hour, so one cell can contribute to open
  ocean in October and pack ice in December.

HOW THE NUMBERS ARE MADE
========================
One streaming pass over the archive (about 4 minutes) accumulates counts per
(day, class, category); the monthly and seasonal tables are sums of the
daily ones, so the three resolutions cannot disagree. The loader, land-sea
mask, surface classifier, season calendar and phase classifier are the
modules' own functions -- the same ones ``lwph.prepare`` calls.

``--verify`` re-runs ``lwph.prepare`` with ``cell_weighting="uniform"``, under
which its monthly accumulators ARE plain cell-hour counts, and requires every
monthly entry to match exactly (another ~4 minutes).

Usage
-----
    python make_era5_hour_count_csvs.py
    python make_era5_hour_count_csvs.py --verify
    python make_era5_hour_count_csvs.py --out-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

# The netCDF script holds the figure settings and puts the analysis modules on
# sys.path; importing it is what keeps the two products on one definition.
from make_era5_utqiagvik_filtered_netcdf import HERE, SETTINGS, build_args, git_describe

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from cloud_classification import fraction_phase_masks  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    CLASS_ORDER,
    UNCLASSIFIED,
    align_lsm_to_grid,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_OUT_DIR = HERE / "era5_hour_count_csv"

# ----------------------------------------------------------------------------
# Axes of the count array
# ----------------------------------------------------------------------------
# Output name of each spatial slot. UTQ is NOT a sixth class: its cell is also
# counted in whichever class it falls in that hour (land or coastal).
SLOT_NAMES: tuple[str, ...] = ("UTQ", "Land", "Coastal", "OpenOcean",
                               "MarginalSeaIce", "PackIce")
# Module class name behind each class slot (pack ice is "sea_ice" there:
# siconc > 0.95).
SLOT_CLASS = {"Land": "land", "Coastal": "coastal", "OpenOcean": "open_ocean",
              "MarginalSeaIce": "marginal_ice", "PackIce": "sea_ice"}
I_UTQ = 0

# Categories counted per (day, slot). The last is the denominator: every
# valid cell-hour of the slot, before the cloud filters.
# overcast_no_condensate (cloudy, dry, but no cloud water above the floors)
# is not written to any table; it is counted only so the readME can state how
# far figure 1's ice-only bars (which include it) differ from ice_only here.
CATS: tuple[str, ...] = ("liquid_containing", "liquid_only", "ice_only",
                         "all_valid", "overcast_no_condensate")
FILE_TAG = {"liquid_containing": "liquidContaining", "liquid_only": "liquidOnly",
            "ice_only": "iceOnly"}
PHASE_CATS = CATS[:3]

# '1 Oct. 2014'. All six season months abbreviate to three letters + '.'.
MONTH_ABBR = {1: "Jan.", 2: "Feb.", 3: "Mar.", 4: "Apr.", 5: "May", 6: "Jun.",
              7: "Jul.", 8: "Aug.", 9: "Sep.", 10: "Oct.", 11: "Nov.", 12: "Dec."}


def season_label(y: int) -> str:
    """2014 -> '2014/15'."""
    return f"{y}/{(y + 1) % 100:02d}"


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def count_cell_hours(args: argparse.Namespace) -> dict:
    """Counts of cell-hours per (day, slot, category) over the whole domain.

    Returns ``counts`` shaped ``(n_day, n_slot, n_cat)`` (int64) with the day
    axis described by ``dates`` (datetime64[D], UTC calendar days),
    ``day_season`` (season start year) and ``day_month`` (calendar month).
    """
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)

    # Land fraction on the archive's grid; time-invariant.
    lsm = align_lsm_to_grid(
        load_land_sea_mask(args.region,
                           lwph.resolve_data_root(args.storage, args.data_root),
                           args.mask_grid),
        ds,
    )

    # Season bookkeeping from the time axis only (same as the figures).
    layout = season_layout(ds, args)
    keep_idx, used, _label = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    # Day axis: every UTC calendar date holding at least one kept hour.
    times = np.asarray(ds["valid_time"].values)
    step_date = times.astype("datetime64[D]")
    dates, day_of_kept = np.unique(step_date[use_step], return_inverse=True)
    day_of_step = np.full(times.size, -1, dtype=np.intp)
    day_of_step[use_step] = day_of_kept
    season_years = np.asarray(layout["seasons"])
    # Season of each day from any of its hours (a day never straddles two).
    day_season = np.zeros(dates.size, dtype=int)
    day_season[day_of_kept] = season_years[s_idx[use_step]]

    site_mask, site_lat, site_lon = site_cell_mask(ds)
    n_day, n_slot, n_cat = dates.size, len(SLOT_NAMES), len(CATS)
    counts = np.zeros((n_day, n_slot, n_cat), dtype=np.int64)
    n_unclassified = 0

    # Output slot of each module class code. Unclassified cells go to a
    # scratch slot (index n_slot) that is dropped after the bincount.
    slot_of_code = np.empty(len(CLASS_ORDER), dtype=np.intp)
    for slot_name, cls in SLOT_CLASS.items():
        slot_of_code[CLASS_CODES[cls]] = SLOT_NAMES.index(slot_name)
    SCRATCH = n_slot                                   # unclassified

    read_vars = ["tcc", args.liquid_var, "tciw", "tp", "siconc"]
    print(f"  Streaming {int(use_step.sum()):,} hours x "
          f"{ds.sizes['latitude']} x {ds.sizes['longitude']} cells ...", flush=True)
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        sl = slice(i0, i0 + block.sizes["valid_time"])
        keep = use_step[sl]
        if not keep.any():
            continue
        day = day_of_step[sl][keep]                              # (t,)

        # --- surface class of every cell-hour ------------------------------
        classes = classify_cells(lsm, block["siconc"].values, args.lsm_tol,
                                 args.open_ocean_max_siconc,
                                 args.sea_ice_min_siconc,
                                 args.land_max_siconc)[keep]     # (t, lat, lon)
        n_unclassified += int((classes == UNCLASSIFIED).sum())
        slot = np.where(classes == UNCLASSIFIED, SCRATCH,
                        slot_of_code[np.clip(classes, 0, None)])

        # --- the filters and the phase, exactly as the netCDF script -------
        tcc = block["tcc"].values[keep]                          # 0-1
        lwp_g_m2 = block[args.liquid_var].values[keep] * 1000.0  # kg -> g m-2
        iwp_g_m2 = block["tciw"].values[keep] * 1000.0
        rate_mm_hr = block["tp"].values[keep] * 1000.0           # m h-1 -> mm h-1
        valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
        f = fraction_phase_masks(lwp_g_m2, iwp_g_m2,
                                 args.liquid_fraction_min, args.ice_fraction_min,
                                 args.min_lwp, args.min_iwp)
        raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
        kept = (valid & (tcc >= args.min_cloud_fraction)
                & (f["cwp_g"] > 0.0) & ~raining)
        masks = {
            "liquid_containing": kept & (f["liquid"] | f["mixed"]),
            "liquid_only": kept & f["liquid"],
            "ice_only": kept & f["ice"],
            "all_valid": valid,
            "overcast_no_condensate": (valid & (tcc >= args.min_cloud_fraction)
                                       & (f["cwp_g"] == 0.0) & ~raining),
        }

        # --- accumulate: one bincount per category over (day, slot) --------
        # Flat index day * (n_slot + 1) + slot over every cell-hour.
        flat = (day[:, None, None] * (n_slot + 1) + slot).ravel()
        for ci, cat in enumerate(CATS):
            m = masks[cat]
            c = np.bincount(flat[m.ravel()], minlength=n_day * (n_slot + 1))
            counts[:, :, ci] += c.reshape(n_day, n_slot + 1)[:, :n_slot]
            # UTQ: the one site cell, added on top of its class slot.
            counts[:, I_UTQ, ci] += np.bincount(
                day[m[:, site_mask][:, 0]], minlength=n_day)

    if n_unclassified:
        print(f"  !! {n_unclassified:,} unclassified cell-hours (not in any class)")
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return {"counts": counts, "dates": dates, "day_season": day_season,
            "day_month": months, "seasons": list(used),
            "keep_idx": list(keep_idx), "n_unclassified": n_unclassified,
            "site_lat": site_lat, "site_lon": site_lon,
            "grid": (ds.sizes["latitude"], ds.sizes["longitude"])}


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
SEASON_MONTHS = (10, 11, 12, 1, 2, 3)


def per_season(C: dict, slot: int, cats=PHASE_CATS) -> pd.DataFrame:
    """Rows = seasons, columns = categories."""
    rows = []
    for y in C["seasons"]:
        d = C["day_season"] == y
        rows.append([season_label(y)]
                    + [int(C["counts"][d, slot, CATS.index(c)].sum()) for c in cats])
    return pd.DataFrame(rows, columns=["season", *cats])


def per_month(C: dict, slot: int, cat: str) -> pd.DataFrame:
    """Rows = Oct..Mar, columns = seasons."""
    ci = CATS.index(cat)
    data = {"month": [MONTH_ABBR[m].rstrip(".") for m in SEASON_MONTHS]}
    for y in C["seasons"]:
        d_s = C["day_season"] == y
        data[season_label(y)] = [
            int(C["counts"][d_s & (C["day_month"] == m), slot, ci].sum())
            for m in SEASON_MONTHS]
    return pd.DataFrame(data)


def per_day(C: dict, slot: int, cats=PHASE_CATS) -> pd.DataFrame:
    """Rows = every day of the record, first column the date."""
    dt = pd.to_datetime(C["dates"])
    labels = [f"{d.day} {MONTH_ABBR[d.month]} {d.year}" for d in dt]
    out = pd.DataFrame({"date": labels})
    for c in cats:
        out[c] = C["counts"][:, slot, CATS.index(c)]
    return out


def write_all(C: dict, out_dir: Path) -> list[Path]:
    """Every CSV, the normalization tables and nothing else."""
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = out_dir / "normalization"
    norm_dir.mkdir(exist_ok=True)
    y0, y1 = C["seasons"][0], C["seasons"][-1] + 1          # 2014, 2025
    span = f"{y0}_{y1}"
    written = []

    def save(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False)
        written.append(path)

    for s, name in enumerate(SLOT_NAMES):
        save(per_season(C, s), out_dir / f"ERA5_totalHours_perSeason_{span}_{name}.csv")
        for cat in PHASE_CATS:
            save(per_month(C, s, cat),
                 out_dir / f"ERA5_totalHours_perMonth_{span}_{FILE_TAG[cat]}_{name}.csv")
        save(per_day(C, s), out_dir / f"ERA5_totalHours_perDay_{span}_{name}.csv")

    # Denominators: all valid cell-hours of each slot, one column per slot.
    ai = CATS.index("all_valid")
    season_tbl = pd.DataFrame({"season": [season_label(y) for y in C["seasons"]]})
    for s, name in enumerate(SLOT_NAMES):
        season_tbl[name] = [int(C["counts"][C["day_season"] == y, s, ai].sum())
                            for y in C["seasons"]]
    save(season_tbl, norm_dir / f"ERA5_allValidCellHours_perSeason_{span}.csv")
    for s, name in enumerate(SLOT_NAMES):
        save(per_month(C, s, "all_valid"),
             norm_dir / f"ERA5_allValidCellHours_perMonth_{span}_{name}.csv")
    day_tbl = per_day(C, I_UTQ, cats=())
    for s, name in enumerate(SLOT_NAMES):
        day_tbl[name] = C["counts"][:, s, ai]
    save(day_tbl, norm_dir / f"ERA5_allValidCellHours_perDay_{span}.csv")
    return written


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def self_checks(C: dict) -> None:
    """Internal consistency that must hold whatever the data."""
    k = C["counts"]
    lc, lo, io, av = (k[..., CATS.index(c)] for c in CATS[:4])
    assert (lo <= lc).all(), "liquid only exceeds liquid containing"
    assert ((lc + io) <= av).all(), "phases exceed the valid cell-hours"
    # UTQ is one cell: at most 24 h per day of anything.
    assert (av[:, I_UTQ] <= 24).all(), "more than 24 h in a UTQ day"
    # The five classes cannot hold more cell-hours than the grid has.
    n_lat, n_lon = C["grid"]
    n_hours = int(av[:, I_UTQ].sum())
    cap = n_lat * n_lon * n_hours
    assert int(av[:, 1:].sum()) + C["n_unclassified"] <= cap
    print(f"  checks OK: {int(av[:, 1:].sum()):,} valid classified cell-hours of "
          f"{cap:,} ({n_lat}x{n_lon} cells x {n_hours:,} h); "
          f"{C['n_unclassified']:,} unclassified")


def verify_against_module(C: dict, args: argparse.Namespace) -> None:
    """Every monthly entry against ``lwph.prepare(cell_weighting='uniform')``.

    With uniform weights the module's ``w_phase_month[season, month, class,
    phase]`` and ``w_valid_month`` are plain cell-hour counts of the same
    filtered population, so they must equal these tables exactly. Its ice
    phase is ``ice``; the overcast-but-empty hours it calls ``none`` are not
    part of ice only here and are not compared.
    """
    print("\n  --verify: lwph.prepare with cell_weighting='uniform' ...")
    args_u = build_args({**SETTINGS, "cell_weighting": "uniform"})
    A = lwph.prepare(args=args_u)
    sec = A.sec
    months_mod = list(sec["months"])                    # calendar months, Oct..Mar
    P = {p: i for i, p in enumerate(lwph.PHASE_ORDER_ACC)}
    slot_code = {"UTQ": sec["site_code"]}
    slot_code.update({n: CLASS_CODES[c] for n, c in SLOT_CLASS.items()})

    worst = 0
    for s, name in enumerate(SLOT_NAMES):
        code = slot_code[name]
        for si_k, (y, si_mod) in enumerate(zip(C["seasons"], C["keep_idx"])):
            for m in SEASON_MONTHS:
                mi = months_mod.index(m)
                sel = (C["day_season"] == y) & (C["day_month"] == m)
                mine = {c: int(C["counts"][sel, s, CATS.index(c)].sum())
                        for c in CATS[:4]}
                ph = sec["w_phase_month"][si_mod, mi, code]
                ref = {
                    "liquid_containing": ph[P["liquid"]] + ph[P["mixed"]],
                    "liquid_only": ph[P["liquid"]],
                    "ice_only": ph[P["ice"]],
                    "all_valid": sec["w_valid_month"][si_mod, mi, code],
                }
                for c in CATS[:4]:
                    worst = max(worst, abs(mine[c] - int(round(ref[c]))))
    print(f"  max |CSV - module| over every class, season, month, category = {worst}")
    if worst != 0:
        raise AssertionError("the CSV counts do not reproduce the module's counts")
    print("  IDENTICAL")


# ----------------------------------------------------------------------------
# Plot colours, for the readME
# ----------------------------------------------------------------------------
# The fig-1 observed-mean line's halo opacity as drawn in the talk:
# ocean_visions_figures.ipynb calls fig_era5_vs_obs_simple_forOV(...,
# halo_alpha=0.7); the module default (lwph.DEFAULT_HALO_ALPHA) is 0.5.
HALO_ALPHA_OV = 0.7


def plot_color_rows() -> list[tuple[str, list[tuple[str, str, str]]]]:
    """(group title, [(label, matplotlib colour spec, note), ...]).

    Every colour is read from the constant the figures themselves use, so the
    readME cannot drift from the plots. Imported here rather than at the top
    because cloud_spatial_extent / cloud_level_wind are only needed for this.
    """
    import cloud_level_wind as clw
    import cloud_spatial_extent as cse
    import plot_surface_class_timeseries as sct
    from cloud_classification import PHASE_COLORS
    from surface_classification import CLASS_COLORS, CLASS_LABELS

    return [
        ("Cloud phase -- figures 1 and 2", [
            ("liquid containing", lwph.GENIE_LIQUID_COLOR,
             "fig 1 red bars; fig 2 ERA5 boxes (solid fill)"),
            ("ice only", lwph.GENIE_ICE_COLOR, "fig 1 blue bars"),
            ("ARM observed mean (fig 1)", lwph.GENIE_LIQUID_COLOR,
             "dashed line, drawn on the halo below"),
            ("  halo under the ARM line", lwph.DEFAULT_HALO_COLOR,
             f"opacity (alpha) {HALO_ALPHA_OV:g}"),
            ("ARM observations (fig 2)", lwph.GENIE_LIQUID_COLOR,
             "dotted box outline, white fill"),
        ]),
        ("Cloud phase -- working figures only (not in the talk)", [
            ("liquid only", PHASE_COLORS["liquid"],
             "SAME colour as OpenOcean below -- pick another if both appear"),
            ("mixed phase", PHASE_COLORS["mixed"], ""),
        ]),
        ("Surface classes -- plot_surface_class_timeseries.ipynb and the talk", [
            *[(tag, CLASS_COLORS[k], f"labelled '{CLASS_LABELS[k]}' in the figures")
              for tag, k in SLOT_CLASS.items()],
            ("ARM site (that notebook)", sct.SITE_COLOR, "dashed line"),
        ]),
        ("Figure 4", [
            ("ARM cell / cloud duration", cse.SITE_COLOR,
             "duration histogram and its axis"),
            ("cloud-level wind speed", clw.CLOUD_WIND_COLOR,
             "wind histogram and its axis"),
        ]),
    ]


def plot_colors_text() -> str:
    """The colour table as fixed-width text: RGB 0-255, RGB 0-1, hex."""
    lines = []
    head = (f"  {'element':<30s} {'RGB (0-255)':<15s} {'RGB (0-1)':<22s} "
            f"{'hex':<8s} {'matplotlib':<10s} note")
    for title, rows in plot_color_rows():
        lines += ["", f"{title}:", head, "  " + "-" * (len(head) + 10)]
        for label, spec, note in rows:
            r, g, b = to_rgb(spec)                         # floats on [0, 1]
            r255 = f"({round(255 * r)}, {round(255 * g)}, {round(255 * b)})"
            r01 = f"({r:.3f}, {g:.3f}, {b:.3f})"
            lines.append(f"  {label:<30s} {r255:<15s} {r01:<22s} "
                         f"{to_hex(spec):<8s} {str(spec):<10s} {note}".rstrip())
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# readME
# ----------------------------------------------------------------------------
def write_readme(C: dict, args: argparse.Namespace, out_dir: Path) -> Path:
    s0, s1 = C["seasons"][0], C["seasons"][-1]
    span = f"{s0}_{s1 + 1}"
    k = C["counts"]
    utq_tot = {c: int(k[:, I_UTQ, CATS.index(c)].sum()) for c in CATS}
    # The ARM season-months figure 2 drops: Genie's flagged list plus the
    # month her precipitation-filtered monthly file leaves blank (Dec 2018),
    # i.e. the union used with MONTHLY_OBS_SOURCE = "noprecip".
    dropped = sorted(set(lwph.GENIE_EXCLUDED_MONTHS) | {(2018, 12)})
    dropped_txt = ", ".join(f"{MONTH_ABBR[m]} {y}" for y, m in dropped)
    text = f"""ERA5 CLOUD CELL-HOUR COUNTS, UTQIAGVIK AND THE BARROW DOMAIN
=============================================================

Andrew Buggee, Scripps Institution of Oceanography, UC San Diego
Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_hour_count_csvs.py (repo {git_describe()})

These tables are the ERA5 statistics behind figures 1, 2 and 4 of the Ocean
Visions research update (28 Sept. 2026), extended to the five surface classes
of the surrounding domain.


1. DATA
-------
ERA5 hourly data on single levels (Hersbach et al. 2020, Q. J. R. Meteorol.
Soc., doi:10.1002/qj.3803), 0.25 deg grid, from the Copernicus Climate Data
Store. Domain: {C['grid'][0]} x {C['grid'][1]} cells, 70-80 N, 165-150 W.
Fields used: total cloud cover (tcc), total column cloud liquid water (tclw),
total column cloud ice water (tciw), total precipitation (tp), sea ice area
fraction (siconc), and the ERA5 land-sea mask (lsm).

UTQ is the single ERA5 cell nearest the DOE ARM North Slope of Alaska central
facility (71.323 N, 156.609 W); its centre is {C['site_lat']:.2f} N,
{abs(C['site_lon']):.2f} W.

Time: hourly, UTC. A "day" is a UTC calendar day.


2. WHICH HOURS ARE COUNTED
--------------------------
Every cell-hour is tested independently. It is counted when ALL hold:

  a. Season window   1 Oct - 31 Mar (inclusive), seasons {season_label(s0)} through
                     {season_label(s1)} ({len(C['seasons'])} seasons).
  b. Cloudy          tcc >= {args.min_cloud_fraction:g}.
  c. Cloud water     LWP = 1000*tclw and IWP = 1000*tciw (g m-2). Each path
                     at or below its floor ({args.min_lwp:g} g m-2 for liquid,
                     {args.min_iwp:g} g m-2 for ice) is set to zero, and the
                     hour must keep a condensed water path CWP = LWP + IWP > 0.
  d. No precipitation  tp < {args.precip_rate_max:g} mm hr-1 (tp is the
                     accumulation over the hour ending at the time stamp).

Each counted hour is then labelled by the share of CWP that is liquid or ice
(using the floored paths):

  liquid containing  IWP/CWP <  {args.ice_fraction_min:.2f}   (liquid only + mixed phase)
  liquid only        LWP/CWP >= {args.liquid_fraction_min:.2f}   (a subset of liquid containing)
  ice only           IWP/CWP >= {args.ice_fraction_min:.2f}

Liquid containing and ice only together make up every counted hour; mixed
phase = liquid containing - liquid only.


3. SURFACE CLASSES
------------------
Each cell is classified EVERY HOUR, from the static land fraction and that
hour's sea ice concentration:

  Land            lsm >= {1 - args.lsm_tol:g} (and siconc undefined or < {args.land_max_siconc:g})
  Coastal         {args.lsm_tol:g} < lsm < {1 - args.lsm_tol:g} (mixed land/sea cells)
  OpenOcean       lsm <= {args.lsm_tol:g} and siconc < {args.open_ocean_max_siconc:g}
  MarginalSeaIce  lsm <= {args.lsm_tol:g} and {args.open_ocean_max_siconc:g} <= siconc <= {args.sea_ice_min_siconc:g}
  PackIce         lsm <= {args.lsm_tol:g} and siconc > {args.sea_ice_min_siconc:g}

So a cell can move between the three ocean classes as the ice edge moves, and
the number of cells in each ocean class changes from hour to hour. The UTQ
cell is also counted within its own class (Land or Coastal) in the class
tables; it is not a separate area.


4. UNITS: CELL-HOURS (PLAIN COUNTS)
-----------------------------------
Every entry is the number of (grid cell, hour) pairs meeting the criteria.
For UTQ (one cell) this is simply hours. For a surface class it is summed
over every cell that belonged to the class in that hour: no area weighting,
no averaging over cells, no rescaling for missing data (there are no gaps in
the archive over these seasons).

A class total therefore grows with the size of the class. To compare classes,
divide by the class's total valid cell-hours in the same period, given in
normalization/ (same rows and layout; "all valid" = every cell-hour of the
class with finite tcc, tclw and tciw, before filters b-d). That gives the
unweighted fraction of the class's area-time in each state. Note that the
notebook's domain figures use cos(latitude)-weighted fractions, which differ
slightly from these unweighted ones because pack ice lies further north.


5. FILES
--------
Per season ({len(SLOT_NAMES)} files), rows = seasons, columns = liquid_containing,
liquid_only, ice_only:
    ERA5_totalHours_perSeason_{span}_<CLASS>.csv

Per month ({len(SLOT_NAMES) * 3} files), rows = Oct ... Mar, columns = seasons
{season_label(s0)} ... {season_label(s1)}; one file per cloud type:
    ERA5_totalHours_perMonth_{span}_<TYPE>_<CLASS>.csv
    <TYPE> = liquidContaining | liquidOnly | iceOnly

Per day ({len(SLOT_NAMES)} files), one row per day of the record
({C['dates'].size} days, days with no cloud included as 0), columns = date
('1 Oct. 2014'), liquid_containing, liquid_only, ice_only:
    ERA5_totalHours_perDay_{span}_<CLASS>.csv

<CLASS> = UTQ | Land | Coastal | OpenOcean | MarginalSeaIce | PackIce

normalization/ : all valid cell-hours per class, per season (one file,
one column per class), per month (one file per class, same layout as the
monthly tables) and per day (one file, one column per class).

The monthly and seasonal tables are sums of the daily ones.


6. RELATION TO THE FIGURES
--------------------------
UTQ totals over all {len(C['seasons'])} seasons: {utq_tot['liquid_containing']:,} h liquid containing,
{utq_tot['liquid_only']:,} h liquid only, {utq_tot['ice_only']:,} h ice only.

Figure 1 (hours per season, UTQ): the red bars are the liquid_containing
column of the UTQ per-season file. The blue ice-only bars additionally
include the few cloudy, non-precipitating hours with NO cloud water above
the floors ({utq_tot['overcast_no_condensate']} hours over all seasons), which these tables leave out.

Figure 2 (monthly box plots, UTQ): ERA5 boxes are built from the UTQ
liquidContaining monthly table (one value per season per month). The figure
omits the {len(dropped)} season-months for which the ARM record is incomplete
({dropped_txt}) from both ERA5 and the observations; these tables include
every month.

Figure 4 (cloud duration / hours per day, UTQ): the daily-mode histogram is
the distribution of the liquid_containing column of the UTQ per-day file.

The thresholds are the same for the class tables; the class tables are not
shown in figures 1, 2 or 4.


7. PLOT COLORS
--------------
The colours used in the Ocean Visions figures and the surface-class figures,
so plots made from these tables can match. RGB is given on both the 0-255
and 0-1 scales; "matplotlib" is the colour as written in the plotting code
(a name, a hex string, or a grey level between 0 and 1).
{plot_colors_text()}
"""
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"output directory (default {DEFAULT_OUT_DIR.name}/ beside this script)")
    p.add_argument("--verify", action="store_true",
                   help="also require exact agreement with lwph.prepare (~4 min more)")
    opts = p.parse_args(argv)

    args = build_args(SETTINGS)
    print("=" * 72)
    print("ERA5 cloud cell-hour counts: UTQ + five surface classes")
    print("=" * 72)
    C = count_cell_hours(args)
    self_checks(C)
    if opts.verify:
        verify_against_module(C, args)

    written = write_all(C, opts.out_dir)
    readme = write_readme(C, args, opts.out_dir)
    print(f"\n  -> {len(written)} CSV files + {readme.name} in {opts.out_dir}")
    print(per_season(C, I_UTQ).to_string(index=False))


if __name__ == "__main__":
    main()
