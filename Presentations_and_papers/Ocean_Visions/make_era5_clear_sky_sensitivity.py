#!/usr/bin/env python3
"""How many ERA5 hours count as clear sky, and how does that depend on the definition?

WHY THIS EXISTS
===============
In Figure 7 of Genie's draft (OV_SLCFrequency_20261102), ERA5 has far fewer
clear-sky hours than the ARM observations at UTQ: 1-17 h per month against
35-165 h. This script measures how the ERA5 count changes when each part of the
clear-sky definition is relaxed, so the strictness of the definition can be
separated from ERA5 simply being cloudier.

THE DEFINITION IN FIGURE 7
==========================
make_era5_hourly_values_csvs.py, which wrote Genie's ERA5 files, calls an hour
clear when ALL of these hold (plot_dlr_by_phase.DEFAULT_CLEAR_TCC_MAX = 0.05):

    LWP <= 0.01 g m-2 and IWP <= 0.01 g m-2   (both paths at or below the floors)
    tcc < 0.05
    tp  < 0.05 mm hr-1

So the cloud cover limit is already 0.05, not 0.

THE RELAXED DEFINITIONS
=======================
Each variant keeps tp < 0.05 mm hr-1. The precipitation test makes no
difference to clear hours (checked 2026-11-12: the counts with and without it
agree to 0.1 h per month), so it is not varied.

    fig7          the Figure 7 definition above
    tcc20         tcc < 0.20, paths unchanged
    tcc20_iwp01   tcc < 0.20, LWP <= 0.01, IWP <= 0.1 g m-2
    tcc20_paths1  tcc < 0.20, LWP and IWP <= 1 g m-2
    paths1        LWP and IWP <= 1 g m-2, no cloud cover test
    cre5          DLR - DLR_clear < 5 W m-2, no cloud cover or path test
    cre10         DLR - DLR_clear < 10 W m-2, no cloud cover or path test
    arm80         tcc <= 0.20, no path test (an analogue of Genie's ARM rule; below)
    arm80_gt1h    arm80, and the same cell also passes arm80 in the hour before
                  or the hour after (a clear spell longer than 1 h)

DLR is msdwlwrf and DLR_clear is msdwlwrfcs, ERA5's own clear-sky downward
longwave: the same radiation calculation with the clouds removed (ECMWF IFS
documentation, Part IV, chapter 2). Their difference is ERA5's longwave cloud
radiative effect at the surface. The cre variants therefore ask whether the
clouds ERA5 does have are too thin to matter radiatively. The thresholds 0.20, 0.1, 1, 5
and 10 are sensitivity choices of mine, not values from a paper.

AN ERA5 ANALOGUE OF GENIE'S ARM CLEAR-SKY RULE
==============================================
Genie calls an hour clear when at least 80% of the 30-s thermocldphase time
stamps in it have a column labelled clear, i.e. at most 20% of the samples see
any hydrometeor anywhere in the column. Her Figure 7a counts clear spells
longer than 1 h.

ERA5 has no sub-hourly samples, but tcc is the fraction of the grid box covered
by cloud at any level, seen from above. My reasoning (an analogy I have not seen
in a paper): an hour of 30-s columns samples the cloud field advected over the
site, roughly 20-30 km at 6-8 m/s, which is about one ERA5 cell (0.25 deg is
about 28 km north-south). If that time sample is representative of the cell
(Taylor's frozen-turbulence hypothesis plus an ergodic assumption), "at most 20%
of columns cloudy" corresponds to "cloud cover at most 0.20". Hence arm80:
tcc <= 0.20. It has no water-path test, because Genie's rule allows cloud in up
to 20% of the samples, so the grid-box mean paths need not be near zero.

arm80_gt1h adds persistence for the "> 1 h" in Figure 7a: an hour counts only if
the same cell also passes arm80 at an adjacent hour stamp (same season, one
hour apart). That is my reading of "> 1 h"; check it against Genie's code.

Where the analogy is weakest (my inference, not tested):
  * ERA5 tcc is a model diagnostic from the cloud scheme, not a detection by
    radar and lidar, so it can count cloud that a lidar would miss, and miss
    thin cloud that a lidar would see.
  * tcc is an instantaneous value at the hour stamp, while Genie's 80% is
    over the hour. arm80_gt1h, which needs two consecutive stamps, is the
    closer match to a clear hour.

Timing caveat: tcc, tclw, tciw are instantaneous at the time stamp, while tp,
msdwlwrf and msdwlwrfcs are means over the hour ending at it.

TWO WAYS TO COUNT HOURS FOR A SURFACE CLASS
===========================================
(a) Per grid cell, as in the class_normalized tables:

        H = sum_t n(t) / N(t)     [h]

    N(t) is the weight of the class's valid cells at hour t, and n(t) the weight
    of those that are clear (weight 1, or cos(lat) for area weighting). This is
    the clear-sky time of a typical cell of the class, so it compares with the
    single UTQ cell and with the ARM site.

(b) Hours with at least one clear cell of the class. This is what the numbers in
    Figure 7 count, because Genie's files hold the class-mean DLR over the cells
    in a state and are blank only when no cell is in it. For a class of hundreds
    of cells this is far larger than (a), and it is not comparable with a
    single site. For UTQ, (a) and (b) are the same.

Both are given as hours in each month, October to March, averaged over the
seasons. A class that does not exist in a month (open ocean in January)
contributes zero hours, as in the other tables.

THE FIGURES
===========
Three figures in the layout of Figure 7: panels (a)-(d) are UTQ, Land, Open
ocean and Pack ice, each with the six months on the x axis and one bar per
definition. They differ only in how a class is counted:

    ..._definitionSensitivity.png                count (a), equal weight per cell
    ..._definitionSensitivity_areaWeighted.png   count (a), cos(latitude) weight
    ..._definitionSensitivity_anyCellCount.png   count (b), as Figure 7 counts

UTQ is the same in all three. Its panel also marks the ARM clear-sky hours for
each month as a dashed line.

THE ARM REFERENCE
=================
ARM_CLEAR_GT1H_H holds the ARM clear-sky hours (> 1 h durations) printed above
the boxes of Figure 7a in the 2026-11-02 draft. They are read off the figure, not
computed here.

CHECK
=====
For the fig7 variant, count (b) must equal the number of filled clear_sky hours in
era5_hourly_values_csv/ERA5_DLR_perHour_<span>_<class>.csv. check_against_hourly_csvs
tests this for every month and class.

REGIONS: BARROW OR RUI
======================
The domain comes from ov_config, as for the other make_era5_*.py scripts. The
UTQ panel is the same grid cell in every domain, so its numbers do not depend
on the domain; Land, Open ocean and Pack ice do.

    Barrow (laptop, the local files)    python make_era5_clear_sky_sensitivity.py
    Rui region (Casper; GDEX only)       ./submit.sh --region Rui-region --years 2014-2024 \
                                             --csvs clear_sky --out <run folder>

The Figure 7 check looks for the hourly-values folder of the same run
(era5_hourly_values_csv or era5_hourly_values_csv_areaWeighted beside the
hour-count folder), or --hourly-dir, and is skipped with a message if none is
found.

USAGE
=====
    python make_era5_clear_sky_sensitivity.py              # Barrow, both weightings, ~1 min
    python make_era5_clear_sky_sensitivity.py --plot-only  # redraw from the CSVs

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules on sys.path.
import ov_config as ov

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from make_era5_hour_count_csvs import (  # noqa: E402
    OUT_PRODUCT,
    SETTINGS,
    SLOT_CLASS,
    build_args,
    git_describe,
    season_label,
)

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
    site_in_grid,
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

# ----------------------------------------------------------------------------
# What is computed
# ----------------------------------------------------------------------------
# The four regions of Figure 7, in its panel order, with their labels.
REGIONS = {"UTQ": "UTQ", "Land": "Land", "OpenOcean": "Open ocean", "PackIce": "Pack ice"}
MONTHS = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
WEIGHTINGS = ("area", "uniform")

# Each variant: an upper limit on tcc (fraction), on LWP and IWP (g m-2), and on
# DLR - DLR_clear (W m-2). None means that test is not applied.
VARIANTS = {
    "fig7": dict(tcc=0.05, lwp=0.01, iwp=0.01, cre=None,
                 label="Figure 7: tcc < 0.05, LWP and IWP ≤ 0.01 g m⁻²"),
    "tcc20": dict(tcc=0.20, lwp=0.01, iwp=0.01, cre=None,
                  label="tcc < 0.20, LWP and IWP ≤ 0.01 g m⁻²"),
    "tcc20_iwp01": dict(tcc=0.20, lwp=0.01, iwp=0.1, cre=None,
                        label="tcc < 0.20, LWP ≤ 0.01, IWP ≤ 0.1 g m⁻²"),
    "tcc20_paths1": dict(tcc=0.20, lwp=1.0, iwp=1.0, cre=None,
                         label="tcc < 0.20, LWP and IWP ≤ 1 g m⁻²"),
    "paths1": dict(tcc=None, lwp=1.0, iwp=1.0, cre=None,
                   label="No tcc test, LWP and IWP ≤ 1 g m⁻²"),
    "cre5": dict(tcc=None, lwp=None, iwp=None, cre=5.0,
                 label="Radiatively clear: DLR − clear-sky DLR < 5 W m⁻²"),
    "cre10": dict(tcc=None, lwp=None, iwp=None, cre=10.0,
                  label="Radiatively clear: DLR − clear-sky DLR < 10 W m⁻²"),
    # The ERA5 analogue of Genie's ARM rule (see docstring). tcc_le is inclusive
    # (<=), matching "at least 80% of samples clear" = "at most 20% cloudy".
    "arm80": dict(tcc=None, tcc_le=0.20, lwp=None, iwp=None, cre=None,
                  label="ARM-like: tcc ≤ 0.20, no water-path test"),
    "arm80_gt1h": dict(tcc=None, tcc_le=0.20, lwp=None, iwp=None, cre=None, persist=True,
                       label="ARM-like, > 1 h: tcc ≤ 0.20 at this and an adjacent hour"),
}

# ARM clear sky with durations > 1 h at UTQ, hours per month, read from the
# numbers above the boxes of Figure 7a (draft OV_SLCFrequency_20261102).
ARM_CLEAR_GT1H_H = {"Oct": 35, "Nov": 57, "Dec": 137, "Jan": 165, "Feb": 143, "Mar": 160}

# Colours: the five definitions built from cloud cover and paths are an ordered
# relaxation, so they share one hue from light to dark (grey-blue, near the grey
# Figure 7 uses for clear sky). The two radiative definitions are a different
# kind of test and get a second hue.
VARIANT_COLORS = {
    "fig7": "#c6d3df",
    "tcc20": "#9fb4c7",
    "tcc20_iwp01": "#7393ae",
    "tcc20_paths1": "#4d7192",
    "paths1": "#2c4d6b",
    "cre5": "#f0a95a",
    "cre10": "#b8621b",
    # A third hue for the definitions modelled on the ARM rule.
    "arm80": "#b39ddb",
    "arm80_gt1h": "#5e3c99",
}
INK_PRIMARY = "#222222"
INK_MUTED = "#666666"
GRID_COLOR = "#e4e4e4"


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def clear_sky_hits(fields: dict, spec: dict, args: argparse.Namespace) -> np.ndarray:
    """Cells that pass one clear-sky variant, (t, lat, lon), before any persistence test.

    ``fields`` holds the block's tcc, lwp_g_m2, iwp_g_m2, cre_w_m2, rate_mm_hr
    and ``base`` (valid and not raining). make_era5_iceOnly_sensitivity.py passes its own
    function of the same shape to accumulate().
    """
    hit = fields["base"].copy()
    tcc, lwp_g_m2, iwp_g_m2, cre_w_m2 = (fields[k] for k in ("tcc", "lwp_g_m2", "iwp_g_m2", "cre_w_m2"))
    if spec["tcc"] is not None:
        hit &= tcc < spec["tcc"]
    if spec.get("tcc_le") is not None:
        hit &= tcc <= spec["tcc_le"]
    if spec["lwp"] is not None:
        # "At or below the floor" in the project means value <= floor.
        hit &= (lwp_g_m2 <= spec["lwp"]) & (iwp_g_m2 <= spec["iwp"])
    if spec["cre"] is not None:
        hit &= np.isfinite(cre_w_m2) & (cre_w_m2 < spec["cre"])
    return hit


def accumulate(args: argparse.Namespace, variants: dict | None = None,
               hits_fn=clear_sky_hits) -> dict:
    """Per hour and region: class weight, state weight per variant, and 'any cell in state'.

    ``variants`` (default VARIANTS) and ``hits_fn(fields, spec, args)`` define
    the sky state; a variant with ``persist=True`` also needs the state at an
    adjacent hour of the same cell. The result keys say "clear" for history,
    but they hold whatever state hits_fn tests.

    Weighted sums are kept for both weightings at once, so one pass serves both.
    Returns arrays shaped (hour, region, variant) for the state sums and (hour,
    region) for the class sums, plus each hour's season and month.
    """
    variants = VARIANTS if variants is None else variants
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(
            args.region, lwph.resolve_data_root(args.storage, args.data_root), args.mask_grid
        ),
        ds,
    )
    layout = season_layout(ds, args)
    keep_idx, used, _label = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    times = np.asarray(ds["valid_time"].values)
    hour_season = np.asarray(layout["seasons"])[s_idx[use_step]]       # (hour,)
    hour_month = pd.DatetimeIndex(times[use_step]).month.values         # (hour,)
    row_of_step = np.cumsum(use_step) - 1

    if not site_in_grid(ds):
        raise ValueError("the ARM site is outside this domain; this script compares with it")
    site_mask = site_cell_mask(ds)[0]                                   # (lat, lon)
    weights = {
        "area": area_weights_2d(ds["latitude"].values, ds.sizes["longitude"]),
        "uniform": np.ones((ds.sizes["latitude"], ds.sizes["longitude"])),
    }                                                                   # each (lat, lon)

    n_hour, n_reg, n_var = int(use_step.sum()), len(REGIONS), len(variants)
    class_w = {w: np.zeros((n_hour, n_reg)) for w in WEIGHTINGS}         # N(t)
    clear_w = {w: np.zeros((n_hour, n_reg, n_var)) for w in WEIGHTINGS}  # n(t)
    any_clear = np.zeros((n_hour, n_reg, n_var), dtype=bool)

    persist_v = [v for v, spec in enumerate(variants.values()) if spec.get("persist")]
    hour_ns = np.int64(3_600_000_000_000)                              # one hour in ns

    def add_hits(rows_: np.ndarray, v: int, clear_: np.ndarray, masks_: list) -> None:
        """Add one variant's clear cells to the sums, per region. clear_ is (t, lat, lon)."""
        for r, region in enumerate(masks_):
            hit = region & clear_
            any_clear[rows_, r, v] = hit.any(axis=(1, 2))
            for w in WEIGHTINGS:
                clear_w[w][rows_, r, v] = (hit * weights[w]).sum(axis=(1, 2))

    # The persistence variants need each hour's neighbours, and a block's last
    # hour has its next neighbour in the following block. So that hour is held
    # back ("carry") and finished when the next block arrives.
    # The carry also keeps "was the hour before it clear and adjacent?", because
    # that hour has already been finished and would otherwise be forgotten.
    carry = None   # dict(row, time, base and prev (lat, lon) per persist variant, masks)

    def finish_persist(rows_, times_, bases_, masks_, next_base, next_time, prev0) -> None:
        """Persistence variants for a run of hours: clear now and at an adjacent hour.

        prev0[v] is (lat, lon): whether the hour before rows_[0] was clear and one
        hour earlier (None when there is no such hour).
        """
        for v in persist_v:
            b = bases_[v]                                                # (t, lat, lon)
            adj = (np.diff(times_.astype("int64")) == hour_ns)           # (t-1,) consecutive?
            prev_ok = np.zeros_like(b)
            prev_ok[1:] = b[:-1] & adj[:, None, None]
            if prev0 is not None:
                prev_ok[0] = prev0[v]
            next_ok = np.zeros_like(b)
            next_ok[:-1] = b[1:] & adj[:, None, None]
            if next_base is not None and (next_time - times_[-1]).astype("int64") == hour_ns:
                next_ok[-1] = next_base[v]
            add_hits(rows_, v, b & (prev_ok | next_ok), masks_)

    read_vars = ["tcc", args.liquid_var, "tciw", "tp", "siconc", "msdwlwrf", "msdwlwrfcs"]
    print(f"  Streaming {n_hour:,} hours ...", flush=True)
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours, keep_mask=use_step):
        sl = slice(i0, i0 + block.sizes["valid_time"])
        keep = use_step[sl]
        if not keep.any():
            continue
        rows = row_of_step[sl][keep]
        block_times = times[sl][keep]                                    # (t,) datetime64

        # Region masks, (t, lat, lon): the UTQ cell, then the three classes.
        classes = classify_cells(
            lsm, block["siconc"].values, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )[keep]
        region_masks = [np.broadcast_to(site_mask, classes.shape)]
        region_masks += [classes == CLASS_CODES[SLOT_CLASS[name]] for name in list(REGIONS)[1:]]

        tcc = block["tcc"].values[keep]                                 # fraction, 0-1
        lwp_g_m2 = block[args.liquid_var].values[keep] * 1000.0         # kg m-2 -> g m-2
        iwp_g_m2 = block["tciw"].values[keep] * 1000.0                  # kg m-2 -> g m-2
        rate_mm_hr = block["tp"].values[keep] * 1000.0                  # m h-1 -> mm h-1
        cre_w_m2 = block["msdwlwrf"].values[keep] - block["msdwlwrfcs"].values[keep]  # W m-2

        # Same validity and rain tests as make_era5_hourly_values_csvs.py.
        valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
        raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
        base = valid & ~raining
        fields = {"tcc": tcc, "lwp_g_m2": lwp_g_m2, "iwp_g_m2": iwp_g_m2,
                  "cre_w_m2": cre_w_m2, "rate_mm_hr": rate_mm_hr, "valid": valid, "base": base}

        bases = {}
        for v, spec in enumerate(variants.values()):
            clear = hits_fn(fields, spec, args)
            if spec.get("persist"):
                bases[v] = clear              # finished below, with its neighbours
            else:
                add_hits(rows, v, clear, region_masks)

        if persist_v:
            # Prepend the held-back hour, finish all but this block's last hour,
            # and hold that one back in turn.
            masks_all = [np.asarray(m) for m in region_masks]
            if carry is not None:
                rows_c = np.concatenate([[carry["row"]], rows])
                times_c = np.concatenate([[carry["time"]], block_times])
                bases_c = {v: np.concatenate([carry["base"][v][None], bases[v]]) for v in persist_v}
                masks_c = [np.concatenate([cm[None], m]) for cm, m in zip(carry["masks"], masks_all)]
            else:
                rows_c, times_c, bases_c, masks_c = rows, block_times, bases, masks_all
            prev0 = carry["prev"] if carry is not None else None
            if rows_c.size > 1:
                finish_persist(rows_c[:-1], times_c[:-1], {v: b[:-1] for v, b in bases_c.items()},
                               [m[:-1] for m in masks_c],
                               {v: b[-1] for v, b in bases_c.items()}, times_c[-1], prev0)
                # The new carry's previous hour is the second-to-last one here.
                gap_ok = (times_c[-1] - times_c[-2]).astype("int64") == hour_ns
                new_prev = {v: b[-2] & gap_ok for v, b in bases_c.items()}
            else:
                new_prev = prev0
            carry = {"row": rows_c[-1], "time": times_c[-1],
                     "base": {v: b[-1] for v, b in bases_c.items()},
                     "prev": new_prev,
                     "masks": [m[-1] for m in masks_c]}

        for r, region in enumerate(region_masks):
            in_class = region & valid
            for w in WEIGHTINGS:
                class_w[w][rows, r] = (in_class * weights[w]).sum(axis=(1, 2))

    if persist_v and carry is not None:
        # The very last hour: no later neighbour.
        finish_persist(np.array([carry["row"]]), np.array([carry["time"]]),
                       {v: b[None] for v, b in carry["base"].items()},
                       [m[None] for m in carry["masks"]], None, None, carry["prev"])

    return {"class_w": class_w, "clear_w": clear_w, "any_clear": any_clear,
            "hour_season": hour_season, "hour_month": hour_month, "seasons": list(used)}


def build_table(A: dict, variants: dict | None = None) -> pd.DataFrame:
    """Hours per month, one row per region, variant, month and counting method.

    Columns: count ('per_cell_area', 'per_cell_uniform', 'any_cell'), one column
    per season, and mean_of_seasons.
    """
    variants = VARIANTS if variants is None else variants
    labels = [season_label(y) for y in A["seasons"]]
    hour_values = {"any_cell": A["any_clear"].astype(float)}             # (hour, region, variant)
    for w in WEIGHTINGS:
        N = A["class_w"][w][..., None]                                   # (hour, region, 1)
        hour_values[f"per_cell_{w}"] = np.where(N > 0, A["clear_w"][w] / np.where(N > 0, N, 1.0), 0.0)

    rows = []
    for count, values in hour_values.items():
        for r, region in enumerate(REGIONS):
            for v, variant in enumerate(variants):
                for month, month_name in MONTHS.items():
                    row = {"count": count, "region": region, "variant": variant, "month": month_name}
                    for year, label in zip(A["seasons"], labels):
                        sel = (A["hour_season"] == year) & (A["hour_month"] == month)
                        row[label] = values[sel, r, v].sum()
                    rows.append(row)
    df = pd.DataFrame(rows)
    df["mean_of_seasons"] = df[labels].mean(axis=1)
    return df


def find_hourly_dir(hour_count_dir: Path, given: Path | None, domain: str) -> Path | None:
    """The same run's hourly-values folder, or ``given``; None if there is none.

    Tries the names make_era5_hourly_values_csvs.py writes, then the
    "<name>_<domain>Region" form used for the renamed Barrow folder
    (era5_hourly_values_csv_barrowRegion). A folder for another domain is never
    picked up by accident, because each name is matched exactly.
    """
    if given is not None:
        return given
    for name in ("era5_hourly_values_csv", "era5_hourly_values_csv_areaWeighted",
                 f"era5_hourly_values_csv_{domain}Region"):
        candidate = hour_count_dir.parent / name
        if candidate.is_dir():
            return candidate
    return None


def check_against_hourly_csvs(df: pd.DataFrame, span: str, hourly_dir: Path | None,
                              column: str = "clear_sky", variant: str = "fig7") -> None:
    """The ``variant`` 'any cell' counts must equal the filled ``column`` hours in Genie's files."""
    if hourly_dir is None:
        print("  !! no hourly-values folder for this run: Figure 7 check skipped "
              "(pass --hourly-dir to run it)")
        return
    worst, n_checked = 0.0, 0
    sub = df[(df["count"] == "any_cell") & (df["variant"] == variant)]
    seasons = [c for c in df.columns if "/" in c]
    for region in REGIONS:
        # The DLR files are written per weighting, but which hours are filled does not depend on it.
        candidates = sorted(hourly_dir.glob(f"ERA5_DLR*_perHour_{span}_{region}.csv"))
        if not candidates:
            print(f"  !! no hourly DLR file for {region} in {hourly_dir}")
            continue
        hourly = pd.read_csv(candidates[0], usecols=["date", column])
        when = pd.to_datetime(hourly["date"])
        season_start = np.where(when.dt.month >= 10, when.dt.year, when.dt.year - 1)
        for month, month_name in MONTHS.items():
            for season in seasons:
                year = int(season.split("/")[0])
                theirs = hourly.loc[(when.dt.month == month) & (season_start == year),
                                    column].notna().sum()
                mine = sub[(sub["region"] == region) & (sub["month"] == month_name)][season].iloc[0]
                worst = max(worst, abs(mine - theirs))
                n_checked += 1
    print(f"  check: Figure 7 counts reproduced for {n_checked} region-months, worst {worst:.0f} h")
    if n_checked and worst > 0:
        raise AssertionError(f"the {variant} variant does not reproduce Genie's {column} hours")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def season_month_mean(df: pd.DataFrame, count: str, region: str, variant: str) -> float:
    """Mean over the six months of the mean over seasons: hours per month (for the log)."""
    sel = (df["count"] == count) & (df["region"] == region) & (df["variant"] == variant)
    return float(df.loc[sel, "mean_of_seasons"].mean())


def month_values(df: pd.DataFrame, count: str, region: str, variant: str) -> list[float]:
    """Mean over seasons for each month, October to March: hours in that month."""
    sel = (df["count"] == count) & (df["region"] == region) & (df["variant"] == variant)
    by_month = df.loc[sel].set_index("month")["mean_of_seasons"]
    return [float(by_month[m]) for m in MONTHS.values()]


# The three figures: which count each one draws, and how it is described.
COUNT_NOTES = {
    "per_cell_uniform": "Hours per month for a typical grid cell, equal weight per cell",
    "per_cell_area": "Hours per month for a typical grid cell, area weighted (cos latitude)",
    "any_cell": "Hours per month with at least one clear cell of the class, as Figure 7 counts",
}


def plot(df: pd.DataFrame, path: Path, count: str, n_seasons: int, domain: str) -> Path:
    """Figure 7 layout: one panel per region, months on the x axis, one bar per definition."""
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(12.0, 14.0), sharex=True)
    n_var = len(VARIANTS)
    width = 0.84 / n_var
    x_month = np.arange(len(MONTHS))
    for p, (ax, (region, region_label)) in enumerate(zip(axes, REGIONS.items())):
        top = 0.0
        # Months with no clear hours under any definition (open ocean Jan-Mar, when
        # the class does not exist) get one note instead of a row of zeros.
        all_values = np.array([month_values(df, count, region, v) for v in VARIANTS])  # (variant, month)
        empty_month = (all_values == 0.0).all(axis=0)                                # (month,)
        for v, variant in enumerate(VARIANTS):
            x = x_month + (v - (n_var - 1) / 2) * width
            values = all_values[v]
            top = max(top, *values)
            ax.bar(x, values, width, color=VARIANT_COLORS[variant], edgecolor="white",
                   linewidth=0.6, label=VARIANTS[variant]["label"])
            for m, (xi, value) in enumerate(zip(x, values)):
                if not empty_month[m]:
                    ax.text(xi, value, f"{value:.0f}", ha="center", va="bottom", fontsize=6.3,
                            color=INK_PRIMARY, zorder=4)
        for m in np.flatnonzero(empty_month):
            ax.text(x_month[m], 0, "0 h under every definition", ha="center", va="bottom",
                    fontsize=8, color=INK_MUTED)
        if region == "UTQ":
            # ARM clear sky (> 1 h) for each month: a short dashed line across that month's bars.
            half = 0.44
            for xm, month_name in zip(x_month, MONTHS.values()):
                arm_h = ARM_CLEAR_GT1H_H[month_name]
                ax.plot([xm - half, xm + half], [arm_h, arm_h], color=INK_PRIMARY,
                        linestyle="--", linewidth=1.3, zorder=3,
                        label="ARM clear sky > 1 h (numbers in Fig. 7a)" if xm == 0 else None)
                top = max(top, arm_h)
        ax.set_ylim(0, max(top, 1.0) * 1.15)
        ax.text(0.005, 0.96, f"({'abcd'[p]}) {region_label}", transform=ax.transAxes,
                ha="left", va="top", fontsize=13, color=INK_PRIMARY)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", colors=INK_MUTED)
    axes[-1].set_xticks(x_month)
    axes[-1].set_xticklabels(list(MONTHS.values()), fontsize=13, color=INK_PRIMARY)
    fig.supylabel("Clear-sky hours per month", fontsize=13, color=INK_PRIMARY)
    # One legend for all panels, above them, with the ARM line from panel (a).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.995),
               title="Clear-sky definition (all with tp < 0.05 mm/hr)", title_fontsize=10)
    fig.text(
        0.01, 0.003,
        f"{domain}. {COUNT_NOTES[count]}. Mean of {n_seasons} seasons. Months when a class "
        f"does not exist count as zero.",
        fontsize=9, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.905))
    fig.savefig(path, dpi=350)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Files and entry point
# ----------------------------------------------------------------------------
def names(span: str) -> dict:
    """Output file names: one CSV, and one figure per way of counting."""
    stem = f"ERA5_clearSkyHours_perMonth_{span}_definitionSensitivity"
    return {"csv": f"{stem}.csv",
            "fig": {"per_cell_uniform": f"{stem}.png",
                    "per_cell_area": f"{stem}_areaWeighted.png",
                    "any_cell": f"{stem}_anyCellCount.png"}}


def find_span(out_dir: Path) -> str:
    """The season span tag of the CSV already in ``out_dir``."""
    prefix, suffix = "ERA5_clearSkyHours_perMonth_", "_definitionSensitivity.csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {prefix}<span>{suffix} in {out_dir}; run without --plot-only")
    return found[0].name[len(prefix) : -len(suffix)]


def draw(out_dir: Path, span: str, domain: str) -> list[Path]:
    """The three figures, from the CSV only."""
    df = pd.read_csv(out_dir / names(span)["csv"])
    n_seasons = len([c for c in df.columns if "/" in c])
    return [plot(df, out_dir / name, count, n_seasons, domain)
            for count, name in names(span)["fig"].items()]


def write_readme(out_dir: Path, span: str, n_seasons: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = names(span)
    text = (
        f"Generated {stamp} by make_era5_clear_sky_sensitivity.py (repo {git_describe()})\n"
        f"Seasons: {n_seasons}.\n\n"
        f"{n['csv']}\n    Clear-sky hours per month by count method, region, variant, month, season\n"
        "Figures: Figure 7 layout, panels (a)-(d) UTQ, Land, Open ocean, Pack ice\n"
        + "".join(f"{name}\n    {COUNT_NOTES[count]}\n" for count, name in n["fig"].items())
        + "\n"
        f"{__doc__}"
    )
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default <hour-count folder>/clear_sky_sensitivity)")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the figures from the CSV already in the output folder")
    p.add_argument("--hourly-dir", type=Path, default=None,
                   help="hourly-values folder for the Figure 7 check (default: the run's own)")
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    out_dir = opts.out_dir or hour_count_dir / "clear_sky_sensitivity"

    if opts.plot_only:
        for path in draw(out_dir, find_span(out_dir), run.domain.name):
            print(f"  -> {path}")
        return

    print(run.report())
    # Thresholds (floors, precipitation, class limits) come from the project's SETTINGS.
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": "uniform"})
    A = accumulate(args)
    span = ov.span_tag(A["seasons"])
    df = build_table(A)
    check_against_hourly_csvs(df, span, find_hourly_dir(hour_count_dir, opts.hourly_dir, run.domain.name))

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / names(span)["csv"], index=False, float_format="%.4f")
    write_readme(out_dir, span, len(A["seasons"]))
    for path in draw(out_dir, span, run.domain.name):
        print(f"  -> {path}")

    # The numbers in the figure, for the log: hours per month, mean of Oct-Mar.
    for count in ("per_cell_area", "per_cell_uniform", "any_cell"):
        table = pd.DataFrame({
            REGIONS[r]: [season_month_mean(df, count, r, v) for v in VARIANTS] for r in REGIONS
        }, index=list(VARIANTS))
        print(f"\nClear-sky hours per month, {count}")
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
