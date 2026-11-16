#!/usr/bin/env python3
"""Surface-class time series (Figures 1-6 of plot_surface_class_timeseries.ipynb), any domain.

WHAT IT MAKES
=============
The six figures of ERA5/surface_energy_budget/plot_surface_class_timeseries.ipynb,
for whatever domain and seasons ov_config resolves (the Rui region, the Barrow
strip, any --box). Each has the quantity by surface class on top and the share
of the domain's area each class occupies underneath, against day of season:

    1  <region>_surfaceclass_dlr_<tag>.png               downwelling longwave
    2  <region>_surfaceclass_lwp_<tag>.png               liquid water path
    3  <region>_surfaceclass_iwp_<tag>.png               ice water path
    4  <region>_surfaceclass_liquid_fraction_<tag>.png   cloudy scenes containing liquid
    5  <region>_surfaceclass_dlr_dlat_byclass_<tag>.png  dDLR/dlat within each class
    6  <region>_surfaceclass_dlr_dlat_transitions_<tag>.png  dDLR/dlat across class edges

(<tag> is "mean<first>-<last>" for several seasons, "season<year>" for one.)

plus, for comparing the AREA FRACTION of each class between domains:

    ERA5_surfaceClassAreaPercent_perDay_meanOfSeasons_<span>.csv
        one row per day of season; % of the domain's area in each class,
        averaged over the seasons (the bottom panel of every figure)
    ERA5_surfaceClassAreaPercent_perDay_perSeason_<span>.csv
        the same, one row per season and day
    summary.txt
        mean area share per class, where the ARM site cell landed, and the
        season-mean DLR gradients behind figures 5 and 6 (notebook cell 22)

HOW
===
Every number comes from plot_surface_class_timeseries.prepare() and its fig_*
functions, imported, not copied, so this script and the notebook cannot drift
apart. ERA5/surface_energy_budget is part of the sparse checkout on Casper
(every make_era5_*.py job already imports from it), so nothing needs copying.

The science settings are the notebook's (SETTINGS below): thresholds, class
boundaries, 5-day smoothing, and a 1 Aug - 31 Mar season so the open-water
months before freeze-up are included. Domain, years and storage come from
ov_config, as in the other make_era5_*.py scripts. On Casper the remote reader
loads exactly the windows of --years x [season start, season end].

The supercooled-only filter (ov_config SUPERCOOLED_ONLY) does NOT apply here:
these are area-mean LWP/IWP/DLR fields and a liquid-PRESENCE fraction, not the
liquid-containing cloud statistics the filter is defined for. The output folder
is therefore never tagged "_supercooled".

USAGE
=====
    python make_era5_surface_class_timeseries.py --box 80 -180 65 -140 --region-name Rui-region
    python make_era5_surface_class_timeseries.py --region barrow --years 2022-2025   # laptop check
    ./submit.sh --box 80 -180 65 -140 --region-name Rui-region --csvs surface_class   # Casper

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules (ERA5/surface_energy_budget) on sys.path.
import ov_config as ov

import matplotlib

matplotlib.use("Agg")                      # batch job: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plot_surface_class_timeseries as sct  # noqa: E402
from make_era5_utqiagvik_filtered_netcdf import git_describe  # noqa: E402

# Folder name of this product; ov_config decides which directory holds it.
OUT_PRODUCT = "surface_class_timeseries"

# The notebook's prepare() call (plot_surface_class_timeseries.ipynb, cell 4),
# minus region/years/storage, which come from ov_config. Units in the comments.
SETTINGS = dict(
    season_start=(8, 1),          # month, day: includes open water before freeze-up
    season_end=(3, 31),           # month, day, inclusive; the season wraps the new year
    lwp_threshold=0.1,            # g m-2, liquid PRESENT test (figure 4)
    iwp_threshold=0.1,            # g m-2, ice PRESENT test (figure 4's cloudy test)
    trace_threshold=0.03,         # g m-2, hours at or below are zeroed before the LWP/IWP means
    min_cloud_fraction=0.98,      # tcc at or above which a scene counts as cloudy (figure 4)
    lsm_tol=0.01,                 # land-sea mask tolerance for land / sea cells
    open_ocean_max_siconc=0.05,   # siconc below which a sea cell is open ocean
    sea_ice_min_siconc=0.95,      # siconc above which a sea cell is pack ice
    smooth=5,                     # days, centred running mean on the TOP panels only
    area_style="stacked",         # bottom panel: stacked area shares (they sum to 100%)
    dpi=350,                      # dots per inch of the saved PNGs
)

# Figures 1-6 of the notebook, in its order.
FIGURES = (
    sct.fig_dlr,                        # 1
    sct.fig_lwp,                        # 2
    sct.fig_iwp,                        # 3
    sct.fig_liquid_fraction,            # 4
    sct.fig_dlr_gradient_by_class,      # 5
    sct.fig_dlr_gradient_transitions,   # 6
)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def class_columns(A) -> list[tuple[int, str]]:
    """(index into the class axis, column name) for the five surface classes."""
    return [(sct.CLASS_CODES[n], sct.CLASS_LABELS[n]) for n in sct.CLASS_ORDER]


def area_tables(A) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily % of the domain area per class: mean of seasons, and per season.

    ``A.col["area_pct"]`` is (slot, class) and ``A.sec["area_pct"]`` is
    (season, slot, class), slot = day of season; the ARM cell, if any, has its
    own index on the class axis and is left out (it is one cell inside a class,
    not a class).
    """
    slots = A.col["slots"]                                   # [(month, day), ...]
    day_cols = {"month": [m for m, _ in slots], "day": [d for _, d in slots]}
    cols = class_columns(A)

    mean = pd.DataFrame(day_cols)
    for ci, name in cols:
        mean[name] = A.col["area_pct"][:, ci]

    rows = []
    seasons = np.asarray(A.layout["seasons"])
    for si in A.keep_idx:
        part = pd.DataFrame({"season_start_year": int(seasons[si]), **day_cols})
        for ci, name in cols:
            part[name] = A.sec["area_pct"][si, :, ci]
        rows.append(part)
    per_season = pd.concat(rows, ignore_index=True)
    return mean, per_season


def summary_text(A, run: ov.RunConfig) -> str:
    """Area shares, the ARM cell, and the season-mean DLR gradients, as text."""
    lines = [
        "SURFACE-CLASS TIME SERIES",
        "=========================",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by "
        f"make_era5_surface_class_timeseries.py (repo {git_describe()})",
        f"Domain {run.domain.name}: {run.domain.describe()}; seasons {A.used} "
        f"({A.args.season_start[0]:02d}-{A.args.season_start[1]:02d} to "
        f"{A.args.season_end[0]:02d}-{A.args.season_end[1]:02d}); ERA5 from {run.source_text()}.",
        "Area shares are cos(latitude)-weighted % of the domain (plot_surface_class_timeseries).",
        "",
        "Mean area share over the season window [% of domain]:",
    ]
    for ci, name in class_columns(A):
        lines.append(f"  {name:<20}{sct.nanmean_quiet(A.col['area_pct'][:, ci]):7.2f}")

    # Notebook cell 22: the numbers behind figures 5 and 6.
    lines += ["", "dDLR/dlat within each class, season mean [W m-2 deg-1]:"]
    f = A.col["series"]["dlr_dlat"]
    for ci, name in class_columns(A):
        v = np.nanmean(f[:, ci]) if np.isfinite(f[:, ci]).any() else np.nan
        if np.isfinite(v):
            lines.append(f"  {name:<20}{v:+8.2f}")
    lines += ["", "dDLR/dlat across class transitions [W m-2 deg-1]  (mean share of all edges):"]
    tm, ts = A.col["trans_mean"], A.col["trans_share"]
    for i, lab in enumerate(sct.TRANSITION_LABELS):
        if np.isfinite(tm[:, i]).any():
            lines.append(f"  {lab:<42}{np.nanmean(tm[:, i]):+8.2f}   "
                         f"({np.nanmean(ts[:, i]):5.1f}%)")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"output directory (default {OUT_PRODUCT}/, placed by ov_config)")
    p.add_argument("--season-start", default=None, metavar="MM-DD",
                   help="season start (default 08-01, as the notebook)")
    p.add_argument("--season-end", default=None, metavar="MM-DD",
                   help="season end, inclusive (default 03-31)")
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    # Not a liquid-containing statistic, so the supercooled filter never applies
    # (and the run folder is the unfiltered one).
    opts.supercooled_only = "off"
    run = ov.run_from_args(opts)
    out_dir = opts.out_dir or run.out_dir(OUT_PRODUCT)

    settings = dict(SETTINGS)
    if opts.season_start:
        settings["season_start"] = sct.parse_month_day(opts.season_start)
    if opts.season_end:
        settings["season_end"] = sct.parse_month_day(opts.season_end)

    print("=" * 72)
    print("ERA5 surface-class time series, figures 1-6")
    print(run.report().replace("filter  supercooled only (cloud temperature < 0 C): off",
                               "filter  supercooled only: not applicable"))
    print("=" * 72)

    # region/years/storage/block_hours from ov_config; years must be explicit
    # for the remote reader, which loads exactly those season windows.
    A = sct.prepare(region=run.domain.name, years=tuple(run.years),
                    storage=run.storage, block_hours=run.block_hours, **settings)
    sct.print_report(A)

    for fn in FIGURES:
        fig = fn(A, out_dir=out_dir)
        plt.close(fig)                  # a batch job keeps no figure open

    mean, per_season = area_tables(A)
    span = ov.span_tag(A.used)
    mean.to_csv(out_dir / f"ERA5_surfaceClassAreaPercent_perDay_meanOfSeasons_{span}.csv",
                index=False, float_format="%.4f")
    per_season.to_csv(out_dir / f"ERA5_surfaceClassAreaPercent_perDay_perSeason_{span}.csv",
                      index=False, float_format="%.4f")
    text = summary_text(A, run)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"  -> {len(FIGURES)} figures + 2 CSV files + summary.txt in {out_dir}")


if __name__ == "__main__":
    main()
