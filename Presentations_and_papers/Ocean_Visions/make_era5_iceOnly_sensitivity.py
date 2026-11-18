#!/usr/bin/env python3
"""How many ERA5 hours count as ice-only cloud, and how does that depend on the definition?

WHY THIS EXISTS
===============
In Figure 7a of Genie's draft (OV_SLCFrequency_20261102), ERA5 has far more
ice-only hours at UTQ than the ARM observations: 44-198 h per month against
2-9 h. Her ARM rule is strict in time (below), the ERA5 rule is a snapshot at
the hour stamp. This script measures how much of the gap a more ARM-like ERA5
rule removes.

GENIE'S ARM RULE (as described by the user, 2026-11-17)
=======================================================
From the thermocldphase VAP (30-s columns): a column is ice-only when every
level is either clear or ice, with no liquid, mixed or precipitation at any
level. An hour is ice-only when at least 80% of its 30-s columns are. Figure 7a
then counts ice-only spells longer than 1 h.

THE ERA5 RULE IN FIGURE 7
=========================
make_era5_hourly_values_csvs.py, which wrote Genie's ERA5 files, calls a cell
ice-only at an hour stamp when ALL of these hold (SETTINGS of
make_era5_utqiagvik_filtered_netcdf.py, imported here):

    tcc >= 0.95                                   (overcast gate)
    IWP/CWP >= 0.90, with CWP = LWP + IWP and each path set to zero at or
        below its 0.01 g m-2 floor (cloud_classification.fraction_phase_masks)
    tp < 0.05 mm hr-1                             (no precipitation)

THE VARIANTS
============
    fig7              the Figure 7 rule above
    fig7_gt1h         fig7, and the same cell is also ice-only (fig7) at the hour
                      stamp before or after: an ice-only spell longer than 1 h
    tcc80_gt1h        as fig7_gt1h with tcc >= 0.80 instead of 0.95
    gt1h_ice99        as fig7_gt1h with IWP/CWP >= 0.99
    gt1h_tp001        as fig7_gt1h with tp < 0.01 mm hr-1
    gt1h_ice99_tp001  as fig7_gt1h with IWP/CWP >= 0.99 and tp < 0.01 mm hr-1

In every "> 1 h" variant the adjacent hour must pass that variant's own full
test. The last three (user's choices, 2026-11-17) follow from the UTQ fig7
ice-only hours: 71% have some liquid above the floor and 83% have tp >= 0.001
mm hr-1, whereas Genie's rule allows no liquid and no precipitation at any level.
Precipitation limits keep the project's form: an hour AT or above is dropped.

The adjacency test is the one in make_era5_clear_sky_sensitivity.py
(accumulate(), imported here, so both scripts use the same code): same season,
stamps exactly one hour apart. Reading "> 1 h" as "this and an adjacent stamp"
is my interpretation; check it against Genie's code.

Why these are only partly ARM-like (my inference, not tested):
  * tcc, tclw and tciw are instantaneous at the stamp (ERA5 documentation,
    Tables 2 and 6: no hourly-mean cloud cover exists), while Genie's 80% is a
    fraction of the hour. Two consecutive stamps that pass is the closest
    available match to "most of the hour".
  * tcc >= 0.80 is the user's sensitivity choice. It is a loose
    analogy to "80% of the columns": ARM's other 20% may be anything, including
    liquid, whereas ERA5's IWP/CWP test is applied to the whole cell.
  * ARM's ice-only column excludes any liquid level; ERA5's rule allows up to
    10% of the condensed water as liquid. Hours where ERA5 has a little liquid
    above its floor may be called mixed by ARM.
  * Lidar is attenuated by liquid and the radar misses small ice, so the ARM
    phase mask and ERA5's cloud scheme do not see the same ice.

HOW HOURS ARE COUNTED
=====================
Main figure: hours per month for a typical grid cell, area weighted,

        H = sum_t n(t) / N(t)     [h]

with N(t) the cos(lat)-weighted count of the class's valid cells at hour t and
n(t) that of the cells in the state. UTQ is one cell, so its numbers are the
same in every weighting. The CSV also holds equal-weight per-cell hours and
the Figure 7 count (hours with at least one cell of the class in the state).
All are hours in each month, October to March, mean over seasons; a class that
does not exist in a month contributes zero.

THE ARM REFERENCE
=================
ARM_ICE_GT1H_H and ERA5_FIG7_ICE_H are the numbers printed above the boxes of
Figure 7a in the 2026-11-02 draft, read off the figure, not computed here.
The ERA5 ones must match the fig7 variant at UTQ (printed as a check).

CHECK
=====
For the fig7 variant, the 'any cell' count must equal the number of filled
ice_only hours in the run's hourly-values files
(era5_hourly_values_csv*/ERA5_DLR_perHour_<span>_<class>.csv), for every
month, season and region. The run stops if it does not.

REGIONS
=======
As for make_era5_clear_sky_sensitivity.py: Barrow locally with no options,
the Rui region on Casper (raw data there only), e.g.

    ./submit.sh --region Rui-region --years 2014-2024 --csvs ice_only --out <run folder>

The UTQ panel is the same cell in every domain.

USAGE
=====
    python make_era5_iceOnly_sensitivity.py              # Barrow, ~1 min on the laptop
    python make_era5_iceOnly_sensitivity.py --plot-only  # redraw from the CSV

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

from cloud_classification import fraction_phase_masks  # noqa: E402
from make_era5_hour_count_csvs import OUT_PRODUCT, SETTINGS, build_args, git_describe  # noqa: E402

# The streaming pass (with the persistence carry), the table builder and the
# Figure 7 check are shared with the clear-sky script, so the two cannot drift.
from make_era5_clear_sky_sensitivity import (  # noqa: E402
    GRID_COLOR,
    INK_MUTED,
    INK_PRIMARY,
    MONTHS,
    REGIONS,
    accumulate,
    build_table,
    check_against_hourly_csvs,
    find_hourly_dir,
    month_values,
    season_month_mean,
)

# ----------------------------------------------------------------------------
# What is computed
# ----------------------------------------------------------------------------
# Each variant: the lower limit on tcc (fraction) and whether an adjacent hour
# must also be ice-only. The phase floors and IWP/CWP limit come from SETTINGS.
# ice_min: IWP/CWP limit (None = SETTINGS' ice_fraction_min, 0.90).
# tp_max: precipitation limit in mm hr-1, an hour AT or above is dropped as in
# the project (None = SETTINGS' precip_rate_max, 0.05).
VARIANTS = {
    "fig7": dict(tcc_min=0.95, persist=False, ice_min=None, tp_max=None,
                 label="Figure 7: tcc ≥ 0.95 at the hour stamp"),
    "fig7_gt1h": dict(tcc_min=0.95, persist=True, ice_min=None, tp_max=None,
                      label="tcc ≥ 0.95, also at an adjacent hour (> 1 h)"),
    "tcc80_gt1h": dict(tcc_min=0.80, persist=True, ice_min=None, tp_max=None,
                       label="tcc ≥ 0.80, also at an adjacent hour (> 1 h)"),
    # Stricter phase and precipitation tests, closer to Genie's "no liquid, no
    # precipitation at any level" (user's choices, 2026-11-17).
    "gt1h_ice99": dict(tcc_min=0.95, persist=True, ice_min=0.99, tp_max=None,
                       label="> 1 h, tcc ≥ 0.95, IWP/CWP ≥ 0.99"),
    "gt1h_tp001": dict(tcc_min=0.95, persist=True, ice_min=None, tp_max=0.01,
                       label="> 1 h, tcc ≥ 0.95, tp < 0.01 mm/hr"),
    "gt1h_ice99_tp001": dict(tcc_min=0.95, persist=True, ice_min=0.99, tp_max=0.01,
                             label="> 1 h, tcc ≥ 0.95, IWP/CWP ≥ 0.99, tp < 0.01 mm/hr"),
}

# Figure 7a (draft OV_SLCFrequency_20261102), UTQ, ice-only hours per month,
# read from the numbers above the boxes. ARM is "> 1 h"; ERA5 is the snapshot rule.
ARM_ICE_GT1H_H = {"Oct": 2, "Nov": 3, "Dec": 4, "Jan": 5, "Feb": 3, "Mar": 9}
ERA5_FIG7_ICE_H = {"Oct": 44, "Nov": 70, "Dec": 155, "Jan": 198, "Feb": 194, "Mar": 162}

# Colours: Figure 7 draws ice in blue, so the tcc >= 0.95 pair is blue and the
# tcc >= 0.80 pair purple; within each pair light = snapshot, dark = > 1 h.
VARIANT_COLORS = {
    "fig7": "#9ecae1",
    "fig7_gt1h": "#2171b5",
    "tcc80_gt1h": "#6a51a3",
    # The stricter phase / precipitation variants: green (phase), orange
    # (precipitation), and a dark red-brown for both together.
    "gt1h_ice99": "#41ab5d",
    "gt1h_tp001": "#fd8d3c",
    "gt1h_ice99_tp001": "#8c2d04",
}

# Which counts get a figure. The area-weighted per-cell count is the one asked
# for; the Figure 7 count is drawn too so the any-cell numbers can be seen.
COUNT_NOTES = {
    "per_cell_area": "Hours per month for a typical grid cell, area weighted (cos latitude)",
    "any_cell": "Hours per month with at least one ice-only cell of the class, as Figure 7 counts",
}


def ice_only_hits(fields: dict, spec: dict, args: argparse.Namespace) -> np.ndarray:
    """Cells ice-only under one variant at the hour stamp, (t, lat, lon).

    Same tests as make_era5_hourly_values_csvs.py: valid, not raining, tcc at
    or above the variant's limit, and fraction_phase_masks' ice class (IWP/CWP
    >= ice_min with both paths floored), which already requires cloud water
    above the floors. A variant's tp_max adds a stricter precipitation test on
    top of the base one, written the same way (an hour AT or above is dropped;
    a missing tp does not drop the hour).
    """
    ice_min = args.ice_fraction_min if spec["ice_min"] is None else spec["ice_min"]
    phase = fraction_phase_masks(fields["lwp_g_m2"], fields["iwp_g_m2"],
                                 args.liquid_fraction_min, ice_min,
                                 args.min_lwp, args.min_iwp)
    hit = fields["base"] & (fields["tcc"] >= spec["tcc_min"]) & phase["ice"]
    if spec["tp_max"] is not None:
        rate = fields["rate_mm_hr"]                                   # mm hr-1
        hit &= ~(np.isfinite(rate) & (rate >= spec["tp_max"]))
    return hit


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def plot(df: pd.DataFrame, path: Path, count: str, n_seasons: int, domain: str) -> Path:
    """Figure 7 layout: one panel per region, months on the x axis, one bar per variant."""
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(12.0, 13.0), sharex=True)
    n_var = len(VARIANTS)
    width = 0.84 / n_var
    x_month = np.arange(len(MONTHS))
    for p, (ax, (region, region_label)) in enumerate(zip(axes, REGIONS.items())):
        all_values = np.array([month_values(df, count, region, v) for v in VARIANTS])  # (variant, month)
        empty_month = (all_values == 0.0).all(axis=0)                                 # (month,)
        top = float(all_values.max())
        for v, variant in enumerate(VARIANTS):
            x = x_month + (v - (n_var - 1) / 2) * width
            ax.bar(x, all_values[v], width, color=VARIANT_COLORS[variant], edgecolor="white",
                   linewidth=0.6, label=VARIANTS[variant]["label"])
            for m, (xi, value) in enumerate(zip(x, all_values[v])):
                if not empty_month[m]:
                    ax.text(xi, value, f"{value:.0f}", ha="center", va="bottom", fontsize=6.5,
                            color=INK_PRIMARY, zorder=4)
        for m in np.flatnonzero(empty_month):
            ax.text(x_month[m], 0, "0 h under every definition", ha="center", va="bottom",
                    fontsize=8, color=INK_MUTED)
        if region == "UTQ":
            # ARM ice-only (> 1 h) per month: a dashed line across that month's bars,
            # with its value printed at the right end so the small numbers are readable.
            half = 0.44
            for xm, month_name in zip(x_month, MONTHS.values()):
                arm_h = ARM_ICE_GT1H_H[month_name]
                ax.plot([xm - half, xm + half], [arm_h, arm_h], color=INK_PRIMARY,
                        linestyle="--", linewidth=1.3, zorder=5,
                        label="ARM ice only > 1 h (numbers in Fig. 7a)" if xm == 0 else None)
                ax.text(xm + half, arm_h, f" {arm_h}", ha="left", va="center", fontsize=7.5,
                        color=INK_PRIMARY, zorder=5)
        ax.set_ylim(0, max(top, 1.0) * 1.15)
        ax.text(0.005, 0.96, f"({'abcd'[p]}) {region_label}", transform=ax.transAxes,
                ha="left", va="top", fontsize=13, color=INK_PRIMARY)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", colors=INK_MUTED)
    axes[-1].set_xticks(x_month)
    axes[-1].set_xticklabels(list(MONTHS.values()), fontsize=13, color=INK_PRIMARY)
    fig.supylabel("Ice-only hours per month", fontsize=13, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.995),
               title="ERA5 ice-only definition (unless stated: IWP/CWP ≥ 0.90, paths floored at "
                     "0.01 g m⁻², tp < 0.05 mm/hr)", title_fontsize=10)
    fig.text(0.01, 0.003,
             f"{domain}. {COUNT_NOTES[count]}. Mean of {n_seasons} seasons. Months when a "
             f"class does not exist count as zero.", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.92))
    fig.savefig(path, dpi=350)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Files and entry point
# ----------------------------------------------------------------------------
def names(span: str) -> dict:
    """Output file names: one CSV, and one figure per drawn count."""
    stem = f"ERA5_iceOnlyHours_perMonth_{span}_definitionSensitivity"
    return {"csv": f"{stem}.csv",
            "fig": {"per_cell_area": f"{stem}_areaWeighted.png",
                    "any_cell": f"{stem}_anyCellCount.png"}}


def find_span(out_dir: Path) -> str:
    """The season span tag of the CSV already in ``out_dir``."""
    prefix, suffix = "ERA5_iceOnlyHours_perMonth_", "_definitionSensitivity.csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {prefix}<span>{suffix} in {out_dir}; run without --plot-only")
    return found[0].name[len(prefix) : -len(suffix)]


def draw(out_dir: Path, span: str, domain: str) -> list[Path]:
    """The figures, from the CSV only."""
    df = pd.read_csv(out_dir / names(span)["csv"])
    n_seasons = len([c for c in df.columns if "/" in c])
    return [plot(df, out_dir / name, count, n_seasons, domain)
            for count, name in names(span)["fig"].items()]


def write_readme(out_dir: Path, span: str, n_seasons: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = names(span)
    text = (
        f"Generated {stamp} by make_era5_iceOnly_sensitivity.py (repo {git_describe()})\n"
        f"Seasons: {n_seasons}.\n\n"
        f"{n['csv']}\n    Ice-only hours per month by count method (per_cell_area, "
        "per_cell_uniform, any_cell), region, variant, month, season\n"
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
                   help="output directory (default <hour-count folder>/ice_only_sensitivity)")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the figures from the CSV already in the output folder")
    p.add_argument("--hourly-dir", type=Path, default=None,
                   help="hourly-values folder for the Figure 7 check (default: the run's own)")
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    out_dir = opts.out_dir or hour_count_dir / "ice_only_sensitivity"

    if opts.plot_only:
        for path in draw(out_dir, find_span(out_dir), run.domain.name):
            print(f"  -> {path}")
        return

    print(run.report())
    # Thresholds (floors, IWP/CWP limit, precipitation, class limits) from SETTINGS.
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": "uniform"})
    A = accumulate(args, VARIANTS, ice_only_hits)
    span = ov.span_tag(A["seasons"])
    df = build_table(A, VARIANTS)
    check_against_hourly_csvs(df, span, find_hourly_dir(hour_count_dir, opts.hourly_dir,
                                                        run.domain.name),
                              column="ice_only", variant="fig7")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / names(span)["csv"], index=False, float_format="%.4f")
    write_readme(out_dir, span, len(A["seasons"]))
    for path in draw(out_dir, span, run.domain.name):
        print(f"  -> {path}")

    # UTQ against the numbers printed in Figure 7a.
    utq = pd.DataFrame({v: month_values(df, "per_cell_area", "UTQ", v) for v in VARIANTS},
                       index=list(MONTHS.values()))
    utq.insert(0, "fig7a_ERA5_printed", [ERA5_FIG7_ICE_H[m] for m in MONTHS.values()])
    utq.insert(0, "ARM_gt1h", [ARM_ICE_GT1H_H[m] for m in MONTHS.values()])
    print("\nUTQ ice-only hours per month (mean of seasons)")
    print(utq.round(1).to_string())

    for count in ("per_cell_area", "per_cell_uniform", "any_cell"):
        table = pd.DataFrame({
            REGIONS[r]: [season_month_mean(df, count, r, v) for v in VARIANTS] for r in REGIONS
        }, index=list(VARIANTS))
        print(f"\nIce-only hours per month (Oct-Mar mean), {count}")
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
