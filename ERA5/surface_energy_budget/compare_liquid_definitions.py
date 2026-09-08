#!/usr/bin/env python3
"""Cloud hours under two definitions of liquid: tclw against tcslw.

THE QUESTION
============
How much cloud time is lost -- or gained -- by counting only SUPERCOOLED liquid
water (``tcslw``) instead of all cloud liquid water (``tclw``)? Every other
threshold is held fixed, so any difference is attributable to the swap alone.

WHY "LOST" IS THE WRONG WORD, AND THE FIGURES SAY SO
====================================================
Physically, supercooled liquid is liquid below 0 C and ought to be a subset of
cloud liquid. In ERA5's ARCHIVED SINGLE-LEVEL FIELDS it is not. Over the Barrow
strip, ``tcslw`` exceeds ``tclw`` by more than the 0.031 g m-2 storage quantum
in 22.5% of cells in January 2023 and 38.5% in November 2022, reaching
+361 g m-2, with a p95 ratio near 3.5. Three candidate explanations were tested
and rejected -- see the note at ``LIQUID_VARS`` in the histogram module for the
numbers.

The consequence for this comparison is concrete: the change in cloud hours is
SIGNED. Some scenes drop out of a liquid category because their supercooled
path falls under the floor; others enter it because their supercooled path is
larger. So these figures report a signed difference and a percent change, never
a "loss", and :func:`print_signed_report` breaks the two directions apart.

WHAT IS HELD FIXED
==================
Both runs share --min-cloud-fraction, --min-lwp, --min-iwp,
--liquid-fraction-min, --ice-fraction-min, the season window, the region and
the precipitation setting. :func:`check_comparable` enforces that and raises
rather than letting a stray difference be read as a supercooled-water effect.
"""

from __future__ import annotations

import calendar

import numpy as np

import plot_lwp_histogram_by_surface_class as lwph
from plot_lwp_histogram_by_surface_class import (
    GENIE_EXCLUDED_MONTHS,
    LIQUID_VAR_LABEL,
    LIQUID_VAR_SHORT,
    PHASE_COLORS,
    PHASE_LABELS,
    _sh,
    excluded_month_mask,
    nanmean_quiet,
    resolve_series_code,
    season_phase_hours,
    window_label,
)

# Categories drawn in the three-way figure, bottom of the stack first.
THREE_WAY: tuple[str, ...] = ("liquid", "mixed", "ice")
# ...and in the two-way figure, where liquid-only and mixed are merged.
TWO_WAY: tuple[str, ...] = ("liquid_containing", "ice")

TWO_WAY_COLORS = {"liquid_containing": "red", "ice": "blue"}
TWO_WAY_LABELS = {"liquid_containing": "liquid containing", "ice": "ice only"}

# The tcslw bar is stippled, the way ERA5 is marked apart from observations in
# the Genie comparison, so the two notebooks read the same way.
TCSLW_HATCH = "////"

# Options that must agree for the difference to mean what the figure says.
SHARED_OPTIONS: tuple[str, ...] = (
    "region", "season_start", "season_end", "min_cloud_fraction",
    "min_lwp", "min_iwp", "liquid_fraction_min", "ice_fraction_min",
    "phase_mode", "no_precip", "precip_var", "precip_rate_max",
)


def check_comparable(A_all, A_sc) -> None:
    """Raise unless the two runs differ ONLY in --liquid-var.

    A difference in any other threshold would show up in these figures as a
    supercooled-water effect, which is exactly the mistake worth making
    impossible rather than merely unlikely.
    """
    if A_all.args.liquid_var == A_sc.args.liquid_var:
        raise ValueError(
            f"both runs use --liquid-var {A_all.args.liquid_var}; the "
            f"comparison needs one tclw run and one tcslw run")
    bad = []
    for name in SHARED_OPTIONS:
        a, b = getattr(A_all.args, name, None), getattr(A_sc.args, name, None)
        if a != b:
            bad.append(f"{name}: {a!r} vs {b!r}")
    if list(A_all.used) != list(A_sc.used):
        bad.append(f"seasons: {A_all.used} vs {A_sc.used}")
    if bad:
        raise ValueError("the two runs differ in more than --liquid-var, so "
                         "the difference would not be attributable to the "
                         "liquid definition:\n  " + "\n  ".join(bad))


# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------
def season_categories(A, surface_class: str = "arm_site"):
    """Hours per season in each drawn category, for one run.

    Returns ``(labels, {category: (n_season,) hours}, season_h)``. ``ice``
    absorbs the "no phase" residual exactly as the Genie-comparison figures do,
    so the categories still sum to the overcast total.
    """
    labels, hours, _cloudy, season_h = season_phase_hours(A)
    code, series_label = resolve_series_code(A.col, surface_class)
    i = {p: lwph.SEASON_STACK_ORDER.index(p) for p in lwph.SEASON_STACK_ORDER}
    h = hours[:, code, :]
    out = {
        "liquid": h[:, i["liquid"]],
        "mixed": h[:, i["mixed"]],
        "ice": h[:, i["ice"]] + h[:, i["none"]],
    }
    out["liquid_containing"] = out["liquid"] + out["mixed"]
    return labels, out, season_h, series_label


def month_categories(A, surface_class: str = "arm_site",
                     exclude_months=GENIE_EXCLUDED_MONTHS):
    """Monthly MEAN hours per category, averaged over seasons.

    Uses the module's own monthly route, so the excluded season-months and the
    per-season month lengths are handled identically to the Genie comparison.
    """
    col = A.col
    code, series_label = resolve_series_code(col, surface_class)
    frac = col["month_fraction"]["per_season"]        # (s, month, class, phase)
    month_h2 = col.get("month_hours")
    scale = (lwph.month_window_hours(A.sec["slots"])[None, :]
             if month_h2 is None else month_h2)
    i = {p: lwph.PHASE_ORDER_ACC.index(p) for p in lwph.PHASE_ORDER_ACC}
    per = {
        "liquid": frac[:, :, code, i["liquid"]] * scale,
        "mixed": frac[:, :, code, i["mixed"]] * scale,
        "ice": (frac[:, :, code, i["ice"]] + frac[:, :, code, i["none"]]) * scale,
    }
    per["liquid_containing"] = per["liquid"] + per["mixed"]
    if exclude_months:
        drop, _hit = excluded_month_mask(A.used, col["months"], A.args,
                                         exclude_months)
        per = {k: np.where(drop, np.nan, v) for k, v in per.items()}
    means = {k: nanmean_quiet(v, axis=0) for k, v in per.items()}
    sds = {k: _nanstd_quiet(v, axis=0) for k, v in per.items()}
    n = np.isfinite(per["liquid"]).sum(axis=0)
    return col["months"], means, sds, n, series_label


def _nanstd_quiet(a, axis=None):
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


def percent_change(new, old):
    """100 x (supercooled - all liquid) / all liquid, NaN where the base is 0."""
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(old > 0, 100.0 * (new - old) / np.where(old > 0, old, 1.0),
                        np.nan)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _save(fig, A, out_dir, stem, dpi):
    from pathlib import Path
    if out_dir is None:
        return fig
    path = Path(out_dir) / f"{A.args.region}_{stem}_{A.tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def _cat_style(cat):
    if cat in TWO_WAY_COLORS:
        return TWO_WAY_COLORS[cat], TWO_WAY_LABELS[cat]
    return PHASE_COLORS[cat], PHASE_LABELS[cat].lower()


def _grouped_stack(ax, x, w, cats, left, right, off_l, off_r):
    """Two stacked bars per group: tclw solid on the left, tcslw stippled."""
    for series, off, hatch in ((left, off_l, None), (right, off_r, TCSLW_HATCH)):
        bottom = np.zeros(len(x))
        for cat in cats:
            color, lab = _cat_style(cat)
            ax.bar(x + off, series[cat], width=w, bottom=bottom,
                   color="white" if hatch else color,
                   edgecolor=color if hatch else "white",
                   hatch=hatch, linewidth=0.7 if hatch else 0.4)
            bottom = bottom + series[cat]
    return bottom


def _legend(ax, cats, fontsize, loc="upper right"):
    import matplotlib.pyplot as plt
    handles, labels = [], []
    for cat in cats:
        color, lab = _cat_style(cat)
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="none"))
        labels.append(lab)
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.55", ec="none"))
    labels.append(f"{LIQUID_VAR_SHORT['tclw']} (left, solid)")
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.35", lw=0.8,
                                 hatch=TCSLW_HATCH))
    labels.append(f"{LIQUID_VAR_SHORT['tcslw']} (right, striped)")
    ax.legend(handles, labels, fontsize=fontsize, ncol=2, framealpha=0.9,
              loc=loc)


def _threshold_box(ax, A, loc="upper left", fontsize=9.0):
    pk = A.phase_kw
    lines = [f"min cloud fraction = {100.0 * A.args.min_cloud_fraction:g}%"]
    if pk["mode"] == "fraction":
        lines.append(f"ice only: IWP/(LWP+IWP) $\\geq$ "
                     f"{100.0 * pk['ice_fraction_min']:g}%")
        lines.append(f"liquid only: LWP/(LWP+IWP) $\\geq$ "
                     f"{100.0 * pk['liquid_fraction_min']:g}%")
    lines.append(f"min LWP {pk.get('min_lwp_g', float('nan')):g} / "
                 f"IWP {pk.get('min_iwp_g', float('nan')):g} g m$^{{-2}}$")
    place = {"upper left": (0.005, 1.10, "left", "top"),
             "upper right": (0.995, 1.10, "right", "top")}[loc]
    ax.text(place[0], place[1], "\n".join(lines), transform=ax.transAxes,
            ha=place[2], va=place[3], fontsize=fontsize, linespacing=1.4,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="0.55", linewidth=0.8, alpha=0.94))


def fig_season_liquid_definitions(A_all, A_sc, cats=THREE_WAY, out_dir=None,
                                  dpi=None, surface_class="arm_site",
                                  label_fontsize=13.0, tick_fontsize=11.5,
                                  legend_fontsize=10.5):
    """Hours per season under both liquid definitions, with percent change.

    Upper panel: two stacked bars per season -- all cloud liquid (tclw) solid on
    the left, supercooled liquid (tcslw) stippled on the right.

    Lower panel: percent change per category, ``100 x (tcslw - tclw) / tclw``.
    Signed, because the swap is not a strict narrowing -- see the module
    docstring.
    """
    import matplotlib.pyplot as plt

    check_comparable(A_all, A_sc)
    labels, left, season_h, series_label = season_categories(A_all, surface_class)
    _l2, right, _sh2, _sl2 = season_categories(A_sc, surface_class)

    x = np.arange(len(labels))
    w = 0.38
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(2.0 + 1.35 * len(labels), 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    _grouped_stack(ax, x, w, cats, left, right, -w / 2, +w / 2)

    tot_l = sum(left[c] for c in cats)
    tot_r = sum(right[c] for c in cats)
    top = float(max(np.nanmax(tot_l), np.nanmax(tot_r))) * 1.34
    for xi in x:
        for off, tot in ((-w / 2, tot_l), (+w / 2, tot_r)):
            if not np.isfinite(tot[xi]) or tot[xi] <= 0:
                continue
            ax.text(xi + off, tot[xi] + 0.012 * top,
                    f"{tot[xi]:,.0f} h\n"
                    f"{100.0 * tot[xi] / _sh(season_h, xi):.1f}%",
                    ha="center", va="bottom", fontsize=label_fontsize - 5.0,
                    linespacing=1.15)
    ax.set_ylim(0, top)
    ax.set_ylabel("Hours per season", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _legend(ax, cats, legend_fontsize, loc="upper right")
    _threshold_box(ax, A_all, loc="upper left", fontsize=legend_fontsize - 1.5)

    # ---- percent-change panel ---------------------------------------------
    n_c = len(cats)
    bw = 0.8 / n_c
    for k, cat in enumerate(cats):
        color, lab = _cat_style(cat)
        d = percent_change(right[cat], left[cat])
        ax_r.bar(x + (k - (n_c - 1) / 2) * bw, d, width=bw * 0.9, color=color,
                 edgecolor="white", linewidth=0.5, label=lab)
    ax_r.axhline(0.0, color="0.3", lw=1.0)
    ax_r.set_ylabel("100 $\\times$ (tcslw $-$ tclw) / tclw   [%]",
                    fontsize=label_fontsize - 1.0)
    ax_r.tick_params(axis="both", labelsize=tick_fontsize)
    ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax_r.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_r.spines[sp].set_visible(False)
    ax_r.legend(fontsize=legend_fontsize, ncol=n_c, framealpha=0.9,
                loc="lower right")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)

    kind = "three categories" if len(cats) == 3 else "two categories"
    fig.suptitle(
        f"Cloud hours under two definitions of liquid — {series_label}"
        f"\n{A_all.args.region}   |   {kind}   |   season window "
        f"{window_label(season_h)}   |   {lwph.precip_label(A_all.args)}",
        fontsize=12.5, y=0.965)
    fig.subplots_adjust(top=0.90, bottom=0.115, left=0.09, right=0.985)
    stem = f"liquid_defs_season_{len(cats)}cat"
    return _save(fig, A_all, out_dir, stem, dpi)


def fig_month_liquid_definitions(A_all, A_sc, cats=THREE_WAY, out_dir=None,
                                 dpi=None, surface_class="arm_site",
                                 exclude_months=GENIE_EXCLUDED_MONTHS,
                                 label_fontsize=13.0, tick_fontsize=11.5,
                                 legend_fontsize=10.5):
    """Monthly mean hours under both liquid definitions, with percent change.

    The season axis collapsed: for each calendar month, the mean over seasons of
    that month's hours in each category. Whiskers on the total are +/- one
    standard deviation ACROSS SEASONS -- interannual variability, not
    uncertainty in the mean.

    Excluded season-months and per-season month lengths follow the same rules as
    the Genie comparison, so the two notebooks count the same hours.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    check_comparable(A_all, A_sc)
    months, left, sd_l, n_l, series_label = month_categories(
        A_all, surface_class, exclude_months)
    _m2, right, sd_r, _n2, _s2 = month_categories(A_sc, surface_class,
                                                  exclude_months)
    n_lo, n_hi = int(np.min(n_l)), int(np.max(n_l))
    n_txt = f"{n_hi}" if n_lo == n_hi else f"{n_lo}–{n_hi}"

    x = np.arange(len(months))
    w = 0.38
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(2.0 + 1.7 * len(months), 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    _grouped_stack(ax, x, w, cats, left, right, -w / 2, +w / 2)

    tot_l = sum(left[c] for c in cats)
    tot_r = sum(right[c] for c in cats)
    # Spread of the TOTAL, so the whisker matches the bar it sits on.
    sd_tot_l = np.sqrt(sum(sd_l[c] ** 2 for c in cats))
    sd_tot_r = np.sqrt(sum(sd_r[c] ** 2 for c in cats))
    ax.errorbar(x - w / 2, tot_l, yerr=sd_tot_l, fmt="none", ecolor="0.15",
                elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)
    ax.errorbar(x + w / 2, tot_r, yerr=sd_tot_r, fmt="none", ecolor="0.15",
                elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)

    top = float(max(np.nanmax(tot_l + sd_tot_l),
                    np.nanmax(tot_r + sd_tot_r))) * 1.30
    for xi in x:
        for off, tot, sd in ((-w / 2, tot_l, sd_tot_l), (+w / 2, tot_r, sd_tot_r)):
            if not np.isfinite(tot[xi]) or tot[xi] <= 0:
                continue
            ax.text(xi + off, tot[xi] + sd[xi] + 0.012 * top,
                    f"{tot[xi]:,.0f} h", ha="center", va="bottom",
                    fontsize=label_fontsize - 4.5)
    ax.set_ylim(0, top)
    ax.set_ylabel("Mean hours per month", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _legend(ax, cats, legend_fontsize, loc="upper right")
    _threshold_box(ax, A_all, loc="upper left", fontsize=legend_fontsize - 1.5)
    ax.add_artist(ax.legend(
        [Line2D([0], [0], color="0.15", lw=1.4)],
        [f"$\\pm$1 s.d. across {n_txt} seasons"],
        fontsize=legend_fontsize, loc="upper center", framealpha=0.9))

    n_c = len(cats)
    bw = 0.8 / n_c
    for k, cat in enumerate(cats):
        color, lab = _cat_style(cat)
        d = percent_change(right[cat], left[cat])
        ax_r.bar(x + (k - (n_c - 1) / 2) * bw, d, width=bw * 0.9, color=color,
                 edgecolor="white", linewidth=0.5, label=lab)
    ax_r.axhline(0.0, color="0.3", lw=1.0)
    ax_r.set_ylabel("100 $\\times$ (tcslw $-$ tclw) / tclw   [%]",
                    fontsize=label_fontsize - 1.0)
    ax_r.tick_params(axis="both", labelsize=tick_fontsize)
    ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax_r.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_r.spines[sp].set_visible(False)
    ax_r.legend(fontsize=legend_fontsize, ncol=n_c, framealpha=0.9,
                loc="lower right")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([calendar.month_abbr[m] for m in months],
                         fontsize=tick_fontsize)

    kind = "three categories" if len(cats) == 3 else "two categories"
    fig.suptitle(
        f"Monthly mean cloud hours under two definitions of liquid — "
        f"{series_label}\n{A_all.args.region}   |   {kind}   |   mean over "
        f"{n_txt} seasons   |   {lwph.precip_label(A_all.args)}",
        fontsize=12.5, y=0.965)
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.09, right=0.985)
    stem = f"liquid_defs_month_{len(cats)}cat"
    return _save(fig, A_all, out_dir, stem, dpi)


# ----------------------------------------------------------------------------
# The signed accounting
# ----------------------------------------------------------------------------
def print_signed_report(A_all, A_sc, surface_class="arm_site") -> None:
    """Season-by-season table, and the two directions kept apart.

    Calling the difference a "loss" would hide that some seasons gain hours.
    The record-mean line therefore reports the gains and the losses separately
    as well as the net, so a small net cannot disguise large offsetting moves.
    """
    check_comparable(A_all, A_sc)
    labels, left, season_h, series_label = season_categories(A_all, surface_class)
    _l, right, _s, _sl = season_categories(A_sc, surface_class)

    print(f"{series_label}   |   {LIQUID_VAR_LABEL['tclw']} vs "
          f"{LIQUID_VAR_LABEL['tcslw']}")
    print(f"{'season':<10}" + "".join(
        f"{c[:12]:>26}" for c in ("liquid only", "mixed phase", "liq containing")))
    print(f"{'':<10}" + "".join(f"{'tclw':>8}{'tcslw':>9}{'%':>9}"
                                for _ in range(3)))
    for i, lab in enumerate(labels):
        row = f"{lab:<10}"
        for cat in ("liquid", "mixed", "liquid_containing"):
            a, b = left[cat][i], right[cat][i]
            row += f"{a:>8,.0f}{b:>9,.0f}{percent_change(b, a):>+8.1f}%"
        print(row)

    print()
    for cat in ("liquid", "mixed", "ice", "liquid_containing"):
        a, b = left[cat], right[cat]
        d = b - a
        gain = float(abs(d[d > 0].sum()))
        loss = float(abs(d[d < 0].sum()))
        _color, lab = _cat_style(cat)
        print(f"  {lab:<18} mean {a.mean():>7,.0f} -> {b.mean():>7,.0f} h"
              f"   net {d.mean():>+7,.0f} h/season "
              f"({percent_change(b.mean(), a.mean()):+.1f}%)"
              f"   [+{gain / len(d):,.0f} gained, -{loss / len(d):,.0f} lost "
              f"per season]")
    print("\n  Gains and losses are reported apart because tcslw is NOT "
          "nested inside tclw in\n  the archived fields -- see the note at "
          "LIQUID_VARS. A small net can hide large\n  offsetting moves.")


# ----------------------------------------------------------------------------
# The forecast-lead structure of tcslw
# ----------------------------------------------------------------------------
# tclw and tcslw are NOT the same kind of product:
#
#   tclw   the 4D-Var analysis at that hour (step 0), constrained by every
#          observation in the 12-hour assimilation window.
#   tcslw  a free-running short forecast initialised at 06 and 18 UTC, at a
#          lead of 1 to 12 hours.
#
# So the lead is a deterministic function of the UTC hour: step 1 at 07 and
# 19 UTC, rising to step 12 at 18 and 06 UTC. MEASURED over 20 winter months of
# the Barrow strip, RMS |tcslw - tclw| tracks that exactly -- 3.5 g m-2 at step
# 1 rising monotonically to 10.3 at step 12, with the correlation falling
# 0.9905 -> 0.9055, and a 2.9x discontinuity across both 18->19 and 06->07.
# Spearman against lead: rho = +0.96 on RMS, -0.97 on correlation.
#
# Consequence: most of the apparent disagreement between the two fields is
# FORECAST ERROR, not a difference between supercooled and total liquid.
# Eighteen hours in twenty-four carry a lead of 4 or more. Restricting to short
# leads is the only way to isolate the liquid definition, which is what
# hours_for_lead is for.
FORECAST_INIT_HOURS: tuple[int, ...] = (6, 18)
MAX_FORECAST_LEAD = 12


def forecast_lead(utc_hour):
    """Forecast step of the tcslw field at a given UTC hour, 1 to 12."""
    h = np.asarray(utc_hour)
    return ((h - (FORECAST_INIT_HOURS[0] + 1)) % MAX_FORECAST_LEAD) + 1


def hours_for_lead(max_lead: int) -> tuple[int, ...]:
    """UTC hours whose tcslw forecast lead is at most ``max_lead``.

    ``hours_for_lead(3)`` gives 07, 08, 09, 19, 20, 21 -- the six hours where
    tcslw is closest to an analysis. Pass the result to ``--utc-hours``.
    """
    if not 1 <= max_lead <= MAX_FORECAST_LEAD:
        raise ValueError(f"max_lead must be 1-{MAX_FORECAST_LEAD}")
    return tuple(h for h in range(24) if forecast_lead(h) <= max_lead)


def lead_diagnostic(region="barrow", months=("202111", "202112", "202201",
                                             "202202", "202211", "202212",
                                             "202301", "202302"),
                    data_dir=None, quantum_g=0.031):
    """Measure |tcslw - tclw| against UTC hour, straight from the files.

    Returns a dict of per-hour arrays. Deliberately independent of the analysis
    pipeline -- it reads the raw fields -- so it tests the forecast-lead
    hypothesis without any of this module's phase logic in the way.
    """
    import glob
    import xarray as xr
    from pathlib import Path

    root = Path(data_dir) if data_dir else Path("data") / region
    files = sorted(f for f in glob.glob(str(root / "*.nc"))
                   if any(f"_{m}" in f for m in months))
    if not files:
        raise FileNotFoundError(f"no files under {root} matching {months}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    times = np.asarray(ds["valid_time"].values)
    hour = times.astype("datetime64[h]").astype(np.int64) % 24
    lw = np.asarray(ds["tclw"].values) * 1000.0
    sl = np.asarray(ds["tcslw"].values) * 1000.0

    hrs, rms, corr, exc, bias, n = [], [], [], [], [], []
    for h in range(24):
        m = hour == h
        if not m.any():
            continue
        a, b = lw[m].ravel(), sl[m].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        d = b - a
        hrs.append(h)
        rms.append(float(np.sqrt((d ** 2).mean())))
        corr.append(float(np.corrcoef(a, b)[0, 1]))
        exc.append(float(100 * np.mean(b > a + quantum_g)))
        bias.append(float(d.mean()))
        n.append(int(ok.sum()))
    hrs = np.array(hrs)
    return {"hour": hrs, "lead": forecast_lead(hrs), "rms": np.array(rms),
            "corr": np.array(corr), "exceed_pct": np.array(exc),
            "bias": np.array(bias), "n": np.array(n),
            "n_files": len(files), "n_times": int(times.size)}


def fig_forecast_lead(diag, out_dir=None, dpi=200, label_fontsize=12.0):
    """The sawtooth: error against UTC hour, and against forecast lead.

    Left panel is the falsifiable prediction -- two identical ramps a day with
    a cliff at 18->19 and 06->07. Right panel collapses the two ramps onto the
    lead axis; if the hypothesis holds they land on one curve.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    h, lead, rms, corr = diag["hour"], diag["lead"], diag["rms"], diag["corr"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    ax.plot(h, rms, "o-", color="#1f77b4", lw=1.8, ms=6, label="RMS |tcslw $-$ tclw|")
    for b in (6.5, 18.5):
        ax.axvline(b, color="0.35", ls="--", lw=1.2)
    ax.text(18.6, ax.get_ylim()[1] * 0.96, "forecast\nre-initialised",
            fontsize=8.5, color="0.35", va="top")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("UTC hour", fontsize=label_fontsize)
    ax.set_ylabel("RMS difference [g m$^{-2}$]", fontsize=label_fontsize)
    ax.set_title("Two ramps a day, resetting at 06 and 18 UTC", fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    a2 = ax.twinx()
    a2.plot(h, corr, "s--", color="#d62728", lw=1.2, ms=4.5, alpha=0.8)
    a2.set_ylabel("correlation with tclw", color="#d62728",
                  fontsize=label_fontsize - 1)
    a2.tick_params(axis="y", colors="#d62728")

    for lo, hi, mark, lab in ((7, 18, "o", "init 06 UTC"),
                              (19, 6, "^", "init 18 UTC")):
        sel = ((h >= 7) & (h <= 18)) if lo == 7 else ((h >= 19) | (h <= 6))
        ax2.plot(lead[sel], rms[sel], mark, ms=8, alpha=0.8, label=lab)
    ax2.set_xlabel("forecast lead [h]", fontsize=label_fontsize)
    ax2.set_ylabel("RMS difference [g m$^{-2}$]", fontsize=label_fontsize)
    ax2.set_title("Both ramps collapse onto one curve", fontsize=11.5)
    ax2.set_xticks(range(1, 13))
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9.5)

    fig.suptitle("tcslw is a forecast, tclw is an analysis: the difference "
                 "grows with forecast lead", fontsize=13, y=1.00)
    fig.tight_layout()
    if out_dir is not None:
        path = Path(out_dir) / "tcslw_forecast_lead_diagnostic.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig
