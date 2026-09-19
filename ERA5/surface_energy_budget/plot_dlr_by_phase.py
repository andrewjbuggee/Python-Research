#!/usr/bin/env python3
"""Downwelling longwave radiation by sky state -- liquid-containing cloud,
ice-only cloud and clear sky -- as monthly distributions, at Utqiagvik and
over the whole Barrow domain, plus the distribution of the flux itself.

Built for the Ocean Visions talk (28 Sep 2026), as the counterpart to the
hour-count figures in ``comparison_with_Genie_Obs.ipynb``: those ask how OFTEN
ERA5 puts liquid cloud over the ARM site, this asks what that liquid cloud
DOES to the surface longwave budget, hour by hour, in the same population.

Sky states
==========
Three states per hour, from the same fields and thresholds the hour-count
figures use (``A.args`` and ``A.phase_kw``):

    liquid containing   tcc >= min_cloud_fraction, cloud water above the
                        minimum paths, and IWP/CWP <  ice_fraction_min
                        (liquid-only plus mixed-phase, as everywhere else)
    ice only            tcc >= min_cloud_fraction, cloud water above the
                        minimum paths, and IWP/CWP >= ice_fraction_min
    clear sky           LWP <= min_lwp AND IWP <= min_iwp, and
                        tcc <  clear_tcc_max (0.05 by default)

The states are exclusive and do not cover every hour. Partly cloudy hours
(clear_tcc_max <= tcc < min_cloud_fraction), cloudy hours with condensate
but below the overcast gate, and overcast hours with NO cloud water above
the floors all fall outside the three boxes; the reports count them.

That last group -- overcast but empty -- is the one place this module departs
from the hour-count figures. There it is folded into ice-only so that liquid
and ice sum to the overcast total. Here it is left out by default, because a
scene with no condensate radiates like clear sky, not like ice cloud, and
folding it in would pull the ice-only box toward the clear one.
``fold_no_phase_into_ice=True`` restores the hour-count convention.

Precipitation
=============
With ``no_precip`` on the run, hours at or above ``precip_rate_max`` (tp, in
mm hr-1) are removed from EVERY state before anything is drawn -- the same
filter as the seasonal comparison, applied to the same hours. The domain
pass accumulates both the filtered and the unfiltered populations, so the
distribution figure can show all hours while the box figures follow the run.

Two populations
===============
ARM cell     one grid cell, every hour: exact values from the per-hour table
             of ``fit_cloud_thresholds.extract_site_series`` with the flux
             (``msdwlwrf``, W m-2, hour-mean, positive down) carried along --
             the table behind ``fig_monthly_lwp_box`` and the duration figure,
             so the liquid-containing population here is, hour for hour, the
             one those figures draw (``verify_against_fit`` asserts it).
domain       every cell of the region, every hour: ~120 million cell-hours
             for eleven seasons, far too many to hold, so one streaming pass
             accumulates HISTOGRAMS of the flux at 1 W m-2 resolution per
             month and sky state (``prepare_domain``). Quantiles are read off
             the histograms by linear interpolation within a bin, so a box
             edge is exact to +/-0.5 W m-2, which no reader can see.
             Cell-hours are cos(latitude)-weighted when the run's
             ``cell_weighting`` is ``area`` (the default), so a fraction of
             hours is a fraction of the domain's AREA-time, and the northern
             half-size cells do not count double.

Why box plots of hourly values
==============================
One sample per hour: a month's box summarises thousands of hours across the
kept seasons. The box is the interquartile range, the line inside it the
median, the whiskers the full range with nothing drawn as an outlier (as in
the module's other box figures). The question the figure answers is "how much
warmer is the surface's longwave environment under liquid cloud than under
ice cloud or clear sky, and how much do the populations overlap" -- a
distribution question, which a mean and a standard deviation would flatten.

The distribution of the flux
============================
Arctic winter downwelling longwave is bimodal at the surface: a radiatively
clear mode and an opaque-cloud mode, with the atmosphere spending little time
between them (Stramler, Del Genio and Rossow 2011, J. Climate, from SHEBA;
the same two states at N-ICE2015 in Graham et al. 2017, J. Climate).
Reanalyses have been found to under-populate the clear mode over Arctic sea
ice (Graham et al. 2019, JGR Atmospheres, six reanalyses including ERA5).
``fig_dlr_pdf`` puts ERA5's distribution at the ARM cell beside the whole
domain's so the shape can be judged directly; ``show_states=True`` splits
the domain curve by sky state to show which state fills the gap.

Usage (notebook)
================
    import plot_dlr_by_phase as dlr
    S = dlr.extract_site_table(A_precip)          # ~25 s, once per run
    dlr.fig_monthly_dlr_box(A_precip, S, out_dir=SAVE_DIR)
    dlr.print_monthly_dlr_table(A_precip, S)

    D = dlr.prepare_domain(A_precip)              # one pass over every cell
    dlr.fig_monthly_dlr_box_domain(A_precip, D, out_dir=SAVE_DIR)
    dlr.print_state_fraction_table(A_precip, D)
    dlr.fig_dlr_pdf(A_precip, D, out_dir=SAVE_DIR)
"""

from __future__ import annotations

import calendar
from types import SimpleNamespace

import numpy as np

import fit_cloud_thresholds as fit
import plot_lwp_histogram_by_surface_class as lwph
from surface_classification import (CLASS_CODES, CLASS_COLORS, CLASS_LABELS,
                                    CLASS_ORDER, UNCLASSIFIED, area_weights_2d,
                                    classify_cells, iter_time_blocks)

# The archive's downwelling longwave flux: hour-mean surface downward
# long-wave radiation, W m-2, positive downward (era5_seb_variables.py).
DLR_VAR = "msdwlwrf"

# Clear sky needs an upper bound on cloud cover as well as no condensate: a
# scene with tcc = 0.5 and column water below the floors is a thin or
# sub-grid cloud, not a clear sky. 0.05 is the complement of the 0.95 overcast
# gate.
DEFAULT_CLEAR_TCC_MAX = 0.05

STATE_ORDER: tuple[str, ...] = ("liquid", "ice", "clear")
STATE_LABELS: dict[str, str] = {
    "liquid": "liquid containing",
    "ice": "ice only",
    "clear": "clear sky",
}
# Liquid and ice in the deck's phase colours, so the boxes read against the
# bars of fig_era5_vs_obs_simple_forOV; clear sky in grey.
CLEAR_COLOR = "0.55"
STATE_COLORS: dict[str, str] = {
    "liquid": lwph.GENIE_LIQUID_COLOR,
    "ice": lwph.GENIE_ICE_COLOR,
    "clear": CLEAR_COLOR,
}

# Whiskers at the full range, no outliers drawn -- the module convention.
DEFAULT_WHIS: tuple[float, float] = (0.0, 100.0)

# A box backed by fewer hours than this is drawn hatched: its quartiles are
# not a statistic of the month's sky state so much as a list of the hours
# that happened to qualify. Clear sky at the ARM cell needs this -- ERA5 has
# tcc < 0.05 there in about 1% of cold-season hours, so some months hold a
# handful of clear hours.
DEFAULT_MIN_HOURS = 30

# The domain pass bins the flux at this resolution, over this range. Arctic
# cold-season values run roughly 100-350 W m-2; the range is generous so that
# nothing is dropped, and the report says if anything was.
DLR_BIN_W_M2 = 1.0
DLR_BIN_EDGES = np.arange(0.0, 500.0 + DLR_BIN_W_M2, DLR_BIN_W_M2)

# What the domain pass accumulates, in order. "all" is every valid hour, the
# three states as above, "no_phase" the overcast-but-empty hours, "other" the
# remainder (partly cloudy). The two filters are every hour and the hours the
# precipitation filter keeps.
GROUPS: tuple[str, ...] = ("all", "liquid", "ice", "clear", "no_phase", "other")
FILTERS: tuple[str, ...] = ("all", "nonprecip")

# ERA5's clear-sky downwelling longwave: the same atmosphere with its cloud
# removed, archived every hour beside the all-sky flux. Their difference is
# the surface longwave cloud radiative effect of that hour, and it is the
# cleanest control for "similar environmental conditions" the archive offers:
# every cloudy hour is compared with its own air mass, not with a clear hour
# somewhere else. Over clear cells it comes out within +/-0.5 W m-2 of zero.
DLR_CS_VAR = "msdwlwrfcs"
CRE_BIN_W_M2 = 1.0
CRE_BIN_EDGES = np.arange(-100.0, 250.0 + CRE_BIN_W_M2, CRE_BIN_W_M2)

# Per-day histograms for the daily-difference method are kept coarser, since
# there are two thousand days: 2 W m-2 puts a daily median within a bin.
DAILY_BIN_W_M2 = 2.0
DAILY_BIN_EDGES = np.arange(0.0, 500.0 + DAILY_BIN_W_M2, DAILY_BIN_W_M2)

# The matched-environment method bins every cloudy hour by its clear-sky
# flux -- the emission of its own column without the cloud, i.e. its
# air-mass state -- and compares liquid and ice CRE within a bin. The
# clear-sky flux runs roughly 100-320 W m-2 in the Arctic cold season; 10
# W m-2 bins resolve the air-mass state without starving the bins. The CRE
# axis of that 2-D histogram is coarser than CRE_BIN_EDGES to keep it small.
CS_BIN_W_M2 = 10.0
CS_BIN_EDGES = np.arange(50.0, 400.0 + CS_BIN_W_M2, CS_BIN_W_M2)
MATCH_CRE_BIN_W_M2 = 2.0
MATCH_CRE_EDGES = np.arange(-100.0, 250.0 + MATCH_CRE_BIN_W_M2, MATCH_CRE_BIN_W_M2)

# How the between-state differences are formed for the ver2 slide figure.
#   median_diff  the monthly medians differenced, one value per season
#   daily_diff   the daily medians over a class's cells differenced, one
#                value per day (the "same day, same surface" pairing)
#   cre          each state's cloud effect against ERA5's own clear-sky flux
#                (DLR - DLR_clearsky), medians differenced per season
#   cre_matched  the same, but within bins of the clear-sky flux, so liquid
#                and ice cloud are compared in the same air-mass state; the
#                bins are averaged with the liquid hours' weights
DIFF_METHODS: tuple[str, ...] = ("median_diff", "daily_diff", "cre", "cre_matched")
DEFAULT_DIFF_METHOD = "median_diff"


# ----------------------------------------------------------------------------
# The ARM-cell per-hour table
# ----------------------------------------------------------------------------
def extract_site_table(A, **kwargs) -> dict:
    """The ARM-cell per-hour table with the downwelling longwave carried along.

    ``fit_cloud_thresholds.extract_site_series(A, extra_vars=(DLR_VAR,))``:
    one streaming pass over the run's seasons. The result is a superset of the
    table ``fig_monthly_lwp_box`` and the duration figure take, so it can be
    passed to them as ``S=`` too.
    """
    extra = tuple(kwargs.pop("extra_vars", ()))
    if DLR_VAR not in extra:
        extra = (DLR_VAR,) + extra
    return fit.extract_site_series(A, extra_vars=extra, **kwargs)


def _require_dlr(S: dict) -> None:
    if DLR_VAR not in S:
        raise KeyError(f"the per-hour table has no {DLR_VAR!r}; build it with "
                       f"plot_dlr_by_phase.extract_site_table(A) or "
                       f"fit.extract_site_series(A, extra_vars=({DLR_VAR!r},))")


# ----------------------------------------------------------------------------
# Sky states
# ----------------------------------------------------------------------------
def _classify(tcc, has_cloud, ice_frac, finite, raining,
              tcc_min: float, ice_frac_min: float, clear_tcc_max: float,
              fold_no_phase_into_ice: bool, no_precip: bool) -> dict:
    """The sky-state masks from already-floored inputs, any array shape.

    Shared by the ARM-cell table and the domain pass so the two populations
    are classified by one piece of code. ``finite`` is where every input is
    usable; ``raining`` where the precipitation filter would drop the hour;
    ``has_cloud`` where either water path is above its floor (a species under
    its floor having been zeroed upstream).
    """
    rain = (finite & raining) if no_precip else np.zeros_like(finite)
    valid = finite & ~rain
    cloudy = valid & (tcc >= tcc_min)
    with np.errstate(invalid="ignore"):
        is_ice = has_cloud & (ice_frac >= ice_frac_min)
    liquid = cloudy & has_cloud & ~is_ice
    no_phase = cloudy & ~has_cloud
    ice = cloudy & is_ice
    if fold_no_phase_into_ice:
        ice = ice | no_phase
    # has_cloud is False exactly when both paths sit at or below their floors,
    # so "below the minimum LWP and IWP thresholds" is ~has_cloud.
    clear = valid & ~has_cloud & (tcc < clear_tcc_max)
    other = valid & ~(liquid | ice | clear)
    return {"liquid": liquid, "ice": ice, "clear": clear,
            "_valid": valid, "_no_phase": no_phase, "_other": other,
            "_raining": rain}


def sky_state_masks(S: dict, A, clear_tcc_max: float = DEFAULT_CLEAR_TCC_MAX,
                    fold_no_phase_into_ice: bool = False) -> dict:
    """Boolean masks over the ARM-cell table's hours, one per sky state.

    Returns ``{"liquid", "ice", "clear"}`` plus bookkeeping keys:
    ``"_valid"`` (hours with finite inputs that survived the precipitation
    filter), ``"_no_phase"`` (overcast, no cloud water above the floors),
    ``"_other"`` (valid but in none of the three states) and
    ``"_raining"`` (hours the filter removed).

    The liquid-containing mask is, by construction, ``fit._phase_hit(...,
    "liquid")`` restricted to hours with a finite flux; :func:`verify_against_fit`
    checks that.
    """
    _require_dlr(S)
    args = A.args
    tcc_min = float(args.min_cloud_fraction)
    if not 0.0 <= clear_tcc_max <= tcc_min:
        raise ValueError(f"clear_tcc_max must lie in [0, min_cloud_fraction="
                         f"{tcc_min:g}], got {clear_tcc_max}")
    finite = S["valid"] & np.isfinite(S[DLR_VAR])
    return _classify(S["tcc"], S["has_cloud"], S["ice_frac"], finite,
                     S["raining"], tcc_min,
                     float(A.phase_kw["ice_fraction_min"]), clear_tcc_max,
                     fold_no_phase_into_ice, bool(args.no_precip))


def verify_against_fit(S: dict, A) -> None:
    """Assert the liquid-containing hours are the fit's, hour for hour.

    The comparison figures' liquid-containing population at the ARM cell is
    ``fit._phase_hit``; this module's must be the same hours, less any with a
    missing flux (none, in practice). Raises AssertionError otherwise.
    """
    m = sky_state_masks(S, A)
    ref = fit._phase_hit(S, float(A.args.min_cloud_fraction),
                         float(A.phase_kw["ice_fraction_min"]),
                         bool(A.args.no_precip), "liquid")
    ref = ref & np.isfinite(S[DLR_VAR])
    if not np.array_equal(m["liquid"], ref):
        raise AssertionError("liquid-containing hours differ from "
                             "fit._phase_hit; the two definitions have drifted")
    print(f"  verified: {int(ref.sum()):,} liquid-containing hours match "
          f"fit_cloud_thresholds hour for hour")


# ----------------------------------------------------------------------------
# ARM cell: monthly distributions
# ----------------------------------------------------------------------------
def monthly_dlr_distributions(A, S: dict, exclude_months=(),
                              clear_tcc_max: float = DEFAULT_CLEAR_TCC_MAX,
                              fold_no_phase_into_ice: bool = False):
    """Every hour's downwelling longwave at the ARM cell, by month and state.

    Returns ``(months, dist, counts)``: ``dist[state][month]`` is the array of
    hourly fluxes [W m-2]; ``counts`` holds, per month, the hours in each
    state and in the bookkeeping groups (``no_phase``, ``other``,
    ``raining``, ``valid``), so the report can say what the boxes leave out.

    ``exclude_months`` takes ``(calendar_year, month)`` pairs, e.g.
    ``lwph.GENIE_EXCLUDED_MONTHS`` to match section 7b's population. Empty by
    default: the ARM instrument outages say nothing about ERA5's radiation,
    and nothing here is compared against the ARM record.
    """
    masks = sky_state_masks(S, A, clear_tcc_max, fold_no_phase_into_ice)
    months = list(S["months"])
    keep = np.ones(S["si"].size, dtype=bool)
    if exclude_months:
        drop, _hit = lwph.excluded_month_mask(S["seasons"], months, A.args,
                                              exclude_months)
        keep &= ~drop[S["si"], S["mi"]]

    dlr = S[DLR_VAR]
    dist = {st: {} for st in STATE_ORDER}
    counts = {k: np.zeros(len(months), dtype=int)
              for k in (*STATE_ORDER, "no_phase", "other", "raining", "valid")}
    for j, m in enumerate(months):
        in_m = keep & (S["mi"] == j)
        for st in STATE_ORDER:
            sel = in_m & masks[st]
            dist[st][m] = dlr[sel]
            counts[st][j] = int(sel.sum())
        counts["no_phase"][j] = int((in_m & masks["_no_phase"]).sum())
        counts["other"][j] = int((in_m & masks["_other"]).sum())
        counts["raining"][j] = int((in_m & masks["_raining"]).sum())
        counts["valid"][j] = int((in_m & masks["_valid"]).sum())
    return months, dist, counts


# ----------------------------------------------------------------------------
# Shared drawing: three boxes per month from precomputed statistics
# ----------------------------------------------------------------------------
def _box_stats(values: np.ndarray, whis) -> dict | None:
    """``ax.bxp`` statistics from raw values; None for an empty array."""
    if values.size == 0:
        return None
    lo, hi = np.percentile(values, [whis[0], whis[1]])
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi,
            "fliers": []}


def _hist_quantiles(counts: np.ndarray, edges: np.ndarray, qs) -> np.ndarray:
    """Quantiles (percent) of a histogram, interpolated linearly within a bin.

    The cumulative count is treated as piecewise linear across each bin, so
    the answer is exact to within the bin width. ``counts`` may be weighted.
    """
    counts = np.asarray(counts, dtype=float)
    cum = np.cumsum(counts)
    total = cum[-1]
    out = np.full(len(qs), np.nan)
    if total <= 0:
        return out
    for k, q in enumerate(qs):
        target = q / 100.0 * total
        if q <= 0:                                # the lowest occupied bin
            i = int(np.argmax(counts > 0))
            out[k] = edges[i]
        elif q >= 100:                            # the highest occupied bin
            i = int(len(counts) - 1 - np.argmax(counts[::-1] > 0))
            out[k] = edges[i + 1]
        else:
            i = int(np.searchsorted(cum, target))
            i = min(i, len(counts) - 1)
            below = cum[i] - counts[i]
            frac = (target - below) / counts[i] if counts[i] > 0 else 0.0
            out[k] = edges[i] + frac * (edges[i + 1] - edges[i])
    return out


def _box_stats_from_hist(counts: np.ndarray, edges: np.ndarray, whis) -> dict | None:
    """``ax.bxp`` statistics from a histogram; None if it is empty."""
    if not np.any(counts > 0):
        return None
    lo, q1, med, q3, hi = _hist_quantiles(counts, edges,
                                          (whis[0], 25, 50, 75, whis[1]))
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi,
            "fliers": []}


def _fmt_count(n: float) -> str:
    """Hours for the labels under the boxes: 1,234 / 56k / 2.3M."""
    n = float(n)
    if n <= 0:
        return "—"
    if n < 1e4:
        return f"{n:,.0f}"
    if n < 1e6:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e6:.2f}M"


# The two between-state differences written above each month's boxes when
# show_differences is on: (minuend, subtrahend, mathtext label). Both take the
# liquid-containing median as the reference -- what liquid cloud adds over an
# ice-only sky, and over a clear one.
DIFF_ANNOTATIONS: tuple[tuple[str, str, str], ...] = (
    ("liquid", "ice", r"$\Delta\mathrm{DLR}_{\mathrm{glaciated}}$"),
    ("liquid", "clear", r"$\Delta\mathrm{DLR}_{\mathrm{clrsky}}$"),
)


def _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                          show_counts, box_alpha, tick_fontsize, legend_fontsize,
                          label_fontsize, ylabel, show_differences: bool = False,
                          draw_legend: bool = True):
    """Three boxes per month on ``ax`` from ``stats[state][j]`` (bxp dicts or
    None) and ``counts[state][j]`` (hours, for the labels and the hatching).

    With ``show_differences``, two lines are written above each month's
    boxes -- the liquid-containing median minus the ice-only median, and
    minus the clear-sky median (``DIFF_ANNOTATIONS``), in W m-2 without the
    unit -- in a band at a fixed height so the row reads straight across.
    The legend then moves below the axes, since the band takes the space it
    would have used; the caller keeps the threshold box out of the way too
    (:func:`_threshold_below`). ``draw_legend=False`` draws none, for a
    multi-row figure that wants one legend for all rows.

    Returns the legend handles and labels (drawn or not).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    x = np.arange(len(months))
    n_state = len(STATE_ORDER)
    w = 0.24                                    # box width, axis units
    offsets = (np.arange(n_state) - (n_state - 1) / 2.0) * (w + 0.04)

    lo_all, hi_all, any_thin = np.inf, -np.inf, False
    if all(s is None for st in STATE_ORDER for s in stats[st]):
        # Nothing at all in this population (a class absent from the run);
        # say so rather than fail on an empty min().
        ax.text(0.5, 0.5, "no hours in any state", transform=ax.transAxes,
                ha="center", va="center", fontsize=label_fontsize, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([calendar.month_abbr[m] for m in months],
                           fontsize=tick_fontsize)
        return [], []
    for st, off in zip(STATE_ORDER, offsets):
        color = STATE_COLORS[st]
        pos = [xi + off for xi, s in zip(x, stats[st]) if s is not None]
        data = [s for s in stats[st] if s is not None]
        n_h = [counts[st][j] for j, s in enumerate(stats[st]) if s is not None]
        if not data:
            continue
        lo_all = min(lo_all, min(s["whislo"] for s in data))
        hi_all = max(hi_all, max(s["whishi"] for s in data))
        bp = ax.bxp(data, positions=pos, widths=w, showfliers=False,
                    patch_artist=True, manage_ticks=False)
        for art, n in zip(bp["boxes"], n_h):
            thin = n < min_hours
            any_thin |= thin
            art.set_facecolor(to_rgba(color, box_alpha * (0.4 if thin else 1.0)))
            art.set_edgecolor(color)
            art.set_linewidth(1.4)
            if thin:
                art.set_hatch("////")
        for key in ("whiskers", "caps"):
            for art in bp[key]:
                art.set_color(color)
                art.set_linewidth(1.3)
        for art in bp["medians"]:
            art.set_color("black")
            art.set_linewidth(1.8)

    span = hi_all - lo_all
    # Room below the whiskers for the hour counts, and above for the legend
    # and the threshold box -- or, with the differences on, for the two-line
    # annotation band (the legend then sits below the axes).
    y_bottom = lo_all - (0.10 if show_counts else 0.04) * span
    ax.set_ylim(y_bottom, hi_all + (0.32 if show_differences else 0.26) * span)
    if show_differences:
        # x in data coordinates (the month), y in axes fraction, so the band
        # stays put whatever the whiskers do. Median differences straight
        # from the box statistics, so they are the tables' numbers.
        band = ax.get_xaxis_transform()
        for j, xi in enumerate(x):
            for k, (a, b, label) in enumerate(DIFF_ANNOTATIONS):
                sa, sb = stats[a][j], stats[b][j]
                txt = (f"{label} = {sa['med'] - sb['med']:+.1f}"
                       if sa is not None and sb is not None else f"{label} = \u2014")
                ax.text(xi, 0.985 - 0.075 * k, txt, transform=band,
                        ha="center", va="top", fontsize=tick_fontsize - 2.5,
                        color="black", zorder=6)
    if show_counts:
        y_txt = lo_all - 0.045 * span
        for st, off in zip(STATE_ORDER, offsets):
            for j, xi in enumerate(x):
                ax.text(xi + off, y_txt, _fmt_count(counts[st][j]),
                        ha="center", va="top", fontsize=tick_fontsize - 4.0,
                        color=STATE_COLORS[st])

    ax.set_ylabel(ylabel, fontsize=label_fontsize - 1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months],
                       fontsize=tick_fontsize)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=to_rgba(STATE_COLORS[st], box_alpha),
                             ec=STATE_COLORS[st], lw=1.4) for st in STATE_ORDER]
    labels = [STATE_LABELS[st] for st in STATE_ORDER]
    handles.append(Line2D([0], [0], color="black", lw=1.8))
    labels.append("median")
    if any_thin:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=to_rgba("0.5", 0.18),
                                     ec="0.5", lw=1.4, hatch="////"))
        labels.append(f"fewer than {min_hours} h")
    whis_txt = ("whiskers: min to max, no outliers drawn"
                if tuple(whis) == tuple(DEFAULT_WHIS)
                else f"whiskers: {whis[0]:g}th to {whis[1]:g}th percentile")
    counts_txt = ";  hours behind each box beneath it" if show_counts else ""
    if draw_legend:
        _state_legend(ax, handles, labels, whis_txt + counts_txt, legend_fontsize,
                      below=show_differences)
    return handles, labels


def _state_legend(ax, handles, labels, title, legend_fontsize, below: bool,
                  anchor=None, ncol: int | None = None):
    """The sky-state legend: in the upper-right corner, or below the axes,
    right-aligned, leaving the bottom-left for the threshold lines -- or, with
    ``anchor``, with its upper-right corner at that axes-fraction point (for
    a panel with an empty region to spare), in ``ncol`` columns."""
    kw = dict(title=title, title_fontsize=legend_fontsize - 1,
              fontsize=legend_fontsize, ncol=ncol or len(handles),
              framealpha=0.9, loc="upper right")
    if anchor is not None:
        kw.update(bbox_to_anchor=anchor, borderaxespad=0.0)
    elif below:
        kw.update(bbox_to_anchor=(1.0, -0.085), borderaxespad=0.0)
    ax.legend(handles, labels, **kw)


_THRESHOLD_BBOX = dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="0.55", linewidth=0.8, alpha=0.94)


def _threshold_below(ax, A, fontsize: float) -> None:
    """The threshold box, below the axes at the left -- the same lines
    :func:`lwph.draw_threshold_box` stamps in a corner, for figures whose
    corners are taken by the annotation band."""
    ax.text(0.0, -0.085, "\n".join(lwph.threshold_box_lines(A)),
            transform=ax.transAxes, ha="left", va="top", fontsize=fontsize,
            linespacing=1.4, zorder=6, bbox=_THRESHOLD_BBOX)


def _threshold_at(ax, A, fontsize: float, xy, ha: str, va: str) -> None:
    """The threshold box at an axes-fraction point, for a panel with room."""
    ax.text(xy[0], xy[1], "\n".join(lwph.threshold_box_lines(A)),
            transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            linespacing=1.4, zorder=6, bbox=_THRESHOLD_BBOX)


def _clear_txt(min_lwp_g: float, min_iwp_g: float, clear_tcc_max: float) -> str:
    return (f"clear sky: tcc < {clear_tcc_max:g}, LWP ≤ {min_lwp_g:g} and "
            f"IWP ≤ {min_iwp_g:g} g m$^{{-2}}$")


def _seasons_txt(A) -> str:
    return (f"{A.used[0]}/{(A.used[0] + 1) % 100:02d}–"
            f"{A.used[-1]}/{(A.used[-1] + 1) % 100:02d}")


def _variant_suffix(clear_tcc_max, fold_no_phase_into_ice, whis,
                    exclude_months=()) -> str:
    suffix = ("" if clear_tcc_max == DEFAULT_CLEAR_TCC_MAX
              else f" - clear-tcc{clear_tcc_max:g}")
    suffix += " - nophase-in-ice" if fold_no_phase_into_ice else ""
    suffix += " - months-excluded" if exclude_months else ""
    suffix += ("" if tuple(whis) == tuple(DEFAULT_WHIS)
               else f" - whis{whis[0]:g}-{whis[1]:g}")
    return suffix


# ----------------------------------------------------------------------------
# ARM cell: the figure and the numbers
# ----------------------------------------------------------------------------
def fig_monthly_dlr_box(A, S: dict, out_dir=None, dpi: int | None = None,
                        exclude_months=(),
                        clear_tcc_max: float = DEFAULT_CLEAR_TCC_MAX,
                        fold_no_phase_into_ice: bool = False,
                        whis=DEFAULT_WHIS,
                        min_hours: int = DEFAULT_MIN_HOURS,
                        show_counts: bool = True,
                        show_differences: bool = False,
                        box_alpha: float = 0.45,
                        label_fontsize: float = lwph.DEFAULT_COMPARISON_LABEL_FONTSIZE,
                        tick_fontsize: float = lwph.DEFAULT_COMPARISON_TICK_FONTSIZE,
                        legend_fontsize: float = lwph.DEFAULT_COMPARISON_LEGEND_FONTSIZE):
    """Box-and-whisker of hourly downwelling longwave at the ARM cell, by
    month and sky state.

    Per month, three boxes: liquid containing (red), ice only (blue), clear
    sky (grey), each over every such hour of that month across the run's
    seasons. Box = interquartile range, black line = median, whiskers per
    ``whis`` -- the full range by default, with nothing drawn as an outlier.
    ``show_counts`` writes the hours behind each box beneath it in the box's
    colour. A box with fewer than ``min_hours`` behind it is hatched and its
    fill faded, and the legend says so. ``show_differences`` writes the two
    median differences (liquid minus ice, liquid minus clear) above each
    month, as :func:`fig_monthly_dlr_box_domain` does by default; the legend
    and the threshold box then move below the axes. ``box_alpha`` is the fill
    opacity; the edge is drawn solid.

    ``clear_tcc_max``, ``fold_no_phase_into_ice`` and ``exclude_months`` are
    the definitions in :func:`sky_state_masks` and
    :func:`monthly_dlr_distributions`; a non-default choice of any of them is
    recorded in the file name.
    """
    import matplotlib.pyplot as plt

    args = A.args
    months, dist, counts = monthly_dlr_distributions(
        A, S, exclude_months, clear_tcc_max, fold_no_phase_into_ice)
    stats = {st: [_box_stats(np.asarray(dist[st][m], dtype=float), whis)
                  for m in months] for st in STATE_ORDER}

    fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.75 * len(months), 6.4))
    _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                          show_counts, box_alpha, tick_fontsize, legend_fontsize,
                          label_fontsize,
                          "downwelling longwave at the surface [W m$^{-2}$]",
                          show_differences=show_differences)
    if show_differences:
        _threshold_below(ax, A, legend_fontsize - 2.0)
    else:
        lwph.draw_threshold_box(ax, A, loc="upper left",
                                fontsize=legend_fontsize - 2.0)

    _drop, dropped_months = lwph.excluded_month_mask(
        A.used, months, args, exclude_months or ())
    if dropped_months:
        drop_txt = ", ".join(f"{calendar.month_abbr[m]} {y}"
                             for y, m in sorted(dropped_months))
        fig.text(0.09, 0.008, f"season-months excluded: {drop_txt}",
                 ha="left", va="bottom", fontsize=7.5, color="0.35")

    _code, series_label = lwph.resolve_series_code(A.col, "arm_site")
    fig.suptitle(f"Downwelling longwave by sky state, ERA5 — {series_label}\n"
                 f"{args.region}   |   every hour, {_seasons_txt(A)}   |   "
                 f"{lwph.precip_label(args)}   |   "
                 f"{_clear_txt(S['min_lwp_g'], S['min_iwp_g'], clear_tcc_max)}",
                 fontsize=12, y=0.965)
    fig.subplots_adjust(top=0.87, bottom=0.21 if show_differences else 0.10,
                        left=0.08, right=0.985)

    tag = "noprecip" if args.no_precip else "allsky"
    suffix = _variant_suffix(clear_tcc_max, fold_no_phase_into_ice, whis,
                             exclude_months)
    suffix += " - with-differences" if show_differences else ""
    return lwph._save_stack(fig, A, out_dir, f"monthly_dlr_box_by_phase_{tag}",
                            dpi, suffix=suffix)


def print_monthly_dlr_table(A, S: dict, exclude_months=(),
                            clear_tcc_max: float = DEFAULT_CLEAR_TCC_MAX,
                            fold_no_phase_into_ice: bool = False) -> dict:
    """The ARM-cell figure's numbers as text: per month and state, hours,
    median, interquartile range and mean of the downwelling longwave
    [W m-2], then the median differences between states, and what the boxes
    leave out.

    Returns ``{"months", "median", "q25", "q75", "mean", "n"}``, each keyed
    by state over months, for reuse.
    """
    months, dist, counts = monthly_dlr_distributions(
        A, S, exclude_months, clear_tcc_max, fold_no_phase_into_ice)

    stats = {k: {st: np.full(len(months), np.nan) for st in STATE_ORDER}
             for k in ("median", "q25", "q75", "mean")}
    for st in STATE_ORDER:
        for j, m in enumerate(months):
            d = dist[st][m]
            if d.size:
                stats["median"][st][j] = np.median(d)
                stats["q25"][st][j] = np.percentile(d, 25)
                stats["q75"][st][j] = np.percentile(d, 75)
                stats["mean"][st][j] = d.mean()

    _code, series_label = lwph.resolve_series_code(A.col, "arm_site")
    pair = (f"non-precipitating (tp < {S['precip_rate_max']:g} mm/hr)"
            if A.args.no_precip else "all sky")
    print(f"\n  Downwelling longwave at the surface [W m-2] by month and sky "
          f"state, ERA5 ({series_label})")
    print(f"  {pair}   |   {len(A.used)} seasons {_seasons_txt(A)}   |   "
          f"clear sky: tcc < {clear_tcc_max:g}, LWP <= {S['min_lwp_g']:g} and "
          f"IWP <= {S['min_iwp_g']:g} g m-2\n")
    diff = _print_state_stats(months, stats,
                              {st: counts[st] for st in STATE_ORDER})

    # What the boxes leave out. "other" is every valid hour in none of the
    # three states; the overcast-but-empty hours sit inside it unless they
    # were folded into ice only, so subtract them to isolate the partly
    # cloudy remainder.
    n_valid = int(counts["valid"].sum())
    n_in = int(sum(counts[st].sum() for st in STATE_ORDER))
    n_other = int(counts["other"].sum())
    n_no_phase = int(counts["no_phase"].sum())
    n_partly = n_other if fold_no_phase_into_ice else n_other - n_no_phase
    print(f"\n    hours after the filter {n_valid:,}; in the three boxes "
          f"{n_in:,} ({100.0 * n_in / n_valid:.1f}%); outside them "
          f"{n_valid - n_in:,}")
    print(f"      partly cloudy or non-overcast cloud ({clear_tcc_max:g} <= tcc "
          f"< {float(A.args.min_cloud_fraction):g}, or cloud water under a "
          f"partial sky): {n_partly:,}")
    print(f"      overcast without cloud water above the floors: "
          f"{n_no_phase:,}"
          f"{' (folded into ice only)' if fold_no_phase_into_ice else ' (left out)'}")
    if A.args.no_precip:
        print(f"    hours removed by the precipitation filter: "
              f"{int(counts['raining'].sum()):,}")
    return {"months": months, "n": {st: counts[st] for st in STATE_ORDER},
            "diff_median": diff, **stats}


# The between-state differences of the median flux, in the order they are
# printed. The first two are the talk's numbers -- what liquid cloud adds over
# ice cloud, and over a clear sky -- the third completes the triangle.
DIFF_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("liquid", "ice", "liq-ice"),
    ("liquid", "clear", "liq-clr"),
    ("ice", "clear", "ice-clr"),
)


def median_differences(median: dict) -> dict:
    """``median[state]`` arrays -> ``{"liq-ice": liquid - ice, ...}`` [W m-2].

    NaN wherever either state has no hours in that month.
    """
    return {name: np.asarray(median[a], dtype=float) - np.asarray(median[b], dtype=float)
            for a, b, name in DIFF_PAIRS}


def _print_state_stats(months, stats: dict, n: dict) -> dict:
    """The month x state block shared by the two DLR tables, with the
    between-state differences of the medians as the last three columns of
    every month's row. Returns those differences."""
    diff = median_differences(stats["median"])
    head = f"    {'month':<6}"
    for st in STATE_ORDER:
        head += f"{STATE_LABELS[st]:>34}"
    head += f"{'median differences [W m-2]':>29}"
    print(head)
    print(f"    {'':<6}" + "".join(
        f"{'hours':>9}{'median':>8}{'IQR':>12}{'mean':>5}" for _ in STATE_ORDER)
        + "".join(f"{name:>9}" for _a, _b, name in DIFF_PAIRS) + "  ")
    for j, m in enumerate(months):
        row = f"    {calendar.month_abbr[m]:<6}"
        for st in STATE_ORDER:
            if n[st][j] > 0 and np.isfinite(stats["median"][st][j]):
                row += (f"{_fmt_count(n[st][j]):>9}{stats['median'][st][j]:>8.1f}"
                        f"{stats['q25'][st][j]:>6.0f}-{stats['q75'][st][j]:<5.0f}"
                        f"{stats['mean'][st][j]:>5.0f}")
            else:
                row += f"{_fmt_count(n[st][j]):>9}{'--':>8}{'--':>12}{'--':>5}"
        for _a, _b, name in DIFF_PAIRS:
            v = diff[name][j]
            row += f"{v:>+9.1f}" if np.isfinite(v) else f"{'--':>9}"
        print(row)
    return diff


# ----------------------------------------------------------------------------
# The whole domain: one streaming pass, histograms per month and sky state
# ----------------------------------------------------------------------------
class DomainDLR(SimpleNamespace):
    """What :func:`prepare_domain` accumulates.

    ``hist[f, g, j, b]`` -- weighted cell-hours in flux bin ``b`` of month
    ``j`` for group ``g`` (``GROUPS``) under filter ``f`` (``FILTERS``), with
    ``n`` the same unweighted, and ``site_hist`` the ARM cell alone (one cell,
    so unweighted). ``hist_class[f, c, g, j, b]`` and ``n_class`` split the
    domain by surface class ``c`` (``CLASS_ORDER``: land, coastal, open
    ocean, marginal ice zone, sea ice, from the run's own thresholds);
    unclassified cell-hours are in ``hist`` but in no class. ``edges`` are
    the bin edges [W m-2]; ``months`` the calendar months in season order;
    ``dropped`` the cell-hours whose flux fell outside the binned range
    (should be zero).

    For the difference methods: ``hist_cs[f, s, c, g, j, b]`` is
    ``hist_class`` split by season ``s``; ``cre_hist[f, s, c, st, j, b]``
    bins the cloud radiative effect (``CRE_BIN_EDGES``) per season, class,
    state and month; ``daily_hist[f, d, c, st, b]`` bins the flux
    (``DAILY_BIN_EDGES``) per day ``d`` of the run, with ``day_month`` and
    ``day_season`` giving each day's month index and season index;
    ``match_hist[f, s, c, st, j, k, b]`` bins the cloud effect
    (``MATCH_CRE_EDGES``) against the clear-sky flux (``CS_BIN_EDGES``, bin
    ``k``) per season, class, state and month, for the matched method.
    """


def prepare_domain(A, clear_tcc_max: float = DEFAULT_CLEAR_TCC_MAX,
                   fold_no_phase_into_ice: bool = False) -> DomainDLR:
    """Stream the archive once and bin the flux for every cell and hour.

    Reuses ``A.ds`` and ``A.layout``, so the seasons, the window and the site
    cell are the run's own. Per block of hours, every cell-hour is classified
    with :func:`_classify` -- the same code as the ARM-cell table -- and its
    flux is binned at ``DLR_BIN_W_M2`` resolution into the histogram of its
    month, group and filter. Cell-hours are weighted by cos(latitude) when
    the run's ``cell_weighting`` is ``area``, so that a share of cell-hours is
    a share of the domain's area-time; ``n`` keeps the plain counts.

    Both the unfiltered and the precipitation-filtered populations are
    accumulated, whatever ``no_precip`` says, so the distribution figure can
    show every hour while the box figure follows the run. Every cell-hour is
    also assigned its surface class from the run's land-sea mask and the
    hour's sea ice concentration (``classify_cells``, the thresholds the
    run's classes use), and binned again per class, so the by-class figure
    costs nothing extra.
    """
    ds, args, layout = A.ds, A.args, A.layout
    tcc_min = float(args.min_cloud_fraction)
    if not 0.0 <= clear_tcc_max <= tcc_min:
        raise ValueError(f"clear_tcc_max must lie in [0, min_cloud_fraction="
                         f"{tcc_min:g}], got {clear_tcc_max}")
    ice_frac_min = float(A.phase_kw["ice_fraction_min"])
    min_lwp_g = float(A.phase_kw["min_lwp_g"])
    min_iwp_g = float(A.phase_kw["min_iwp_g"])
    rate_max = float(args.precip_rate_max)

    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    slots, seasons = layout["slots"], layout["seasons"]
    months, mi_of_slot = lwph.season_month_axis(slots)
    wanted = np.zeros(len(seasons), dtype=bool)
    wanted[A.keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    # Season index among the SELECTED seasons, per step, and a day index for
    # the run: one integer per (season, day of season) that carries an hour,
    # so the daily method can group a block's hours by day.
    n_season = len(A.keep_idx)
    remap = np.full(len(seasons), -1, dtype=np.intp)
    remap[A.keep_idx] = np.arange(n_season, dtype=np.intp)
    si_all = np.where(use_step, remap[np.clip(s_idx, 0, None)], -1)
    day_key = np.where(use_step, si_all * 400 + dos, -1)
    uniq_days, inv = np.unique(day_key[use_step], return_inverse=True)
    day_idx_all = np.full(use_step.size, -1, dtype=np.intp)
    day_idx_all[use_step] = inv
    n_day = uniq_days.size
    day_season = (uniq_days // 400).astype(np.intp)
    day_month = mi_of_slot[uniq_days % 400]

    site_mask, site_lat, site_lon = lwph.site_cell_mask(ds)
    iy, ix = (int(v) for v in np.argwhere(site_mask)[0])
    if getattr(args, "cell_weighting", "area") == "uniform":
        weights_2d = np.ones((ds.sizes["latitude"], ds.sizes["longitude"]))
    else:
        weights_2d = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])

    edges = DLR_BIN_EDGES
    n_bin = edges.size - 1
    n_month = len(months)
    shape = (len(FILTERS), len(GROUPS), n_month, n_bin)
    hist = np.zeros(shape)
    n = np.zeros(shape)
    site_hist = np.zeros(shape)
    n_class_codes = len(CLASS_ORDER)
    hist_class = np.zeros((len(FILTERS), n_class_codes, len(GROUPS), n_month, n_bin))
    n_class = np.zeros_like(hist_class)
    # Per season as well, for the seasonal spread of the differences; the
    # cloud effect per season; and the flux per day. All weighted like hist.
    n_st = len(STATE_ORDER)
    n_cre = CRE_BIN_EDGES.size - 1
    n_bin2 = DAILY_BIN_EDGES.size - 1
    hist_cs = np.zeros((len(FILTERS), n_season, n_class_codes, len(GROUPS),
                        n_month, n_bin))
    cre_hist = np.zeros((len(FILTERS), n_season, n_class_codes, n_st, n_month, n_cre))
    daily_hist = np.zeros((len(FILTERS), n_day, n_class_codes, n_st, n_bin2),
                          dtype=np.float32)
    n_cs = CS_BIN_EDGES.size - 1
    n_cre2 = MATCH_CRE_EDGES.size - 1
    match_hist = np.zeros((len(FILTERS), n_season, n_class_codes, n_st, n_month,
                           n_cs, n_cre2), dtype=np.float32)
    dropped = 0
    dropped_cre = 0
    dropped_cs = 0
    n_step = 0
    n_unclassified = 0

    read_vars = ["tcc", "tclw", "tciw", "tp", "siconc", DLR_VAR, DLR_CS_VAR]
    print(f"  Domain DLR pass: {ds.sizes['latitude']} x {ds.sizes['longitude']} "
          f"cells, {int(use_step.sum()):,} hours, bins of {DLR_BIN_W_M2:g} W m-2")
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        keep = use_step[i0:i0 + n_t]
        if not keep.any():
            continue
        tcc = block["tcc"].values[keep]                        # (t, y, x)
        lwp_g = block["tclw"].values[keep] * 1000.0            # kg m-2 -> g m-2
        iwp_g = block["tciw"].values[keep] * 1000.0
        rate = block["tp"].values[keep] * 1000.0               # m/h -> mm/hr
        dlr = block[DLR_VAR].values[keep]                      # W m-2
        dlr_cs = block[DLR_CS_VAR].values[keep]                # W m-2, clear sky
        mi_rows = mi_of_slot[dos[i0:i0 + n_t][keep]]           # (t,)
        si_rows = si_all[i0:i0 + n_t][keep]                    # (t,)
        day_rows = day_idx_all[i0:i0 + n_t][keep]              # (t,)
        n_step += int(keep.sum())
        # Surface class of every cell-hour, as the run's classes are cut.
        classes = classify_cells(
            A.lsm, block["siconc"].values[keep], args.lsm_tol,
            args.open_ocean_max_siconc, args.sea_ice_min_siconc,
            args.land_max_siconc)                              # (t, y, x) int8
        n_unclassified += int((classes == UNCLASSIFIED).sum())

        finite = (np.isfinite(tcc) & np.isfinite(lwp_g) & np.isfinite(iwp_g)
                  & np.isfinite(dlr))
        # Same flooring as fraction_phase_masks and extract_site_series.
        with np.errstate(invalid="ignore"):
            lwp_eff = np.where(finite & (lwp_g > min_lwp_g), lwp_g, 0.0)
            iwp_eff = np.where(finite & (iwp_g > min_iwp_g), iwp_g, 0.0)
        cwp = lwp_eff + iwp_eff
        has_cloud = finite & (cwp > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ice_frac = np.where(has_cloud, iwp_eff / np.where(cwp > 0, cwp, 1.0),
                                np.nan)
        raining = np.isfinite(rate) & (rate >= rate_max)

        # Classify twice: every hour, and with the precipitation filter on.
        masks_by_filter = tuple(
            _classify(tcc, has_cloud, ice_frac, finite, raining, tcc_min,
                      ice_frac_min, clear_tcc_max, fold_no_phase_into_ice,
                      no_precip=npf)
            for npf in (False, True))
        with np.errstate(invalid="ignore"):
            b = np.floor((dlr - edges[0]) / DLR_BIN_W_M2)
            cre = dlr - dlr_cs
            b_cre = np.floor((cre - CRE_BIN_EDGES[0]) / CRE_BIN_W_M2)
            b2 = np.floor((dlr - DAILY_BIN_EDGES[0]) / DAILY_BIN_W_M2)
        in_range = finite & (b >= 0) & (b < n_bin)
        dropped += int((finite & ~in_range).sum())
        b = np.where(in_range, b, 0).astype(np.intp)
        finite_cs = finite & np.isfinite(dlr_cs)
        in_range_cre = finite_cs & (b_cre >= 0) & (b_cre < n_cre)
        dropped_cre += int((finite_cs & ~in_range_cre).sum())
        b_cre = np.where(in_range_cre, b_cre, 0).astype(np.intp)
        in_range2 = finite & (b2 >= 0) & (b2 < n_bin2)
        b2 = np.where(in_range2, b2, 0).astype(np.intp)
        # The matched method's two axes: clear-sky flux, and CRE again on
        # its coarser grid.
        with np.errstate(invalid="ignore"):
            b_cs = np.floor((dlr_cs - CS_BIN_EDGES[0]) / CS_BIN_W_M2)
            b_cre2 = np.floor((cre - MATCH_CRE_EDGES[0]) / MATCH_CRE_BIN_W_M2)
        in_range_m = (finite_cs & (b_cs >= 0) & (b_cs < n_cs)
                      & (b_cre2 >= 0) & (b_cre2 < n_cre2))
        dropped_cs += int((finite_cs & ~in_range_m).sum())
        b_cs = np.where(in_range_m, b_cs, 0).astype(np.intp)
        b_cre2 = np.where(in_range_m, b_cre2, 0).astype(np.intp)
        w3 = np.broadcast_to(weights_2d, tcc.shape)
        cls3 = classes.astype(np.intp)

        # Month by month AND season by season within the block (a 30-day
        # block straddles at most one month boundary and never a season).
        sm_rows = si_rows * n_month + mi_rows
        for sm in np.unique(sm_rows):
            s_i, j = int(sm // n_month), int(sm % n_month)
            rows = sm_rows == sm
            for f, masks in enumerate(masks_by_filter):
                group_masks = (masks["_valid"], masks["liquid"], masks["ice"],
                               masks["clear"], masks["_no_phase"], masks["_other"])
                for g, gm in enumerate(group_masks):
                    sel = (gm & in_range)[rows]                 # (t_j, y, x)
                    if not sel.any():
                        continue
                    idx = b[rows][sel]
                    wsel = w3[rows][sel]
                    hist[f, g, j] += np.bincount(idx, weights=wsel,
                                                 minlength=n_bin)
                    n[f, g, j] += np.bincount(idx, minlength=n_bin)
                    # Per class in one bincount: key = class * n_bin + bin,
                    # over the classified cell-hours only.
                    cls = cls3[rows][sel]
                    ok = cls >= 0
                    key = cls[ok] * n_bin + idx[ok]
                    per_class = np.bincount(
                        key, weights=wsel[ok],
                        minlength=n_class_codes * n_bin).reshape(n_class_codes, n_bin)
                    hist_class[f, :, g, j] += per_class
                    hist_cs[f, s_i, :, g, j] += per_class
                    n_class[f, :, g, j] += np.bincount(
                        key, minlength=n_class_codes * n_bin
                    ).reshape(n_class_codes, n_bin)
                    sel_s = sel[:, iy, ix]
                    if sel_s.any():
                        site_hist[f, g, j] += np.bincount(
                            b[rows][:, iy, ix][sel_s], minlength=n_bin)
                # The cloud effect, per state, class and season.
                for st_i, st in enumerate(STATE_ORDER):
                    sel = (masks[st] & in_range_cre)[rows]
                    if not sel.any():
                        continue
                    cls = cls3[rows][sel]
                    ok = cls >= 0
                    key = cls[ok] * n_cre + b_cre[rows][sel][ok]
                    cre_hist[f, s_i, :, st_i, j] += np.bincount(
                        key, weights=w3[rows][sel][ok],
                        minlength=n_class_codes * n_cre).reshape(n_class_codes, n_cre)
                    # ... and against the clear-sky flux, for matching.
                    selm = (masks[st] & in_range_m)[rows]
                    if not selm.any():
                        continue
                    cls = cls3[rows][selm]
                    ok = cls >= 0
                    key = ((cls[ok] * n_cs + b_cs[rows][selm][ok]) * n_cre2
                           + b_cre2[rows][selm][ok])
                    match_hist[f, s_i, :, st_i, j] += np.bincount(
                        key, weights=w3[rows][selm][ok],
                        minlength=n_class_codes * n_cs * n_cre2
                    ).reshape(n_class_codes, n_cs, n_cre2).astype(np.float32)

        # The flux per day, state and class, for the daily method: one
        # bincount per (filter, state) over the whole block, keyed by the
        # day within the block, the class and the bin.
        uniq_d, d_local = np.unique(day_rows, return_inverse=True)
        n_d = uniq_d.size
        d3 = np.broadcast_to(d_local[:, None, None], tcc.shape)
        for f, masks in enumerate(masks_by_filter):
            for st_i, st in enumerate(STATE_ORDER):
                sel = masks[st] & in_range2 & (cls3 >= 0)
                if not sel.any():
                    continue
                key = (d3[sel] * n_class_codes + cls3[sel]) * n_bin2 + b2[sel]
                daily_hist[f, uniq_d, :, st_i, :] += np.bincount(
                    key, weights=w3[sel],
                    minlength=n_d * n_class_codes * n_bin2
                ).reshape(n_d, n_class_codes, n_bin2).astype(np.float32)

    if dropped:
        print(f"  WARNING: {dropped:,} cell-hours had a flux outside "
              f"[{edges[0]:g}, {edges[-1]:g}] W m-2 and were not binned")
    if dropped_cre:
        print(f"  WARNING: {dropped_cre:,} cell-hours had a cloud effect outside "
              f"[{CRE_BIN_EDGES[0]:g}, {CRE_BIN_EDGES[-1]:g}] W m-2 and were "
              f"not binned")
    if dropped_cs:
        print(f"  WARNING: {dropped_cs:,} cell-hours fell outside the matched "
              f"method's clear-sky range [{CS_BIN_EDGES[0]:g}, "
              f"{CS_BIN_EDGES[-1]:g}] W m-2 or its CRE range and were not binned")
    print(f"  done: {n_step:,} hours x {weights_2d.size:,} cells; "
          f"{n[0, 0].sum():,.0f} valid cell-hours, "
          f"{n[1, 0].sum():,.0f} after the precipitation filter; "
          f"{n_unclassified:,} cell-hours unclassified by surface")
    return DomainDLR(
        hist=hist, n=n, site_hist=site_hist, edges=edges, months=list(months),
        groups=list(GROUPS), filters=list(FILTERS), dropped=dropped,
        hist_class=hist_class, n_class=n_class, classes=list(CLASS_ORDER),
        n_unclassified=n_unclassified,
        hist_cs=hist_cs, cre_hist=cre_hist, cre_edges=CRE_BIN_EDGES,
        daily_hist=daily_hist, daily_edges=DAILY_BIN_EDGES,
        day_month=day_month, day_season=day_season, n_day=int(n_day),
        match_hist=match_hist, cs_edges=CS_BIN_EDGES, match_cre_edges=MATCH_CRE_EDGES,
        weighting=getattr(args, "cell_weighting", "area"),
        n_cells=int(weights_2d.size), site_lat=site_lat, site_lon=site_lon,
        clear_tcc_max=clear_tcc_max, fold_no_phase_into_ice=fold_no_phase_into_ice,
        min_lwp_g=min_lwp_g, min_iwp_g=min_iwp_g, precip_rate_max=rate_max,
        seasons=list(A.used),
    )


def _filter_index(D: DomainDLR, A, precip_filter: bool | None) -> tuple[int, str]:
    """Which accumulated population to read: the run's, or as asked."""
    if precip_filter is None:
        precip_filter = bool(A.args.no_precip)
    f = D.filters.index("nonprecip" if precip_filter else "all")
    label = (f"non-precipitating: tp < {D.precip_rate_max:g} mm hr$^{{-1}}$"
             if precip_filter else "all hours, no precipitation filter")
    return f, label


def _weighting_txt(D: DomainDLR) -> str:
    return ("cos-latitude weighted cell-hours" if D.weighting == "area"
            else "cell-hours, unweighted")


# ----------------------------------------------------------------------------
# Domain: the monthly box figure
# ----------------------------------------------------------------------------
def fig_monthly_dlr_box_domain(A, D: DomainDLR, out_dir=None,
                               dpi: int | None = None,
                               precip_filter: bool | None = None,
                               whis=DEFAULT_WHIS,
                               min_hours: int = DEFAULT_MIN_HOURS,
                               show_counts: bool = True,
                               show_differences: bool = True,
                               box_alpha: float = 0.45,
                               label_fontsize: float = lwph.DEFAULT_COMPARISON_LABEL_FONTSIZE,
                               tick_fontsize: float = lwph.DEFAULT_COMPARISON_TICK_FONTSIZE,
                               legend_fontsize: float = lwph.DEFAULT_COMPARISON_LEGEND_FONTSIZE):
    """:func:`fig_monthly_dlr_box` for every cell of the domain.

    Same layout and conventions; the statistics come from the histograms
    :func:`prepare_domain` accumulated (weighted as the run weights cells),
    so a quartile is exact to half a bin. The counts under the boxes are
    plain cell-hours. ``precip_filter`` is ``None`` to follow the run,
    otherwise a bool.

    ``show_differences`` (on by default here) writes above each month the two
    median differences the tables give -- the liquid-containing median minus
    the ice-only median, and minus the clear-sky median -- in mathtext, no
    unit; the legend and the threshold box move below the axes to make room.
    """
    import matplotlib.pyplot as plt

    f, filt_txt = _filter_index(D, A, precip_filter)
    months = D.months
    stats, counts = {}, {}
    for st in STATE_ORDER:
        g = D.groups.index(st)
        stats[st] = [_box_stats_from_hist(D.hist[f, g, j], D.edges, whis)
                     for j in range(len(months))]
        counts[st] = [D.n[f, g, j].sum() for j in range(len(months))]

    fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.75 * len(months), 6.4))
    _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                          show_counts, box_alpha, tick_fontsize, legend_fontsize,
                          label_fontsize,
                          "downwelling longwave at the surface [W m$^{-2}$]",
                          show_differences=show_differences)
    if show_differences:
        _threshold_below(ax, A, legend_fontsize - 2.0)
    else:
        lwph.draw_threshold_box(ax, A, loc="upper left",
                                fontsize=legend_fontsize - 2.0)
    fig.suptitle(f"Downwelling longwave by sky state, ERA5 — every cell of "
                 f"the {A.args.region} domain ({D.n_cells:,} cells, "
                 f"{_weighting_txt(D)})\n"
                 f"every hour, {_seasons_txt(A)}   |   {filt_txt}   |   "
                 f"{_clear_txt(D.min_lwp_g, D.min_iwp_g, D.clear_tcc_max)}",
                 fontsize=12, y=0.965)
    fig.subplots_adjust(top=0.87, bottom=0.21 if show_differences else 0.10,
                        left=0.08, right=0.985)

    tag = "noprecip" if D.filters[f] == "nonprecip" else "allsky"
    suffix = _variant_suffix(D.clear_tcc_max, D.fold_no_phase_into_ice, whis)
    suffix += "" if show_differences else " - no-differences"
    return lwph._save_stack(fig, A, out_dir,
                            f"monthly_dlr_box_by_phase_domain_{tag}", dpi,
                            suffix=suffix)


def fig_monthly_dlr_box_domain_forOV(A, D: DomainDLR, out_dir=None,
                                     dpi: int | None = None,
                                     precip_filter: bool | None = None,
                                     whis=DEFAULT_WHIS,
                                     min_hours: int = DEFAULT_MIN_HOURS,
                                     box_alpha: float = 0.5,
                                     figsize: tuple[float, float] = (13.33, 6.0),
                                     show_title: bool = True,
                                     label_fontsize: float = 15.0,
                                     tick_fontsize: float = 14.0):
    """The Ocean Visions slide version of :func:`fig_monthly_dlr_box_domain`.

    The same eighteen boxes, stripped for a slide: no difference annotations,
    no hour counts, no legend, no threshold box -- the boxes in the deck's
    colours alone, with the colours spelled out on the grey subtitle line so
    the figure still explains itself (``show_title=False`` drops both title
    lines for a slide that carries its own). A 16:9-ish canvas (``figsize``)
    and larger fonts, so it stays legible scaled onto a slide beside a
    title. Boxes with fewer than ``min_hours`` behind them are still hatched.
    """
    import matplotlib.pyplot as plt

    f, filt_txt = _filter_index(D, A, precip_filter)
    months = D.months
    stats, counts = {}, {}
    for st in STATE_ORDER:
        g = D.groups.index(st)
        stats[st] = [_box_stats_from_hist(D.hist[f, g, j], D.edges, whis)
                     for j in range(len(months))]
        counts[st] = [D.n[f, g, j].sum() for j in range(len(months))]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                          show_counts=False, box_alpha=box_alpha,
                          tick_fontsize=tick_fontsize, legend_fontsize=10.0,
                          label_fontsize=label_fontsize,
                          ylabel="downwelling longwave at the surface [W m$^{-2}$]",
                          show_differences=False, draw_legend=False)
    # The working figure's headroom was for the legend; a little is enough.
    lo = min(bs["whislo"] for st in STATE_ORDER for bs in stats[st] if bs is not None)
    hi = max(bs["whishi"] for st in STATE_ORDER for bs in stats[st] if bs is not None)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

    if show_title:
        fig.suptitle(f"Downwelling longwave by sky state, ERA5 \u2014 {A.args.region} "
                     f"domain, Oct\u2013Mar {_seasons_txt(A)}",
                     fontsize=label_fontsize + 1, y=0.985)
        fig.text(0.5, 0.915,
                 f"every cell, every hour   |   {filt_txt}   |   "
                 f"red: liquid containing, blue: ice only, grey: clear sky "
                 f"(tcc < {D.clear_tcc_max:g}, no cloud water)",
                 ha="center", va="top", fontsize=tick_fontsize - 2.0, color="0.4")
    fig.subplots_adjust(top=0.86 if show_title else 0.97, bottom=0.09,
                        left=0.075, right=0.99)

    tag = "noprecip" if D.filters[f] == "nonprecip" else "allsky"
    suffix = _variant_suffix(D.clear_tcc_max, D.fold_no_phase_into_ice, whis)
    suffix += "" if show_title else " - no-title"
    return lwph._save_stack(fig, A, out_dir,
                            f"monthly_dlr_box_by_phase_domain_OV_{tag}", dpi,
                            suffix=suffix)


# The three classes a talk has room for: the two ends of the surface spectrum
# and the open water in between, in the order they stack on the figure.
DEFAULT_BOX_CLASSES: tuple[str, ...] = ("land", "open_ocean", "sea_ice")


def fig_monthly_dlr_box_by_class(A, D: DomainDLR, out_dir=None,
                                 dpi: int | None = None,
                                 classes: tuple[str, ...] = DEFAULT_BOX_CLASSES,
                                 precip_filter: bool | None = None,
                                 whis=DEFAULT_WHIS,
                                 min_hours: int = DEFAULT_MIN_HOURS,
                                 show_counts: bool = True,
                                 show_differences: bool = True,
                                 legend_in: str | None = "open_ocean",
                                 box_alpha: float = 0.45,
                                 row_height: float = 4.6,
                                 label_fontsize: float = lwph.DEFAULT_COMPARISON_LABEL_FONTSIZE,
                                 tick_fontsize: float = lwph.DEFAULT_COMPARISON_TICK_FONTSIZE,
                                 legend_fontsize: float = lwph.DEFAULT_COMPARISON_LEGEND_FONTSIZE):
    """:func:`fig_monthly_dlr_box_domain`, one row per surface class.

    Each row is the domain figure restricted to the cell-hours of one class
    -- land, open ocean and sea ice by default (``classes``, any names from
    ``CLASS_ORDER``) -- from the per-class histograms of
    :func:`prepare_domain`, weighted as the run weights cells. Rows share the
    month axis; each has its own flux axis, since a class's range is its
    own. Row titles carry the class colours the deck already uses. One
    legend and one threshold box serve every row: inside the right half of
    the ``legend_in`` panel when that class is one of the rows -- open ocean
    by default, whose January-March columns are empty because the domain
    freezes over -- otherwise below the bottom row.

    A class absent from a month has no boxes there; a class absent from the
    run altogether says so in its row.
    """
    import matplotlib.pyplot as plt

    bad = [c for c in classes if c not in D.classes]
    if bad:
        raise ValueError(f"unknown classes {bad}; choose from {D.classes}")
    f, filt_txt = _filter_index(D, A, precip_filter)
    months = D.months
    n_rows = len(classes)
    fig, axes = plt.subplots(n_rows, 1, sharex=True,
                             figsize=(2.0 + 1.75 * len(months),
                                      row_height * n_rows + 1.6))
    axes = np.atleast_1d(axes)
    host = legend_in if legend_in in classes else None
    handles, labels = [], []
    for r, (ax, cname) in enumerate(zip(axes, classes)):
        c = D.classes.index(cname)
        stats, counts = {}, {}
        for st in STATE_ORDER:
            g = D.groups.index(st)
            stats[st] = [_box_stats_from_hist(D.hist_class[f, c, g, j], D.edges, whis)
                         for j in range(len(months))]
            counts[st] = [D.n_class[f, c, g, j].sum() for j in range(len(months))]
        last = r == n_rows - 1
        h, l = _draw_sky_state_boxes(
            ax, months, stats, counts, whis, min_hours, show_counts, box_alpha,
            tick_fontsize, legend_fontsize, label_fontsize,
            "DLR at the surface [W m$^{-2}$]",
            show_differences=show_differences, draw_legend=False)
        if h:                                    # any row with data will do
            handles, labels = h, l
        n_valid = D.n_class[f, c, D.groups.index("all")].sum()
        ax.set_title(f"{CLASS_LABELS[cname]}   ({_fmt_count(n_valid)} cell-hours)",
                     loc="left", color=CLASS_COLORS[cname], fontweight="bold",
                     fontsize=label_fontsize, pad=6)
        if not last:
            ax.tick_params(axis="x", labelbottom=False)

    whis_txt = ("whiskers: min to max, no outliers drawn"
                if tuple(whis) == tuple(DEFAULT_WHIS)
                else f"whiskers: {whis[0]:g}th to {whis[1]:g}th percentile")
    counts_txt = ";  hours behind each box beneath it" if show_counts else ""
    if handles and host is not None:
        # Into the host panel's right half: the legend in two columns just
        # under the annotation band, the threshold box above the count
        # labels -- the region the frozen-over months leave empty.
        ax_h = axes[classes.index(host)]
        _state_legend(ax_h, handles, labels, whis_txt + counts_txt,
                      legend_fontsize, below=False, anchor=(0.995, 0.84), ncol=2)
        _threshold_at(ax_h, A, legend_fontsize - 2.0, (0.995, 0.13),
                      ha="right", va="bottom")
    elif handles:
        _state_legend(axes[-1], handles, labels, whis_txt + counts_txt,
                      legend_fontsize, below=True)
        _threshold_below(axes[-1], A, legend_fontsize - 2.0)

    fig.suptitle(f"Downwelling longwave by sky state and surface class, ERA5 "
                 f"\u2014 {A.args.region} domain ({_weighting_txt(D)})\n"
                 f"every hour, {_seasons_txt(A)}   |   {filt_txt}   |   "
                 f"{_clear_txt(D.min_lwp_g, D.min_iwp_g, D.clear_tcc_max)}",
                 fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.93,
                        bottom=(0.05 if host is not None
                                else 0.32 / n_rows * 0.9 + 0.02),
                        left=0.08, right=0.985, hspace=0.30)

    tag = "noprecip" if D.filters[f] == "nonprecip" else "allsky"
    suffix = _variant_suffix(D.clear_tcc_max, D.fold_no_phase_into_ice, whis)
    suffix += "" if tuple(classes) == DEFAULT_BOX_CLASSES else " - " + "-".join(classes)
    suffix += "" if show_differences else " - no-differences"
    return lwph._save_stack(fig, A, out_dir,
                            f"monthly_dlr_box_by_phase_by_class_{tag}", dpi,
                            suffix=suffix)


def fig_monthly_dlr_box_by_class_forOV(A, D: DomainDLR, out_dir=None,
                                       dpi: int | None = None,
                                       classes: tuple[str, ...] = DEFAULT_BOX_CLASSES,
                                       precip_filter: bool | None = None,
                                       whis=DEFAULT_WHIS,
                                       min_hours: int = DEFAULT_MIN_HOURS,
                                       box_alpha: float = 0.5,
                                       figsize: tuple[float, float] = (13.33, 6.4),
                                       show_title: bool = True,
                                       label_fontsize: float = 14.0,
                                       tick_fontsize: float = 13.0):
    """The Ocean Visions slide version of :func:`fig_monthly_dlr_box_by_class`.

    The same three rows of boxes, stripped for a slide: no difference
    annotations, no hour counts, no legend, no threshold box. What remains
    is the boxes in the deck's colours -- red liquid containing, blue ice
    only, grey clear sky -- with the class named inside each panel in its
    own colour, and the colours spelled out on the grey subtitle line so the
    figure still explains itself without a legend (``show_title=False``
    drops both title lines for a slide that carries its own).

    The rows share one flux axis, so a box can be read across surfaces as
    well as across months; the full figure gives each row its own. The
    canvas is 16:9-ish (``figsize``), and the fonts are larger than the
    working figure's, so it stays legible when scaled onto a slide beside a
    title. Boxes with fewer than ``min_hours`` behind them are still hatched
    -- the one diagnostic kept, since a thin box should never pass as a
    statistic on a slide.
    """
    import matplotlib.pyplot as plt

    bad = [c for c in classes if c not in D.classes]
    if bad:
        raise ValueError(f"unknown classes {bad}; choose from {D.classes}")
    f, filt_txt = _filter_index(D, A, precip_filter)
    months = D.months
    n_rows = len(classes)
    fig, axes = plt.subplots(n_rows, 1, sharex=True, sharey=True, figsize=figsize)
    axes = np.atleast_1d(axes)

    lo_all, hi_all = np.inf, -np.inf
    for ax, cname in zip(axes, classes):
        c = D.classes.index(cname)
        stats, counts = {}, {}
        for st in STATE_ORDER:
            g = D.groups.index(st)
            stats[st] = [_box_stats_from_hist(D.hist_class[f, c, g, j], D.edges, whis)
                         for j in range(len(months))]
            counts[st] = [D.n_class[f, c, g, j].sum() for j in range(len(months))]
            for bs in stats[st]:
                if bs is not None:
                    lo_all = min(lo_all, bs["whislo"])
                    hi_all = max(hi_all, bs["whishi"])
        _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                              show_counts=False, box_alpha=box_alpha,
                              tick_fontsize=tick_fontsize, legend_fontsize=10.0,
                              label_fontsize=label_fontsize, ylabel="",
                              show_differences=False, draw_legend=False)
        # The class as a left-aligned title above the panel, in its colour:
        # inside the axes it would sit on October's whiskers.
        ax.set_title(CLASS_LABELS[cname], loc="left", color=CLASS_COLORS[cname],
                     fontweight="bold", fontsize=label_fontsize, pad=3)
        ax.tick_params(axis="x", labelbottom=ax is axes[-1])

    # One flux axis for every row, with a little room above the tallest
    # whisker -- the working figure's headroom was for the annotation band
    # and the legend, neither of which is here.
    if np.isfinite(lo_all) and np.isfinite(hi_all):
        span = hi_all - lo_all
        axes[0].set_ylim(lo_all - 0.05 * span, hi_all + 0.06 * span)
    fig.supylabel("downwelling longwave at the surface [W m$^{-2}$]",
                  fontsize=label_fontsize, x=0.012)

    if show_title:
        fig.suptitle(f"Downwelling longwave by sky state and surface, ERA5 \u2014 "
                     f"Oct\u2013Mar {_seasons_txt(A)}", fontsize=label_fontsize + 1,
                     y=0.985)
        fig.text(0.5, 0.925,
                 f"{A.args.region} domain   |   {filt_txt}   |   "
                 f"red: liquid containing, blue: ice only, grey: clear sky "
                 f"(tcc < {D.clear_tcc_max:g}, no cloud water)",
                 ha="center", va="top", fontsize=tick_fontsize - 1.5, color="0.4")
    fig.subplots_adjust(top=0.845 if show_title else 0.955, bottom=0.075,
                        left=0.075, right=0.99, hspace=0.32)

    tag = "noprecip" if D.filters[f] == "nonprecip" else "allsky"
    suffix = _variant_suffix(D.clear_tcc_max, D.fold_no_phase_into_ice, whis)
    suffix += "" if tuple(classes) == DEFAULT_BOX_CLASSES else " - " + "-".join(classes)
    suffix += "" if show_title else " - no-title"
    return lwph._save_stack(fig, A, out_dir,
                            f"monthly_dlr_box_by_phase_by_class_OV_{tag}", dpi,
                            suffix=suffix)


# ----------------------------------------------------------------------------
# Domain: the share of hours in each state, and the numbers behind the boxes
# ----------------------------------------------------------------------------
def state_fractions(D: DomainDLR, precip_filter: bool, scope: str = "domain"):
    """Share of valid (post-filter) cell-hours in each group, per month and
    over the whole window.

    ``scope`` is ``"domain"`` (weighted as the run weights cells) or
    ``"site"`` (the ARM cell). Returns ``(months, frac, n_valid)`` with
    ``frac[group]`` an array over months plus a final all-months entry, and
    ``n_valid`` the plain count of valid hours behind each.
    """
    f = D.filters.index("nonprecip" if precip_filter else "all")
    H = D.hist if scope == "domain" else D.site_hist
    N = D.n if scope == "domain" else D.site_hist
    tot = H[f, D.groups.index("all")].sum(axis=1)             # (month,)
    tot = np.append(tot, tot.sum())
    frac = {}
    for g_name in D.groups:
        if g_name == "all":
            continue
        h = H[f, D.groups.index(g_name)].sum(axis=1)
        h = np.append(h, h.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            frac[g_name] = np.where(tot > 0, h / tot, np.nan)
    n_valid = N[f, D.groups.index("all")].sum(axis=1)
    n_valid = np.append(n_valid, n_valid.sum())
    return D.months, frac, n_valid


def state_medians(D: DomainDLR, precip_filter: bool, scope: str = "domain"):
    """Median flux [W m-2] per state, per month plus the whole window, from
    the histograms; ``scope`` as in :func:`state_fractions`.

    Returns ``(months, median, diff)``. ``median[state]`` has one entry per
    month and a final POOLED entry (the median over every hour of the
    window). ``diff`` holds the between-state differences per month, but
    its final entry is NOT the difference of the pooled medians: it is the
    mean of the monthly differences, weighted by each month's valid hours.

    The distinction matters. The states have different seasonal make-up --
    liquid-containing hours peak in October-November, ice-only and clear
    hours in December-March -- so a difference of pooled medians mostly
    measures autumn against winter, not one sky state against another
    (domain-wide, liquid minus ice comes out near +44 W m-2 pooled against
    +19 to +26 in every month). Differencing within each month holds the
    season fixed, which is what a phase effect needs. NaN where a state has
    no hours.
    """
    f = D.filters.index("nonprecip" if precip_filter else "all")
    H = D.hist if scope == "domain" else D.site_hist
    median = {}
    for st in STATE_ORDER:
        h = H[f, D.groups.index(st)]                              # (month, bin)
        rows = list(h) + [h.sum(axis=0)]
        median[st] = np.array([_hist_quantiles(r, D.edges, (50,))[0]
                               if r.sum() > 0 else np.nan for r in rows])
    diff = median_differences(median)
    # Replace the pooled difference by the season-matched one: the mean of
    # the monthly differences, each month weighted by its valid hours in
    # this scope (the same weight for every state, so the composition is
    # held fixed). Months where either state has no hours drop out.
    w_month = H[f, D.groups.index("all")].sum(axis=1)             # (month,)
    for name, d in diff.items():
        m = d[:-1]
        ok = np.isfinite(m) & (w_month > 0)
        d[-1] = (float(np.average(m[ok], weights=w_month[ok])) if ok.any()
                 else np.nan)
    return D.months, median, diff


def print_state_fraction_table(A, D: DomainDLR,
                               precip_filter: bool | None = None,
                               both_filters: bool = True,
                               with_medians: bool = True) -> dict:
    """Share of cold-season hours in each sky state, domain and ARM cell.

    Per month and over the whole window: the fraction of valid hours that are
    liquid containing, ice only, clear sky, and outside the three states
    (partly cloudy; and, separately, overcast without condensate). The
    domain share is weighted as the run weights cells, so it reads as a share
    of the domain's area-time; the ARM-cell share is a plain share of hours.

    With ``with_medians`` a second block follows: the median flux in each
    state and the differences between them -- liquid minus ice, liquid minus
    clear, ice minus clear -- domain beside ARM cell, from the same
    histograms (so the ARM-cell medians are exact to a bin here, where
    :func:`print_monthly_dlr_table` has them exactly).

    Printed for the run's filter and, with ``both_filters``, for every hour
    too -- the filter removes cloudy hours preferentially, so the shares
    shift. Returns the fractions, medians and differences as nested dicts.
    """
    out = {}
    if precip_filter is None:
        precip_filter = bool(A.args.no_precip)
    variants = [precip_filter] + ([not precip_filter] if both_filters else [])
    cols = ("liquid", "ice", "clear", "other", "no_phase")
    names = ("liq cont", "ice only", "clear", "partly*", "no phase")
    for pf in variants:
        filt = (f"non-precipitating hours (tp < {D.precip_rate_max:g} mm/hr)"
                if pf else "all hours (no precipitation filter)")
        print(f"\n  Share of cold-season hours in each sky state, ERA5 -- "
              f"{filt}")
        print(f"  {len(D.seasons)} seasons {_seasons_txt(A)}   |   domain: "
              f"{D.n_cells:,} cells, {_weighting_txt(D)}   |   clear sky: tcc < "
              f"{D.clear_tcc_max:g}, LWP <= {D.min_lwp_g:g} and IWP <= "
              f"{D.min_iwp_g:g} g m-2\n")
        block_w = 9 * len(cols) + 7
        print(f"    {'':<8}{'--- ' + A.args.region + ' domain ---':^{block_w}}"
              f"   {'--- ARM cell ---':^{block_w}}")
        print(f"    {'month':<8}" + "".join(f"{nm:>9}" for nm in names)
              + f"{'hours':>7}" + "   " + "".join(f"{nm:>9}" for nm in names)
              + f"{'hours':>7}")
        res = {}
        for scope in ("domain", "site"):
            months, frac, n_valid = state_fractions(D, pf, scope)
            res[scope] = {"months": months, "frac": frac, "n_valid": n_valid}
        labels = [calendar.month_abbr[m] for m in months] + ["all"]
        for j, lab in enumerate(labels):
            row = f"    {lab:<8}"
            for scope in ("domain", "site"):
                fr, nv = res[scope]["frac"], res[scope]["n_valid"]
                for c in cols:
                    v = fr[c][j]
                    row += f"{100 * v:>8.1f}%" if np.isfinite(v) else f"{'--':>9}"
                row += f"{_fmt_count(nv[j]):>7}"
                if scope == "domain":
                    row += "   "
            print(row)
        print(f"    * partly: valid hours in none of the three states "
              f"({D.clear_tcc_max:g} <= tcc < {float(A.args.min_cloud_fraction):g}, "
              f"or cloud water under a partial sky), excluding the no-phase "
              f"hours listed separately"
              f"{'; no phase is inside ice only' if D.fold_no_phase_into_ice else ''}")

        if with_medians:
            # Median flux per state and the differences between states, the
            # domain beside the ARM cell, same rows as the shares above.
            short = {"liquid": "liq", "ice": "ice", "clear": "clear"}
            mcols = [short[st] for st in STATE_ORDER] + [nm for _a, _b, nm in DIFF_PAIRS]
            mblock_w = 8 * len(mcols)
            print(f"\n    Median downwelling longwave [W m-2] by sky state, "
                  f"and the differences between states (from the "
                  f"{DLR_BIN_W_M2:g} W m-2 histograms)")
            print(f"    'all' row: medians pooled over every hour of the window; "
                  f"differences = hour-weighted mean of the MONTHLY differences, "
                  f"not the difference of the pooled medians (see below)")
            print(f"    {'':<8}{'--- ' + A.args.region + ' domain ---':^{mblock_w}}"
                  f"   {'--- ARM cell ---':^{mblock_w}}")
            print(f"    {'month':<8}" + "".join(f"{nm:>8}" for nm in mcols)
                  + "   " + "".join(f"{nm:>8}" for nm in mcols))
            for scope in ("domain", "site"):
                _m, med, diff = state_medians(D, pf, scope)
                res[scope]["median"] = med
                res[scope]["diff_median"] = diff
            for j, lab in enumerate(labels):
                row = f"    {lab:<8}"
                for scope in ("domain", "site"):
                    med, diff = res[scope]["median"], res[scope]["diff_median"]
                    for st in STATE_ORDER:
                        v = med[st][j]
                        row += f"{v:>8.1f}" if np.isfinite(v) else f"{'--':>8}"
                    for _a, _b, nm in DIFF_PAIRS:
                        v = diff[nm][j]
                        row += f"{v:>+8.1f}" if np.isfinite(v) else f"{'--':>8}"
                    if scope == "domain":
                        row += "   "
                print(row)
            print(f"    A difference of the POOLED medians would mix the seasonal "
                  f"cycle into the phase effect: liquid-containing hours peak "
                  f"in Oct-Nov, ice-only and clear hours in Dec-Mar, so pooled "
                  f"liquid is an autumn\n    number and pooled ice a winter one. "
                  f"Differencing within each month holds the season fixed; the "
                  f"'all' row averages those monthly differences.")
        out["nonprecip" if pf else "all"] = res
    return out


def print_monthly_dlr_table_domain(A, D: DomainDLR,
                                   precip_filter: bool | None = None) -> dict:
    """The domain box figure's numbers: per month and state, cell-hours,
    median, interquartile range and mean of the flux [W m-2] from the
    histograms, then the median differences between states."""
    f, _txt = _filter_index(D, A, precip_filter)
    months = D.months
    stats = {k: {st: np.full(len(months), np.nan) for st in STATE_ORDER}
             for k in ("median", "q25", "q75", "mean")}
    n = {}
    centres = 0.5 * (D.edges[:-1] + D.edges[1:])
    for st in STATE_ORDER:
        g = D.groups.index(st)
        n[st] = D.n[f, g].sum(axis=1)
        for j in range(len(months)):
            h = D.hist[f, g, j]
            if h.sum() > 0:
                q25, med, q75 = _hist_quantiles(h, D.edges, (25, 50, 75))
                stats["q25"][st][j] = q25
                stats["median"][st][j] = med
                stats["q75"][st][j] = q75
                stats["mean"][st][j] = float((h * centres).sum() / h.sum())
    filt = (f"non-precipitating (tp < {D.precip_rate_max:g} mm/hr)"
            if D.filters[f] == "nonprecip" else "all hours")
    print(f"\n  Downwelling longwave at the surface [W m-2] by month and sky "
          f"state, ERA5 -- every cell of the {A.args.region} domain")
    print(f"  {filt}   |   {len(D.seasons)} seasons {_seasons_txt(A)}   |   "
          f"statistics from {DLR_BIN_W_M2:g} W m-2 histograms, "
          f"{_weighting_txt(D)}; hours are plain counts\n")
    diff = _print_state_stats(months, stats, n)
    return {"months": months, "n": n, "diff_median": diff, **stats}


# ----------------------------------------------------------------------------
# The distribution of the flux: ARM cell beside the whole domain
# ----------------------------------------------------------------------------
def _rebin(counts: np.ndarray, edges: np.ndarray, bin_w: float):
    """Sum adjacent bins so the histogram is ``bin_w`` wide."""
    k = int(round(bin_w / DLR_BIN_W_M2))
    if k < 1 or not np.isclose(k * DLR_BIN_W_M2, bin_w):
        raise ValueError(f"bin_w must be a multiple of {DLR_BIN_W_M2:g} W m-2")
    n_full = (counts.size // k) * k
    c = counts[:n_full].reshape(-1, k).sum(axis=1)
    e = edges[:n_full + 1:k]
    return c, e


def fig_dlr_pdf(A, D: DomainDLR, out_dir=None, dpi: int | None = None,
                precip_filter: bool = False, bin_w: float = 2.0,
                show_states: bool = False, xlim=(80.0, 360.0),
                label_fontsize: float = lwph.DEFAULT_COMPARISON_LABEL_FONTSIZE,
                tick_fontsize: float = lwph.DEFAULT_COMPARISON_TICK_FONTSIZE,
                legend_fontsize: float = lwph.DEFAULT_COMPARISON_LEGEND_FONTSIZE):
    """Probability density of hourly downwelling longwave over the whole
    window: the ARM cell beside every cell of the domain.

    Both curves are histograms at ``bin_w`` W m-2 normalised to unit area, so
    the y axis is density per W m-2 and the two are comparable whatever their
    sample sizes (about 48,000 hours against 120 million cell-hours). The
    domain curve is weighted as the run weights cells. ``precip_filter`` is
    False by default -- the shape of the distribution is a property of the
    whole atmosphere, and the filter removes cloudy hours preferentially --
    and the title says which population is drawn.

    ``show_states=True`` adds the domain curve's decomposition by sky state,
    each scaled by its share of hours so the three (plus the partly cloudy
    remainder, dotted grey) sum to the domain curve; it shows which state
    fills, or fails to fill, each mode.
    """
    import matplotlib.pyplot as plt

    f = D.filters.index("nonprecip" if precip_filter else "all")
    g_all = D.groups.index("all")
    dom_c, e = _rebin(D.hist[f, g_all].sum(axis=0), D.edges, bin_w)
    site_c, _ = _rebin(D.site_hist[f, g_all].sum(axis=0), D.edges, bin_w)
    centres = 0.5 * (e[:-1] + e[1:])
    dom_pdf = dom_c / (dom_c.sum() * bin_w)
    site_pdf = site_c / (site_c.sum() * bin_w)
    n_dom = D.n[f, g_all].sum()
    n_site = D.site_hist[f, g_all].sum()

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.6))
    ax.plot(centres, dom_pdf, color="black", lw=2.2,
            label=f"whole {A.args.region} domain: {D.n_cells:,} cells, "
                  f"{n_dom / 1e6:.0f} M cell-hours ({_weighting_txt(D)})")
    ax.plot(centres, site_pdf, color="#12395E", lw=2.2, ls="--",
            label=f"grid cell containing Utqiaġvik: {n_site:,.0f} hours")
    if show_states:
        # The domain curve split by sky state, each scaled by its share of
        # hours so the pieces sum to the black curve.
        tot = dom_c.sum()
        rest = dom_c.copy()
        for st in STATE_ORDER:
            c, _ = _rebin(D.hist[f, D.groups.index(st)].sum(axis=0), D.edges, bin_w)
            rest = rest - c
            ax.plot(centres, c / (tot * bin_w), color=STATE_COLORS[st], lw=1.5,
                    alpha=0.9,
                    label=f"domain, {STATE_LABELS[st]}: "
                          f"{100 * c.sum() / tot:.1f}% of hours")
        ax.plot(centres, rest / (tot * bin_w), color="0.45", lw=1.2, ls=":",
                label=f"domain, remainder (partly cloudy): "
                      f"{100 * rest.sum() / tot:.1f}% of hours")

    ax.set_xlim(*xlim)
    ax.set_ylim(0, None)
    ax.set_xlabel("downwelling longwave at the surface [W m$^{-2}$]",
                  fontsize=label_fontsize)
    ax.set_ylabel(f"probability density [(W m$^{{-2}}$)$^{{-1}}$]   "
                  f"({bin_w:g} W m$^{{-2}}$ bins)", fontsize=label_fontsize - 1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Legend below the axes: both curves fill the upper part of the panel, so
    # no corner inside it is free.
    ax.legend(fontsize=legend_fontsize - 1, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2 if show_states else 1)

    filt_txt = (f"non-precipitating: tp < {D.precip_rate_max:g} mm hr$^{{-1}}$"
                if precip_filter else "all hours, no precipitation filter")
    fig.suptitle(f"Distribution of hourly downwelling longwave, ERA5 — "
                 f"ARM cell and the whole {A.args.region} domain\n"
                 f"Oct–Mar, {_seasons_txt(A)}   |   {filt_txt}",
                 fontsize=12.5, y=0.975)
    fig.subplots_adjust(top=0.88, bottom=0.26 if show_states else 0.22,
                        left=0.09, right=0.985)

    tag = "noprecip" if precip_filter else "allhours"
    suffix = ("" if bin_w == 2.0 else f" - bin{bin_w:g}") + \
             (" - by-state" if show_states else "")
    return lwph._save_stack(fig, A, out_dir, f"dlr_pdf_site_vs_domain_{tag}",
                            dpi, suffix=suffix)


# ----------------------------------------------------------------------------
# Between-state differences, three ways
# ----------------------------------------------------------------------------
def resolve_diff_method(method, args) -> str:
    """The difference method: the argument, else the run's
    ``--dlr-diff-method``, else ``DEFAULT_DIFF_METHOD``."""
    if method is None:
        method = getattr(args, "dlr_diff_method", DEFAULT_DIFF_METHOD)
    if method not in DIFF_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {DIFF_METHODS}")
    return method


DIFF_METHOD_LABELS: dict[str, str] = {
    "median_diff": "differences of the monthly medians, one value per season",
    "daily_diff": "differences of the daily medians over the class's cells, "
                  "one value per day",
    "cre": "cloud effect against ERA5's clear-sky flux (DLR − DLR$_{\\mathrm{clear}}$), "
           "medians differenced per season",
    "cre_matched": "cloud effect differenced within bins of the clear-sky flux "
                   "(same air-mass state), bins weighted by the liquid hours, "
                   "per season",
}


def _hist_median(h: np.ndarray, edges: np.ndarray, min_weight: float) -> float:
    """Median of one histogram, NaN below ``min_weight`` total."""
    tot = float(h.sum())
    if tot < min_weight or tot <= 0:
        return np.nan
    return float(_hist_quantiles(h, edges, (50,))[0])


def dlr_differences(A, D: DomainDLR, cname: str, method: str | None = None,
                    precip_filter: bool | None = None,
                    min_cell_hours: float = 100.0,
                    daily_min_hours: float = 20.0,
                    match_min_hours: float = 20.0) -> dict:
    """The two between-state differences for one surface class, per month,
    as collections of values a box can summarise.

    Returns ``{"liq-ice": [array per month], "liq-clr": [...],
    "method": ..., "unit": "W m-2"}``. What the arrays hold depends on
    ``method`` (``None`` follows the run's ``--dlr-diff-method``):

    ``median_diff``
        For each season and month, the class's liquid-containing median
        minus its ice-only (clear-sky) median -- the numbers the tables
        print, but per season, so the box shows the spread over the run's
        seasons. A (season, month, state) with fewer than ``min_cell_hours``
        cell-hours is left out.
    ``daily_diff``
        For each day of the run, the class's liquid-containing median over
        that day's cells minus its ice-only (clear-sky) median over that
        day's cells -- the "same day, same surface" pairing. Days where
        either state has fewer than ``daily_min_hours`` cell-hours in the
        class are left out. The box shows the spread over all such days of
        the month across the seasons.
    ``cre``
        Each hour's cloud radiative effect, CRE = DLR - DLR_clearsky, from
        ERA5's own clear-sky flux; per season and month, the class's median
        CRE under liquid-containing cloud minus its median CRE under ice-only
        cloud (under clear sky, which is about zero). Every hour is compared
        with its own atmosphere without cloud, so the air-mass difference
        between the populations is removed by construction; the box shows
        the spread over seasons. ``min_cell_hours`` applies as above.
    ``cre_matched``
        As ``cre``, but liquid and ice (clear) hours are compared only
        within the same bin of clear-sky flux -- the same air-mass emission
        state -- and the per-bin differences of median CRE are averaged with
        the liquid hours' bin weights, so the answer is "in the environments
        where liquid cloud occurs, how much more does it add than ice cloud
        would in the same environment". Bins where either state has fewer
        than ``match_min_hours`` cell-hours are skipped; the returned
        ``coverage`` says what share of the liquid hours found a match. One
        value per season and month, as for ``cre``.
    """
    method = resolve_diff_method(method, A.args)
    f, _txt = _filter_index(D, A, precip_filter)
    c = D.classes.index(cname)
    n_month = len(D.months)
    st_i = {st: i for i, st in enumerate(STATE_ORDER)}
    out = {"liq-ice": [], "liq-clr": [], "method": method, "unit": "W m-2"}

    if method in ("median_diff", "cre"):
        if method == "median_diff":
            H = D.hist_cs[f, :, c]                              # (s, g, j, b)
            edges = D.edges
            idx = {st: D.groups.index(st) for st in STATE_ORDER}
        else:
            H = D.cre_hist[f, :, c]                             # (s, st, j, b)
            edges = D.cre_edges
            idx = st_i
        n_season = H.shape[0]
        med = {st: np.array([[_hist_median(H[s_, idx[st], j], edges, min_cell_hours)
                              for j in range(n_month)] for s_ in range(n_season)])
               for st in STATE_ORDER}                           # (s, j)
        for j in range(n_month):
            for name, other in (("liq-ice", "ice"), ("liq-clr", "clear")):
                d = med["liquid"][:, j] - med[other][:, j]
                out[name].append(d[np.isfinite(d)])
        return out

    if method == "cre_matched":
        H = D.match_hist[f, :, c]                               # (s, st, j, k, b)
        n_season = H.shape[0]
        out["coverage"] = {"liq-ice": [], "liq-clr": []}
        for j in range(n_month):
            for name, other in (("liq-ice", "ice"), ("liq-clr", "clear")):
                vals, cov = [], []
                for s_ in range(n_season):
                    hl = H[s_, st_i["liquid"], j].astype(float)   # (k, b)
                    ho = H[s_, st_i[other], j].astype(float)
                    w_l, w_o = hl.sum(axis=1), ho.sum(axis=1)
                    ok = (w_l >= match_min_hours) & (w_o >= match_min_hours)
                    if not ok.any() or w_l[ok].sum() < min_cell_hours:
                        continue
                    d_k = np.array([_hist_quantiles(hl[k], D.match_cre_edges, (50,))[0]
                                    - _hist_quantiles(ho[k], D.match_cre_edges, (50,))[0]
                                    for k in np.flatnonzero(ok)])
                    vals.append(float(np.average(d_k, weights=w_l[ok])))
                    cov.append(float(w_l[ok].sum() / w_l.sum()) if w_l.sum() > 0 else np.nan)
                out[name].append(np.array(vals))
                out["coverage"][name].append(np.array(cov))
        return out

    # daily_diff
    H = D.daily_hist[f, :, c]                                   # (d, st, b)
    med = {st: np.array([_hist_median(H[d_, st_i[st]], D.daily_edges,
                                      daily_min_hours)
                         for d_ in range(D.n_day)]) for st in STATE_ORDER}
    for j in range(n_month):
        in_m = D.day_month == j
        for name, other in (("liq-ice", "ice"), ("liq-clr", "clear")):
            d = (med["liquid"] - med[other])[in_m]
            out[name].append(d[np.isfinite(d)])
    return out


def print_cre_check(A, D: DomainDLR, precip_filter: bool | None = None,
                    classes: tuple[str, ...] = DEFAULT_BOX_CLASSES) -> None:
    """Is ERA5's clear-sky flux a fair reference? Per month and class, the
    median cloud effect under each state. Under clear sky it should sit
    within a watt or so of zero -- the check that the reference is the same
    atmosphere -- while under cloud it is the state's longwave forcing."""
    f, txt = _filter_index(D, A, precip_filter)
    print(f"\n  Median cloud radiative effect DLR - DLR_clearsky [W m-2] by state "
          f"({txt.replace('$^{{-1}}$', '-1')})")
    print(f"    {'month':<6}" + "".join(
        f"{CLASS_LABELS[c]:>30}" for c in classes))
    print(f"    {'':<6}" + "".join(
        f"{'liquid':>10}{'ice':>10}{'clear':>10}" for _ in classes))
    for j, m in enumerate(D.months):
        row = f"    {calendar.month_abbr[m]:<6}"
        for cname in classes:
            c = D.classes.index(cname)
            for st_i, st in enumerate(STATE_ORDER):
                h = D.cre_hist[f, :, c, st_i, j].sum(axis=0)
                v = _hist_median(h, D.cre_edges, 1.0)
                row += f"{v:>10.1f}" if np.isfinite(v) else f"{'--':>10}"
        print(row)


def print_diff_methods_table(A, D: DomainDLR,
                             classes: tuple[str, ...] = ("open_ocean", "sea_ice"),
                             methods: tuple[str, ...] = DIFF_METHODS,
                             precip_filter: bool | None = None,
                             **kwargs) -> dict:
    """The methods side by side: per class and month, the median of the box
    values each method gives for liquid minus ice and liquid minus clear
    [W m-2], with the matched method's coverage (share of liquid hours that
    found an ice, or clear, match in their clear-sky bin). ``kwargs`` go to
    :func:`dlr_differences`. Returns the numbers."""
    out = {}
    for cname in classes:
        res = {m: dlr_differences(A, D, cname, m, precip_filter, **kwargs)
               for m in methods}
        out[cname] = res
        print(f"\n  Liquid containing minus ice only / minus clear sky [W m-2], "
              f"median of each method's box values -- {CLASS_LABELS[cname]}")
        head = f"    {'month':<6}"
        for m in methods:
            head += f"{m:^20}"
        if "cre_matched" in methods:
            head += f"{'matched coverage':^18}"
        print(head)
        sub = f"    {'':<6}" + "".join(f"{'liq-ice':>10}{'liq-clr':>10}" for _ in methods)
        if "cre_matched" in methods:
            sub += f"{'ice':>9}{'clear':>9}"
        print(sub)
        for j, mo in enumerate(D.months):
            row = f"    {calendar.month_abbr[mo]:<6}"
            for m in methods:
                for name in ("liq-ice", "liq-clr"):
                    v = res[m][name][j]
                    row += f"{np.median(v):>+10.1f}" if v.size else f"{'--':>10}"
            if "cre_matched" in methods:
                for name in ("liq-ice", "liq-clr"):
                    cv = res["cre_matched"]["coverage"][name][j]
                    row += (f"{100 * np.nanmean(cv):>8.0f}%"
                            if cv.size and np.isfinite(cv).any() else f"{'--':>9}")
            print(row)
    print("    (box values: one per season for median_diff, cre and cre_matched; "
          "one per day for daily_diff)")
    return out


# ----------------------------------------------------------------------------
# The ver2 slide figure: two classes, and their differences below
# ----------------------------------------------------------------------------
DIFF_COLORS: dict[str, str] = {"liq-ice": lwph.GENIE_ICE_COLOR,
                               "liq-clr": CLEAR_COLOR}
DIFF_LABELS: dict[str, str] = {
    "liq-ice": r"$\Delta\mathrm{DLR}_{\mathrm{glaciated}}$",
    "liq-clr": r"$\Delta\mathrm{DLR}_{\mathrm{clrsky}}$",
}


def _draw_difference_boxes(ax, months, diffs: dict, whis, min_values: int,
                           box_alpha: float, tick_fontsize: float):
    """Two boxes per month on ``ax`` from ``diffs[name][j]`` (arrays of
    values): liquid minus ice in the ice-only blue, liquid minus clear in
    the clear-sky grey. Hatched below ``min_values`` values."""
    from matplotlib.colors import to_rgba

    x = np.arange(len(months))
    w = 0.30
    names = ("liq-ice", "liq-clr")
    offsets = (-0.5 * (w + 0.05), 0.5 * (w + 0.05))
    lo, hi = np.inf, -np.inf
    for name, off in zip(names, offsets):
        color = DIFF_COLORS[name]
        pos, data, n_v = [], [], []
        for j in range(len(months)):
            v = np.asarray(diffs[name][j], dtype=float)
            bs = _box_stats(v, whis)
            if bs is None:
                continue
            pos.append(x[j] + off)
            data.append(bs)
            n_v.append(v.size)
        if not data:
            continue
        lo = min(lo, min(d["whislo"] for d in data))
        hi = max(hi, max(d["whishi"] for d in data))
        bp = ax.bxp(data, positions=pos, widths=w, showfliers=False,
                    patch_artist=True, manage_ticks=False)
        for art, nv in zip(bp["boxes"], n_v):
            thin = nv < min_values
            art.set_facecolor(to_rgba(color, box_alpha * (0.4 if thin else 1.0)))
            art.set_edgecolor(color)
            art.set_linewidth(1.4)
            if thin:
                art.set_hatch("////")
        for key in ("whiskers", "caps"):
            for art in bp[key]:
                art.set_color(color)
                art.set_linewidth(1.3)
        for art in bp["medians"]:
            art.set_color("black")
            art.set_linewidth(1.8)
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=(0, (3, 3)),
               zorder=1)
    # More major ticks than the default locator gives, plus minor ticks
    # in between, so the difference rows read off more precisely.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, axis="y", which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, axis="y", which="minor", alpha=0.12, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months],
                       fontsize=tick_fontsize)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.tick_params(axis="y", which="major", labelsize=tick_fontsize, length=4.5)
    ax.tick_params(axis="y", which="minor", length=2.5)
    return lo, hi


def fig_monthly_dlr_box_by_class_forOV_ver2(
        A, D: DomainDLR, out_dir=None, dpi: int | None = None,
        classes: tuple[str, ...] = ("open_ocean", "sea_ice"),
        method: str | None = None,
        precip_filter: bool | None = None,
        whis=DEFAULT_WHIS,
        min_hours: int = DEFAULT_MIN_HOURS,
        min_values: int = 5,
        min_cell_hours: float = 100.0,
        daily_min_hours: float = 20.0,
        box_alpha: float = 0.5,
        figsize: tuple[float, float] = (13.33, 9.0),
        show_title: bool = True,
        label_fontsize: float = 14.0,
        tick_fontsize: float = 13.0,
        class_title_fontsize: float = 17.0):
    """:func:`fig_monthly_dlr_box_by_class_forOV` for open ocean and sea ice,
    with the between-state differences beneath.

    Rows, top to bottom: the flux by sky state for each class (as the slide
    figure draws it, class titles a few points larger), then for each class
    a row of two boxes per month -- liquid containing minus ice only in the
    ice-only blue, liquid containing minus clear sky in the clear-sky grey --
    formed by ``method`` (:func:`dlr_differences`; ``None`` follows the run's
    ``--dlr-diff-method``). The flux rows share one axis and the difference
    rows share another. A difference box built from fewer than ``min_values``
    values (seasons, or days) is hatched. The subtitle names the method, so
    a saved PNG says which comparison it shows; the file name carries it too.
    """
    import matplotlib.pyplot as plt

    bad = [c for c in classes if c not in D.classes]
    if bad:
        raise ValueError(f"unknown classes {bad}; choose from {D.classes}")
    method = resolve_diff_method(method, A.args)
    f, filt_txt = _filter_index(D, A, precip_filter)
    months = D.months
    n_c = len(classes)
    fig, axes = plt.subplots(2 * n_c, 1, sharex=True, figsize=figsize)
    axes = np.atleast_1d(axes)
    flux_axes, diff_axes = axes[:n_c], axes[n_c:]

    # The flux rows.
    lo_all, hi_all = np.inf, -np.inf
    for ax, cname in zip(flux_axes, classes):
        c = D.classes.index(cname)
        stats, counts = {}, {}
        for st in STATE_ORDER:
            g = D.groups.index(st)
            stats[st] = [_box_stats_from_hist(D.hist_class[f, c, g, j], D.edges, whis)
                         for j in range(len(months))]
            counts[st] = [D.n_class[f, c, g, j].sum() for j in range(len(months))]
            for bs in stats[st]:
                if bs is not None:
                    lo_all = min(lo_all, bs["whislo"])
                    hi_all = max(hi_all, bs["whishi"])
        _draw_sky_state_boxes(ax, months, stats, counts, whis, min_hours,
                              show_counts=False, box_alpha=box_alpha,
                              tick_fontsize=tick_fontsize, legend_fontsize=10.0,
                              label_fontsize=label_fontsize, ylabel="",
                              show_differences=False, draw_legend=False)
        ax.set_title(CLASS_LABELS[cname], loc="left", color=CLASS_COLORS[cname],
                     fontweight="bold", fontsize=class_title_fontsize, pad=3)
        ax.tick_params(axis="x", labelbottom=False)
    if np.isfinite(lo_all) and np.isfinite(hi_all):
        span = hi_all - lo_all
        for ax in flux_axes:
            ax.set_ylim(lo_all - 0.05 * span, hi_all + 0.06 * span)

    # The difference rows.
    d_lo, d_hi = np.inf, -np.inf
    for ax, cname in zip(diff_axes, classes):
        diffs = dlr_differences(A, D, cname, method, precip_filter,
                                min_cell_hours, daily_min_hours)
        lo, hi = _draw_difference_boxes(ax, months, diffs, whis, min_values,
                                        box_alpha, tick_fontsize)
        d_lo, d_hi = min(d_lo, lo), max(d_hi, hi)
        ax.set_title(f"{CLASS_LABELS[cname]}: {DIFF_LABELS['liq-ice']} (blue), "
                     f"{DIFF_LABELS['liq-clr']} (grey)", loc="left",
                     color=CLASS_COLORS[cname], fontweight="bold",
                     fontsize=class_title_fontsize - 3, pad=3)
        ax.tick_params(axis="x", labelbottom=ax is diff_axes[-1])
    if np.isfinite(d_lo) and np.isfinite(d_hi):
        span = max(d_hi - d_lo, 1.0)
        for ax in diff_axes:
            ax.set_ylim(min(d_lo, 0.0) - 0.06 * span, max(d_hi, 0.0) + 0.06 * span)

    fig.text(0.012, 0.5 + 0.25, "DLR [W m$^{-2}$]", rotation=90, ha="center",
             va="center", fontsize=label_fontsize)
    fig.text(0.012, 0.5 - 0.25, "$\\Delta$DLR [W m$^{-2}$]", rotation=90,
             ha="center", va="center", fontsize=label_fontsize + 3.0)

    if show_title:
        fig.suptitle(f"Downwelling longwave by sky state, and what liquid cloud "
                     f"adds — ERA5, Oct–Mar {_seasons_txt(A)}",
                     fontsize=label_fontsize + 1, y=0.99)
        # Two subtitle lines: one this long would overflow the canvas and
        # the tight bounding box would pad the saved file to fit it.
        fig.text(0.5, 0.957,
                 f"{A.args.region} domain   |   {filt_txt}   |   "
                 f"red: liquid containing, blue: ice only, grey: clear sky\n"
                 f"differences: {DIFF_METHOD_LABELS[method]}",
                 ha="center", va="top", fontsize=tick_fontsize - 2.0, color="0.4",
                 linespacing=1.5)
    fig.subplots_adjust(top=0.885 if show_title else 0.96, bottom=0.055,
                        left=0.075, right=0.99, hspace=0.42)

    tag = "noprecip" if D.filters[f] == "nonprecip" else "allsky"
    suffix = _variant_suffix(D.clear_tcc_max, D.fold_no_phase_into_ice, whis)
    suffix += f" - {method}"
    suffix += ("" if tuple(classes) == ("open_ocean", "sea_ice")
               else " - " + "-".join(classes))
    suffix += "" if show_title else " - no-title"
    return lwph._save_stack(fig, A, out_dir,
                            f"monthly_dlr_box_by_class_OV_ver2_{tag}", dpi,
                            suffix=suffix)
