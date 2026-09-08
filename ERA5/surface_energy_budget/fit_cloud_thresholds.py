#!/usr/bin/env python3
"""Fit --min-cloud-fraction and --ice-fraction-min to the ARM observations.

THE PROBLEM
===========
Two thresholds decide how many hours ERA5 calls "liquid containing" at the
Utqiagvik grid cell:

    tcc >= min_cloud_fraction                       the overcast gate
    IWP/(LWP+IWP) >= ice_fraction_min               the ice-only cut

A scene is liquid-containing when it is cloudy, holds cloud water above the
minimum paths, and is NOT ice-only -- that is, when its ice share sits BELOW
``ice_fraction_min``. (``--liquid-fraction-min`` moves the liquid-only/mixed
boundary and is invisible to this merge, which is why it is not a free
parameter here.)

The cost minimised is the root-sum-square of the LIQUID-CONTAINING residuals,
ERA5 minus observations, over whichever seasons or months the target uses.

WHY EXHAUSTIVE GRID SEARCH, AND NOT AN OPTIMISER
================================================
Three facts about this objective decide the method, and they point the same way.

1. IT IS PIECEWISE CONSTANT, SO IT HAS NO USABLE GRADIENT.
   Each hour flips category at one discrete threshold value. Moving
   min_cloud_fraction from 0.9500 to 0.9501 changes the answer only if some
   hour has tcc inside that interval; otherwise the cost is bit-identical. So
   the surface is a staircase: the derivative is zero almost everywhere and
   undefined on the jumps. Gradient descent, L-BFGS and friends have nothing to
   descend. Nelder-Mead and Powell fare little better -- a simplex that lands
   entirely on one tread sees a flat function and collapses. This alone rules
   out the standard optimisers, whatever the evaluation cost were.

2. EVALUATION IS ESSENTIALLY FREE, ONCE THE ARCHIVE HAS BEEN READ ONCE.
   This is the fact that inverts the usual cost model. Both free parameters are
   post-hoc scalar tests applied hour by hour, and the target is ONE grid cell.
   So a single streaming pass reduces the whole problem to a table of about
   50,000 rows -- season, month, tcc, ice share, cloud flag, rain flag -- after
   which scoring any (min_cloud_fraction, ice_fraction_min) pair is two
   comparisons and a bincount, on the order of a millisecond. A full 91 x 101
   grid costs seconds. There is nothing for a smarter search to save: an
   optimiser converging in 40 evaluations would save ~9 seconds and give up
   everything in point 3.

3. THE SURFACE IS WORTH MORE THAN THE MINIMUM.
   Two thresholds that both remove cloudy hours can trade against each other,
   and a lone optimum cannot show that. The full surface distinguishes a sharp,
   well-identified minimum from a long flat valley in which the data constrain
   only a combination of the two -- and it shows whether the minimum sits in
   the interior or is pinned to a bound, which is the difference between a
   fitted value and a bound the fit wanted to leave.

A refinement worth stating because it bounds the error: since the cost only
changes at values present in the data, the exact global optimum lies at one of
those breakpoints. A grid at 0.001 spacing is far finer than either threshold
is physically meaningful to, so the reported optimum differs from the exact one
by at most one grid step, and only inside a tread where the cost is flat anyway.

WHAT THIS IS AND IS NOT
=======================
This is CALIBRATION, not validation. Thresholds tuned to reproduce one
observational record cannot then be cited as evidence that ERA5 agrees with it.
The honest use is diagnostic: if a good match needs a min_cloud_fraction far
from 1.0, that says ERA5 needs partly cloudy scenes counted as overcast to
match a point instrument -- a statement about sampling, not about tuning.

The cost also uses the LIQUID-CONTAINING residual alone, as asked. Ice-only is
therefore free to get worse, so :func:`report_optimum` prints the ice residual
at the optimum as a check. Treat a fit that fixes liquid while breaking ice as
having relabelled hours rather than found any.
"""

from __future__ import annotations

import numpy as np

import plot_lwp_histogram_by_surface_class as lwph
from plot_lwp_histogram_by_surface_class import (
    GENIE_EXCLUDED_MONTHS,
    HOURS_PER_STEP,
    excluded_month_mask,
    season_month_axis,
    season_month_window_hours,
    season_window_hours,
)
from surface_classification import iter_time_blocks

# ----------------------------------------------------------------------------
# Seasons dropped from the SEASONAL fits
# ----------------------------------------------------------------------------
# Genie's record is substantially incomplete in these four, and the seasonal
# observation file reports the shortfall as raw missing hours rather than
# scaling what remains up to a full season. ERA5 has no gaps and IS scaled up,
# so a residual in one of these seasons is dominated by how much of the season
# the instruments actually watched. Fitting to them would tune the thresholds to
# instrument downtime.
#
# Season START years, matching the convention used everywhere in this project:
# 2016 is the 2016/17 season.
EXCLUDED_FIT_SEASONS: tuple[int, ...] = (2016, 2019, 2020, 2021)


# ----------------------------------------------------------------------------
# One pass over the archive
# ----------------------------------------------------------------------------
def extract_site_series(A, precip_rate_max_mm_hr: float | None = None) -> dict:
    """Reduce the archive to a per-hour table for the ARM grid cell.

    This is the only expensive step, and it runs once. Everything the fit needs
    that does NOT depend on the two free parameters is computed here: the cloud
    water shares, the rain flag, the season and month index of every hour, and
    the denominators.

    Reuses ``A.ds`` and ``A.layout``, so the season selection, the site cell and
    the window bookkeeping are the same objects the figures used -- the fit
    cannot silently disagree with the plots about which hours exist.

    Returns a dict of parallel arrays over KEPT hours, plus the denominators.
    """
    ds, args, layout = A.ds, A.args, A.layout
    if precip_rate_max_mm_hr is None:
        precip_rate_max_mm_hr = float(args.precip_rate_max)

    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    slots, seasons = layout["slots"], layout["seasons"]
    months, mi_of_slot = season_month_axis(slots)

    wanted = np.zeros(len(seasons), dtype=bool)
    wanted[A.keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    site_mask, site_lat, site_lon = lwph.site_cell_mask(ds)
    i, j = (int(v) for v in np.argwhere(site_mask)[0])

    min_lwp_g = float(A.phase_kw["min_lwp_g"])
    min_iwp_g = float(A.phase_kw["min_iwp_g"])

    si_l, mi_l, tcc_l, ice_frac_l, has_cloud_l, rain_l, valid_l = ([] for _ in range(7))
    read_vars = ["tcc", "tclw", "tciw", "tp"]
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        keep = use_step[slice(i0, i0 + n_t)]
        if not keep.any():
            continue
        si_l.append(s_idx[slice(i0, i0 + n_t)][keep])
        mi_l.append(mi_of_slot[dos[slice(i0, i0 + n_t)][keep]])

        tcc = block["tcc"].values[keep, i, j]
        lwp_g = block["tclw"].values[keep, i, j] * 1000.0   # kg m-2 -> g m-2
        iwp_g = block["tciw"].values[keep, i, j] * 1000.0
        rate = block["tp"].values[keep, i, j] * 1000.0      # m/h -> mm/hr

        valid = np.isfinite(tcc) & np.isfinite(lwp_g) & np.isfinite(iwp_g)
        # Same flooring as fraction_phase_masks: a species under its floor
        # contributes nothing to the cloud water path.
        with np.errstate(invalid="ignore"):
            lwp_eff = np.where(valid & (lwp_g > min_lwp_g), lwp_g, 0.0)
            iwp_eff = np.where(valid & (iwp_g > min_iwp_g), iwp_g, 0.0)
        cwp = lwp_eff + iwp_eff
        has_cloud = valid & (cwp > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ice_frac = np.where(has_cloud, iwp_eff / np.where(cwp > 0.0, cwp, 1.0),
                                np.nan)

        tcc_l.append(tcc)
        ice_frac_l.append(ice_frac)
        has_cloud_l.append(has_cloud)
        valid_l.append(valid)
        rain_l.append(np.isfinite(rate) & (rate >= precip_rate_max_mm_hr))

    cat = lambda parts: np.concatenate(parts)               # noqa: E731
    # layout["seasons"] lists every season in the ARCHIVE; the module's arrays
    # are indexed by the SELECTED ones, A.keep_idx. Remap here so this table
    # shares an index with season_phase_binary rather than being off by however
    # many seasons --years and --min-season-coverage dropped. use_step already
    # restricts to selected seasons, so no kept hour maps to -1.
    remap = np.full(len(seasons), -1, dtype=np.intp)
    remap[A.keep_idx] = np.arange(len(A.keep_idx), dtype=np.intp)
    si = remap[cat(si_l).astype(np.intp)]
    if (si < 0).any():
        raise AssertionError("an hour outside the selected seasons was kept")
    mi = cat(mi_l).astype(np.intp)
    n_season, n_month = len(A.keep_idx), len(months)
    valid = cat(valid_l)

    S = {
        "si": si, "mi": mi,
        "tcc": cat(tcc_l),
        "ice_frac": cat(ice_frac_l),
        "has_cloud": cat(has_cloud_l),
        "raining": cat(rain_l),
        "valid": valid,
        "seasons": list(A.used),
        "months": list(months),
        "n_season": n_season,
        "n_month": n_month,
        # Per season, exactly as to_hours_per_season does: a common-year
        # Oct-Mar window is 182 days and a leap-year one 183, so a complete
        # common year must scale by 1.0 rather than by 183/182.
        "season_hours": season_window_hours(layout, A.keep_idx),   # (season,)
        "month_hours": season_month_window_hours(layout, A.keep_idx),
        "season_start": tuple(args.season_start),
        "season_end": tuple(args.season_end),
        "site_lat": site_lat, "site_lon": site_lon,
        "precip_rate_max": precip_rate_max_mm_hr,
        "min_lwp_g": min_lwp_g, "min_iwp_g": min_iwp_g,
    }
    # Denominators: valid hours per season, and per (season, month). Neither
    # depends on the free parameters, so both are computed once.
    S["den_season"] = np.bincount(si[valid], minlength=n_season).astype(float)
    S["den_month"] = np.bincount(
        (si * n_month + mi)[valid],
        minlength=n_season * n_month).reshape(n_season, n_month).astype(float)
    return S


# ----------------------------------------------------------------------------
# The evaluator
# ----------------------------------------------------------------------------
def _phase_hit(S: dict, tcc_min: float, ice_frac_min: float, no_precip: bool,
               want: str) -> np.ndarray:
    """Boolean over hours: liquid-containing (or ice-only) at these thresholds.

    ``liquid`` is cloudy AND holding cloud water AND not ice-only. ``ice`` is
    the complement within the cloudy population, which is ice-only PLUS the
    "no phase" residual -- matching how the figures fold the two together.
    """
    cloudy = S["valid"] & (S["tcc"] >= tcc_min)
    if no_precip:
        cloudy = cloudy & ~S["raining"]
    with np.errstate(invalid="ignore"):
        is_ice = S["has_cloud"] & (S["ice_frac"] >= ice_frac_min)
    if want == "liquid":
        return cloudy & S["has_cloud"] & ~is_ice
    return cloudy & ~(S["has_cloud"] & ~is_ice)


def hours_seasonal(S: dict, tcc_min: float, ice_frac_min: float,
                   no_precip: bool, want: str = "liquid") -> np.ndarray:
    """Hours per season, normalised exactly as ``season_phase_binary`` is."""
    hit = _phase_hit(S, tcc_min, ice_frac_min, no_precip, want)
    num = np.bincount(S["si"][hit], minlength=S["n_season"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(S["den_season"] > 0,
                        num / np.where(S["den_season"] > 0, S["den_season"], 1.0)
                        * np.asarray(S["season_hours"], dtype=float), np.nan)


def hours_monthly(S: dict, tcc_min: float, ice_frac_min: float,
                  no_precip: bool, want: str = "liquid") -> np.ndarray:
    """Hours per (season, month), normalised as ``monthly_phase_binary`` is."""
    hit = _phase_hit(S, tcc_min, ice_frac_min, no_precip, want)
    num = np.bincount((S["si"] * S["n_month"] + S["mi"])[hit],
                      minlength=S["n_season"] * S["n_month"]
                      ).reshape(S["n_season"], S["n_month"]).astype(float)
    den = S["den_month"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    mh = np.asarray(S["month_hours"], dtype=float)
    return frac * (mh if mh.ndim == 2 else mh[None, :])


# ----------------------------------------------------------------------------
# The three targets
# ----------------------------------------------------------------------------
DEFAULT_TCC_GRID = (0.90, 0.99, 0.001)          # (lo, hi, step), inclusive
DEFAULT_ICE_FRAC_GRID = (0.85, 0.95, 0.001)


def _grid(spec) -> np.ndarray:
    lo, hi, step = spec
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


class Target:
    """One fitting problem: which ERA5 hours, against which observed column.

    ``kind`` is 'season' or 'month'; ``no_precip`` selects the ERA5 population
    and, for the seasonal targets, which observation columns pair with it.
    """

    def __init__(self, key: str, label: str, kind: str, no_precip: bool):
        self.key, self.label = key, label
        self.kind, self.no_precip = kind, no_precip

    def observed(self, S: dict, obs: dict, obs_monthly: dict):
        """(observed hours, x labels) on the axis this target is fitted over."""
        if self.kind == "month":
            o_idx = {m: i for i, m in enumerate(obs_monthly["months"])}
            shared = [m for m in S["months"] if m in o_idx]
            return (np.array([obs_monthly["with_liquid"][o_idx[m]] for m in shared]),
                    shared)
        o_liq, _o_ice = lwph.obs_binary(obs, exclude_precip=self.no_precip)
        o_idx = {y: i for i, y in enumerate(obs["seasons"])}
        shared = [y for y in S["seasons"]
                  if y in o_idx and y not in EXCLUDED_FIT_SEASONS]
        return np.array([o_liq[o_idx[y]] for y in shared]), shared

    def modelled(self, S: dict, tcc_min: float, ice_frac_min: float,
                 obs: dict, obs_monthly: dict, want: str = "liquid"):
        """ERA5 hours on the same axis, with the same exclusions applied."""
        if self.kind == "month":
            h = hours_monthly(S, tcc_min, ice_frac_min, self.no_precip, want)
            drop, _hit = excluded_month_mask(S["seasons"], S["months"],
                                             _ArgsShim(S), GENIE_EXCLUDED_MONTHS)
            h = np.where(drop, np.nan, h)
            o_idx = {m: i for i, m in enumerate(obs_monthly["months"])}
            m_of = {m: k for k, m in enumerate(S["months"])}
            shared = [m for m in S["months"] if m in o_idx]
            return np.array([lwph.nanmean_quiet(h[:, m_of[m]]) for m in shared])
        h = hours_seasonal(S, tcc_min, ice_frac_min, self.no_precip, want)
        s_of = {y: k for k, y in enumerate(S["seasons"])}
        o_idx = {y: i for i, y in enumerate(obs["seasons"])}
        shared = [y for y in S["seasons"]
                  if y in o_idx and y not in EXCLUDED_FIT_SEASONS]
        return np.array([h[s_of[y]] for y in shared])


class _ArgsShim:
    """Just enough of ``args`` for :func:`excluded_month_mask`.

    Carries the run's actual season window, copied in ``extract_site_series``,
    so the calendar-month -> season mapping here is the same one the figures
    use rather than a re-derivation from the month axis.
    """

    def __init__(self, S):
        self.season_start = S["season_start"]
        self.season_end = S["season_end"]


TARGETS: tuple[Target, ...] = (
    Target("season_noprecip", "Seasonal, precipitation filtered (section 5)",
           "season", True),
    Target("season_allsky", "Seasonal, all sky (section 6)", "season", False),
    Target("monthly_allsky", "Monthly means, all sky (section 7)",
           "month", False),
)


# ----------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------
def cost_surface(S: dict, target: Target, obs: dict, obs_monthly: dict,
                 tcc_grid=None, ice_frac_grid=None):
    """Root-sum-square liquid-containing residual over the whole grid.

    Returns ``(tcc_values, ice_frac_values, cost)`` with ``cost`` shaped
    ``(n_tcc, n_ice_frac)`` in hours.

    Exhaustive by design -- see the module docstring. The inner call is a
    comparison and a bincount over ~50,000 rows, so the whole surface costs
    seconds and no evaluation is wasted: every point is reported.
    """
    tcc_v = _grid(tcc_grid or DEFAULT_TCC_GRID)
    ifm_v = _grid(ice_frac_grid or DEFAULT_ICE_FRAC_GRID)
    o_liq, _ = target.observed(S, obs, obs_monthly)

    cost = np.full((tcc_v.size, ifm_v.size), np.nan)
    for a, tcc_min in enumerate(tcc_v):
        for b, ifm in enumerate(ifm_v):
            e_liq = target.modelled(S, float(tcc_min), float(ifm), obs,
                                    obs_monthly)
            r = e_liq - o_liq
            cost[a, b] = float(np.sqrt(np.nansum(r * r)))
    return tcc_v, ifm_v, cost


def argmin_2d(tcc_v, ifm_v, cost):
    """(tcc, ice_fraction, cost) at the minimum, and how many cells tie it."""
    k = int(np.nanargmin(cost))
    a, b = np.unravel_index(k, cost.shape)
    best = float(cost[a, b])
    # The surface is a staircase, so exact ties are expected, not a curiosity:
    # they say the data cannot separate those threshold pairs at all.
    n_tie = int(np.count_nonzero(np.isclose(cost, best, rtol=0, atol=1e-9)))
    return float(tcc_v[a]), float(ifm_v[b]), best, n_tie


# ----------------------------------------------------------------------------
# Verification: the fast evaluator must reproduce the figures exactly
# ----------------------------------------------------------------------------
def verify(S: dict, A, tol_h: float = 0.5) -> bool:
    """Check the single-cell evaluator against the module's own numbers.

    The fit reimplements the normalisation on a reduced table, so it is only
    trustworthy if it reproduces ``season_phase_binary`` and
    ``monthly_phase_binary`` at the run's OWN thresholds. Raises if it does not:
    a silent disagreement here would move every optimum below.
    """
    tcc0 = float(A.args.min_cloud_fraction)
    ifm0 = float(A.phase_kw["ice_fraction_min"])
    ok = True

    _labels, liquid, _ice, _clear, _sh = lwph.season_phase_binary(A)
    code, _lab = lwph.resolve_series_code(A.col, "arm_site")
    ref = liquid[:, code]
    got = hours_seasonal(S, tcc0, ifm0, A.args.no_precip)
    d = np.nanmax(np.abs(got - ref))
    print(f"  seasonal   max |fit - module| = {d:9.4f} h "
          f"({'OK' if d <= tol_h else 'MISMATCH'})")
    ok &= d <= tol_h

    if not A.args.no_precip:
        _m, mliq, _mi, _mh = lwph.monthly_phase_binary(A, exclude_months=())
        got_m = hours_monthly(S, tcc0, ifm0, False)
        d = np.nanmax(np.abs(got_m - mliq))
        print(f"  monthly    max |fit - module| = {d:9.4f} h "
              f"({'OK' if d <= tol_h else 'MISMATCH'})")
        ok &= d <= tol_h

    if not ok:
        raise AssertionError(
            "the fast evaluator does not reproduce the module's hours at the "
            "run's own thresholds; the fit below would be meaningless")
    return ok


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report_optimum(S: dict, target: Target, obs: dict, obs_monthly: dict,
                   tcc_v, ifm_v, cost, A=None) -> dict:
    """Print the optimum, the baseline it beats, and the ice-only side effect.

    Ice is NOT in the cost, so it is free to get worse. Printing it is the check
    on whether the fit found hours or merely relabelled them.
    """
    tcc_b, ifm_b, best, n_tie = argmin_2d(tcc_v, ifm_v, cost)
    o_liq, xs = target.observed(S, obs, obs_monthly)
    o_ice = (lwph.obs_binary(obs, exclude_precip=target.no_precip)[1]
             if target.kind == "season" else None)

    e_liq = target.modelled(S, tcc_b, ifm_b, obs, obs_monthly)
    edge = []
    if tcc_b in (tcc_v[0], tcc_v[-1]):
        edge.append(f"tcc pinned to the {'lower' if tcc_b == tcc_v[0] else 'upper'} bound")
    if ifm_b in (ifm_v[0], ifm_v[-1]):
        edge.append(f"ice fraction pinned to the "
                    f"{'lower' if ifm_b == ifm_v[0] else 'upper'} bound")

    print(f"\n{target.label}")
    print(f"  fitted   min_cloud_fraction = {tcc_b:.3f}   "
          f"ice_fraction_min = {ifm_b:.3f}")
    print(f"  RSS      {best:,.1f} h   over {len(xs)} "
          f"{'seasons' if target.kind == 'season' else 'months'}"
          f"   (RMS {best / np.sqrt(len(xs)):,.1f} h)")
    if n_tie > 1:
        print(f"  {n_tie:,} grid cells tie this exactly -- the surface is flat "
              f"here, so these thresholds are not separately identified")
    if edge:
        print(f"  !! {'; '.join(edge)} -- the fit wanted to leave the allowed "
              f"range, so this is a bound, not a fitted value")
    if A is not None:
        c0 = _cost_at(S, target, obs, obs_monthly,
                      float(A.args.min_cloud_fraction),
                      float(A.phase_kw["ice_fraction_min"]))
        print(f"  baseline RSS at ({A.args.min_cloud_fraction:.3f}, "
              f"{A.phase_kw['ice_fraction_min']:.3f}) = {c0:,.1f} h"
              f"   -> improvement {100 * (1 - best / c0):.1f}%")

    # Ice-only is NOT in the cost, so it is free to get worse. Reporting it for
    # BOTH kinds of target is the check on whether the fit found hours or just
    # moved them across the liquid/ice boundary.
    if target.kind == "season":
        o_idx = {y: i for i, y in enumerate(obs["seasons"])}
        oi = np.array([o_ice[o_idx[y]] for y in xs])
    else:
        o_idx = {m: i for i, m in enumerate(obs_monthly["months"])}
        oi = np.array([obs_monthly["ice_only"][o_idx[m]] for m in xs])
    ei = target.modelled(S, tcc_b, ifm_b, obs, obs_monthly, want="ice")
    rss_ice = float(np.sqrt(np.nansum((ei - oi) ** 2)))
    extra = ""
    if A is not None:
        ei0 = target.modelled(S, float(A.args.min_cloud_fraction),
                              float(A.phase_kw["ice_fraction_min"]), obs,
                              obs_monthly, want="ice")
        extra = (f"   (was {float(np.sqrt(np.nansum((ei0 - oi) ** 2))):,.1f} h "
                 f"at the baseline)")
    print(f"  ice-only RSS at this optimum = {rss_ice:,.1f} h{extra}"
          f"   [not minimised]")
    return {"tcc": tcc_b, "ice_frac": ifm_b, "rss": best, "n_tie": n_tie,
            "x": xs, "obs": o_liq, "era5": e_liq,
            "obs_ice": oi, "era5_ice": ei, "rss_ice": rss_ice}


def _cost_at(S, target, obs, obs_monthly, tcc_min, ifm) -> float:
    o_liq, _ = target.observed(S, obs, obs_monthly)
    r = target.modelled(S, tcc_min, ifm, obs, obs_monthly) - o_liq
    return float(np.sqrt(np.nansum(r * r)))


def fig_cost_surfaces(S, results, surfaces, A=None, out_dir=None, dpi=200):
    """The three cost surfaces side by side, each with its minimum marked.

    The surface is the point of the exercise: a tight closed contour means the
    two thresholds are separately identified, a diagonal trough means only some
    combination of them is.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6.0 * len(TARGETS), 5.6))
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, TARGETS):
        tcc_v, ifm_v, cost = surfaces[target.key]
        r = results[target.key]
        pcm = ax.pcolormesh(ifm_v, tcc_v, cost, shading="nearest",
                            cmap="viridis")
        levels = np.nanpercentile(cost, [2, 5, 10, 20, 40, 70])
        ax.contour(ifm_v, tcc_v, cost, levels=np.unique(levels),
                   colors="white", linewidths=0.6, alpha=0.6)
        ax.plot(r["ice_frac"], r["tcc"], marker="*", ms=20, mfc="red",
                mec="white", mew=1.2, ls="none",
                label=f"min: {r['tcc']:.3f}, {r['ice_frac']:.3f}")
        # if A is not None:
        #     ax.plot(A.phase_kw["ice_fraction_min"], A.args.min_cloud_fraction,
        #             marker="o", ms=10, mfc="none", mec="white", mew=2.0,
        #             ls="none", label="current setting")
        ax.set_xlabel("Ice-Only Threshold   IWP/(LWP+IWP)")
        ax.set_ylabel("Cloud Fraction threshold")
        ax.set_title(target.label, fontsize=10.5)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
        fig.colorbar(pcm, ax=ax, label="RSS of liquid-containing residual [h]")
    fig.suptitle("Cost surface: root-sum-square liquid-containing residual, "
                 "ERA5 minus ARM observations", fontsize=13, y=1.0)
    fig.tight_layout()
    if out_dir is not None:
        path = Path(out_dir) / "threshold_fit_cost_surfaces.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig
