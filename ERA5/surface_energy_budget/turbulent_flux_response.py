#!/usr/bin/env python3
"""How the surface turbulent fluxes respond to a longwave radiative forcing.

THE QUESTION
------------
The surface sensible heat flux is a bulk flux driven by the skin-to-air
temperature difference,

    SH_up  ~  rho * c_p * C_H * U * (T_skin - T_2m)          [W m-2, up]

so it is not independent of the radiation field. Lower the downwelling
longwave (DLR) over a surface with little heat capacity and T_skin falls; the
skin-to-air difference falls with it, and the turbulent flux that had been
carrying heat AWAY from the surface weakens. The turbulent term therefore acts
as a DAMPER on longwave forcing: it gives back part of what the radiation took.
How much it gives back is a property of the surface, because T_skin's freedom
to move is a property of the surface -- open water is pinned near the freezing
point by an effectively infinite heat capacity, pack ice is not.

SIGN CONVENTION -- READ THIS BEFORE INTERPRETING ANY NUMBER HERE
----------------------------------------------------------------
Every ERA5 surface flux is stored POSITIVE DOWNWARD (into the surface), and
this module keeps that convention for the stored variables:

    msshf > 0   heat flowing from the air INTO the surface
    msshf < 0   heat flowing from the surface INTO the air (upward SH)

MEASURED on the Barrow strip under liquid-bearing overcast (Nov-Dec 2023,
2.37 million cell-hours), T_skin exceeds T_2m 69.2% of the time, median +0.40 K,
so msshf is typically NEGATIVE there: the sensible heat
flux is carrying heat UP, out of the surface. In that regime a DLR decrease
cools the skin, shrinks (T_skin - T_2m), and makes msshf LESS negative -- the
upward flux weakens, less heat leaves the surface, and the cooling is damped.
That is the mechanism this module measures. Note that "the sensible heat flux
weakens" and "the sensible heat flux warms the surface relative to what it
would otherwise have been" are the same statement here only because the flux is
upward to begin with. Over pack ice in midwinter the sign often reverses
(T_skin < T_2m under the surface inversion, msshf > 0), and there the same DLR
decrease STRENGTHENS a downward flux. Both are damping; the sign of the flux is
not the sign of the feedback. The figures below separate the two regimes by
surface class rather than averaging over them.

WHAT IS ACTUALLY BEING MEASURED -- AN IMPORTANT CAVEAT
------------------------------------------------------
Nothing here is a controlled forcing experiment. Every slope below is a
COVARIANCE across the natural synoptic variability of the record: hours with
high DLR are cloudy, and cloudy hours also differ in wind speed, air mass
origin, and boundary-layer stability. The regression of a surface response on
DLR therefore mixes the direct radiative response with everything else that
travels with a cloud. Read these numbers as "how the surface state co-varies
with DLR", which is an upper bound on the pure radiative response and is the
quantity a reanalysis-versus-observations comparison can actually check. A
clean partition would need a perturbed-DLR experiment, which ERA5 cannot
supply. The MIZ figure below is the closest thing available: it holds the air
mass roughly fixed and varies the surface.

THE MARGINAL ICE ZONE AS A NATURAL EXPERIMENT
---------------------------------------------
Across a few tens of kilometres the surface goes from open water pinned near
-1.8 C to pack ice tens of degrees colder, while the overlying air mass is
nearly unchanged. So the MIZ is where the surface-response gradient is
steepest, and it is the one place the record offers something close to a
controlled contrast: same forcing, different surface. ``fig_miz_transect``
uses sea ice concentration as the coordinate and resolves the transition
continuously, rather than binning it into the three discrete ocean classes.

THE PARTITION
-------------
With every term positive downward, the surface energy balance is

    R = LWD - LWU + SWnet + SH + LH                          [W m-2, down]

with R the net convergence, which over ice goes into conduction, storage and
melt. Differentiating with respect to LWD and rearranging gives an exact
partition of each additional W m-2 of DLR:

    1 = f_LWU + f_SH + f_LH + f_SW + f_res

    f_LWU =  d(LWU)/d(LWD)      radiated straight back out (Stefan-Boltzmann)
    f_SH  = -d(msshf)/d(LWD)    carried away by upward sensible heat flux
    f_LH  = -d(mslhf)/d(LWD)    carried away by upward latent heat flux
    f_SW  = -d(SWnet)/d(LWD)    shortwave co-variation (~0 in polar night)
    f_res =  d(R)/d(LWD)        left over, into the subsurface / melt

The five sum to one BY CONSTRUCTION, not by luck: the slope operator is linear
and d(LWD)/d(LWD) = 1, so f_res is defined as the remainder. That is what makes
the stacked bar in ``fig_response_partition`` a genuine partition and not a
collection of unrelated regressions. The companion number, d(T_skin)/d(LWD) in
K per W m-2, is the surface's thermal freedom -- and it is what f_LWU and f_SH
are both proportional to, which is why they rise and fall together across the
ice edge.

POPULATION
----------
The scatter figures use exactly the population
``plot_lwp_histogram_by_surface_class.py`` draws: overcast
(tcc >= --min-cloud-fraction) and liquid-bearing (the union of that module's
"liquid only" and "mixed phase" categories). The masks are IMPORTED from that
module rather than reimplemented, so the two cannot drift apart. An unfiltered
all-sky population is accumulated in the same pass at negligible cost, so any
figure can be drawn either way with ``population="all"`` and the filtered and
unfiltered answers compared without a reload.

FIGURES
-------
1. ``fig_shf_vs_lwp``       SHF against liquid water path, density-coloured,
                            per surface class, with a least-squares fit.
2. ``fig_shf_vs_dlr``       SHF against downwelling longwave, same treatment.
3. ``fig_shf_vs_dskt``      SHF against (T_skin - T_2m): the bulk relation the
                            whole argument rests on, shown rather than assumed.

   All three take ``fit_mode=``, which selects what is drawn over the density:
   "default" is the cos(latitude)-weighted fit plus the mean of y in each x
   column, "regression" is the unweighted regression alone as a thin solid
   black line. Both are ordinary least squares on the full unbinned sample and
   both are annotated with slope and r^2; see FIT_MODES for why both weightings
   are worth having. Both moment sets are accumulated in the same pass, so the
   mode is a drawing option and never needs a reload.
4. ``fig_response_partition`` d(T_skin)/d(LWD) and the five-way partition of
                            d(LWD), by surface class, plus binned response
                            curves.
5. ``fig_miz_transect``     the same sensitivities against sea ice
                            concentration, resolving the ice edge continuously.

USAGE
-----
    A = prepare(region="barrow", years=tuple(range(2022, 2026)),
                season_start=(10, 1), season_end=(3, 31),
                min_cloud_fraction=0.99)
    print_report(A)
    fig_shf_vs_lwp(A)                        # whatever --fit-mode was set
    fig_shf_vs_lwp(A, fit_mode="regression")  # no reload

or from the command line::

    python turbulent_flux_response.py --region barrow --years 2022-2025 \
        --season-start 10-01 --season-end 03-31 --min-cloud-fraction 0.99 \
        --fit-mode regression --output-dir figures/turbulent_response
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    UNCLASSIFIED,
    add_classification_args,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)
from plot_surface_class_timeseries import (
    SITE_COLOR,
    SITE_KEY,
    SITE_LABEL,
    parse_month_day,
    parse_years,
    season_layout,
    select_seasons,
    site_cell_mask,
)

# The cloud-phase definition and the precipitation filter are IMPORTED, not
# restated. "Liquid-bearing" here means exactly what it means on the LWP
# histogram: the union of that module's liquid-only and mixed-phase masks,
# under whatever thresholds the caller passed. Reimplementing either would let
# the two figures disagree about which hours they are describing.
from plot_lwp_histogram_by_surface_class import (
    PHASE_STACK,
    add_fraction_phase_args,
    add_phase_args,
    phase_definition_label,
    phase_masks,
    precip_label,
    precip_mask,
    resolve_phase_thresholds,
    PHASE_MODES,
    DEFAULT_PHASE_MODE,
    DEFAULT_MIN_CLOUD_FRACTION,
)

# ERA5 hourly data: one time step is one cell-hour, so a count IS an hour count.
HOURS_PER_STEP = 1.0

# Time steps held in memory at once. Smaller than the classification default
# because this pass materialises eleven fields rather than four, and builds a
# dense (n_var, n_cell_hour) working array on top of them.
DEFAULT_BLOCK_HOURS = 24 * 15


# ----------------------------------------------------------------------------
# The tracked variables
# ----------------------------------------------------------------------------
class Tracked(NamedTuple):
    """One field carried through the moment accumulation.

    Attributes
    ----------
    key : Name used everywhere downstream, carrying its units.
    label : Axis label, mathtext.
    units : Units, mathtext, for axis labels and slope units.
    center : Constant subtracted before accumulation. Purely numerical: a
        second moment of an absolute temperature is ~7e4 per sample while its
        variance is ~50, and differencing those at 1e8 samples throws away
        three digits for no reason. Slopes and covariances are invariant to it;
        means have it added back in ``mean_of``.
    """

    key: str
    label: str
    units: str
    center: float


TRACKED: tuple[Tracked, ...] = (
    Tracked("shf_W_m2", "Sensible heat flux", "W m$^{-2}$", 0.0),
    Tracked("lhf_W_m2", "Latent heat flux", "W m$^{-2}$", 0.0),
    Tracked("lwd_W_m2", "Downwelling longwave", "W m$^{-2}$", 230.0),
    Tracked("lwu_W_m2", "Upwelling longwave", "W m$^{-2}$", 250.0),
    Tracked("swnet_W_m2", "Net shortwave", "W m$^{-2}$", 0.0),
    Tracked("skt_K", "Skin temperature", "K", 260.0),
    Tracked("t2m_K", "2 m air temperature", "K", 260.0),
    Tracked("dskt_t2m_K", r"$T_{skin} - T_{2m}$", "K", 0.0),
    Tracked("lwp_g_m2", "Liquid water path", "g m$^{-2}$", 0.0),
    Tracked("iwp_g_m2", "Ice water path", "g m$^{-2}$", 0.0),
    Tracked("wspd_m_s", "10 m wind speed", "m s$^{-1}$", 5.0),
)

VAR_INDEX: dict[str, int] = {t.key: i for i, t in enumerate(TRACKED)}
N_VAR = len(TRACKED)
CENTERS = np.array([t.center for t in TRACKED])

# siconc is deliberately NOT tracked: it is NaN over land, and a moment matrix
# that required it finite would silently drop every land cell-hour. It is the
# BINNING coordinate of the MIZ figure instead, where the population is sea
# cells only and it is always defined.
READ_VARS: tuple[str, ...] = (
    "tcc", "tclw", "tciw", "siconc",
    "msshf", "mslhf", "msdwlwrf", "msnlwrf", "msnswrf",
    "skt", "t2m", "u10", "v10",
)


def derived_fields(block, keep: np.ndarray) -> dict[str, np.ndarray]:
    """Build every tracked field for one block, already subset to ``keep``.

    Returns float64 arrays shaped ``(n_kept, n_lat, n_lon)``. Two are derived
    rather than read: ``lwu_W_m2`` from ``LWD - (LWD - LWU)``, since ERA5
    stores the net rather than the upwelling component, and ``dskt_t2m_K``,
    which is the bulk-flux driver and is wanted often enough to be worth a
    column of its own.
    """
    v = {name: block[name].values[keep].astype(np.float64) for name in
         ("msshf", "mslhf", "msdwlwrf", "msnlwrf", "msnswrf", "skt", "t2m",
          "u10", "v10", "tclw", "tciw")}
    out = {
        "shf_W_m2": v["msshf"],
        "lhf_W_m2": v["mslhf"],
        "lwd_W_m2": v["msdwlwrf"],
        # msnlwrf = LWD - LWU, so LWU = LWD - msnlwrf.
        "lwu_W_m2": v["msdwlwrf"] - v["msnlwrf"],
        "swnet_W_m2": v["msnswrf"],
        "skt_K": v["skt"],
        "t2m_K": v["t2m"],
        "dskt_t2m_K": v["skt"] - v["t2m"],
        "lwp_g_m2": v["tclw"] * 1000.0,          # kg m-2 -> g m-2
        "iwp_g_m2": v["tciw"] * 1000.0,
        "wspd_m_s": np.hypot(v["u10"], v["v10"]),
    }
    return out


# ----------------------------------------------------------------------------
# Class slots
# ----------------------------------------------------------------------------
# The five surface classes, then the ARM site cell, then every valid cell. The
# last two are NOT extra classes: the site already sits inside whichever class
# it falls in on a given hour, and "all" is the whole domain. They are extra
# SLOTS so a panel can be drawn for each without a second pass.
SITE_SLOT = len(CLASS_ORDER)
ALL_SLOT = len(CLASS_ORDER) + 1
N_SLOT = len(CLASS_ORDER) + 2

SLOT_ORDER: tuple[str, ...] = CLASS_ORDER + (SITE_KEY, "all")
SLOT_LABELS: dict[str, str] = dict(CLASS_LABELS)
SLOT_LABELS[SITE_KEY] = SITE_LABEL
SLOT_LABELS["all"] = "All cells"
SLOT_COLORS: dict[str, str] = dict(CLASS_COLORS)
SLOT_COLORS[SITE_KEY] = SITE_COLOR
SLOT_COLORS["all"] = "#555555"

# The six panels the scatter figures draw, in reading order: the five classes
# then the site cell. "all" is accumulated but not panelled -- pooling five
# surfaces with opposite turbulent regimes into one scatter is the exact
# average the rest of this module exists to avoid.
PANEL_SLOTS: tuple[int, ...] = tuple(range(len(CLASS_ORDER))) + (SITE_SLOT,)

POPULATIONS: tuple[str, ...] = ("cloud", "all")


# ----------------------------------------------------------------------------
# 2-D density histograms
# ----------------------------------------------------------------------------
class Panel2D(NamedTuple):
    """Axis configuration for one density-scatter figure."""

    x_key: str
    y_key: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_bins: int
    y_bins: int


# Ranges are fixed rather than derived from the data so that runs over
# different seasons stay directly comparable panel to panel. They were chosen
# from the 1st-99th percentile span of the Barrow strip under liquid-bearing
# overcast, widened to the nearest round number; the fraction of samples
# falling outside is accumulated and printed on the figure, so a range that
# turns out to be wrong for another region announces itself instead of
# silently clipping.
DEFAULT_PANELS: dict[str, Panel2D] = {
    "shf_lwp": Panel2D("shf_W_m2", "lwp_g_m2", (-250.0, 60.0), (0.0, 300.0),
                       180, 180),
    "shf_dlr": Panel2D("shf_W_m2", "lwd_W_m2", (-250.0, 60.0), (140.0, 330.0),
                       180, 180),
    "shf_dskt": Panel2D("shf_W_m2", "dskt_t2m_K", (-250.0, 60.0),
                        (-8.0, 14.0), 180, 180),
}

# Bin edges of the binned-response curves in fig_response_partition. Wide
# enough to hold the whole DLR distribution; bins holding less than
# MIN_CURVE_HOURS cell-hours are dropped rather than drawn as noise.
DEFAULT_DLR_CURVE_EDGES = np.linspace(150.0, 310.0, 17)
MIN_CURVE_HOURS = 200.0

# Sea ice concentration bins for the MIZ transect. Twenty equal bins resolve
# the 0.05-0.95 marginal band in eighteen of them, which is the point.
DEFAULT_SICONC_EDGES = np.linspace(0.0, 1.0, 21)
MIN_MIZ_HOURS = 500.0


# ----------------------------------------------------------------------------
# Weighted moment accumulation
# ----------------------------------------------------------------------------
def new_moments(n_group: int) -> dict:
    """An empty moment accumulator for ``n_group`` groups.

    Holds the zeroth, first and second moments of every tracked variable, which
    between them supply every mean, variance, covariance, regression slope and
    correlation this module reports -- from ONE streaming pass, and without
    ever holding a sample in memory.

    BOTH WEIGHTINGS ARE CARRIED. ``w``/``x``/``xy`` are weighted by
    cos(latitude); ``n``/``x_u``/``xy_u`` are the same moments with every
    cell-hour counted once. The two answer different questions and neither is
    the correction of the other:

      * AREA-WEIGHTED is the right denominator for a statement about the
        REGION -- a cell at 79 N covers half the ground of one at 70 N, so
        counting them equally over-represents the north of the strip.
      * UNWEIGHTED is the right denominator for a statement about the SAMPLE --
        "of the cell-hours in this class, how do the two variables co-vary" --
        and it is what a reader comparing against a point measurement or a
        published scatter plot will assume.

    On this domain the two barely differ, because cos(latitude) varies by only
    a factor of two across 70-80 N. Carrying both costs two extra BLAS calls
    per group and removes the question.
    """
    return {
        "w": np.zeros(n_group),
        "n": np.zeros(n_group),
        "x": np.zeros((n_group, N_VAR)),
        "xy": np.zeros((n_group, N_VAR, N_VAR)),
        "x_u": np.zeros((n_group, N_VAR)),
        "xy_u": np.zeros((n_group, N_VAR, N_VAR)),
    }


def accumulate_moments(acc: dict, groups, values: np.ndarray,
                       weights: np.ndarray) -> None:
    """Fold one block's samples into ``acc``.

    Parameters
    ----------
    groups : Iterable of ``(slot, mask)``, mask flat and boolean over the
        sample axis. Slots may overlap -- the ARM site cell is also a member of
        whichever class it falls in -- so this is a loop rather than one
        grouped reduction.
    values : ``(N_VAR, n_sample)``, already centred.
    weights : ``(n_sample,)`` area weights.
    """
    for slot, mask in groups:
        k = int(np.count_nonzero(mask))
        if k == 0:
            continue
        xs = values[:, mask]                       # (N_VAR, k)
        ws = weights[mask]                         # (k,)
        acc["w"][slot] += float(ws.sum())
        acc["n"][slot] += float(k)
        acc["x"][slot] += xs @ ws
        # One GEMM fills the whole symmetric second-moment block. Doing it
        # entry by entry costs N_VAR^2 passes over the samples for the same
        # numbers.
        acc["xy"][slot] += (xs * ws) @ xs.T
        # The same two moments with every cell-hour counted once. See
        # new_moments for why both weightings are kept.
        acc["x_u"][slot] += xs.sum(axis=1)
        acc["xy_u"][slot] += xs @ xs.T


def moment_stats(acc: dict, slot: int, x_key: str, y_key: str,
                 weighted: bool = True) -> dict:
    """Means, the ordinary least-squares fit of y on x, and its r and r^2.

    Everything comes out of the accumulated moments, so this is exact on the
    full unbinned sample -- the 2-D histograms are for DRAWING only and the fit
    never touches them. Returns NaNs rather than raising when the slot is empty
    or x has no spread, since an absent class is a normal state, not an error.

    ``weighted`` selects which of the two accumulated moment sets to read:
    True for the cos(latitude)-weighted one, False for the plain per-cell-hour
    one. BOTH ARE ORDINARY LEAST SQUARES -- the only difference is the
    denominator, so "the weighted fit" and "the regression" below are the same
    estimator applied to two different populations, not two different methods.

    ``r2`` is the square of the Pearson correlation, which for a simple OLS fit
    is also the fraction of the variance in y that the fit explains. On a
    two-step relation such as LWP against SHF -- LWP drives DLR, DLR drives
    T_skin, T_skin drives SHF -- expect it to be small, and read it as a
    statement about how much of the scatter one straight line accounts for
    rather than as a verdict on whether the mechanism is there.
    """
    ix, iy = VAR_INDEX[x_key], VAR_INDEX[y_key]
    w_key, x_key_m, xy_key = (("w", "x", "xy") if weighted
                              else ("n", "x_u", "xy_u"))
    w = acc[w_key][slot]
    out = {"n_hours": float(acc["n"][slot]) * HOURS_PER_STEP,
           "weight": float(acc["w"][slot]), "weighted": bool(weighted)}
    if w <= 0.0:
        out.update(slope=np.nan, intercept=np.nan, r=np.nan, r2=np.nan,
                   x_mean=np.nan, y_mean=np.nan, x_sd=np.nan, y_sd=np.nan)
        return out

    mx_c = acc[x_key_m][slot, ix] / w
    my_c = acc[x_key_m][slot, iy] / w
    sxx = acc[xy_key][slot, ix, ix] / w - mx_c * mx_c
    syy = acc[xy_key][slot, iy, iy] / w - my_c * my_c
    sxy = acc[xy_key][slot, ix, iy] / w - mx_c * my_c
    x_mean = mx_c + CENTERS[ix]
    y_mean = my_c + CENTERS[iy]

    slope = sxy / sxx if sxx > 0.0 else np.nan
    denom = np.sqrt(sxx * syy)
    r = float(sxy / denom) if denom > 0.0 else np.nan
    out.update(
        slope=float(slope),
        intercept=float(y_mean - slope * x_mean) if np.isfinite(slope) else np.nan,
        r=r,
        r2=float(r * r) if np.isfinite(r) else np.nan,
        x_mean=float(x_mean), y_mean=float(y_mean),
        # Rounding can push a variance a hair below zero when a class is nearly
        # constant; clip rather than emit a NaN standard deviation.
        x_sd=float(np.sqrt(max(sxx, 0.0))), y_sd=float(np.sqrt(max(syy, 0.0))),
    )
    return out


# ----------------------------------------------------------------------------
# Partial regression: separating the surface response from the air mass
# ----------------------------------------------------------------------------
# THE CONFOUND, AND WHY A PLAIN SLOPE IS NOT ENOUGH.
#
# An hour with high DLR is an hour with a cloud, and an hour with a cloud in the
# Arctic is usually an hour of warm, moist advection. So T_2m rises with DLR --
# and the sensible heat flux responds to T_2m directly, through the same bulk
# formula, with nothing radiative about it. A simple regression of SHF on DLR
# therefore charges the radiation for the air mass's own contribution.
#
# MEASURED, Barrow strip, Nov-Dec 2023, liquid-bearing overcast, OPEN OCEAN:
# the simple regression returns d(SHF)/d(LWD) = +3.2 W m-2 per W m-2, which as a
# partition coefficient is nonsense -- more energy moves than arrives. It is not
# a bug in the arithmetic; it is the air mass. Open water is the extreme case
# because the surface is pinned near freezing, so essentially the whole
# skin-to-air difference is the AIR moving.
#
# The fix is a partial derivative: regress on DLR AND the confounder together,
# and read the DLR coefficient. Holding T_2m fixed leaves only the pathway that
# runs through the surface, which is the one the question is about, since
# SH ~ (T_skin - T_2m) and T_2m is now held still.
#
# NEITHER ESTIMATE IS THE CAUSAL ANSWER, and they bracket it from opposite
# sides. The simple slope is an UPPER bound: it credits the radiation with
# everything the air mass did too. The partial slope is a LOWER bound: part of
# the T_2m variation is itself a genuine downstream response to the surface
# warming, and controlling for T_2m removes that real link along with the
# spurious one. Both are printed by ``print_report`` for exactly this reason.
CONTROL_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "t2m": ("t2m_K",),
    "t2m_wind": ("t2m_K", "wspd_m_s"),
}
# DEFAULT IS THE PLAIN REGRESSION, and the reason is that T_2m is a MEDIATOR as
# well as a confounder. Over sea ice the 2 m air temperature is largely slaved
# to the skin temperature beneath it, so the chain DLR -> T_skin -> T_2m is a
# real part of the response; holding T_2m fixed deletes that link along with
# the spurious advective one and biases every sensitivity toward zero.
# MEASURED, same window: controlling on T_2m drops d(T_skin)/d(LWD) over sea
# ice from 0.134 to 0.006 K per W m-2, which is not a corrected estimate, it is
# an over-corrected one. The plain slope is reported as the headline and the
# controlled slope is drawn beside it, so the reader sees the bracket rather
# than one end of it.
DEFAULT_CONTROL = "none"

# Which estimator is drawn BESIDE the chosen one. Always the opposite end of
# the bracket, so a figure never shows a marker sitting exactly on its own bar.
ALT_CONTROL: dict[str, str] = {"none": "t2m", "t2m": "none", "t2m_wind": "none"}

CONTROL_LABELS: dict[str, str] = {
    "none": "plain regression",
    "t2m": r"holding $T_{2m}$ fixed",
    "t2m_wind": r"holding $T_{2m}$ and wind fixed",
}


def _covariance_block(acc: dict, slot: int, keys: tuple[str, ...]):
    """Weighted covariance matrix over ``keys``, from the accumulated moments."""
    idx = [VAR_INDEX[k] for k in keys]
    w = acc["w"][slot]
    m = acc["x"][slot, idx] / w
    return acc["xy"][slot][np.ix_(idx, idx)] / w - np.outer(m, m)


def partial_slope(acc: dict, slot: int, y_key: str,
                  x_key: str = "lwd_W_m2",
                  control: tuple[str, ...] = ()) -> float:
    """d(y)/d(x) holding ``control`` fixed, by weighted multiple regression.

    With ``control`` empty this is the ordinary least-squares slope and agrees
    with ``moment_stats`` exactly. Solved from the accumulated second moments,
    so it costs a 2x2 or 3x3 solve rather than another pass over the archive.

    Returns NaN when the group is empty or the design matrix is singular -- an
    absent class and a class whose control variable never varies are both
    normal states here, not errors.
    """
    if acc["w"][slot] <= 0.0:
        return float("nan")
    keys = (x_key,) + tuple(k for k in control if k != x_key)
    cov = _covariance_block(acc, slot, keys + (y_key,))
    a = cov[:-1, :-1]
    b = cov[:-1, -1]
    try:
        beta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(beta[0])


def slope_of(acc: dict, slot: int, y_key: str, x_key: str = "lwd_W_m2",
             control: tuple[str, ...] = ()) -> float:
    """d(y)/d(x) for one group. Shorthand for the sensitivity tables."""
    return partial_slope(acc, slot, y_key, x_key, control)


def mean_of(acc: dict, slot: int, key: str) -> float:
    """Weighted mean of one tracked variable, with its centring added back."""
    w = acc["w"][slot]
    if w <= 0.0:
        return float("nan")
    return float(acc["x"][slot, VAR_INDEX[key]] / w + CENTERS[VAR_INDEX[key]])


# ----------------------------------------------------------------------------
# The partition of d(LWD)
# ----------------------------------------------------------------------------
# Order is stack order from the bottom of the bar. The residual is last because
# it is defined as the remainder.
PARTITION_TERMS: tuple[tuple[str, str, str], ...] = (
    ("f_lwu", "Upwelling LW", "#B2182B"),
    ("f_sh", "Sensible heat", "#4C72B0"),
    ("f_lh", "Latent heat", "#55A868"),
    ("f_sw", "Net shortwave", "#DD8452"),
    ("f_res", "Subsurface / storage", "#BBBBBB"),
)


def partition(acc: dict, slot: int,
              control: tuple[str, ...] = ()) -> dict:
    """Split each additional W m-2 of DLR into where it goes.

    Returns the five fractions of ``PARTITION_TERMS`` plus ``dskt_dlwd`` in
    K per W m-2 and the cell-hour count. See the module docstring for the
    derivation; the point to keep in mind when reading the numbers is that the
    five sum to exactly one by construction, so ``f_res`` absorbs every
    misfit -- including the part of the co-variation that has nothing to do
    with a radiative response.
    """
    f_lwu = slope_of(acc, slot, "lwu_W_m2", control=control)
    f_sh = -slope_of(acc, slot, "shf_W_m2", control=control)
    f_lh = -slope_of(acc, slot, "lhf_W_m2", control=control)
    f_sw = -slope_of(acc, slot, "swnet_W_m2", control=control)
    return {
        "f_lwu": f_lwu,
        "f_sh": f_sh,
        "f_lh": f_lh,
        "f_sw": f_sw,
        "f_res": 1.0 - f_lwu - f_sh - f_lh - f_sw,
        "dskt_dlwd": slope_of(acc, slot, "skt_K", control=control),
        "dt2m_dlwd": slope_of(acc, slot, "t2m_K", control=control),
        # THE SIGN OF THE TURBULENT RESPONSE LIVES HERE. The sensible heat flux
        # tracks (T_skin - T_2m), so whether a DLR increase strengthens or
        # weakens the upward flux depends on which of the two temperatures
        # moves further -- not on the skin temperature alone. Where this is
        # negative, the air warms faster than the surface, the skin-to-air
        # difference shrinks, and the flux moves energy INTO the surface: the
        # turbulent term adds to the radiative warming instead of damping it.
        "ddskt_dlwd": slope_of(acc, slot, "dskt_t2m_K", control=control),
        "control": control,
        "n_hours": float(acc["n"][slot]) * HOURS_PER_STEP,
        "lwd_mean": mean_of(acc, slot, "lwd_W_m2"),
        "skt_mean": mean_of(acc, slot, "skt_K"),
        "shf_mean": mean_of(acc, slot, "shf_W_m2"),
    }


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def collect(ds, lsm: np.ndarray, args, layout: dict, wanted_idx: list[int],
            panels: dict, dlr_edges: np.ndarray,
            siconc_edges: np.ndarray, phase_kw: dict) -> dict:
    """Accumulate every moment and histogram this module needs, in one pass.

    The archive open is the expensive step -- minutes -- and everything below
    is filled from the same blocks, so adding a figure costs nothing as long as
    what it needs is accumulated here. Only the seasons in ``wanted_idx`` are
    read at all.

    Returns a dict of accumulators; see ``prepare`` for what goes where.
    """
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]

    wanted = np.zeros(len(uniq_seasons), dtype=bool)
    wanted[wanted_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    site_mask, site_lat, site_lon = site_cell_mask(ds)
    weights_2d = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])

    n_dlr = len(dlr_edges) - 1
    n_ice = len(siconc_edges) - 1

    mom = {p: new_moments(N_SLOT) for p in POPULATIONS}
    # Response curves: the same first moments, but grouped by DLR bin instead
    # of pooled. Mean of each tracked variable in each bin, per class slot.
    curve_w = np.zeros((N_SLOT, n_dlr))
    curve_n = np.zeros((N_SLOT, n_dlr))
    curve_x = np.zeros((N_SLOT, n_dlr, N_VAR))
    # The MIZ transect: full moments per sea ice concentration bin, over SEA
    # cells only. One group axis, no class axis -- siconc IS the class here,
    # resolved continuously instead of cut into three.
    ice_mom = {p: new_moments(n_ice) for p in POPULATIONS}

    hist = {name: np.zeros((N_SLOT, p.x_bins, p.y_bins))
            for name, p in panels.items()}
    hist_out = {name: np.zeros(N_SLOT) for name in panels}   # off-range weight

    n_unclassified = 0
    n_valid = 0.0
    n_cloudy = 0.0
    n_precip_removed = 0.0
    site_class_counts = np.zeros(len(CLASS_ORDER) + 1, dtype=np.int64)

    read_vars = list(READ_VARS)
    if args.no_precip:
        read_vars += [v for v in ("tp", "tcrw", "tcsw")
                      if v in ds.data_vars and v not in read_vars]

    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        keep = use_step[i0:i0 + n_t]
        if not keep.any():
            continue

        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )[keep]
        n_unclassified += int((classes == UNCLASSIFIED).sum())
        for code in range(len(CLASS_ORDER)):
            site_class_counts[code] += int((classes[:, site_mask] == code).sum())
        site_class_counts[-1] += int((classes[:, site_mask] == UNCLASSIFIED).sum())

        fields = derived_fields(block, keep)
        tcc = block["tcc"].values[keep]
        siconc_k = siconc[keep]

        # One sample is one cell-hour. Everything below works on the flattened
        # sample axis, so a class mask, a phase mask and a bin index are all
        # just boolean or integer vectors of the same length.
        shape = tcc.shape
        finite = np.isfinite(tcc)
        for arr in fields.values():
            finite &= np.isfinite(arr)

        raining = precip_mask(block, keep, args)
        with np.errstate(invalid="ignore"):
            overcast = finite & (tcc >= args.min_cloud_fraction)
        n_precip_removed += float(np.count_nonzero(overcast & raining))
        cloudy = overcast & ~raining
        # "Liquid-bearing" is the union of the two phases the LWP histogram
        # draws, under the caller's thresholds. Imported, not restated.
        ph = phase_masks(fields["lwp_g_m2"], fields["iwp_g_m2"], phase_kw)
        liquid_bearing = np.zeros(shape, dtype=bool)
        for name in PHASE_STACK:
            liquid_bearing |= ph[name]

        pop_masks = {
            "cloud": (cloudy & liquid_bearing).ravel(),
            "all": finite.ravel(),
        }
        n_valid += float(np.count_nonzero(finite))
        n_cloudy += float(np.count_nonzero(cloudy))

        values = np.empty((N_VAR, tcc.size), dtype=np.float64)
        for vi, t in enumerate(TRACKED):
            values[vi] = fields[t.key].ravel() - t.center
        w_flat = np.broadcast_to(weights_2d, shape).ravel().astype(np.float64)
        cls_flat = classes.ravel()
        site_flat = np.broadcast_to(site_mask, shape).ravel()
        ice_flat = siconc_k.ravel()
        is_sea = np.isfinite(ice_flat)

        for pop, pmask in pop_masks.items():
            groups = [(CLASS_CODES[name], pmask & (cls_flat == CLASS_CODES[name]))
                      for name in CLASS_ORDER]
            groups.append((SITE_SLOT, pmask & site_flat))
            groups.append((ALL_SLOT, pmask))
            accumulate_moments(mom[pop], groups, values, w_flat)

            # The MIZ transect. np.digitize returns 0 below the first edge and
            # n_ice+1 above the last, so shifting by one and masking the ends
            # drops out-of-range rather than piling it into an edge bin --
            # siconc is on [0, 1] by definition, so anything outside is a data
            # problem and should not be quietly absorbed.
            ib = np.digitize(ice_flat, siconc_edges) - 1
            sea_pop = pmask & is_sea & (ib >= 0) & (ib < n_ice)
            ice_groups = [(b, sea_pop & (ib == b)) for b in range(n_ice)]
            accumulate_moments(ice_mom[pop], ice_groups, values, w_flat)

        # Everything below is the FILTERED population only: the density
        # scatters and the response curves both describe liquid-bearing
        # overcast, which is the regime the argument is about.
        sel = pop_masks["cloud"]
        if not sel.any():
            continue

        db = np.digitize(fields["lwd_W_m2"].ravel(), dlr_edges) - 1
        in_dlr = sel & (db >= 0) & (db < n_dlr)
        slot_masks = [(CLASS_CODES[name], cls_flat == CLASS_CODES[name])
                      for name in CLASS_ORDER]
        slot_masks.append((SITE_SLOT, site_flat))
        slot_masks.append((ALL_SLOT, np.ones(cls_flat.shape, dtype=bool)))

        for slot, smask in slot_masks:
            m = in_dlr & smask
            if m.any():
                b = db[m]
                ws = w_flat[m]
                curve_w[slot] += np.bincount(b, weights=ws, minlength=n_dlr)
                curve_n[slot] += np.bincount(b, minlength=n_dlr)
                for vi, t in enumerate(TRACKED):
                    curve_x[slot, :, vi] += np.bincount(
                        b, weights=values[vi, m] * ws, minlength=n_dlr)

            m2 = sel & smask
            if not m2.any():
                continue
            ws2 = w_flat[m2]
            for name, p in panels.items():
                xv = values[VAR_INDEX[p.x_key], m2] + CENTERS[VAR_INDEX[p.x_key]]
                yv = values[VAR_INDEX[p.y_key], m2] + CENTERS[VAR_INDEX[p.y_key]]
                xi = np.digitize(xv, np.linspace(*p.x_range, p.x_bins + 1)) - 1
                yi = np.digitize(yv, np.linspace(*p.y_range, p.y_bins + 1)) - 1
                inside = ((xi >= 0) & (xi < p.x_bins)
                          & (yi >= 0) & (yi < p.y_bins))
                hist_out[name][slot] += float(ws2[~inside].sum())
                if inside.any():
                    flat = xi[inside] * p.y_bins + yi[inside]
                    hist[name][slot] += np.bincount(
                        flat, weights=ws2[inside],
                        minlength=p.x_bins * p.y_bins,
                    ).reshape(p.x_bins, p.y_bins)

    # Curve means, with the centring added back. Bins holding too little are
    # NaN so the figure draws a gap instead of a spike.
    with np.errstate(invalid="ignore", divide="ignore"):
        curve_mean = curve_x / curve_w[:, :, None]
    curve_mean += CENTERS[None, None, :]
    curve_mean[curve_n < MIN_CURVE_HOURS] = np.nan

    return {
        "mom": mom,
        "ice_mom": ice_mom,
        "hist": hist,
        "hist_out": hist_out,
        "curve_mean": curve_mean,
        "curve_n": curve_n,
        "curve_w": curve_w,
        "dlr_edges": dlr_edges,
        "siconc_edges": siconc_edges,
        "panels": panels,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "site_class_counts": site_class_counts,
        "n_unclassified": n_unclassified,
        "n_valid": n_valid,
        "n_cloudy": n_cloudy,
        "n_precip_removed": n_precip_removed,
    }


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    parser.add_argument("--phase-mode", choices=PHASE_MODES,
                        default=DEFAULT_PHASE_MODE,
                        help="How 'liquid-bearing' is defined; same meaning as "
                             "in plot_lwp_histogram_by_surface_class.py "
                             f"(default {DEFAULT_PHASE_MODE}).")
    add_phase_args(parser)
    add_fraction_phase_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(10, 1),
                        metavar="MM-DD", help="Season start (default 10-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 03-31, wrapping "
                             "the year).")
    parser.add_argument("--years", type=parse_years, default=None,
                        metavar="SPEC",
                        help="Seasons to pool, by the year each season STARTS "
                             "in: '2022', '2022-2025', or '2019,2022-2025'. "
                             "Default: every season meeting "
                             "--min-season-coverage.")
    parser.add_argument("--min-cloud-fraction", type=float,
                        default=DEFAULT_MIN_CLOUD_FRACTION, metavar="F",
                        help="Total cloud cover at or above which a scene "
                             f"counts as cloudy (default "
                             f"{DEFAULT_MIN_CLOUD_FRACTION:g}, fully overcast).")
    parser.add_argument("--min-season-coverage", type=float, default=0.6,
                        metavar="F",
                        help="Minimum fraction of the season window a season "
                             "must cover to be included automatically "
                             "(default 0.6). Ignored when --years names "
                             "seasons explicitly.")
    parser.add_argument("--no-precip", action="store_true",
                        help="Drop precipitating scenes from the cloudy "
                             "population. Off by default, matching the LWP "
                             "histogram.")
    parser.add_argument("--precip-var", choices=("rate", "path"),
                        default="rate",
                        help="Which field --no-precip tests (default rate: tp).")
    parser.add_argument("--precip-rate-max", type=float, default=0.1,
                        metavar="MM_HR")
    parser.add_argument("--precip-path-max", type=float, default=50.0,
                        metavar="G")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--population", choices=POPULATIONS, default="cloud",
                        help="Which population the sensitivity figures "
                             "describe: 'cloud' is liquid-bearing overcast, "
                             "'all' is every valid hour (default cloud). Both "
                             "are accumulated, so this can be changed on an "
                             "existing Analysis without reloading.")
    parser.add_argument("--fit-mode", choices=FIT_MODES,
                        default=DEFAULT_FIT_MODE,
                        help="What the scatter figures draw over the density: "
                             "'default' is the cos(latitude)-weighted least-"
                             "squares fit plus the mean of y in each x column; "
                             "'regression' is the unweighted least-squares "
                             "regression alone, as a thin solid black line "
                             f"(default {DEFAULT_FIT_MODE}). Both fit the same "
                             "model on the same unbinned sample and differ "
                             "only in the weighting. Slope and r^2 are "
                             "annotated either way. Both moment sets are "
                             "accumulated, so this can be changed on an "
                             "existing Analysis without reloading.")
    parser.add_argument("--control", choices=tuple(CONTROL_SETS),
                        default=DEFAULT_CONTROL,
                        help="Confounders held fixed when the DLR "
                             "sensitivities are computed: 'none' is the plain "
                             "regression, 't2m' holds the 2 m air temperature "
                             "fixed so only the pathway through the surface "
                             "remains, 't2m_wind' also holds 10 m wind speed "
                             f"(default {DEFAULT_CONTROL}). The two bracket "
                             "the causal answer rather than bounding it "
                             "statistically; see CONTROL_SETS.")
    parser.add_argument("--layout", type=int, nargs=2, default=(2, 3),
                        metavar=("ROWS", "COLS"),
                        help="Panel grid for the scatter figures (default 2 3).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-figures", action="store_true",
                        help="Print the tables only.")
    return parser.parse_args(argv)


class Analysis(SimpleNamespace):
    """Everything the figures need, accumulated in one pass by :func:`prepare`.

    Both populations, both sensitivity groupings and all three density
    histograms are filled together, so every figure below is available without
    touching the archive again.
    """

    def acc(self, population: str | None = None) -> dict:
        """The class-slot moment accumulator for a population."""
        return self.sec["mom"][population or self.args.population]

    def ice_acc(self, population: str | None = None) -> dict:
        """The sea-ice-concentration-binned moment accumulator."""
        return self.sec["ice_mom"][population or self.args.population]

    def control(self, control: str | None = None) -> tuple[str, ...]:
        """The tracked variables held fixed by the sensitivity regressions."""
        return CONTROL_SETS[control or self.args.control]


def prepare(argv=None, args=None, **overrides) -> Analysis:
    """Open the archive, classify, and accumulate. The slow step.

    ``argv`` takes the same strings as the command line; ``overrides`` sets
    individual options by name, e.g. ``prepare(region="barrow",
    years=(2022, 2023))``. Anything left out keeps its command-line default.
    """
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    print("=" * 72)
    print("Turbulent flux response to longwave forcing, by surface class")
    print("=" * 72)

    # Validate before opening anything: a bad threshold pair should fail in
    # milliseconds, not after a multi-minute read.
    phase_kw = resolve_phase_thresholds(args)
    panels = dict(DEFAULT_PANELS)

    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm_da = load_land_sea_mask(
        args.region, resolve_data_root(args.storage, args.data_root),
        args.mask_grid,
    )
    lsm = align_lsm_to_grid(lsm_da, ds)

    missing = sorted(set(READ_VARS) - set(ds.data_vars))
    if missing:
        raise KeyError(f"dataset is missing {missing}. Re-download with "
                       f"--var-set recommended or extended")
    # Checked here rather than at the first block: the precipitation fields are
    # only loaded when the filter is on, and finding out they are absent after
    # a multi-minute open is the wrong time to find out.
    if args.no_precip:
        need = {"rate": ("tp",), "path": ("tcrw", "tcsw")}[args.precip_var]
        absent = sorted(set(need) - set(ds.data_vars))
        if absent:
            raise KeyError(
                f"--no-precip --precip-var {args.precip_var} needs {absent}, "
                f"which this archive does not carry. Use --precip-var "
                f"{'path' if args.precip_var == 'rate' else 'rate'} or drop "
                f"--no-precip.")

    print(f"  Source     : {region_dir}")
    print(f"  Grid       : {ds.sizes['latitude']} x {ds.sizes['longitude']} "
          f"cells, {ds.sizes['valid_time']:,} time steps")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d}"
          f" to {args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)"
             if args.season_end < args.season_start else ""))
    print(f"  Classes    : lsm tol {args.lsm_tol:g} | open ocean < "
          f"{args.open_ocean_max_siconc:g} | pack ice > "
          f"{args.sea_ice_min_siconc:g}")
    print(f"  Cloudy     : tcc >= {args.min_cloud_fraction:g}")
    print(f"  Liquid-bearing: {phase_definition_label(phase_kw, mathtext=False)}")
    print(f"  Precip     : {precip_label(args)}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    print(f"\n  Reading {len(used)} season(s): {used}")

    sec = collect(ds, lsm, args, layout, keep_idx, panels,
                  DEFAULT_DLR_CURVE_EDGES, DEFAULT_SICONC_EDGES, phase_kw)
    if sec["n_unclassified"]:
        print(f"  !! {sec['n_unclassified']:,} unclassified cell-times; run "
              f"surface_classification.py for the breakdown.", file=sys.stderr)

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    return Analysis(args=args, ds=ds, lsm=lsm, layout=layout,
                    keep_idx=keep_idx, used=used, mode_label=mode_label,
                    sec=sec, phase_kw=phase_kw, tag=tag)


# ----------------------------------------------------------------------------
# Numeric report
# ----------------------------------------------------------------------------
def print_report(A: Analysis, population: str | None = None) -> None:
    """Sample sizes, the two scatter fits, and the DLR partition per class.

    The numbers the figures draw, in a form that can be pasted into a note. The
    fit lines are the same weighted least-squares fits the scatter panels
    annotate; the partition is the stacked bar of ``fig_response_partition``.
    """
    pop = population or A.args.population
    acc = A.acc(pop)
    sec = A.sec

    print("\n" + "=" * 78)
    print(f"Population: {'liquid-bearing overcast' if pop == 'cloud' else 'all sky'}"
          f"   |   seasons {A.used[0]}-{A.used[-1]}   |   region {A.args.region}")
    print("=" * 78)
    print(f"  valid cell-hours       {sec['n_valid']:>14,.0f}")
    print(f"  overcast cell-hours    {sec['n_cloudy']:>14,.0f}"
          f"  ({100 * sec['n_cloudy'] / max(sec['n_valid'], 1):.1f}% of valid)")
    print(f"  in this population     {acc['n'][ALL_SLOT]:>14,.0f}")
    print(f"  ARM site cell          {sec['site_lat']:.2f} N, "
          f"{sec['site_lon']:.2f} E")
    counts = sec["site_class_counts"]
    named = ", ".join(f"{CLASS_LABELS[n]} {100 * counts[c] / max(counts.sum(), 1):.0f}%"
                      for n, c in CLASS_CODES.items() if counts[c])
    print(f"  site cell class share  {named}")

    # Both weightings, side by side. They are the same estimator on two
    # denominators (see new_moments), and printing only the one the figure
    # happens to be drawing hides how little the choice matters here.
    fmode = A.args.fit_mode
    for x_key, y_key, title in (
        ("shf_W_m2", "lwp_g_m2", "LWP  vs  sensible heat flux"),
        ("shf_W_m2", "lwd_W_m2", "DLR  vs  sensible heat flux"),
        ("shf_W_m2", "dskt_t2m_K", "T_skin - T_2m  vs  sensible heat flux"),
    ):
        yt = TRACKED[VAR_INDEX[y_key]]
        print(f"\n  {title}   (OLS fit of y on x: y = a + b x)")
        print(f"    {'':<20} {'':>12} {'':>9} {'':>9} "
              f"{'--- cos(lat) weighted ---':>26}  {'---- unweighted ----':>22}")
        print(f"    {'class':<20} {'hours':>12} {'x_mean':>9} {'y_mean':>9} "
              f"{'slope':>11} {'r2':>6} {'r':>7}  {'slope':>11} {'r2':>6}")
        for slot in PANEL_SLOTS + (ALL_SLOT,):
            sw = moment_stats(acc, slot, x_key, y_key, weighted=True)
            su = moment_stats(acc, slot, x_key, y_key, weighted=False)
            print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20} "
                  f"{sw['n_hours']:>12,.0f} {sw['x_mean']:>9.2f} "
                  f"{sw['y_mean']:>9.2f} {sw['slope']:>11.4f} "
                  f"{sw['r2']:>6.3f} {sw['r']:>7.3f}  "
                  f"{su['slope']:>11.4f} {su['r2']:>6.3f}")
        print(f"      slope units: {plain_units(yt.units)} per W m-2 | "
              f"figures draw --fit-mode {fmode} "
              f"({'weighted' if FIT_IS_WEIGHTED[fmode] else 'unweighted'})")

    # Both estimators, always. They bracket the causal answer from opposite
    # sides and the gap between them is the size of the air-mass confound --
    # which over open water is the whole story. Printing only one invites the
    # reader to treat it as the answer.
    for cname in dict.fromkeys(("none", "t2m", A.args.control)):
        ctrl = CONTROL_SETS[cname]
        held = ", ".join(ctrl) if ctrl else "nothing (plain regression)"
        print(f"\n  Partition of d(LWD): where each additional W m-2 goes"
              f"   [holding {held} fixed]")
        print(f"    {'class':<20} {'dTskin':>9} {'dT2m':>9} {'d(dT)':>9} "
              f"{'f_LWU':>8} {'f_SH':>8} {'f_LH':>8} {'f_SW':>8} {'f_res':>8}")
        print(f"    {'':<20} {'--- K per W m-2 ---':>29}")
        for slot in PANEL_SLOTS + (ALL_SLOT,):
            pt = partition(acc, slot, control=ctrl)
            print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20} "
                  f"{pt['dskt_dlwd']:>9.4f} {pt['dt2m_dlwd']:>9.4f} "
                  f"{pt['ddskt_dlwd']:>9.4f} {pt['f_lwu']:>8.3f} "
                  f"{pt['f_sh']:>8.3f} {pt['f_lh']:>8.3f} "
                  f"{pt['f_sw']:>8.3f} {pt['f_res']:>8.3f}")
    print("    (the five fractions sum to 1 by construction; f_res is the "
          "remainder)")
    print("    d(dT) is d(T_skin - T_2m)/d(LWD): the driver of the turbulent "
          "response. Where it is")
    print("    negative the air warms faster than the surface, the "
          "skin-to-air difference shrinks,")
    print("    and the sensible heat flux moves energy INTO the surface "
          "(f_SH < 0) rather than out.")

    for name, p in sec["panels"].items():
        out = sec["hist_out"][name]
        tot = np.array([acc["w"][s] for s in range(N_SLOT)])
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = 100.0 * out / tot
        worst = int(np.nanargmax(np.where(np.isfinite(frac), frac, -1)))
        print(f"  {name}: at most {frac[worst]:.2f}% of one panel's weight "
              f"falls outside the drawn range ({SLOT_LABELS[SLOT_ORDER[worst]]}); "
              f"the fits use every sample regardless.")


# ----------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------
# Seawater at a salinity of 34 psu freezes at -1.8 C. It is drawn on the MIZ
# figure because it is the physical reason an open-water skin temperature
# cannot fall: the phase change, not the heat capacity alone, pins it.
SEAWATER_FREEZING_K = 271.35

# A partition coefficient larger than this in magnitude is not a partition: it
# means more energy moved than arrived, which happens where the surface is
# pinned and the regression is describing the air mass instead. Used to set
# readable axis limits, never to hide a bar.
PARTITION_SANE_MAX = 2.5

# Floor of the shared density colour scale, as a fraction of the densest bin in
# a panel. Three decades: below that a bin holds a handful of cell-hours and is
# tail, not structure.
DENSITY_VMIN = 1e-3

DENSITY_CMAP = "viridis"
FIT_COLOR = "#B2182B"
BINNED_MEAN_COLOR = "#FFFFFF"
REGRESSION_COLOR = "#000000"

# ---------------------------------------------------------------------------
# What the scatter panels draw on top of the density
# ---------------------------------------------------------------------------
# BOTH MODES FIT THE SAME MODEL, y = a + b x, BY ORDINARY LEAST SQUARES ON THE
# FULL UNBINNED SAMPLE. They differ in the denominator and in what is drawn
# beside the line:
#
#   "default"     cos(latitude)-weighted OLS, drawn as a red dashed line, with
#                 the mean of y in each x column beside it. The binned mean is
#                 there to show WHERE THE STRAIGHT LINE MISREPRESENTS a
#                 relation that bends, which on a variable as skewed as LWP it
#                 always does somewhere. Weighting by area is the right choice
#                 when the number wanted is a property of the region.
#
#   "regression"  unweighted OLS -- every cell-hour counted once -- drawn alone
#                 as a thin solid black line. This is the conventional
#                 presentation for a scatter of this kind, and it is what a
#                 reader comparing against a point measurement or a published
#                 figure will assume was done.
#
# Neither is a correction of the other. On the Barrow strip they agree closely,
# because cos(latitude) only varies by a factor of two across 70-80 N; where
# they disagree, the disagreement is telling you the relation differs with
# latitude within the class.
FIT_MODES: tuple[str, ...] = ("default", "regression")
DEFAULT_FIT_MODE = "default"

FIT_IS_WEIGHTED: dict[str, bool] = {"default": True, "regression": False}
FIT_LINE_LABELS: dict[str, str] = {
    "default": "weighted linear fit",
    "regression": "linear regression (unweighted)",
}

NOTE_BOX = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.82,
                edgecolor="#999999", linewidth=0.6)


def plain_units(units: str) -> str:
    """Strip mathtext from a units string, for terminal output."""
    return (units.replace("$", "").replace("^{-2}", "-2")
            .replace("^{-1}", "-1").replace("_{", "").replace("}", ""))


def _save(fig, A: Analysis, stem: str, out_dir, dpi: int | None):
    """Write a figure under the naming the other scripts in this directory use."""
    if out_dir is None:
        return fig
    path = Path(out_dir) / f"{A.args.region}_{stem}_{A.tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


TITLE_FS = 13.5
SUB_FS = 8.0
SUB_LINESPACING = 1.55
NOTE_FS = 8.2


def _header_block(fig, title: str, subtitle: str, note: str | None = None,
                  title_fs: float = TITLE_FS) -> float:
    """Lay the title / subtitle / caveat block above the panels.

    Returns the figure-height fraction the panels may occupy, for
    ``subplots_adjust(top=)`` or ``tight_layout(rect=)``.

    THE SUBTITLE'S LINE COUNT IS NOT FIXED. It gains a line when the population
    is filtered (the phase definition), another when a fit is described, and
    loses them for an all-sky figure -- so any hard-coded y overlaps the panel
    titles on some runs and strands white space on others. Measuring the block
    instead removes a whole class of layout bug that only shows up once an
    option is changed.
    """
    fig_h = fig.get_figheight()

    def _h(pts: float, n: int, spacing: float) -> float:
        return n * pts * spacing / 72.0 / fig_h

    title_h = _h(title_fs, 1, 1.6)
    sub_h = _h(SUB_FS, subtitle.count("\n") + 1, SUB_LINESPACING)
    note_h = _h(NOTE_FS, note.count("\n") + 1, 1.5) if note else 0.0

    y = 1.0 - 0.14 / fig_h
    fig.suptitle(title, y=y, va="top", fontsize=title_fs)
    y -= title_h + 0.10 / fig_h
    fig.text(0.5, y, subtitle, ha="center", va="top", fontsize=SUB_FS,
             color="#444444", linespacing=SUB_LINESPACING)
    y -= sub_h
    if note:
        y -= 0.05 / fig_h
        fig.text(0.5, y, note, ha="center", va="top", fontsize=NOTE_FS,
                 color="#8a5a00", style="italic")
        y -= note_h
    return max(y - 0.12 / fig_h, 0.55)


def _bin_centers(lo: float, hi: float, n: int) -> np.ndarray:
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _binned_mean_y(h: np.ndarray, panel: Panel2D) -> tuple[np.ndarray, np.ndarray]:
    """Mean y in each x column of a 2-D histogram, and the x centres.

    Drawn alongside the straight-line fit because the two disagree wherever the
    relation is not linear, and on a strongly skewed y such as LWP they always
    do. Columns holding under 0.5% of the panel's weight are dropped: the tails
    of a density plot are exactly where a column mean is one or two hours.
    """
    xc = _bin_centers(panel.x_range[0], panel.x_range[1], panel.x_bins)
    yc = _bin_centers(panel.y_range[0], panel.y_range[1], panel.y_bins)
    col_w = h.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ymean = (h @ yc) / col_w
    ymean[col_w < 0.005 * col_w.sum()] = np.nan
    return xc, ymean


def _density_panel(ax, h: np.ndarray, panel: Panel2D, stats: dict,
                   label: str, color: str, out_weight: float,
                   fit_mode: str = DEFAULT_FIT_MODE,
                   marker_size: float = 3.0):
    """One density-coloured scatter panel with its weighted linear fit.

    Colour is the area-weighted frequency of each bin, normalised to the
    DENSEST BIN OF THIS PANEL. Normalising panel by panel rather than across
    the figure is deliberate: the classes differ in size by three orders of
    magnitude -- the ARM site is one grid cell, sea ice is thousands -- and a
    shared scale would render four of the six panels as a single flat colour.
    The cost is that colour is not comparable between panels, so the cell-hour
    count is printed on each.
    """
    from matplotlib.colors import LogNorm

    xc = _bin_centers(panel.x_range[0], panel.x_range[1], panel.x_bins)
    yc = _bin_centers(panel.y_range[0], panel.y_range[1], panel.y_bins)
    occupied = h > 0
    sm = None
    # ONE norm for every panel, spanning three decades of relative frequency.
    # A per-panel norm would make the shared colourbar describe only whichever
    # panel happened to be drawn last, which is worse than no colourbar.
    norm = LogNorm(vmin=DENSITY_VMIN, vmax=1.0)
    if occupied.any():
        xi, yi = np.nonzero(occupied)
        frac = np.clip(h[xi, yi] / h.max(), DENSITY_VMIN, 1.0)
        order = np.argsort(frac)             # densest markers drawn on top
        sm = ax.scatter(xc[xi][order], yc[yi][order], c=frac[order],
                        s=marker_size, cmap=DENSITY_CMAP, linewidths=0,
                        norm=norm)

        if fit_mode == "default":
            xb, ymean = _binned_mean_y(h, panel)
            ax.plot(xb, ymean, color=BINNED_MEAN_COLOR, lw=2.4, alpha=0.85,
                    solid_capstyle="round", zorder=4)
            ax.plot(xb, ymean, color="#222222", lw=1.1, zorder=5,
                    label="mean y per x bin")

    if np.isfinite(stats["slope"]):
        # Drawn only across the x range the data actually occupy. Extending a
        # fit line over an axis the class never visits -- Utqiagvik spans about
        # 30 W m-2 of an axis 310 wide -- makes a slope fitted to a sliver look
        # like a claim about the whole panel.
        if occupied.any():
            lo, hi = xc[xi.min()], xc[xi.max()]
        else:
            lo, hi = panel.x_range
        xs = np.array([lo, hi])
        ys = stats["intercept"] + stats["slope"] * xs
        if fit_mode == "regression":
            # Thin, solid, black, and alone: the conventional rendering, and
            # with the binned mean removed there is nothing else on the panel
            # for it to be confused with.
            ax.plot(xs, ys, color=REGRESSION_COLOR, lw=1.2, ls="-", zorder=6,
                    label=FIT_LINE_LABELS[fit_mode])
        else:
            ax.plot(xs, ys, color=FIT_COLOR, lw=1.8, ls="--", zorder=6,
                    label=FIT_LINE_LABELS[fit_mode])

    ax.set_xlim(*panel.x_range)
    ax.set_ylim(*panel.y_range)
    ax.axvline(0.0, color="#444444", lw=0.7, ls=":", zorder=1)
    ax.set_title(label, fontsize=10.5, color=color, fontweight="bold", pad=4)

    y_units = plain_units(TRACKED[VAR_INDEX[panel.y_key]].units)
    note = (f"slope = {stats['slope']:+.3g} {y_units} / (W m-2)\n"
            f"$r^2$ = {stats['r2']:.3f}   (r = {stats['r']:+.3f})\n"
            f"{stats['n_hours']:,.0f} cell-hours")
    if out_weight > 0.005:
        note += f"\n{100 * out_weight:.1f}% outside axes"
    ax.text(0.03, 0.965, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.4, bbox=NOTE_BOX, zorder=7)
    ax.grid(alpha=0.18, lw=0.5)
    return sm


def _figure_subtitle(A: Analysis, pop: str) -> str:
    """The lines that say what population the figure is describing.

    The phase definition is spelled out rather than named, because "liquid-
    bearing" means two quite different populations depending on
    ``--phase-mode`` and the difference is not small: under the fraction scheme
    it is every cloud whose liquid share clears 1 - ice_fraction_min, under the
    absolute scheme it is every cloud clearing a g m-2 floor. A figure that
    only said "liquid-bearing" would be ambiguous between them.
    """
    a = A.args
    window = (f"{a.season_start[0]:02d}-{a.season_start[1]:02d} to "
              f"{a.season_end[0]:02d}-{a.season_end[1]:02d}")
    if pop == "cloud":
        what = (f"overcast (tcc $\\geq$ {a.min_cloud_fraction:g}) and "
                f"liquid-bearing")
    else:
        what = "all sky"
    lines = [f"{a.region} | {window} | seasons {A.used[0]}/{A.used[0] + 1}-"
             f"{A.used[-1]}/{A.used[-1] + 1} | {what}"]
    if pop == "cloud":
        lines.append("liquid-bearing = liquid only + mixed phase, with "
                     + phase_definition_label(A.phase_kw, mathtext=True))
    lines.append("ERA5, all fluxes positive DOWNWARD: SHF $>$ 0 is heat "
                 "entering the surface, SHF $<$ 0 is heat leaving it")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Figures 1-3: the density scatters
# ----------------------------------------------------------------------------
def _scatter_figure(A: Analysis, panel_name: str, title: str, stem: str,
                    out_dir=None, dpi: int | None = None,
                    population: str | None = None,
                    fit_mode: str | None = None):
    """Six-panel density scatter of one y against sensible heat flux.

    Panels are the five surface classes then the ARM site cell. The site is not
    a sixth class -- it is inside whichever class it falls in that hour -- so it
    is drawn last and separated by its colour rather than mixed into the row.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    if pop != "cloud":
        raise ValueError(
            "the density histograms are accumulated for the liquid-bearing "
            "overcast population only; population='all' is available for the "
            "sensitivity figures, which read the moment matrix instead.")

    fmode = fit_mode or A.args.fit_mode
    if fmode not in FIT_MODES:
        raise ValueError(f"unknown fit_mode {fmode!r}; choose from "
                         f"{list(FIT_MODES)}")
    weighted = FIT_IS_WEIGHTED[fmode]
    acc = A.acc("cloud")
    panel = A.sec["panels"][panel_name]
    h_all = A.sec["hist"][panel_name]
    n_r, n_c = A.args.layout

    fig, axes = plt.subplots(n_r, n_c, figsize=(4.1 * n_c, 3.6 * n_r),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    sm = None
    for ax, slot in zip(axes, PANEL_SLOTS):
        name = SLOT_ORDER[slot]
        stats = moment_stats(acc, slot, panel.x_key, panel.y_key,
                             weighted=weighted)
        w_tot = acc["w"][slot]
        out_frac = (A.sec["hist_out"][panel_name][slot] / w_tot
                    if w_tot > 0 else 0.0)
        s = _density_panel(ax, h_all[slot], panel, stats,
                           SLOT_LABELS[name], SLOT_COLORS[name], out_frac,
                           fit_mode=fmode)
        sm = s if s is not None else sm
    for ax in axes[len(PANEL_SLOTS):]:
        ax.set_visible(False)

    x_t = TRACKED[VAR_INDEX[panel.x_key]]
    y_t = TRACKED[VAR_INDEX[panel.y_key]]
    x_label = f"{x_t.label} [{x_t.units}]"
    for ax in axes[:len(PANEL_SLOTS)]:
        if ax.get_subplotspec().is_last_row() or n_r * n_c > len(PANEL_SLOTS):
            ax.set_xlabel(x_label)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"{y_t.label} [{y_t.units}]")
    # With six panels on a 2x3 grid every column has a bottom axis, but a
    # layout that leaves a hole would otherwise strand an unlabelled x axis.
    axes[len(PANEL_SLOTS) - 1].set_xlabel(x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, -0.035))

    subtitle = (_figure_subtitle(A, pop)
                + f"\nfit: {FIT_LINE_LABELS[fmode]} of y on x, ordinary least "
                  f"squares on the full unbinned sample")
    # Header first, so the colourbar below sizes itself against the panel
    # rectangle the block actually leaves.
    fig.subplots_adjust(top=_header_block(fig, title, subtitle))
    if sm is not None:
        cb = fig.colorbar(sm, ax=axes[:len(PANEL_SLOTS)].tolist(),
                          fraction=0.022, pad=0.015)
        cb.set_label("area-weighted frequency,\nrelative to the densest bin "
                     "in the panel", fontsize=8.5)
    return _save(fig, A, f"{stem}_{fmode}", out_dir, dpi)


def fig_shf_vs_lwp(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None):
    """Liquid water path against sensible heat flux, by surface class.

    The first of the two the request asks for. Read it as a joint distribution,
    not a causal chain: LWP does not drive SHF directly, it drives DLR, which
    drives T_skin, which drives SHF. The straight-line fit is therefore a
    summary of a two-step relation and is not expected to be tight -- what it
    is good for is the SIGN and the contrast between classes.
    """
    return _scatter_figure(
        A, "shf_lwp",
        "Liquid water path against surface sensible heat flux",
        "shf_vs_lwp", out_dir, dpi, fit_mode=fit_mode)


def fig_shf_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None):
    """Downwelling longwave against sensible heat flux, by surface class.

    The same population as ``fig_shf_vs_lwp`` with the intermediate variable
    substituted in: DLR is what the cloud actually delivers to the surface, so
    this panel is one step closer to the mechanism and the relation is
    correspondingly tighter.
    """
    return _scatter_figure(
        A, "shf_dlr",
        "Downwelling longwave against surface sensible heat flux",
        "shf_vs_dlr", out_dir, dpi, fit_mode=fit_mode)


def fig_shf_vs_dskt(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None):
    """(T_skin - T_2m) against sensible heat flux: the bulk relation itself.

    Not requested, but it is the premise the other two figures rest on, and it
    costs nothing to draw from the same pass. If ERA5's sensible heat flux is
    the bulk flux the argument assumes, this panel is a line through the origin
    whose slope is rho*c_p*C_H*U -- and the scatter about it is the wind-speed
    and stability dependence that the LWP and DLR panels inherit.
    """
    return _scatter_figure(
        A, "shf_dskt",
        r"Skin-to-air temperature difference against sensible heat flux",
        "shf_vs_dskt", out_dir, dpi, fit_mode=fit_mode)


# ----------------------------------------------------------------------------
# Figure 4: the partition, by surface class
# ----------------------------------------------------------------------------
def fig_response_partition(A: Analysis, out_dir=None, dpi: int | None = None,
                           population: str | None = None,
                           control: str | None = None):
    """How a change in DLR partitions, as a function of surface type.

    Four panels:

    (a) d(T_skin)/d(LWD), the surface's thermal freedom. This is the single
        number the rest of the figure follows from: a surface that cannot warm
        cannot radiate more, and cannot change its skin-to-air difference, so
        it cannot return the energy through either channel.
    (b) The five-way partition of each additional W m-2 of DLR. Sums to one by
        construction (see the module docstring), so the bar is a genuine
        budget and the residual is what the other four leave.
    (c), (d) The underlying binned relations, drawn so the partition slopes can
        be checked against the curves they came from rather than trusted. A
        slope taken across a curve that bends is a summary, not a coefficient.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    cname = control or A.args.control
    ctrl = CONTROL_SETS[cname]
    acc = A.acc(pop)
    slots = list(PANEL_SLOTS)
    names = [SLOT_ORDER[s] for s in slots]
    labels = [SLOT_LABELS[n] for n in names]
    colors = [SLOT_COLORS[n] for n in names]
    parts = [partition(acc, s, control=ctrl) for s in slots]
    # The plain regression is drawn alongside as an open marker, so the size of
    # the air-mass confound is visible on the figure instead of living in a
    # caveat the reader has to remember.
    alt_name = ALT_CONTROL[cname]
    alt = [partition(acc, s, control=CONTROL_SETS[alt_name]) for s in slots]
    x = np.arange(len(slots))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) thermal freedom -----------------------------------------------------
    vals = np.array([p["dskt_dlwd"] for p in parts])
    raw = np.array([p["dskt_dlwd"] for p in alt])
    # The second series is what actually sets the SIGN of the turbulent
    # response. Plotting only d(T_skin)/d(LWD) invites the reading that a
    # surface free to warm must therefore give energy back through the
    # turbulent term, and on this record that inference is wrong: the air warms
    # faster than the surface almost everywhere, so the skin-to-air difference
    # shrinks and the flux reverses.
    drive = np.array([p["ddskt_dlwd"] for p in parts])
    bw = 0.38
    ax_a.bar(x - bw / 2, vals, bw, color=colors, edgecolor="#333333",
             linewidth=0.6, label=r"$dT_{skin}/dLWD$  (thermal freedom)")
    ax_a.bar(x + bw / 2, drive, bw, color=colors, edgecolor="#333333",
             linewidth=0.6, hatch="///", alpha=0.85,
             label=r"$d(T_{skin}-T_{2m})/dLWD$  (drives the SH response)")
    ax_a.scatter(x - bw / 2, raw, marker="D", s=30, facecolor="white",
                 edgecolor="#333333", linewidth=1.1, zorder=5,
                 label=CONTROL_LABELS[alt_name] + r", $T_{skin}$ only")
    for xi, v in zip(x, vals):
        ax_a.annotate(f"{v:.3f}", (xi - bw / 2, v), textcoords="offset points",
                      xytext=(0, 3 if v >= 0 else -12), ha="center",
                      fontsize=8)
    ax_a.set_ylabel("K per W m$^{-2}$ of DLR")
    ax_a.set_title("(a)  How far the surface, and the skin-to-air "
                   "difference, move", fontsize=11, loc="left",
                   fontweight="bold")
    ax_a.axhline(0.0, color="#333333", lw=0.9)
    both = np.concatenate([vals, raw, drive])
    lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
    rng = hi - lo if hi > lo else max(abs(hi), 1e-3)
    # Extra headroom at the top so the three-entry legend clears the tallest
    # bar and its value label instead of sitting on top of them.
    ax_a.set_ylim(min(lo - 0.15 * rng, -1e-3), hi + 0.6 * rng)
    ax_a.legend(fontsize=7.6, frameon=False, loc="upper left", ncol=1)

    # (b) the partition -------------------------------------------------------
    bottom = np.zeros(len(slots))
    for key, term_label, color in PARTITION_TERMS:
        v = np.array([p[key] for p in parts])
        ax_b.bar(x, v, bottom=bottom, color=color, edgecolor="white",
                 linewidth=0.7, label=term_label)
        for xi, vi, bi in zip(x, v, bottom):
            if abs(vi) > 0.12:
                ax_b.annotate(f"{vi:.2f}", (xi, bi + vi / 2), ha="center",
                              va="center", fontsize=8, color="white"
                              if color != "#BBBBBB" else "#333333")
        bottom = bottom + v
    # One breakdown case must not destroy the other five panels' readability.
    # Over open water the surface cannot respond at all, so every fraction is
    # the air mass instead and the bar runs to several times its own total; it
    # is clipped and labelled rather than dropped, because "this estimator
    # fails here" is itself the result for that class.
    stack = np.array([[p[k] for k, _, _ in PARTITION_TERMS] for p in parts])
    tops = np.maximum.accumulate(np.cumsum(stack, axis=1), axis=1)
    bots = np.minimum.accumulate(np.cumsum(stack, axis=1), axis=1)
    sane = np.abs(stack).max(axis=1) <= PARTITION_SANE_MAX
    if sane.any():
        y_hi = min(float(np.nanmax(tops[sane])) + 0.15, PARTITION_SANE_MAX)
        y_lo = max(float(np.nanmin(np.minimum(bots[sane], 0.0))) - 0.15,
                   -PARTITION_SANE_MAX)
        ax_b.set_ylim(min(y_lo, -0.15), max(y_hi, 1.15))
    for xi in np.flatnonzero(~sane):
        ax_b.annotate("off scale:\nsurface\npinned", (xi, 0.42),
                      xycoords=("data", "axes fraction"), ha="center",
                      va="center", fontsize=7.6, color="#8a5a00",
                      fontweight="bold", bbox=NOTE_BOX, zorder=8)
    ax_b.axhline(1.0, color="#333333", lw=1.0, ls="--")
    ax_b.axhline(0.0, color="#333333", lw=0.8)
    ax_b.set_ylabel("fraction of $d(LWD)$\n"
                    "(dashed line = the whole 1 W m$^{-2}$)")
    ax_b.set_title("(b)  Where each additional W m$^{-2}$ of DLR goes",
                   fontsize=11, loc="left", fontweight="bold")
    ax_b.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.14))

    for ax in (ax_a, ax_b):
        ax.set_xticks(x)
        ax.set_xticklabels([lab.replace(" (", "\n(").replace(" zone", "\nzone")
                            .replace("Utqiagvik ", "Utqiagvik\n")
                            for lab in labels], fontsize=8.5)
        ax.grid(axis="y", alpha=0.2, lw=0.5)

    # (c), (d) the binned curves ---------------------------------------------
    edges = A.sec["dlr_edges"]
    dlr_c = 0.5 * (edges[:-1] + edges[1:])
    curve = A.sec["curve_mean"]
    for slot, name, color in zip(slots, names, colors):
        ax_c.plot(dlr_c, curve[slot, :, VAR_INDEX["skt_K"]], marker="o", ms=3.5,
                  lw=1.6, color=color, label=SLOT_LABELS[name])
        ax_d.plot(dlr_c, curve[slot, :, VAR_INDEX["shf_W_m2"]], marker="o",
                  ms=3.5, lw=1.6, color=color, label=SLOT_LABELS[name])

    ax_c.axhline(SEAWATER_FREEZING_K, color="#1f5fa8", lw=1.0, ls=":")
    ax_c.annotate("seawater freezing, $-1.8$ $\\degree$C",
                  (edges[0], SEAWATER_FREEZING_K), textcoords="offset points",
                  xytext=(4, 4), fontsize=8, color="#1f5fa8")
    ax_c.set_ylabel(r"mean $T_{skin}$   [K]")
    ax_c.set_title("(c)  Skin temperature against DLR", fontsize=11,
                   loc="left", fontweight="bold")

    ax_d.axhline(0.0, color="#333333", lw=0.8)
    ax_d.set_ylabel("mean sensible heat flux   [W m$^{-2}$]")
    ax_d.set_title("(d)  Sensible heat flux against DLR", fontsize=11,
                   loc="left", fontweight="bold")

    for ax in (ax_c, ax_d):
        ax.set_xlabel("downwelling longwave, DLR   [W m$^{-2}$]")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(fontsize=8, frameon=False, ncol=2)

    top = _header_block(
        fig, "Partition of a longwave forcing by surface type",
        _figure_subtitle(A, pop),
        note=(f"bars: {CONTROL_LABELS[cname]}; markers in (a): "
              f"{CONTROL_LABELS[alt_name]}. These are covariances across "
              "synoptic variability, not a controlled perturbation, and the "
              "two estimators bracket the causal answer - see the module "
              "docstring"),
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, A, f"dlr_partition_{pop}_{cname}", out_dir, dpi)


# ----------------------------------------------------------------------------
# Figure 5: the marginal ice zone as a natural experiment
# ----------------------------------------------------------------------------
def fig_miz_transect(A: Analysis, out_dir=None, dpi: int | None = None,
                     population: str | None = None,
                     control: str | None = None):
    """The surface response across the ice edge, with siconc as the coordinate.

    The three discrete ocean classes cut a continuum into three pieces and then
    report a mean for each. That is the wrong shape for the question: the whole
    point of the marginal ice zone is that the surface changes CONTINUOUSLY
    across it, over a distance short enough that the air mass above does not.
    So here sea ice concentration replaces the class label, the population is
    sea cells only, and the panels are read left to right as a transect from
    open water to pack ice.

    (a) sets up the experiment: what the surface does across the edge, and how
        little the forcing does. (b) is the skin temperature's freedom to
        respond, the quantity that changes most. (c) is the resulting partition
        -- the same five fractions as ``fig_response_partition``, drawn as
        curves. (d) is the turbulent flux itself, which changes sign somewhere
        in the band.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    cname = control or A.args.control
    ctrl = CONTROL_SETS[cname]
    acc = A.ice_acc(pop)
    edges = A.sec["siconc_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bin = len(centers)

    ok = acc["n"] >= MIN_MIZ_HOURS
    if not ok.any():
        raise ValueError("no sea ice concentration bin holds enough cell-hours; "
                         "widen the season window or lower MIN_MIZ_HOURS.")

    def series(fn) -> np.ndarray:
        out = np.full(n_bin, np.nan)
        for b in range(n_bin):
            if ok[b]:
                out[b] = fn(b)
        return out

    skt = series(lambda b: mean_of(acc, b, "skt_K"))
    t2m = series(lambda b: mean_of(acc, b, "t2m_K"))
    lwd = series(lambda b: mean_of(acc, b, "lwd_W_m2"))
    shf = series(lambda b: mean_of(acc, b, "shf_W_m2"))
    dskt = series(lambda b: mean_of(acc, b, "dskt_t2m_K"))
    dskt_dlwd = series(lambda b: partition(acc, b, ctrl)["dskt_dlwd"])
    alt_name = ALT_CONTROL[cname]
    alt_ctrl = CONTROL_SETS[alt_name]
    dskt_alt = series(lambda b: partition(acc, b, alt_ctrl)["dskt_dlwd"])
    ddskt_dlwd = series(lambda b: partition(acc, b, ctrl)["ddskt_dlwd"])
    frac = {key: series(lambda b, k=key: partition(acc, b, ctrl)[k])
            for key, _, _ in PARTITION_TERMS}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    miz_lo = A.args.open_ocean_max_siconc
    miz_hi = A.args.sea_ice_min_siconc
    for ax in axes.ravel():
        ax.axvspan(miz_lo, miz_hi, color="#4eb3d3", alpha=0.10, zorder=0)
        ax.grid(alpha=0.2, lw=0.5)

    # (a) the experiment: surface varies, forcing does not --------------------
    ax_a.plot(centers, skt, marker="o", ms=4, lw=1.8, color="#B2182B",
              label=r"$T_{skin}$")
    ax_a.plot(centers, t2m, marker="s", ms=3.5, lw=1.4, ls="--",
              color="#8a5a00", label=r"$T_{2m}$")
    ax_a.axhline(SEAWATER_FREEZING_K, color="#1f5fa8", lw=1.0, ls=":")
    ax_a.annotate("seawater freezing", (1.0, SEAWATER_FREEZING_K),
                  textcoords="offset points", xytext=(-4, 4), ha="right",
                  fontsize=8, color="#1f5fa8")
    ax_a.set_ylabel("temperature   [K]")
    ax_a.legend(fontsize=9, frameon=False, loc="lower left")
    ax_a2 = ax_a.twinx()
    ax_a2.plot(centers, lwd, marker="^", ms=3.5, lw=1.5, color="#4C72B0",
               label="DLR")
    ax_a2.set_ylabel("mean DLR   [W m$^{-2}$]", color="#4C72B0")
    ax_a2.tick_params(axis="y", colors="#4C72B0")
    ax_a2.legend(fontsize=9, frameon=False, loc="upper right")
    ax_a2.grid(False)
    span_t = np.nanmax(skt) - np.nanmin(skt)
    span_f = np.nanmax(lwd) - np.nanmin(lwd)
    ax_a.set_title("(a)  The natural experiment: the surface varies, "
                   "the forcing much less",
                   fontsize=10.5, loc="left", fontweight="bold")
    ax_a.text(0.5, 0.06,
              f"across the edge: $T_{{skin}}$ spans {span_t:.1f} K,\n"
              f"mean DLR spans only {span_f:.0f} W m$^{{-2}}$",
              transform=ax_a.transAxes, ha="center", va="bottom",
              fontsize=8.5, bbox=NOTE_BOX, zorder=6)

    # (b) thermal freedom -----------------------------------------------------
    ax_b.plot(centers, dskt_dlwd, marker="o", ms=4, lw=1.9, color="#B2182B",
              label=r"$dT_{skin}/dLWD$  (" + CONTROL_LABELS[cname] + ")")
    ax_b.plot(centers, dskt_alt, marker="D", ms=3.2, lw=1.2, ls="--",
              color="#8a8a8a",
              label=r"$dT_{skin}/dLWD$  (" + CONTROL_LABELS[alt_name] + ")")
    # The skin temperature's freedom is only half the story: the turbulent
    # response follows the skin-to-air DIFFERENCE, and this curve is where its
    # sign is decided.
    ax_b.plot(centers, ddskt_dlwd, marker="^", ms=3.6, lw=1.7, color="#4C72B0",
              label=r"$d(T_{skin}-T_{2m})/dLWD$  (drives SH)")
    ax_b.axhline(0.0, color="#333333", lw=0.9)
    ax_b.set_ylabel("K per W m$^{-2}$ of DLR")
    ax_b.legend(fontsize=7.8, frameon=False, loc="upper left")
    ax_b.set_title("(b)  The surface response gradient across the ice edge",
                   fontsize=10.5, loc="left", fontweight="bold")

    # (c) the partition -------------------------------------------------------
    for key, term_label, color in PARTITION_TERMS:
        ax_c.plot(centers, frac[key], marker="o", ms=3.2, lw=1.7, color=color,
                  label=term_label)
    ax_c.axhline(0.0, color="#333333", lw=0.8)
    ax_c.axhline(1.0, color="#333333", lw=0.8, ls="--")
    stack = np.array([frac[k] for k, _, _ in PARTITION_TERMS])
    inside = np.abs(stack) <= PARTITION_SANE_MAX
    if inside.any():
        ax_c.set_ylim(min(float(np.nanmin(stack[inside])) - 0.15, -0.15),
                      max(float(np.nanmax(stack[inside])) + 0.15, 1.15))
    ax_c.set_ylabel("fraction of $d(LWD)$")
    ax_c.set_title("(c)  Partition of each additional W m$^{-2}$ of DLR",
                   fontsize=10.5, loc="left", fontweight="bold")
    ax_c.legend(fontsize=7.8, frameon=False, ncol=1, loc="upper left")
    # The left end of this panel is where the estimator stops meaning what its
    # axis label says, and the figure should say so rather than leave a reader
    # to wonder why a partition coefficient is -1.7.
    ax_c.text(0.98, 0.03,
              "open water: the surface is pinned near freezing and\n"
              "cannot respond, so these coefficients describe the air\n"
              "mass co-varying with DLR, not a partition of it",
              transform=ax_c.transAxes, fontsize=7.4, color="#8a5a00",
              va="bottom", ha="right", bbox=NOTE_BOX, zorder=6)

    # (d) the turbulent flux itself ------------------------------------------
    ax_d.plot(centers, shf, marker="o", ms=4, lw=1.9, color="#4C72B0",
              label="sensible heat flux")
    ax_d.axhline(0.0, color="#333333", lw=0.9)
    ax_d.set_ylabel("mean sensible heat flux   [W m$^{-2}$]",
                    color="#4C72B0")
    ax_d.tick_params(axis="y", colors="#4C72B0")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(centers, dskt, marker="s", ms=3.5, lw=1.4, ls="--",
               color="#B2182B", label=r"$T_{skin}-T_{2m}$")
    ax_d2.axhline(0.0, color="#B2182B", lw=0.7, ls=":")
    ax_d2.set_ylabel(r"$T_{skin}-T_{2m}$   [K]", color="#B2182B")
    ax_d2.tick_params(axis="y", colors="#B2182B")
    ax_d2.grid(False)
    ax_d.set_title("(d)  The flux and its driver change sign together",
                   fontsize=10.5, loc="left", fontweight="bold")

    for ax in (ax_c, ax_d):
        ax.set_xlabel("sea ice concentration   [0 = open water, 1 = pack ice]")
    ax_c.set_xlim(0.0, 1.0)
    ax_c.annotate(f"marginal ice zone  ({miz_lo:g} - {miz_hi:g})",
                  (0.5 * (miz_lo + miz_hi), 0.965),
                  xycoords=("data", "axes fraction"), ha="center", va="top",
                  fontsize=8.5, color="#1f5fa8")

    top = _header_block(
        fig, "The marginal ice zone as a natural experiment: "
             "one air mass, every surface",
        _figure_subtitle(A, pop) + " | sea cells only",
        note=(f"sensitivities in (b) and (c): {CONTROL_LABELS[cname]}; the "
              f"grey curve in (b) is {CONTROL_LABELS[alt_name]}, the other end "
              "of the bracket"),
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, A, f"miz_transect_{pop}_{cname}", out_dir, dpi)


ALL_FIGURES = (
    fig_shf_vs_lwp,
    fig_shf_vs_dlr,
    fig_shf_vs_dskt,
    fig_response_partition,
    fig_miz_transect,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    A = prepare(args=args)
    print_report(A)
    if args.no_figures:
        return 0
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    for fn in ALL_FIGURES:
        fn(A, out_dir=args.output_dir, dpi=args.dpi)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
