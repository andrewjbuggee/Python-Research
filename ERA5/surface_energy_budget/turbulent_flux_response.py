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
to move is a property of the surface -- and over open water in ERA5 it has
almost none, for a reason that is as much about the model as about the ocean.
See "THE OPEN-OCEAN SURFACE DOES NOT RESPOND, AND WHY" below.

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

THE OPEN-OCEAN SURFACE DOES NOT RESPOND, AND WHY
------------------------------------------------
Every figure here shows open water behaving unlike the other classes: the skin
temperature barely moves with DLR, the partition does not close, and the
turbulent coefficients exceed one. The reason is worth stating precisely,
because the obvious physical explanation is NOT the one operating.

THE OBVIOUS EXPLANATION, WHICH IS RIGHT PHYSICS BUT THE WRONG DIAGNOSIS. Water
at the freezing point cannot cool further: extract heat and it forms ice
instead, and the latent heat of fusion, 334 kJ per kg, is an enormous buffer.
An ocean surface sitting at its freezing point genuinely is pinned.

BUT THE ERA5 OPEN OCEAN HERE IS MOSTLY NOT AT ITS FREEZING POINT. MEASURED,
Barrow strip, Oct-Nov 2024, cells with siconc < 0.05:

    mean skin temperature                                 273.35 K  (+0.2 C)
    range                                        262.40 to 279.76 K
    within 0.1 K of the seawater freezing point            0.2% of cell-hours
    within 1.0 K                                          17.8% of cell-hours

so for most hours the phase-change floor is one to two kelvin away and is not
what is holding the surface still.

WHAT IS HOLDING IT STILL IS THE MODEL. MEASURED, same window, the standard
deviation of the HOUR-TO-HOUR CHANGE in skin temperature:

    open ocean       0.053 K
    marginal ice     0.283 K
    sea ice          0.303 K
    land             0.609 K

The open-ocean surface moves roughly six times less per hour than pack ice and
eleven times less than land. That is the signature of a field that is imposed
rather than computed: ERA5 prescribes sea surface temperature and sea ice
concentration from an external daily analysis and interpolates them in time, so
over open water the skin temperature is essentially a boundary condition, with
only a small cool-skin correction responding to the fluxes. Over land and ice
ERA5 solves a surface energy balance for a skin layer of zero heat capacity, so
there the skin temperature IS a prognostic response.

CONSEQUENCE FOR EVERY FIGURE BELOW. The open-ocean column is measuring
something structurally different from the land and ice columns. Over ice and
land, d(T_skin)/d(DLR) is a genuine model response to the radiation. Over open
water it is the residual correlation between a daily external SST field and an
hourly DLR field, and the flux sensitivities there describe the ATMOSPHERE
moving over an effectively fixed surface. That is why those bars are labelled
and excluded from the axis scaling rather than read as a partition.

For the real ocean the reader's instinct is the right one -- phase change would
buffer the temperature once the freezing point is reached -- but ERA5 does not
express that mechanism at hourly resolution, and in this window the water is
mostly still above the freezing point in any case. Confidence: the measurements
above are direct; the attribution to prescribed SST follows from ERA5's
documented design rather than from anything tested here.

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
    available_regions,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
)
from convert_specific_to_absolute import G_M_S2, layer_thickness_pa
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
    strip_frequency_suffix,
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
    lowest_drawn_lwp,
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
    # --- the latent-heat side, and the multiplicative bulk predictors ------
    Tracked("q2m_g_kg", "2 m specific humidity", "g kg$^{-1}$", 1.0),
    Tracked("dq_g_kg", r"$q_{sat}(T_{skin}) - q_{2m}$", "g kg$^{-1}$", 0.0),
    # The bulk formulae are MULTIPLICATIVE in wind speed: a 5 K skin-to-air
    # difference moves twice the heat at 10 m/s that it does at 5 m/s. Adding
    # wind as an extra ADDITIVE term in a multiple regression only
    # approximates that. Carrying the products lets the regression be run on
    # the predictor the physics actually names, whose slope is rho*c_p*C_H
    # directly -- no wind speed left in the units.
    Tracked("u_dskt_K_m_s", r"$U\,(T_{skin}-T_{2m})$", "K m s$^{-1}$", 0.0),
    Tracked("u_dq_g_kg_m_s", r"$U\,\Delta q$", "g kg$^{-1}$ m s$^{-1}$", 0.0),
    # Surface pressure as a circulation proxy, following the way Bertrand et
    # al. (2025) stratify on daily surface pressure anomaly to test whether
    # synoptic variability explains their result.
    Tracked("sp_hPa", "Surface pressure", "hPa", 1010.0),
    # The net surface flux, R = LWD - LWU + SW_net + SH + LH, every term
    # positive downward. Tracked rather than assembled from the other five
    # afterwards for two reasons: a density scatter needs the per-sample
    # value, which a covariance matrix cannot supply; and once it is a column
    # of its own, regressing it on DLR is an INDEPENDENT route to f_res, which
    # the partition otherwise defines as a remainder.
    Tracked("rnet_W_m2", r"Net surface flux $R$", "W m$^{-2}$", 0.0),
)

# ---------------------------------------------------------------------------
# Humidity, for the latent-heat side of the budget
# ---------------------------------------------------------------------------
# The latent flux is a bulk flux too,
#
#     LH_up ~ rho * L_v * C_E * U * (q_sat(T_skin) - q_2m)
#
# so its driver is a HUMIDITY difference in the same way the sensible flux's
# driver is a temperature difference. ERA5 stores neither q_2m nor the surface
# saturation value, so both are derived here from the 2 m dewpoint and the skin
# temperature with Alduchov & Eskridge (1996) saturation vapour pressures.
#
# APPROXIMATE, AND KNOWN TO BE. ERA5's own turbulence scheme computes the
# surface humidity with a resistance formulation over land and snow, so this is
# not a reconstruction of what the model did -- it is an independent estimate of
# the same physical quantity, good enough to serve as a predictor and as a
# control, and not good enough to close the model's latent flux exactly. The
# saturation is taken over ICE when the skin is below freezing, which is what a
# frozen surface actually presents to the air; using the water value there
# overstates q_sat by about 10% at -20 C.
EPSILON_RD_RV = 0.621981          # R_dry / R_vapour


def sat_vapour_pressure_hpa(t_k: np.ndarray, over_ice: np.ndarray | None = None):
    """Saturation vapour pressure in hPa (Alduchov & Eskridge 1996).

    ``over_ice`` selects the ice formulation element-wise; None means water
    everywhere, which is the right choice for a dewpoint temperature since
    dewpoint is defined with respect to water.
    """
    tc = t_k - 273.15
    e_w = 6.1094 * np.exp(17.625 * tc / (tc + 243.04))
    if over_ice is None:
        return e_w
    e_i = 6.1121 * np.exp(22.587 * tc / (tc + 273.86))
    return np.where(over_ice, e_i, e_w)


def specific_humidity(e_hpa: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Specific humidity [kg kg-1] from vapour pressure and total pressure."""
    denom = p_hpa - (1.0 - EPSILON_RD_RV) * e_hpa
    return EPSILON_RD_RV * e_hpa / np.where(denom > 0, denom, np.nan)


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
    "skt", "t2m", "u10", "v10", "d2m", "sp",
)

# Read only when with_cloud_temperature is on: sp clips the lowest pressure
# levels at the ground so a level below terrain contributes no liquid.
CLOUD_T_EXTRA_VARS: tuple[str, ...] = ("sp",)


# ---------------------------------------------------------------------------
# Mean cloud temperature, from the pressure-level archive
# ---------------------------------------------------------------------------
# Rachel's Barrow figure colours its LWP-against-DLR scatter by MEAN CLOUD
# TEMPERATURE, and that colour is most of the physics in the plot: at fixed LWP
# the downwelling longwave is set by how warm the emitting layer is, so the
# LWP-DLR slope is partly a temperature covariance rather than an opacity
# effect. Reproducing the comparison without it would only match the axes.
#
# ERA5's single-level archive carries no cloud temperature, so this comes from
# the pressure-level archive as the LIQUID-WEIGHTED mean:
#
#     T_cloud = sum_l (clwc_l * dp_l * T_l) / sum_l (clwc_l * dp_l)
#
# i.e. each level weighted by the liquid water path it contributes. Liquid
# rather than total condensate, because the quantity on the x axis is the
# LIQUID water path and the population is liquid-bearing cloud; weighting by
# total condensate would let a deep ice layer aloft drag the mean toward a
# temperature no liquid ever had. Columns with no liquid above the floor get
# NaN and are simply absent from the coloured figure -- they are not dropped
# from anything else, which is why this is NOT a tracked variable: requiring it
# finite would silently delete every ice-only column from every other figure.
#
# This costs a second pass over a second archive, so it is opt-in:
# prepare(with_cloud_temperature=True).
CLOUD_T_MIN_LWP_G = 1e-3    # g m-2 of column liquid needed to define a mean

EXTRA_FIELDS: dict[str, tuple[str, str]] = {
    "tcld_C": ("Mean cloud temperature", r"$^{\circ}$C"),
}


# ---------------------------------------------------------------------------
# UNCERTAINTY: why the textbook standard error is unusable, and what replaces it
# ---------------------------------------------------------------------------
# THE PROBLEM. The ordinary least-squares standard error,
#
#     SE(b) = (sd_y / sd_x) * sqrt((1 - r^2) / (n - 2)),
#
# is derived under the assumption that the n residuals are INDEPENDENT draws.
# ERA5 cell-hours are not remotely independent, in either dimension:
#
#   MEASURED, Barrow strip, Oct 2024 - Mar 2025
#     lag-1 hourly autocorrelation      DLR 0.988,  SHF 0.984
#     -> one independent sample per ~72 h, not per hour
#     DLR correlation across the domain  r = 0.99 at 50 km, still 0.65 at 600 km
#     -> the 1100 x 550 km strip is ONE weather system wide, so 2501 grid cells
#        carry on the order of one to three independent pieces of information,
#        not 2501
#
# So a season of 4368 hours over 2501 cells is not 10.9 million independent
# samples; it is roughly 60 independent synoptic situations. The naive SE is
# too small by about sqrt(n / n_eff) ~ 130, and it produces intervals that
# claim four significant figures on a slope that is not even distinguishable
# from zero.
#
# WHAT A CONFIDENCE INTERVAL EVEN MEANS HERE. It is not "how much would this
# number move if we re-measured", because there is no measurement error to
# speak of -- ERA5 hands us the same value every time we read the file. The
# useful question is different:
#
#     The record is ONE REALISATION of the atmosphere's synoptic variability.
#     If the same climate had thrown a different sequence of weather systems at
#     this domain, how different would the fitted slope have been?
#
# That is a real sampling distribution, and the record contains enough distinct
# weather systems to estimate it -- just far fewer than it contains cell-hours.
#
# THE ESTIMATOR: A MOVING-BLOCK BOOTSTRAP.
#
#   1. Cut the record into contiguous blocks of time LONGER than the
#      decorrelation time, so different blocks are near-independent. Each block
#      is one or two synoptic situations' worth of information.
#   2. Draw blocks at random WITH REPLACEMENT until a synthetic record of the
#      original length is assembled. Each synthetic record is a plausible
#      alternative answer to "which weather happened".
#   3. Refit the slope on each synthetic record. The spread across synthetic
#      records IS the sampling distribution; its 2.5th and 97.5th percentiles
#      are the 95% interval.
#
# WHY BLOCKS AND NOT INDIVIDUAL POINTS. Resampling cell-hours one at a time
# destroys the correlation structure and reproduces the too-narrow naive
# answer. Keeping a block contiguous preserves the correlation WITHIN a weather
# system, so each block contributes one system's worth of information, which is
# the correct quantum.
#
# WHY THE SPATIAL CORRELATION NEEDS NO SEPARATE TREATMENT. A block is a slab of
# TIME containing the entire spatial field. Whatever correlation exists between
# grid cells travels inside the block untouched, and the resampling only shuffles
# blocks along the time axis -- the one axis along which the record really does
# supply repeated, near-independent draws. So the spatial degrees of freedom
# never have to be estimated; the method is agnostic to them.
#
# CHOOSING THE BLOCK LENGTH. It must exceed the decorrelation time (~3 days
# here) so that blocks are near-independent, but stay short enough to leave
# many blocks to resample. The diagnostic is convergence: lengthen the block
# and the interval widens until the blocks are genuinely independent, then
# stops. MEASURED on sea ice, one season: SE = 0.0256 at 3-day blocks, 0.0285
# at 7 days, 0.0299 at 14 days -- converged by about a week, which is why
# DEFAULT_BOOTSTRAP_BLOCK_DAYS is 7.
#
# THIS IS MADE CHEAP BY THE MOMENTS BEING ADDITIVE. The per-block moment
# matrices are accumulated during the SAME streaming pass as everything else,
# and a bootstrap replicate is then just a weighted sum of block matrices --
# no re-reading, no refitting from raw samples. Two thousand replicates cost
# well under a second.
#
# WHAT THE INTERVAL STILL DOES NOT COVER. It is the sampling variability of the
# weather, for THIS domain and THESE seasons. It says nothing about whether
# ERA5's physics is right, and it does not convert a covariance into a causal
# effect.
DEFAULT_BOOTSTRAP_BLOCK_DAYS = 7
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260904


def block_index(ds, use_step: np.ndarray, block_days: int):
    """Map each time step to a contiguous block, compactly numbered.

    Returns ``(blk_of_step, n_block)``; steps outside the selected seasons get
    -1. Blocks are cut on absolute calendar days, so a block never spans two
    seasons -- the out-of-season steps between them are simply absent, and the
    blocks either side keep their own identities.
    """
    times = ds["valid_time"].values
    day = ((times - times[0]) / np.timedelta64(1, "D")).astype(np.int64)
    raw = day // int(block_days)
    used = np.unique(raw[use_step])
    lookup = {int(b): i for i, b in enumerate(used)}
    out = np.full(raw.shape, -1, dtype=np.int64)
    out[use_step] = [lookup[int(b)] for b in raw[use_step]]
    return out, len(used)


def steps_in_seasons(layout: dict, wanted_idx: list[int]) -> np.ndarray:
    """Boolean over the time axis: steps inside one of the wanted seasons."""
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[wanted_idx] = True
    return in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]


def cloud_temperature_field(ds, args, use_step: np.ndarray):
    """Liquid-weighted mean cloud temperature for every wanted cell-hour.

    Returns ``(row_of, tcld_C)``. ``row_of`` maps a single-level time index to
    a row of ``tcld_C`` (shape ``(n_matched, n_lat, n_lon)``, degrees Celsius),
    or -1 where the pressure archive has no matching timestamp.

    Held in memory rather than streamed alongside the main pass because the two
    archives are separate files with their own time axes; at four seasons this
    is about 175 MB, which is cheaper than the bookkeeping of interleaving two
    block iterators.
    """
    region_pl = f"{strip_frequency_suffix(args.region)}_pressure"
    data_root = resolve_data_root(args.storage, args.data_root)
    if region_pl not in available_regions(data_root):
        raise FileNotFoundError(
            f"cloud temperature needs the pressure-level archive "
            f"{data_root / region_pl}, which is not there. Download it with "
            f"download_era5_pressure.py, or call prepare() without "
            f"with_cloud_temperature.")

    pds = load_seb_data(region_pl, None, None, data_root)
    missing = sorted({"t", "clwc"} - set(pds.data_vars))
    if missing:
        raise KeyError(f"pressure archive is missing {missing}")

    want = np.flatnonzero(use_step)
    t_want = ds["valid_time"].values[want]
    t_pl = pds["valid_time"].values
    pos = np.searchsorted(t_pl, t_want)
    inside = pos < t_pl.size
    match = np.zeros(want.size, dtype=bool)
    match[inside] = t_pl[pos[inside]] == t_want[inside]

    n_lat, n_lon = ds.sizes["latitude"], ds.sizes["longitude"]
    tcld = np.full((want.size, n_lat, n_lon), np.nan, dtype=np.float32)
    row_of = np.full(ds.sizes["valid_time"], -1, dtype=np.int64)
    row_of[want] = np.arange(want.size)

    n_hit = int(match.sum())
    print(f"  Cloud temp : {n_hit:,} of {want.size:,} wanted steps matched in "
          f"{region_pl}"
          + ("" if n_hit == want.size else "; the rest are uncoloured"))
    if n_hit == 0:
        return row_of, tcld

    lv = pds["pressure_level"].values.astype(float)
    order = np.argsort(lv)                       # ascending pressure
    p_pa = lv[order] * 100.0

    rows = np.flatnonzero(match)
    step = max(1, args.block_hours // 8)         # 23 levels: smaller blocks
    for i0 in range(0, rows.size, step):
        sel = rows[i0:i0 + step]
        blk = pds[["t", "clwc"]].isel(valid_time=pos[sel]).load()
        sp_pa = ds["sp"].isel(valid_time=want[sel]).values.astype(float)

        dims = [d for d in blk["t"].dims if d != "pressure_level"]
        t_k = blk["t"].transpose(*dims, "pressure_level").values.astype(
            np.float64)[..., order]
        clwc = blk["clwc"].transpose(*dims, "pressure_level").values.astype(
            np.float64)[..., order]

        dp = layer_thickness_pa(p_pa, sp_pa)                 # (..., n_lev)
        lwp_l = np.clip(clwc, 0.0, None) * dp / G_M_S2 * 1000.0   # g m-2
        denom = lwp_l.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_t = (lwp_l * t_k).sum(axis=-1) / denom
        mean_t[denom < CLOUD_T_MIN_LWP_G] = np.nan
        tcld[sel] = (mean_t - 273.15).astype(np.float32)

    return row_of, tcld


def tcc_shape_placeholder(block, keep: np.ndarray) -> tuple:
    """Shape of one block's kept cell-hours, for allocating an extra field."""
    return block["tcc"].values[keep].shape


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
          "u10", "v10", "tclw", "tciw", "d2m", "sp")}
    p_hpa = v["sp"] / 100.0
    wspd = np.hypot(v["u10"], v["v10"])
    d_skt = v["skt"] - v["t2m"]
    q2m = specific_humidity(sat_vapour_pressure_hpa(v["d2m"]), p_hpa)
    q_surf = specific_humidity(
        sat_vapour_pressure_hpa(v["skt"], over_ice=v["skt"] < 273.15), p_hpa)
    dq = (q_surf - q2m) * 1000.0                     # kg kg-1 -> g kg-1
    out = {
        "shf_W_m2": v["msshf"],
        "lhf_W_m2": v["mslhf"],
        "lwd_W_m2": v["msdwlwrf"],
        # msnlwrf = LWD - LWU, so LWU = LWD - msnlwrf.
        "lwu_W_m2": v["msdwlwrf"] - v["msnlwrf"],
        "swnet_W_m2": v["msnswrf"],
        "skt_K": v["skt"],
        "t2m_K": v["t2m"],
        "dskt_t2m_K": d_skt,
        "lwp_g_m2": v["tclw"] * 1000.0,          # kg m-2 -> g m-2
        "iwp_g_m2": v["tciw"] * 1000.0,
        "wspd_m_s": wspd,
        "q2m_g_kg": q2m * 1000.0,
        "dq_g_kg": dq,
        "u_dskt_K_m_s": wspd * d_skt,
        "u_dq_g_kg_m_s": wspd * dq,
        "sp_hPa": p_hpa,
    }
    out["rnet_W_m2"] = (out["lwd_W_m2"] - out["lwu_W_m2"] + out["swnet_W_m2"]
                        + out["shf_W_m2"] + out["lhf_W_m2"])
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
# Short forms for axes that carry all six groups side by side: the full
# labels overlap below about an inch per group, and a tick label that
# overlaps its neighbour is worse than an abbreviated one.
SLOT_SHORT: dict[str, str] = {
    "land": "Land", "coastal": "Coastal", "open_ocean": "Open\nocean",
    "marginal_ice": "Marginal\nice", "sea_ice": "Sea ice",
    SITE_KEY: "Utqiagvik", "all": "All",
}

# The six panels the scatter figures draw, in reading order: the five classes
# then the site cell. "all" is accumulated but not panelled -- pooling five
# surfaces with opposite turbulent regimes into one scatter is the exact
# average the rest of this module exists to avoid.
PANEL_SLOTS: tuple[int, ...] = tuple(range(len(CLASS_ORDER))) + (SITE_SLOT,)

POPULATIONS: tuple[str, ...] = ("cloud", "all")


# ----------------------------------------------------------------------------
# 2-D density histograms
# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# WHICH WAY TO RUN THE REGRESSION -- this is not a presentation choice
# ---------------------------------------------------------------------------
# The two ordinary least-squares fits of a pair are DIFFERENT ESTIMATORS, not
# one estimator drawn two ways:
#
#     slope(y|x) = cov / var_x        slope(x|y) = cov / var_y
#     slope(y|x) * slope(x|y) = r^2
#
# So 1/slope(y|x) = slope(x|y) / r^2. INVERTING THE WRONG FIT INFLATES THE
# ANSWER BY 1/r^2, away from zero, and the inflation is worst exactly where the
# scatter is largest and a reader is least likely to notice.
#
# MEASURED, Barrow strip, Oct-Mar 2022/23-2025/26, fraction-mode liquid-bearing
# overcast, on (T_skin - T_2m) against sensible heat flux -- the bulk coupling
# coefficient d(SHF)/d(dT), in W m-2 K-1:
#
#     class            1/slope(dT|SHF)   slope(SHF|dT)   inflation
#     Land                        28.3             3.7       7.8x
#     Coastal                     21.7            14.4       1.5x
#     Open ocean                  23.9            19.1       1.3x
#     Marginal ice zone           23.4            19.3       1.2x
#     Sea ice                     26.7             9.9       2.7x
#     Utqiagvik                   22.5             4.0       5.6x
#
# Read the first column and the coupling looks REMARKABLY UNIFORM at 22-28
# across every surface. It is not: the correct column spans a factor of five,
# and the ordering is different. The apparent uniformity was the 1/r^2 factor
# tracking the scatter, not the physics. The corrected values are also the ones
# that make physical sense -- rho*c_p*C_H*U is about 1300 * 1.3e-3 * 6 = 10
# W m-2 K-1 for a stable Arctic boundary layer over pack ice and roughly twice
# that over open water with stronger winds and a rougher surface, which is what
# the second column says and the first does not.
#
# HOW THIS IS HANDLED HERE. Every panel now puts the RESPONSE ON THE Y AXIS, so
# the physically meaningful fit is the ordinary y-on-x one and no panel needs a
# special direction. The DLR panel's annotated slope is then exactly -f_SH from
# the partition figure -- VERIFIED equal to machine precision, since both are
# cov(SHF, LWD) / var(LWD).
#
# The option below survives because the WRONG direction is worth being able to
# draw: "both" overlays the two fits, and watching them fan apart on the
# low-r^2 panels is the clearest available statement of why the direction
# matters.
#
#     "y_on_x"   fit y = a + b x. For LWP against SHF, where neither variable
#                drives the other and the fit is descriptive.
#     "x_on_y"   fit x = a + b y, i.e. d(SHF)/d(other). For DLR and for
#                (T_skin - T_2m), where the question is how the FLUX responds.
#     "both"     draw both, which makes the divergence visible; they coincide
#                only as r^2 -> 1.
#
# r^2 is symmetric, so it is the same number whichever direction is fitted, and
# it is annotated once.
FIT_ORIENTS: tuple[str, ...] = ("y_on_x", "x_on_y", "both")
DEFAULT_FIT_ORIENT = "panel"      # honour each panel's own declaration


class Panel2D(NamedTuple):
    """Axis configuration for one density-scatter figure."""

    x_key: str
    y_key: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_bins: int
    y_bins: int
    fit_orient: str = "y_on_x"
    color_key: str | None = None      # colour by this field's MEAN, not density


# Ranges are fixed rather than derived from the data so that runs over
# different seasons stay directly comparable panel to panel. They were chosen
# from the 1st-99th percentile span of the Barrow strip under liquid-bearing
# overcast, widened to the nearest round number; the fraction of samples
# falling outside is accumulated and printed on the figure, so a range that
# turns out to be wrong for another region announces itself instead of
# silently clipping.
DEFAULT_PANELS: dict[str, Panel2D] = {
    # THE RESPONSE GOES ON Y. Every panel is a plain y-on-x least-squares fit
    # whose slope is the derivative named in the figure title, so the axes and
    # the reported number agree without the reader having to reconcile them.
    #
    # These four originally had the flux on the x axis, and the fit ran x on y
    # to recover the right slope. That returns the identical number -- both are
    # cov(flux, driver) / var(driver) -- but it asks a reader to hold "the
    # regression runs the other way from the axes" in their head while looking
    # at the plot. Putting the response on y removes the reconciliation step.
    "lwp_shf": Panel2D("lwp_g_m2", "shf_W_m2", (0.0, 300.0), (-250.0, 60.0),
                       180, 180, "y_on_x"),
    "dlr_shf": Panel2D("lwd_W_m2", "shf_W_m2", (140.0, 330.0), (-250.0, 60.0),
                       180, 180, "y_on_x"),
    # The bulk coupling coefficient rho*c_p*C_H*U is d(SHF)/d(T_skin - T_2m),
    # so the temperature difference is the predictor and the flux the response.
    "dskt_shf": Panel2D("dskt_t2m_K", "shf_W_m2", (-8.0, 14.0), (-250.0, 60.0),
                        180, 180, "y_on_x"),
    "dlr_lhf": Panel2D("lwd_W_m2", "lhf_W_m2", (140.0, 330.0), (-150.0, 40.0),
                       180, 180, "y_on_x"),
    # The comparison against the ARM observations at Barrow. Axes and colour
    # follow that figure so the two can be read side by side: LWP 0-350 g m-2,
    # DLR 100-350 W m-2, coloured by mean cloud temperature. DLR is the
    # response to LWP here, which is also the direction the published
    # y = 0.27x + 228.26 was fitted in.
    "lwp_dlr": Panel2D("lwp_g_m2", "lwd_W_m2", (0.0, 350.0), (100.0, 350.0),
                       180, 180, "y_on_x", "tcld_C"),
    # The three remaining terms of the DLR partition, on the same axes and the
    # same fit direction as "dlr_shf" and "dlr_lhf", so all five panels of the
    # budget can be read the same way. Each fitted slope IS the corresponding
    # partition fraction: +d(LWU)/d(DLR) for LWU, -d/d(DLR) for SW_net (the
    # fraction is defined with the sign flipped), and f_res for R.
    "dlr_lwu": Panel2D("lwd_W_m2", "lwu_W_m2", (140.0, 330.0), (150.0, 350.0),
                       180, 180, "y_on_x"),
    # SW_net cannot be negative, and in the dark half of the season it is
    # exactly zero, so the mass piles on the bottom axis. That is the result
    # for this panel, not a range that wants widening.
    "dlr_swnet": Panel2D("lwd_W_m2", "swnet_W_m2", (140.0, 330.0),
                         (0.0, 120.0), 180, 180, "y_on_x"),
    "dlr_rnet": Panel2D("lwd_W_m2", "rnet_W_m2", (140.0, 330.0),
                        (-450.0, 200.0), 180, 180, "y_on_x"),
}


# ---------------------------------------------------------------------------
# LWP REGIMES: why 10 and 40 g m-2 and not terciles
# ---------------------------------------------------------------------------
# A liquid cloud's longwave emissivity follows eps = 1 - exp(-a * LWP) with
# a ~ 0.15 m2 g-1 (Stephens 1978), so:
#
#     LWP = 10 g m-2   eps = 0.78     thin: DLR still responds to more liquid
#     LWP = 20 g m-2   eps = 0.95
#     LWP = 40 g m-2   eps = 0.998    radiatively BLACK: DLR no longer responds
#
# The two edges therefore split the population where the PHYSICS changes, which
# is what a matched comparison needs. Data-driven edges -- terciles, say --
# would fall in different places for each surface class, and then a bar chart
# comparing classes within a "regime" would be comparing different regimes, and
# would reintroduce exactly the confounding the stratification exists to
# remove. Fixed physical edges keep the comparison matched.
#
# The floor of the low bin is the phase scheme's own liquid floor, so the bins
# partition precisely the population the rest of the module describes.
LWP_REGIME_EDGES_G: tuple[float, ...] = (10.0, 40.0)
LWP_REGIME_LABELS: tuple[str, ...] = ("low", "medium", "high")

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
def regime_of(lwp_g: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """LWP regime index per sample: 0 = low, 1 = medium, 2 = high."""
    return np.digitize(lwp_g, np.asarray(edges, dtype=float))


def regime_labels(edges: tuple[float, ...], floor_g: float) -> list[str]:
    """Axis labels naming the actual g m-2 span of each regime."""
    lo = [floor_g] + list(edges)
    hi = list(edges) + [np.inf]
    out = []
    for name, a, b in zip(LWP_REGIME_LABELS, lo, hi):
        span = (f"$>$ {a:g}" if not np.isfinite(b)
                else f"{a:g}$-${b:g}")
        out.append(f"{name}\nLWP {span} g m$^{{-2}}$")
    return out


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


def moments_from_arrays(fields: dict, weights=None, group=None,
                        n_group: int | None = None) -> dict:
    """Build a moment accumulator from in-memory arrays.

    The streaming path fills these accumulators from the ERA5 archive; this
    fills the SAME structure from arrays already in memory, so an entirely
    different dataset -- a shipboard record, say -- can be handed to
    ``moment_stats``, ``multiple_stats``, ``partial_slope`` and the bootstrap
    without any of those estimators being reimplemented. That is the point:
    a comparison between two datasets is only a comparison if the estimator is
    literally the same code.

    Parameters
    ----------
    fields : ``{TRACKED key: 1-D array}``. Keys must be names in ``TRACKED``;
        slots not supplied are left at zero, so they carry no variance and any
        regression asking for them returns NaN rather than a wrong number.
    weights : per-sample weights, default 1 (unweighted).
    group : optional integer group index per sample, for building a
        per-block accumulator in one call. ``-1`` drops a sample.

    Every array must be finite: the moment update is a matrix product, so a
    single NaN would propagate into every entry. Drop missing rows first.
    """
    bad = sorted(set(fields) - set(VAR_INDEX))
    if bad:
        raise KeyError(f"not tracked variables: {bad}. "
                       f"Choose from {sorted(VAR_INDEX)}")
    n = len(next(iter(fields.values())))
    values = np.zeros((N_VAR, n), dtype=np.float64)
    for k, v in fields.items():
        a = np.asarray(v, dtype=np.float64)
        if a.size != n:
            raise ValueError(f"{k!r} has length {a.size}, expected {n}")
        if not np.isfinite(a).all():
            raise ValueError(f"{k!r} contains non-finite values; drop them first")
        values[VAR_INDEX[k]] = a - CENTERS[VAR_INDEX[k]]

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if group is None:
        group = np.zeros(n, dtype=np.intp)
        n_group = 1
    group = np.asarray(group, dtype=np.intp)
    n_group = int(n_group if n_group is not None else group.max() + 1)
    acc = new_moments(n_group)
    accumulate_moments(acc, [(g, group == g) for g in range(n_group)], values, w)
    return acc


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
# because the ERA5 open-ocean surface barely moves at all, so essentially the
# whole skin-to-air difference is the AIR moving. See the note on the
# open-ocean surface below.
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


def fit_pair(acc: dict, slot: int, x_key: str, y_key: str,
             weighted: bool = True) -> dict:
    """Both ordinary least-squares fits of a pair, and what relates them.

    Returns ``{"y_on_x": ..., "x_on_y": ..., "r2": ..., "inflation": ...}``.
    The two stat dicts are ``moment_stats`` with the arguments swapped -- there
    is no second estimator to implement, only a second question to ask.

    ``inflation`` is ``1 / r2``, the factor by which INVERTING the y-on-x slope
    overstates the x-on-y slope. It is reported because that inversion is the
    natural mistake to make when a figure plots the response on the x axis, and
    because its size is not obvious: at the sea-ice r^2 of 0.37 it is a factor
    of 2.7, and at the DLR-against-SHF r^2 of 0.025 it is a factor of forty.
    """
    yx = moment_stats(acc, slot, x_key, y_key, weighted=weighted)
    xy = moment_stats(acc, slot, y_key, x_key, weighted=weighted)
    r2 = yx["r2"]
    return {
        "y_on_x": yx,
        "x_on_y": xy,
        "r2": r2,
        "inflation": (1.0 / r2) if np.isfinite(r2) and r2 > 0 else np.nan,
    }


def multiple_stats(acc: dict, slot: int, y_key: str,
                   x_keys: tuple[str, ...], weighted: bool = True) -> dict:
    """Full multiple regression of y on ``x_keys``: coefficients and R^2.

    ``partial_slope`` returns only the first coefficient, which is all the
    sensitivity tables need. This returns the whole fit, including the
    MULTIPLE coefficient of determination

        R^2 = beta . cov(x, y) / var(y),

    the fraction of the variance in y that all the predictors together account
    for. It is not comparable to the r^2 of a single-predictor fit on a
    different predictor -- adding a variable can only raise R^2, so the
    interesting comparison is always between specifications with the same y,
    and between predictors chosen for physical reasons rather than for fit.
    """
    keys = tuple(x_keys)
    w_key, xk, xyk = (("w", "x", "xy") if weighted else ("n", "x_u", "xy_u"))
    w = acc[w_key][slot]
    out = {"x_keys": keys, "n_hours": float(acc["n"][slot]) * HOURS_PER_STEP}
    if w <= 0.0:
        out.update(coef=np.full(len(keys), np.nan), intercept=np.nan, r2=np.nan)
        return out

    idx = [VAR_INDEX[k] for k in keys] + [VAR_INDEX[y_key]]
    m = acc[xk][slot, idx] / w
    cov = acc[xyk][slot][np.ix_(idx, idx)] / w - np.outer(m, m)
    a, b, var_y = cov[:-1, :-1], cov[:-1, -1], cov[-1, -1]
    try:
        beta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        out.update(coef=np.full(len(keys), np.nan), intercept=np.nan, r2=np.nan)
        return out

    means = m + CENTERS[idx]
    out.update(
        coef=beta,
        intercept=float(means[-1] - float(beta @ means[:-1])),
        r2=float(beta @ b / var_y) if var_y > 0 else np.nan,
        x_means=means[:-1], y_mean=float(means[-1]),
    )
    return out


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


# ---------------------------------------------------------------------------
# The two turbulent terms, as partial derivatives with respect to DLR
# ---------------------------------------------------------------------------
# These are the quantities the whole module is for, so they get their own
# function rather than living as negated entries of the partition. The
# direction matters and is fixed here once: the FLUX is regressed ON the
# radiation,
#
#     d(SHF)/d(DLR) = cov(SHF, DLR) / var(DLR)
#
# not the reverse and not the reciprocal of the reverse. Regressing DLR on SHF
# and inverting would inflate the answer by 1/r^2, which at the sea-ice
# r^2 = 0.025 is a factor of forty. See FIT_ORIENTS.
#
# SIGN. Everything is in ERA5's positive-downward convention, so a POSITIVE
# d(SHF)/d(DLR) means more DLR is accompanied by more heat flowing INTO the
# surface -- the turbulent term adding to the radiative warming rather than
# opposing it. A negative value is the damping the naive argument expects.
TURBULENT_TERMS: tuple[tuple[str, str, str, str], ...] = (
    ("dshf_dlwd", "shf_W_m2", "Sensible", "#4C72B0"),
    ("dlhf_dlwd", "lhf_W_m2", "Latent", "#55A868"),
)


# ---------------------------------------------------------------------------
# WHAT MAY BE CONTROLLED FOR, AND WHAT MUST NOT BE
# ---------------------------------------------------------------------------
# A control variable is not a free improvement. Adding the wrong one biases the
# answer as surely as omitting the right one, and which is which follows from
# the causal structure, not from whether the fit looks better.
#
# The bulk formulae name the structure:
#
#     SH_up ~ rho c_p C_H U (T_skin - T_2m)
#     LH_up ~ rho L_v C_E U (q_sat(T_skin) - q_2m)
#
# and the pathway under test is
#
#     DLR ---> T_skin ---> (T_skin - T_2m) ---> SHF
#
# THE RULE. Control for a variable that influences y through a path that does
# NOT run through x, and that is correlated with x. Three kinds must be left
# alone:
#
#   MEDIATORS -- anything on the path x -> ... -> y. Controlling one removes
#     the very effect being estimated. T_skin and (T_skin - T_2m) are pure
#     mediators for d(SHF)/d(DLR): hold the skin temperature fixed and DLR by
#     construction can no longer do anything.
#
#   COLLIDERS -- anything caused by BOTH x and y. Controlling one manufactures
#     an association that is not there.
#
#   CAUSES OF x ALONE -- harmless but useless. They remove variance in x
#     without removing bias. THIS ANSWERS THE OBVIOUS QUESTION ABOUT CLOUDS:
#     liquid water path and cloud opacity are causes of DLR, and in polar night
#     they reach the turbulent fluxes only THROUGH DLR, so controlling for them
#     strips out exactly the variance the regression is using and buys nothing.
#     Do not control for the drivers of the predictor.
#
# WHAT IS LEFT, AND WHY EACH QUALIFIES:
#
#   WIND SPEED. Multiplies the exchange coefficient, so it drives both fluxes
#     directly, and it is correlated with DLR because Arctic storms are both
#     cloudy and windy. Not on the DLR -> T_skin -> flux path. A clean
#     confounder, and the least controversial control here.
#
#   2 m AIR TEMPERATURE. Enters SHF directly through the skin-to-air
#     difference, and warm advection raises both it and DLR -- so it is a
#     confounder. But it is ALSO a mediator, because DLR warms the skin which
#     warms the air. Controlling it therefore removes real signal along with
#     the spurious part, which is why this module reports the controlled and
#     uncontrolled slopes as a bracket rather than picking one.
#
#   2 m HUMIDITY, for the latent flux. Enters LHF directly through the
#     humidity difference, and water vapour is itself a strong longwave
#     emitter, so it raises DLR too. The confounding is stronger here than the
#     temperature case because the path from humidity to DLR is direct
#     radiative physics rather than advection. Same mediator caveat applies.
#
#   SURFACE PRESSURE, as a circulation proxy. Bertrand et al. (2025) test
#     whether synoptic variability explains their result by stratifying on
#     daily surface pressure anomaly; carrying it as a control is the
#     regression form of the same check.
#
# HOW TO READ THE LADDER FIGURE. Adding a control that matters MOVES the slope.
# A slope that barely moves as controls are added is evidence that the
# confounder in question was not doing much -- which is a result, not a
# non-result, and is the same logic Bertrand et al. use when they conclude that
# circulation variability cannot explain their trend.
ControlStep = tuple[str, tuple[str, ...]]

CONTROL_LADDERS: dict[str, dict] = {
    "shf_lwp": {
        "y": "shf_W_m2", "x": "lwp_g_m2",
        "title": r"(a)  $d\mathrm{SHF}/d\mathrm{LWP}$",
        "units": "W m$^{-2}$ per g m$^{-2}$",
        "steps": [
            ("no control", ()),
            ("+ wind", ("wspd_m_s",)),
            ("+ wind, $T_{2m}$", ("wspd_m_s", "t2m_K")),
        ],
    },
    "shf_dlr": {
        "y": "shf_W_m2", "x": "lwd_W_m2",
        "title": r"(b)  $d\mathrm{SHF}/d\mathrm{DLR}$",
        "units": "W m$^{-2}$ per W m$^{-2}$",
        "steps": [
            ("no control", ()),
            ("+ wind", ("wspd_m_s",)),
            ("+ wind, $T_{2m}$", ("wspd_m_s", "t2m_K")),
            ("+ wind, $T_{2m}$, $p_s$", ("wspd_m_s", "t2m_K", "sp_hPa")),
        ],
    },
    "shf_dskt": {
        "y": "shf_W_m2", "x": "dskt_t2m_K",
        "title": r"(c)  $\rho c_p C_H U = d\mathrm{SHF}/d\Delta T$",
        "units": "W m$^{-2}$ K$^{-1}$",
        "steps": [
            ("no control", ()),
            ("+ wind", ("wspd_m_s",)),
            ("+ wind, $p_s$", ("wspd_m_s", "sp_hPa")),
        ],
    },
    "lhf_dlr": {
        "y": "lhf_W_m2", "x": "lwd_W_m2",
        "title": r"(d)  $d\mathrm{LHF}/d\mathrm{DLR}$",
        "units": "W m$^{-2}$ per W m$^{-2}$",
        "steps": [
            ("no control", ()),
            ("+ wind", ("wspd_m_s",)),
            ("+ wind, $q_{2m}$", ("wspd_m_s", "q2m_g_kg")),
            ("+ wind, $q_{2m}$, $T_{2m}$",
             ("wspd_m_s", "q2m_g_kg", "t2m_K")),
        ],
    },
}


def turbulent_response(acc: dict, slot: int,
                       control: tuple[str, ...] = ()) -> dict:
    """d(SHF)/d(DLR), d(LHF)/d(DLR) and their sum, for one group.

    All three in W m-2 per W m-2 of downwelling longwave, ERA5's
    positive-downward convention throughout. ``r2_shf`` and ``r2_lhf`` are the
    ZERO-ORDER coefficients of determination of each flux against DLR alone --
    they describe the scatter a reader sees on the corresponding panel, and are
    unchanged by ``control``, which alters the slope but not that panel.

    ``dturb_dlwd`` is the total turbulent response, and it is the sum of the
    two by construction: the regression operator is linear, so regressing
    (SHF + LHF) on DLR and adding the two separate slopes give the same number.
    """
    out = {"n_hours": float(acc["n"][slot]) * HOURS_PER_STEP,
           "control": control}
    for key, var, _, _ in TURBULENT_TERMS:
        out[key] = slope_of(acc, slot, var, control=control)
        st = moment_stats(acc, slot, "lwd_W_m2", var)
        out[f"r2_{var.split('_')[0]}"] = st["r2"]
    out["dturb_dlwd"] = out["dshf_dlwd"] + out["dlhf_dlwd"]
    return out


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
# The partition, term by term: the net flux, and the sign-honest ledger
# ----------------------------------------------------------------------------
# The surface energy balance with every term positive downward,
#
#     R = LWD - LWU + SW_net + SH + LH,
#
# as a coefficient vector over the tracked variables. R is the NET flux into
# the surface: what the skin has left after the four exchanges with the
# atmosphere, and therefore what must be conducted or stored below. ERA5's
# single-level archive carries no ground heat flux, so this combination is the
# only route to that term, and it is what ``partition`` calls ``f_res``.
NET_FLUX_COMBO: dict[str, float] = {
    "lwd_W_m2": 1.0,
    "lwu_W_m2": -1.0,
    "swnet_W_m2": 1.0,
    "shf_W_m2": 1.0,
    "lhf_W_m2": 1.0,
}


def combo_slope(acc: dict, slot: int, coefs: dict[str, float],
                x_key: str = "lwd_W_m2",
                control: tuple[str, ...] = ()) -> float:
    """d(sum_k c_k v_k)/d(x) holding ``control`` fixed, in ONE regression.

    The point of this function is that it does not sum slopes. It forms the
    linear combination inside the covariance matrix and then solves once, so
    the number it returns comes back through a different matrix solve than
    adding the individual ``partial_slope`` calls does. Agreement between the
    two is what ``partition_closure`` checks.

    That agreement is guaranteed in exact arithmetic -- covariance is linear in
    each argument, and so is the multiple-regression coefficient -- which is
    exactly why the partition is allowed to define one of its five terms as the
    remainder of the other four. The check is on the implementation and on the
    floating-point path, not on a physical claim.
    """
    if acc["w"][slot] <= 0.0:
        return float("nan")
    pred = (x_key,) + tuple(k for k in control if k != x_key)
    keys = pred + tuple(k for k in coefs if k not in pred)
    cov = _covariance_block(acc, slot, keys)
    idx = {k: i for i, k in enumerate(keys)}
    c = np.zeros(len(keys))
    for k, v in coefs.items():
        c[idx[k]] += v
    n_p = len(pred)
    try:
        beta = np.linalg.solve(cov[:n_p, :n_p], cov[:n_p, :] @ c)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(beta[0])


def net_flux_slope(acc: dict, slot: int,
                   control: tuple[str, ...] = ()) -> float:
    """d(R)/d(DLR): the net flux into the surface per W m-2 of DLR."""
    return combo_slope(acc, slot, NET_FLUX_COMBO, "lwd_W_m2", control)


# The five fractions are regression coefficients, and nothing constrains a
# regression coefficient to lie in [0, 1]. Where one comes out NEGATIVE the
# channel is disposing of LESS than it did before, so relative to the base
# state the surface keeps more.
#
# BE CAREFUL WHAT THAT IS CLAIMING. Over open water the turbulent fluxes are
# upward 90% of the time and stay upward: the binned mean sensible flux runs
# from -105 W m-2 in the lowest DLR bin to -13 in the highest without reaching
# zero. Nothing new arrives from the atmosphere. An ordinary heat loss is
# suppressed -- which is exactly the turbulent damping this module set out to
# measure -- and in an ANOMALY budget a suppressed loss is indistinguishable
# from a gain. Over sea ice, where the mean flux is downward and strengthens
# with DLR, the flux really is carrying more energy in; the sign of the
# fraction does not distinguish the two cases, and only the mean flux does.
#
# Either way, stacking a negative share on a bar that is supposed to read as
# "where the 1 W m-2 went" is a category error: the stack still sums to one,
# but only because the sinks overshoot to compensate, which is why open water
# shows f_res above three.
#
# The ledger below fixes the presentation without touching the estimates. Split
# the five terms by sign, put the DLR unit itself on the supply side, and
# normalise both sides by the same GROSS total
#
#     G = 1 + sum of the magnitudes of the negative terms
#       =     sum of the positive terms                     (identically)
#
# so supply and disposal each sum to one for EVERY surface class, open water
# included. What was an off-scale bar becomes a readable statement: over open
# water only a quarter of the energy arriving with a DLR anomaly is the DLR
# anomaly, three quarters is turbulent, and essentially all of it goes into a
# surface that ERA5 will not let warm.
LEDGER_DLR_LABEL = "DLR anomaly (the 1 W m$^{-2}$)"
LEDGER_DLR_COLOR = "#C8A02C"


def partition_ledger(part: dict) -> dict:
    """Re-cast one ``partition`` result as a two-sided ledger summing to one.

    Returns ``{"gross": G, "supply": {...}, "disposal": {...}}`` with both
    inner dicts keyed by the ``PARTITION_TERMS`` keys plus ``"dlr"`` on the
    supply side, and both summing to 1 up to rounding. ``gross`` is the total
    energy per W m-2 of DLR that changes hands, and it is the number to report
    beside the bar: G = 1 means the DLR anomaly is the whole story, G = 4 means
    it is a quarter of it.
    """
    vals = {k: part[k] for k, _, _ in PARTITION_TERMS}
    gross = 1.0 + sum(-v for v in vals.values() if v < 0)
    if not np.isfinite(gross) or gross <= 0.0:
        nan = {k: float("nan") for k in vals}
        return {"gross": float("nan"), "supply": nan, "disposal": dict(nan)}
    supply = {"dlr": 1.0 / gross}
    disposal = {}
    for k, v in vals.items():
        if v < 0:
            supply[k] = -v / gross
        else:
            disposal[k] = v / gross
    return {"gross": gross, "supply": supply, "disposal": disposal}


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def collect(ds, lsm: np.ndarray, args, layout: dict, wanted_idx: list[int],
            panels: dict, dlr_edges: np.ndarray,
            siconc_edges: np.ndarray, phase_kw: dict,
            cloud_t: tuple | None = None,
            use_step: np.ndarray | None = None) -> dict:
    """Accumulate every moment and histogram this module needs, in one pass.

    The archive open is the expensive step -- minutes -- and everything below
    is filled from the same blocks, so adding a figure costs nothing as long as
    what it needs is accumulated here. Only the seasons in ``wanted_idx`` are
    read at all.

    Returns a dict of accumulators; see ``prepare`` for what goes where.
    """
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]

    if use_step is None:
        use_step = steps_in_seasons(layout, wanted_idx)

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
    # Stratified by LWP regime AND surface class: the matched comparison. Full
    # moments per cell, so the bar chart gets a mean, the whisker gets a
    # standard deviation, and any within-bin regression is available later
    # without another pass. Flattened as slot * n_regime + regime.
    n_regime = len(LWP_REGIME_LABELS)
    regime_mom = {p: new_moments(N_SLOT * n_regime) for p in POPULATIONS}
    # Per-block moments for the bootstrap, cloud population only: the intervals
    # are wanted on the filtered figures, and carrying both populations would
    # double a bookkeeping cost for nothing.
    blk_of, n_block = block_index(ds, use_step, args.bootstrap_block_days)
    block_mom = new_moments(n_block * N_SLOT)

    hist = {name: np.zeros((N_SLOT, p.x_bins, p.y_bins))
            for name, p in panels.items()}
    hist_out = {name: np.zeros(N_SLOT) for name in panels}   # off-range weight
    # Weighted SUM of the colour field per bin, and the weight that went with
    # it. Kept separate from `hist` because a colour field may be undefined
    # (no liquid, so no cloud temperature) where the bin still holds samples,
    # and the colour must then be the mean over the defined ones only.
    colour_panels = {name: p.color_key for name, p in panels.items()
                     if p.color_key}
    hist_cs = {name: np.zeros((N_SLOT, p.x_bins, p.y_bins))
               for name, p in panels.items() if p.color_key}
    hist_cw = {name: np.zeros((N_SLOT, p.x_bins, p.y_bins))
               for name, p in panels.items() if p.color_key}
    row_of, tcld_all = cloud_t if cloud_t is not None else (None, None)

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
        extra: dict[str, np.ndarray] = {}
        if row_of is not None:
            rows = row_of[i0:i0 + n_t][keep]
            got = rows >= 0
            tc = np.full(tcc_shape_placeholder(block, keep), np.nan)
            if got.any():
                tc[got] = tcld_all[rows[got]]
            extra["tcld_C"] = tc
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

        reg_flat = regime_of(fields["lwp_g_m2"].ravel(), args.lwp_regime_edges)

        for pop, pmask in pop_masks.items():
            slot_of = [(CLASS_CODES[name], cls_flat == CLASS_CODES[name])
                       for name in CLASS_ORDER]
            slot_of.append((SITE_SLOT, site_flat))
            slot_of.append((ALL_SLOT, np.ones(cls_flat.shape, dtype=bool)))

            groups = [(slot, pmask & m) for slot, m in slot_of]
            accumulate_moments(mom[pop], groups, values, w_flat)

            reg_groups = [(slot * n_regime + r, pmask & m & (reg_flat == r))
                          for slot, m in slot_of for r in range(n_regime)]
            accumulate_moments(regime_mom[pop], reg_groups, values, w_flat)

            if pop == "cloud":
                # Only the blocks this streaming chunk actually touches, so the
                # loop stays a handful of groups rather than all n_block.
                bi = np.broadcast_to(blk_of[i0:i0 + n_t][keep][:, None, None],
                                     classes.shape).ravel()
                for b in np.unique(bi[pmask]):
                    if b < 0:
                        continue
                    inb = pmask & (bi == b)
                    accumulate_moments(
                        block_mom,
                        [(int(b) * N_SLOT + slot, inb & m) for slot, m in slot_of],
                        values, w_flat)

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
                    ck = colour_panels.get(name)
                    if ck and ck in extra:
                        cv = extra[ck].ravel()[m2][inside]
                        fin = np.isfinite(cv)
                        if fin.any():
                            fl, wf = flat[fin], ws2[inside][fin]
                            hist_cs[name][slot] += np.bincount(
                                fl, weights=wf * cv[fin],
                                minlength=p.x_bins * p.y_bins,
                            ).reshape(p.x_bins, p.y_bins)
                            hist_cw[name][slot] += np.bincount(
                                fl, weights=wf,
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
        "regime_mom": regime_mom,
        "block_mom": block_mom,
        "n_block": n_block,
        "block_days": int(args.bootstrap_block_days),
        "regime_edges": tuple(args.lwp_regime_edges),
        "n_regime": n_regime,
        "hist": hist,
        "hist_out": hist_out,
        "hist_colour_sum": hist_cs,
        "hist_colour_weight": hist_cw,
        "has_cloud_t": row_of is not None,
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
    parser.add_argument("--bootstrap-block-days", type=int,
                        default=DEFAULT_BOOTSTRAP_BLOCK_DAYS, metavar="D",
                        help="Length of the contiguous time blocks the "
                             "confidence intervals resample (default "
                             f"{DEFAULT_BOOTSTRAP_BLOCK_DAYS}). Must exceed "
                             "the decorrelation time, ~3 days here; see the "
                             "note above block_index.")
    parser.add_argument("--bootstrap-samples", type=int,
                        default=DEFAULT_BOOTSTRAP_SAMPLES, metavar="N",
                        help=f"Bootstrap replicates (default "
                             f"{DEFAULT_BOOTSTRAP_SAMPLES}).")
    parser.add_argument("--with-cloud-temperature", action="store_true",
                        help="Also read the pressure-level archive and compute "
                             "the liquid-weighted mean cloud temperature, "
                             "which colours the LWP-against-DLR figure. OFF by "
                             "default: it is a second pass over a second "
                             "archive and only one figure uses it.")
    parser.add_argument("--lwp-regime-edges", type=float, nargs=2,
                        default=LWP_REGIME_EDGES_G, metavar=("G1", "G2"),
                        help="The two LWP thresholds, g m-2, splitting low / "
                             "medium / high for the stratified bar chart "
                             f"(default {LWP_REGIME_EDGES_G[0]:g} "
                             f"{LWP_REGIME_EDGES_G[1]:g}). Chosen where the "
                             "longwave emissivity changes, not from the data -- "
                             "see LWP_REGIME_EDGES_G.")
    parser.add_argument("--fit-orient",
                        choices=("panel",) + FIT_ORIENTS,
                        default=DEFAULT_FIT_ORIENT,
                        help="Which of the two least-squares fits the scatter "
                             "figures draw and annotate. 'panel' (default) "
                             "uses the direction each panel declares as "
                             "physically meaningful -- descriptive y-on-x for "
                             "LWP, flux-as-response x-on-y for DLR and for "
                             "T_skin-T_2m. 'both' draws both, which makes the "
                             "1/r^2 gap between them visible. The two are "
                             "DIFFERENT ESTIMATORS: inverting the wrong one "
                             "overstates the answer by 1/r^2. See FIT_ORIENTS.")
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

    def regime_acc(self, population: str | None = None) -> dict:
        """Moments stratified by (surface class, LWP regime).

        Index a group with ``slot * A.sec["n_regime"] + regime``, or use
        :func:`regime_slot`.
        """
        return self.sec["regime_mom"][population or self.args.population]

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

    use_step = steps_in_seasons(layout, keep_idx)
    cloud_t = None
    if args.with_cloud_temperature:
        if "sp" not in ds.data_vars:
            raise KeyError("cloud temperature needs 'sp' in the single-level "
                           "archive to clip levels at the ground")
        cloud_t = cloud_temperature_field(ds, args, use_step)
        # RESTRICT THE WHOLE ANALYSIS to the hours the pressure archive covers.
        # The alternative -- keep every hour and colour only some of them --
        # would put the fit, the density and the colour on three different
        # populations of the same panel, which is precisely the kind of
        # mismatch a comparison figure must not have. The pressure archive is
        # not continuous, so this can drop whole seasons; which ones survive is
        # printed below rather than left to be discovered.
        row_of, tcld = cloud_t
        have = np.flatnonzero(use_step)
        # A row exists for every wanted step; a step the pressure archive did
        # not supply leaves its row entirely NaN. That, not the row index, is
        # the test for "covered".
        supplied = np.isfinite(tcld).any(axis=(1, 2))
        matched = np.zeros_like(use_step)
        matched[have] = supplied[row_of[have]]
        if not matched.any():
            raise ValueError(
                "the pressure-level archive covers none of the requested "
                "seasons, so there is no cloud temperature to colour with. "
                "Choose seasons it covers, or drop with_cloud_temperature.")
        use_step = use_step & matched
        surviving = sorted({int(layout["seasons"][i])
                            for i in np.unique(layout["s_idx"][use_step])})
        dropped = [y for y in used if y not in surviving]
        used = surviving
        keep_idx = [layout["seasons"].index(y) for y in used]
        print(f"  Cloud temp : restricted to seasons {used}"
              + (f"; dropped {dropped} (no pressure-level data)"
                 if dropped else ""))

    sec = collect(ds, lsm, args, layout, keep_idx, panels,
                  DEFAULT_DLR_CURVE_EDGES, DEFAULT_SICONC_EDGES, phase_kw,
                  cloud_t=cloud_t, use_step=use_step)
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
def print_report(A: Analysis, population: str | None = None,
                 bootstrap_ci: bool = True) -> None:
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
        ("lwp_g_m2", "shf_W_m2", "sensible heat flux  vs  LWP"),
        ("lwd_W_m2", "shf_W_m2", "sensible heat flux  vs  DLR"),
        ("lwd_W_m2", "lhf_W_m2", "latent heat flux  vs  DLR"),
        ("dskt_t2m_K", "shf_W_m2", "sensible heat flux  vs  T_skin - T_2m"),
    ):
        xt = TRACKED[VAR_INDEX[x_key]]
        yt = TRACKED[VAR_INDEX[y_key]]
        x_u, y_u = plain_units(xt.units), plain_units(yt.units)
        # BOTH DIRECTIONS, ALWAYS, with the inverted slope printed beside the
        # correct one. The inverted column is not there to be used -- it is
        # there so the size of the error sits on the page next to the number it
        # would replace. See FIT_ORIENTS.
        print(f"\n  {title}")
        print(f"    {'':<20} {'':>12}{'-- dy/dx: y on x --':>21}"
              f"{'-- dx/dy: x on y --':>21}{'1/(dy/dx)':>12}")
        print(f"    {'class':<20} {'hours':>12}{'slope':>12}{'r2':>9}"
              f"{'slope':>13}{'infl':>8}{'  (NOT dx/dy)':>12}")
        for slot in PANEL_SLOTS + (ALL_SLOT,):
            f = fit_pair(acc, slot, x_key, y_key, weighted=True)
            yx, xy = f["y_on_x"], f["x_on_y"]
            naive = (1.0 / yx["slope"] if np.isfinite(yx["slope"])
                     and yx["slope"] != 0.0 else np.nan)
            print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20} "
                  f"{yx['n_hours']:>12,.0f}{yx['slope']:>12.4f}"
                  f"{f['r2']:>9.3f}{xy['slope']:>13.4f}"
                  f"{f['inflation']:>7.1f}x{naive:>12.2f}")
        print(f"      dy/dx in {y_u} per {x_u}  |  dx/dy in {x_u} per {y_u}  |  "
              f"the last column overstates dx/dy by 1/r2")
        print(f"      figures draw --fit-mode {fmode} "
              f"({'weighted' if FIT_IS_WEIGHTED[fmode] else 'unweighted'}), "
              f"--fit-orient {A.args.fit_orient}")

    # The matched comparison, in numbers. The pooled LWP regression above
    # returns r2 ~ 0.03; these rows are what that number is averaging over.
    n_reg = A.sec["n_regime"]
    racc = A.regime_acc(pop)
    print("\n  Mean sensible heat flux by LWP regime and surface class "
          "[W m-2, + into the surface]")
    edges = A.sec["regime_edges"]
    floor = lowest_drawn_lwp(A.phase_kw)
    heads = [f"{lab} ({a:g}-{b})" for lab, a, b in zip(
        LWP_REGIME_LABELS, [floor] + list(edges),
        [f"{e:g}" for e in edges] + ["inf"])]
    print(f"    {'class':<20}" + "".join(f"{h:>20}" for h in heads)
          + f"{'high - low':>13}")
    for slot in PANEL_SLOTS + (ALL_SLOT,):
        row = f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20}"
        vals = []
        for r in range(n_reg):
            g = regime_slot(A, slot, r)
            v = mean_of(racc, g, "shf_W_m2")
            vals.append(v)
            n = racc["n"][g]
            row += (f"{v:>11.1f}"
                    + (f"{n / 1e6:>7.1f}M" if n >= 1e6
                       else f"{n / 1e3:>7.0f}k" if n >= 1e3
                       else f"{n:>8.0f}"))
        row += f"{vals[-1] - vals[0]:>13.1f}"
        print(row)
    print("    each cell is the mean and the cell-hours behind it; the last "
          "column is the change across regimes,")
    print("    which is the conditional cloud effect at fixed surface type.")

    # THE HEADLINE ESTIMATE: both turbulent terms regressed ON DLR.
    for cname in dict.fromkeys(("none", A.args.control)):
        ctrl = CONTROL_SETS[cname]
        print(f"\n  Turbulent flux response to DLR   [W m-2 per W m-2, ERA5 "
              f"positive downward]   ({CONTROL_LABELS[cname]})")
        print(f"    {'class':<20}{'dSHF/dDLR':>11}{'r2':>8}"
              f"{'dLHF/dDLR':>11}{'r2':>8}{'sum':>11}{'hours':>14}")
        for slot in PANEL_SLOTS + (ALL_SLOT,):
            r = turbulent_response(acc, slot, control=ctrl)
            print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20}"
                  f"{r['dshf_dlwd']:>11.4f}{r['r2_shf']:>8.3f}"
                  f"{r['dlhf_dlwd']:>11.4f}{r['r2_lhf']:>8.3f}"
                  f"{r['dturb_dlwd']:>11.4f}{r['n_hours']:>14,.0f}")
        print("    POSITIVE means more DLR goes with more heat INTO the "
              "surface: the turbulent term")
        print("    ADDS to the radiative warming. Negative is the damping the "
              "bulk argument expects.")
        print("    r2 is the zero-order fit of that flux against DLR alone, so "
              "it does not change with the control.")

    if bootstrap_ci:
        try:
            bs = bootstrap_turbulent_response(A)
        except ValueError as exc:
            print(f"\n  No bootstrap intervals: {exc}")
        else:
            nb, nboot = (bs["dshf_dlwd"][ALL_SLOT]["n_block"],
                         bs["dshf_dlwd"][ALL_SLOT]["n_boot"])
            print(f"\n  95% confidence intervals, moving-block bootstrap "
                  f"({nb} blocks of {A.sec['block_days']} days, "
                  f"{nboot:,} replicates)")
            print(f"    {'class':<20}{'dSHF/dDLR 95% CI':>28}"
                  f"{'dLHF/dDLR 95% CI':>28}{'naive SE':>10}{'boot SE':>10}"
                  f"{'infl':>7}")
            for slot in PANEL_SLOTS + (ALL_SLOT,):
                a_, b_ = bs["dshf_dlwd"][slot], bs["dlhf_dlwd"][slot]
                ns = naive_se(acc, slot, "lwd_W_m2", "shf_W_m2")
                infl = a_["se"] / ns if ns and np.isfinite(ns) else np.nan
                star = "" if (a_["lo"] > 0) == (a_["hi"] > 0) else "  spans 0"
                print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20}"
                      f"{a_['value']:>9.4f} [{a_['lo']:+.4f},{a_['hi']:+.4f}]"
                      f"{b_['value']:>10.4f} [{b_['lo']:+.4f},{b_['hi']:+.4f}]"
                      f"{ns:>10.5f}{a_['se']:>10.5f}{infl:>6.0f}x{star}")
            print("    'naive SE' is the textbook OLS formula. It assumes "
                  "independent cell-hours, which these")
            print("    are not: DLR has a lag-1 hourly autocorrelation of 0.99 "
                  "and stays correlated across the")
            print("    whole domain, so the record holds tens of independent "
                  "weather systems, not millions of")
            print("    independent samples. The inflation factor is the price "
                  "of that assumption.")

    # SPECIFICATION CHECK: the bulk formulae are multiplicative in wind speed,
    # so U*Delta is the predictor the physics names. If ERA5's fluxes really
    # are bulk fluxes, that predictor should fit far better than Delta alone --
    # and the gap is a measure of how much of the "unexplained scatter" was
    # only ever wind speed.
    # The DLR ladder, in variance-explained terms: what does adding wind buy,
    # and why does it stop well short of the bulk specification?
    print("\n  Variance explained for SHF, by specification  [R2]")
    print(f"    {'class':<20}{'DLR':>8}{'DLR+U':>8}{'DLR+U+T2m':>11}"
          f"{'dT':>8}{'U.dT':>8}{'U,dT,U.dT':>11}")
    for slot in PANEL_SLOTS + (ALL_SLOT,):
        def r2(keys, sl=slot):
            return multiple_stats(acc, sl, "shf_W_m2", keys)["r2"]
        print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20}"
              f"{r2(('lwd_W_m2',)):>8.3f}{r2(('lwd_W_m2', 'wspd_m_s')):>8.3f}"
              f"{r2(('lwd_W_m2', 'wspd_m_s', 't2m_K')):>11.3f}"
              f"{r2(('dskt_t2m_K',)):>8.3f}{r2(('u_dskt_K_m_s',)):>8.3f}"
              f"{r2(('wspd_m_s', 'dskt_t2m_K', 'u_dskt_K_m_s')):>11.3f}")
    print("    Adding wind to the DLR fit helps, but cannot reach the bulk "
          "columns: DLR reaches the flux")
    print("    only through the skin temperature, whereas dT IS the driver. "
          "Closing the gap would mean")
    print("    putting the mediator into the regression, which destroys the "
          "DLR effect being estimated.")

    print("\n  Specification check: does the physically correct predictor fit "
          "better?")
    print(f"    {'class':<20}{'SHF~dT':>9}{'SHF~U.dT':>11}"
          f"{'LHF~dq':>10}{'LHF~U.dq':>11}   [r2]")
    for slot in PANEL_SLOTS + (ALL_SLOT,):
        a = moment_stats(acc, slot, "dskt_t2m_K", "shf_W_m2")["r2"]
        b = moment_stats(acc, slot, "u_dskt_K_m_s", "shf_W_m2")["r2"]
        c = moment_stats(acc, slot, "dq_g_kg", "lhf_W_m2")["r2"]
        d = moment_stats(acc, slot, "u_dq_g_kg_m_s", "lhf_W_m2")["r2"]
        print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20}{a:>9.3f}{b:>11.3f}"
              f"{c:>10.3f}{d:>11.3f}")
    print("    U.dT and U.dq are the bulk predictors: SH ~ rho c_p C_H U dT, "
          "LH ~ rho L_v C_E U dq.")
    print("    Where the second column is much larger than the first, the "
          "missing variance was wind speed,")
    print("    not missing physics -- and the slope on U.dT is rho*c_p*C_H "
          "with no wind left in the units.")

    print("\n  Control ladder: how far each slope moves when a confounder is "
          "held fixed")
    for name, cfg in CONTROL_LADDERS.items():
        print(f"    {cfg['title'].replace('$', '').replace(chr(92), ''):<44}"
              + "".join(f"{lab:>22}" for lab, _ in cfg["steps"]))
        for slot in PANEL_SLOTS:
            row = f"      {SLOT_LABELS[SLOT_ORDER[slot]]:<42}"
            for _, ctrl in cfg["steps"]:
                row += f"{partial_slope(acc, slot, cfg['y'], cfg['x'], ctrl):>22.4f}"
            print(row)
    print("    A slope that barely moves means that confounder was not doing "
          "much. T_2m and q_2m are")
    print("    MEDIATORS as well as confounders, so their rungs bound the "
          "answer from below.")

    # The bulk coupling coefficient, which is the one number from these
    # scatters that has a physical name and an independent value to check
    # against. Printed both ways so the 1/r^2 error is on the page beside the
    # number it would replace, rather than left for a reader to rediscover.
    print("\n  Bulk coupling coefficient  rho*c_p*C_H*U = "
          "-d(SHF)/d(T_skin - T_2m)   [W m-2 K-1]")
    print(f"    {'class':<20} {'correct':>10} {'from 1/slope':>14} "
          f"{'inflation':>11} {'r2':>8}")
    for slot in PANEL_SLOTS + (ALL_SLOT,):
        f = fit_pair(acc, slot, "dskt_t2m_K", "shf_W_m2", weighted=True)
        good = -f["y_on_x"]["slope"]
        bad = (-1.0 / f["x_on_y"]["slope"]
               if np.isfinite(f["x_on_y"]["slope"]) and f["x_on_y"]["slope"]
               else np.nan)
        print(f"    {SLOT_LABELS[SLOT_ORDER[slot]]:<20} {good:>10.1f} "
              f"{bad:>14.1f} {f['inflation']:>10.1f}x {f['r2']:>8.3f}")
    print("    'correct' regresses SHF on dT -- the panel's own y-on-x fit; "
          "'from 1/slope' inverts the")
    print("    dT-on-SHF fit instead and is wrong by 1/r2.")
    print("    rho*c_p ~ 1300 J m-3 K-1, so 10 W m-2 K-1 is C_H*U ~ 0.008 m s-1 "
          "-- e.g. C_H = 1.3e-3 at U = 6 m s-1,")
    print("    a stable Arctic boundary layer. The inverted column is uniform "
          "across surfaces; the correct one is not.")

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
# means more energy moved than arrived, which happens where the surface cannot
# respond and the regression is describing the air mass instead. Used to set
# readable axis limits, never to hide a bar.
PARTITION_SANE_MAX = 2.5

# Above this magnitude d(flux)/d(DLR) is no longer a surface response: more
# energy is moving than the radiation delivered, which happens where the
# surface cannot respond and the regression tracks the air mass instead.
TURBULENT_SANE_MAX = 0.6

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
ORIENT_LABELS: dict[str, str] = {
    "y_on_x": "regression of y on x",
    "x_on_y": "regression of x on y",
}

# Opaque, not translucent: a fit line seen THROUGH the box changes colour
# where it passes behind, which reads as two different lines.
NOTE_BOX = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95,
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
                  title_fs: float = TITLE_FS,
                  axes_title_pts: float = 0.0) -> float:
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
        # A multi-line note needs a real gap under the subtitle: at the single
        # line these blocks were written for, 0.05 in was enough, but the
        # subtitle's own descenders reach into it once the note is tall enough
        # to sit close.
        y -= (0.05 + 0.06 * note.count("\n")) / fig_h
        fig.text(0.5, y, note, ha="center", va="top", fontsize=NOTE_FS,
                 color="#8a5a00", style="italic")
        y -= note_h
    # An axes title is drawn ABOVE the axes rectangle, so subplots_adjust(top=)
    # does not reserve space for it and the caller has to say how tall it is.
    # tight_layout() accounts for titles itself, so those callers pass nothing.
    y -= axes_title_pts * 1.9 / 72.0 / fig_h
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


ORIENT_LINE_STYLE: dict[str, dict] = {
    # y-on-x keeps the established look; x-on-y is drawn in a different colour
    # so a figure showing both cannot be misread as one line with a kink.
    "y_on_x": dict(color="#B2182B", lw=1.8, ls="--"),
    "x_on_y": dict(color="#000000", lw=1.6, ls="-"),
}


def _fit_line_xy(stats: dict, orient: str, panel: Panel2D,
                 x_span: tuple[float, float], y_span: tuple[float, float]):
    """Endpoints of one fitted line, in plot coordinates.

    For ``y_on_x`` the fit is y = a + b x and is swept over the occupied x
    span. For ``x_on_y`` the fit is x = a + b y -- a regression whose PREDICTOR
    is the vertical axis -- so it is swept over the occupied y span and the
    result is still drawn on the same axes. Sweeping x for an x-on-y fit and
    inverting the slope is exactly the mistake the whole orientation option
    exists to prevent.
    """
    if not np.isfinite(stats["slope"]):
        return None
    if orient == "y_on_x":
        xs = np.array(x_span)
        return xs, stats["intercept"] + stats["slope"] * xs
    ys = np.array(y_span)
    return stats["intercept"] + stats["slope"] * ys, ys


def _slope_text(stats: dict, orient: str, panel: Panel2D) -> str:
    """The annotated slope, with the units of the direction actually fitted."""
    x_u = plain_units(TRACKED[VAR_INDEX[panel.x_key]].units)
    y_u = plain_units(TRACKED[VAR_INDEX[panel.y_key]].units)
    if orient == "y_on_x":
        # The equation form, which is how a scatter of this kind is normally
        # reported and what a published figure will state for comparison.
        return (f"y = {stats['slope']:.3g}x + {stats['intercept']:.2f}"
                f"   [{y_u} per {x_u}]")
    return f"dx/dy = {stats['slope']:+.3g} {x_u} / ({y_u})"


def _occupied_span(h: np.ndarray, panel: Panel2D):
    """The x and y ranges the panel's data actually occupy.

    Fits and overlays are swept across this rather than the full axis, so a
    line never makes a claim about a region the class never visits.
    """
    xc = _bin_centers(panel.x_range[0], panel.x_range[1], panel.x_bins)
    yc = _bin_centers(panel.y_range[0], panel.y_range[1], panel.y_bins)
    occ = h > 0
    if not occ.any():
        return panel.x_range, panel.y_range
    xi, yi = np.nonzero(occ)
    return (xc[xi.min()], xc[xi.max()]), (yc[yi.min()], yc[yi.max()])


def _density_panel(ax, h: np.ndarray, panel: Panel2D, fits: dict,
                   label: str, color: str, out_weight: float,
                   fit_mode: str = DEFAULT_FIT_MODE,
                   fit_orient: str = "y_on_x",
                   marker_size: float = 3.0,
                   colour: np.ndarray | None = None,
                   colour_norm: tuple[float, float] | None = None,
                   colour_cmap: str = "viridis",
                   note_loc: str = "upper left",
                   overlay_lines=None, extra_note=None,
                   draw_fit: bool = True):
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
        if colour is None:
            sm = ax.scatter(xc[xi][order], yc[yi][order], c=frac[order],
                            s=marker_size, cmap=DENSITY_CMAP, linewidths=0,
                            norm=norm)
        else:
            # Colour carries a THIRD variable's mean per bin, and density moves
            # to opacity. Keeping density visible matters: a bin holding two
            # cell-hours and one holding twenty thousand would otherwise be the
            # same solid dot, and the eye would read the sparse tail as
            # structure.
            from matplotlib.colors import Normalize
            cv = colour[xi, yi][order]
            good = np.isfinite(cv)
            lo, hi = colour_norm if colour_norm else (np.nanmin(cv),
                                                      np.nanmax(cv))
            alpha = 0.12 + 0.88 * (np.log10(frac[order]) - np.log10(DENSITY_VMIN)) \
                / (-np.log10(DENSITY_VMIN))
            sm = ax.scatter(xc[xi][order][good], yc[yi][order][good],
                            c=cv[good], s=marker_size, cmap=colour_cmap,
                            linewidths=0, norm=Normalize(vmin=lo, vmax=hi),
                            alpha=np.clip(alpha[good], 0.08, 1.0))

        if fit_mode == "default":
            xb, ymean = _binned_mean_y(h, panel)
            ax.plot(xb, ymean, color=BINNED_MEAN_COLOR, lw=2.4, alpha=0.85,
                    solid_capstyle="round", zorder=4)
            ax.plot(xb, ymean, color="#222222", lw=1.1, zorder=5,
                    label="mean y per x bin")

    # Fits are swept only across the range the data actually occupy. Extending
    # a line over an axis the class never visits -- Utqiagvik spans about
    # 30 W m-2 of an axis 310 wide -- makes a slope fitted to a sliver look like
    # a claim about the whole panel.
    if occupied.any():
        x_span = (xc[xi.min()], xc[xi.max()])
        y_span = (yc[yi.min()], yc[yi.max()])
    else:
        x_span, y_span = panel.x_range, panel.y_range

    drawn = ["y_on_x", "x_on_y"] if fit_orient == "both" else [fit_orient]
    # ``draw_fit=False`` keeps the fit's NUMBERS in the annotation but leaves
    # its line off the panel, for a figure whose subject is a different fit and
    # where a second line would only invite the two to be confused.
    for orient in (drawn if draw_fit else []):
        line = _fit_line_xy(fits[orient], orient, panel, x_span, y_span)
        if line is None:
            continue
        lx, ly = line
        style = dict(ORIENT_LINE_STYLE[orient])
        if fit_mode == "regression" and fit_orient != "both":
            # Thin, solid, black, and alone: the conventional rendering, and
            # with the binned mean removed there is nothing else on the panel
            # for it to be confused with.
            style = dict(color=REGRESSION_COLOR, lw=1.2, ls="-")
        ax.plot(lx, ly, zorder=6, label=ORIENT_LABELS[orient], **style)

    # Extra lines the caller wants in the panel's own coordinates -- e.g. a
    # multiple-regression prediction evaluated at several values of a control,
    # which is what "holding wind fixed" looks like on a two-dimensional plot.
    for lx, ly, kw in (overlay_lines or []):
        ax.plot(lx, ly, zorder=7, **kw)

    ax.set_xlim(*panel.x_range)
    ax.set_ylim(*panel.y_range)
    ax.axvline(0.0, color="#444444", lw=0.7, ls=":", zorder=1)
    ax.set_title(label, fontsize=10.5, color=color, fontweight="bold", pad=18)

    # Sample-size context (cell-hours, fraction outside the axes) lives in a
    # subtitle rather than the fit box below -- it describes the PANEL, not
    # the fit, and crowds the corner annotation otherwise.
    ref = fits[drawn[0]]
    sub_bits = [f"{ref['n_hours']:,.0f} cell-hours"]  # panel context, always
    if out_weight > 0.005:
        sub_bits.append(f"{100 * out_weight:.1f}% outside axes")
    ax.annotate("  |  ".join(sub_bits), xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.2, color="#555555")

    # r^2 is symmetric, so it is annotated once no matter how many fits are
    # drawn. In "both" mode the inflation factor is spelled out, because that
    # number IS the reason the two lines differ.
    # When the fit's line is not drawn, its equation does not belong in the box
    # either: a number in the annotation with no line on the panel is exactly
    # the ambiguity suppressing the line was meant to remove.
    lines = []
    if draw_fit:
        lines = [_slope_text(fits[o], o, panel) for o in drawn]
        lines.append(f"$r^2$ = {fits['r2']:.3f}   (r = {ref['r']:+.3f})")
    lines.extend(extra_note or [])
    if fit_orient == "both" and np.isfinite(fits["inflation"]):
        lines.append(f"inverting dy/dx overstates dx/dy "
                     f"{fits['inflation']:.1f}$\\times$")
    note = "\n".join(lines)
    note_xy, note_va, note_ha = {
        "upper left": ((0.03, 0.965), "top", "left"),
        "lower right": ((0.97, 0.035), "bottom", "right"),
        "lower left": ((0.03, 0.035), "bottom", "left"),
    }[note_loc]
    ax.text(*note_xy, note, transform=ax.transAxes, va=note_va, ha=note_ha,
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
                    fit_mode: str | None = None,
                    fit_orient: str | None = None,
                    note_loc: str | None = None,
                    show_legend: bool = True,
                    overlay=None, stem_suffix: str = "",
                    draw_fit: bool = True):
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
    # "panel" defers to the direction the panel declares as physically
    # meaningful; anything else overrides it for every panel in the figure.
    orient = fit_orient or A.args.fit_orient
    if orient == "panel":
        orient = panel.fit_orient
    if orient not in FIT_ORIENTS:
        raise ValueError(f"unknown fit_orient {orient!r}; choose from "
                         f"{list(FIT_ORIENTS) + ['panel']}")
    h_all = A.sec["hist"][panel_name]
    n_r, n_c = A.args.layout

    # Colour by a third variable's per-bin mean, when the panel asks for one
    # and the run actually computed it.
    colour_all = colour_norm = None
    if panel.color_key and panel_name in A.sec.get("hist_colour_sum", {}):
        cs = A.sec["hist_colour_sum"][panel_name]
        cw = A.sec["hist_colour_weight"][panel_name]
        with np.errstate(invalid="ignore", divide="ignore"):
            colour_all = np.where(cw > 0, cs / np.where(cw > 0, cw, 1.0), np.nan)
        if np.isfinite(colour_all).any():
            # ONE norm across every panel: the colour is a physical quantity,
            # not a within-panel rank, so a per-panel scale would make the same
            # colour mean different temperatures in different panels.
            lo = float(np.nanpercentile(colour_all, 0.5))
            hi = float(np.nanpercentile(colour_all, 99.5))
            colour_norm = (np.floor(lo / 5) * 5, np.ceil(hi / 5) * 5)
        else:
            colour_all = None

    fig, axes = plt.subplots(n_r, n_c, figsize=(4.1 * n_c, 3.6 * n_r),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    sm = None
    for ax, slot in zip(axes, PANEL_SLOTS):
        name = SLOT_ORDER[slot]
        fits = fit_pair(acc, slot, panel.x_key, panel.y_key, weighted=weighted)
        w_tot = acc["w"][slot]
        out_frac = (A.sec["hist_out"][panel_name][slot] / w_tot
                    if w_tot > 0 else 0.0)
        ov_lines, ov_note = (
            overlay(A, acc, slot, panel, _occupied_span(h_all[slot], panel)[0])
            if overlay is not None else ([], []))
        s = _density_panel(ax, h_all[slot], panel, fits,
                           SLOT_LABELS[name], SLOT_COLORS[name], out_frac,
                           fit_mode=fmode, fit_orient=orient,
                           colour=None if colour_all is None
                           else colour_all[slot],
                           colour_norm=colour_norm,
                           note_loc=(note_loc or
                                     ("lower right" if panel_name == "lwp_dlr"
                                      else "upper left")),
                           overlay_lines=ov_lines, extra_note=ov_note,
                           draw_fit=draw_fit)
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

    if show_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", ncol=2, frameon=False,
                       fontsize=9, bbox_to_anchor=(0.5, -0.035))

    what = {
        "y_on_x": "y on x (descriptive)",
        "x_on_y": "x on y - the flux as the response",
        "both": "both directions; they differ by 1/$r^2$ and answer "
                "different questions",
    }[orient]
    subtitle = (_figure_subtitle(A, pop)
                + f"\nfit: {FIT_LINE_LABELS[fmode]}, {what}")
    # Header first, so the colourbar below sizes itself against the panel
    # rectangle the block actually leaves. axes_title_pts is 18, not the 10.5pt
    # panel title font, because each panel title now sits above a cell-hours
    # subtitle line (see _density_panel) and needs the taller reserved band.
    fig.subplots_adjust(top=_header_block(fig, title, subtitle,
                                          axes_title_pts=18.0))
    if sm is not None:
        cb = fig.colorbar(sm, ax=axes[:len(PANEL_SLOTS)].tolist(),
                          fraction=0.022, pad=0.015)
        if colour_all is None:
            cb.set_label("area-weighted frequency,\nrelative to the densest "
                         "bin in the panel", fontsize=8.5)
        else:
            lab, unit = EXTRA_FIELDS[panel.color_key]
            cb.set_label(f"{lab} ({unit})\n"
                         f"opacity $\\propto$ log(frequency)", fontsize=8.5)
    return _save(fig, A, f"{stem}{stem_suffix}_{fmode}_{orient}",
                 out_dir, dpi)


def fig_shf_vs_lwp(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None, fit_orient: str | None = None):
    """Liquid water path against sensible heat flux, by surface class.

    The first of the two the request asks for. Read it as a joint distribution,
    not a causal chain: LWP does not drive SHF directly, it drives DLR, which
    drives T_skin, which drives SHF. The straight-line fit is therefore a
    summary of a two-step relation and is not expected to be tight -- what it
    is good for is the SIGN and the contrast between classes.
    """
    return _scatter_figure(
        A, "lwp_shf",
        "Surface sensible heat flux against liquid water path",
        "shf_vs_lwp", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower right", show_legend=False)


def fig_shf_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None, fit_orient: str | None = None):
    """Downwelling longwave against sensible heat flux, by surface class.

    The same population as ``fig_shf_vs_lwp`` with the intermediate variable
    substituted in: DLR is what the cloud actually delivers to the surface, so
    this panel is one step closer to the mechanism and the relation is
    correspondingly tighter.
    """
    return _scatter_figure(
        A, "dlr_shf",
        "Surface sensible heat flux against downwelling longwave",
        "shf_vs_dlr", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower right", show_legend=False)


def fig_lwu_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                   fit_mode: str | None = None, fit_orient: str | None = None):
    """Upwelling longwave against downwelling longwave, by surface class.

    Figure 2 with the radiative term substituted in. The fitted slope is
    f_LWU, the first term of the DLR partition, and it is the one term of the
    five whose panel is close to a straight line -- so its slope is a
    coefficient rather than a summary of a curve.
    """
    return _scatter_figure(
        A, "dlr_lwu",
        "Upwelling longwave against downwelling longwave",
        "lwu_vs_dlr", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower right", show_legend=False)


def fig_swnet_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                     fit_mode: str | None = None,
                     fit_orient: str | None = None):
    """Net shortwave against downwelling longwave, by surface class.

    The fitted slope is -f_SW. Read the panel before the number: the mass sits
    on SW_net = 0 through the dark half of the season, so the line is drawn
    through a distribution that is mostly a single point, and r^2 runs 0.000 to
    0.011. There is no mechanism here -- see the polar-night check in the
    notebook -- only the covariance of cloud with itself.
    """
    return _scatter_figure(
        A, "dlr_swnet",
        "Net surface shortwave against downwelling longwave",
        "swnet_vs_dlr", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="upper left", show_legend=False)


def fig_rnet_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                    fit_mode: str | None = None,
                    fit_orient: str | None = None):
    """Net surface flux against downwelling longwave, by surface class.

    R = LWD - LWU + SW_net + SH + LH, positive into the surface: what the skin
    has left after the four exchanges with the atmosphere, and therefore what
    is conducted or stored below.

    THE FITTED SLOPE ON THIS PANEL IS f_res, AND IT IS FITTED HERE DIRECTLY.
    The partition defines f_res as the remainder of the other four fractions;
    this panel regresses an independently accumulated column on DLR and gets
    the same number, which is checked in ``self_check``. It is the difference
    between a residual and a leftover.
    """
    return _scatter_figure(
        A, "dlr_rnet",
        "Net surface flux against downwelling longwave",
        "rnet_vs_dlr", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower left", show_legend=False)


def fig_shf_vs_dskt(A: Analysis, out_dir=None, dpi: int | None = None,
                 fit_mode: str | None = None, fit_orient: str | None = None):
    """(T_skin - T_2m) against sensible heat flux: the bulk relation itself.

    Not requested, but it is the premise the other two figures rest on, and it
    costs nothing to draw from the same pass. If ERA5's sensible heat flux is
    the bulk flux the argument assumes, this panel is a line through the origin
    whose slope is rho*c_p*C_H*U -- and the scatter about it is the wind-speed
    and stability dependence that the LWP and DLR panels inherit.
    """
    return _scatter_figure(
        A, "dskt_shf",
        "Surface sensible heat flux against the skin-to-air temperature "
        "difference",
        "shf_vs_dskt", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower left", show_legend=False)


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
        ax_b.annotate("off scale:\nsurface\nprescribed", (xi, 0.42),
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
# Figures 13 and 14: the partition on its own, term by term
# ----------------------------------------------------------------------------
def partition_closure(acc: dict, slot: int,
                      control: tuple[str, ...] = ()) -> dict:
    """``f_res`` by all three routes that reach it, and the gaps between them.

    ``f_res``        the remainder of the other four fractions, as ``partition``
                     defines it -- four separate least-squares solves, summed.
    ``f_res_combo``  the same linear combination formed INSIDE the covariance
                     matrix and solved once (``combo_slope``). A different
                     matrix solve; agreement is guaranteed in exact arithmetic
                     and checks the implementation, not a physical claim.
    ``f_res_direct`` the regression of the accumulated ``rnet_W_m2`` column on
                     DLR. This one is INDEPENDENT: R is built per sample during
                     the streaming pass and carries its own moments, so nothing
                     about this number is guaranteed by the algebra of the
                     other four.
    """
    p = partition(acc, slot, control=control)
    combo = net_flux_slope(acc, slot, control=control)
    direct = partial_slope(acc, slot, "rnet_W_m2", "lwd_W_m2", control)
    return {"f_res": p["f_res"], "f_res_combo": combo,
            "f_res_direct": direct,
            "gap": p["f_res"] - direct,
            "gap_combo": p["f_res"] - combo,
            "sum": sum(p[k] for k, _, _ in PARTITION_TERMS)}


def fig_partition_terms(A: Analysis, out_dir=None, dpi: int | None = None,
                        population: str | None = None,
                        control: str | None = None):
    """The five partition fractions as grouped bars, with the closure check.

    The companion to the five scatter panels -- ``fig_shf_vs_dlr``,
    ``fig_lhf_vs_dlr``, ``fig_lwu_vs_dlr``, ``fig_swnet_vs_dlr`` and
    ``fig_rnet_vs_dlr`` -- each of which fits one term of this bar.

    GROUPED, NOT STACKED. Stacking hides the sign of a term, and over open
    water the sign is the whole story: three of the five fractions are negative
    there, which is why the residual runs past three. The stacked version is
    panel (b) of ``fig_response_partition``, kept because it is the readable
    form wherever the negatives are small.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    cname = control or A.args.control
    ctrl = CONTROL_SETS[cname]
    acc = A.acc(pop)
    slots = list(PANEL_SLOTS)
    names = [SLOT_ORDER[s] for s in slots]
    parts = [partition(acc, s, control=ctrl) for s in slots]
    closes = [partition_closure(acc, s, control=ctrl) for s in slots]

    fig, ax_d = plt.subplots(figsize=(10.6, 5.6))

    x = np.arange(len(slots))
    n_term = len(PARTITION_TERMS)
    bw = 0.8 / n_term
    for j, (key, term_label, color) in enumerate(PARTITION_TERMS):
        v = np.array([p[key] for p in parts])
        off = (j - (n_term - 1) / 2) * bw
        ax_d.bar(x + off, v, bw * 0.92, color=color, edgecolor="#333333",
                 linewidth=0.5, label=term_label)
        for xi, vi in zip(x, v):
            if abs(vi) > 0.20:
                ax_d.annotate(f"{vi:.2f}", (xi + off, vi),
                              textcoords="offset points",
                              xytext=(0, 3 if vi >= 0 else -14), ha="center",
                              fontsize=6.8, rotation=90)
    ax_d.axhline(0.0, color="#333333", lw=0.8)
    stack = np.array([[p[k] for k, _, _ in PARTITION_TERMS] for p in parts])
    lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
    pad = 0.12 * (hi - lo)
    ax_d.set_ylim(min(lo - pad, -0.1), max(hi + pad, 1.25))
    ax_d.set_ylabel("fraction of $d(LWD)$")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([SLOT_SHORT[n] for n in names], fontsize=9)
    ax_d.grid(axis="y", alpha=0.2, lw=0.5)
    ax_d.legend(fontsize=8.5, ncol=5, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.10))

    # The closure, stated on the figure rather than left to the reader's
    # arithmetic. The DIRECT number is the one worth reporting: R is
    # accumulated per sample, so its regression on DLR is not constrained by
    # the algebra that defines f_res as a remainder.
    gap = max(abs(c["gap"]) for c in closes if np.isfinite(c["gap"]))
    gapc = max(abs(c["gap_combo"]) for c in closes
               if np.isfinite(c["gap_combo"]))
    dsum = max(abs(c["sum"] - 1.0) for c in closes if np.isfinite(c["sum"]))
    ax_d.annotate(
        "closure, over the six groups:\n"
        f"max |sum of the five $-$ 1|  =  {dsum:.1e}\n"
        f"max |$f_{{res}}$ $-$ fit of $R$ on DLR|  =  {gap:.1e}   "
        "(independent column)\n"
        f"max |$f_{{res}}$ $-$ same combination in the covariance|  = "
        f" {gapc:.1e}",
        (0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
        fontsize=7.8, bbox=NOTE_BOX, zorder=7)
    if min(min(p[k] for k, _, _ in PARTITION_TERMS) for p in parts) < -0.1:
        ax_d.annotate("bars below zero: that channel removes LESS\n"
                      "than before, so the surface keeps more and\n"
                      "the residual must exceed 1 to close",
                      (0.985, 0.03), xycoords="axes fraction", ha="right",
                      va="bottom", fontsize=7.6, bbox=NOTE_BOX, zorder=7)

    top = _header_block(
        fig, "Where each additional W m$^{-2}$ of DLR goes, term by term",
        _figure_subtitle(A, pop),
        note=(f"{CONTROL_LABELS[cname]}; each bar is the slope of one scatter "
              "figure: LWU, SH, LH, SW$_{net}$ and R, each fitted on DLR\n"
              "slopes are d/d(DLR) across synoptic variability, not a "
              "controlled perturbation"),
        title_fs=14,
    )
    fig.subplots_adjust(top=top - 0.02, bottom=0.16, left=0.09, right=0.985)
    return _save(fig, A, f"dlr_partition_terms_{pop}_{cname}", out_dir, dpi)


def fig_partition_ledger(A: Analysis, out_dir=None, dpi: int | None = None,
                         population: str | None = None,
                         control: str | None = None,
                         alt_control: str | None = None):
    """The same five numbers as a ledger that sums to one for every class.

    The stacked partition fails over open water for a presentational reason,
    not a numerical one: three of the five fractions come out negative there,
    so the stack only closes because the residual overshoots past three.
    Splitting the terms by sign fixes it. In an anomaly budget a negative term
    is a SOURCE -- not because energy arrives that did not before, but because
    a loss weakened -- and it belongs on the supply side beside the DLR anomaly
    itself; positive terms are SINKS.
    Normalising both sides by the same gross total G (see ``partition_ledger``)
    makes each side sum to one for every class, open water included.

    Panels (a) and (b) are the same ledger under the single-variable regression
    and under the multiple regression, so the two estimators can be compared
    bar for bar. Panel (c) carries G itself, which is the one number that says
    how much of the co-varying energy the DLR anomaly actually is.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    cname = control or A.args.control
    aname = alt_control or ALT_CONTROL[cname]
    acc = A.acc(pop)
    slots = list(PANEL_SLOTS)
    labels = [SLOT_LABELS[SLOT_ORDER[s]] for s in slots]
    x = np.arange(len(slots))
    bw = 0.36

    # Two rows rather than three panels abreast: six groups of stacked bars
    # need about an inch of width each before the tick labels collide, and the
    # gross-factor panel reads better full width anyway, since it is the one
    # panel where the classes are meant to be compared against each other.
    fig = plt.figure(figsize=(13.6, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.42,
                          wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    order = [k for k, _, _ in PARTITION_TERMS]
    color_of = {k: c for k, _, c in PARTITION_TERMS}
    label_of = {k: lab for k, lab, _ in PARTITION_TERMS}

    def _ledger_panel(ax, name, title):
        ctrl = CONTROL_SETS[name]
        led = [partition_ledger(partition(acc, s, control=ctrl)) for s in slots]
        for side, off, hatch in (("supply", -bw / 2, "///"),
                                 ("disposal", bw / 2, None)):
            bottom = np.zeros(len(slots))
            for key in ["dlr"] + order:
                v = np.array([L[side].get(key, 0.0) for L in led])
                if not np.any(np.abs(v) > 1e-12):
                    continue
                col = LEDGER_DLR_COLOR if key == "dlr" else color_of[key]
                lab = LEDGER_DLR_LABEL if key == "dlr" else label_of[key]
                ax.bar(x + off, v, bw, bottom=bottom, color=col,
                       edgecolor="white", linewidth=0.7,
                       hatch=hatch if key != "dlr" else None)
                del lab
                for xi, vi, bi in zip(x, v, bottom):
                    if vi > 0.10:
                        ax.annotate(f"{vi:.2f}", (xi + off, bi + vi / 2),
                                    ha="center", va="center", fontsize=7.2,
                                    color="white" if col != "#BBBBBB"
                                    else "#333333")
                bottom = bottom + v
        for xi, L in zip(x, led):
            ax.annotate(f"{L['gross']:.2f}", (xi, 1.015), ha="center",
                        va="bottom", fontsize=7.8, fontweight="bold",
                        color="#333333")
        ax.axhline(1.0, color="#333333", lw=1.0, ls="--")
        ax.set_ylim(0.0, 1.20)
        ax.set_ylabel("share of the gross energy $G$\n"
                      "left bar: supply (hatched);  right bar: disposal")
        ax.set_title(title + "\nnumber above each pair is $G$", fontsize=10.5,
                     loc="left", fontweight="bold")
        return led

    def _panel_title(letter: str, name: str) -> str:
        kind = ("single-variable regression on DLR" if not CONTROL_SETS[name]
                else "multiple regression: the DLR coefficient")
        return f"({letter})  {kind}, {CONTROL_LABELS[name]}"

    _ledger_panel(ax_a, cname, _panel_title("a", cname))
    _ledger_panel(ax_b, aname, _panel_title("b", aname))

    for ax in (ax_a, ax_b):
        ax.set_xticks(x)
        ax.set_xticklabels([SLOT_SHORT[SLOT_ORDER[s_]] for s_ in slots],
                           fontsize=8.2)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
    # Proxy handles rather than the bars' own labels: a term that is a source
    # in one panel and a sink in the other would otherwise appear twice, or
    # once with whichever hatch it happened to be drawn with first.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=LEDGER_DLR_COLOR, label=LEDGER_DLR_LABEL)]
    handles += [Patch(facecolor=c, label=lab) for _, lab, c in PARTITION_TERMS]
    handles += [Patch(facecolor="white", edgecolor="#333333", hatch="///",
                      label="hatched = supply side")]
    ax_a.legend(handles=handles, fontsize=8.2, ncol=4, frameon=False,
                loc="upper center", bbox_to_anchor=(1.13, -0.10))

    # (c) the gross factor: how much of the moving energy is the DLR anomaly --
    cnames = [n for n in CONTROL_SETS]
    gw = 0.8 / len(cnames)
    g_max = 1.0
    for j, n in enumerate(cnames):
        ctrl = CONTROL_SETS[n]
        g = np.array([partition_ledger(partition(acc, s, control=ctrl))["gross"]
                      for s in slots])
        off = (j - (len(cnames) - 1) / 2) * gw
        ax_c.bar(x + off, g, gw * 0.9, edgecolor="#333333", linewidth=0.5,
                 color=matplotlib.colormaps["Greys"](0.30 + 0.25 * j),
                 label=CONTROL_LABELS[n])
        for xi, gi in zip(x, g):
            if np.isfinite(gi):
                ax_c.annotate(f"{gi:.2f}", (xi + off, gi),
                              textcoords="offset points", xytext=(0, 2),
                              ha="center", fontsize=6.8, rotation=90)
        g_max = max(g_max, float(np.nanmax(g)))
    ax_c.set_ylim(0.0, g_max * 1.32)
    ax_c.axhline(1.0, color="#B2182B", lw=1.1, ls="--")
    ax_c.annotate("$G = 1$: the DLR anomaly is the only energy moving,\n"
                  "and the stacked partition means what it says",
                  (0.985, 0.42), xycoords="axes fraction", ha="right",
                  va="top", fontsize=8.0, color="#B2182B", bbox=NOTE_BOX,
                  zorder=7)
    ax_c.set_ylabel("gross energy $G$ per W m$^{-2}$ of DLR\n"
                    "[W m$^{-2}$ changing hands]")
    ax_c.set_title("(c)  How much energy co-varies with each W m$^{-2}$ of "
                   "DLR, and how much of it the DLR anomaly is",
                   fontsize=11, loc="left", fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=8.5)
    ax_c.grid(axis="y", alpha=0.2, lw=0.5)
    ax_c.legend(fontsize=8.0, frameon=False, loc="upper right", ncol=3)

    top = _header_block(
        fig, "The DLR partition as a ledger that closes for every surface",
        _figure_subtitle(A, pop),
        note=("negative fractions are re-read as SUPPLY: this is an ANOMALY "
              "budget, in which a loss that weakens looks the same as a gain\n"
              "over open water the upward turbulent loss shrinks from -105 to "
              "-13 W m-2 across the DLR range without reversing\n"
              "both sides are normalised by the same gross total G, so each "
              "sums to one for every class; the estimates are untouched, only "
              "their presentation"),
        title_fs=14,
    )
    fig.subplots_adjust(top=top - 0.03, bottom=0.06, left=0.075, right=0.985)
    return _save(fig, A, f"dlr_partition_ledger_{pop}_{cname}", out_dir, dpi)


def print_partition_detail(A: Analysis, population: str | None = None,
                           controls: tuple[str, ...] | None = None) -> None:
    """The numbers behind figures 13 and 14, per class and per control set.

    Prints the five fractions, their sum, the residual fitted directly on the
    net flux, the gross factor G, and the normalised supply shares -- so the
    single-variable and multiple-regression partitions can be compared line by
    line rather than by eye across two panels.
    """
    pop = population or A.args.population
    acc = A.acc(pop)
    cnames = tuple(controls) if controls else tuple(CONTROL_SETS)
    keys = [k for k, _, _ in PARTITION_TERMS]

    for cname in cnames:
        ctrl = CONTROL_SETS[cname]
        print()
        print(f"{CONTROL_LABELS[cname]}  (control = {cname}, "
              f"{'single-variable' if not ctrl else 'multiple'} regression"
              + (f" on DLR + {', '.join(ctrl)}" if ctrl else " on DLR") + ")")
        print(f"  {'class':<22}" + "".join(f"{k[2:]:>10}" for k in keys)
              + f"{'sum':>9}{'R on DLR':>11}{'gap':>10}{'G':>8}"
              + f"{'DLR/G':>8}")
        for slot in PANEL_SLOTS:
            p = partition(acc, slot, control=ctrl)
            c = partition_closure(acc, slot, control=ctrl)
            L = partition_ledger(p)
            print(f"  {SLOT_LABELS[SLOT_ORDER[slot]]:<22}"
                  + "".join(f"{p[k]:>10.3f}" for k in keys)
                  + f"{c['sum']:>9.5f}{c['f_res_direct']:>11.3f}"
                  + f"{c['gap']:>10.1e}{L['gross']:>8.2f}"
                  + f"{L['supply']['dlr']:>8.2f}")


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
              "open water: ERA5's ocean surface is prescribed, so it\n"
              "cannot respond hourly; these coefficients describe the\n"
              "air mass co-varying with DLR, not a partition of it",
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


def fig_lhf_vs_dlr(A: Analysis, out_dir=None, dpi: int | None = None,
                   fit_mode: str | None = None, fit_orient: str | None = None):
    """Downwelling longwave against LATENT heat flux, by surface class.

    The companion to ``fig_shf_vs_dlr``. Fitted x on y, so the annotated slope
    is d(LHF)/d(DLR) directly -- the second of the two turbulent terms, and the
    one the sensible-heat argument leaves out. Over open water the latent term
    is the larger of the two in the mean, so a turbulent-flux budget that
    stops at sensible heat is incomplete there.
    """
    return _scatter_figure(
        A, "dlr_lhf",
        "Surface latent heat flux against downwelling longwave",
        "lhf_vs_dlr", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient,
        note_loc="lower right", show_legend=False)


def fig_turbulent_response(A: Analysis, out_dir=None, dpi: int | None = None,
                           population: str | None = None,
                           control: str | None = None,
                           bootstrap_ci: bool = True,
                           n_boot: int | None = None):
    """d(SHF)/d(DLR) and d(LHF)/d(DLR) by surface type, side by side.

    The headline estimate, with both turbulent terms shown together because
    they are the two halves of one flux and can have opposite signs. Bars are
    the flux regressed ON the radiation; the r^2 printed under each pair is the
    zero-order fit of that flux against DLR, so a reader can see immediately
    how much scatter the slope was drawn through.

    Open water is drawn but its bars run off the scale, for the reason
    ``fig_response_partition`` gives: ERA5's open-ocean surface is prescribed
    and barely moves, the air is not and does, so the regression there is
    describing the air mass rather than a surface response. Rescaling the panel to fit it would compress the other
    five classes into a flat line.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    cname = control or A.args.control
    ctrl = CONTROL_SETS[cname]
    acc = A.acc(pop)
    slots = list(PANEL_SLOTS)
    names = [SLOT_ORDER[s] for s in slots]
    resp = [turbulent_response(acc, s, control=ctrl) for s in slots]
    x = np.arange(len(slots))
    bw = 0.34

    bs = None
    if bootstrap_ci:
        try:
            bs = bootstrap_turbulent_response(A, n_boot=n_boot,
                                              control=cname)
        except ValueError:
            bs = None            # too few blocks; the bars stand on their own

    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    for i, (key, var, label, colour) in enumerate(TURBULENT_TERMS):
        vals = np.array([r[key] for r in resp])
        pos = x + (i - 0.5) * bw
        ax.bar(pos, vals, bw, color=colour, edgecolor="#333333", linewidth=0.6,
               label=f"{label} heat flux", zorder=3)
        if bs is not None:
            # Asymmetric by construction: these are PERCENTILES of the
            # bootstrap distribution, not value +/- k*sigma, so a skewed
            # sampling distribution shows as an off-centre bar.
            lo = np.array([bs[key][sl]["lo"] for sl in slots])
            hi = np.array([bs[key][sl]["hi"] for sl in slots])
            ax.errorbar(pos, vals,
                        yerr=np.vstack([vals - lo, hi - vals]),
                        fmt="none", ecolor="#222222", elinewidth=1.1,
                        capsize=3.0, zorder=6)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:+.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, -11 if v < 0 else 4), ha="center",
                        fontsize=7.4, zorder=5)
    total = np.array([r["dturb_dlwd"] for r in resp])
    ax.scatter(x, total, marker="_", s=520, color="#B2182B", linewidth=2.2,
               zorder=6, label="sum: d(SH+LH)/d(DLR)")

    # Scale to the classes where the estimator is a surface response, and mark
    # the ones that run past it rather than dropping or rescaling for them.
    stack = np.concatenate([[r[k] for k, _, _, _ in TURBULENT_TERMS]
                            for r in resp] + [total[:, None].ravel()])
    if bs is not None:
        stack = np.concatenate([stack] + [
            [bs[k][sl][e] for sl in slots]
            for k, _, _, _ in TURBULENT_TERMS for e in ("lo", "hi")])
    sane = np.abs(stack) <= TURBULENT_SANE_MAX
    if sane.any():
        m = float(np.nanmax(np.abs(stack[sane])))
        ax.set_ylim(-1.55 * m, 1.55 * m)
    lo, hi = ax.get_ylim()
    for j, r in enumerate(resp):
        if max(abs(r["dshf_dlwd"]), abs(r["dlhf_dlwd"])) > TURBULENT_SANE_MAX:
            # The values still have to be readable: a bar that runs past the
            # axis with no number on it is worse than no bar.
            ax.annotate(f"off scale\nSH {r['dshf_dlwd']:+.2f}   "
                        f"LH {r['dlhf_dlwd']:+.2f}\n"
                        f"surface prescribed:\nthis is the air mass",
                        (x[j], 0.55 * hi), ha="center", va="center",
                        fontsize=7.4, color="#8a5a00", fontweight="bold",
                        bbox=NOTE_BOX, zorder=7)

    ax.axhline(0.0, color="#333333", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SLOT_LABELS[n].replace(" (", "\n(").replace(" zone", "\nzone")
         .replace("Utqiagvik ", "Utqiagvik\n") for n in names], fontsize=9)
    ax.set_ylabel("d(flux) / d(DLR)   [W m$^{-2}$ per W m$^{-2}$]")
    ax.grid(axis="y", alpha=0.2, lw=0.5, zorder=0)
    ax.legend(fontsize=9, ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))

    for j, r in enumerate(resp):
        ax.annotate(f"$r^2$ {r['r2_shf']:.3f} / {r['r2_lhf']:.3f}",
                    (x[j], lo), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.2, color="#666666")

    top = _header_block(
        fig, "Turbulent flux response to downwelling longwave, by surface type",
        _figure_subtitle(A, pop),
        note=(f"each flux REGRESSED ON DLR ({CONTROL_LABELS[cname]}); "
              "positive = more DLR accompanies more heat INTO the surface. "
              + (f"whiskers are 95% moving-block bootstrap intervals "
                 f"({A.sec['n_block']} blocks of {A.sec['block_days']} days) - "
                 f"a bar whose whisker crosses zero is not distinguishable "
                 f"from no response"
                 if bs is not None else
                 "r^2 under each pair is sensible / latent against DLR alone")),
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, top))
    return _save(fig, A, f"turbulent_response_{pop}_{cname}", out_dir, dpi)


# The multiple-regression prediction is drawn at ONE wind speed, the class's
# own mean. In an additive model the wind coefficient only slides the line up
# and down without tilting it, so drawing it at several wind speeds adds
# parallel copies and no information about the slope, which is the quantity the
# figure is for. The wind coefficient is reported in the annotation instead.
MULTIPLE_FIT_COLOR = "#000000"


def _wind_overlay(A: Analysis, acc: dict, slot: int, panel: Panel2D,
                  x_span: tuple[float, float]):
    """Multiple regression of SHF on DLR AND wind, drawn at three wind speeds.

    An additive multiple regression predicts

        SHF = a + b_DLR * DLR + b_U * U,

    so at a fixed wind speed the prediction is a straight line in DLR with
    slope b_DLR, and changing U slides that line up or down without tilting
    it. THREE PARALLEL LINES ARE THEREFORE WHAT "HOLDING WIND FIXED" LOOKS LIKE
    -- and their being parallel is the additive model's assumption made
    visible, not a property of the data. Where the true dependence is
    multiplicative (it is; see the U*dT specification check) the spacing
    between the lines is right on average but the common slope is a
    compromise across wind speeds.
    """
    weighted = FIT_IS_WEIGHTED[A.args.fit_mode]
    ms = multiple_stats(acc, slot, panel.y_key,
                        (panel.x_key, "wspd_m_s"), weighted=weighted)
    simple = moment_stats(acc, slot, panel.x_key, panel.y_key,
                          weighted=weighted)
    wst = moment_stats(acc, slot, "wspd_m_s", "wspd_m_s", weighted=weighted)
    if not np.isfinite(ms["r2"]) or not np.isfinite(wst["x_sd"]):
        return [], []

    b_x, b_u = float(ms["coef"][0]), float(ms["coef"][1])
    xs = np.array(x_span)
    u_bar = max(wst["x_mean"], 0.0)
    lines = [(xs, ms["intercept"] + b_x * xs + b_u * u_bar,
              dict(color=MULTIPLE_FIT_COLOR, lw=1.6, ls="-",
                   label="multiple fit, wind held at its mean"))]

    y_u = plain_units(TRACKED[VAR_INDEX[panel.y_key]].units)
    x_u = plain_units(TRACKED[VAR_INDEX[panel.x_key]].units)
    note = [
        "multiple fit, $U$ held fixed",
        f"dy/dx = {b_x:+.4g} {y_u}/({x_u})",
        f"dy/d$U$ = {b_u:+.3g} {y_u}/(m s$^{{-1}}$)",
        f"$R^2$ = {ms['r2']:.3f}   at $\\bar{{U}}$ = {u_bar:.1f} m s$^{{-1}}$",
    ]
    return lines, note


def _flux_vs_dlr_multiple(A: Analysis, panel_name: str, flux_label: str,
                          stem: str, out_dir, dpi, fit_mode):
    """Shared body of the two multiple-regression scatters.

    The simple fit's LINE is suppressed and only the multiple-regression
    prediction is drawn, so the panel carries one line and no ambiguity about
    which it is; the simple slope stays available in the report and on the
    corresponding single-predictor figure for comparison.
    """
    return _scatter_figure(
        A, panel_name,
        f"Surface {flux_label} heat flux against downwelling longwave, "
        "with wind speed controlled",
        stem, out_dir, dpi, fit_mode=fit_mode, fit_orient="y_on_x",
        overlay=_wind_overlay, stem_suffix="_multiple",
        note_loc="lower right", show_legend=False, draw_fit=False)


def fig_shf_vs_dlr_multiple(A: Analysis, out_dir=None, dpi: int | None = None,
                            fit_mode: str | None = None):
    """SHF against DLR, with wind speed added as a second predictor.

    The line is the multiple-regression prediction
    ``SHF = a + b_DLR * DLR + b_U * U`` evaluated at the class's mean wind, so
    its slope is d(SHF)/d(DLR) with wind HELD FIXED -- the partial derivative,
    and the same number as the "+ wind" rung of the control ladder. The simple
    fit is on the previous figure for comparison.

    WHAT THIS DOES AND DOES NOT BUY. Adding wind raises the variance explained
    substantially -- over open water from 0.35 to 0.54 -- because wind is a
    genuine confounder: Arctic storms are both cloudy and windy. It does NOT
    get near the 0.88-0.98 of the bulk specification, and it cannot, because
    those two fits answer different questions. The bulk fit regresses SHF on
    U*(T_skin - T_2m), the flux's DIRECT driver, which nearly determines it.
    This fit regresses SHF on DLR, which reaches the flux only through the skin
    temperature and can therefore explain only the part of the skin-to-air
    difference that DLR itself moves. Closing that gap would mean putting the
    temperature difference into the regression -- the one variable that must
    stay out, because it is the mediator the DLR effect travels through. High
    R^2 and a causal estimate are different goals here, and the variable that
    serves one destroys the other.
    """
    return _flux_vs_dlr_multiple(A, "dlr_shf", "sensible", "shf_vs_dlr",
                                 out_dir, dpi, fit_mode)


def fig_lhf_vs_dlr_multiple(A: Analysis, out_dir=None, dpi: int | None = None,
                            fit_mode: str | None = None):
    """LHF against DLR, with wind speed added as a second predictor.

    The latent-heat counterpart of ``fig_shf_vs_dlr_multiple``, same treatment
    and same control, so the two slopes are directly comparable and their sum
    is the total turbulent response of the bar chart.

    Wind is the control here for symmetry with the sensible figure. It is not
    the strongest confounder on this side: 2 m humidity is, because water
    vapour is itself a longwave emitter and so raises DLR directly rather than
    by advection. That rung is on the control ladder, where the humidity column
    moves the latent slope considerably further than wind alone does.
    """
    return _flux_vs_dlr_multiple(A, "dlr_lhf", "latent", "lhf_vs_dlr",
                                 out_dir, dpi, fit_mode)


def fig_control_ladder(A: Analysis, out_dir=None, dpi: int | None = None,
                       population: str | None = None,
                       bootstrap_ci: bool = True, n_boot: int | None = None,
                       ladders: dict | None = None):
    """How each slope moves as confounders are added to the regression.

    Four panels, one per regression, each showing every surface class under a
    ladder of control sets. This is the figure that turns "what should I
    control for" from an argument into a measurement: a slope that barely moves
    as a control is added is evidence that the confounder was not doing much,
    and one that moves a lot is evidence that it was.

    Read it with the note above CONTROL_LADDERS in hand. In particular the
    2 m temperature and humidity controls are mediators as well as confounders,
    so the last rung of a ladder is a LOWER bound on the response rather than a
    better estimate of it -- the bracket, not the answer.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    acc = A.acc(pop)
    lad = ladders or CONTROL_LADDERS
    slots = list(PANEL_SLOTS)
    x = np.arange(len(slots))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4))
    for ax, (name, cfg) in zip(axes.ravel(), lad.items()):
        steps = cfg["steps"]
        bw = 0.82 / len(steps)
        shades = plt.get_cmap("cividis")(np.linspace(0.15, 0.85, len(steps)))
        vals = np.full((len(steps), len(slots)), np.nan)
        for si, (label, ctrl) in enumerate(steps):
            for j, slot in enumerate(slots):
                vals[si, j] = partial_slope(acc, slot, cfg["y"], cfg["x"], ctrl)
            pos = x - 0.41 + bw * (si + 0.5)
            ax.bar(pos, vals[si], bw, color=shades[si], edgecolor="#333333",
                   linewidth=0.5, label=label, zorder=3)
            if bootstrap_ci:
                try:
                    bs = bootstrap(
                        A, lambda a, sl, c=ctrl, cf=cfg: partial_slope(
                            a, sl, cf["y"], cf["x"], c),
                        n_boot=n_boot or 500, slots=slots)
                except ValueError:
                    bootstrap_ci = False
                else:
                    lo = np.array([bs[sl]["lo"] for sl in slots])
                    hi = np.array([bs[sl]["hi"] for sl in slots])
                    ax.errorbar(pos, vals[si],
                                yerr=np.vstack([np.maximum(vals[si] - lo, 0),
                                                np.maximum(hi - vals[si], 0)]),
                                fmt="none", ecolor="#333333", elinewidth=0.8,
                                capsize=1.6, alpha=0.8, zorder=5)

        # Scale to the classes where the estimator is a surface response. Open
        # water runs off every one of these panels for the reason given in
        # fig_response_partition, and rescaling for it flattens the rest.
        finite = vals[np.isfinite(vals)]
        med = np.median(np.abs(finite)) if finite.size else 1.0
        keep = np.abs(finite) <= max(8.0 * med, 1e-6)
        if keep.any():
            m = float(np.max(np.abs(finite[keep])))
            ax.set_ylim(-1.5 * m, 1.5 * m)
        lo_a, hi_a = ax.get_ylim()
        for j in range(len(slots)):
            if np.any(np.abs(vals[:, j]) > hi_a):
                ax.annotate("off\nscale", (x[j], 0.72 * hi_a), ha="center",
                            va="center", fontsize=7, color="#8a5a00",
                            fontweight="bold", bbox=NOTE_BOX, zorder=7)

        ax.axhline(0.0, color="#333333", lw=0.9, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SLOT_LABELS[SLOT_ORDER[sl]].replace(" (", "\n(")
             .replace(" zone", "\nzone").replace("Utqiagvik ", "Utqiagvik\n")
             for sl in slots], fontsize=7.6)
        ax.set_ylabel(cfg["units"], fontsize=9)
        ax.set_title(cfg["title"], fontsize=11, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2, lw=0.5, zorder=0)
        ax.legend(fontsize=7.4, frameon=False, ncol=2, loc="upper left")

    top = _header_block(
        fig, "Does controlling for a confounder move the answer?",
        _figure_subtitle(A, pop),
        note=("bars are the slope with the named variables held fixed; "
              "whiskers are 95% block-bootstrap intervals. A slope that does "
              "not move as a control is added is evidence that confounder was "
              "not doing much. $T_{2m}$ and $q_{2m}$ are MEDIATORS as well as "
              "confounders, so their rungs are a lower bound, not a better "
              "estimate - see CONTROL_LADDERS"),
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, A, f"control_ladder_{pop}", out_dir, dpi)


def fig_dlr_vs_lwp(A: Analysis, out_dir=None, dpi: int | None = None,
                   fit_mode: str | None = None, fit_orient: str | None = None):
    """Downwelling longwave against liquid water path -- the ERA5 counterpart
    to the ARM observational figure from Barrow.

    Axes, colour and fit direction follow that figure so the two can be read
    side by side: LWP 0-350 g m-2, DLR 100-350 W m-2, coloured by mean cloud
    temperature, and DLR fitted on LWP so the reported equation is in the same
    sense as the published y = 0.27x + 228.26.

    READ THE COLOUR, NOT ONLY THE SLOPE. The scatter fans out at low LWP into a
    near-vertical spread of more than 100 W m-2, and that spread is almost
    entirely temperature: a thin cloud at -35 C and a thin cloud at -5 C sit at
    the same x and 100 W m-2 apart in y. The LWP-DLR correlation is therefore
    part opacity -- more liquid, higher emissivity, more DLR -- and part
    covariance, because warm Arctic air masses are also the moist ones that
    carry more liquid. A single r^2 does not separate those, which is what
    makes the colour worth the second pass over the pressure archive.

    Requires ``prepare(with_cloud_temperature=True)`` for the colour; without
    it the panels fall back to density shading and everything else is unchanged.
    """
    return _scatter_figure(
        A, "lwp_dlr",
        "Downwelling longwave against liquid water path",
        "dlr_vs_lwp", out_dir, dpi, fit_mode=fit_mode, fit_orient=fit_orient)


# ----------------------------------------------------------------------------
# Figure 6: the matched comparison
# ----------------------------------------------------------------------------
def regime_slot(A: Analysis, slot: int, regime: int) -> int:
    """Flat group index into the regime-stratified accumulator."""
    return slot * A.sec["n_regime"] + regime


def fig_shf_by_lwp_regime(A: Analysis, out_dir=None, dpi: int | None = None,
                          population: str | None = None,
                          value_key: str = "shf_W_m2"):
    """Mean sensible heat flux by LWP regime and surface class.

    WHY THIS FIGURE EXISTS. A weak correlation across a heterogeneous
    population can hide a strong conditional relationship. The scatter figures
    pool every surface and every cloud thickness together and return
    r^2 = 0.001-0.06 on LWP; that number is a statement about the pooled
    population, not about the physics inside it. Stratifying by LWP regime AND
    by surface class holds the two dominant confounders roughly fixed within
    each bar, so a difference BETWEEN bars in the same regime is a difference
    between surfaces at comparable cloud forcing, and a difference ACROSS
    regimes within one surface is a cloud effect at fixed surface. That is a
    matched comparison, and it is a better estimator of the thing in question
    than a regression slope through the pooled cloud.

    SAMPLE SIZE IS ENCODED TWICE, deliberately. The classes differ in sample
    size by more than a hundredfold -- tens of thousands of cell-hours over
    land against millions over sea ice -- and a grouped bar chart that hides
    that invites the obvious objection. Bar WIDTH is proportional to
    log10(cell-hours), which makes the disparity visible at a glance, and the
    count is PRINTED on every bar, which makes it exact. The width encoding is
    logarithmic and is labelled as such: at a linear encoding the land bars
    would be invisible.

    The whisker is +/- one standard deviation WITHIN the bin, not a standard
    error. With millions of cell-hours the standard error is a fraction of a
    W m-2 and would draw a line thinner than the bar edge, which would imply a
    precision the matching does not have; the spread is what tells you whether
    two bars are really different.
    """
    import matplotlib.pyplot as plt

    pop = population or A.args.population
    acc = A.regime_acc(pop)
    n_reg = A.sec["n_regime"]
    slots = list(PANEL_SLOTS)
    names = [SLOT_ORDER[s] for s in slots]
    v = TRACKED[VAR_INDEX[value_key]]

    means = np.full((n_reg, len(slots)), np.nan)
    sds = np.zeros((n_reg, len(slots)))
    hours = np.zeros((n_reg, len(slots)))
    for j, slot in enumerate(slots):
        for r in range(n_reg):
            g = regime_slot(A, slot, r)
            if acc["w"][g] <= 0:
                continue
            means[r, j] = mean_of(acc, g, value_key)
            st = moment_stats(acc, g, value_key, value_key)
            sds[r, j] = st["x_sd"]
            hours[r, j] = acc["n"][g] * HOURS_PER_STEP

    # Bar width from log10(cell-hours), floored so an empty-ish bin is still a
    # visible sliver rather than nothing at all.
    with np.errstate(divide="ignore"):
        lg = np.where(hours > 0, np.log10(np.maximum(hours, 1.0)), np.nan)
    lo, hi = np.nanmin(lg), np.nanmax(lg)
    span = hi - lo if hi > lo else 1.0
    frac = np.clip((lg - lo) / span, 0.0, 1.0)
    slot_w = 0.86 / len(slots)
    widths = slot_w * (0.30 + 0.70 * np.nan_to_num(frac))

    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    x = np.arange(n_reg)
    for j, (slot, name) in enumerate(zip(slots, names)):
        centre = x - 0.43 + slot_w * (j + 0.5)
        ax.bar(centre, means[:, j], widths[:, j], color=SLOT_COLORS[name],
               edgecolor="#333333", linewidth=0.6, label=SLOT_LABELS[name],
               zorder=3)
        ax.errorbar(centre, means[:, j], yerr=sds[:, j], fmt="none",
                    ecolor="#555555", elinewidth=0.7, capsize=1.8, alpha=0.45,
                    zorder=4)
        # The count goes at the BAR's own tip, not the whisker's. Chasing the
        # whisker end scatters the labels over the whole panel and breaks the
        # association between a number and the bar it belongs to.
        for r in range(n_reg):
            if not np.isfinite(means[r, j]):
                continue
            h = hours[r, j]
            txt = (f"{h / 1e6:.1f}M" if h >= 1e6
                   else f"{h / 1e3:.0f}k" if h >= 1e3 else f"{h:.0f}")
            ax.annotate(txt, (centre[r], means[r, j]),
                        textcoords="offset points",
                        xytext=(0, -10 if means[r, j] < 0 else 4),
                        ha="center", fontsize=7.0, color="#222222", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.7))

    ax.axhline(0.0, color="#333333", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels(A.sec["regime_edges"],
                                     lowest_drawn_lwp(A.phase_kw)), fontsize=9.5)
    ax.set_ylabel(f"mean {v.label.lower()}   [{v.units}]")
    # No x label: the tick labels already name the quantity and its units, and
    # a third line of text there collides with the legend.
    ax.grid(axis="y", alpha=0.2, lw=0.5, zorder=0)

    ax.legend(fontsize=8.6, ncol=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.115))
    # The encoding legend is a text note, not a legend entry: a blank swatch
    # standing for "width means something" reads as a seventh surface class.
    fig.text(0.5, 0.015,
             "bar width $\\propto$ log$_{10}$(cell-hours)  |  printed count is "
             "exact  |  whisker is $\\pm$1 standard deviation WITHIN the bin, "
             "not a standard error",
             ha="center", fontsize=8.0, color="#555555")

    top = _header_block(
        fig, "Sensible heat flux stratified by cloud regime and surface type",
        _figure_subtitle(A, pop),
        note=("a matched comparison: within a regime the classes see "
              "comparable cloud forcing, so a difference between bars is a "
              "difference between surfaces, not between cloud populations"),
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0.055, 1, top))
    return _save(fig, A, f"shf_by_lwp_regime_{pop}", out_dir, dpi)


# ----------------------------------------------------------------------------
# The bootstrap
# ----------------------------------------------------------------------------
MOMENT_KEYS: tuple[str, ...] = ("w", "n", "x", "xy", "x_u", "xy_u")


def acc_from_block_counts(block_mom: dict, n_block: int, n_slot: int,
                          counts: np.ndarray) -> dict:
    """Assemble a moment accumulator from a weighted selection of blocks.

    ``counts[b]`` is how many times block ``b`` was drawn. Because the moments
    are additive, a bootstrap replicate is exactly this contraction -- no
    resampling of individual samples and no refitting from raw data. That is
    what makes two thousand replicates cost less than a second.
    """
    out = {}
    for k in MOMENT_KEYS:
        arr = block_mom[k].reshape((n_block, n_slot) + block_mom[k].shape[1:])
        out[k] = np.tensordot(counts.astype(float), arr, axes=(0, 0))
    return out


def _acc_from_block_counts(A: Analysis, counts: np.ndarray) -> dict:
    """``acc_from_block_counts`` for this Analysis's own per-block moments."""
    return acc_from_block_counts(A.sec["block_mom"], A.sec["n_block"],
                                 N_SLOT, counts)


def bootstrap_blocks(block_mom: dict, n_block: int, n_slot: int, stat,
                     n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
                     seed: int = DEFAULT_BOOTSTRAP_SEED, ci: float = 95.0,
                     slots=None) -> dict:
    """Moving-block bootstrap over any per-block moment accumulator.

    The estimator core, independent of where the blocks came from: ``bootstrap``
    calls it with an ERA5 ``Analysis``'s blocks, and any other dataset can call
    it with blocks built by ``moments_from_arrays(..., group=block_index)``.
    See the long note above ``block_index`` for why blocks and not points.
    """
    if n_block < 8:
        raise ValueError(
            f"only {n_block} blocks; a percentile interval from that few is "
            f"not worth quoting. Use more data or shorter blocks.")
    slots = list(range(n_slot)) if slots is None else list(slots)
    rng = np.random.default_rng(seed)
    full = acc_from_block_counts(block_mom, n_block, n_slot, np.ones(n_block))

    draws = {sl: np.empty(n_boot) for sl in slots}
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, n_block, n_block),
                             minlength=n_block)
        acc = acc_from_block_counts(block_mom, n_block, n_slot, counts)
        for sl in slots:
            draws[sl][i] = stat(acc, sl)

    lo_q, hi_q = 50.0 - ci / 2.0, 50.0 + ci / 2.0
    out = {}
    for sl in slots:
        d = draws[sl][np.isfinite(draws[sl])]
        out[sl] = {
            "value": stat(full, sl),
            "lo": float(np.percentile(d, lo_q)) if d.size else np.nan,
            "hi": float(np.percentile(d, hi_q)) if d.size else np.nan,
            "se": float(d.std(ddof=1)) if d.size > 1 else np.nan,
            "n_block": n_block, "n_boot": n_boot,
        }
    return out


def bootstrap(A: Analysis, stat, n_boot: int | None = None,
              seed: int = DEFAULT_BOOTSTRAP_SEED, ci: float = 95.0,
              slots=None) -> dict:
    """Moving-block bootstrap confidence intervals for any moment statistic.

    ``stat(acc, slot) -> float`` is evaluated on the full record and on each
    synthetic record assembled by drawing ``n_block`` blocks with replacement.
    Returns ``{slot: {"value", "lo", "hi", "se", "n_block"}}`` with the
    percentile interval.

    See the long note above ``block_index`` for why this is the right estimator
    and the textbook standard error is not. In one line: the blocks, not the
    cell-hours, are the units the record supplies independently, so they are
    what gets resampled.

    ``value`` is the point estimate from the real record, not the bootstrap
    mean -- the interval describes the uncertainty around the estimate, it does
    not replace it.
    """
    if A.sec["n_block"] < 8:
        raise ValueError(
            f"only {A.sec['n_block']} blocks of {A.sec['block_days']} days; a "
            f"percentile interval from that few is not worth quoting. Load "
            f"more seasons or shorten --bootstrap-block-days.")
    return bootstrap_blocks(
        A.sec["block_mom"], A.sec["n_block"], N_SLOT, stat,
        n_boot=int(n_boot or A.args.bootstrap_samples), seed=seed, ci=ci,
        slots=(list(PANEL_SLOTS) + [ALL_SLOT]) if slots is None else slots)


def naive_se(acc: dict, slot: int, x_key: str, y_key: str) -> float:
    """The textbook OLS standard error -- WRONG HERE, kept for comparison.

    Reported beside the bootstrap interval so the size of the independence
    assumption is visible as a number rather than as an assertion. It is too
    small by roughly sqrt(n / n_eff), which on this domain is a factor of
    order a hundred.
    """
    st = moment_stats(acc, slot, x_key, y_key, weighted=False)
    n, r2 = st["n_hours"], st["r2"]
    if not np.isfinite(r2) or n < 3 or st["x_sd"] <= 0:
        return float("nan")
    return float(st["y_sd"] / st["x_sd"] * np.sqrt(max(1.0 - r2, 0.0) / (n - 2)))


def bootstrap_turbulent_response(A: Analysis, n_boot: int | None = None,
                                 control: str | None = None,
                                 seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Bootstrap intervals for d(SHF)/d(DLR), d(LHF)/d(DLR) and their sum."""
    ctrl = CONTROL_SETS[control or A.args.control]
    out = {}
    for key, var, _, _ in TURBULENT_TERMS:
        out[key] = bootstrap(
            A, lambda a, sl, v=var: partial_slope(a, sl, v, "lwd_W_m2", ctrl),
            n_boot=n_boot, seed=seed)
    out["dturb_dlwd"] = bootstrap(
        A, lambda a, sl: (partial_slope(a, sl, "shf_W_m2", "lwd_W_m2", ctrl)
                          + partial_slope(a, sl, "lhf_W_m2", "lwd_W_m2", ctrl)),
        n_boot=n_boot, seed=seed)
    return out


# ----------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------
def self_check(A: Analysis, tol: float = 1e-9, verbose: bool = True) -> bool:
    """Assert the algebraic identities the estimates depend on.

    None of these can fail for a reason that is visible in a figure, which is
    why they are checked rather than trusted:

    1. Regressing DLR on itself returns exactly 1. If it does not, the moment
       matrix or the centring is wrong and EVERY slope is wrong with it.
    2. d(SHF)/d(DLR) agrees across all four routes that compute it -- the
       partial regression, the swapped-argument ``moment_stats``, the x-on-y
       member of ``fit_pair``, and the negated partition fraction. These share
       the accumulator but not the code path, so agreement rules out a swapped
       argument somewhere.
    3. The same for d(LHF)/d(DLR).
    4. The five partition fractions sum to one, and the residual -- which the
       partition defines as the remainder of the other four -- equals the
       regression on DLR of the separately accumulated net-flux column.
    5. The two turbulent terms sum to the total turbulent response.
    6. Controlling on a variable drives its own slope to zero, which is the
       test that the multiple regression is solving what it claims to.
    7. The regime bins partition the pooled population exactly.

    Returns True when everything holds. Raises nothing: a failure is reported
    and returned, so a notebook cell shows it rather than aborting.
    """
    acc = A.acc()
    bad: list[str] = []

    def near(a, b, what):
        if not (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol):
            bad.append(f"{what}: {a!r} vs {b!r}")

    for slot in PANEL_SLOTS + (ALL_SLOT,):
        nm = SLOT_LABELS[SLOT_ORDER[slot]]
        near(partial_slope(acc, slot, "lwd_W_m2", "lwd_W_m2"), 1.0,
             f"{nm}: d(DLR)/d(DLR)")
        p = partition(acc, slot)
        t = turbulent_response(acc, slot)
        for var, key, frac in (("shf_W_m2", "dshf_dlwd", "f_sh"),
                               ("lhf_W_m2", "dlhf_dlwd", "f_lh")):
            direct = partial_slope(acc, slot, var, "lwd_W_m2")
            near(direct, moment_stats(acc, slot, "lwd_W_m2", var)["slope"],
                 f"{nm}: {var} via moment_stats")
            near(direct, fit_pair(acc, slot, var, "lwd_W_m2")["x_on_y"]["slope"],
                 f"{nm}: {var} via fit_pair")
            near(direct, -p[frac], f"{nm}: {var} via partition")
            near(direct, t[key], f"{nm}: {var} via turbulent_response")
        near(p["f_lwu"] + p["f_sh"] + p["f_lh"] + p["f_sw"] + p["f_res"], 1.0,
             f"{nm}: partition closure")
        # 4b. The residual is a remainder of four fits AND the fit of an
        #     independently accumulated column, R = LWD - LWU + SW_net + SH
        #     + LH, built per sample in the streaming pass. Nothing in the
        #     algebra forces these to agree: a sign error in any one of the
        #     five terms, or in how R was assembled, breaks it.
        near(p["f_res"], partial_slope(acc, slot, "rnet_W_m2", "lwd_W_m2"),
             f"{nm}: residual vs the direct fit of R on DLR")
        near(p["f_res"], net_flux_slope(acc, slot),
             f"{nm}: residual vs the same combination in the covariance")
        near(t["dshf_dlwd"] + t["dlhf_dlwd"], t["dturb_dlwd"],
             f"{nm}: turbulent sum")

    near(partial_slope(acc, ALL_SLOT, "t2m_K", "lwd_W_m2", ("t2m_K",)), 0.0,
         "controlling on T2m zeroes its own slope")

    racc = A.regime_acc()
    pooled = acc["n"][ALL_SLOT]
    binned = sum(racc["n"][regime_slot(A, ALL_SLOT, r)]
                 for r in range(A.sec["n_regime"]))
    near(binned, pooled, "regime bins partition the population")

    # 8. Summing every block once must reproduce the pooled accumulator: the
    #    blocks partition the record, so the bootstrap is resampling exactly
    #    the data the point estimate was computed from.
    full = _acc_from_block_counts(A, np.ones(A.sec["n_block"]))
    for slot in PANEL_SLOTS + (ALL_SLOT,):
        near(full["n"][slot], acc["n"][slot],
             f"{SLOT_LABELS[SLOT_ORDER[slot]]}: blocks partition the record")
        near(partial_slope(full, slot, "shf_W_m2", "lwd_W_m2"),
             partial_slope(acc, slot, "shf_W_m2", "lwd_W_m2"),
             f"{SLOT_LABELS[SLOT_ORDER[slot]]}: block sum reproduces the slope")

    if verbose:
        if bad:
            print(f"SELF-CHECK FAILED ({len(bad)}):")
            for b in bad[:20]:
                print(f"  {b}")
        else:
            print(f"self-check passed: every identity holds to {tol:g} "
                  f"across {len(PANEL_SLOTS) + 1} groups")
    return not bad


ALL_FIGURES = (
    fig_shf_vs_lwp,
    fig_shf_vs_dlr,
    fig_shf_vs_dskt,
    fig_lwu_vs_dlr,
    fig_swnet_vs_dlr,
    fig_rnet_vs_dlr,
    fig_response_partition,
    fig_partition_terms,
    fig_partition_ledger,
    fig_miz_transect,
    fig_shf_by_lwp_regime,
    fig_lhf_vs_dlr,
    fig_turbulent_response,
    fig_control_ladder,
    fig_shf_vs_dlr_multiple,
    fig_lhf_vs_dlr_multiple,
    fig_dlr_vs_lwp,
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
