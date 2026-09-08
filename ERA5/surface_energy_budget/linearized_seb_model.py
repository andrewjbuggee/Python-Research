#!/usr/bin/env python3
r"""A four-term linearised skin-layer surface energy balance, fitted to ERA5.

THE QUESTION THIS ANSWERS
-------------------------
"How much of a cloud longwave perturbation goes into turbulence?" -- Lynn's
question. ``turbulent_flux_response.py`` answers a *covariance* version of it:
regress each surface flux on the downwelling longwave across the record and
read off the slope. That answer is correct but it is not a partition of a
perturbation, because everything else that travels with an Arctic cloud (warm
moist advection, wind, stability) is inside the slope too.

This module answers the *perturbation* version instead. Perturb the downwelling
longwave, hold the atmosphere above fixed, let the skin temperature find its
new equilibrium, and ask which term takes up each W m-2. That is a well-posed
question with a closed-form answer, and the answer is a set of four numbers
that sum to one.

WHY THE FORCING MAY BE TREATED AS EXTERNAL
------------------------------------------
Arctic cloud radiative forcing is largely set by synoptic advection -- the
clouds arrive from elsewhere -- while the skin layer responds within minutes to
hours. On the timescale of the response the forcing is therefore external, and
that asymmetry is what licenses treating DLR as the independent variable. It is
the same licence Miller et al. (2017) and Sledd et al. (2025) take when they
regress SEB response terms on (LW_down + SW_net). The cloud-boundary-layer
feedback is real, but it operates on the boundary layer over longer timescales
and through a much smaller loop gain than the direct skin response.

Note what this framing does NOT claim: it does not claim a particular joule of
longwave became a particular joule of turbulence. It measures how the system
responds to a perturbation, which sidesteps the attribution problem entirely.

THE MODEL
---------
The skin layer has no heat capacity, so its energy balance is diagnostic: the
fluxes into it must sum to zero at every instant. Linearising each flux about
the mean state in the skin temperature gives

    dLWD = ( lam_LW + lam_SH + lam_LH + lam_G ) * dT_skin              (1)

and the fraction of the perturbation absorbed by each term is that term's
lambda divided by the sum. Every lambda is a positive damping coefficient in
W m-2 K-1:

    lam_LW = 4 eps sigma T^3     more emission as the skin warms
    lam_SH = rho c_p C_H U       stronger upward sensible heat flux
    lam_LH = rho L_s C_E U dq/dT stronger sublimation
    lam_G                        stronger conduction into the ice

THIS IS NOT AN ANALOGY -- IT IS ERA5'S OWN SURFACE EQUATION
-----------------------------------------------------------
The IFS solves, for each tile i, exactly this balance with a zero-heat-capacity
skin (IFS Documentation Cy41r2, Part IV, eq. 8.22):

    (1 - f_Rs) (1 - alb) R_s + eps (R_T - sigma T_sk^4) + H + L E
                                              = Lam_sk (T_sk - T_1)

with the net longwave then re-linearised about the new skin temperature using
precisely ``eps 4 sigma T_sk^3`` (their eq. 8.23). So (1) is the IFS skin
equation with the quartic linearised and the bulk fluxes written out -- which
means the coefficients below are not an external model imposed on ERA5 output,
they are ERA5's own coefficients, and any disagreement between (1) and the
regressions in ``turbulent_flux_response.py`` is informative rather than a
units error.

WHERE EACH COEFFICIENT COMES FROM
---------------------------------
lam_LW   THEORY, checked against the data. 4 eps sigma T^3 with eps the
         Planck-weighted broadband emissivity the IFS constructs from its
         spectral values (0.99 outside the 800-1250 cm-1 window; 0.98 in the
         window for the sea-ice tile, 0.99 for open sea, 0.93-0.96 for
         vegetation and bare soil -- IFS Cy41r2 Part IV Table 2.8 and
         Section 2.8.5). About 3.5 W m-2 K-1 at 250 K.
         MEASURED equivalent: d(LWU)/d(T_skin) from the accumulated moments,
         which should reproduce it and does -- see ``print_report``.

lam_SH   MEASURED, per surface class, as the direct regression of the sensible
         heat flux on (T_skin - T_2m). This is the coupling coefficient
         rho c_p C_H U of the bulk formula, and it is the reason this module
         exists: nothing else in the literature supplies a per-surface-type
         value for the Beaufort in winter. NOTE THE DIRECTION OF THE FIT --
         d(SHF)/d(dT), never 1/[d(dT)/d(SHF)], which overstates it by 1/r^2
         (a factor of 2.7 over pack ice). See FIT_ORIENTS in
         ``turbulent_flux_response.py``.

lam_LH   MEASURED the same way, as the regression of the latent heat flux on
         (T_skin - T_2m); cross-checked against the Clausius-Clapeyron estimate
         lam_SH * (L_s / c_p) * dq_sat/dT, which is the wet-surface limit and
         is the right limit over snow and ice because the skin is sublimating.

lam_G    THEORY, and it is the term that needs the most care, because IT
         DEPENDS ON THE TIMESCALE OF THE FORCING. See below.

THE CONDUCTION TERM IS FREQUENCY DEPENDENT, AND BY A FACTOR OF FORTY
--------------------------------------------------------------------
A slab of ice does not present one conductance to the surface. Force the
surface sinusoidally at angular frequency w and the heat only penetrates a
diffusive damping depth

    d(P) = sqrt( kappa P / pi ),      kappa = k / (rho c)

so the conductance the skin feels is roughly k / d(P) -- large for a fast
perturbation, small for a slow one. Exactly, for a slab of thickness h over a
base held at the freezing point,

    Y(w) = k q coth(q h),      q = sqrt(i w rho c / k)                 (2)

whose real part is the in-phase damping and is what belongs in (1). Both limits
come out of (2) correctly: Y -> k/h as w -> 0, and Y -> k sqrt(i w rho c / k)
(the semi-infinite result) at high frequency.

MEASURED against ERA5's own ice column (k = 2.03 W m-1 K-1, rho c = 1.88e6
J m-3 K-1, h = 1.5 m, IFS Cy41r2 eq. 8.149-8.150):

    forcing period    Re Y  [W m-2 K-1]    damping depth
    1 hour                     57.7            3.5 cm
    6 hours                    23.6            8.6 cm
    12 hours                   16.7           12.2 cm
    1 day                      11.8           17.2 cm
    1 week                      4.5           45.6 cm
    1 month                     2.0           94.4 cm
    steady state                1.35          (whole slab)

TWO THINGS TO NOTICE. First, the one-hour value 57.7 is not a coincidence: the
IFS gives its top ice layer a depth of 0.07 m, so 2 k / D_1 = 58.0 W m-2 K-1,
and 0.035 m is exactly the hourly damping depth in ice. The IFS layer
discretisation is built so that its top-layer conductance IS the hourly
diffusive admittance, and the skin conductivity it uses for the ice-cap tile
is 58.0 W m-2 K-1 (Table 8.2). This module's Lam_sk for sea ice is therefore an
inference, but a tightly constrained one, and the arithmetic is on the page.

Second, AND THIS CORRECTS THE NOTE THIS MODULE WAS WRITTEN FROM: the
conduction term does not grow for a sustained forcing, it SHRINKS, from 58 to
1.35 W m-2 K-1. The physical statement is that a slower perturbation has to
push heat further into the ice, which puts more thermal resistance in series
with the skin. The conductive FLUX shrinks too -- MEASURED, ERA5 sea ice: 21.0 W m-2 at one
hour against 2.0 W m-2 in steady state for a 27 W m-2 perturbation -- because
the fall in lam_G outruns the rise in dT_skin. What genuinely does grow with time is the
ICE GROWTH response, which is a different question, is not a term in (1) at
all, and needs a time-dependent column model.

THE AIR-COUPLING FACTOR, AND WHY THE MODEL AND THE REGRESSION DISAGREE
----------------------------------------------------------------------
Equation (1) holds T_2m fixed. In the record it is not fixed: an hour with more
DLR is an hour with a warmer air mass, and the turbulent fluxes respond to the
DIFFERENCE. Writing dT_2m = alpha dT_skin, the balance becomes

    dLWD = [ lam_LW + (1 - alpha)(lam_SH + lam_LH) + lam_G ] dT_skin   (3)

and the turbulent damping is scaled by (1 - alpha). MEASURED on the Barrow
strip under liquid-bearing overcast, alpha = d(T_2m)/d(LWD) / d(T_skin)/d(LWD)
is close to ONE over sea ice and land -- the air warms right along with the
surface -- which collapses the turbulent term and is precisely why
``turbulent_flux_response.py`` finds |f_SH| <= 0.14 where (1) with alpha = 0
predicts 0.67.

BOTH ARE CORRECT ANSWERS TO DIFFERENT QUESTIONS, and the pair brackets the
one Lynn asked:

    alpha = 0   The fixed-atmosphere limit. "If a cloud added 27 W m-2 to this
                surface and nothing else changed, where would it go?" This is
                the model-physics question, the one a single-column or
                offline-surface experiment would answer, and the UPPER bound
                on the turbulent share.
    alpha = 1   The realised-covariance limit. "Across the hours in this
                record, how did the fluxes actually co-vary with DLR?" This is
                what the reanalysis shows and what an observational comparison
                can check, and it is the LOWER bound on the turbulent share.

Reporting only the first overstates turbulence by roughly a factor of five;
reporting only the second understates the surface's own physics. ``partition``
takes ``alpha=`` and every figure draws the bracket.

WHAT THE ANSWER IS
------------------
MEASURED, ERA5 sea ice under liquid-bearing overcast on the Barrow strip
(Oct-Mar 2022/23-2025/26, 8.30 million cell-hours), at alpha = 0:

    forcing period    f_LW    f_SH    f_LH     f_G    dT_skin for -27 W m-2
    12 hours          11.2%   29.8%    8.8%   50.2%      -0.81 K
    steady state      20.7%   55.3%   16.3%    7.6%      -1.51 K

So the turbulent share (SH + LH together) runs from 39% on a synoptic timescale
to 72% for a sustained forcing, and lam_G crosses lam_SH at a 34-hour period
over pack ice and a 9-hour period at the ice edge. Set alpha to its measured
value instead and the turbulent share collapses toward the regression answer.
The exact numbers come out of ``print_report`` for whatever population and
season the caller loaded -- they are not hard-coded here.

TWO HONEST CAVEATS TO CARRY WITH ANY OF THESE NUMBERS
-----------------------------------------------------
1. This is the INSTANTANEOUS SKIN balance, made timescale-aware only through
   lam_G. The heat capacity of the ice column is in (2), but nothing here
   evolves: for a forcing sustained over weeks the ice column cools, its
   temperature profile changes, and the ice-growth response begins -- a
   genuinely time-dependent problem that (1) cannot address.
2. lam_SH inherits whatever bias ERA5's skin temperature has, and over Arctic
   sea ice that bias is known and substantial: the IFS carries NO SNOW LAYER on
   sea ice (IFS Cy41r2 Part IV Section 8.9, assumption iii), which warms the
   modelled ice surface by 5-10 K in winter (Batrak and Muller 2019). A skin
   temperature biased warm sits on the wrong side of the stability transition,
   so lam_SH is the coefficient of a boundary layer that is less stable than
   the real one. ``COLUMNS["sea_ice_snow"]`` puts 30 cm of snow in series so
   the size of the lam_G half of that error can at least be seen.

State this as a framework with numbers attached, not as a finished result.

SOURCES
-------
Batrak, Y. and Muller, M. (2019). On the warm bias in atmospheric reanalyses
    induced by the missing snow over Arctic sea-ice. Nature Communications 10,
    4170. https://doi.org/10.1038/s41467-019-11975-3
ECMWF (2016). IFS Documentation Cy41r2, Part IV: Physical Processes. Sections
    2.8.5 (surface emissivity, Table 2.8), 8.2.2 and eq. 8.22-8.23 (the tiled
    skin energy balance and its longwave linearisation), Table 8.2 (skin
    conductivities), 8.9 and eq. 8.149-8.150 (the four-layer sea-ice model).
    https://www.ecmwf.int/en/elibrary/79697-ifs-documentation-cy41r2-part-iv-physical-processes
Carslaw, H. S. and Jaeger, J. C. (1959). Conduction of Heat in Solids, 2nd ed.,
    Oxford. Section 2.6 -- the periodic-forcing damping depth and the surface
    admittance of a slab, eq. (2) here.
Maykut, G. A. (1978). Energy exchange over young sea ice in the central Arctic.
    J. Geophys. Res. 83, 3646-3658. -- conductive fluxes through thin ice.
Miller, N. B., Shupe, M. D., Cox, C. J., Noone, D., Persson, P. O. G., and
    Steffen, K. (2017). Surface energy budget responses to radiative forcing at
    Summit, Greenland. The Cryosphere 11, 497-516.
    https://doi.org/10.5194/tc-11-497-2017
Sledd, A., Shupe, M. D., Solomon, A., and Cox, C. J. (2025). Surface Energy
    Balance Responses to Radiative Forcing in the Central Arctic From MOSAiC
    and Models. J. Geophys. Res. Atmospheres 130, e2024JD042578.
    https://doi.org/10.1029/2024JD042578
Stephens, G. L. (1978). Radiation profiles in extended water clouds. II:
    Parameterization schemes. J. Atmos. Sci. 35, 2123-2132. -- the cloud
    emissivity relation behind the LWP regimes.
Sturm, M., Holmgren, J., Konig, M., and Morris, K. (1997). The thermal
    conductivity of seasonal snow. J. Glaciology 43, 26-41.

USAGE
-----
    import turbulent_flux_response as tfr
    import linearized_seb_model as lsm

    A = tfr.prepare(region="barrow", years=tuple(range(2022, 2026)))
    lsm.print_report(A)                       # the tables
    lsm.fig_lambda_bars(A)                    # the four coefficients
    lsm.fig_partition_vs_timescale(A)         # the answer, vs forcing period
    lsm.fig_forcing_response(A)               # dT_skin for a given perturbation
    lsm.fig_model_vs_era5(A)                  # the closure check

Individual numbers, without a figure::

    lam = lsm.lambdas(A, "sea_ice", period_s=12 * 3600.0)
    lsm.partition(lam)["f_sh"]                # the turbulent share
    lsm.partition(lam, d_dlr_W_m2=-27.0)["dT_skin_K"]
"""

from __future__ import annotations

import argparse
import sys
from typing import NamedTuple

import numpy as np

import turbulent_flux_response as tfr
from turbulent_flux_response import (
    Analysis,
    CLASS_ORDER,
    SLOT_LABELS,
    SLOT_ORDER,
)

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
# CODATA 2018 recommended values.
SIGMA_SB = 5.670374419e-8      # W m-2 K-4, Stefan-Boltzmann
PLANCK_H = 6.62607015e-34      # J s
BOLTZMANN_K = 1.380649e-23     # J K-1
LIGHT_C = 2.99792458e8         # m s-1

# Dry-air and moist thermodynamic constants as the IFS defines them
# (IFS Cy41r2, Part IV, Chapter 12 "Basic physical constants").
CP_AIR = 1004.7                # J kg-1 K-1, specific heat of dry air
L_SUBLIMATION = 2.8345e6       # J kg-1, latent heat of sublimation
L_VAPORISATION = 2.5008e6      # J kg-1, latent heat of vaporisation
R_VAPOUR = 461.5               # J kg-1 K-1, gas constant for water vapour
EPS_MOLAR = 0.621981           # M_water / M_dry_air
P_SURFACE_REF = 1013.0e2       # Pa, reference surface pressure for q_sat


# ----------------------------------------------------------------------------
# Surface longwave emissivity, exactly as the IFS builds it
# ----------------------------------------------------------------------------
# IFS Cy41r2, Part IV, Section 2.8.5: "The thermal emissivity of the surface
# outside the 800-1250 cm-1 spectral region is assumed to be 0.99 everywhere.
# In the window region, the spectral emissivity is constant for open water, sea
# ice, the interception layer and exposed snow tiles. ... Finally, a broadband
# emissivity is obtained by convolution of the spectral emissivity and the
# Planck function at the skin temperature."
#
# So the broadband value is not a constant -- it is weighted by the Planck
# function and therefore depends on the skin temperature. It barely moves (the
# window and out-of-window values differ by at most 0.06), which is exactly why
# it is worth computing once rather than arguing about: the answer is that
# lam_LW is set by temperature, not by emissivity uncertainty.
WINDOW_CM1 = (800.0, 1250.0)   # the IFS atmospheric-window band
EPS_OUT_WINDOW = 0.99          # every tile, outside the window

# Window-region emissivity per tile, IFS Cy41r2 Part IV Table 2.8. Mapped onto
# this project's surface classes: the ocean classes are mixtures of the open-sea
# and sea-ice tiles, so the MIZ value is the midpoint and is flagged as such.
EPS_WINDOW: dict[str, float] = {
    "land": 0.945,             # low vegetation / bare ground, 0.93-0.96
    "coastal": 0.96,           # land-sea mixture; nearer the water value
    "open_ocean": 0.99,        # tile 1, open sea
    "marginal_ice": 0.985,     # tile 1 / tile 2 mixture
    "sea_ice": 0.98,           # tile 2, sea ice
}
EPS_WINDOW_DEFAULT = 0.98


def planck_spectral_radiance(nu_cm1: np.ndarray, T_K: float) -> np.ndarray:
    """Planck function B_nu at wavenumber ``nu_cm1`` [cm-1] and ``T_K`` [K].

    Returned in W m-2 sr-1 (cm-1)-1. Only its SHAPE matters here -- it is used
    as a weighting function -- so the constant prefactor is carried for
    correctness rather than for the value.
    """
    nu_m = np.asarray(nu_cm1, dtype=float) * 100.0        # cm-1 -> m-1
    c1 = 2.0 * PLANCK_H * LIGHT_C**2                      # W m2 sr-1
    c2 = PLANCK_H * LIGHT_C / BOLTZMANN_K                 # K m
    # 1e2 converts per-m-1 to per-cm-1.
    return c1 * nu_m**3 / np.expm1(c2 * nu_m / T_K) * 1e2


def planck_window_fraction(T_K: float,
                           window_cm1: tuple[float, float] = WINDOW_CM1,
                           nu_max_cm1: float = 3000.0,
                           n: int = 6000) -> float:
    """Fraction of the Planck emission at ``T_K`` inside the window band.

    Integrated on a uniform wavenumber grid out to ``nu_max_cm1``, which holds
    better than 99.9% of the emission at Arctic temperatures. At 250 K the
    answer is 0.234, so a window emissivity 0.01 below the out-of-window value
    pulls the broadband value down by only 0.0023 -- which is why the
    emissivity is not worth arguing about and the temperature is.
    """
    nu = np.linspace(1.0, nu_max_cm1, n)
    b = planck_spectral_radiance(nu, T_K)
    total = np.trapezoid(b, nu)
    inside = (nu >= window_cm1[0]) & (nu <= window_cm1[1])
    return float(np.trapezoid(b[inside], nu[inside]) / total)


def broadband_emissivity(T_K: float, slot_name: str = "sea_ice") -> float:
    """Planck-weighted broadband emissivity for one surface class.

    The convolution the IFS describes in Section 2.8.5, with the spectral
    values of Table 2.8. Returns a number between the window and out-of-window
    emissivities.
    """
    f_win = planck_window_fraction(T_K)
    eps_win = EPS_WINDOW.get(slot_name, EPS_WINDOW_DEFAULT)
    return float(f_win * eps_win + (1.0 - f_win) * EPS_OUT_WINDOW)


def lambda_lw(T_K: float, slot_name: str = "sea_ice",
              eps: float | None = None) -> float:
    """lam_LW = 4 eps sigma T^3, in W m-2 K-1.

    The linearisation of the Stefan-Boltzmann law about the mean skin
    temperature. This is the same expression the IFS itself uses to re-linearise
    the net longwave after solving the skin balance (Cy41r2 eq. 8.23), so it is
    not an approximation imported from outside -- it is the model's own.

    About 3.5 W m-2 K-1 at 250 K and 4.9 at 280 K; the T^3 dependence means the
    radiative damping over midwinter pack ice is roughly 30% weaker than over
    open water, which by itself would make the ice surface MORE volatile, not
    less.
    """
    e = broadband_emissivity(T_K, slot_name) if eps is None else eps
    return float(4.0 * e * SIGMA_SB * T_K**3)


# ----------------------------------------------------------------------------
# The conduction term: a frequency-dependent surface admittance
# ----------------------------------------------------------------------------
class Layer(NamedTuple):
    """One conducting layer, top-down.

    Attributes
    ----------
    k_W_m_K : Thermal conductivity, W m-1 K-1.
    rho_c_J_m3_K : Volumetric heat capacity, J m-3 K-1.
    thickness_m : Layer thickness, m.
    label : What it is, for figure legends.
    """

    k_W_m_K: float
    rho_c_J_m3_K: float
    thickness_m: float
    label: str


# ERA5's sea ice, verbatim from IFS Cy41r2 Part IV eq. 8.149-8.150: volumetric
# heat capacity 1.88e6 J m-3 K-1, conductivity 2.03 W m-1 K-1, total slab depth
# prescribed at 1.5 m in four layers, base held at the sea-water freezing point
# T_0 - 1.7 K. No snow accumulates on it (Section 8.9, assumption iii), which is
# both a documented simplification and the source of the known warm bias in the
# modelled ice surface temperature (Batrak and Muller 2019).
ICE_ERA5 = Layer(2.03, 1.88e6, 1.5, "ERA5 sea ice, 1.5 m")

# Arctic snow. Conductivity 0.31 W m-1 K-1 is the seasonal-snow value of Sturm
# et al. (1997) used for Arctic sea ice; Miller et al. (2017) measure 0.47 at
# Summit, where the pack is denser (413 kg m-3), so 0.31 is the sea-ice end of
# the range rather than a universal value. Volumetric heat capacity from
# rho = 300 kg m-3 and c = 2100 J kg-1 K-1.
SNOW_ARCTIC = Layer(0.31, 0.63e6, 0.30, "30 cm Arctic snow")

# Frozen soil, for the land and coastal classes. Layer depths are the IFS soil
# discretisation (0.07 / 0.21 / 0.72 / 1.89 m, IFS Cy41r2 Part IV Table 8.6),
# collapsed to one 2.89 m slab because nothing here resolves the profile.
# Properties are for frozen, moist tundra soil: conductivity ~2.2 W m-1 K-1 and
# volumetric heat capacity ~2.0e6 J m-3 K-1 are mid-range values from the
# Peters-Lidard/Johansen scheme the IFS uses; treat the land numbers as
# indicative, not measured -- lam_SH over land has r^2 = 0.13 anyway.
SOIL_FROZEN = Layer(2.2, 2.0e6, 2.89, "frozen soil, 2.89 m")

COLUMNS: dict[str, tuple[Layer, ...]] = {
    # What ERA5 actually has under a sea-ice tile.
    "sea_ice": (ICE_ERA5,),
    # What is really there. Included to size the missing-snow error, not
    # because ERA5's lam_G should be computed from it.
    "sea_ice_snow": (SNOW_ARCTIC, ICE_ERA5),
    "frozen_soil": (SOIL_FROZEN,),
}

# OPEN WATER HAS NO CONDUCTING COLUMN IN THIS SENSE. ERA5 prescribes the sea
# surface temperature from the OSTIA analysis and holds it fixed through the
# forecast (IFS Cy41r2 Part IV Section 8.10), so the water surface is CLAMPED,
# not diffusive. In the language of this model that is lam_G -> infinity, which
# is why the measured d(T_skin)/d(LWD) over open ocean is 0.017 K per W m-2 --
# an order of magnitude below every other class -- and why the ordinary
# regression coefficients there exceed one and stop being a partition. The
# model is still evaluated over open water for comparability, but with the
# sea-ice column, and every table and figure marks it.
PINNED_SLOTS: tuple[str, ...] = ("open_ocean",)

# Which conducting column each surface class gets by default. Applying one
# column to every class would be simpler and would also be wrong: 1.5 m of sea
# ice is not what sits under a tundra grid cell.
COLUMN_BY_SLOT: dict[str, str] = {
    "land": "frozen_soil",
    "coastal": "frozen_soil",
    "open_ocean": "sea_ice",      # nominal only; see PINNED_SLOTS
    "marginal_ice": "sea_ice",
    "sea_ice": "sea_ice",
}

# Forcing periods the timescale figures sweep, in seconds. The synoptic band --
# a cloud system passing in 6 to 48 hours -- is where Arctic cloud radiative
# forcing actually lives, so it is resolved most finely.
DEFAULT_PERIODS_S: tuple[float, ...] = (
    1 * 3600.0, 3 * 3600.0, 6 * 3600.0, 12 * 3600.0, 24 * 3600.0,
    3 * 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0,
)
# The single period the headline numbers use. Twelve hours is the middle of the
# synoptic band and close to the median duration of a liquid-bearing overcast
# episode on this record; nothing about the model depends on the choice, which
# is why the timescale figure exists.
DEFAULT_PERIOD_S = 12 * 3600.0


def damping_depth_m(period_s: float, layer: Layer = ICE_ERA5) -> float:
    """Diffusive damping depth sqrt(kappa P / pi), in m.

    The depth at which a surface temperature oscillation of period ``period_s``
    has decayed by 1/e (Carslaw and Jaeger 1959, Section 2.6). Reported beside
    lam_G because it says WHY lam_G changes: 3.5 cm of ice at one hour, 17 cm
    at a day, 95 cm at a month.
    """
    kappa = layer.k_W_m_K / layer.rho_c_J_m3_K       # m2 s-1
    return float(np.sqrt(kappa * period_s / np.pi))


def surface_admittance(period_s: float,
                       column: tuple[Layer, ...] = (ICE_ERA5,)) -> complex:
    """Complex surface admittance of a layered column, W m-2 K-1.

    Solved as a thermal transmission line from the base upward. For a single
    slab of thickness h over a base held at a fixed temperature this is

        Y = k q coth(q h),      q = sqrt(i w rho c / k)

    and each layer above transforms the admittance below it by the standard
    finite-layer relation

        Y_top = Y_l (Y_below + Y_l tanh(q h)) / (Y_l + Y_below tanh(q h))

    with Y_l = k q the layer's own characteristic admittance. The two limits are
    both correct: as w -> 0 the whole thing collapses to the series resistance
    1 / sum(h_j / k_j), and at high frequency only the top layer is felt.

    The REAL part is the component in phase with the surface temperature
    perturbation, and that is the damping coefficient that belongs in the skin
    balance. The imaginary part is the quadrature (storage) component -- for a
    sinusoid it shifts the phase of the response rather than damping it, and it
    is why ``abs(Y)`` exceeds ``Y.real`` by up to sqrt(2).
    """
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    w = 2.0 * np.pi / period_s
    # Start at the base of the deepest layer, held at the freezing point: an
    # infinite admittance (a perfect temperature clamp), which the coth form
    # below reproduces without needing a special case.
    y = None
    for layer in reversed(column):
        q = np.sqrt(1j * w * layer.rho_c_J_m3_K / layer.k_W_m_K)
        y_char = layer.k_W_m_K * q                 # characteristic admittance
        t = np.tanh(q * layer.thickness_m)
        if y is None:
            y = y_char / t                         # fixed-temperature base
        else:
            y = y_char * (y + y_char * t) / (y_char + y * t)
    return complex(y)


def lambda_g(period_s: float = DEFAULT_PERIOD_S,
             column: tuple[Layer, ...] | str = "sea_ice") -> float:
    """lam_G, the in-phase conductive damping at one forcing period.

    ``column`` may be a key of ``COLUMNS`` or an explicit layer tuple.
    """
    col = COLUMNS[column] if isinstance(column, str) else column
    return float(surface_admittance(period_s, col).real)


def effective_period_s(lambda_g_target: float,
                       column: tuple[Layer, ...] | str = "sea_ice",
                       bracket_s: tuple[float, float] = (60.0,
                                                         3650 * 86400.0)
                       ) -> float:
    """Invert ``lambda_g``: the forcing period at which lam_G takes a value.

    ``surface_admittance`` is monotonically decreasing in the period, so this
    is a clean bisection. It is the diagnostic that turns "ERA5's residual
    damping is 3.9 W m-2 K-1" into "which is the ice column responding on a
    ~10-day timescale", which is a statement one can argue with. Returns NaN
    when the target is outside what the column can produce -- above the
    high-frequency limit at the bracket's fast end, or below the steady-state
    value, which no finite period can reach.
    """
    lo, hi = bracket_s
    f_lo, f_hi = lambda_g(lo, column), lambda_g(hi, column)
    if not (f_hi <= lambda_g_target <= f_lo):
        return float("nan")
    for _ in range(200):
        mid = np.sqrt(lo * hi)                  # bisect in log period
        if lambda_g(mid, column) > lambda_g_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def lambda_g_steady(column: tuple[Layer, ...] | str = "sea_ice") -> float:
    """The w -> 0 limit, 1 / sum(h/k). The value for a sustained forcing.

    1.35 W m-2 K-1 for ERA5's bare 1.5 m ice; 0.59 with 30 cm of snow in
    series. This is the SMALLEST lam_G, not the largest -- a slower
    perturbation has to push heat deeper and therefore feels more thermal
    resistance.
    """
    col = COLUMNS[column] if isinstance(column, str) else column
    return float(1.0 / sum(l.thickness_m / l.k_W_m_K for l in col))


# ----------------------------------------------------------------------------
# The turbulent terms: measured from ERA5, and cross-checked against theory
# ----------------------------------------------------------------------------
def q_sat_ice(T_K: float, p_Pa: float = P_SURFACE_REF) -> float:
    """Saturation specific humidity over ice, kg kg-1.

    Uses the IFS Tetens-form saturation vapour pressure over ice (Cy41r2 Part
    IV eq. 7.5 with the ice coefficients a3 = 22.587, a4 = -0.7 K, referenced
    to 611.21 Pa at the triple point).
    """
    e_sat = 611.21 * np.exp(22.587 * (T_K - 273.16) / (T_K - 0.7))
    return float(EPS_MOLAR * e_sat / (p_Pa - (1.0 - EPS_MOLAR) * e_sat))


def dq_sat_dT_ice(T_K: float, p_Pa: float = P_SURFACE_REF) -> float:
    """d(q_sat)/dT over ice, K-1, by Clausius-Clapeyron.

    q_sat * L_s / (R_v T^2), the standard form. About 4.9e-5 K-1 at 250 K,
    which is small in absolute terms but is multiplied by L_s / c_p ~ 2800.
    """
    return float(q_sat_ice(T_K, p_Pa) * L_SUBLIMATION / (R_VAPOUR * T_K**2))


def lambda_lh_clausius(lambda_sh: float, T_K: float,
                       p_Pa: float = P_SURFACE_REF) -> float:
    """lam_LH from lam_SH, assuming a saturated skin and C_E = C_H.

    Both bulk fluxes carry the same rho * C * U, so their ratio is the
    equilibrium (wet-surface) Bowen relation

        lam_LH / lam_SH = (L_s / c_p) * dq_sat/dT

    which is about 0.14 at 250 K. This is the RIGHT limit over snow and ice,
    because a frozen skin is saturated by construction -- there is no moisture
    availability factor to guess. It is the cross-check on the measured value,
    and the two agreeing is the evidence that regressing the latent heat flux
    on a TEMPERATURE difference is a legitimate reduction.
    """
    return float(lambda_sh * (L_SUBLIMATION / CP_AIR) * dq_sat_dT_ice(T_K, p_Pa))


def _slot_index(slot: int | str) -> int:
    """Accept a slot index or a class name; return the index."""
    if isinstance(slot, str):
        return SLOT_ORDER.index(slot)
    return int(slot)


def measured_coupling(A: Analysis, slot: int | str, flux_key: str,
                      population: str | None = None,
                      weighted: bool = True) -> dict:
    """Regression of one turbulent flux on (T_skin - T_2m), as a damping.

    Returns ``{"lambda": ..., "r2": ..., "inverted": ..., "n_hours": ...}``,
    with ``lambda`` positive for a flux that damps a skin warming.

    THE SIGN. ERA5 stores every surface flux positive DOWNWARD, so an upward
    sensible heat flux is negative and d(msshf)/d(dT) is negative; lam_SH is
    its negation, which makes it a positive damping coefficient in the sense of
    equation (1).

    THE DIRECTION OF THE FIT. This is d(flux)/d(dT), taken directly. The
    reciprocal of the other least-squares fit, 1/[d(dT)/d(flux)], overstates it
    by 1/r^2 -- 2.7x over pack ice, where r^2 = 0.37 -- and is returned as
    ``inverted`` only so the size of that error sits beside the number it would
    replace. See FIT_ORIENTS in ``turbulent_flux_response.py``.
    """
    si = _slot_index(slot)
    acc = A.acc(population)
    f = tfr.fit_pair(acc, si, flux_key, "dskt_t2m_K", weighted=weighted)
    fwd = f["x_on_y"]["slope"]            # d(flux)/d(dT), the correct one
    bwd = f["y_on_x"]["slope"]            # d(dT)/d(flux)
    return {
        "lambda": -fwd,
        "r2": f["r2"],
        "inverted": (-1.0 / bwd if np.isfinite(bwd) and bwd != 0.0 else np.nan),
        "inflation": f["inflation"],
        "n_hours": f["x_on_y"]["n_hours"],
    }


def measured_lambda_lw(A: Analysis, slot: int | str,
                       population: str | None = None) -> float:
    """d(LWU)/d(T_skin) from the accumulated moments, W m-2 K-1.

    The empirical counterpart of 4 eps sigma T^3, and a genuine test rather
    than a tautology: ERA5's upwelling longwave is
    eps sigma T_skin^4 + (1 - eps) LWD, so the regression also picks up
    (1 - eps) d(LWD)/d(T_skin). With eps ~ 0.99 that contribution is about
    0.01 * 8 = 0.08 W m-2 K-1, well inside the rounding, so agreement with the
    theoretical value is a real check on both.
    """
    return float(tfr.partial_slope(A.acc(population), _slot_index(slot),
                                   "lwu_W_m2", "skt_K"))


def air_coupling(A: Analysis, slot: int | str,
                 population: str | None = None,
                 control: tuple[str, ...] = ()) -> float:
    """alpha = d(T_2m)/d(T_skin), from the two DLR sensitivities.

    The fraction of a skin temperature perturbation the 2 m air temperature
    follows. Computed as the ratio of two slopes on the same predictor,
    [d(T_2m)/d(LWD)] / [d(T_skin)/d(LWD)], because that is the pathway the
    question is about -- a direct regression of T_2m on T_skin would be
    dominated by the enormous shared seasonal cycle.

    Near ONE over sea ice and land on this record, which is what collapses the
    turbulent damping in equation (3), and near 7 over open water, where the
    skin barely moves at all and the ratio stops being meaningful.
    """
    acc = A.acc(population)
    si = _slot_index(slot)
    d_skt = tfr.partial_slope(acc, si, "skt_K", "lwd_W_m2", control)
    d_t2m = tfr.partial_slope(acc, si, "t2m_K", "lwd_W_m2", control)
    if not np.isfinite(d_skt) or d_skt == 0.0:
        return float("nan")
    return float(d_t2m / d_skt)


# ----------------------------------------------------------------------------
# Assembling the four coefficients, and solving
# ----------------------------------------------------------------------------
class Lambdas(NamedTuple):
    """The four damping coefficients of equation (1), all W m-2 K-1.

    Every one is positive for a term that opposes a skin temperature
    perturbation. ``alpha`` is the air-coupling factor of equation (3), applied
    to the two turbulent terms when the partition is solved.
    """

    lw: float
    sh: float
    lh: float
    g: float
    alpha: float = 0.0
    # Provenance, carried so a figure or a table can say where the numbers came
    # from without the caller having to remember.
    slot_name: str = ""
    T_skin_K: float = float("nan")
    period_s: float = DEFAULT_PERIOD_S
    column: str = "sea_ice"
    sh_r2: float = float("nan")
    lh_r2: float = float("nan")
    n_hours: float = float("nan")
    lh_source: str = "measured"


def lambdas(A: Analysis, slot: int | str,
            period_s: float = DEFAULT_PERIOD_S,
            column: tuple[Layer, ...] | str | None = None,
            population: str | None = None,
            alpha: float = 0.0,
            lh_source: str = "measured",
            eps: float | None = None) -> Lambdas:
    """Build the four coefficients for one surface class.

    Parameters
    ----------
    slot : Class name (``"sea_ice"``) or slot index.
    period_s : Forcing period the conduction term is evaluated at. See
        ``surface_admittance`` -- this is the single most consequential option
        in the module, because lam_G spans a factor of forty across the range
        of periods a cloud system can have.
    column : Conducting column below the skin, a ``COLUMNS`` key or an explicit
        tuple of ``Layer``. ``None`` (the default) gives each class the column
        ``COLUMN_BY_SLOT`` assigns it, which is the only setting under which
        the land rows mean anything.
    alpha : Air-coupling factor of equation (3). 0 is the fixed-atmosphere
        limit; pass ``"measured"`` semantics by calling ``air_coupling`` and
        handing the result in.
    lh_source : ``"measured"`` regresses the latent heat flux on (T_skin-T_2m);
        ``"clausius"`` uses the saturated-skin relation to lam_SH instead.
    eps : Override the Planck-weighted broadband emissivity.
    """
    si = _slot_index(slot)
    name = SLOT_ORDER[si]
    if column is None:
        column = COLUMN_BY_SLOT.get(name, "sea_ice")
    T_skin = tfr.mean_of(A.acc(population), si, "skt_K")
    sh = measured_coupling(A, si, "shf_W_m2", population)
    lh = measured_coupling(A, si, "lhf_W_m2", population)
    lh_value = (lh["lambda"] if lh_source == "measured"
                else lambda_lh_clausius(sh["lambda"], T_skin))
    return Lambdas(
        lw=lambda_lw(T_skin, name, eps=eps),
        sh=sh["lambda"],
        lh=lh_value,
        g=lambda_g(period_s, column),
        alpha=alpha,
        slot_name=name,
        T_skin_K=T_skin,
        period_s=period_s,
        column=column if isinstance(column, str) else "custom",
        sh_r2=sh["r2"],
        lh_r2=lh["r2"],
        n_hours=sh["n_hours"],
        lh_source=lh_source,
    )


def partition(lam: Lambdas, d_dlr_W_m2: float = 1.0,
              alpha: float | None = None) -> dict:
    """Solve equation (3) and split the perturbation four ways.

    Parameters
    ----------
    lam : The four coefficients.
    d_dlr_W_m2 : The downwelling longwave perturbation. Positive warms the
        surface; the fractions do not depend on it, only ``dT_skin_K`` does.
    alpha : Air-coupling factor, defaulting to ``lam.alpha``. The turbulent
        coefficients are scaled by ``(1 - alpha)`` -- see equation (3) and the
        module docstring on why both alpha = 0 and the measured alpha are
        reported.

    Returns
    -------
    dict with the four fractions ``f_lw``, ``f_sh``, ``f_lh``, ``f_g`` (summing
    to one), the effective coefficients actually used, their sum
    ``lambda_sum``, the skin temperature response ``dT_skin_K``, and each
    term's flux change in W m-2.

    A NOTE ON WHAT "f_sh" MEANS HERE. It is the share of the perturbation the
    sensible heat flux absorbs *in this model*, with the atmosphere held to
    ``alpha``. It is NOT the same quantity as ``f_sh`` in
    ``turbulent_flux_response.partition``, which is a regression coefficient
    across the record and carries the air mass with it. ``fig_model_vs_era5``
    exists to put the two side by side.

    ``stable`` IS THE FLAG TO CHECK BEFORE QUOTING ANYTHING. With alpha > 1 the
    turbulent coefficients turn negative -- the air overshoots the surface, so
    the flux amplifies rather than damps the perturbation -- and once
    (alpha - 1)(lam_SH + lam_LH) exceeds lam_LW + lam_G the sum goes negative
    and the linearised balance has no stable solution at all: dT_skin flips
    sign and the fractions become meaningless. That is not a numerical
    accident, it is what a measured alpha of 6.9 over open water means. Every
    table and figure marks those rows rather than plotting the nonsense.
    """
    a = lam.alpha if alpha is None else alpha
    # The turbulent terms respond to (T_skin - T_2m); if the air follows the
    # surface with slope alpha, only the fraction (1 - alpha) of a skin
    # perturbation reaches the bulk driver. Radiation and conduction respond to
    # the skin temperature itself and are untouched.
    eff = {
        "lw": lam.lw,
        "sh": lam.sh * (1.0 - a),
        "lh": lam.lh * (1.0 - a),
        "g": lam.g,
    }
    total = sum(eff.values())
    stable = np.isfinite(total) and total > 0.0
    dT = d_dlr_W_m2 / total if total != 0.0 else float("nan")
    out = {
        "lambda_sum": total,
        "dT_skin_K": dT,
        "alpha": a,
        "stable": bool(stable),
        "d_dlr_W_m2": d_dlr_W_m2,
        "lambdas": lam,
    }
    for term, value in eff.items():
        out[f"f_{term}"] = value / total if total != 0.0 else float("nan")
        out[f"lambda_eff_{term}"] = value
        out[f"dflux_{term}_W_m2"] = value * dT
    return out


# Stack order and colours, matched to fig_response_partition in
# turbulent_flux_response.py so the same term is the same colour in both.
MODEL_TERMS: tuple[tuple[str, str, str], ...] = (
    ("f_lw", "Upwelling LW", "#B2182B"),
    ("f_sh", "Sensible heat", "#4C72B0"),
    ("f_lh", "Latent heat", "#55A868"),
    ("f_g", "Conduction", "#8172B2"),
)

# The perturbation the headline numbers are quoted for: the cloud longwave
# forcing this project has been measuring, taken negative so the numbers read
# as a cooling. Nothing in the model depends on it -- the fractions are
# independent of amplitude and only dT_skin scales.
DEFAULT_D_DLR = -27.0

# Classes the tables and figures cover, in reading order. "all" is excluded on
# purpose: pooling surfaces whose lam_SH differs by a factor of five is the
# average this whole module exists to avoid.
MODEL_SLOTS: tuple[str, ...] = CLASS_ORDER


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def coefficient_table(A: Analysis,
                      period_s: float = DEFAULT_PERIOD_S,
                      column: tuple[Layer, ...] | str | None = None,
                      population: str | None = None,
                      slots: tuple[str, ...] = MODEL_SLOTS,
                      lh_source: str = "measured",
                      d_dlr_W_m2: float = DEFAULT_D_DLR) -> list[dict]:
    """One row per surface class: the four lambdas, alpha, and the partition.

    Both alpha limits are solved for every row, so the bracket is in the table
    rather than requiring two calls. Each row also carries the INVERSE problem:
    given ERA5's own total damping and its measured alpha, what lam_G would the
    balance need, and what forcing timescale does that correspond to? That is
    ``lambda_g_implied`` / ``period_implied_s``, and it is the closest thing
    here to an independent estimate of the timescale the record is responding
    on.
    """
    rows = []
    for name in slots:
        col_key = column if column is not None else COLUMN_BY_SLOT.get(
            name, "sea_ice")
        col = COLUMNS[col_key] if isinstance(col_key, str) else col_key
        lam = lambdas(A, name, period_s, col_key, population,
                      lh_source=lh_source)
        a_meas = air_coupling(A, name, population)
        era5 = tfr.partition(A.acc(population), _slot_index(name))
        d_skt = era5["dskt_dlwd"]
        era5_sum = (1.0 / d_skt if np.isfinite(d_skt) and d_skt != 0.0
                    else float("nan"))
        # What lam_G would have to be for the model at the MEASURED alpha to
        # reproduce ERA5's own total damping. Everything else in the sum is
        # already pinned, so this is a subtraction, not a fit.
        g_implied = era5_sum - lam.lw - (lam.sh + lam.lh) * (1.0 - a_meas)
        rows.append({
            "slot": name,
            "label": SLOT_LABELS[name],
            "column": col,
            "column_key": col_key if isinstance(col_key, str) else "custom",
            "pinned": name in PINNED_SLOTS,
            "lam": lam,
            "alpha_measured": a_meas,
            "fixed_air": partition(lam, d_dlr_W_m2, alpha=0.0),
            "measured_air": partition(lam, d_dlr_W_m2, alpha=a_meas),
            "era5": era5,
            "era5_lambda_sum": era5_sum,
            "lambda_g_implied": g_implied,
            "period_implied_s": effective_period_s(g_implied, col),
        })
    return rows


def _period_label(p_s: float) -> str:
    """A short human label for a forcing period."""
    if p_s < 2 * 86400:
        return f"{p_s / 3600:.4g} h"
    if p_s < 90 * 86400:
        return f"{p_s / 86400:.4g} d"
    return f"{p_s / (365 * 86400):.3g} yr"


def print_report(A: Analysis,
                 period_s: float = DEFAULT_PERIOD_S,
                 column: tuple[Layer, ...] | str | None = None,
                 population: str | None = None,
                 d_dlr_W_m2: float = DEFAULT_D_DLR,
                 lh_source: str = "measured") -> None:
    """Every number the figures draw, in a form that can be pasted into a note."""
    pop = population or A.args.population
    rows = coefficient_table(A, period_s, column, pop, lh_source=lh_source,
                             d_dlr_W_m2=d_dlr_W_m2)
    ice_col = COLUMNS["sea_ice"]

    print("\n" + "=" * 80)
    print("LINEARISED SKIN-LAYER SURFACE ENERGY BALANCE")
    print("  dLWD = (lam_LW + lam_SH + lam_LH + lam_G) dT_skin")
    print("=" * 80)
    print(f"  population       "
          f"{'liquid-bearing overcast' if pop == 'cloud' else 'all sky'}")
    print(f"  seasons          {A.used[0]}-{A.used[-1]}, region {A.args.region}")
    print(f"  forcing period   {_period_label(period_s)}  "
          f"(over ERA5's bare ice: lam_G = {lambda_g(period_s, ice_col):.2f} "
          f"W m-2 K-1, damping depth {100 * damping_depth_m(period_s):.1f} cm)")
    print("  columns          " + ",  ".join(
        f"{r['label']}: {r['column_key']}" for r in rows))
    print(f"  perturbation     {d_dlr_W_m2:+.1f} W m-2")

    # --- the coefficients -------------------------------------------------
    print("\n  THE FOUR COEFFICIENTS  [W m-2 K-1]")
    print(f"    {'class':<20}{'T_skin':>7}{'lam_LW':>8}{'(meas)':>8}"
          f"{'lam_SH':>8}{'r2':>6}{'lam_LH':>8}{'r2':>6}{'(C-C)':>7}"
          f"{'lam_G':>7}{'sum':>8}")
    for r in rows:
        lam = r["lam"]
        meas_lw = measured_lambda_lw(A, r["slot"], pop)
        cc = lambda_lh_clausius(lam.sh, lam.T_skin_K)
        flag = " *" if r["pinned"] else ""
        print(f"    {r['label']:<20}{lam.T_skin_K:>7.1f}{lam.lw:>8.2f}"
              f"{meas_lw:>8.2f}{lam.sh:>8.2f}{lam.sh_r2:>6.2f}"
              f"{lam.lh:>8.2f}{lam.lh_r2:>6.2f}{cc:>7.2f}"
              f"{lam.g:>7.2f}{sum(lam[:4]):>8.2f}{flag}")
    print("      lam_LW  theory, 4*eps*sigma*T^3 with the Planck-weighted "
          "IFS emissivity")
    print("      (meas)  d(LWU)/d(T_skin) from the moments -- the one "
          "INDEPENDENT check in this table")
    print("      lam_SH, lam_LH  d(flux)/d(T_skin - T_2m) regressions, "
          "measured, per class")
    print("      (C-C)   saturated-skin Clausius-Clapeyron estimate of lam_LH, "
          "for comparison")
    print("      lam_G   theory, Re[k q coth(q h)] for that class's column at "
          "the period above")
    print("      *       ERA5 prescribes this surface temperature; lam_G is "
          "effectively infinite there")

    # --- the partition, both alpha limits ---------------------------------
    for key, title in (("fixed_air", "FIXED ATMOSPHERE (alpha = 0)"),
                       ("measured_air", "MEASURED AIR COUPLING (alpha "
                                        "from the record)")):
        print(f"\n  PARTITION -- {title}")
        print(f"    {'class':<20}{'alpha':>8}{'f_LW':>8}{'f_SH':>8}"
              f"{'f_LH':>8}{'f_G':>8}{'dT_skin':>9}{'ERA5 dT':>9}")
        for r in rows:
            p = r[key]
            era5_dT = r["era5"]["dskt_dlwd"] * d_dlr_W_m2
            if not p["stable"]:
                msg = (f"-- no stable balance: sum(lam) = "
                       f"{p['lambda_sum']:.1f} --")
                print(f"    {r['label']:<20}{p['alpha']:>8.2f}"
                      f"{msg:>40}{era5_dT:>9.2f}")
                continue
            print(f"    {r['label']:<20}{p['alpha']:>8.2f}"
                  f"{p['f_lw']:>8.3f}{p['f_sh']:>8.3f}"
                  f"{p['f_lh']:>8.3f}{p['f_g']:>8.3f}"
                  f"{p['dT_skin_K']:>9.2f}{era5_dT:>9.2f}")
        if key == "fixed_air":
            print("      the four fractions sum to 1 by construction")
        else:
            print("      alpha = [dT_2m/dLWD] / [dT_skin/dLWD]. alpha -> 1 "
                  "collapses the turbulent terms;")
            print("      alpha > 1 turns them into an amplifier, and far "
                  "enough past 1 there is no")
            print("      equilibrium at all -- which is what the open-water "
                  "row is telling you.")
        print(f"      dT_skin for {d_dlr_W_m2:+.1f} W m-2; 'ERA5 dT' is "
              f"d(T_skin)/d(LWD) x the same perturbation")

    # --- the closure check -------------------------------------------------
    print("\n  CLOSURE AGAINST THE REGRESSIONS")
    print(f"    {'class':<20}{'model sum':>11}{'ERA5 sum':>10}{'ratio':>7}"
          f"{'model f_SH':>12}{'ERA5 f_SH':>11}{'lam_G needed':>14}"
          f"{'implies':>10}")
    for r in rows:
        model_sum = sum(r["lam"][:4])
        e5 = r["era5_lambda_sum"]
        print(f"    {r['label']:<20}{model_sum:>11.2f}{e5:>10.2f}"
              f"{model_sum / e5:>7.2f}"
              f"{r['fixed_air']['f_sh']:>12.3f}{r['era5']['f_sh']:>11.3f}"
              f"{r['lambda_g_implied']:>14.2f}"
              f"{_period_label(r['period_implied_s']):>10}")
    print("      'ERA5 sum' is 1 / [d(T_skin)/d(LWD)]: the total damping the")
    print("      record actually exhibits, air-mass response included. ratio")
    print("      > 1 means the fixed-atmosphere model over-damps, which is")
    print("      the alpha effect of equation (3), not an inconsistency.")
    print("      'lam_G needed' is what conduction would have to supply for")
    print("      the MEASURED-alpha model to reproduce 'ERA5 sum', and")
    print("      'implies' inverts the admittance to the forcing period that")
    print("      would give it. That is a diagnostic, not a measurement.")

    # --- the timescale sweep ----------------------------------------------
    print("\n  lam_G AND THE SEA-ICE PARTITION vs FORCING PERIOD  "
          "(sea ice, alpha = 0)")
    lam_ref = lambdas(A, "sea_ice", DEFAULT_PERIOD_S, "sea_ice", pop,
                      lh_source=lh_source)
    print(f"    {'period':>10}{'depth':>10}{'lam_G':>8}{'f_LW':>8}{'f_SH':>8}"
          f"{'f_LH':>8}{'f_G':>8}{'dT_skin':>9}")
    for p_s in DEFAULT_PERIODS_S + (float("inf"),):
        if np.isinf(p_s):
            g, depth, label = lambda_g_steady(ice_col), None, "steady"
        else:
            g = lambda_g(p_s, ice_col)
            depth = 100 * damping_depth_m(p_s)
            label = _period_label(p_s)
        pr = partition(lam_ref._replace(g=g), d_dlr_W_m2, alpha=0.0)
        depth_s = "--" if depth is None else f"{depth:.1f} cm"
        print(f"    {label:>10}{depth_s:>10}{g:>8.2f}{pr['f_lw']:>8.3f}"
              f"{pr['f_sh']:>8.3f}{pr['f_lh']:>8.3f}{pr['f_g']:>8.3f}"
              f"{pr['dT_skin_K']:>9.2f}")
    print("      lam_G FALLS as the forcing slows: a slower perturbation "
          "reaches deeper")
    print("      and meets more thermal resistance in series with the skin. "
          "So the")
    print("      turbulent share is SMALLEST for a fast perturbation and "
          "largest for a")
    print("      sustained one -- the opposite of the usual intuition.")

    # --- published comparisons --------------------------------------------
    print("\n  PUBLISHED RESPONSE SHARES, for comparison")
    print("    Miller et al. 2017, Summit Greenland, regression on "
          "(LW_down + SW_net):")
    print("      annual  LW_up 77%   sensible 11%   latent 1.5%   "
          "conduction 10%   storage 6%")
    print("      winter  LW_up 65-85%, conduction 23%; summer conduction 9%")
    print("    Sledd et al. 2025, MOSAiC central Arctic: during winter ice "
          "growth the")
    print("      forcing change appears in upwelling longwave, sensible heat "
          "and")
    print("      subsurface heat flux; in summer melt the surface temperature "
          "is fixed")
    print("      and the response goes into melt instead.")
    print("    Both are COVARIANCE partitions with the air mass free, so they "
          "belong")
    print("    beside the MEASURED-alpha table above, not the alpha = 0 one.")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

NOTE_BOX = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.86,
                edgecolor="#BBBBBB")


def _resolve_column(column, slot_name: str = "sea_ice") -> tuple[Layer, ...]:
    """Turn a ``column`` argument into a layer tuple.

    ``None`` means "whatever ``COLUMN_BY_SLOT`` gives this class", which is the
    default everywhere and the only setting under which the land rows are
    physically meaningful.
    """
    if column is None:
        column = COLUMN_BY_SLOT.get(slot_name, "sea_ice")
    return COLUMNS[column] if isinstance(column, str) else column


def _subtitle(A: Analysis, pop: str, period_s: float,
              column: tuple[Layer, ...] | str | None) -> str:
    """The population line the ``tfr`` figures use, plus the model settings.

    Kept to ONE extra line. The header block is measured, not hard-coded, but
    it is already three lines deep before this module adds anything, and the
    detail that does not fit belongs in ``_column_note`` on the panel it
    applies to rather than stacked above the figure.
    """
    ice_g = lambda_g(period_s, COLUMNS["sea_ice"])
    return (f"{tfr._figure_subtitle(A, pop)}\n"
            f"linearised skin balance  |  forcing period "
            f"{_period_label(period_s)}  |  "
            f"$\\lambda_G$ = {ice_g:.1f} W m$^{{-2}}$ K$^{{-1}}$ over ERA5's "
            f"bare 1.5 m ice\n")


def _column_note(column: tuple[Layer, ...] | str | None) -> str:
    """Which conducting column each class was given. For an in-axes box."""
    if column is None:
        return ("conduction column per class:\n"
                + "\n".join(f"  {SLOT_LABELS[k]}: {COLUMN_BY_SLOT[k]}"
                             for k in MODEL_SLOTS))
    col = _resolve_column(column)
    return "conduction: " + " + ".join(l.label for l in col)


def fig_lambda_bars(A: Analysis, out_dir=None, dpi: int | None = None,
                    period_s: float = DEFAULT_PERIOD_S,
                    column: tuple[Layer, ...] | str | None = None,
                    population: str | None = None,
                    lh_source: str = "measured"):
    """The four coefficients per surface class, and what they sum to.

    (a) is the stack: each class's total damping, split by term, so the height
    is the model's 1 / [dT_skin/dLWD] and the segments are the partition. (b)
    is the same information as fractions, which is the direct answer to "how
    much goes into turbulence".
    """
    pop = population or A.args.population
    rows = coefficient_table(A, period_s, column, pop, lh_source=lh_source)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    x = np.arange(len(rows))
    labels = [r["label"].replace(" ", "\n", 1) for r in rows]

    # --- (a) the coefficients, stacked ------------------------------------
    ax = axes[0]
    bottom = np.zeros(len(rows))
    for key, name, colour in MODEL_TERMS:
        vals = np.array([r["fixed_air"][f"lambda_eff_{key[2:]}"] for r in rows])
        ax.bar(x, vals, bottom=bottom, color=colour, label=name,
               edgecolor="white", linewidth=0.6)
        bottom += vals
    # ERA5's own total damping, 1 / [dT_skin/dLWD]. A marker rather than a bar,
    # because it is a different estimator and should not look like a fifth term.
    era5_sum = np.array([r["era5_lambda_sum"] for r in rows])
    ax.plot(x, era5_sum, "o", color="black", ms=8, mfc="white", mew=1.8,
            label=r"ERA5 $1/[dT_{skin}/dLWD]$", zorder=5)
    for xi, (tot, e5) in enumerate(zip(bottom, era5_sum)):
        ax.text(xi, tot, f"{tot:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel(r"damping coefficient  [W m$^{-2}$ K$^{-1}$]")
    # Headroom for the legend, which would otherwise sit on the value labels.
    ax.set_ylim(0.0, 1.45 * float(np.nanmax(bottom)))
    ax.set_title(r"(a) the four $\lambda$, and the total damping", fontsize=11)
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
    ax.text(0.985, 0.985, _column_note(column), transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0, bbox=NOTE_BOX)

    # --- (b) the fractions, both alpha limits -----------------------------
    # A row whose measured alpha admits no stable balance is left BLANK on the
    # right rather than drawn: a stack of fractions that do not sum to one, or
    # sum to one with negative members, is worse than an absence.
    ax = axes[1]
    w = 0.38
    for off, key, hatch in ((-w / 2, "fixed_air", ""),
                            (+w / 2, "measured_air", "///")):
        bottom = np.zeros(len(rows))
        for key_f, name, colour in MODEL_TERMS:
            vals = np.array([r[key][key_f] if r[key]["stable"] else np.nan
                             for r in rows])
            ax.bar(x + off, np.nan_to_num(vals), w, bottom=bottom,
                   color=colour, hatch=hatch, edgecolor="white", linewidth=0.6)
            bottom += np.nan_to_num(vals)
    ax.axhline(1.0, color="#666666", lw=0.9, ls=":")
    for xi, r in enumerate(rows):
        if not r["measured_air"]["stable"]:
            ax.text(xi + w / 2, 0.5, "no stable\nbalance at\nthis $\\alpha$",
                    ha="center", va="center", fontsize=7.4, color="#8a5a00",
                    bbox=NOTE_BOX)
    ax.set_ylabel("fraction of the DLR perturbation absorbed")
    ax.set_ylim(0.0, 1.45)
    ax.set_title(r"(b) the partition: left $\alpha=0$, right $\alpha$ measured",
                 fontsize=11)
    handles = [Patch(facecolor=c, label=n) for _, n, c in MODEL_TERMS]
    handles.append(Patch(facecolor="#BBBBBB", hatch="///",
                         label=r"hatched: $\alpha$ measured"))
    ax.legend(handles=handles, fontsize=8.0, loc="upper right",
              ncol=2, framealpha=0.92)

    for ai, ax in enumerate(axes):
        ax.set_xticks(x)
        # Panel (b) carries the measured alpha under each class name, since
        # that is the number its right-hand bar depends on entirely.
        ax.set_xticklabels(
            labels if ai == 0 else
            [f"{lb}\n$\\alpha$={r['alpha_measured']:.2f}"
             for lb, r in zip(labels, rows)], fontsize=8.6)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    top = tfr._header_block(
        fig,
        "The four-term linearised skin balance, by surface class",
        _subtitle(A, pop, period_s, column),
        note="(b) right-hand bars: where $\\alpha \\approx 1$ the air follows "
             "the surface and the turbulent terms drop out.\n"
             "The two are bounds on the same quantity, not competing "
             "estimates -- see the module docstring.",
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return tfr._save(fig, A, f"lsm_lambda_bars_{pop}", out_dir, dpi)


def fig_partition_vs_timescale(A: Analysis, out_dir=None,
                               dpi: int | None = None,
                               period_s: float = DEFAULT_PERIOD_S,
                               slots: tuple[str, ...] = ("sea_ice",
                                                         "marginal_ice"),
                               column: tuple[Layer, ...] | str | None = None,
                               population: str | None = None,
                               n_period: int = 160,
                               lh_source: str = "measured"):
    """The partition as a function of how long the forcing lasts.

    THE FIGURE THE MODULE IS FOR. lam_LW, lam_SH and lam_LH do not depend on
    the forcing timescale but lam_G spans a factor of forty across it, so the
    answer to "how much goes into turbulence" is a curve, not a number. The
    synoptic band -- 6 to 48 hours, where Arctic cloud forcing actually lives
    -- is shaded.

    ``period_s`` is accepted so this figure takes the same keywords as the
    others in ``ALL_FIGURES``, but it only marks the reference period in the
    subtitle: the whole point of the figure is that lam_G is swept.
    """
    pop = population or A.args.population
    periods = np.logspace(np.log10(0.5 * 3600), np.log10(400 * 86400),
                          n_period)
    hours = periods / 3600.0

    fig, axes = plt.subplots(2, len(slots), figsize=(6.6 * len(slots), 8.2),
                             sharex=True, squeeze=False)
    for ci, name in enumerate(slots):
        col = _resolve_column(column, name)
        lam0 = lambdas(A, name, period_s, col, pop, lh_source=lh_source)
        lam_g_curve = np.array([lambda_g(p, col) for p in periods])
        parts = [partition(lam0._replace(g=g), DEFAULT_D_DLR, alpha=0.0)
                 for g in lam_g_curve]

        # --- top: the fractions, stacked --------------------------------
        ax = axes[0, ci]
        stack = np.array([[p[k] for p in parts] for k, _, _ in MODEL_TERMS])
        ax.stackplot(hours, stack, colors=[c for _, _, c in MODEL_TERMS],
                     labels=[n for _, n, _ in MODEL_TERMS], alpha=0.92)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of the perturbation")
        ax.set_title(f"{SLOT_LABELS[name]}"
                     f"   ($\\lambda_{{SH}}$ = {lam0.sh:.1f}, "
                     f"$\\lambda_{{LW}}$ = {lam0.lw:.1f} "
                     f"W m$^{{-2}}$ K$^{{-1}}$)", fontsize=10.5)
        if ci == 0:
            ax.legend(loc="center left", fontsize=8.4, framealpha=0.9)

        # --- bottom: lam_G, the total, and the skin response -------------
        ax = axes[1, ci]
        ax.plot(hours, lam_g_curve, color="#8172B2", lw=2.0,
                label=r"$\lambda_G$")
        ax.plot(hours, [p["lambda_sum"] for p in parts], color="black", lw=1.6,
                ls="--", label=r"$\sum \lambda$")
        ax.axhline(lam0.sh, color="#4C72B0", lw=1.3, ls=":",
                   label=r"$\lambda_{SH}$ (timescale independent)")
        ax.axhline(lambda_g_steady(col), color="#8172B2", lw=1.0, ls="-.",
                   label=r"$\lambda_G$ steady state")
        ax.set_yscale("log")
        ax.set_ylabel(r"W m$^{-2}$ K$^{-1}$")
        ax.set_xlabel("forcing period  [hours]")

        # The skin response on a twin axis: the number a reader wants.
        ax2 = ax.twinx()
        ax2.plot(hours, [abs(p["dT_skin_K"]) for p in parts],
                 color="#C44E52", lw=1.4, alpha=0.85)
        ax2.set_ylabel(f"$|\\Delta T_{{skin}}|$ for "
                       f"{DEFAULT_D_DLR:+.0f} W m$^{{-2}}$  [K]",
                       color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        if ci == 0:
            ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9)

        for ax in (axes[0, ci], axes[1, ci]):
            ax.set_xscale("log")
            # The synoptic band: where Arctic cloud radiative forcing lives.
            ax.axvspan(6, 48, color="#FFD27F", alpha=0.30, zorder=0)
            for p_s, tag in ((12.0 * 3600, "12 h"), (86400.0, "1 d"),
                             (7 * 86400.0, "1 wk"), (30 * 86400.0, "1 mo")):
                ax.axvline(p_s / 3600, color="#888888", lw=0.7, ls=":")
            ax.grid(alpha=0.2, lw=0.5)
            ax.set_axisbelow(True)
        # Named above the axes rather than inside the stack, where a label
        # sits on whichever colour happens to be there at that period.
        axes[0, ci].text(np.sqrt(6 * 48), 1.01, "synoptic band",
                         ha="center", va="bottom", fontsize=8.2,
                         color="#7a5a00")

    top = tfr._header_block(
        fig,
        "How much goes into turbulence? It depends on how long the cloud stays",
        _subtitle(A, pop, period_s, column),
        note="$\\lambda_G$ is the in-phase surface admittance of the ice "
             "column and FALLS as the forcing slows (deeper penetration, more "
             "resistance in series)\n"
             "so the conductive share is largest for a fast perturbation and "
             "the turbulent share largest for a slow one. $\\alpha = 0$.",
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return tfr._save(fig, A, f"lsm_partition_vs_timescale_{pop}", out_dir, dpi)


def fig_forcing_response(A: Analysis, out_dir=None, dpi: int | None = None,
                         period_s: float = DEFAULT_PERIOD_S,
                         d_dlr_W_m2: float = DEFAULT_D_DLR,
                         periods_s: tuple[float, ...] | None = None,
                         column: tuple[Layer, ...] | str | None = None,
                         population: str | None = None,
                         lh_source: str = "measured"):
    """The answer with a number on it: skin cooling, and where the energy goes.

    (a) the skin temperature change a ``d_dlr_W_m2`` perturbation produces, per
    class, at several forcing timescales, with ERA5's own regression response
    marked. (b) the flux change in each term, in W m-2, for the middle
    timescale -- the physically legible form of the partition, since the four
    bars sum to the perturbation itself.

    ``period_s`` sets the middle of the three timescales, and is the one panel
    (b) uses. Pass ``periods_s`` to choose all three explicitly.
    """
    pop = population or A.args.population
    if periods_s is None:
        periods_s = (3600.0, period_s, 7 * 86400.0)
    slots = MODEL_SLOTS
    x = np.arange(len(slots))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # --- (a) skin response vs timescale, per class ------------------------
    ax = axes[0]
    w = 0.8 / len(periods_s)
    greys = plt.cm.viridis(np.linspace(0.15, 0.8, len(periods_s)))
    for pi, p_s in enumerate(periods_s):
        vals, off = [], (pi - (len(periods_s) - 1) / 2) * w
        for name in slots:
            lam = lambdas(A, name, p_s, _resolve_column(column, name), pop,
                          lh_source=lh_source)
            vals.append(partition(lam, d_dlr_W_m2, alpha=0.0)["dT_skin_K"])
        # The legend quotes lam_G over ERA5's bare ice; the land classes use
        # their own column, which is why the label says which surface it is for.
        ax.bar(x + off, vals, w, color=greys[pi],
               label=f"{_period_label(p_s)}  (ice $\\lambda_G$="
                     f"{lambda_g(p_s, COLUMNS['sea_ice']):.1f})")
    era5 = [tfr.partition(A.acc(pop), _slot_index(n))["dskt_dlwd"] * d_dlr_W_m2
            for n in slots]
    ax.plot(x, era5, "o", color="black", ms=8, mfc="white", mew=1.8,
            label="ERA5 regression", zorder=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"$\\Delta T_{{skin}}$ for {d_dlr_W_m2:+.0f} "
                  f"W m$^{{-2}}$  [K]")
    # Every bar hangs below zero, so the space above it is free for the legend.
    lo = min(min(era5), ax.get_ylim()[0])
    ax.set_ylim(1.12 * lo, -0.42 * lo)
    ax.set_title("(a) skin temperature response, $\\alpha = 0$", fontsize=11)
    ax.legend(fontsize=8.0, framealpha=0.92, loc="upper center", ncol=2)

    # --- (b) the flux changes, in W m-2 -----------------------------------
    ax = axes[1]
    p_mid = periods_s[len(periods_s) // 2]
    w2 = 0.2
    for ki, (key, name, colour) in enumerate(MODEL_TERMS):
        vals = []
        for n in slots:
            lam = lambdas(A, n, p_mid, _resolve_column(column, n), pop,
                          lh_source=lh_source)
            vals.append(partition(lam, d_dlr_W_m2,
                                  alpha=0.0)[f"dflux_{key[2:]}_W_m2"])
        ax.bar(x + (ki - 1.5) * w2, vals, w2, color=colour, label=name)
    ax.axhline(d_dlr_W_m2, color="#666666", lw=1.1, ls="--",
               label=f"the perturbation, {d_dlr_W_m2:+.0f} W m$^{{-2}}$")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(r"flux change  [W m$^{-2}$]")
    ax.set_ylim(1.10 * d_dlr_W_m2, -0.42 * d_dlr_W_m2)
    ax.set_title(f"(b) where the energy goes, {_period_label(p_mid)} forcing "
                 f"period\n(each term signed as the perturbation; the four "
                 f"sum to it)", fontsize=10)
    ax.legend(fontsize=8.0, framealpha=0.92, loc="upper center", ncol=3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([SLOT_LABELS[n].replace(" ", "\n", 1)
                            for n in slots], fontsize=8.6)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    top = tfr._header_block(
        fig,
        f"Response to a {d_dlr_W_m2:+.0f} W m$^{{-2}}$ downwelling "
        f"longwave perturbation",
        _subtitle(A, pop, p_mid, column),
        note="(a) open circles are ERA5's own $dT_{skin}/dLWD$ times the same "
             "perturbation -- a covariance with the air mass free, so it "
             "cools further than the fixed-atmosphere bars.\n"
             "Open ocean is the exception, and for the opposite reason: ERA5 "
             "prescribes its surface temperature, so the skin barely moves.",
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return tfr._save(fig, A, f"lsm_forcing_response_{pop}", out_dir, dpi)


def fig_model_vs_era5(A: Analysis, out_dir=None, dpi: int | None = None,
                      period_s: float = DEFAULT_PERIOD_S,
                      column: tuple[Layer, ...] | str | None = None,
                      population: str | None = None,
                      lh_source: str = "measured"):
    """The closure check: does the model reproduce ERA5's own sensitivities?

    Three tests, in increasing difficulty:

    (a) lam_LW. Theory 4 eps sigma T^3 against the measured d(LWU)/d(T_skin).
        These MUST agree -- if they do not, something is wrong with the
        emissivity or the units, not with the physics.
    (b) The total damping. The model's sum(lambda) against ERA5's
        1 / [dT_skin/dLWD]. These need not agree, and the ratio is the
        air-coupling effect.
    (c) The turbulent share. The model's f_SH at both alpha limits against
        ERA5's regression f_SH. The bracket should contain the regression
        value; where it does not, the missing physics is named on the panel.
    """
    pop = population or A.args.population
    rows = coefficient_table(A, period_s, column, pop, lh_source=lh_source)
    x = np.arange(len(rows))
    labels = [r["label"].replace(" ", "\n", 1) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))

    # --- (a) lam_LW: theory vs measured -----------------------------------
    ax = axes[0]
    theory = np.array([r["lam"].lw for r in rows])
    meas = np.array([measured_lambda_lw(A, r["slot"], pop) for r in rows])
    ax.bar(x - 0.2, theory, 0.4, color="#B2182B", label=r"$4\epsilon\sigma T^3$")
    ax.bar(x + 0.2, meas, 0.4, color="#EE9999",
           label=r"measured $d(LWU)/dT_{skin}$")
    for xi, (t, m) in enumerate(zip(theory, meas)):
        ax.text(xi, max(t, m) * 1.01, f"{100 * (m / t - 1):+.1f}%",
                ha="center", va="bottom", fontsize=8.0)
    ax.set_ylabel(r"$\lambda_{LW}$  [W m$^{-2}$ K$^{-1}$]")
    ax.set_ylim(0.0, 1.55 * float(np.nanmax(np.r_[theory, meas])))
    ax.set_title("(a) the term that must close", fontsize=11)
    ax.legend(fontsize=8.2, loc="upper left")

    # --- (b) total damping -------------------------------------------------
    ax = axes[1]
    model_sum = np.array([sum(r["lam"][:4]) for r in rows])
    era5_sum = np.array([r["era5_lambda_sum"] for r in rows])
    ax.bar(x - 0.2, model_sum, 0.4, color="#4C72B0",
           label=r"model $\sum\lambda$, $\alpha=0$")
    ax.bar(x + 0.2, era5_sum, 0.4, color="#9FB8D8",
           label=r"ERA5 $1/[dT_{skin}/dLWD]$")
    ax.set_ylabel(r"total damping  [W m$^{-2}$ K$^{-1}$]")
    ax.set_yscale("log")
    ax.set_ylim(0.7 * float(np.nanmin(np.r_[model_sum, era5_sum])),
                6.0 * float(np.nanmax(np.r_[model_sum, era5_sum])))
    ax.set_title("(b) the total, which need not close", fontsize=11)
    ax.legend(fontsize=8.2, loc="upper left")
    # The ratio on the page, so panel (c)'s alpha bracket has a size attached.
    for xi, (m, e) in enumerate(zip(model_sum, era5_sum)):
        ax.text(xi, max(m, e) * 1.15, f"{m / e:.1f}$\\times$", ha="center",
                va="bottom", fontsize=8.0, color="#333333")

    # --- (c) the turbulent share, bracketed --------------------------------
    ax = axes[2]
    lo = np.array([r["measured_air"]["f_sh"] for r in rows])
    hi = np.array([r["fixed_air"]["f_sh"] for r in rows])
    era5_fsh = np.array([r["era5"]["f_sh"] for r in rows])
    for xi, (a, b) in enumerate(zip(lo, hi)):
        ax.plot([xi, xi], [a, b], color="#4C72B0", lw=7.0, solid_capstyle="butt",
                alpha=0.55, zorder=2)
    ax.plot(x, hi, "_", color="#1F3D6B", ms=22, mew=2.4,
            label=r"model $f_{SH}$, $\alpha = 0$", zorder=3)
    ax.plot(x, lo, "_", color="#7FA6D9", ms=22, mew=2.4,
            label=r"model $f_{SH}$, $\alpha$ measured", zorder=3)
    ax.plot(x, era5_fsh, "o", color="black", ms=8, mfc="white", mew=1.8,
            label=r"ERA5 regression $f_{SH}$", zorder=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.11, color="#55A868", lw=1.2, ls="--",
               label="Miller+ 2017 Summit, 11%")
    ax.set_ylabel(r"share of the perturbation taken by $SH$")
    ax.set_ylim(-0.35, 1.0)
    ax.set_title("(c) the answer to the question, bracketed", fontsize=11)
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.4)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    top = tfr._header_block(
        fig,
        "Does the linearised model reproduce ERA5's own response?",
        _subtitle(A, pop, period_s, column),
        note="(a) is a units-and-emissivity check and closes. (b) and (c) are "
             "not expected to close: the model holds the atmosphere\n"
             "fixed and the regression does not, which is the whole content "
             "of the $\\alpha$ bracket in (c).",
        title_fs=14,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return tfr._save(fig, A, f"lsm_model_vs_era5_{pop}", out_dir, dpi)


def fig_ground_admittance(out_dir=None, dpi: int | None = None,
                          A: Analysis | None = None):
    """lam_G against forcing period, for both columns, with the two limits.

    Standalone -- it needs no ERA5 data, only the column properties, so it can
    be drawn before anything is loaded. Included because the frequency
    dependence of lam_G is the least familiar part of the model and the easiest
    to get backwards.
    """
    periods = np.logspace(np.log10(0.2 * 3600), np.log10(1000 * 86400), 400)
    hours = periods / 3600.0

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    styles = {"sea_ice": ("#8172B2", "-"), "sea_ice_snow": ("#55A868", "--")}
    for key, col in COLUMNS.items():
        colour, ls = styles.get(key, ("#666666", ":"))
        y = np.array([surface_admittance(p, col).real for p in periods])
        ax.plot(hours, y, color=colour, ls=ls, lw=2.0,
                label=" + ".join(l.label for l in col))
        ax.axhline(lambda_g_steady(col), color=colour, lw=0.9, ls=":")
        ax.text(hours[-1], lambda_g_steady(col),
                f"  steady {lambda_g_steady(col):.2f}", fontsize=8.2,
                color=colour, va="center")
    # The semi-infinite high-frequency asymptote, sqrt(w k rho c / 2), which is
    # what the curve follows until the damping depth reaches the ice base.
    w = 2.0 * np.pi / periods
    ax.plot(hours, np.sqrt(w * ICE_ERA5.k_W_m_K * ICE_ERA5.rho_c_J_m3_K / 2.0),
            color="#BBBBBB", lw=1.2, ls="-.",
            label=r"semi-infinite ice, $\sqrt{\omega k \rho c / 2}$")
    # The IFS's own top-layer conductance, 2 k / D_1 with D_1 = 0.07 m.
    ax.plot([1.0], [2 * ICE_ERA5.k_W_m_K / 0.07], "*", color="#B2182B", ms=16,
            zorder=5, label=r"IFS $2k/D_1$ = 58.0 (top layer, $D_1$=0.07 m)")

    for p_s, tag in ((3600.0, "1 h"), (12 * 3600.0, "12 h"), (86400.0, "1 d"),
                     (7 * 86400.0, "1 wk"), (30 * 86400.0, "1 mo"),
                     (365 * 86400.0, "1 yr")):
        ax.axvline(p_s / 3600, color="#CCCCCC", lw=0.7, ls=":")
        ax.text(p_s / 3600, 0.52, tag, rotation=90, fontsize=7.6,
                color="#888888", ha="right", va="bottom")
    ax.axvspan(6, 48, color="#FFD27F", alpha=0.30, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.15, 1.0e5)          # room for the steady-state labels
    ax.set_ylim(0.35, 400.0)
    ax.set_xlabel("forcing period  [hours]")
    ax.set_ylabel(r"$\lambda_G$ = Re$[k q \coth(q h)]$  "
                  r"[W m$^{-2}$ K$^{-1}$]")
    ax.set_title(r"The conductive damping is frequency dependent, "
                 r"over a factor of 40", fontsize=12)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.22, lw=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(0.98, 0.96,
            "shaded: the synoptic band, where Arctic cloud forcing lives\n"
            "$\\lambda_G$ FALLS as the forcing slows -- a slower perturbation\n"
            "penetrates deeper and meets more resistance in series",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            bbox=NOTE_BOX)
    fig.tight_layout()
    if out_dir is not None and A is not None:
        return tfr._save(fig, A, "lsm_ground_admittance", out_dir, dpi)
    return fig


ALL_FIGURES = (
    fig_lambda_bars,
    fig_partition_vs_timescale,
    fig_forcing_response,
    fig_model_vs_era5,
)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """This module's own options, on top of every option ``tfr`` accepts."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--period-hours", type=float,
                        default=DEFAULT_PERIOD_S / 3600.0, metavar="H",
                        help="Forcing period the conduction term is evaluated "
                             f"at (default {DEFAULT_PERIOD_S / 3600:g} h). "
                             "lam_G spans a factor of 40 across the plausible "
                             "range; see surface_admittance.")
    parser.add_argument("--column", choices=tuple(COLUMNS), default="sea_ice",
                        help="Conducting column below the skin. 'sea_ice' is "
                             "what ERA5 has (bare 1.5 m ice); 'sea_ice_snow' "
                             "puts 30 cm of snow in series, which is what is "
                             "really there (default sea_ice).")
    parser.add_argument("--d-dlr", type=float, default=DEFAULT_D_DLR,
                        metavar="W",
                        help="The DLR perturbation the response is quoted for "
                             f"(default {DEFAULT_D_DLR:g} W m-2). The "
                             "fractions do not depend on it.")
    parser.add_argument("--lh-source", choices=("measured", "clausius"),
                        default="measured",
                        help="How lam_LH is obtained (default measured).")
    argv = sys.argv[1:] if argv is None else list(argv)
    # This module's parser carries add_help=False so the four options above can
    # ride on top of every turbulent_flux_response option. The cost is that -h
    # would otherwise reach only the parent parser and these four would be
    # invisible, so print them first and let the parent print the rest and exit.
    if "-h" in argv or "--help" in argv:
        print(parser.format_help())
        print("Plus every option of turbulent_flux_response.py:\n")
    known, rest = parser.parse_known_args(argv)
    tfr_args = tfr.parse_args(rest)
    for k, v in vars(known).items():
        setattr(tfr_args, k, v)
    return tfr_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    A = tfr.prepare(args=args)
    period_s = args.period_hours * 3600.0
    print_report(A, period_s, args.column, d_dlr_W_m2=args.d_dlr,
                 lh_source=args.lh_source)
    if not args.no_figures:
        out = args.output_dir
        fig_ground_admittance(out_dir=out, dpi=args.dpi, A=A)
        for fn in ALL_FIGURES:
            fn(A, out_dir=out, dpi=args.dpi, period_s=period_s,
               column=args.column, lh_source=args.lh_source)
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
