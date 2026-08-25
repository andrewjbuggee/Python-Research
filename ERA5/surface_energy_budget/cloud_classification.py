#!/usr/bin/env python3
"""Cloud STATE and cloud PHASE filters, to pair with ``surface_classification.py``.

Two independent axes
====================
State, from total cloud cover ``tcc`` alone::

    clear      tcc <= tol                     nothing in the column
    cloudy     tcc >  --cloudy-threshold      partly to fully covered
    overcast   tcc >= 1 - tol                 completely covered
    all_sky    (no filter)                    every scene, clouds or not

These are deliberately NOT mutually exclusive: overcast is a subset of cloudy,
and all_sky contains everything. That is the point -- each is a conditioning
choice, and reading them side by side is how the cloud effect shows up.

A NOTE ON THE TERM "ALL-SKY"
----------------------------
In the radiation literature all-sky means the actual atmosphere with whatever
clouds are in it, i.e. NO cloud filtering. It is the counterpart to clear-sky,
which is the hypothetical flux computed with the clouds removed; the difference
of the two is the cloud radiative effect. ERA5 archives both, ``msdwlwrf``
(all-sky) and ``msdwlwrfcs`` (clear-sky), for exactly that pairing.

A "tcc > 0.5" filter is therefore NOT all-sky -- it is a cloudy-scene filter,
and is named ``cloudy`` here. ``all_sky`` is kept for its standard meaning so
that a number labelled all-sky in a figure means what a reader will assume.

No single threshold for "cloudy" is standard in the ERA5 literature. The choice
matters less than it looks: ERA5's Arctic tcc distribution is strongly U-shaped,
piling up near 0 and near 1 with relatively little in between, so moving the cut
within the sparse middle moves few scenes. Worth confirming on your own data with
``--report-tcc``.

Phase, from the column condensate of a cloudy scene::

    liquid   LWP > --lwp-min  and  IWP < --iwp-max-liquid
    ice      IWP > --iwp-min  and  LWP < --lwp-max-ice
    mixed    LWP > --lwp-min  and  IWP > --iwp-min

These three ARE mutually exclusive, but not exhaustive: a scene whose condensate
falls between the "essentially none" ceiling and the "definitely present" floor
(e.g. LWP of 0.01 g m-2, above --lwp-max-ice but below --lwp-min) belongs to no
phase and is counted as unclassified rather than forced into one. Widen the gap
by lowering --lwp-min, or close it by raising --lwp-max-ice.

Note that ``cloud_phase_masks`` gives liquid and mixed a SHARED liquid floor.
When the two categories need floors of their own -- a high LWP bar for "liquid
only" and a low one for "mixed" -- use ``liquid_mixed_masks`` further down,
which takes four independent thresholds and enforces disjointness explicitly.

UNITS
=====
Every threshold here is in **g m-2**, and every ``*_g`` variable name says so.
ERA5 stores ``tclw``/``tciw`` in kg m-2, so callers must convert (multiply by
1000) before calling into this module. The defaults follow the trace threshold
already used in ``plot_monthly_lwp_maps.py`` (0.03 g m-2), which is the level at
which ERA5's near-zero background condensate is treated as no cloud at all.

If you meant kg m-2 -- a far heavier cloud, 0.03 kg m-2 = 30 g m-2 -- pass
``--lwp-min 30 --iwp-min 30 --lwp-max-ice 1 --iwp-max-liquid 1``.
"""

from __future__ import annotations

import argparse

import numpy as np

# ----------------------------------------------------------------------------
# Cloud state
# ----------------------------------------------------------------------------
STATE_ORDER: tuple[str, ...] = ("clear", "cloudy", "overcast", "all_sky")

STATE_LABELS: dict[str, str] = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "overcast": "Overcast",
    "all_sky": "All-sky",
}

# all_sky is the reference, so it is given the one warm colour; the other three
# run pale to dark with increasing cloud.
STATE_COLORS: dict[str, str] = {
    "clear": "#8ecae6",
    "cloudy": "#adb5bd",
    "overcast": "#495057",
    "all_sky": "#e07a5f",
}

# tcc is stored packed, so an exactly-clear or exactly-overcast scene can come
# back as 1e-5 or 0.99999. Same round-off insurance as the lsm tolerance.
DEFAULT_TCC_TOL = 1e-4
DEFAULT_CLOUDY_THRESHOLD = 0.5


# ----------------------------------------------------------------------------
# Cloud phase
# ----------------------------------------------------------------------------
PHASE_ORDER: tuple[str, ...] = ("liquid", "ice", "mixed")

PHASE_LABELS: dict[str, str] = {
    "liquid": "Liquid only",
    "ice": "Ice only",
    "mixed": "Mixed phase",
}

PHASE_COLORS: dict[str, str] = {
    "liquid": "#1f5fa8",
    "ice": "#7fd4e8",
    "mixed": "#8e5ea2",
}

# Condensate floors: above this, the phase is present. Matches the trace
# threshold in plot_monthly_lwp_maps.py.
# MEASURED: ERA5's tclw/tciw are quantised, because GRIB stores each field with
# a binary scale factor. NOTE 2**-15 below is TWO TO THE POWER OF MINUS FIFTEEN,
# i.e. 1/32768 = 3.05e-5 kg m-2 -- NOT 2e-15. The Python operator reads like
# scientific notation and the two differ by ten orders of magnitude.
#
# The scale factor is chosen PER GRIB MESSAGE, so the step is not one fixed
# number. It varies by variable, by file, and even between time steps of the
# SAME file: era5_seb_barrow_20000113.nc carries all three steps below across
# its 24 hours. Surveyed over all 642 barrow files:
#
#     tclw   2**-15 = 0.0305176 g m-2   642 files   (uniform)
#     tciw   2**-14 = 0.0610352 g m-2     1 file
#            2**-15 = 0.0305176 g m-2   618 files
#            2**-16 = 0.0152588 g m-2    23 files
#
# So the finest path this archive expresses is 0.0152588 g m-2 -- HALF the
# figure this note used to quote -- and the coarsest step is 0.0610352.
#
# WHAT THAT MEANS FOR A THRESHOLD. Below 0.0152588 a threshold selects exactly
# the same cell-hours as "> 0" and guards nothing. Above it the guard becomes
# archive-dependent: 0.031 drops the single-quantum population where the step is
# 2**-15, drops the first TWO quanta where it is 2**-16, and drops none where it
# is 2**-14. No single value means "one quantum" everywhere, and any value large
# enough to clear one 2**-14 quantum (> 0.0610352) clears four 2**-16 ones.
#
# This note used to recommend ">= 0.031" for that purpose. It is withdrawn: the
# target it names does not exist. Single-quantum values are 2.00% of non-zero
# tclw and 1.43% of non-zero tciw, too small a population to steer a threshold
# by in any case. Choose a threshold for a physical reason instead -- an
# instrument detection limit, or the level below which the condensate is
# radiatively irrelevant -- and say which. Read 0.03 g m-2 below as "essentially
# zero", NOT as "exactly one quantum".
DEFAULT_LWP_MIN_G = 0.03
DEFAULT_IWP_MIN_G = 0.03

# Condensate ceilings: below this, the phase is treated as absent. These ARE
# deliberately below the FINEST step measured above (0.0152588 g m-2), not just
# below the common one, so the test means exactly "not a single quantum of this
# phase is present" whichever scale factor the message happened to use.
DEFAULT_LWP_MAX_ICE_G = 0.001
DEFAULT_IWP_MAX_LIQUID_G = 0.001


def cloud_state_masks(
    tcc: np.ndarray,
    tcc_tol: float = DEFAULT_TCC_TOL,
    cloudy_threshold: float = DEFAULT_CLOUDY_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Boolean mask per cloud state. Overlapping by design -- see module docs.

    Parameters
    ----------
    tcc : Total cloud cover on [0, 1], any shape. NaN is excluded from every
        state, including ``all_sky``.
    """
    finite = np.isfinite(tcc)
    with np.errstate(invalid="ignore"):
        return {
            "clear": finite & (tcc <= tcc_tol),
            "cloudy": finite & (tcc > cloudy_threshold),
            "overcast": finite & (tcc >= 1.0 - tcc_tol),
            "all_sky": finite,
        }


def cloud_phase_masks(
    lwp_g: np.ndarray,
    iwp_g: np.ndarray,
    lwp_min_g: float = DEFAULT_LWP_MIN_G,
    iwp_min_g: float = DEFAULT_IWP_MIN_G,
    lwp_max_ice_g: float = DEFAULT_LWP_MAX_ICE_G,
    iwp_max_liquid_g: float = DEFAULT_IWP_MAX_LIQUID_G,
) -> dict[str, np.ndarray]:
    """Boolean mask per cloud phase. Mutually exclusive, not exhaustive.

    Parameters
    ----------
    lwp_g, iwp_g : Column liquid and ice water path in **g m-2** (ERA5 stores
        kg m-2; convert before calling).
    """
    finite = np.isfinite(lwp_g) & np.isfinite(iwp_g)
    with np.errstate(invalid="ignore"):
        has_liquid = lwp_g > lwp_min_g
        has_ice = iwp_g > iwp_min_g
        no_liquid = lwp_g < lwp_max_ice_g
        no_ice = iwp_g < iwp_max_liquid_g
        return {
            "liquid": finite & has_liquid & no_ice,
            "ice": finite & has_ice & no_liquid,
            "mixed": finite & has_liquid & has_ice,
        }


# ----------------------------------------------------------------------------
# Liquid-only vs mixed-phase, with INDEPENDENT thresholds
# ----------------------------------------------------------------------------
# ``cloud_phase_masks`` above ties both categories to one condensate floor:
# liquid-only and mixed-phase must share ``lwp_min_g``. That is the right model
# when the aim is to label every scene on one consistent set of cuts, and the
# wrong one when the two categories answer different questions -- "is there a
# radiatively significant liquid deck with no ice in it" wants a HIGH liquid
# floor and a LOW ice ceiling, while "does this cloud hold both species" wants
# a low floor on each. Those need four numbers, not two.
DEFAULT_LIQUID_LWP_MIN_G = 5.0    # liquid only: liquid must be this substantial
DEFAULT_LIQUID_IWP_MAX_G = 1.0    # liquid only: and ice this close to absent
DEFAULT_MIXED_LWP_MIN_G = 1.0     # mixed: both species merely need to be
DEFAULT_MIXED_IWP_MIN_G = 1.0     # mixed: present at all


def check_liquid_mixed_disjoint(liquid_iwp_max_g: float,
                                mixed_iwp_min_g: float) -> None:
    """Raise unless no scene can satisfy both category definitions at once.

    The two sets are

        liquid only   LWP > a  and  IWP < b
        mixed phase   LWP > c  and  IWP > d

    so their intersection is ``{LWP > max(a, c)} and {d < IWP < b}``. LWP is
    unbounded above, so the first factor is NEVER empty and the LWP floors
    cannot separate the categories at all -- no matter how far apart ``a`` and
    ``c`` are set. Disjointness rests entirely on the IWP axis, and the whole
    condition is

        b <= d          i.e.  liquid_iwp_max_g <= mixed_iwp_min_g

    VERIFIED by exhaustive sweep: over all 625 threshold quadruples drawn from
    {0, 0.5, 1, 2, 5} against a dense (LWP, IWP) grid, "the sets intersect" and
    "b > d" agree in every case.

    Equality is allowed and is the tightest useful setting: it leaves no IWP gap
    between the categories, so the only scene falling between them is one with
    IWP EXACTLY equal to the shared value, which both strict tests reject.
    """
    if liquid_iwp_max_g > mixed_iwp_min_g:
        raise ValueError(
            f"liquid-only and mixed-phase would overlap: a scene with "
            f"LWP above both floors and IWP between {mixed_iwp_min_g:g} and "
            f"{liquid_iwp_max_g:g} g m-2 matches BOTH definitions, so it would "
            f"be counted twice.\n"
            f"  Require liquid_iwp_max <= mixed_iwp_min "
            f"({liquid_iwp_max_g:g} <= {mixed_iwp_min_g:g} is false).\n"
            f"  Raising the LWP floors cannot fix this -- LWP is unbounded "
            f"above, so the categories are separated on the IWP axis alone."
        )


def liquid_mixed_masks(
    lwp_g: np.ndarray,
    iwp_g: np.ndarray,
    liquid_lwp_min_g: float = DEFAULT_LIQUID_LWP_MIN_G,
    liquid_iwp_max_g: float = DEFAULT_LIQUID_IWP_MAX_G,
    mixed_lwp_min_g: float = DEFAULT_MIXED_LWP_MIN_G,
    mixed_iwp_min_g: float = DEFAULT_MIXED_IWP_MIN_G,
) -> dict[str, np.ndarray]:
    """Liquid-only and mixed-phase masks on four independent thresholds.

        liquid only   LWP > liquid_lwp_min_g  and  IWP < liquid_iwp_max_g
        mixed phase   LWP > mixed_lwp_min_g   and  IWP > mixed_iwp_min_g

    Guaranteed DISJOINT -- ``check_liquid_mixed_disjoint`` runs first and raises
    rather than let a scene be counted twice. Deliberately NOT exhaustive: at
    the defaults a scene with LWP 3 and IWP 0.5 g m-2 has too little liquid for
    the first and too little ice for the second, and belongs to neither. That
    population is real and should be reported by the caller, not absorbed.

    Parameters
    ----------
    lwp_g, iwp_g : Column liquid and ice water path in **g m-2** (ERA5 stores
        kg m-2; convert before calling).
    """
    check_liquid_mixed_disjoint(liquid_iwp_max_g, mixed_iwp_min_g)
    finite = np.isfinite(lwp_g) & np.isfinite(iwp_g)
    with np.errstate(invalid="ignore"):
        return {
            "liquid": finite & (lwp_g > liquid_lwp_min_g)
                             & (iwp_g < liquid_iwp_max_g),
            "mixed": finite & (lwp_g > mixed_lwp_min_g)
                            & (iwp_g > mixed_iwp_min_g),
        }


# ----------------------------------------------------------------------------
# Phase by FRACTION of cloud water path
# ----------------------------------------------------------------------------
# An alternative to the absolute-threshold schemes above. Define
#
#     CWP = LWP + IWP        (cloud water path, the total condensate)
#
# and classify on each species' SHARE of it rather than on its magnitude:
#
#     liquid only   LWP/CWP >= liquid_fraction_min   (default 0.90)
#     ice only      IWP/CWP >= ice_fraction_min      (default 0.90)
#     mixed phase   everything else
#
# Because LWP/CWP and IWP/CWP sum to exactly 1, the three categories are
# EXHAUSTIVE and MUTUALLY EXCLUSIVE by construction over every scene holding
# condensate -- there is no gap to report and no overlap to guard, given the one
# condition below. That is the scheme's main attraction over the absolute one:
# it asks "what is this cloud made of" rather than "how much of each does it
# hold", so a thin cloud and a thick one are classified on the same footing.
#
# The trade-off is the mirror image. A scene with 0.05 g m-2 of liquid and
# nothing else is 100% liquid by share and would be called a liquid cloud, which
# is why the minimum paths below are not optional decoration: they are what stops
# the ratio being computed on numerical dust. See ``min_lwp_g`` / ``min_iwp_g``.
DEFAULT_LIQUID_FRACTION_MIN = 0.90
DEFAULT_ICE_FRACTION_MIN = 0.90

# A species below its floor is treated as ABSENT -- contributing neither to CWP
# nor to its own share -- rather than the scene being discarded. So a cloud with
# LWP 50 and IWP 0.05 g m-2 is liquid-only outright, instead of liquid-only by a
# ratio of 0.999 that happens to clear the cut. A scene where BOTH species are
# below their floor has no cloud water at all and is left unclassified.
DEFAULT_MIN_LWP_G = 0.1
DEFAULT_MIN_IWP_G = 0.1


def check_fraction_thresholds_disjoint(liquid_fraction_min: float,
                                       ice_fraction_min: float) -> None:
    """Raise unless the two fractional cuts cannot both be met at once.

    Writing f = LWP/CWP, the categories are ``f >= a`` (liquid only) and
    ``1 - f >= b``, i.e. ``f <= 1 - b`` (ice only). They intersect on
    ``a <= f <= 1 - b``, which is empty exactly when

        a > 1 - b      i.e.      liquid_fraction_min + ice_fraction_min > 1

    Note the inequality is STRICT. At a + b == 1 -- say 0.5 and 0.5 -- the two
    intervals still touch at the single point f = 0.5, and a scene sitting
    exactly there would be counted in both. The default pair sums to 1.8.
    """
    if not 0.0 < liquid_fraction_min <= 1.0:
        raise ValueError(f"liquid_fraction_min must be in (0, 1], got "
                         f"{liquid_fraction_min:g}")
    if not 0.0 < ice_fraction_min <= 1.0:
        raise ValueError(f"ice_fraction_min must be in (0, 1], got "
                         f"{ice_fraction_min:g}")
    if liquid_fraction_min + ice_fraction_min <= 1.0:
        raise ValueError(
            f"liquid-only and ice-only would overlap: with "
            f"liquid_fraction_min {liquid_fraction_min:g} and "
            f"ice_fraction_min {ice_fraction_min:g}, any cloud whose liquid "
            f"share falls between {liquid_fraction_min:g} and "
            f"{1 - ice_fraction_min:g} satisfies BOTH and would be counted "
            f"twice.\n"
            f"  Require liquid_fraction_min + ice_fraction_min > 1 "
            f"({liquid_fraction_min:g} + {ice_fraction_min:g} = "
            f"{liquid_fraction_min + ice_fraction_min:g})."
        )


def fraction_phase_masks(
    lwp_g: np.ndarray,
    iwp_g: np.ndarray,
    liquid_fraction_min: float = DEFAULT_LIQUID_FRACTION_MIN,
    ice_fraction_min: float = DEFAULT_ICE_FRACTION_MIN,
    min_lwp_g: float = DEFAULT_MIN_LWP_G,
    min_iwp_g: float = DEFAULT_MIN_IWP_G,
) -> dict[str, np.ndarray]:
    """Phase from each species' share of the cloud water path.

        CWP = LWP + IWP, counting only species above their own floor

        liquid only   LWP/CWP >= liquid_fraction_min
        ice only      IWP/CWP >= ice_fraction_min
        mixed phase   any remaining scene that holds cloud water

    Mutually exclusive AND exhaustive over scenes with CWP > 0; a scene with
    both species below their floors is in none of the three, which is the only
    unclassified case. Returns the three masks plus ``cwp_g`` and
    ``liquid_fraction`` so a caller can bin or report on them without redoing
    the flooring.

    Parameters
    ----------
    lwp_g, iwp_g : Column liquid and ice water path in **g m-2** (ERA5 stores
        kg m-2; convert before calling).
    """
    check_fraction_thresholds_disjoint(liquid_fraction_min, ice_fraction_min)
    finite = np.isfinite(lwp_g) & np.isfinite(iwp_g)
    with np.errstate(invalid="ignore"):
        lwp_eff = np.where(finite & (lwp_g > min_lwp_g), lwp_g, 0.0)
        iwp_eff = np.where(finite & (iwp_g > min_iwp_g), iwp_g, 0.0)
    cwp = lwp_eff + iwp_eff
    has_cloud = finite & (cwp > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        liquid_fraction = np.where(has_cloud, lwp_eff / np.where(cwp > 0.0, cwp,
                                                                1.0), np.nan)
    liquid = has_cloud & (liquid_fraction >= liquid_fraction_min)
    ice = has_cloud & ((1.0 - liquid_fraction) >= ice_fraction_min)
    return {
        "liquid": liquid,
        "ice": ice,
        "mixed": has_cloud & ~liquid & ~ice,
        "cwp_g": cwp,
        "liquid_fraction": liquid_fraction,
    }


def add_cloud_state_args(parser: argparse.ArgumentParser) -> None:
    """Cloud state thresholds, shared across scripts."""
    group = parser.add_argument_group("cloud state")
    group.add_argument(
        "--cloudy-threshold", type=float, default=DEFAULT_CLOUDY_THRESHOLD,
        metavar="F",
        help=(
            "Total cloud cover above which a scene counts as cloudy (default "
            f"{DEFAULT_CLOUDY_THRESHOLD:g}). Note this is the 'cloudy' category, "
            "NOT all-sky: all-sky means no cloud filter at all and is reported "
            "separately."
        ),
    )
    group.add_argument(
        "--tcc-tol", type=float, default=DEFAULT_TCC_TOL, metavar="TOL",
        help=(
            "How close tcc must be to 0 or 1 to count as exactly clear or exactly "
            f"overcast (default {DEFAULT_TCC_TOL:g}). Packing round-off "
            "insurance, not physics."
        ),
    )


def add_cloud_phase_args(parser: argparse.ArgumentParser) -> None:
    """Cloud phase thresholds, all in g m-2."""
    group = parser.add_argument_group("cloud phase (all thresholds in g m-2)")
    group.add_argument(
        "--lwp-min", type=float, default=DEFAULT_LWP_MIN_G, metavar="G",
        help=f"LWP above which liquid is present (default {DEFAULT_LWP_MIN_G:g}).",
    )
    group.add_argument(
        "--iwp-min", type=float, default=DEFAULT_IWP_MIN_G, metavar="G",
        help=f"IWP above which ice is present (default {DEFAULT_IWP_MIN_G:g}).",
    )
    group.add_argument(
        "--lwp-max-ice", type=float, default=DEFAULT_LWP_MAX_ICE_G, metavar="G",
        help=(
            "LWP below which an ice-only scene is treated as having no liquid "
            f"(default {DEFAULT_LWP_MAX_ICE_G:g})."
        ),
    )
    group.add_argument(
        "--iwp-max-liquid", type=float, default=DEFAULT_IWP_MAX_LIQUID_G,
        metavar="G",
        help=(
            "IWP below which a liquid-only scene is treated as having no ice "
            f"(default {DEFAULT_IWP_MAX_LIQUID_G:g})."
        ),
    )
