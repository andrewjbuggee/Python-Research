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
# MEASURED: ERA5's tclw/tciw here are quantised to exact multiples of
# 2**-15 kg m-2 = 0.0305176 g m-2 (a GRIB binary scale factor), so the smallest
# non-zero path the archive can express is 0.03052 g m-2. A threshold BELOW that
# quantum selects exactly the same cell-hours as "> 0" and guards nothing. To
# actually drop the single-quantum population, use >= 0.031.
DEFAULT_LWP_MIN_G = 0.03
DEFAULT_IWP_MIN_G = 0.03

# Condensate ceilings: below this, the phase is treated as absent. These
# ARE deliberately below the 0.0305 g m-2 quantum above, so the test means
# exactly "not a single quantum of this phase is present".
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
