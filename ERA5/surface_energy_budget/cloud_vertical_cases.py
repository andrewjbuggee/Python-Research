#!/usr/bin/env python3
"""Rachel Gillespie's radiative cloud cases A-F, applied to ERA5 profiles.

Ports the phase-structure detection in ``identify_radiative_cases.py`` from
ARM cloud radar/lidar columns onto ERA5 pressure-level profiles, so the two can
be compared. The case definitions, the check order, the purity tests and the
quality filters are all reproduced; what changes is where the phase labels come
from and how "fraction of the cloud" is measured.

    A  pure liquid, single layer
    B  liquid-topped mixed-phase: liquid -> mixed, no ice, single layer
    C  liquid-topped: liquid -> ice, no mixed, single layer
    D  lower pure-liquid + upper ice, two layers
    E  lower liquid-topped liquid+mixed + upper ice, two layers
    F  liquid-topped: liquid -> mixed -> ice, single layer

FOUR THINGS DIFFER FROM THE ARM VERSION, AND THEY MATTER
========================================================

1. PHASE COMES FROM WATER CONTENT, NOT A RETRIEVAL
   ARM's ``cloud_phase_mplgr`` assigns a discrete phase per range gate. ERA5
   gives continuous liquid and ice water content, so a phase has to be derived:

       clear   both LWC and IWC at or below --phase-threshold
       liquid  LWC above,  IWC at or below
       ice     IWC above,  LWC at or below
       mixed   both above

   The threshold is on water content DENSITY (g m-3), not on the layer's mass
   path, because density is the intensive quantity a radar range gate responds
   to and it does not change when the layer thickness does. Default 1e-3
   g m-3, near a cloud radar's detection floor. Everything downstream is
   sensitive to it, so ``sensitivity_to_threshold`` sweeps it.

2. FRACTIONS ARE MASS-WEIGHTED BY DEFAULT, NOT LEVEL COUNTS
   This is the change that makes the comparison mean anything at all. Rachel's
   purity rule -- a contaminating phase may be at most 10% of the cloud's
   pixels -- works on ~500 ARM range gates. ERA5 has 23 levels, of which a
   typical cloud occupies 3-8, and 10% of 8 rounds down to ZERO. On level
   counts the rule stops being "mostly one phase" and silently becomes "no
   trace of any other phase at all", which no real ERA5 profile satisfies, and
   cases B, C and F collapse to nearly nothing.

   So the default basis is MASS: frac_ice = IWP / (LWP + IWP) integrated over
   the cloud, which is resolution-independent and is what "90% liquid" means
   physically. ``basis="levels"`` reproduces the literal ARM arithmetic, and is
   worth running once to see the degeneracy for yourself.

3. NO TEMPORAL POST-PROCESSING
   Rachel bridges gaps up to 90 s and drops events shorter than 30 min, both
   defined on a 30-second sampling grid. ERA5 is hourly: one sample already
   spans 60 minutes, so a 90-second bridge and a 30-minute minimum are both
   below the sampling interval and cannot be applied. Every ERA5 statistic here
   is therefore the equivalent of Rachel's "hours passing column filters"
   column -- her RAW count -- and must be compared against that one, never
   against her ">= 30 min" column.

4. ONE FILTER IS VACUOUS IN BOTH CODES
   "Clear sky below cloud base" is computed in the ARM version as
   ``all(phase[:cloud_base] == CLEAR)`` where ``cloud_base`` is the first
   non-clear index. That is true by construction and rejects nothing. It is
   kept here for faithfulness, and reported, so the agreement is visible rather
   than assumed.

QUALITY FILTERS
===============
Applied to every case, as in the original, with the ERA5 source in brackets:

    total cloud cover >= --min-cloud-fraction   [tcc, single level]
    no snow anywhere in the column              [cswc above the threshold]
    no drizzle/rain anywhere in the column      [crwc above the threshold]
    precipitation rate < 0.01 mm/hr             [tp, single level]
    liquid water path > 2 g m-2                 [integrated clwc]
    clear sky below cloud base                  [vacuous; see above]
"""

from __future__ import annotations

import numpy as np

from convert_specific_to_absolute import (
    G_M_S2,
    layer_thickness_pa,
    moist_density,
)

# Phase codes, matching the ARM encoding for the subset ERA5 can express.
CLEAR, LIQUID, ICE, MIXED = 0, 1, 2, 3

# Rachel's detection parameters, unchanged.
PURITY_THRESHOLD = 0.10
MIXED_ZONE_MIN_PURITY = 0.60
PRECIP_THRESHOLD_MM_HR = 0.01
LWP_THRESHOLD_G_M2 = 2.0

# Water-content density above which a level counts as containing that phase.
# g m-3. A cloud radar's detection floor is around 1e-3 to 1e-2 g m-3.
# MEASURED storage floors on this archive, as specific content:
#     clwc  2^-22 = 2.384e-07 kg/kg   -> 3.2e-04 g m-3 at rho = 1.35
#     ciwc  2^-26 = 1.490e-08 kg/kg   -> 2.0e-05 g m-3
# The two are NOT the same: ERA5 stores ice 16x finer than liquid. A threshold
# below about 3e-4 g m-3 therefore sits INSIDE the liquid floor -- it would keep
# every non-zero liquid value while still cutting ice, biasing the phase balance
# by construction.
#
# 1e-3 g m-3 sits ~3x above the liquid floor and cuts a near-identical share of
# each species (66.3% of non-zero LWC kept, 68.1% of IWC), so it discriminates
# on physics rather than on storage resolution. Raising it to a radar-like
# 5e-3 to 1e-2 g m-3 is also defensible; going below 5e-4 is not.
DEFAULT_PHASE_THRESHOLD_G_M3 = 1e-3

# How a level's phase is decided. See assign_phase.
DEFAULT_PHASE_RULE = "fraction"

# Rachel's numeric codes and letters, kept identical so results line up with
# her CSV columns and summary table without a translation step.
CASE_LABELS = {3: "A", 1: "B", 2: "C", 4: "D", 5: "E", 6: "F"}
CASE_ORDER = [3, 1, 2, 4, 5, 6]                       # A, B, C, D, E, F
CASE_COLORS = {"A": "#F9A825", "B": "#5E35B1", "C": "#00838F",
               "D": "#28A745", "E": "#00B5B8", "F": "#D62728"}
CASE_DESC = {
    "A": "Pure liquid  (single layer)",
    "B": "Liquid-topped MPC: liquid -> mixed  (single layer)",
    "C": "Liquid-topped: liquid -> ice  (single layer)",
    "D": "Lower pure-liquid + upper ice  (two layers)",
    "E": "Lower liquid-topped liq+mix + upper ice  (two layers)",
    "F": "Liquid-topped: liquid -> mixed -> ice  (single layer)",
}


# ----------------------------------------------------------------------------
# Vectorised equivalents of the ARM scalar helpers
# ----------------------------------------------------------------------------
def _first_last(mask: np.ndarray):
    """First and last True index along the last axis; -1 where the row is all
    False. Returns ``(first, last, any_)``."""
    n = mask.shape[-1]
    any_ = mask.any(axis=-1)
    first = np.where(any_, np.argmax(mask, axis=-1), -1)
    last = np.where(any_, n - 1 - np.argmax(mask[..., ::-1], axis=-1), -1)
    return first, last, any_


def count_runs(mask: np.ndarray) -> np.ndarray:
    """Number of contiguous True runs along the last axis.

    A run starts at index 0 if that element is True, and thereafter wherever a
    False is followed by a True -- which is exactly the count of 0->1
    transitions in the integer view.
    """
    m = mask.astype(np.int8)
    return m[..., 0] + (np.diff(m, axis=-1) == 1).sum(axis=-1)


def zone_pure(phase: np.ndarray, val: int, min_purity: float,
              limit: np.ndarray | None = None) -> np.ndarray:
    """Vectorised ``_zone_pure``: is ``val``'s own span at least this pure?

    The envelope runs from the first to the last level labelled ``val``; the
    test is the fraction of that envelope which is actually ``val``. False
    where ``val`` never occurs, matching the original.

    ``limit`` optionally restricts the test to a sub-range of the column (used
    for the lower layer of cases D and E), as a boolean mask of the same shape.
    """
    m = phase == val
    if limit is not None:
        m = m & limit
    first, last, any_ = _first_last(m)
    idx = np.arange(phase.shape[-1])
    span = (idx >= first[..., None]) & (idx <= last[..., None])
    if limit is not None:
        span = span & limit
    span_len = np.maximum(span.sum(axis=-1), 1)
    return any_ & ((m & span).sum(axis=-1) / span_len >= min_purity)


# ----------------------------------------------------------------------------
# Phase assignment
# ----------------------------------------------------------------------------
def assign_phase(lwc_g_m3: np.ndarray, iwc_g_m3: np.ndarray,
                 threshold: float = DEFAULT_PHASE_THRESHOLD_G_M3,
                 rule: str = DEFAULT_PHASE_RULE) -> np.ndarray:
    """Discrete phase per level from liquid and ice water content, g m-3.

    Two rules, and the difference is not cosmetic.

    "absolute" -- each species tested against the threshold INDEPENDENTLY, so a
        level is MIXED whenever both clear the floor no matter how lopsided the
        ratio. This is inconsistent with the column-level purity test, which is
        fractional: a level that is 99% liquid by mass gets labelled MIXED, and
        then in the mass basis its ENTIRE condensate -- the liquid included --
        counts toward frac_mix. Measured on this archive, 15.9% of levels
        labelled MIXED are more than 90% liquid by mass.

    "fraction" -- the threshold decides only whether there is CLOUD at all
        (total condensate above it), and the phase then follows the same 90/10
        rule used at column level:

            liquid   ice mass share < 1 - PURITY_THRESHOLD complement, i.e. <10%
            ice      ice mass share > 90%
            mixed    in between

        Consistent with the column test and with the ARM purity concept, and the
        recommended setting.
    """
    if rule not in ("absolute", "fraction"):
        raise ValueError(f"rule must be 'absolute' or 'fraction', not {rule!r}")

    phase = np.full(lwc_g_m3.shape, CLEAR, dtype=np.int8)
    if rule == "absolute":
        has_l = lwc_g_m3 > threshold
        has_i = iwc_g_m3 > threshold
        phase[has_l & ~has_i] = LIQUID
        phase[has_i & ~has_l] = ICE
        phase[has_l & has_i] = MIXED
        return phase

    total = lwc_g_m3 + iwc_g_m3
    cloudy = total > threshold
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, iwc_g_m3 / np.where(total > 0, total, 1.0), 0.0)
    phase[cloudy] = MIXED
    phase[cloudy & (share < PURITY_THRESHOLD)] = LIQUID
    phase[cloudy & (share > 1.0 - PURITY_THRESHOLD)] = ICE
    return phase


# ----------------------------------------------------------------------------
# Case detection
# ----------------------------------------------------------------------------
def detect_columns(phase: np.ndarray, lwp: np.ndarray, iwp: np.ndarray,
                   basis: str = "mass") -> dict:
    """Classify every column into a case code, following the ARM logic exactly.

    ``phase`` is ``(n_col, n_lev)`` with level index increasing UPWARD, matching
    the ARM convention where index 0 is the lowest gate. ``lwp``/``iwp`` are the
    per-level mass paths, same shape, used for the mass-weighted fractions.

    ``basis`` selects how "fraction of the cloud" is measured:
      "mass"   frac_ice = sum(IWP) / sum(LWP + IWP) over the cloud
      "levels" frac_ice = n_ice_levels / n_cloud_levels, the literal ARM rule

    Returns a dict of arrays: ``case`` plus the diagnostic counts and flags the
    ARM version records.
    """
    if basis not in ("mass", "levels"):
        raise ValueError(f"basis must be 'mass' or 'levels', not {basis!r}")

    n_col, n_lev = phase.shape
    idx = np.arange(n_lev)

    is_liq, is_ice, is_mix = phase == LIQUID, phase == ICE, phase == MIXED
    is_cloud = phase != CLEAR
    liq_count = is_liq.sum(axis=-1)
    ice_count = is_ice.sum(axis=-1)
    mix_count = is_mix.sum(axis=-1)

    base, top, has_cloud = _first_last(is_cloud)
    interior = (idx >= base[:, None]) & (idx <= top[:, None]) & has_cloud[:, None]

    # Vacuous in the ARM code too: everything below the first cloudy level is
    # clear by construction. Recorded so the comparison can show it agrees.
    clear_below = np.ones(n_col, dtype=bool)

    # Phase weights, on the requested basis.
    if basis == "mass":
        w_mix = np.where(is_mix, lwp + iwp, 0.0).sum(axis=-1)
        # A mixed level's mass sits in both phases, so normalise on the total
        # condensate rather than on a sum that double-counts it.
        total = np.where(is_cloud, lwp + iwp, 0.0).sum(axis=-1)
        w_ice_only = np.where(is_ice, iwp, 0.0).sum(axis=-1)
    else:
        total = (liq_count + mix_count + ice_count).astype(float)
        w_ice_only = ice_count.astype(float)
        w_mix = mix_count.astype(float)

    safe = np.where(total > 0, total, 1.0)
    frac_ice = np.where(total > 0, w_ice_only / safe, 0.0)
    frac_mix = np.where(total > 0, w_mix / safe, 0.0)

    top_pixel = np.where(has_cloud, phase[np.arange(n_col), np.clip(top, 0, None)],
                         CLEAR)

    gap_mask = (phase == CLEAR) & interior
    n_gaps = count_runs(gap_mask)
    liqmix_regions = count_runs((is_liq | is_mix) & interior)

    pure_liq = zone_pure(phase, LIQUID, 1.0 - PURITY_THRESHOLD)
    pure_ice = zone_pure(phase, ICE, 1.0 - PURITY_THRESHOLD)
    pure_mix = zone_pure(phase, MIXED, MIXED_ZONE_MIN_PURITY)

    has_liq, has_mix, has_ice = liq_count > 0, mix_count > 0, ice_count > 0
    case = np.zeros(n_col, dtype=np.int8)

    # ---- single layer: A, then B, then C, then F -- same order as the ARM
    # code, so a column only ever matches the first case it satisfies.
    single = has_cloud & (n_gaps == 0) & (total > 0)

    a = single & has_liq & ((frac_mix + frac_ice) <= PURITY_THRESHOLD)
    case = np.where(a, 3, case)

    b_slot = single & ~a & has_liq & has_mix & (frac_ice <= PURITY_THRESHOLD)
    b = b_slot & (top_pixel == LIQUID) & pure_liq & pure_mix
    case = np.where(b, 1, case)

    c_slot = (single & ~a & ~b_slot & has_liq & has_ice
              & (frac_mix <= PURITY_THRESHOLD))
    c = c_slot & (top_pixel == LIQUID) & (liqmix_regions == 1) & pure_liq & pure_ice
    case = np.where(c, 2, case)

    f_slot = single & ~a & ~b_slot & ~c_slot & has_liq & has_mix & has_ice
    f = (f_slot & (top_pixel == LIQUID) & (liqmix_regions == 1)
         & pure_liq & pure_mix & pure_ice)
    case = np.where(f, 6, case)

    # ---- two layers separated by one clear gap: D, then E
    two = has_cloud & (n_gaps == 1)
    if two.any():
        g_first, g_last, _ = _first_last(gap_mask)
        lower = interior & (idx < g_first[:, None])
        upper = interior & (idx > g_last[:, None])

        if basis == "mass":
            up_tot = np.where(upper, lwp + iwp, 0.0).sum(axis=-1)
            up_nonice = np.where(upper & ~is_ice, lwp + iwp, 0.0).sum(axis=-1)
            lo_tot = np.where(lower, lwp + iwp, 0.0).sum(axis=-1)
            lo_nonliq = np.where(lower & ~is_liq, lwp + iwp, 0.0).sum(axis=-1)
            lo_ice = np.where(lower & is_ice, iwp, 0.0).sum(axis=-1)
        else:
            up_tot = upper.sum(axis=-1).astype(float)
            up_nonice = (upper & ~is_ice).sum(axis=-1).astype(float)
            lo_tot = lower.sum(axis=-1).astype(float)
            lo_nonliq = (lower & ~is_liq).sum(axis=-1).astype(float)
            lo_ice = (lower & is_ice).sum(axis=-1).astype(float)

        up_ok = (up_tot > 0) & (upper & is_ice).any(axis=-1) & (
            up_nonice / np.where(up_tot > 0, up_tot, 1.0) <= PURITY_THRESHOLD)
        lo_has_liq = (lower & is_liq).any(axis=-1)
        lo_ok = (lo_tot > 0) & lo_has_liq

        d = two & up_ok & lo_ok & (
            lo_nonliq / np.where(lo_tot > 0, lo_tot, 1.0) <= PURITY_THRESHOLD)
        case = np.where(d, 4, case)

        # E: same two-layer setup, lower layer liquid-topped with mixed present.
        lo_top = np.where(lo_has_liq, np.argmax(np.where(lower, idx, -1), axis=-1), 0)
        lo_top_is_liq = phase[np.arange(n_col), np.clip(lo_top, 0, None)] == LIQUID
        e = (two & up_ok & lo_ok & ~d & (lower & is_mix).any(axis=-1)
             & (lo_ice / np.where(lo_tot > 0, lo_tot, 1.0) <= PURITY_THRESHOLD)
             & lo_top_is_liq
             & zone_pure(phase, LIQUID, 1.0 - PURITY_THRESHOLD, limit=lower)
             & zone_pure(phase, MIXED, MIXED_ZONE_MIN_PURITY, limit=lower))
        case = np.where(e, 5, case)

    return {
        "case": case,
        "liq_count": liq_count, "mix_count": mix_count, "ice_count": ice_count,
        "n_gaps": n_gaps, "frac_ice": frac_ice, "frac_mix": frac_mix,
        "has_cloud": has_cloud, "clear_below": clear_below,
        "cloud_base": base, "cloud_top": top,
    }


# ----------------------------------------------------------------------------
# From an ERA5 pressure-level dataset to phase and mass paths
# ----------------------------------------------------------------------------
def profile_fields(ds, sp_pa: np.ndarray, threshold_g_m3: float,
                   phase_rule: str = DEFAULT_PHASE_RULE):
    """Phase, LWP and IWP per level from one pressure-level dataset.

    Returns ``(phase, lwp_g_m2, iwp_g_m2, extras)`` with the level axis LAST and
    increasing UPWARD, so the ARM convention holds. ``extras`` carries the
    per-column snow/rain flags and the column liquid path the filters need.
    """
    level_name = "pressure_level"
    lv = ds[level_name].values.astype(float)
    order = np.argsort(lv)                       # ascending pressure = downward
    p_pa = lv[order] * 100.0

    dims = [d for d in ds["t"].dims if d != level_name]

    def arr(name):
        return ds[name].transpose(*dims, level_name).values.astype(float)[..., order]

    t, q = arr("t"), arr("q")
    clwc, ciwc = arr("clwc"), arr("ciwc")
    crwc = arr("crwc") if "crwc" in ds else np.zeros_like(clwc)
    cswc = arr("cswc") if "cswc" in ds else np.zeros_like(clwc)

    q_cond = clwc + ciwc + crwc + cswc
    rho = moist_density(p_pa, t, q, q_cond)                     # kg m-3
    dp = layer_thickness_pa(p_pa, sp_pa)                        # Pa

    # Density in g m-3 for the phase test; mass path in g m-2 for the fractions.
    lwc_g_m3 = clwc * rho * 1000.0
    iwc_g_m3 = ciwc * rho * 1000.0
    lwp = clwc * dp / G_M_S2 * 1000.0
    iwp = ciwc * dp / G_M_S2 * 1000.0
    rwp = crwc * dp / G_M_S2 * 1000.0
    swp = cswc * dp / G_M_S2 * 1000.0

    phase = assign_phase(lwc_g_m3, iwc_g_m3, threshold_g_m3, phase_rule)
    # A level below ground has zero thickness and must not be called cloudy.
    phase[dp <= 0.0] = CLEAR

    extras = {
        "no_snow": ~((cswc * rho * 1000.0 > threshold_g_m3) & (dp > 0)).any(axis=-1),
        "no_drizzle": ~((crwc * rho * 1000.0 > threshold_g_m3) & (dp > 0)).any(axis=-1),
        "lwp_column": lwp.sum(axis=-1),
        "iwp_column": iwp.sum(axis=-1),
        "rwp_column": rwp.sum(axis=-1),
        "swp_column": swp.sum(axis=-1),
        "dims": dims,
    }
    return phase, lwp, iwp, extras
