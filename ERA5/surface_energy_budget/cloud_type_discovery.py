#!/usr/bin/env python3
"""Discover the common cloud types in the ERA5 pressure-level archive.

Where ``cloud_vertical_cases.py`` ASKS whether a profile matches one of six
cloud structures defined in advance, this module asks what structures are
actually there and how often. Nothing about the taxonomy is decided before the
data is read: every column is reduced to a short symbolic signature, and the
"types" are whichever signatures turn out to be common.

WHY NOT JUST ENUMERATE THE CASES BY HAND
========================================
The obvious approach -- step through every hour, decide phase / arrangement /
layer count, and look for patterns afterwards -- is right in spirit and wrong
in execution, for two reasons.

First, writing the decision tree by hand pre-decides the answer. You find the
structures you thought to write down and stay blind to the ones you did not.
The frequencies you get back are then a property of your imagination as much as
of the atmosphere.

Second, done naively it is slow. 63 files x 48 hours x 41 x 61 cells is 7.6
million columns; a Python loop that builds a tuple per column takes tens of
minutes and has to be re-run for every parameter you want to change.

Both problems have the same fix. Reduce each column to a small integer -- a
SIGNATURE that packs the per-layer attributes into base-64 digits -- and the
taxonomy becomes ``np.unique(codes, return_counts=True)``. The alphabet is
fixed and known (phase x altitude regime x internal arrangement, per layer),
but which WORDS in that alphabet occur, and how often, is measured. The whole
archive is one vectorised streaming pass, and only a counter survives it, so
memory is set by the number of distinct signatures rather than the number of
columns.

THE THREE ATTRIBUTES, AND THE ALPHABET THEY SPAN
================================================
A column becomes a list of cloud LAYERS, bottom to top. Each layer carries:

  PHASE          from the layer's ice mass fraction f_ice = IWP / (IWP + LWP)
                 L  f_ice <= 0.1     liquid
                 M  0.1 < f_ice < 0.9  mixed
                 I  f_ice >= 0.9     ice
                 Mass, not level counts. A level count of "10% ice" is
                 meaningless when a cloud occupies 3 levels -- see
                 ``cloud_vertical_cases.py`` for what that degeneracy does.

  REGIME         where the layer BASE sits, in height above ground
                 BL   < 1 km      LOW  1-3 km    MID  3-6 km
                 HIGH 6 km - tropopause          TP+  above the tropopause
                 "BL" is a height band, not a diagnosed boundary layer; see
                 CAVEATS.

  ARRANGEMENT    the phase of the layer's top level against its bottom level
                 -    same, or the layer is one level thick
                 ^L   more liquid at the top than the bottom (liquid-topped)
                 ^I   more ice at the top

A signature is those layers concatenated bottom to top, e.g.

    BL:L^L | MID:I        liquid-topped boundary-layer cloud with cirrus above
    BL:M                  a single mixed-phase layer near the surface
    LOW:I | HIGH:I        two ice layers

FOUR GRANULARITIES, AND A COVERAGE CURVE
========================================
The full signature has a long tail, so it is built at four nested
granularities and every one of them is counted in the same pass:

    nlayers        1, 2, 3, ...
    phase          L | I
    phase+regime   BL:L | MID:I
    full           BL:L^L | MID:I

``coverage_curve`` then reports how many distinct signatures are needed to
account for 50 / 80 / 90 / 95% of cloudy hours at each granularity. That is the
quantitative form of "what are the common types": a taxonomy that needs 200
entries to cover 80% of the sky is not a taxonomy.

THE UNSUPERVISED CROSS-CHECK
============================
A symbolic scheme can only find structure its alphabet can express. So the pass
also keeps a stratified sample of raw LWC/IWC profiles binned onto a common
height grid, which ``cluster_profiles`` runs k-means over. Cross-tabulating the
clusters against the signatures says whether the symbolic types are real modes
of the data or artefacts of the thresholds. Where they agree, the taxonomy is
doing its job; where a cluster splits across several signatures, the alphabet is
throwing away something the data cares about.

WHAT COMES OUT
==============
``prepare()`` returns an ``Analysis`` holding the weighted signature counters,
the marginal tallies that answer the layer-count / boundary-layer /
stratosphere questions directly, per-signature composite profiles, the
diagnostic histograms, and the profile sample. ``save_analysis`` /
``load_analysis`` round-trip it to an .npz so a notebook pays for the pass once.

CAVEATS -- READ THESE BEFORE QUOTING A NUMBER
=============================================
1. VERTICAL RESOLUTION SETS THE LAYER COUNT. The 23 pressure levels are ~200 m
   apart below 700 hPa and ~700-1000 m apart above 500 hPa. Arctic
   boundary-layer stratus is often 200-400 m thick, so a real cloud can occupy
   ONE level and two stacked decks can merge into one. Layer counts here are a
   lower bound, and the bound is tighter aloft than near the ground.

2. "BL" IS A HEIGHT BAND. The single-level archive has no ``blh``, and a
   diagnosed boundary-layer top from a profile sampled every ~200 m is not
   meaningful in an Arctic winter inversion. ``BL`` means "base below 1 km above
   ground", nothing more. ``--regime-edges`` changes it.

3. "STRATOSPHERE" IS TRUNCATED AT 200 hPa. The archive stops there (~11.8 km).
   The Arctic tropopause sits near 250-300 hPa in winter, so some
   above-tropopause layers are visible, but anything higher is invisible and the
   TP+ frequency is a LOWER BOUND. ERA5 does not represent polar stratospheric
   clouds in any case.

4. CONTENTS ARE GRID-BOX MEANS. ERA5's clwc/ciwc are already diluted by cloud
   fraction, so a threshold on them conflates a thin cloud with a small one.
   ``--content-basis incloud`` divides by ``cc`` first, which is what a ground
   instrument under a cloud sees. The default is ``gridmean``; the difference is
   large and ``sensitivity`` reports it.

5. NEIGHBOURING CELLS ARE NOT INDEPENDENT. The frequencies are area-weighted
   climatologies over a 41 x 61 grid at 0.25 deg, and adjacent cells are highly
   correlated, so the effective sample size is far below the cell-hour count.
   Use ``--stride`` to thin the grid and check that the ranking does not move.

6. LEVEL AXIS. Everything in this module is BOTTOM-UP: index 0 is the level
   nearest the ground. This is asserted at run time (``_check_bottom_up``)
   rather than assumed, because the same convention is documented but NOT held
   in ``cloud_vertical_cases.profile_fields``, where the axis runs top-down and
   silently inverts every "liquid-topped" test.

USAGE
=====
    python cloud_type_discovery.py --region barrow --storage local
    python cloud_type_discovery.py --region barrow --max-files 4 --stride 4
    python cloud_type_discovery.py --region barrow --content-basis incloud
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from convert_specific_to_absolute import (
    G_M_S2,
    R_DRY,
    VIRTUAL_COEF,
    layer_thickness_pa,
)
from download_era5_seb import STORAGE_ROOTS, days_covered_by_file

# ----------------------------------------------------------------------------
# The alphabet
# ----------------------------------------------------------------------------
PHASE_LIQ, PHASE_MIX, PHASE_ICE = 0, 1, 2
PHASE_SYM = ("L", "M", "I")
PHASE_NAME = ("liquid", "mixed", "ice")

REGIME_BL, REGIME_LOW, REGIME_MID, REGIME_HIGH, REGIME_TP = 0, 1, 2, 3, 4
REGIME_SYM = ("BL", "LOW", "MID", "HIGH", "TP+")
REGIME_NAME = ("boundary layer (<1 km)", "low (1-3 km)", "mid (3-6 km)",
               "high (6 km-tropopause)", "above tropopause")

ARR_UNIFORM, ARR_LIQ_TOP, ARR_ICE_TOP = 0, 1, 2
ARR_SYM = ("", "^L", "^I")
ARR_NAME = ("uniform", "liquid-topped", "ice-topped")

N_PHASE, N_REGIME, N_ARR = 3, 5, 3

# Granularity -> which layer attributes enter the signature.
GRANULARITIES = ("nlayers", "phase", "phase+regime", "full")
GRAN_FIELDS = {
    "nlayers":      (False, False, False),
    "phase":        (True, False, False),
    "phase+regime": (True, True, False),
    "full":         (True, True, True),
}

# One layer slot packs into a base-64 digit: 1 + phase*1 + regime*3 + arr*15,
# max 1 + 2 + 12 + 30 = 45 < 64. 0 means "no layer in this slot".
SLOT_BASE = 64

# ----------------------------------------------------------------------------
# Defaults. Every one of these is a free parameter; ``sensitivity`` sweeps them.
# ----------------------------------------------------------------------------
DEFAULT_CONTENT_THRESHOLD_G_M3 = 1e-3   # a cloud radar's detection floor
DEFAULT_MIN_GAP_M = 500.0               # thinner clear slabs do not split a layer
DEFAULT_MIN_LAYER_PATH_G_M2 = 1.0       # layers below this are dropped;
#                                         the 5th percentile of layer path
#                                         is 0.5-1.3 g m-2, so this trims
#                                         numerical traces, not clouds
DEFAULT_LIQ_MAX_FICE = 0.1              # f_ice <= this  -> liquid
DEFAULT_ICE_MIN_FICE = 0.9              # f_ice >= this  -> ice
DEFAULT_REGIME_EDGES_M = (1000.0, 3000.0, 6000.0)
DEFAULT_MAX_LAYERS = 5
DEFAULT_SAMPLE_PER_FILE = 1200
DEFAULT_SEED = 20260826

# Height grid the sampled profiles are binned onto, for clustering.
PROFILE_BIN_EDGES_M = np.arange(0.0, 12001.0, 500.0)

SINGLE_VARS = ("sp", "tcc", "lcc", "mcc", "hcc", "siconc")

# Diagnostic histogram bins.
HEIGHT_BINS_M = np.arange(0.0, 12001.0, 250.0)
THICK_BINS_M = np.arange(0.0, 6001.0, 100.0)
FICE_BINS = np.linspace(0.0, 1.0, 51)
TEMP_BINS_K = np.arange(200.0, 293.1, 2.0)
LOGPATH_BINS = np.linspace(-3.0, 3.5, 66)     # log10 g m-2

# ERA5's own low/medium/high cloud boundaries, for the lcc/mcc/hcc cross-check.
ERA5_LOW_MID_HPA = 800.0
ERA5_MID_HIGH_HPA = 450.0


class Analysis(SimpleNamespace):
    """Everything one streaming pass produces. See ``prepare``."""


# ----------------------------------------------------------------------------
# Vertical geometry
# ----------------------------------------------------------------------------
def _check_bottom_up(p_pa: np.ndarray) -> None:
    """Assert the level axis runs bottom-up (pressure strictly decreasing).

    Checked rather than assumed: the sibling module documents this convention
    and does not hold it, which inverts every top-vs-base test downstream.
    """
    if p_pa.ndim != 1 or p_pa.size < 2:
        raise ValueError("pressure axis must be 1-D with at least two levels")
    if not np.all(np.diff(p_pa) < 0):
        raise ValueError(
            "level axis is not bottom-up: pressure must decrease with index, "
            f"got {p_pa[:3]} ... {p_pa[-3:]} Pa"
        )


def virtual_temperature(t_k, q_v, q_cond):
    """T_v = T (1 + 0.608 q_v - q_c), K. Condensate is subtracted: suspended
    water carries mass but no partial pressure, so it makes the parcel denser.
    Same form as ``convert_specific_to_absolute.moist_density``."""
    return t_k * (1.0 + VIRTUAL_COEF * q_v - q_cond)


def layer_geometry(p_pa, t_k, q_v, q_cond, sp_pa):
    """Pressure thickness, geometric thickness, and edge heights per level.

    ``p_pa`` is 1-D bottom-up (decreasing). The rest are ``(n_col, n_lev)``.
    Returns ``(dp_pa, dz_m, z_bot_m, z_top_m, rho)`` with the level axis last
    and heights measured ABOVE GROUND, from the hydrostatic thickness of each
    layer rather than from ``z`` -- 50 of the 63 archive files have no ``z``.

    dz = dp / (rho g) is the hypsometric thickness of the slab the level owns,
    so the edge heights are the running sum of dz from the surface upward and
    are consistent by construction with the mass paths computed from the same
    dp. A level entirely below ground gets dp = dz = 0 and contributes nothing.
    """
    _check_bottom_up(p_pa)
    # layer_thickness_pa wants ascending pressure; flip in and back out.
    dp_pa = layer_thickness_pa(p_pa[::-1], np.asarray(sp_pa, dtype=float))[..., ::-1]
    t_v = virtual_temperature(t_k, q_v, q_cond)
    rho = p_pa / (R_DRY * t_v)                                  # kg m-3
    dz_m = np.where(dp_pa > 0.0, dp_pa / (rho * G_M_S2), 0.0)   # m
    z_top_m = np.cumsum(dz_m, axis=-1)
    z_bot_m = z_top_m - dz_m
    return dp_pa, dz_m, z_bot_m, z_top_m, rho


def tropopause_height_m(t_k, z_mid_m, p_pa, min_p_hpa=500.0, span_m=2000.0):
    """WMO lapse-rate tropopause, as far as a 23-level profile can resolve it.

    The lowest level above ``min_p_hpa`` where the lapse rate falls to 2 K/km or
    less AND the mean lapse rate from there to 2 km higher is also 2 K/km or
    less. Returns ``(z_trop_m, found)``; where no such level exists below the
    top of the archive, ``z_trop_m`` is the top of the profile and ``found`` is
    False. Those columns cannot distinguish HIGH from TP+, which is why the
    fraction of them is reported.
    """
    _check_bottom_up(p_pa)
    n_lev = p_pa.size
    dz = np.diff(z_mid_m, axis=-1)
    lapse = -np.diff(t_k, axis=-1) / np.maximum(dz, 1.0) * 1000.0   # K km-1

    # Mean lapse rate from each level to ``span_m`` above it, by linear
    # interpolation of T onto that height. Vectorised over columns with
    # searchsorted-free arithmetic: for each level k, find the bracketing pair.
    z_target = z_mid_m + span_m
    t_at = np.empty_like(t_k)
    for k in range(n_lev):
        zt = z_target[:, k]
        above = z_mid_m >= zt[:, None]
        hi = np.where(above.any(axis=-1), np.argmax(above, axis=-1), n_lev - 1)
        lo = np.maximum(hi - 1, 0)
        rows = np.arange(t_k.shape[0])
        z0, z1 = z_mid_m[rows, lo], z_mid_m[rows, hi]
        t0, t1 = t_k[rows, lo], t_k[rows, hi]
        w = np.where(z1 > z0, (zt - z0) / np.maximum(z1 - z0, 1.0), 0.0)
        t_at[:, k] = t0 + np.clip(w, 0.0, 1.0) * (t1 - t0)
    mean_lapse = -(t_at - t_k) / span_m * 1000.0

    ok = np.zeros_like(t_k, dtype=bool)
    ok[:, :-1] = lapse <= 2.0
    ok &= mean_lapse <= 2.0
    ok &= (p_pa / 100.0 <= min_p_hpa)[None, :]
    # The 2-km test needs 2 km of profile above the level to mean anything.
    ok &= (z_mid_m + span_m) <= z_mid_m[:, -1][:, None]

    found = ok.any(axis=-1)
    idx = np.where(found, np.argmax(ok, axis=-1), t_k.shape[-1] - 1)
    z_trop = z_mid_m[np.arange(t_k.shape[0]), idx]
    return np.where(found, z_trop, z_mid_m[:, -1]), found


# ----------------------------------------------------------------------------
# Runs, layers, and their attributes
# ----------------------------------------------------------------------------
def _first_last(mask):
    """First and last True index along the last axis; -1 where all False."""
    n = mask.shape[-1]
    any_ = mask.any(axis=-1)
    first = np.where(any_, np.argmax(mask, axis=-1), -1)
    last = np.where(any_, n - 1 - np.argmax(mask[..., ::-1], axis=-1), -1)
    return first, last, any_


def run_labels(mask):
    """1-based contiguous-run id per True element along the last axis, 0 where
    False. Runs are numbered bottom-up, so run 1 is the lowest."""
    m = mask.astype(np.int8)
    starts = np.zeros_like(m)
    starts[..., 0] = m[..., 0]
    starts[..., 1:] = ((m[..., 1:] == 1) & (m[..., :-1] == 0)).astype(np.int8)
    ids = np.cumsum(starts, axis=-1, dtype=np.int16)
    return np.where(mask, ids, 0).astype(np.int16)


def _sum_by_run(values, ids, n_run):
    """Sum ``values`` within each run. ``ids`` is 0..n_run; slot 0 is the
    not-in-a-run bucket and is returned but never meaningful."""
    n_col, _ = ids.shape
    lin = (np.arange(n_col)[:, None] * (n_run + 1) + ids).ravel()
    out = np.bincount(lin, weights=values.ravel(),
                      minlength=n_col * (n_run + 1))
    return out.reshape(n_col, n_run + 1)


def merge_thin_gaps(is_cloud, dz_m, min_gap_m):
    """Fill clear slabs INSIDE the cloud that are thinner than ``min_gap_m``.

    At 25 hPa spacing a single sub-threshold level is ~200 m near the ground.
    Calling that a cloud-free gap turns one deck into two and makes the layer
    count an artefact of the level spacing rather than a property of the sky.
    Only interior gaps are considered; clear air below the lowest cloud and
    above the highest is untouched.
    """
    if min_gap_m <= 0:
        return is_cloud
    first, last, any_ = _first_last(is_cloud)
    idx = np.arange(is_cloud.shape[-1])
    interior = ((idx >= first[:, None]) & (idx <= last[:, None])
                & any_[:, None])
    gap = interior & ~is_cloud
    if not gap.any():
        return is_cloud
    gid = run_labels(gap)
    n_run = int(gid.max())
    thick = _sum_by_run(np.where(gap, dz_m, 0.0), gid, n_run)
    gap_thick = np.take_along_axis(thick, gid.astype(np.intp), axis=-1)
    return is_cloud | (gap & (gap_thick < min_gap_m))


def layer_attributes(is_cloud, lwp, iwp, dz_m, z_bot_m, z_top_m, t_k,
                     max_layers, min_layer_path_g_m2):
    """Per-layer aggregates, bottom-up, as ``(n_col, max_layers)`` arrays.

    Layers whose total condensate path is below ``min_layer_path_g_m2`` are
    dropped and the remaining layers are RE-PACKED so slot 0 is always the
    lowest surviving layer -- otherwise a discarded layer would leave a hole and
    the signature would encode the hole.

    ``n_layers`` counts surviving layers and saturates at ``max_layers``;
    ``n_truncated`` counts columns that had more.
    """
    lid = run_labels(is_cloud)
    n_run = int(lid.max())
    n_col, n_lev = is_cloud.shape
    if n_run == 0:
        z = np.zeros((n_col, max_layers))
        return dict(
            present=np.zeros((n_col, max_layers), bool),
            lwp=z.copy(), iwp=z.copy(), f_ice=z.copy(),
            base_m=z.copy(), top_m=z.copy(), thick_m=z.copy(),
            top_t_k=np.full((n_col, max_layers), np.nan),
            base_idx=np.zeros((n_col, max_layers), np.int16),
            top_idx=np.zeros((n_col, max_layers), np.int16),
            phase_top=np.zeros((n_col, max_layers), np.int8),
            phase_base=np.zeros((n_col, max_layers), np.int8),
            n_layers=np.zeros(n_col, np.int8),
            n_truncated=0,
        )

    lwp_r = _sum_by_run(np.where(is_cloud, lwp, 0.0), lid, n_run)[:, 1:]
    iwp_r = _sum_by_run(np.where(is_cloud, iwp, 0.0), lid, n_run)[:, 1:]
    keep = (lwp_r + iwp_r) >= min_layer_path_g_m2                # (n_col, n_run)

    # Re-pack: the surviving runs, in order, into slots 0..max_layers-1.
    order = np.argsort(~keep, axis=-1, kind="stable")            # kept first
    n_keep = keep.sum(axis=-1)
    n_truncated = int((n_keep > max_layers).sum())
    # A file may hold fewer runs than max_layers; pad so every slot exists.
    # The padding points at run 1 and is masked out by ``present``.
    if order.shape[1] < max_layers:
        order = np.concatenate(
            [order, np.zeros((n_col, max_layers - order.shape[1]), order.dtype)],
            axis=1)
    slot_run = order[:, :max_layers] + 1                         # 1-based run id
    present = (np.arange(max_layers)[None, :] < n_keep[:, None])

    rows = np.arange(n_col)[:, None]
    out = dict(
        present=present,
        lwp=np.where(present, lwp_r[rows, slot_run - 1], 0.0),
        iwp=np.where(present, iwp_r[rows, slot_run - 1], 0.0),
    )
    tot = out["lwp"] + out["iwp"]
    out["f_ice"] = np.where(tot > 0.0, out["iwp"] / np.where(tot > 0, tot, 1.0), 0.0)

    base_idx = np.zeros((n_col, max_layers), np.int16)
    top_idx = np.zeros((n_col, max_layers), np.int16)
    for s in range(max_layers):
        sel = (lid == slot_run[:, s][:, None]) & present[:, s][:, None]
        has = sel.any(axis=-1)
        base_idx[:, s] = np.where(has, np.argmax(sel, axis=-1), 0)
        top_idx[:, s] = np.where(
            has, n_lev - 1 - np.argmax(sel[:, ::-1], axis=-1), 0)
    out["base_idx"], out["top_idx"] = base_idx, top_idx
    out["base_m"] = np.where(present, z_bot_m[rows, base_idx], 0.0)
    out["top_m"] = np.where(present, z_top_m[rows, top_idx], 0.0)
    out["thick_m"] = out["top_m"] - out["base_m"]
    out["top_t_k"] = np.where(present, t_k[rows, top_idx], np.nan)
    out["n_layers"] = np.minimum(n_keep, max_layers).astype(np.int8)
    out["n_truncated"] = n_truncated
    out["_slot_run"] = slot_run
    out["_lid"] = lid
    return out


# ----------------------------------------------------------------------------
# Attribute -> code
# ----------------------------------------------------------------------------
def phase_code(f_ice, liq_max=DEFAULT_LIQ_MAX_FICE, ice_min=DEFAULT_ICE_MIN_FICE):
    """L / M / I from the ice mass fraction."""
    return np.where(f_ice <= liq_max, PHASE_LIQ,
                    np.where(f_ice >= ice_min, PHASE_ICE, PHASE_MIX)).astype(np.int8)


def regime_code(base_m, z_trop_m, edges=DEFAULT_REGIME_EDGES_M):
    """BL / LOW / MID / HIGH / TP+ from the layer base height above ground."""
    code = np.digitize(base_m, np.asarray(edges, dtype=float)).astype(np.int8)
    return np.where(base_m >= z_trop_m[:, None], REGIME_TP, code).astype(np.int8)


def arrangement_code(phase_top, phase_base, present, single_level):
    """Uniform / liquid-topped / ice-topped, from the top and base LEVELS.

    A one-level layer has no internal arrangement and is always uniform. Note
    the sign: PHASE_LIQ < PHASE_MIX < PHASE_ICE, so a top code SMALLER than the
    base code means more liquid at the top.
    """
    arr = np.where(phase_top < phase_base, ARR_LIQ_TOP,
                   np.where(phase_top > phase_base, ARR_ICE_TOP, ARR_UNIFORM))
    return np.where(present & ~single_level, arr, ARR_UNIFORM).astype(np.int8)


def pack_signature(present, phase, regime, arr, granularity):
    """Pack the per-layer codes into one int64 per column.

    Slot s occupies base-64 digit s, bottom layer in the least significant
    digit. Attributes outside the granularity are zeroed BEFORE packing, so a
    coarser signature is a genuine coarsening of the same code space and
    ``decode_signature`` needs only the granularity to read it back.
    """
    use_p, use_r, use_a = GRAN_FIELDS[granularity]
    p = phase if use_p else np.zeros_like(phase)
    r = regime if use_r else np.zeros_like(regime)
    a = arr if use_a else np.zeros_like(arr)
    slot = np.where(present, 1 + p + 3 * r + 15 * a, 0).astype(np.int64)
    powers = SLOT_BASE ** np.arange(present.shape[1], dtype=np.int64)
    return (slot * powers[None, :]).sum(axis=1)


def decode_signature(code, granularity, sep=" | "):
    """Human-readable signature string, bottom layer first."""
    use_p, use_r, use_a = GRAN_FIELDS[granularity]
    code = int(code)
    if code == 0:
        return "(clear)"
    parts = []
    while code > 0:
        slot = code % SLOT_BASE
        code //= SLOT_BASE
        if slot == 0:
            parts.append("?")
            continue
        s = slot - 1
        p, r, a = s % 3, (s // 3) % 5, s // 15
        if granularity == "nlayers":
            parts.append("cloud")
        else:
            txt = PHASE_SYM[p] if use_p else "*"
            if use_r:
                txt = f"{REGIME_SYM[r]}:{txt}"
            if use_a:
                txt += ARR_SYM[a]
            parts.append(txt)
    if granularity == "nlayers":
        n = len(parts)
        return f"{n} layer" + ("s" if n != 1 else "")
    return sep.join(parts)


# ----------------------------------------------------------------------------
# Profile extraction, bottom-up
# ----------------------------------------------------------------------------
def profile_fields(ds, sp_pa, content_threshold_g_m3, content_basis="gridmean",
                   min_cc=0.01):
    """Bottom-up phase-relevant fields from one pressure-level dataset.

    Returns a dict with the level axis LAST and index 0 nearest the ground.
    Every array is ``(n_col, n_lev)`` with the non-level dims flattened;
    ``dims`` and ``shape`` record how to fold them back.

    ``content_basis``:
      "gridmean"  threshold ERA5's grid-box mean content, as stored
      "incloud"   divide by the cloud fraction first, which is what an
                  instrument inside the cloud measures. Grid boxes with
                  ``cc < min_cc`` are treated as clear rather than divided by a
                  near-zero number.

    Mass paths are always the grid-box means, because they are what integrates
    to the column water path -- only the THRESHOLD moves with the basis.
    """
    level_name = "pressure_level"
    lv = ds[level_name].values.astype(float)
    order = np.argsort(-lv)                       # descending pressure: bottom-up
    p_pa = lv[order] * 100.0
    _check_bottom_up(p_pa)

    dims = [d for d in ds["t"].dims if d != level_name]
    shape = tuple(ds.sizes[d] for d in dims)
    n_col = int(np.prod(shape))

    def arr(name):
        if name not in ds:
            return np.zeros((n_col, p_pa.size))
        v = ds[name].transpose(*dims, level_name).values.astype(np.float64)
        return v[..., order].reshape(n_col, p_pa.size)

    t_k, q_v = arr("t"), arr("q")
    clwc, ciwc = arr("clwc"), arr("ciwc")
    crwc, cswc = arr("crwc"), arr("cswc")
    cc = arr("cc") if "cc" in ds else np.ones((n_col, p_pa.size))

    q_cond = clwc + ciwc + crwc + cswc
    dp_pa, dz_m, z_bot_m, z_top_m, rho = layer_geometry(
        p_pa, t_k, q_v, q_cond, np.asarray(sp_pa, dtype=float).reshape(n_col))
    valid = dp_pa > 0.0

    # Mass path per level, g m-2. Grid-box mean, always.
    to_path = dp_pa / G_M_S2 * 1000.0
    lwp = clwc * to_path
    iwp = ciwc * to_path
    rwp = crwc * to_path
    swp = cswc * to_path

    # Density for the threshold test, g m-3.
    lwc = clwc * rho * 1000.0
    iwc = ciwc * rho * 1000.0
    if content_basis == "incloud":
        denom = np.where(cc >= min_cc, cc, np.nan)
        with np.errstate(invalid="ignore"):
            lwc, iwc = lwc / denom, iwc / denom
        lwc = np.nan_to_num(lwc)
        iwc = np.nan_to_num(iwc)
    elif content_basis != "gridmean":
        raise ValueError(f"content_basis must be gridmean or incloud, "
                         f"not {content_basis!r}")

    has_l = (lwc > content_threshold_g_m3) & valid
    has_i = (iwc > content_threshold_g_m3) & valid
    is_cloud = has_l | has_i

    z_mid_m = 0.5 * (z_bot_m + z_top_m)
    z_trop_m, trop_found = tropopause_height_m(t_k, z_mid_m, p_pa)

    return dict(
        p_pa=p_pa, dims=dims, shape=shape, n_col=n_col,
        t_k=t_k, cc=cc, valid=valid,
        dp_pa=dp_pa, dz_m=dz_m, z_bot_m=z_bot_m, z_top_m=z_top_m, z_mid_m=z_mid_m,
        lwc_g_m3=lwc, iwc_g_m3=iwc,
        lwp=lwp, iwp=iwp, rwp=rwp, swp=swp,
        is_cloud=is_cloud, has_l=has_l, has_i=has_i,
        z_trop_m=z_trop_m, trop_found=trop_found,
    )


def classify_columns(fields, *, min_gap_m=DEFAULT_MIN_GAP_M,
                     min_layer_path_g_m2=DEFAULT_MIN_LAYER_PATH_G_M2,
                     liq_max=DEFAULT_LIQ_MAX_FICE, ice_min=DEFAULT_ICE_MIN_FICE,
                     regime_edges=DEFAULT_REGIME_EDGES_M,
                     max_layers=DEFAULT_MAX_LAYERS):
    """Layer decomposition plus the coded attributes, for one file's columns."""
    is_cloud = merge_thin_gaps(fields["is_cloud"], fields["dz_m"], min_gap_m)
    lay = layer_attributes(is_cloud, fields["lwp"], fields["iwp"],
                           fields["dz_m"], fields["z_bot_m"], fields["z_top_m"],
                           fields["t_k"], max_layers, min_layer_path_g_m2)

    rows = np.arange(fields["n_col"])[:, None]
    lwp_lev, iwp_lev = fields["lwp"], fields["iwp"]

    def level_f_ice(lev_idx):
        l = lwp_lev[rows, lev_idx]
        i = iwp_lev[rows, lev_idx]
        tot = l + i
        return np.where(tot > 0.0, i / np.where(tot > 0, tot, 1.0), 0.0)

    phase = phase_code(lay["f_ice"], liq_max, ice_min)
    p_top = phase_code(level_f_ice(lay["top_idx"]), liq_max, ice_min)
    p_base = phase_code(level_f_ice(lay["base_idx"]), liq_max, ice_min)
    single = lay["top_idx"] == lay["base_idx"]
    arr = arrangement_code(p_top, p_base, lay["present"], single)
    regime = regime_code(lay["base_m"], fields["z_trop_m"], regime_edges)

    phase = np.where(lay["present"], phase, 0).astype(np.int8)
    regime = np.where(lay["present"], regime, 0).astype(np.int8)
    lay.update(phase=phase, regime=regime, arrangement=arr,
               phase_top=p_top, phase_base=p_base, is_cloud=is_cloud)
    return lay


# ----------------------------------------------------------------------------
# Weighted counters
# ----------------------------------------------------------------------------
def _accumulate(counter, codes, weights):
    """Add weighted code counts into a ``{code: weight}`` dict."""
    uniq, inv = np.unique(codes, return_inverse=True)
    tot = np.bincount(inv, weights=weights, minlength=uniq.size)
    for c, w in zip(uniq.tolist(), tot.tolist()):
        counter[c] = counter.get(c, 0.0) + w


def counter_table(counter, granularity, total=None, top=None):
    """A counter as a DataFrame sorted by weight, with percentages."""
    import pandas as pd
    if not counter:
        return pd.DataFrame(columns=["signature", "weight", "percent",
                                     "cum_percent", "code"])
    codes = np.array(sorted(counter, key=lambda c: -counter[c]), dtype=np.int64)
    w = np.array([counter[int(c)] for c in codes], dtype=float)
    denom = float(w.sum()) if total is None else float(total)
    pct = 100.0 * w / max(denom, 1e-30)
    df = pd.DataFrame({
        "signature": [decode_signature(c, granularity) for c in codes],
        "weight": w, "percent": pct, "cum_percent": np.cumsum(pct),
        "code": codes,
    })
    return df.head(top) if top else df


def coverage_curve(counter, levels=(50.0, 80.0, 90.0, 95.0, 99.0)):
    """How many distinct signatures cover each percentage of the weight.

    The number that says whether a taxonomy is useful. Returns
    ``{level: n_signatures}`` plus the total number of distinct signatures under
    the key ``"distinct"``.
    """
    if not counter:
        return {"distinct": 0, **{lv: 0 for lv in levels}}
    w = np.sort(np.fromiter(counter.values(), dtype=float))[::-1]
    cum = 100.0 * np.cumsum(w) / w.sum()
    out = {"distinct": int(w.size)}
    for lv in levels:
        out[lv] = int(np.searchsorted(cum, lv) + 1)
    return out


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def prepare(argv=None, args=None, quiet=False, **overrides):
    """Read the pressure-level archive once and build every counter.

    Slow -- it opens every file. A notebook should call this once, keep the
    result, and ``save_analysis`` it.
    """
    import pandas as pd
    import xarray as xr

    warnings.filterwarnings("ignore", category=FutureWarning)
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    root = Path(args.data_root or STORAGE_ROOTS[args.storage])
    pl_dir = root / f"{args.region}_pressure"
    single_dir = root / args.region
    pl_files = sorted(glob.glob(str(pl_dir / "*.nc")))
    if not pl_files:
        raise FileNotFoundError(f"no pressure-level files in {pl_dir}")
    if args.max_files:
        pl_files = pl_files[:args.max_files]
    single_files = sorted(glob.glob(str(single_dir / "*.nc")))
    if not single_files:
        raise FileNotFoundError(f"no single-level files in {single_dir}")
    n_all_single = len(single_files)
    single_files = matching_single_files(pl_files, single_files)

    def say(*a, **kw):
        if not quiet:
            print(*a, **kw)

    say("=" * 78)
    say("ERA5 cloud-type discovery")
    say("=" * 78)
    say(f"  Pressure   : {len(pl_files)} file(s) from {pl_dir}")
    say(f"  Single lvl : {len(single_files)} of {n_all_single} file(s) "
        f"overlap in time")
    say(f"  Content    : {args.content_basis}, threshold "
        f"{args.content_threshold:g} g m-3")
    say(f"  Gap merge  : clear slabs thinner than {args.min_gap_m:g} m")
    say(f"  Layer floor: {args.min_layer_path:g} g m-2 condensate path")
    say(f"  Phase      : L if f_ice <= {args.liq_max:g}, "
        f"I if f_ice >= {args.ice_min:g}")
    say(f"  Regimes    : {', '.join(REGIME_SYM)} at "
        f"{args.regime_edges} m and the tropopause")
    say(f"  Grid stride: {args.stride}")

    single = xr.open_mfdataset(single_files, combine="by_coords", join="outer",
                               compat="no_conflicts")[list(SINGLE_VARS)]

    rng = np.random.default_rng(args.seed)
    n_reg, n_ph = N_REGIME, N_PHASE
    ml = args.max_layers

    sig = {g: [{}, {}] for g in GRANULARITIES}          # [domain, site]
    sig_month = {g: {} for g in GRANULARITIES}
    prof_sum = {}                                       # code -> (2, n_lev) sums
    prof_w = {}
    nlayer_w = np.zeros((2, ml + 1))
    regime_presence_w = np.zeros((2, n_reg))
    regime_phase_w = np.zeros((2, n_reg, n_ph))
    layer_phase_w = np.zeros((2, n_ph))
    total_w = np.zeros(2)
    cloudy_w = np.zeros(2)
    trop_found_w = np.zeros(2)
    n_truncated = 0
    n_steps = 0

    h_base = np.zeros((n_ph, HEIGHT_BINS_M.size - 1))
    h_top = np.zeros((n_ph, HEIGHT_BINS_M.size - 1))
    h_thick = np.zeros((n_ph, THICK_BINS_M.size - 1))
    h_fice = np.zeros(FICE_BINS.size - 1)
    h_toptemp = np.zeros((n_ph, TEMP_BINS_K.size - 1))
    h_lwp = np.zeros(LOGPATH_BINS.size - 1)
    h_iwp = np.zeros(LOGPATH_BINS.size - 1)
    h_base_fice = np.zeros((HEIGHT_BINS_M.size - 1, FICE_BINS.size - 1))

    era5_band = np.zeros((3, 2, 2))       # band x (mine 0/1) x (ERA5 0/1)

    samples = {"prof": [], "w": [], "sig_full": [], "sig_pr": [],
               "nlay": [], "month": []}
    p_pa_ref = None
    lat_ref = lon_ref = None
    months_seen = set()

    for i, f in enumerate(pl_files, 1):
        with xr.open_dataset(f) as pl_raw:
            pl = pl_raw.isel(latitude=slice(None, None, args.stride),
                             longitude=slice(None, None, args.stride)).load()
        times = pl["valid_time"].values
        try:
            sl = single.sel(valid_time=times).isel(
                latitude=slice(None, None, args.stride),
                longitude=slice(None, None, args.stride)).load()
        except KeyError:
            print(f"  !! {Path(f).name}: no single-level data for these times; "
                  f"skipped.", file=sys.stderr)
            continue

        dims = [d for d in pl["t"].dims if d != "pressure_level"]
        shape = tuple(pl.sizes[d] for d in dims)
        sp = sl["sp"].transpose(*dims).values.reshape(-1)

        fields = profile_fields(pl, sp, args.content_threshold,
                                args.content_basis)
        lay = classify_columns(
            fields, min_gap_m=args.min_gap_m,
            min_layer_path_g_m2=args.min_layer_path,
            liq_max=args.liq_max, ice_min=args.ice_min,
            regime_edges=args.regime_edges, max_layers=args.max_layers)
        n_truncated += lay["n_truncated"]

        if p_pa_ref is None:
            p_pa_ref = fields["p_pa"]
            lat_ref = pl["latitude"].values
            lon_ref = pl["longitude"].values

        # ---- weights: cos(latitude) cell-hours, and the single ARM cell
        lat = np.asarray(pl["latitude"].values, dtype=float)
        w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None],
                              (lat.size, pl.sizes["longitude"]))
        w_dom = np.broadcast_to(w2d, shape).reshape(-1).astype(float)
        smask, site_lat, site_lon = _site_mask(pl)
        w_site = np.broadcast_to(smask, shape).reshape(-1).astype(float)

        month = pd.DatetimeIndex(times).month.values
        m_col = np.broadcast_to(month[:, None, None], shape).reshape(-1)
        months_seen.update(int(m) for m in np.unique(month))

        present = lay["present"]
        n_lay = lay["n_layers"]
        cloudy = n_lay > 0
        # Occupancy is a property of the column, not of the weighting, so it is
        # computed once here rather than again inside the per-series loop.
        occupancy = [_regime_occupancy(lay, fields["z_trop_m"],
                                       args.regime_edges, r)
                     for r in range(n_reg)]

        for s, w in ((0, w_dom), (1, w_site)):
            total_w[s] += w.sum()
            cloudy_w[s] += w[cloudy].sum()
            trop_found_w[s] += w[fields["trop_found"]].sum()
            nlayer_w[s] += np.bincount(n_lay, weights=w, minlength=ml + 1)[:ml + 1]

            wc = w[cloudy]
            for g in GRANULARITIES:
                codes = pack_signature(present, lay["phase"], lay["regime"],
                                       lay["arrangement"], g)[cloudy]
                _accumulate(sig[g][s], codes, wc)
                if s == 0:
                    for m in np.unique(m_col[cloudy]):
                        sel = m_col[cloudy] == m
                        _accumulate(sig_month[g].setdefault(int(m), {}),
                                    codes[sel], wc[sel])

            # per-layer marginals, one row per (column, slot)
            wl = np.broadcast_to(w[:, None], present.shape)[present]
            reg = lay["regime"][present]
            ph = lay["phase"][present]
            np.add.at(regime_phase_w[s], (reg, ph), wl)
            np.add.at(layer_phase_w[s], ph, wl)
            for r in range(n_reg):
                regime_presence_w[s, r] += w[occupancy[r]].sum()

        # ---- composite profiles, keyed by the phase+regime signature
        codes_pr = pack_signature(present, lay["phase"], lay["regime"],
                                  lay["arrangement"], "phase+regime")
        _accumulate_profiles(prof_sum, prof_w, codes_pr[cloudy],
                             fields["lwp"][cloudy], fields["iwp"][cloudy],
                             w_dom[cloudy], args.max_profile_keys)

        # ---- diagnostic histograms, per layer
        wl = np.broadcast_to(w_dom[:, None], present.shape)[present]
        ph = lay["phase"][present]
        for p in range(n_ph):
            m = ph == p
            if not m.any():
                continue
            h_base[p] += np.histogram(lay["base_m"][present][m], HEIGHT_BINS_M,
                                      weights=wl[m])[0]
            h_top[p] += np.histogram(lay["top_m"][present][m], HEIGHT_BINS_M,
                                     weights=wl[m])[0]
            h_thick[p] += np.histogram(lay["thick_m"][present][m], THICK_BINS_M,
                                       weights=wl[m])[0]
            h_toptemp[p] += np.histogram(lay["top_t_k"][present][m], TEMP_BINS_K,
                                         weights=wl[m])[0]
        h_fice += np.histogram(lay["f_ice"][present], FICE_BINS, weights=wl)[0]
        h_base_fice += np.histogram2d(lay["base_m"][present],
                                      lay["f_ice"][present],
                                      bins=[HEIGHT_BINS_M, FICE_BINS],
                                      weights=wl)[0]
        with np.errstate(divide="ignore"):
            h_lwp += np.histogram(np.log10(np.maximum(lay["lwp"][present], 1e-6)),
                                  LOGPATH_BINS, weights=wl)[0]
            h_iwp += np.histogram(np.log10(np.maximum(lay["iwp"][present], 1e-6)),
                                  LOGPATH_BINS, weights=wl)[0]

        # ---- cross-check against ERA5's own lcc / mcc / hcc
        era5_band += _band_contingency(fields, lay, sl, dims, w_dom,
                                       args.era5_cover_threshold)

        # ---- stratified sample of raw profiles, for the clustering check
        _draw_sample(samples, rng, args.sample_per_file, cloudy, fields, lay,
                     w_dom, m_col, present)

        n_steps += pl.sizes["valid_time"]
        if i % 10 == 0 or i == len(pl_files):
            say(f"    [{i}/{len(pl_files)}] {Path(f).name}")

    sample = _finish_sample(samples, p_pa_ref)
    # Fingerprint of what was actually read, so a cache written against a
    # smaller archive can be detected instead of silently reused. The archive
    # grows: this analysis was first built on 63 files and the directory held
    # 120 a week later.
    source = [(Path(f).name, Path(f).stat().st_size) for f in pl_files]
    return Analysis(
        args=args, p_pa=p_pa_ref, lat=lat_ref, lon=lon_ref, source=source,
        sig=sig, sig_month=sig_month, prof_sum=prof_sum, prof_w=prof_w,
        nlayer_w=nlayer_w, regime_presence_w=regime_presence_w,
        regime_phase_w=regime_phase_w, layer_phase_w=layer_phase_w,
        total_w=total_w, cloudy_w=cloudy_w, trop_found_w=trop_found_w,
        n_truncated=n_truncated, n_steps=n_steps, n_files=len(pl_files),
        months=sorted(months_seen),
        h_base=h_base, h_top=h_top, h_thick=h_thick, h_fice=h_fice,
        h_toptemp=h_toptemp, h_lwp=h_lwp, h_iwp=h_iwp, h_base_fice=h_base_fice,
        era5_band=era5_band, sample=sample,
        site_lat=site_lat, site_lon=site_lon,
    )


# ----------------------------------------------------------------------------
# Pass helpers
# ----------------------------------------------------------------------------
def matching_single_files(pl_files, single_files):
    """The single-level files whose days overlap the pressure-level files.

    The single-level archive runs from 2000 and the pressure-level one covers a
    few months, so opening all 676 of the former to read ``sp`` for 63 of the
    latter costs minutes before the first byte of science is read. Both
    downloaders emit contiguous day ranges and rename only completed files into
    place, so the DAYS a file covers can be read off its name -- which is what
    ``days_covered_by_file`` does, and what the resume logic already relies on.

    Falls back to the full list if no name parses, rather than silently
    returning nothing.
    """
    want = set()
    for f in pl_files:
        want |= days_covered_by_file(Path(f))
    if not want:
        return list(single_files)
    keep = [f for f in single_files
            if days_covered_by_file(Path(f)) & want]
    return keep or list(single_files)


def _site_mask(ds, lat_deg=71.323, lon_deg=-156.616):
    """Boolean (lat, lon) mask with one True: the cell holding the ARM site at
    Utqiagvik. Same nearest-neighbour rule as
    ``plot_surface_class_timeseries.site_cell_mask``, repeated here so this
    module does not drag in a plotting script."""
    lats = np.asarray(ds["latitude"].values, dtype=float)
    lons = np.asarray(ds["longitude"].values, dtype=float)
    lons_signed = ((lons + 180.0) % 360.0) - 180.0
    i = int(np.argmin(np.abs(lats - lat_deg)))
    j = int(np.argmin(np.abs(((lons_signed - lon_deg) + 180.0) % 360.0 - 180.0)))
    mask = np.zeros((lats.size, lons.size), dtype=bool)
    mask[i, j] = True
    return mask, float(lats[i]), float(lons_signed[j])


def _regime_occupancy(lay, z_trop_m, regime_edges, regime):
    """Columns with ANY cloud whose vertical SPAN intersects ``regime``.

    Distinct from the regime code, which places a layer by its base alone. A
    deck based at 800 m and topped at 2.5 km is a BL cloud by base but occupies
    BL and LOW both, and "how often is there cloud in the boundary layer" wants
    the second reading.
    """
    edges = np.asarray(regime_edges, dtype=float)
    lo = np.concatenate([[0.0], edges])
    hi = np.concatenate([edges, [np.inf]])
    present = lay["present"]
    base, top = lay["base_m"], lay["top_m"]
    if regime == REGIME_TP:
        hit = present & (top > z_trop_m[:, None])
    else:
        band_lo, band_hi = lo[regime], hi[regime]
        band_hi = np.minimum(band_hi, z_trop_m[:, None])
        hit = present & (top > band_lo) & (base < band_hi)
    return hit.any(axis=-1)


def _accumulate_profiles(prof_sum, prof_w, codes, lwp, iwp, w, max_keys):
    """Weighted LWP/IWP profile sums per signature, for composite plots.

    Capped at ``max_keys`` distinct signatures: the tail is long and its
    composites are built from too few samples to mean anything. Signatures
    arriving after the cap are counted nowhere, which is fine because the cap is
    far beyond the number of types anyone plots.
    """
    if codes.size == 0:
        return
    uniq, inv = np.unique(codes, return_inverse=True)
    for k, c in enumerate(uniq.tolist()):
        if c not in prof_sum and len(prof_sum) >= max_keys:
            continue
        m = inv == k
        acc = prof_sum.get(c)
        if acc is None:
            acc = np.zeros((2, lwp.shape[1]))
            prof_sum[c] = acc
            prof_w[c] = 0.0
        acc[0] += (lwp[m] * w[m][:, None]).sum(axis=0)
        acc[1] += (iwp[m] * w[m][:, None]).sum(axis=0)
        prof_w[c] = prof_w[c] + float(w[m].sum())


def _band_contingency(fields, lay, sl, dims, w, cover_threshold):
    """2x2 agreement between this decomposition and ERA5's lcc / mcc / hcc.

    ERA5 splits low / medium / high at 800 and 450 hPa, so the same split
    applied to the layer decomposition gives a like-for-like test: does a column
    this code calls "cloudy below 800 hPa" also have lcc above threshold?
    Disagreement is not necessarily an error -- lcc is a maximum-random overlap
    of model-level cloud fraction, not a condensate threshold -- but a large
    one-sided bias means the threshold is wrong.

    Returns ``(3, 2, 2)`` weighted counts indexed [band, mine, era5].
    """
    out = np.zeros((3, 2, 2))
    p_hpa = fields["p_pa"] / 100.0
    bands = ((p_hpa >= ERA5_LOW_MID_HPA),
             (p_hpa < ERA5_LOW_MID_HPA) & (p_hpa >= ERA5_MID_HIGH_HPA),
             (p_hpa < ERA5_MID_HIGH_HPA))
    is_cloud = lay["is_cloud"]
    for b, (band, name) in enumerate(zip(bands, ("lcc", "mcc", "hcc"))):
        if name not in sl:
            continue
        mine = (is_cloud & band[None, :]).any(axis=-1)
        theirs = sl[name].transpose(*dims).values.reshape(-1) >= cover_threshold
        for mi in (0, 1):
            for th in (0, 1):
                sel = (mine == bool(mi)) & (theirs == bool(th))
                out[b, mi, th] += float(w[sel].sum())
    return out


def _bin_profile(values, z_bot, z_top, edges):
    """Redistribute a per-level mass path onto fixed height bins by overlap.

    Levels are 200-1000 m thick and the bins are 500 m, so assigning a level's
    whole path to the bin holding its midpoint would smear a thin cloud across
    the wrong bin. Splitting by the overlap fraction conserves mass and puts it
    where the layer actually is.
    """
    lo = np.maximum(z_bot[..., None], edges[None, None, :-1])
    hi = np.minimum(z_top[..., None], edges[None, None, 1:])
    overlap = np.clip(hi - lo, 0.0, None)                  # (n, n_lev, n_bin)
    dz = np.maximum(z_top - z_bot, 1e-6)[..., None]
    return (values[..., None] * overlap / dz).sum(axis=1)


def _draw_sample(samples, rng, per_file, cloudy, fields, lay, w, m_col, present):
    """Stratified sample of cloudy columns: a fixed quota per file, carrying the
    weight each drawn column stands for, so the sample is unbiased under
    ``sample_weight`` even though files differ in how much cloud they hold."""
    idx = np.flatnonzero(cloudy)
    if idx.size == 0 or per_file <= 0:
        return
    take = min(per_file, idx.size)
    pick = rng.choice(idx, size=take, replace=False)
    scale = idx.size / take
    prof = _bin_profile(np.stack([fields["lwp"][pick], fields["iwp"][pick]], 1)
                        .reshape(-1, fields["lwp"].shape[1]),
                        np.repeat(fields["z_bot_m"][pick], 2, axis=0),
                        np.repeat(fields["z_top_m"][pick], 2, axis=0),
                        PROFILE_BIN_EDGES_M).reshape(take, 2, -1)
    samples["prof"].append(prof.astype(np.float32))
    samples["w"].append((w[pick] * scale).astype(np.float32))
    samples["sig_full"].append(
        pack_signature(present[pick], lay["phase"][pick], lay["regime"][pick],
                       lay["arrangement"][pick], "full"))
    samples["sig_pr"].append(
        pack_signature(present[pick], lay["phase"][pick], lay["regime"][pick],
                       lay["arrangement"][pick], "phase+regime"))
    samples["nlay"].append(lay["n_layers"][pick].astype(np.int8))
    samples["month"].append(m_col[pick].astype(np.int8))


def _finish_sample(samples, p_pa):
    if not samples["prof"]:
        return None
    return dict(
        prof=np.concatenate(samples["prof"]),          # (n, 2, n_bin) g m-2
        weight=np.concatenate(samples["w"]),
        sig_full=np.concatenate(samples["sig_full"]),
        sig_pr=np.concatenate(samples["sig_pr"]),
        n_layers=np.concatenate(samples["nlay"]),
        month=np.concatenate(samples["month"]),
        bin_edges_m=PROFILE_BIN_EDGES_M,
    )


# ----------------------------------------------------------------------------
# The unsupervised cross-check
# ----------------------------------------------------------------------------
def cluster_features(sample, floor_g_m2=1e-3):
    """Feature matrix for clustering: log-scaled binned LWP and IWP profiles.

    ``log10(1 + W / floor)`` compresses five decades of water path into a range
    k-means can work in without letting one thick deck dominate every distance,
    and maps exactly zero to exactly zero so empty bins stay empty. Liquid and
    ice bins are concatenated, so the clustering sees magnitude AND vertical
    arrangement AND phase together -- the same three attributes the symbolic
    signature encodes, but continuously.
    """
    x = np.log10(1.0 + sample["prof"] / floor_g_m2)
    return x.reshape(x.shape[0], -1).astype(np.float64)


def cluster_profiles(sample, k_values=(4, 6, 8, 10, 12, 16), seed=DEFAULT_SEED,
                     n_init=10):
    """k-means over the sampled profiles, swept over k.

    Returns ``{"k": ..., "inertia": ..., "silhouette": ..., "labels": {k: ...},
    "centers": {k: ...}}``. Silhouette is computed on a 5,000-column subsample
    because it is O(n^2); inertia uses every sample. Neither picks k for you --
    look for the knee in inertia and the peak in silhouette, and remember that
    a clean k is not evidence the clusters are physical.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = cluster_features(sample)
    w = sample["weight"].astype(np.float64)
    rng = np.random.default_rng(seed)
    sub = rng.choice(x.shape[0], size=min(5000, x.shape[0]), replace=False)

    out = {"k": list(k_values), "inertia": [], "silhouette": [],
           "labels": {}, "centers": {}}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        lab = km.fit_predict(x, sample_weight=w)
        out["inertia"].append(float(km.inertia_))
        out["silhouette"].append(float(silhouette_score(x[sub], lab[sub])))
        out["labels"][k] = lab.astype(np.int16)
        out["centers"][k] = km.cluster_centers_.reshape(k, 2, -1)
    return out


def cluster_signature_crosstab(sample, labels, granularity="phase+regime",
                               top=12):
    """Cluster membership against the top symbolic signatures.

    Rows are clusters, columns the most common signatures, values the weighted
    percentage of each cluster. A cluster concentrated in one signature means
    the alphabet captured that mode; a cluster spread evenly across several
    means the continuous data holds a distinction the symbols throw away.
    """
    import pandas as pd
    codes = sample["sig_pr"] if granularity == "phase+regime" else sample["sig_full"]
    w = sample["weight"].astype(float)
    counter = {}
    _accumulate(counter, codes, w)
    keep = [c for c, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:top]]
    names = [decode_signature(c, granularity) for c in keep] + ["other"]
    n_k = int(labels.max()) + 1
    tab = np.zeros((n_k, len(names)))
    for ci in range(n_k):
        m = labels == ci
        for j, c in enumerate(keep):
            tab[ci, j] = w[m & (codes == c)].sum()
        tab[ci, -1] = w[m].sum() - tab[ci, :-1].sum()
    tot = np.maximum(tab.sum(axis=1, keepdims=True), 1e-30)
    return pd.DataFrame(100.0 * tab / tot,
                        index=[f"cluster {i}" for i in range(n_k)],
                        columns=names)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_heights(path, sp_da=None, sp_pa=101325.0, ref_level=1):
    """Check the hydrostatic heights against ERA5's own geopotential.

    Only 13 of the 63 archive files carry ``z``, which is why heights here are
    derived from ``t`` and ``q`` instead. Those 13 make the derivation testable.

    THE TEST IS CUMULATIVE, AND IT HAS TO BE. The obvious check -- compare
    ``diff(z_mid)`` against ``diff(z/g)`` level by level -- is invalid wherever
    the level SPACING changes, and this archive changes it twice (25 -> 50 hPa
    at 750, 50 -> 25 hPa at 250). Each level owns the slab between the midpoints
    to its neighbours, so at a spacing change that slab is asymmetric and
    ``0.5*(dz[k] + dz[k+1])`` stops being the distance between levels k and k+1.
    Measured on 2025-08-01/02 the pairwise error at those two joins is +67/-67 m
    and -156/+165 m -- equal and opposite, cancelling exactly, because the
    slabs are right and the pairwise statistic is wrong. Summed heights, which
    is what layer bases and tops actually are, agree to 21 m over 11 km.

    ``ref_level`` is the level the cumulative height is measured from; the
    default skips the lowest slab, whose bottom edge is clipped at the surface
    and so deliberately does NOT match a 1000 hPa geopotential.

    Returns error statistics in metres and percent, keyed by the top pressure of
    each span, plus the overall figures. Raises ``KeyError`` if there is no
    ``z``.
    """
    import xarray as xr
    with xr.open_dataset(path) as ds:
        if "z" not in ds:
            raise KeyError(f"{Path(path).name} has no geopotential")
        dims = [d for d in ds["t"].dims if d != "pressure_level"]
        n_col = int(np.prod([ds.sizes[d] for d in dims]))
        sp = (np.full(n_col, sp_pa) if sp_da is None
              else np.broadcast_to(np.asarray(sp_da, dtype=float).reshape(-1),
                                   (n_col,)))
        fields = profile_fields(ds, sp, DEFAULT_CONTENT_THRESHOLD_G_M3)
        lv = ds["pressure_level"].values.astype(float)
        order = np.argsort(-lv)
        z_era = (ds["z"].transpose(*dims, "pressure_level").values[..., order]
                 .reshape(n_col, -1) / G_M_S2)

    p_hpa = fields["p_pa"] / 100.0
    mine = fields["z_mid_m"] - fields["z_mid_m"][:, [ref_level]]
    era = z_era - z_era[:, [ref_level]]
    per_span = {}
    for k in range(ref_level + 1, p_hpa.size):
        err = mine[:, k] - era[:, k]
        per_span[float(p_hpa[k])] = dict(
            mean_error_m=float(err.mean()),
            rms_error_m=float(np.sqrt((err ** 2).mean())),
            mean_error_pct=float(100.0 * err.mean() / max(era[:, k].mean(), 1.0)),
        )
    top = mine[:, -1] - era[:, -1]
    return dict(
        n_columns=int(n_col), ref_hpa=float(p_hpa[ref_level]),
        per_span=per_span,
        top_mean_error_m=float(top.mean()),
        top_rms_error_m=float(np.sqrt((top ** 2).mean())),
        top_mean_error_pct=float(100.0 * top.mean() / max(era[:, -1].mean(), 1.0)),
    )


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def save_analysis(A, path):
    """Write an Analysis to a compressed .npz.

    The signature counters are dicts, so they go in as parallel code/weight
    arrays under generated keys; ``load_analysis`` reassembles them. Arguments
    are stored as a JSON blob so a reloaded analysis still knows what produced
    it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {}

    def put_counter(prefix, counter):
        codes = np.fromiter(counter.keys(), dtype=np.int64, count=len(counter))
        w = np.fromiter(counter.values(), dtype=np.float64, count=len(counter))
        blob[f"{prefix}__c"] = codes
        blob[f"{prefix}__w"] = w

    for g in GRANULARITIES:
        for s in (0, 1):
            put_counter(f"sig__{g}__{s}", A.sig[g][s])
        for m, c in A.sig_month[g].items():
            put_counter(f"sigm__{g}__{m}", c)
    codes = np.fromiter(A.prof_sum.keys(), dtype=np.int64, count=len(A.prof_sum))
    blob["prof__c"] = codes
    blob["prof__s"] = (np.stack([A.prof_sum[int(c)] for c in codes])
                       if codes.size else np.zeros((0, 2, 1)))
    blob["prof__w"] = np.array([A.prof_w[int(c)] for c in codes], dtype=float)

    for k in ("p_pa", "lat", "lon", "nlayer_w", "regime_presence_w",
              "regime_phase_w", "layer_phase_w", "total_w", "cloudy_w",
              "trop_found_w", "h_base", "h_top", "h_thick", "h_fice",
              "h_toptemp", "h_lwp", "h_iwp", "h_base_fice", "era5_band"):
        blob[k] = np.asarray(getattr(A, k))
    if A.sample is not None:
        for k, v in A.sample.items():
            blob[f"sample__{k}"] = np.asarray(v)

    meta = dict(args={k: (list(v) if isinstance(v, tuple) else
                          str(v) if isinstance(v, Path) else v)
                      for k, v in vars(A.args).items()},
                n_truncated=A.n_truncated, n_steps=A.n_steps,
                n_files=A.n_files, months=list(A.months),
                site_lat=A.site_lat, site_lon=A.site_lon,
                source=[list(x) for x in getattr(A, "source", [])])
    blob["meta__json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **blob)
    return path


def load_analysis(path):
    """Read back what ``save_analysis`` wrote."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta__json"]))

    def get_counter(prefix):
        return dict(zip(z[f"{prefix}__c"].tolist(), z[f"{prefix}__w"].tolist()))

    sig = {g: [get_counter(f"sig__{g}__{s}") for s in (0, 1)]
           for g in GRANULARITIES}
    sig_month = {g: {} for g in GRANULARITIES}
    for key in z.files:
        if key.startswith("sigm__") and key.endswith("__c"):
            _, g, m = key[:-3].split("__")
            sig_month[g][int(m)] = get_counter(f"sigm__{g}__{m}")

    codes = z["prof__c"]
    prof_sum = {int(c): z["prof__s"][i] for i, c in enumerate(codes)}
    prof_w = {int(c): float(z["prof__w"][i]) for i, c in enumerate(codes)}

    sample = None
    skeys = [k for k in z.files if k.startswith("sample__")]
    if skeys:
        sample = {k.split("__", 1)[1]: z[k] for k in skeys}

    args = SimpleNamespace(**meta["args"])
    kw = {k: z[k] for k in ("p_pa", "lat", "lon", "nlayer_w",
                            "regime_presence_w", "regime_phase_w",
                            "layer_phase_w", "total_w", "cloudy_w",
                            "trop_found_w", "h_base", "h_top", "h_thick",
                            "h_fice", "h_toptemp", "h_lwp", "h_iwp",
                            "h_base_fice", "era5_band")}
    return Analysis(args=args, sig=sig, sig_month=sig_month, prof_sum=prof_sum,
                    prof_w=prof_w, sample=sample,
                    source=[(n, int(sz)) for n, sz in meta.get("source", [])],
                    n_truncated=meta["n_truncated"], n_steps=meta["n_steps"],
                    n_files=meta["n_files"], months=meta["months"],
                    site_lat=meta["site_lat"], site_lon=meta["site_lon"], **kw)


def cache_status(A, region="barrow", storage="local", data_root=None,
                 max_files=0):
    """Is a loaded Analysis still built from the archive that is on disk now?

    Returns ``(stale, message)``. A cached pass is the right default -- the full
    archive takes minutes -- but a silently stale one is worse than no cache at
    all, because the notebook goes on to report months the archive no longer
    ends at. The download is incremental, so this WILL happen: the first cache
    here was built on 63 files and the directory held 120 a week later.

    The comparison is on file names and sizes, which catches both new days
    appearing and a short file being re-fetched at full length.
    """
    root = Path(data_root or STORAGE_ROOTS[storage])
    on_disk = sorted(glob.glob(str(root / f"{region}_pressure" / "*.nc")))
    if max_files:
        on_disk = on_disk[:max_files]
    now = [(Path(f).name, Path(f).stat().st_size) for f in on_disk]
    was = [tuple(x) for x in getattr(A, "source", [])]

    if not was:
        return True, ("cache predates the file fingerprint and cannot be "
                      "checked; recomputing")
    if now == was:
        return False, f"cache is current: {len(now)} file(s)"

    added = sorted({n for n, _ in now} - {n for n, _ in was})
    removed = sorted({n for n, _ in was} - {n for n, _ in now})
    resized = sorted({n for n, s in now} & {n for n, s in was}
                     & {n for n, s in set(now) ^ set(was)})
    bits = []
    if added:
        bits.append(f"{len(added)} new file(s), latest {added[-1]}")
    if removed:
        bits.append(f"{len(removed)} file(s) gone")
    if resized:
        bits.append(f"{len(resized)} file(s) changed size")
    return True, (f"cache is STALE ({len(was)} file(s) -> {len(now)} on disk): "
                  + "; ".join(bits))


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
SERIES_LABEL = ("domain (area-weighted)", "Utqiagvik cell")


def layer_count_percent(A, series=0):
    """Percent of CLOUDY cell-hours with each layer count, 1..max_layers."""
    w = A.nlayer_w[series]
    denom = max(float(w[1:].sum()), 1e-30)
    return 100.0 * w[1:] / denom


def print_report(A):
    """Everything the questions asked for, as text."""
    ml = int(A.nlayer_w.shape[1] - 1)
    print()
    print("=" * 78)
    print("What is in the archive")
    print("=" * 78)
    print(f"  Files            : {A.n_files}")
    print(f"  Hourly steps     : {A.n_steps:,}")
    print(f"  Months present   : {', '.join(str(m) for m in A.months)}")
    for s in (0, 1):
        cf = 100.0 * A.cloudy_w[s] / max(A.total_w[s], 1e-30)
        print(f"  Cloudy fraction  : {cf:5.1f}%   [{SERIES_LABEL[s]}]")
    tf = 100.0 * A.trop_found_w[0] / max(A.total_w[0], 1e-30)
    print(f"  Tropopause found : {tf:5.1f}% of columns "
          f"(the rest cannot separate HIGH from TP+)")
    if A.n_truncated:
        print(f"  !! {A.n_truncated:,} column(s) had more than {ml} layers and "
              f"were truncated")

    print()
    print("=" * 78)
    print("How many layers?   (percent of CLOUDY cell-hours)")
    print("=" * 78)
    print(f"  {'layers':>8}  {'domain':>10}  {'Utqiagvik':>10}")
    dom, site = layer_count_percent(A, 0), layer_count_percent(A, 1)
    for i in range(ml):
        tag = f"{i+1}" + ("+" if i == ml - 1 else "")
        print(f"  {tag:>8}  {dom[i]:9.2f}%  {site[i]:9.2f}%")

    print()
    print("=" * 78)
    print("Where are the layers?   (percent of ALL cell-hours with cloud whose")
    print("                         vertical span reaches into each regime)")
    print("=" * 78)
    for r in range(N_REGIME):
        d = 100.0 * A.regime_presence_w[0, r] / max(A.total_w[0], 1e-30)
        s = 100.0 * A.regime_presence_w[1, r] / max(A.total_w[1], 1e-30)
        print(f"  {REGIME_SYM[r]:>5}  {REGIME_NAME[r]:<28}  "
              f"{d:6.2f}%  {s:6.2f}%")
    print("  NOTE: the archive stops at 200 hPa, so TP+ is a LOWER BOUND.")

    print()
    print("=" * 78)
    print("What phase?   (percent of individual LAYERS, area-weighted)")
    print("=" * 78)
    tot = max(A.layer_phase_w[0].sum(), 1e-30)
    for p in range(N_PHASE):
        print(f"  {PHASE_SYM[p]:>2}  {PHASE_NAME[p]:<8}  "
              f"{100.0 * A.layer_phase_w[0, p] / tot:6.2f}%")
    print()
    print("  Layers by regime x phase, percent of all layers:")
    print("        " + "".join(f"{PHASE_SYM[p]:>10}" for p in range(N_PHASE)))
    for r in range(N_REGIME):
        row = "".join(f"{100.0*A.regime_phase_w[0, r, p]/tot:9.2f}%"
                      for p in range(N_PHASE))
        print(f"  {REGIME_SYM[r]:>5} {row}")

    print()
    print("=" * 78)
    print("How concentrated is the taxonomy?")
    print("=" * 78)
    print(f"  {'granularity':<14}{'distinct':>10}{'50%':>7}{'80%':>7}"
          f"{'90%':>7}{'95%':>7}{'99%':>7}")
    for g in GRANULARITIES:
        cov = coverage_curve(A.sig[g][0])
        print(f"  {g:<14}{cov['distinct']:>10}"
              + "".join(f"{cov[lv]:>7}" for lv in (50.0, 80.0, 90.0, 95.0, 99.0)))
    print("  Read: how many distinct signatures account for that share of "
          "cloudy hours.")

    for g in ("phase", "phase+regime", "full"):
        print()
        print("=" * 78)
        print(f"Most common cloud types -- granularity '{g}'")
        print("=" * 78)
        df = counter_table(A.sig[g][0], g, top=15)
        for _, row in df.iterrows():
            print(f"  {row['percent']:6.2f}%  (cum {row['cum_percent']:6.2f}%)  "
                  f"{row['signature']}")

    print()
    print("=" * 78)
    print("Cross-check against ERA5's own lcc / mcc / hcc")
    print("=" * 78)
    print(f"  {'band':<6}{'agree':>9}{'mine only':>12}{'ERA5 only':>12}")
    for b, name in enumerate(("low", "medium", "high")):
        t = A.era5_band[b]
        tot_b = max(t.sum(), 1e-30)
        agree = 100.0 * (t[0, 0] + t[1, 1]) / tot_b
        mine_only = 100.0 * t[1, 0] / tot_b
        era_only = 100.0 * t[0, 1] / tot_b
        print(f"  {name:<6}{agree:8.1f}%{mine_only:11.1f}%{era_only:11.1f}%")
    print("  Disagreement is expected -- lcc is a maximum-random overlap of "
          "model-level\n  cloud fraction, not a condensate threshold -- but a "
          "large one-sided bias\n  means the content threshold is off.")
    print("=" * 78)


def sensitivity(base_args, sweeps, max_files=8, top=8, granularity="phase+regime"):
    """Re-run the pass with one parameter changed at a time.

    ``sweeps`` is ``{attribute: [values]}``. Every combination is a full pass
    over ``max_files`` files, so keep both small. Returns a DataFrame with one
    row per (parameter, value, signature) giving the percentage, so the question
    "does the ranking move" can be answered by looking rather than argued about.

    The threshold parameters are not incidental. ``content_threshold`` decides
    what counts as cloud at all, ``min_gap_m`` decides the layer count, and
    ``content_basis`` decides whether the threshold is applied to a grid-box
    mean or to what an instrument inside the cloud would see. If the top types
    reorder under any of them, the taxonomy is reporting the threshold.
    """
    import pandas as pd
    rows = []
    for attr, values in sweeps.items():
        for v in values:
            A = prepare(args=argparse.Namespace(**vars(base_args)), quiet=True,
                        **{attr: v, "max_files": max_files})
            print(f"    {attr} = {v}")
            df = counter_table(A.sig[granularity][0], granularity, top=top)
            for _, r in df.iterrows():
                rows.append(dict(parameter=attr, value=str(v),
                                 signature=r["signature"],
                                 percent=r["percent"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _emit(fig, out_dir, stem, dpi=None):
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi or 150, bbox_inches="tight")
    print(f"  saved {path}")
    return path


def fig_layer_count(A, out_dir=None, dpi=None):
    """Frequency of 1, 2, 3, ... cloud layers, domain against the ARM cell."""
    import matplotlib.pyplot as plt
    ml = int(A.nlayer_w.shape[1] - 1)
    dom, site = layer_count_percent(A, 0), layer_count_percent(A, 1)
    x = np.arange(ml)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - 0.19, dom, 0.38, label=SERIES_LABEL[0], color="#3A6EA5")
    ax.bar(x + 0.19, site, 0.38, label=SERIES_LABEL[1], color="#D1495B")
    for xi, (d, s) in enumerate(zip(dom, site)):
        ax.text(xi - 0.19, d + 0.6, f"{d:.1f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, s + 0.6, f"{s:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" + ("+" if i == ml - 1 else "")
                        for i in range(ml)])
    ax.set_xlabel("number of cloud layers")
    ax.set_ylabel("percent of cloudy cell-hours")
    ax.set_title("How often is the cloud single- or multi-layered?\n"
                 f"gaps thinner than {A.args.min_gap_m:g} m do not split a layer",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_layer_count", dpi)
    return fig


def fig_coverage(A, out_dir=None, dpi=None):
    """Cumulative share of cloudy hours against the number of signatures."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = ("#2E4057", "#3A6EA5", "#E9724C", "#7A9E7E")
    for g, c in zip(GRANULARITIES, colors):
        counter = A.sig[g][0]
        if not counter:
            continue
        w = np.sort(np.fromiter(counter.values(), dtype=float))[::-1]
        cum = 100.0 * np.cumsum(w) / w.sum()
        ax.plot(np.arange(1, cum.size + 1), cum, label=f"{g}  ({cum.size} distinct)",
                color=c, lw=1.8)
    ax.axhline(80, ls=":", color="0.5", lw=1)
    ax.axhline(95, ls=":", color="0.5", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("number of distinct signatures, most common first")
    ax.set_ylabel("cumulative percent of cloudy cell-hours")
    ax.set_title("How few types cover most of the sky?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_coverage", dpi)
    return fig


def fig_top_types(A, granularity="phase+regime", top=18, out_dir=None, dpi=None):
    """The most common signatures, domain beside the ARM cell."""
    import matplotlib.pyplot as plt
    df = counter_table(A.sig[granularity][0], granularity, top=top)
    site = A.sig[granularity][1]
    site_tot = max(sum(site.values()), 1e-30)
    site_pct = [100.0 * site.get(int(c), 0.0) / site_tot for c in df["code"]]
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.34 * len(df) + 2.0))
    ax.barh(y + 0.19, df["percent"], 0.38, color="#3A6EA5", label=SERIES_LABEL[0])
    ax.barh(y - 0.19, site_pct, 0.38, color="#D1495B", label=SERIES_LABEL[1])
    ax.set_yticks(y)
    ax.set_yticklabels(df["signature"], fontsize=9, family="monospace")
    ax.set_xlabel("percent of cloudy cell-hours")
    ax.set_title(f"Most common cloud types -- '{granularity}'\n"
                 "bottom layer listed first", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, f"cloud_types_top_{granularity.replace('+', '_')}", dpi)
    return fig


def fig_regime_phase(A, out_dir=None, dpi=None):
    """Where layers sit and what phase they are, two views of the same pass."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    tot = max(A.layer_phase_w[0].sum(), 1e-30)
    grid = 100.0 * A.regime_phase_w[0] / tot
    im = ax1.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax1.set_xticks(range(N_PHASE))
    ax1.set_xticklabels([f"{PHASE_SYM[p]}\n{PHASE_NAME[p]}" for p in range(N_PHASE)])
    ax1.set_yticks(range(N_REGIME))
    ax1.set_yticklabels(REGIME_SYM)
    for r in range(N_REGIME):
        for p in range(N_PHASE):
            v = grid[r, p]
            ax1.text(p, r, f"{v:.1f}", ha="center", va="center", fontsize=9,
                     color="white" if v > grid.max() * 0.55 else "black")
    ax1.set_title("percent of all layers, by regime and phase", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.045, label="% of layers")

    pres = 100.0 * A.regime_presence_w[0] / max(A.total_w[0], 1e-30)
    ax2.barh(np.arange(N_REGIME), pres, color="#3A6EA5")
    ax2.set_yticks(range(N_REGIME))
    ax2.set_yticklabels([f"{REGIME_SYM[r]}\n{REGIME_NAME[r]}" for r in range(N_REGIME)],
                        fontsize=8)
    ax2.invert_yaxis()
    for r, v in enumerate(pres):
        ax2.text(v + 0.4, r, f"{v:.1f}%", va="center", fontsize=9)
    ax2.set_xlabel("percent of ALL cell-hours")
    ax2.set_title("how often is there cloud in each regime?\n"
                  "(by vertical span, not base alone)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_regime_phase", dpi)
    return fig


def fig_layer_properties(A, out_dir=None, dpi=None):
    """Base height, thickness, ice fraction and cloud-top temperature."""
    import matplotlib.pyplot as plt
    hb = 0.5 * (HEIGHT_BINS_M[:-1] + HEIGHT_BINS_M[1:])
    tb = 0.5 * (THICK_BINS_M[:-1] + THICK_BINS_M[1:])
    fb = 0.5 * (FICE_BINS[:-1] + FICE_BINS[1:])
    kb = 0.5 * (TEMP_BINS_K[:-1] + TEMP_BINS_K[1:])
    colors = ("#3A6EA5", "#E9724C", "#7A9E7E")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    ax = axes[0, 0]
    for p in range(N_PHASE):
        ax.plot(A.h_base[p] / max(A.h_base.sum(), 1e-30) * 100, hb,
                color=colors[p], label=PHASE_NAME[p], lw=1.6)
    # Log x: the lowest bin holds a quarter of every layer in the archive, so a
    # linear axis shows one spike and hides the entire free troposphere.
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 40)
    ax.set_xlabel("percent of all layers per 250 m (log)")
    ax.set_ylabel("layer base height above ground (m)")
    ax.set_title("where layers begin", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    for p in range(N_PHASE):
        ax.plot(tb, A.h_thick[p] / max(A.h_thick.sum(), 1e-30) * 100,
                color=colors[p], lw=1.6)
    ax.set_xlabel("layer geometric thickness (m)")
    ax.set_ylabel("percent of all layers per 100 m")
    # The sawtooth is the level grid, not the atmosphere: a layer's thickness
    # can only be a sum of whole level slabs, so the allowed values are
    # quantised at ~200 m low down and ~700 m aloft.
    ax.set_title("how thick they are\nsawtooth = the 23-level grid, not weather",
                 fontsize=10)
    ax.set_xlim(0, 4000)

    ax = axes[1, 0]
    ax.bar(fb, A.h_fice / max(A.h_fice.sum(), 1e-30) * 100,
           width=fb[1] - fb[0], color="#2E4057")
    ax.axvline(A.args.liq_max, ls=":", color="#D1495B")
    ax.axvline(A.args.ice_min, ls=":", color="#D1495B")
    ax.set_xlabel("layer ice mass fraction  IWP / (IWP + LWP)")
    ax.set_ylabel("percent of all layers")
    # Deliberately NOT claiming the middle is empty: it is not. The modes at 0
    # and 1 are strong, but roughly a third of all layers sit between the two
    # cuts, so "mixed" is a populated category and its boundaries are a choice,
    # not a natural gap. Sweep them before quoting an M fraction.
    ax.set_title("phase is bimodal but the middle is populated:\n"
                 "the L/M/I cuts (dotted) are a choice, not a gap", fontsize=10)

    ax = axes[1, 1]
    for p in range(N_PHASE):
        ax.plot(kb, A.h_toptemp[p] / max(A.h_toptemp.sum(), 1e-30) * 100,
                color=colors[p], lw=1.6)
    ax.axvline(273.15, ls=":", color="0.4")
    ax.axvline(233.15, ls=":", color="0.4")
    ax.set_xlabel("cloud-top temperature (K)")
    ax.set_ylabel("percent of all layers per 2 K")
    ax.set_title("cloud-top temperature by phase\n"
                 "dotted: 273 K and 233 K (homogeneous freezing)", fontsize=10)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_layer_properties", dpi)
    return fig


def fig_composite_profiles(A, granularity="phase+regime", top=8,
                           out_dir=None, dpi=None):
    """Mean LWP/IWP profile of each common type -- what the label looks like."""
    import matplotlib.pyplot as plt
    df = counter_table(A.sig[granularity][0], granularity, top=top)
    p_hpa = np.asarray(A.p_pa) / 100.0
    n = len(df)
    ncol = min(4, max(n, 1))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow),
                             squeeze=False, sharey=True)
    for k, (_, row) in enumerate(df.iterrows()):
        ax = axes[k // ncol][k % ncol]
        code = int(row["code"])
        w = A.prof_w.get(code, 0.0)
        if w <= 0:
            ax.set_axis_off()
            continue
        lwp, iwp = A.prof_sum[code] / w
        ax.plot(lwp, p_hpa, color="#D1495B", lw=1.6, label="liquid")
        ax.plot(iwp, p_hpa, color="#3A6EA5", lw=1.6, label="ice")
        ax.invert_yaxis()
        ax.set_title(f"{row['signature']}\n{row['percent']:.1f}% of cloudy hours",
                     fontsize=9, family="monospace")
        ax.spines[["top", "right"]].set_visible(False)
        if k % ncol == 0:
            ax.set_ylabel("pressure (hPa)")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    fig.supxlabel("mean water path per level (g m$^{-2}$)", fontsize=10)
    fig.suptitle("Composite profiles of the most common types", fontsize=12)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_composites", dpi)
    return fig


def season_month_order(months):
    """Months in season order rather than calendar order.

    An Aug-Mar record plotted as 1,2,3,8,...,12 cuts the season in half at the
    new year and puts deep winter to the left of late summer. The months are a
    CYCLIC sequence, so the right place to start is the month after the largest
    gap in that cycle -- which gives August for an Aug-Mar record and January
    for a full year, with no special-casing of either.
    """
    ms = sorted(set(int(m) for m in months))
    if len(ms) < 2:
        return ms
    gaps = [(ms[(i + 1) % len(ms)] - m) % 12 for i, m in enumerate(ms)]
    # A gap of 1 everywhere means the months wrap with no break at all -- a full
    # year -- so there is no natural season start and calendar order is right.
    # Without this the rotation is decided by an arbitrary tie-break and a full
    # year comes out starting in December.
    if max(gaps) <= 1:
        return ms
    start = (gaps.index(max(gaps)) + 1) % len(ms)
    return ms[start:] + ms[:start]


def fig_monthly(A, granularity="phase+regime", top=8, out_dir=None, dpi=None):
    """Seasonal march of the common types."""
    import matplotlib.pyplot as plt
    df = counter_table(A.sig[granularity][0], granularity, top=top)
    months = season_month_order(A.sig_month[granularity])
    if not months:
        return None
    codes = [int(c) for c in df["code"]]
    frac = np.zeros((len(codes) + 1, len(months)))
    for j, m in enumerate(months):
        c = A.sig_month[granularity][m]
        tot = max(sum(c.values()), 1e-30)
        for i, code in enumerate(codes):
            frac[i, j] = 100.0 * c.get(code, 0.0) / tot
        frac[-1, j] = 100.0 - frac[:-1, j].sum()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = list(df["signature"]) + ["everything else"]
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(months))
    for i, lab in enumerate(labels):
        ax.bar(range(len(months)), frac[i], bottom=bottom, label=lab,
               color="0.85" if i == len(labels) - 1 else cmap(i % 20))
        bottom += frac[i]
    ax.set_xticks(range(len(months)))
    _mon = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    ax.set_xticklabels([_mon[m - 1] for m in months])
    ax.set_xlabel("month (season order, not calendar order)")
    ax.set_ylabel("percent of cloudy cell-hours")
    ax.set_title(f"Composition of the cloudy hours by month -- '{granularity}'",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_monthly", dpi)
    return fig


def fig_era5_check(A, out_dir=None, dpi=None):
    """Agreement with ERA5's own lcc / mcc / hcc, per band."""
    import matplotlib.pyplot as plt
    names = ("low\n(p >= 800 hPa)", "medium\n(800-450 hPa)", "high\n(p < 450 hPa)")
    agree, mine_only, era_only = [], [], []
    for b in range(3):
        t = A.era5_band[b]
        tot = max(t.sum(), 1e-30)
        agree.append(100.0 * (t[0, 0] + t[1, 1]) / tot)
        mine_only.append(100.0 * t[1, 0] / tot)
        era_only.append(100.0 * t[0, 1] / tot)
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, agree, 0.6, label="agree", color="#7A9E7E")
    ax.bar(x, mine_only, 0.6, bottom=agree, label="condensate only", color="#E9724C")
    ax.bar(x, era_only, 0.6, bottom=np.array(agree) + np.array(mine_only),
           label="ERA5 cover only", color="#3A6EA5")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("percent of cell-hours")
    ax.set_title("This decomposition against ERA5's own lcc / mcc / hcc\n"
                 f"cover threshold {A.args.era5_cover_threshold:g}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_era5_check", dpi)
    return fig


def fig_cluster_sweep(cl, out_dir=None, dpi=None):
    """Inertia and silhouette against k -- neither of which picks k for you."""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(cl["k"], cl["inertia"], "o-", color="#3A6EA5")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia", color="#3A6EA5")
    ax2 = ax1.twinx()
    ax2.plot(cl["k"], cl["silhouette"], "s--", color="#D1495B")
    ax2.set_ylabel("silhouette", color="#D1495B")
    ax1.set_title("k-means over binned LWP/IWP profiles", fontsize=11)
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_cluster_sweep", dpi)
    return fig


def fig_cluster_centers(cl, k, sample, out_dir=None, dpi=None):
    """Centroid profiles -- the types the data finds without an alphabet."""
    import matplotlib.pyplot as plt
    centers = cl["centers"][k]
    lab = cl["labels"][k]
    w = sample["weight"].astype(float)
    share = [100.0 * w[lab == i].sum() / max(w.sum(), 1e-30) for i in range(k)]
    edges = np.asarray(sample["bin_edges_m"])
    zc = 0.5 * (edges[:-1] + edges[1:]) / 1000.0
    ncol = min(4, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for i in range(k):
        ax = axes[i // ncol][i % ncol]
        ax.plot(centers[i, 0], zc, color="#D1495B", lw=1.6, label="liquid")
        ax.plot(centers[i, 1], zc, color="#3A6EA5", lw=1.6, label="ice")
        ax.set_title(f"cluster {i} -- {share[i]:.1f}%", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if i % ncol == 0:
            ax.set_ylabel("height above ground (km)")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    for i in range(k, nrow * ncol):
        axes[i // ncol][i % ncol].set_axis_off()
    fig.supxlabel(r"centroid, $\log_{10}(1 + W/W_0)$", fontsize=10)
    fig.suptitle(f"k-means centroids, k = {k}", fontsize=12)
    fig.tight_layout()
    _emit(fig, out_dir, f"cloud_types_cluster_centers_k{k}", dpi)
    return fig


def fig_cluster_crosstab(tab, out_dir=None, dpi=None):
    """Clusters against symbolic signatures: does the alphabet hold up?"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * tab.shape[1],
                                    1.4 + 0.42 * tab.shape[0]))
    im = ax.imshow(tab.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns, rotation=45, ha="right", fontsize=8,
                       family="monospace")
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index, fontsize=9)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v >= 3.0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else "black")
    ax.set_title("Percent of each cluster falling in each signature\n"
                 "concentrated rows = the alphabet captured that mode",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="% of cluster")
    fig.tight_layout()
    _emit(fig, out_dir, "cloud_types_cluster_crosstab", dpi)
    return fig


ALL_FIGURES = (fig_layer_count, fig_coverage, fig_top_types, fig_regime_phase,
               fig_layer_properties, fig_composite_profiles, fig_monthly,
               fig_era5_check)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--storage", choices=sorted(STORAGE_ROOTS), default="local")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--region", default="barrow")
    p.add_argument("--max-files", type=int, default=0,
                   help="Read only the first N files. 0 = all.")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth grid cell in lat and lon. Neighbouring "
                        "cells are strongly correlated, so a stride of 4 costs "
                        "little information and checks that the ranking holds.")
    p.add_argument("--content-basis", choices=("gridmean", "incloud"),
                   default="gridmean",
                   help="Threshold the grid-box mean content, or the in-cloud "
                        "value (content / cloud fraction).")
    p.add_argument("--content-threshold", type=float,
                   default=DEFAULT_CONTENT_THRESHOLD_G_M3, metavar="G_M3")
    p.add_argument("--min-gap-m", type=float, default=DEFAULT_MIN_GAP_M)
    p.add_argument("--min-layer-path", type=float,
                   default=DEFAULT_MIN_LAYER_PATH_G_M2, metavar="G_M2")
    p.add_argument("--liq-max", type=float, default=DEFAULT_LIQ_MAX_FICE)
    p.add_argument("--ice-min", type=float, default=DEFAULT_ICE_MIN_FICE)
    p.add_argument("--regime-edges", type=float, nargs=3,
                   default=list(DEFAULT_REGIME_EDGES_M), metavar="M")
    p.add_argument("--max-layers", type=int, default=DEFAULT_MAX_LAYERS)
    p.add_argument("--max-profile-keys", type=int, default=4000)
    p.add_argument("--era5-cover-threshold", type=float, default=0.5)
    p.add_argument("--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cache", type=Path, default=None,
                   help="Write the finished Analysis here as .npz.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        A = prepare(args=args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    print_report(A)
    if args.cache:
        print(f"\n  cached -> {save_analysis(A, args.cache)}")
    if not args.no_figures:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        out_dir = args.output_dir or (Path(__file__).resolve().parent
                                      / "figures" / "cloud_types")
        print()
        for fn in ALL_FIGURES:
            fn(A, out_dir=out_dir, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
