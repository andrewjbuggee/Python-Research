"""
compare_lwp_methods_retrieved_profiles.py — compare the ways of turning a
retrieved droplet-size profile + cloud optical depth into a liquid water path
(LWP), and test Dion Ho's droplet-spectrum-width ("k" / "C") correction.

Run on a PROFILE + τ_c sweep run (the PT sweep), whose network retrieves a
7-level r_e profile AND τ_c, using the test-set predictions cached in
<run-dir>/pred_cache.npz and the truth stored in the synthetic HDF5.

=============================================================================
Physics — why r_e is the right moment, and where the width (k) can sneak in
=============================================================================

(1) The exact identity (any droplet size distribution n(r)).
    Hansen & Travis (1974, Space Sci. Rev. 16, 527–610), Eq. 2.48 (p. 547):

        β_ext = ∫ π r² Q_ext(r) n(r) dr  =  Q̄_ext · G ,   G = ∫ π r² n(r) dr

    where G is the geometric cross-section per unit volume and Q̄_ext is the
    area-weighted extinction efficiency (the extinction analogue of H&T
    Eq. 2.58).  With the effective radius of H&T Eq. 2.53 (p. 548),
    r_e = ∫ r·πr² n dr / G, the liquid water content is

        LWC = (4/3) π ρ_w ∫ r³ n dr  =  (4/3) ρ_w r_e G .

    Eliminating G:

        β_ext = 3 Q̄_ext LWC / (4 ρ_w r_e)                           (exact)
        LWP   = ∫ LWC dz = (4 ρ_w / (3 Q̄_ext)) ∫₀^τc r_e(τ) dτ       (exact)

    → (2/3) ρ_w ∫ r_e dτ for Q̄_ext = 2 (geometric-optics limit).
    No assumption about the width of n(r) is made anywhere.  r_e is exactly
    the moment ratio (M3/M2) that links droplet mass to droplet extinction,
    so using r_e instead of integrating over n(r) costs NOTHING, provided τ
    is the true extinction optical depth.

(2) Where the width enters — Dion Ho's claim.
    If β_ext is instead computed as  π Q̄_ext N r_e²  (the "monodisperse
    shortcut", used by the MATLAB profile finders before 2 Sep 2026; fixed in
    find_verticalProfiles_VOCALS_REx_ver2.m and find_verticalProfiles_ORACLES.m,
    commit cee5fbc), then because G = π N <r²> and

        k ≡ <r²>/r_e² = M2³ / (M0 M3²) = (r_v / r_e)³ ≤ 1

    (Martin, Johnson & Spice 1994, JAS 51, 1823; Grosvenor et al. 2018,
    Rev. Geophys. 56, 409, Eq. 13 — Dion calls this factor "C"), the shortcut
    gives τ_shortcut = τ_true / k ≥ τ_true (k ≤ 1 by Lyapunov's moment
    inequality).  An LWP built from (τ_shortcut, r_e) is then biased HIGH by
    1/k, and multiplying by k undoes it — that is Dion's correction.
    For H&T's standard gamma distribution (Eqs. 2.56–2.57, p. 549)

        k = (1 − v_e)(1 − 2 v_e)                 (Grosvenor et al. 2018, Eq. 13)

    and for libRadtran's gamma form n ∝ r^α exp(−(α+3) r / r_e) — the form
    of the custom Mie tables and of +prob/GammaDistribution_libRadtran.m —
    v_e = 1/(α+3), so k = (α+1)(α+2)/(α+3)².
    Note Grosvenor et al. (2018) Eq. 17, LWP = (5/9) ρ_w r_e τ_c, carries NO
    k: k enters their N_d equations, not LWP-from-(τ, r_e).

(3) Paper-2 Eq. 10 — what plot_lwp_analysis_profile_tau.py uses for the
    retrieved profiles:

        LWP = (2/3) ρ_w τ_c ∫ r_e³ dz / ∫ r_e² dz

    derived from LWP = (4/3) π ρ_w N ∫ r_e³ dz and τ_c = 2 π N ∫ r_e² dz with
    N constant.  BOTH lines use the shortcut (r_v → r_e and <r²> → r_e²), each
    off by the same factor k, which CANCELS in the ratio.  Eq. 10 is therefore
    width-independent when τ_c is the true optical depth and k, N are constant
    with height.  It is (2/3) ρ_w ∫ r_e dτ with dτ ∝ r_e² dz.

(4) The τ_c TRAINING LABEL in the synthetic HDF5 is
    750 · Q_lab · ∫ LWC / r_e dz with Q_lab = 2.32
    (06_synthetic_profile_generator.compute_tau_c) — the width-exact form of
    (1), but with Q_lab ≠ 2.  Eq. 10 assumes Q = 2, so applied to this τ it
    returns (Q_lab / 2) · LWP ≈ 1.16 · LWP whenever N is constant with height.

=============================================================================
What this script computes
=============================================================================

Reference ("truth"): LWP_ref = 1000 · |∫ LWC dz|  (LWC g m⁻³, z km → g m⁻²),
the generator's own definition (LWP_g_per_m2).

LWP methods, each applied to TRUE inputs (true r_e profile + true τ_c label;
isolates the bias of the formula itself) and to RETRIEVED inputs (network
r_e profile + network τ_c; formula bias + retrieval error):

    A   Eq. 10 as implemented now (Q = 2)          — Andrew, current
    D1  k(v_e = 0.037) · A                          — Dion, constant v_e
    D2  k(v_e = 1/(α+3)) · A, per profile           — Dion, oracle v_e from the
                                                       synthetic α (the network
                                                       does NOT retrieve α)
    B   (2 / Q_lab) · A                             — Eq. 10 inverted with the
                                                       label's own Q

τ-space scenario (TRUE inputs only — needs LWC(z) to build τ(z)), Dion's
formula exactly as written in his notebook, LWP = (2/3)ρ_w C ∫ r_e dτ:

    S0  τ(z) from the exact extinction profile, no k      → expect ≈ 0 bias
    S1  τ(z) from the shortcut, τ_shortcut(z) = τ(z)/k(α) → expect +(1/k − 1)
    S2  k(α) · S1 (Dion, per-profile v_e)                  → expect ≈ 0 bias
    S3  k(0.037) · S1 (Dion, constant v_e)                 → residual from the
                                                              α spread
    (S0–S3 invert with Q_lab so the Q mismatch of item 4 is not mixed in.)

Mie / libRadtran-geometry diagnostic (optional; needs the libRadtran custom
Mie tables).  Recompute τ from the true profile with Q̄_ext(r_e, α, λ) from
the same tables libRadtran uses, two ways:
    (i)  trapezoid over the 7 levels (what the label integral does), and
    (ii) the layer geometry libRadtran actually receives from
         write_wc_file_from_in_situ.m: an empty line is written at
         z_top + dz, and libRadtran treats each wc_file line as the layer
         ABOVE that altitude (libRadtran user guide, §3.3.5, "wc_file …
         interpreted as layer quantities") → 7 cloudy layers of thickness dz
         spanning z_base … z_top + dz, instead of 6 spanning z_base … z_top.
Comparing both with τ_label tests whether Q_lab = 2.32 reflects Mie
extinction (expected Q̄_ext ≈ 2.05–2.2 at 550 nm) or the extra layer.

Bias convention: bias = 100 · (estimate − reference) / reference
(positive ⇒ overestimate).  Paper 2's "average percent difference" is the
negative of the mean of this quantity.

Outputs (written to --out-dir, default = the run dir)
    lwp_method_comparison.png         2×2 figure
    lwp_method_comparison_stats.json  every number printed to the console

Usage
    python compare_lwp_methods_retrieved_profiles.py \\
        --run-dir hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004

Nothing in this script is stochastic (no RNG is used).

Author: Andrew J. Buggee (with Claude), Scripps / UCSD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "hyper_parameter_sweep"))

# Re-use the repo's HDF5 path resolution (Alpine path → local training_data/
# fallback) and the paper figure style, so this script behaves like
# plot_lwp_analysis_profile_tau.py.
from plot_percentile_profiles_synthetic import _resolve_h5_path  # noqa: E402
from plot_run004_figures import setup_style_paper  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Dion Ho's zero-median-bias constant v_e for VOCALS-REx (from his notebook
# effective_variance.ipynb, §5c: median 0.037, MAD 0.018, n = 125 profiles).
DION_VE_CONST = 0.037

# Q_ext the label τ_c was built with (06_synthetic_profile_generator.QEXT_EFF).
# Used only as a cross-check: the script measures the value implied by the
# HDF5 (τ_c / (750 ∫LWC/r_e dz)) and uses THAT.
QEXT_LABEL_EXPECTED = 2.32

DEFAULT_RUN_DIR = "hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004"
DEFAULT_MIE_DIR = (
    "/Users/andrewbuggee/Documents/libRadtran-2.0.6/data/wc/custom_mieTables"
)
# The synthetic RT runs use the 1–50 μm tables (use_35_or_50 = 50 in
# hysics_refl_from_synthetic_NN_inputs.m).
MIE_TABLE_TEMPLATE = "wc_mieTable_gamma_rEff_1-50microns_gammaDist_alpha_{:03d}.cdf"

DPI = 500

# Okabe-Ito colours (CVD-safe; same palette as the repo's paper figures).
# Each method also gets its own line style so identity is not colour-alone.
STYLE = {
    "A": dict(color="#D55E00", ls="-", label="A: Eq. 10, Q = 2 (current)"),
    "D1": dict(
        color="#CC79A7", ls="--", label=f"D1: Dion, k(v$_e$={DION_VE_CONST}) × A"
    ),
    "D2": dict(
        color="#009E73", ls="-.", label=r"D2: Dion, k(v$_e$=1/($\alpha$+3)) × A"
    ),
    "B": dict(color="#0072B2", ls=":", label=r"B: Eq. 10 with Q = Q$_{\rm label}$"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Size-distribution width factor
# ─────────────────────────────────────────────────────────────────────────────
def k_from_ve(v_e: np.ndarray | float) -> np.ndarray:
    """Width factor k = (r_v/r_e)³ = <r²>/r_e² for a gamma distribution.

    k = (1 − v_e)(1 − 2 v_e)   — Grosvenor et al. (2018), Eq. 13; Dion's "C".
    Valid for 0 ≤ v_e < 0.5 (H&T 1974 p. 549: the standard distribution is
    defined only for b = v_eff < 0.5).
    """
    v_e = np.asarray(v_e, dtype=np.float64)
    if np.any((v_e < 0) | (v_e >= 0.5)):
        raise ValueError("v_e must lie in [0, 0.5) for the gamma closure.")
    return (1.0 - v_e) * (1.0 - 2.0 * v_e)


def ve_from_libradtran_alpha(alpha: np.ndarray) -> np.ndarray:
    """Effective variance of libRadtran's gamma n(r) ∝ r^α exp(−(α+3) r/r_e).

    Moments ∝ b^-k Γ(α+1+k) with b = (α+3)/r_e, so
    v_e = M4 M2 / M3² − 1 = (α+4)/(α+3) − 1 = 1/(α+3).
    Identical to H&T (1974) Eq. 2.56 with b_HT = v_e = 1/(α+3).
    """
    return 1.0 / (np.asarray(alpha, dtype=np.float64) + 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# LWP formulae
# ─────────────────────────────────────────────────────────────────────────────
def lwp_eq10_g_m2(
    r_e_um: np.ndarray, tau_c: np.ndarray, z_km: np.ndarray, q_ext: float = 2.0
) -> np.ndarray:
    """Paper-2 Eq. 10 generalised to any bulk Q_ext.

        LWP = (4 ρ_w / (3 Q_ext)) · τ_c · ∫ r_e³ dz / ∫ r_e² dz

    With r_e in μm and ρ_w = 10⁶ g m⁻³, (4 ρ_w / 3 Q) · r[μm]·10⁻⁶ m
    = (4 / 3Q) · r[μm] g m⁻², i.e. (2/3) τ_c r̄ for Q = 2 — exactly the
    lwp_eq10_g_m2 of plot_lwp_analysis_profile_tau.py when q_ext = 2.

    Parameters
    ----------
    r_e_um : (n, L) effective radius, μm, top → base
    tau_c  : (n,)   cloud optical depth
    z_km   : (n, L) altitude of each level, km, top → base. Only the ratio of
             the two integrals is used, so the sign of dz and the cloud
             thickness cancel; on an evenly spaced grid this is identical to
             the unit-spacing trapezoid used in the existing script.
    q_ext  : bulk extinction efficiency the τ_c was defined with.
    """
    num = np.trapezoid(r_e_um**3, z_km, axis=1)  # ∝ ∫ r³ dz   (μm³ km)
    den = np.trapezoid(r_e_um**2, z_km, axis=1)  # ∝ ∫ r² dz   (μm² km)
    return (4.0 / (3.0 * q_ext)) * tau_c * (num / den)


def lwp_from_lwc_g_m2(lwc_g_m3: np.ndarray, z_km: np.ndarray) -> np.ndarray:
    """Reference LWP = 1000 · |∫ LWC dz| (LWC g m⁻³, z km → g m⁻²)."""
    return 1000.0 * np.abs(np.trapezoid(lwc_g_m3, z_km, axis=1))


def tau_profile_from_lwc(
    r_e_um: np.ndarray, lwc_g_m3: np.ndarray, z_km: np.ndarray, q_ext: float
) -> np.ndarray:
    """Cumulative optical depth τ(z) from cloud top, exact form of item (1).

    dτ/dz = 750 · Q · LWC[g m⁻³] / r_e[μm]   (per km; 750 = 3/(4ρ_w) in
    mixed units, see 06_synthetic_profile_generator.compute_tau_c).
    Returns (n, L) with τ = 0 at the top level (index 0), increasing to the
    base — the same cumulative trapezoid the label τ_c is the last value of.
    """
    beta_per_km = 750.0 * q_ext * lwc_g_m3 / np.maximum(r_e_um, 1e-6)  # (n, L)
    dz_km = np.abs(np.diff(z_km, axis=1))  # (n, L-1)
    dtau = 0.5 * (beta_per_km[:, 1:] + beta_per_km[:, :-1]) * dz_km  # (n, L-1)
    return np.concatenate(
        [np.zeros((r_e_um.shape[0], 1)), np.cumsum(dtau, axis=1)], axis=1
    )


def lwp_tau_space_g_m2(
    r_e_um: np.ndarray, tau_z: np.ndarray, q_ext: float
) -> np.ndarray:
    """LWP = (4 ρ_w / 3Q) ∫₀^τc r_e(τ) dτ — item (1), evaluated in τ-space.

    This is the form Dion writes (with Q = 2 and his C applied outside).
    Trapezoid over the τ levels; units as in lwp_eq10_g_m2.
    """
    return (4.0 / (3.0 * q_ext)) * np.trapezoid(r_e_um, tau_z, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Mie / libRadtran-geometry diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def load_mie_ext_tables(
    mie_dir: Path, alphas: np.ndarray, wavelength_um: float
) -> dict | None:
    """Read ext(r_e) at one wavelength from the libRadtran custom Mie tables.

    ext is the volume extinction coefficient per unit LWC,
    km⁻¹ / (g m⁻³), so β[km⁻¹] = ext(r_e) · LWC[g m⁻³] and the bulk
    efficiency is Q̄_ext = ext · r_e[μm] / 750.

    One table is read per distinct nearest-available α (tables exist for
    integer α 1–40 and 45, 50, …, 150), mirroring
    find_custom_mieTable_closest_to_alpha_profile.m.  Returns None if the
    directory or netCDF4 is unavailable, so the rest of the script still runs.
    """
    if not mie_dir.is_dir():
        print(f"  Mie-table dir not found ({mie_dir}) — skipping Mie diagnostic.")
        return None
    try:
        import netCDF4
    except ImportError:
        print("  netCDF4 not installed — skipping Mie diagnostic.")
        return None

    # Available α values, parsed from the filenames on disk
    available = sorted(
        int(p.stem.split("_alpha_")[-1])
        for p in mie_dir.glob(MIE_TABLE_TEMPLATE.replace("{:03d}", "*"))
    )
    if not available:
        print(f"  No 1–50 μm Mie tables in {mie_dir} — skipping Mie diagnostic.")
        return None
    available_arr = np.asarray(available)
    nearest_alpha = available_arr[
        np.abs(alphas[:, None] - available_arr[None, :]).argmin(axis=1)
    ]

    tables = {}
    for a in np.unique(nearest_alpha):
        with netCDF4.Dataset(mie_dir / MIE_TABLE_TEMPLATE.format(int(a))) as ds:
            wl_um = np.asarray(ds["wavelen"][:], dtype=np.float64)
            reff_um = np.asarray(ds["reff"][:], dtype=np.float64)
            i_wl = int(np.argmin(np.abs(wl_um - wavelength_um)))
            ext = np.asarray(ds["ext"][i_wl, :], dtype=np.float64)  # (n_reff,)
        tables[int(a)] = (reff_um, ext, wl_um[i_wl])
    return {"nearest_alpha": nearest_alpha, "tables": tables}


def tau_mie_two_geometries(
    r_e_um: np.ndarray, lwc_g_m3: np.ndarray, z_km: np.ndarray, mie: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """τ from Mie extinction, (i) 7-level trapezoid, (ii) libRadtran layers.

    (ii) reproduces what write_wc_file_from_in_situ.m hands libRadtran for a
    profile whose base is above the surface:

        z:   z_top+dz   z_1=z_top   z_2  …  z_7=z_base   0
        LWC: 0          LWC_1       LWC_2 … LWC_7        0

    and libRadtran's layer convention (user guide §3.3.5) assigns each line's
    values to the layer between that altitude and the next one ABOVE it, so
    layer [z_i, z_{i-1}] (with z_0 = z_top + dz) carries LWC_i, r_e,i: seven
    cloudy layers of thickness dz.  τ_layers = Σ_i ext(r_e,i) LWC_i dz.

    Returns (tau_trapz, tau_layers, q_bar_ext) with q_bar_ext (n, L).
    """
    n, n_lev = r_e_um.shape
    beta_per_km = np.empty_like(r_e_um)
    for i in range(n):
        reff_tab, ext_tab, _ = mie["tables"][int(mie["nearest_alpha"][i])]
        # libRadtran 'interpolate' → linear interpolation in r_eff
        beta_per_km[i] = np.interp(r_e_um[i], reff_tab, ext_tab) * lwc_g_m3[i]
    q_bar = beta_per_km / np.maximum(lwc_g_m3, 1e-12) * r_e_um / 750.0

    tau_trapz = np.abs(np.trapezoid(beta_per_km, z_km, axis=1))
    # write_wc_file_from_in_situ.m: dz = -mean(diff(z)) for a top→base grid
    dz_km = np.abs(np.mean(np.diff(z_km, axis=1), axis=1))  # (n,)
    tau_layers = beta_per_km.sum(axis=1) * dz_km
    return tau_trapz, tau_layers, q_bar


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def bias_pct(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """100 · (estimate − reference) / reference; positive ⇒ overestimate."""
    return 100.0 * (estimate - reference) / np.maximum(reference, 1e-9)


def bias_stats(b_pct: np.ndarray) -> dict:
    return {
        "mean_pct": float(np.mean(b_pct)),
        "median_pct": float(np.median(b_pct)),
        "mean_abs_pct": float(np.mean(np.abs(b_pct))),
        "median_abs_pct": float(np.median(np.abs(b_pct))),
        "rms_pct": float(np.sqrt(np.mean(b_pct**2))),
        "p05_pct": float(np.percentile(b_pct, 5)),
        "p95_pct": float(np.percentile(b_pct, 95)),
        # Paper-2 Table-3 "average percent difference" = mean of
        # 100 (truth − retrieved)/truth = −mean of this bias
        "paper2_avg_pct_diff": float(-np.mean(b_pct)),
    }


def print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(
        f"  {'method':<46s} {'mean':>8s} {'median':>8s} {'mean|b|':>8s} {'RMS':>8s}"
        f" {'p05':>8s} {'p95':>8s}"
    )
    print("  " + "-" * 98)
    for name, s in rows.items():
        print(
            f"  {name:<46s} {s['mean_pct']:+8.2f} {s['median_pct']:+8.2f}"
            f" {s['mean_abs_pct']:8.2f} {s['rms_pct']:8.2f}"
            f" {s['p05_pct']:+8.2f} {s['p95_pct']:+8.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def _hist_panel(
    ax, biases: dict[str, np.ndarray], title: str, bins: np.ndarray
) -> None:
    """Step histograms (one per method) of the % bias vs the LWC truth."""
    for key, b in biases.items():
        st = STYLE[key]
        ax.hist(
            np.clip(b, bins[0], bins[-1]),
            bins=bins,
            histtype="step",
            lw=1.4,
            color=st["color"],
            ls=st["ls"],
            label=f"{st['label']}  (median {np.median(b):+.1f} %)",
        )
    ax.axvline(0.0, color="k", lw=0.7)
    # Headroom so the legend (upper left) never sits on the histogram peaks
    ax.set_ylim(0.0, ax.get_ylim()[1] * 1.5)
    ax.set_xlabel("LWP bias vs ∫LWC dz  (%)  [tails clipped into the end bins]")
    ax.set_ylabel("profiles")
    ax.set_title(title)
    # Opaque, borderless legend box so the 0 % reference line doesn't cross the text
    ax.legend(
        loc="upper left", frameon=True, framealpha=1.0, edgecolor="none", fontsize=6.3
    )
    ax.grid(alpha=0.25, lw=0.4)


def make_figure(
    out_path: Path,
    bias_true: dict,
    bias_ret: dict,
    ve_alpha: np.ndarray,
    bias_s1: np.ndarray,
    bias_s2: np.ndarray,
    bias_s3: np.ndarray,
    mie_ratios: dict | None,
    q_label: float,
) -> None:
    setup_style_paper()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    bins = np.linspace(-40.0, 50.0, 91)

    _hist_panel(
        axes[0, 0],
        bias_true,
        "(a) formula bias only — TRUE r$_e$ profile + TRUE τ$_c$ label",
        bins,
    )
    _hist_panel(
        axes[0, 1],
        bias_ret,
        "(b) formula + retrieval — RETRIEVED r$_e$ profile + τ$_c$",
        bins,
    )

    # (c) Dion's setting: shortcut τ, then his k correction, vs v_e
    ax = axes[1, 0]
    ax.scatter(
        ve_alpha,
        bias_s1,
        s=4,
        alpha=0.35,
        color="#D55E00",
        edgecolors="none",
        label="S1: τ from π Q N r$_e^2$ (shortcut), no k",
    )
    ax.scatter(
        ve_alpha,
        bias_s2,
        s=4,
        alpha=0.35,
        color="#009E73",
        edgecolors="none",
        marker="s",
        label=r"S2: S1 × k(v$_e$=1/($\alpha$+3))  (Dion, per profile)",
    )
    ax.scatter(
        ve_alpha,
        bias_s3,
        s=4,
        alpha=0.35,
        color="#CC79A7",
        edgecolors="none",
        marker="^",
        label=f"S3: S1 × k(v$_e$={DION_VE_CONST})  (Dion, constant)",
    )
    v_grid = np.linspace(0.0, min(0.45, ve_alpha.max() * 1.05), 200)
    ax.plot(
        v_grid,
        100.0 * (1.0 / k_from_ve(v_grid) - 1.0),
        color="k",
        lw=1.0,
        label="theory: 100 (1/k − 1)",
    )
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"effective variance v$_e$ = 1/($\alpha$+3) of the synthetic cloud")
    ax.set_ylabel("LWP bias vs ∫LWC dz  (%)")
    ax.set_title("(c) Dion's case: τ-space LWP with a shortcut τ(z), true inputs")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3, markerscale=3)
    ax.grid(alpha=0.25, lw=0.4)

    # (d) what Q_label = 2.32 is absorbing
    ax = axes[1, 1]
    if mie_ratios is None:
        ax.text(
            0.5,
            0.5,
            "Mie diagnostic skipped\n(no libRadtran Mie tables)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        rbins = np.linspace(0.8, 1.25, 46)
        for key, color, ls, lab in [
            (
                "label_over_mie_trapz",
                "#0072B2",
                "-",
                r"τ$_{\rm label}$ / τ$_{\rm Mie}$, 7-level trapezoid",
            ),
            (
                "label_over_mie_layers",
                "#E69F00",
                "--",
                r"τ$_{\rm label}$ / τ$_{\rm Mie}$, libRadtran layers (7 × dz)",
            ),
        ]:
            r = mie_ratios[key]
            ax.hist(
                np.clip(r, rbins[0], rbins[-1]),
                bins=rbins,
                histtype="step",
                lw=1.4,
                color=color,
                ls=ls,
                label=f"{lab}  (median {np.median(r):.3f})",
            )
        ax.axvline(1.0, color="k", lw=0.7)
        ax.set_ylim(0.0, ax.get_ylim()[1] * 1.35)  # legend headroom
        ax.set_xlabel(r"ratio of optical depths")
        ax.set_ylabel("profiles")
        ax.set_title(
            f"(d) what Q$_{{\\rm label}}$ = {q_label:.2f} absorbs "
            f"(Mie Q̄$_{{\\rm ext}}$ median {mie_ratios['q_bar_median']:.3f})"
        )
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            edgecolor="none",
            fontsize=6.3,
        )
        ax.grid(alpha=0.25, lw=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="PT-sweep run directory containing pred_cache.npz.",
    )
    p.add_argument(
        "--h5-path", default=None, help="Override the HDF5 path stored in summary.json."
    )
    p.add_argument(
        "--ve-const",
        type=float,
        default=DION_VE_CONST,
        help="Constant v_e for Dion's correction (default 0.037).",
    )
    p.add_argument(
        "--mie-dir",
        default=DEFAULT_MIE_DIR,
        help="libRadtran custom Mie-table directory ('' to skip).",
    )
    p.add_argument(
        "--mie-wavelength-um",
        type=float,
        default=0.55,
        help="Wavelength (μm) at which Q̄_ext is read from the tables.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write the figure and JSON (default: run dir).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (
        (repo_root / args.run_dir).resolve()
        if not Path(args.run_dir).is_absolute()
        else Path(args.run_dir)
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Test-set predictions (written by plot_run_figures_profile_tau.py) ──
    cache_path = run_dir / "pred_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing — run plot_run_figures_profile_tau.py "
            f"--run-dir {run_dir} first."
        )
    cache = np.load(cache_path)
    if "tau_pred" not in cache.files:
        raise RuntimeError("pred_cache.npz has no τ arrays — not a profile+τ run.")
    r_e_ret_um = cache["pred"].astype(np.float64)  # (n, 7) retrieved r_e, μm, top→base
    r_e_true_cache = cache["true"].astype(np.float64)  # (n, 7) true r_e (cross-check)
    tau_ret = cache["tau_pred"].astype(np.float64)  # (n,) retrieved τ_c
    tau_true_cache = cache["tau_true"].astype(np.float64)
    h5_idx = cache["h5_idx"].astype(np.int64)
    n = h5_idx.size

    # ── Truth from the HDF5 (h5py fancy indexing needs increasing indices) ──
    with (run_dir / "summary.json").open() as f:
        summary = json.load(f)
    h5_path = _resolve_h5_path(summary["data"]["h5_path"], args.h5_path)
    order = np.argsort(h5_idx)
    unsort = np.argsort(order)
    with h5py.File(h5_path, "r") as f:
        r_e_true_um = f["profiles_raw"][h5_idx[order]].astype(np.float64)[
            unsort
        ]  # (n, 7) μm
        lwc_g_m3 = f["lwc"][h5_idx[order]].astype(np.float64)[unsort]  # (n, 7)
        z_km = f["profiles_raw_z"][h5_idx[order]].astype(np.float64)[unsort]  # (n, 7)
        alpha = f["alpha"][h5_idx[order]].astype(np.float64)[unsort]  # (n,)
        tau_label = f["tau_c"][h5_idx[order]].astype(np.float64)[unsort]  # (n,)

    print(f"Run dir : {run_dir}")
    print(f"HDF5    : {h5_path}")
    print(f"Test set: {n:,} profiles × {r_e_true_um.shape[1]} levels")

    # Cross-checks: the cache's truth must be the HDF5 truth for these rows
    if not np.allclose(r_e_true_cache, r_e_true_um, rtol=1e-5, atol=1e-4):
        raise RuntimeError("pred_cache 'true' ≠ HDF5 profiles_raw — wrong HDF5?")
    if not np.allclose(tau_true_cache, tau_label, rtol=1e-5):
        raise RuntimeError("pred_cache 'tau_true' ≠ HDF5 tau_c — wrong HDF5?")

    # ── Which Q_ext is the τ_c label defined with? Measure it. ─────────────
    int_lwc_over_re = np.abs(np.trapezoid(lwc_g_m3 / r_e_true_um, z_km, axis=1))
    q_implied = tau_label / (750.0 * int_lwc_over_re)
    q_label = float(np.median(q_implied))
    print(
        f"\nQ_ext implied by the τ_c label: median {q_label:.4f} "
        f"(p01–p99 {np.percentile(q_implied, 1):.4f}–{np.percentile(q_implied, 99):.4f}; "
        f"06_synthetic_profile_generator.QEXT_EFF = {QEXT_LABEL_EXPECTED})"
    )

    # ── Reference LWP and width factors ───────────────────────────────────
    lwp_ref = lwp_from_lwc_g_m2(lwc_g_m3, z_km)  # g m⁻²
    ve_alpha = ve_from_libradtran_alpha(alpha)  # per profile
    k_alpha = k_from_ve(ve_alpha)
    k_const = float(k_from_ve(args.ve_const))
    print(
        f"Width factor k: constant v_e={args.ve_const} → k = {k_const:.4f};  "
        f"per-profile v_e=1/(α+3): median v_e {np.median(ve_alpha):.3f} "
        f"(p05–p95 {np.percentile(ve_alpha, 5):.3f}–{np.percentile(ve_alpha, 95):.3f}), "
        f"median k {np.median(k_alpha):.3f}"
    )

    # ── Methods A, D1, D2, B on TRUE and RETRIEVED inputs ─────────────────
    def methods(r_e_um: np.ndarray, tau_c: np.ndarray) -> dict[str, np.ndarray]:
        a = lwp_eq10_g_m2(r_e_um, tau_c, z_km, q_ext=2.0)
        return {
            "A": a,
            "D1": k_const * a,
            "D2": k_alpha * a,
            "B": lwp_eq10_g_m2(r_e_um, tau_c, z_km, q_ext=q_label),
        }

    lwp_true_in = methods(r_e_true_um, tau_label)
    lwp_ret_in = methods(r_e_ret_um, tau_ret)
    bias_true = {k: bias_pct(v, lwp_ref) for k, v in lwp_true_in.items()}
    bias_ret = {k: bias_pct(v, lwp_ref) for k, v in lwp_ret_in.items()}

    names = {
        "A": "A  Eq.10, Q=2 (current)",
        "D1": f"D1 Dion k(v_e={args.ve_const})·A",
        "D2": "D2 Dion k(v_e=1/(α+3))·A  [oracle α]",
        "B": f"B  Eq.10, Q=Q_label={q_label:.2f}",
    }
    stats_true = {names[k]: bias_stats(v) for k, v in bias_true.items()}
    stats_ret = {names[k]: bias_stats(v) for k, v in bias_ret.items()}
    print_table(
        "LWP bias vs ∫LWC dz, TRUE inputs (formula bias only), % " "[+ = overestimate]",
        stats_true,
    )
    print_table(
        "LWP bias vs ∫LWC dz, RETRIEVED inputs (formula + retrieval), %", stats_ret
    )
    print(
        "\n  B on TRUE inputs is the residual of Eq. 10's constant-N assumption plus\n"
        "  7-level quadrature — the only part of plot_lwp_analysis_profile_tau.py's\n"
        "  'constant-N assumption bias' that is actually due to constant N."
    )

    # ── τ-space scenario: Dion's formula with exact vs shortcut τ(z) ───────
    tau_z_exact = tau_profile_from_lwc(r_e_true_um, lwc_g_m3, z_km, q_ext=q_label)
    tau_z_shortcut = (
        tau_z_exact / k_alpha[:, None]
    )  # what π Q N r_e² dz would integrate to
    lwp_s0 = lwp_tau_space_g_m2(r_e_true_um, tau_z_exact, q_ext=q_label)
    lwp_s1 = lwp_tau_space_g_m2(r_e_true_um, tau_z_shortcut, q_ext=q_label)
    scen = {
        "S0 τ-space, exact τ(z), no k": bias_pct(lwp_s0, lwp_ref),
        "S1 τ-space, shortcut τ(z), no k": bias_pct(lwp_s1, lwp_ref),
        "S2 Dion: k(1/(α+3))·S1": bias_pct(k_alpha * lwp_s1, lwp_ref),
        f"S3 Dion: k({args.ve_const})·S1": bias_pct(k_const * lwp_s1, lwp_ref),
    }
    stats_scen = {k: bias_stats(v) for k, v in scen.items()}
    print_table(
        "τ-space LWP = (4ρ_w/3Q)∫r_e dτ, TRUE inputs — Dion's setting, %", stats_scen
    )
    print(
        "  (The label τ_c equals the last value of the exact τ(z): max |Δ| = "
        f"{np.max(np.abs(tau_z_exact[:, -1] - tau_label)):.2e})"
    )

    # ── Mie / libRadtran-geometry diagnostic ──────────────────────────────
    mie_ratios = None
    mie = (
        load_mie_ext_tables(Path(args.mie_dir), alpha, args.mie_wavelength_um)
        if args.mie_dir
        else None
    )
    if mie is not None:
        wl_used = next(iter(mie["tables"].values()))[2]
        tau_mie_trapz, tau_mie_layers, q_bar = tau_mie_two_geometries(
            r_e_true_um, lwc_g_m3, z_km, mie
        )
        mie_ratios = {
            "wavelength_um": float(wl_used),
            "q_bar_median": float(np.median(q_bar)),
            "q_bar_p05": float(np.percentile(q_bar, 5)),
            "q_bar_p95": float(np.percentile(q_bar, 95)),
            "label_over_mie_trapz": tau_label / tau_mie_trapz,
            "label_over_mie_layers": tau_label / tau_mie_layers,
        }
        print(f"\nMie diagnostic at λ = {wl_used:.3f} μm (custom libRadtran tables):")
        print(
            f"  Q̄_ext per level: median {mie_ratios['q_bar_median']:.3f} "
            f"(p05–p95 {mie_ratios['q_bar_p05']:.3f}–{mie_ratios['q_bar_p95']:.3f})"
            f"  vs Q_label {q_label:.3f}"
        )
        for key, lab in [
            ("label_over_mie_trapz", "τ_label / τ_Mie (7-level trapezoid)"),
            ("label_over_mie_layers", "τ_label / τ_Mie (libRadtran 7 layers)"),
        ]:
            r = mie_ratios[key]
            print(
                f"  {lab:<40s}: median {np.median(r):.3f}  "
                f"p05–p95 [{np.percentile(r, 5):.3f}, {np.percentile(r, 95):.3f}]"
            )

    # ── Figure + JSON ─────────────────────────────────────────────────────
    fig_path = out_dir / "lwp_method_comparison.png"
    make_figure(
        fig_path,
        bias_true,
        bias_ret,
        ve_alpha,
        scen["S1 τ-space, shortcut τ(z), no k"],
        scen["S2 Dion: k(1/(α+3))·S1"],
        scen[f"S3 Dion: k({args.ve_const})·S1"],
        mie_ratios,
        q_label,
    )
    print(f"\nFigure -> {fig_path}")

    out = {
        "run_dir": str(run_dir),
        "h5_path": str(h5_path),
        "n_test": int(n),
        "bias_convention": "100*(estimate - LWC_integrated_truth)/truth; + = overestimate",
        "q_label_implied_median": q_label,
        "ve_const": args.ve_const,
        "k_const": k_const,
        "ve_alpha_median": float(np.median(ve_alpha)),
        "k_alpha_median": float(np.median(k_alpha)),
        "true_inputs": stats_true,
        "retrieved_inputs": stats_ret,
        "tau_space_scenario_true_inputs": stats_scen,
    }
    if mie_ratios is not None:
        out["mie_diagnostic"] = {
            k: (
                v
                if np.isscalar(v)
                else {
                    "median": float(np.median(v)),
                    "p05": float(np.percentile(v, 5)),
                    "p95": float(np.percentile(v, 95)),
                }
            )
            for k, v in mie_ratios.items()
        }
    json_path = out_dir / "lwp_method_comparison_stats.json"
    with json_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Stats  -> {json_path}")


if __name__ == "__main__":
    main()
