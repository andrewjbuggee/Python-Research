"""
pc_truncation_analysis.py
=========================
How many principal components (PCs) of log r_e(z) and log LWC(z) should the
synthetic-profile generator (06_synthetic_profile_generator.py) keep?

The script evaluates the truncation k against two kinds of criteria and
prints / plots them side by side so that a single, stated rule can be chosen
for the paper.

Part A — statistical criteria (how much of the in-situ variability does k
modes describe, and how many modes are distinguishable from sampling noise?)
---------------------------------------------------------------------------
A1. Fraction of variance explained, FVE(k) = Σ_{i≤k} λ_i / Σ_i λ_i, where λ_i
    are the eigenvalues of the sample covariance of the log-transformed,
    mean-removed profiles (Jolliffe 2002, §6.1; Wilks 2019, PCA chapter).
    Reported three ways:
      (i)   in-sample, log space (the textbook number);
      (ii)  leave-one-profile-out (LOO), log space: each profile is projected
            onto a basis fitted WITHOUT it, which removes the optimism of (i);
      (iii) LOO, physical space: the LOO reconstruction is exponentiated and
            compared with the measured profile in μm (r_e) or g/m³ (LWC), so the
            number refers to the quantity the network actually retrieves.
    (ii)/(iii) are NOT used to *select* k: row-wise LOO reconstruction error
    always decreases with k (Bro et al. 2008); they quantify what a chosen k
    retains out of sample.
A2. LOO reconstruction RMSE vs k, in physical units — compared with the
    in-situ instrument uncertainty (draft §III.C.1 quotes ~1 μm for r_e).
A3. North et al. (1982) "rule of thumb": the sampling error of eigenvalue λ_k
    is δλ_k ≈ λ_k √(2/N). If neighbouring eigenvalues are closer than this,
    their EOFs are statistically mixed and should be kept or dropped together.
    N is the number of profiles; profiles from the same flight are not fully
    independent, so this δλ is a lower bound.
A4. Broken-stick rule (Frontier 1976; Jackson 1993; Jolliffe 2002 §6.1):
    keep PC k while its variance fraction exceeds the expected k-th largest
    fraction of a stick of unit length broken at random into p pieces,
    b_k = (1/p) Σ_{i=k}^{p} 1/i.
A5. Parallel analysis (Horn 1965; Preisendorfer's "Rule N" is the same idea,
    Overland & Preisendorfer 1982): eigenvalues of data in which each level is
    permuted independently across profiles (destroys inter-level covariance,
    keeps each level's variance). Keep PCs whose eigenvalue exceeds the 95th
    percentile of the permutation eigenvalues.
A6. Gavish & Donoho (2014) optimal hard threshold for singular values with
    unknown white-noise level: τ* = ω(β)·median(σ_i), β = min(m,n)/max(m,n).

Part B — retrievability criterion (how many modes can HySICS constrain?)
------------------------------------------------------------------------
This is the criterion that speaks to Peter's argument. Linear optimal-estimation
information content (Rodgers 2000, ch. 2) is computed with
  * the PRIOR covariance S_a = the in-situ covariance of log r_e on the common
    grid (its eigenvectors are exactly the PCs), augmented with ln τ_c and its
    in-situ cross-covariance (τ is retrieved jointly, so it must be an unknown);
  * the JACOBIAN K from Platnick's (2000) linearisation
        dR_λ / d ln r_e(τ_j) = (dR_λ/d ln r_e)_homog · w_λ(τ_j) Δτ_j · r̄_e(τ_j)/r_0
    where w_λ(τ) are the HySICS weighting functions behind Fig. 5 of the
    draft and (dR_λ/d ln r_e)_homog comes from compute_hysics_re_sensitivity.py
    (libRadtran/DISORT, homogeneous cloud, τ = 8);
  * the NOISE covariance S_ε = diag(R_λ²)·(σ_rad² I + σ_fm² C), with
    σ_rad = 0.3 % (HySICS), σ_fm the forward-model uncertainty (swept), and C
    the band-to-band correlation of forward-model error (identity by default,
    matching the independent-band assumption of draft Eq. 5).
For each PC k the script reports the diagonal of the averaging kernel A_kk
(Rodgers 2000, §2.4): the fraction of that mode's prior
variability that the measurement — rather than the prior — determines. A mode
with A_kk ≥ 0.5 is "more measured than assumed". The total degrees of freedom
for signal, DFS = tr(A) restricted to the r_e block, is the information-content
estimate of how many independent profile modes HySICS can retrieve.

Caveats (stated in the printed summary as well)
  * Linear, single-scene (τ = 8, sza ≈ 49°, vza ≈ 32°) analysis; the weighting
    functions were computed for a 12 → 5 μm adiabatic cloud.
  * Only r_e(z) and ln τ are unknown; water vapour, cloud height, α, aerosol
    and surface are treated as known, so Part B is an UPPER bound.
  * Independent band noise across 636 bands is optimistic (draft §III.D says
    so too). FM_BAND_CORR > 0 adds a spectrally FLAT correlated error; that is
    not a pessimistic test (a flat relative error is easy to separate from the
    spectrally shaped r_e signal and DFS can even rise), so it is left at 0.

Outputs → pc_truncation/figures/ and pc_truncation/outputs/
  pc_truncation_summary_<date>.txt   human-readable report
  pc_truncation_table_<date>.csv     per-k numbers for r_e and LWC
  fig1_scree_<date>.png              eigenvalue spectra + A3–A6 thresholds
  fig2_variance_retained_<date>.png  FVE(k) curves (A1)
  fig3_loo_rmse_<date>.png           LOO RMSE vs k in physical units (A2)
  fig4_averaging_kernel_<date>.png   per-PC A_kk vs σ_fm (Part B)

References (full list with DOIs in pc_truncation/README_pc_truncation.md)

Author: Claude (assistant) for Andrew J. Buggee, Nov 2026
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# ── Configuration ──────────────────────────────────────────────────────────────
# Same in-situ file and filters as 06_synthetic_profile_generator.py, so the
# PCs analysed here are the PCs the generator truncates.
MAT_FILE = Path('/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/'
                'Hyperspectral_Cloud_Retrievals/Neural_Network/'
                'Training_data_set/synthetic_profile_inputs/'
                'insitu_era5_synthetic_inputs_combined_08-Nov-2026.mat')
TAU_C_MIN, TAU_C_MAX = 3.0, 40.0     # same τ_c window as the generator
TAU_C_SOURCE = 'stored'              # same τ_c source as the generator
L_COMMON = 80                        # common normalised-altitude grid length
LWC_EPS = 1e-3                       # g/m³ floor before log (generator value)

K_CURRENT_RE, K_CURRENT_LWC = 6, 8   # truncation used in draft 23
K_MAX_REPORT = 15                    # largest k shown in tables / plots

# Instrument uncertainty quoted in draft §III.C.1 (CDP r_e, "order 1 μm").
RE_INSTRUMENT_UNC_UM = 1.0

# Part A5 parallel analysis
N_PERMUTATIONS = 1000
RANDOM_SEED = 20261108               # recorded in the summary file

# Part B: weighting functions (DISORT run behind draft Fig. 5) and the
# libRadtran sensitivity file written by compute_hysics_re_sensitivity.py.
WF_FILE = Path('/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/'
               'Hyperspectral_Cloud_Retrievals/HySICS/weighting_functions/'
               'disort_HySICS_weightingFunctions_platnick2000-profile_'
               'cloud-top-view_allBands_sim-ran-on-27-Jun-2025_rev1.mat')
# NB: the sibling '..._near-nadir-view_...' file (sza 10°, vza 0°, Cox–Munk
# ocean) is sun-glint contaminated: R(τ) DEcreases with τ for τ < 1 and its
# weighting functions are negative near cloud top. Do not use it.
SENS_GLOB = 'hysics_re_sensitivity_*.npz'
TAU_C_RT = 8.0                       # optical thickness of the WF / sensitivity runs
SIGMA_RAD = 0.003                    # HySICS radiometric uncertainty (fraction)
SIGMA_FM_LIST = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]   # forward-model unc.
SIGMA_FM_HEADLINE = 0.015            # the "ambitious target" used in draft §IV.A
FM_BAND_CORR = 0.0                   # inter-band correlation of forward-model error
N_PC_STATE = 40                      # PCs carried in the OE state (rest have ~0 variance)

HERE = Path(__file__).parent
FIG_DIR = HERE / 'figures'
OUT_DIR = HERE / 'outputs'
DATE = datetime.date.today().isoformat()
DPI = 300

# Okabe & Ito (2008) colour-blind-safe categorical palette, fixed order.
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PURPLE, C_SKY, C_GREY = (
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#7f7f7f')


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors load_profiles_consolidated() in the generator)
# ═══════════════════════════════════════════════════════════════════════════════
def project_to_common_grid(values_top_to_base: np.ndarray,
                           z_top_to_base: np.ndarray, L: int) -> np.ndarray:
    """Linear interpolation onto u = (z - z_base)/(z_top - z_base) ∈ [0, 1],
    L evenly spaced points, returned base → top (identical to the generator)."""
    z_top, z_base = float(z_top_to_base[0]), float(z_top_to_base[-1])
    if z_top == z_base:
        return np.full(L, np.mean(values_top_to_base))
    u_raw = (np.asarray(z_top_to_base) - z_base) / (z_top - z_base)
    order = np.argsort(u_raw)
    return np.interp(np.linspace(0.0, 1.0, L), u_raw[order],
                     np.asarray(values_top_to_base)[order])


def load_insitu() -> dict:
    """Return the τ-filtered, de-duplicated in-situ profiles on the common grid.

    Keys: re_um, lwc_gm3 (n, L) base → top; tau_c (n,); campaign (n,).
    """
    d = scipy.io.loadmat(MAT_FILE, squeeze_me=True)
    tau_all = np.atleast_1d(np.asarray(
        d['tau_c'] if TAU_C_SOURCE == 'stored' else d['tau_c_computed'], float))
    camp_all = np.atleast_1d(np.asarray(d['campaign_each'], dtype=object))

    re_rows, lwc_rows, taus, camps, seen = [], [], [], [], set()
    for i in range(len(tau_all)):
        if not (TAU_C_MIN <= tau_all[i] <= TAU_C_MAX):
            continue
        re = np.atleast_1d(np.asarray(d['re'][i], float))
        lwc = np.atleast_1d(np.asarray(d['lwc'][i], float))
        z = np.atleast_1d(np.asarray(d['z'][i], float))
        fp = tuple(np.round(re[:5], 4))          # duplicate-profile fingerprint
        if fp in seen:
            continue
        seen.add(fp)
        re_rows.append(project_to_common_grid(re, z, L_COMMON))
        lwc_rows.append(project_to_common_grid(lwc, z, L_COMMON))
        taus.append(float(tau_all[i]))
        camps.append(str(camp_all[i]))
    return dict(re_um=np.asarray(re_rows), lwc_gm3=np.asarray(lwc_rows),
                tau_c=np.asarray(taus), campaign=np.asarray(camps))


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — statistical criteria
# ═══════════════════════════════════════════════════════════════════════════════
def pca(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by SVD of the mean-removed data matrix Y (n_profiles, L).

    Returns mean (L,), eigenvalues λ (min(n,L),) of the sample covariance
    (divisor n-1), and eigenvectors as rows Vt (min(n,L), L).
    """
    mu = Y.mean(axis=0)
    _, S, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    lam = S ** 2 / (Y.shape[0] - 1)
    return mu, lam, Vt


def loo_reconstruction(Y: np.ndarray, k_max: int) -> np.ndarray:
    """Leave-one-profile-out reconstruction of every profile for k = 0..k_max.

    For profile i the mean and basis are fitted on the other n-1 profiles,
    and profile i is projected onto the first k eigenvectors:
        ŷ_i(k) = μ_{-i} + V_k V_kᵀ (y_i − μ_{-i}).
    Returns Yhat with shape (k_max + 1, n, L), in the same (log) space as Y.
    """
    n, L = Y.shape
    Yhat = np.empty((k_max + 1, n, L))
    for i in range(n):
        keep = np.arange(n) != i
        mu, _, Vt = pca(Y[keep])
        a = Y[i] - mu                                # (L,) anomaly
        scores = Vt[:k_max] @ a                      # (k_max,)
        Yhat[0, i] = mu
        # cumulative sum of rank-1 contributions gives every k at once
        contrib = scores[:, None] * Vt[:k_max]      # (k_max, L)
        Yhat[1:, i] = mu + np.cumsum(contrib, axis=0)
    return Yhat


def fve_curves(Y: np.ndarray, X_phys: np.ndarray, k_max: int,
               to_phys) -> dict:
    """Fraction of variance explained vs k: in-sample (log), LOO (log),
    LOO (physical units), plus LOO RMSE in physical units.

    FVE_LOO(k) = 1 − Σ_i ||y_i − ŷ_i(k)||² / Σ_i ||y_i − ȳ||², i.e. the PRESS
    form of R² (Jolliffe 2002, §6.1) with ȳ the grand per-level mean.
    """
    _, lam, _ = pca(Y)
    fve_in = np.concatenate([[0.0], np.cumsum(lam) / lam.sum()])[:k_max + 1]

    Yhat = loo_reconstruction(Y, k_max)
    ss_tot_log = np.sum((Y - Y.mean(axis=0)) ** 2)
    ss_tot_phys = np.sum((X_phys - X_phys.mean(axis=0)) ** 2)
    fve_loo_log = np.empty(k_max + 1)
    fve_loo_phys = np.empty(k_max + 1)
    rmse_phys = np.empty(k_max + 1)
    rmse_phys_p90 = np.empty(k_max + 1)
    for k in range(k_max + 1):
        fve_loo_log[k] = 1.0 - np.sum((Y - Yhat[k]) ** 2) / ss_tot_log
        Xhat = to_phys(Yhat[k])
        fve_loo_phys[k] = 1.0 - np.sum((X_phys - Xhat) ** 2) / ss_tot_phys
        per_profile = np.sqrt(np.mean((X_phys - Xhat) ** 2, axis=1))  # (n,)
        rmse_phys[k] = np.sqrt(np.mean((X_phys - Xhat) ** 2))
        rmse_phys_p90[k] = np.percentile(per_profile, 90)
    return dict(lam=lam, fve_in=fve_in, fve_loo_log=fve_loo_log,
                fve_loo_phys=fve_loo_phys, rmse_phys=rmse_phys,
                rmse_phys_p90=rmse_phys_p90)


def north_separation(lam: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """North et al. (1982) rule of thumb: δλ_k ≈ λ_k √(2/N).

    Returns δλ (len λ) and a boolean 'resolved' array: PC k is separated from
    PC k+1 if λ_k − λ_{k+1} > δλ_k.
    """
    dlam = lam * np.sqrt(2.0 / n)
    resolved = np.append(lam[:-1] - lam[1:] > dlam[:-1], False)
    return dlam, resolved


def broken_stick(p: int) -> np.ndarray:
    """Expected variance fraction of the k-th largest piece, k = 1..p
    (Jolliffe 2002, §6.1)."""
    return np.array([np.sum(1.0 / np.arange(k, p + 1)) / p
                     for k in range(1, p + 1)])


def parallel_analysis(Y: np.ndarray, n_perm: int,
                      rng: np.random.Generator) -> np.ndarray:
    """95th-percentile eigenvalues of level-wise permuted data (Horn 1965)."""
    n, L = Y.shape
    out = np.empty((n_perm, min(n, L)))
    for b in range(n_perm):
        Yp = np.column_stack([rng.permutation(Y[:, j]) for j in range(L)])
        out[b] = pca(Yp)[1]
    return np.percentile(out, 95, axis=0)


def gavish_donoho_k(Y: np.ndarray) -> int:
    """Number of singular values above the Gavish & Donoho (2014) optimal hard
    threshold for unknown noise, τ* = ω(β) · median(σ), with the polynomial
    approximation ω(β) ≈ 0.56β³ − 0.95β² + 1.82β + 1.43 (Gavish & Donoho 2014)."""
    n, L = Y.shape
    s = np.linalg.svd(Y - Y.mean(axis=0), compute_uv=False)
    beta = min(n, L) / max(n, L)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return int(np.sum(s > omega * np.median(s)))


def first_false(mask: np.ndarray) -> int:
    """Number of leading True entries (k retained by a 'keep while' rule)."""
    idx = np.flatnonzero(~mask)
    return int(idx[0]) if idx.size else int(mask.size)


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — retrievability (linear optimal-estimation information content)
# ═══════════════════════════════════════════════════════════════════════════════
def tau_from_top_on_u_grid(re_med_um: np.ndarray, lwc_med_gm3: np.ndarray,
                           tau_c: float) -> np.ndarray:
    """Optical depth measured from cloud top, τ(u), for the median in-situ
    profile on the normalised grid u ∈ [0, 1] (base → top), scaled so that
    τ(u = 0) = tau_c.

    dτ/dz ∝ LWC / r_e (geometric-optics extinction, Q_ext ≈ 2;
    Hansen & Travis 1974), so the SHAPE of τ(u) follows from the median profiles and the
    unknown constant is fixed by tau_c.
    """
    u = np.linspace(0.0, 1.0, re_med_um.size)
    dtau_du = lwc_med_gm3 / re_med_um
    # cumulative integral from the top (u = 1) downward
    seg = 0.5 * (dtau_du[1:] + dtau_du[:-1]) * np.diff(u)
    tau_top_down = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return tau_c * tau_top_down / tau_top_down[0]


def build_jacobian(re_med_um: np.ndarray, lwc_med_gm3: np.ndarray,
                   sens: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobian of HySICS reflectance w.r.t. ln r_e at each grid level and ln τ.

    Returns K_lnre (n_band, L), K_lntau (n_band,), R0 (n_band,), after dropping
    bands whose weighting function integrates to ≤ 0 (saturated gas bands).
    """
    wf = scipy.io.loadmat(WF_FILE, squeeze_me=True, struct_as_record=False)
    w = np.asarray(wf['w'], float)                    # (149 τ, 636 bands)
    tau_mid = np.asarray(wf['tau_midPoint'], float)   # (149,)
    tau_edges = np.asarray(wf['tau_2run'], float)     # (150,)
    dtau_wf = np.diff(tau_edges)                      # (149,)

    # Normalise each band's weighting function to unit integral over the cloud
    # (Platnick 2000 normalises w so that ∫ w dτ = 1).
    integ = np.sum(w * dtau_wf[:, None], axis=0)      # (636,)
    good = np.isfinite(integ) & (integ > 0)
    w_n = np.where(good, w / np.where(good, integ, 1.0), 0.0)

    # Evaluate w_λ at the τ of each u-level and multiply by the τ-thickness
    # that level represents, so Σ_j K_j = homogeneous-cloud sensitivity.
    tau_u = tau_from_top_on_u_grid(re_med_um, lwc_med_gm3, TAU_C_RT)  # (L,)
    # trapezoid quadrature weights in τ along the u-grid
    dtau_level = np.abs(np.gradient(tau_u))                            # (L,)
    W_u = np.empty((w.shape[1], tau_u.size))                           # (636, L)
    for b in range(w.shape[1]):
        W_u[b] = np.interp(tau_u, tau_mid, w_n[:, b]) * dtau_level
    # re-normalise rows so the discrete integral over levels is exactly one
    rs = W_u.sum(axis=1, keepdims=True)
    W_u = np.where(rs > 0, W_u / np.where(rs > 0, rs, 1.0), 0.0)

    # Platnick linearisation: R_λ depends on r_e*_λ = Σ_j W_λj r_e,j, hence
    # dR/d ln r_e,j = (dR/d r_e*) · W_λj · r_e,j = (dR/d ln r_e)_homog · W_λj · r_e,j / r_0
    r0 = float(sens['re0_um'])
    K_lnre = sens['dR_dlnre'][:, None] * W_u * (re_med_um[None, :] / r0)
    K_lntau = sens['dR_dlntau']
    R0 = sens['R0']
    keep = good & (R0 > 0)
    return K_lnre[keep], K_lntau[keep], R0[keep]


def oe_information(K_lnre: np.ndarray, K_lntau: np.ndarray, R0: np.ndarray,
                   Vt: np.ndarray, lam: np.ndarray, cov_pc_lntau: np.ndarray,
                   var_lntau: float, sigma_fm: float,
                   band_corr: float) -> dict:
    """Linear OE information content in the PC basis (Rodgers 2000, ch. 2).

    State x = [PC scores s_1..s_M of ln r_e,  ln τ]. Prior covariance S_a:
    diag(λ_1..λ_M) for the scores, var(ln τ) and the in-situ cross-covariance
    cov(s_k, ln τ). Jacobian in this basis: K_x = [K_lnre V_Mᵀ, K_lntau].

    Returns the averaging-kernel diagonal per PC, DFS of the r_e block,
    and per-PC prior-normalised signal-to-noise
        SNR_k = √λ_k · ‖S_ε^{-1/2} K_lnre v_k‖   (τ held fixed).
    """
    M = Vt.shape[0]
    Kx = np.column_stack([K_lnre @ Vt.T, K_lntau])            # (n_band, M+1)
    Sa = np.zeros((M + 1, M + 1))
    Sa[:M, :M] = np.diag(lam)
    Sa[:M, M] = Sa[M, :M] = cov_pc_lntau
    Sa[M, M] = var_lntau

    # Noise: radiometric (independent) + forward model (correlation band_corr).
    nb = R0.size
    sig_rad = SIGMA_RAD * R0
    sig_fm = sigma_fm * R0
    if band_corr == 0.0:
        Se_inv_diag = 1.0 / (sig_rad ** 2 + sig_fm ** 2)
        KtSiK = Kx.T @ (Kx * Se_inv_diag[:, None])
        Se_isqrt_K = Kx * np.sqrt(Se_inv_diag)[:, None]
    else:
        C = (1 - band_corr) * np.eye(nb) + band_corr * np.ones((nb, nb))
        Se = np.diag(sig_rad ** 2) + np.outer(sig_fm, sig_fm) * C
        Lc = np.linalg.cholesky(Se)
        Se_isqrt_K = np.linalg.solve(Lc, Kx)
        KtSiK = Se_isqrt_K.T @ Se_isqrt_K

    # Posterior covariance and averaging kernel (Rodgers 2000, ch. 2)
    Sa_inv = np.linalg.inv(Sa)
    S_hat = np.linalg.inv(KtSiK + Sa_inv)
    A = S_hat @ KtSiK                     # = I − S_hat S_a⁻¹
    a_diag = np.diag(A)
    snr = np.sqrt(lam) * np.linalg.norm(Se_isqrt_K[:, :M], axis=0)
    return dict(A_diag_pc=a_diag[:M], A_tau=a_diag[M],
                dfs_re=float(np.sum(a_diag[:M])), dfs_total=float(np.trace(A)),
                snr=snr,
                var_reduction=1.0 - np.diag(S_hat)[:M] / lam)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})

    data = load_insitu()
    n, L = data['re_um'].shape
    print(f'In-situ profiles kept: {n}  (L = {L}); '
          f'ORACLES = {(data["campaign"] == "oracles").sum()}, '
          f'VOCALS-REx = {(data["campaign"] == "vocals-rex").sum()}')

    # Log transform exactly as in the generator.
    Y = {'re': np.log(np.maximum(data['re_um'], LWC_EPS)),
         'lwc': np.log(np.maximum(data['lwc_gm3'], LWC_EPS))}
    X = {'re': data['re_um'], 'lwc': data['lwc_gm3']}
    units = {'re': 'μm', 'lwc': 'g m⁻³'}

    # The generator's floor makes exp(log(max(x, eps))) ≠ x where x < eps; use
    # the floored physical values as "truth" so the comparison is like-for-like.
    X_floor = {v: np.exp(Y[v]) for v in Y}

    res = {}
    for v in ('re', 'lwc'):
        r = fve_curves(Y[v], X_floor[v], K_MAX_REPORT, np.exp)
        r['dlam'], r['north_resolved'] = north_separation(r['lam'], n)
        r['bstick'] = broken_stick(r['lam'].size)
        r['pa95'] = parallel_analysis(Y[v], N_PERMUTATIONS, rng)
        frac = r['lam'] / r['lam'].sum()
        r['k_bstick'] = first_false(frac > r['bstick'])
        r['k_pa'] = first_false(r['lam'] > r['pa95'])
        r['k_gd'] = gavish_donoho_k(Y[v])
        r['k_north'] = first_false(r['north_resolved'])
        r['k_fve90'] = int(np.searchsorted(r['fve_in'], 0.90))
        r['k_fve95'] = int(np.searchsorted(r['fve_in'], 0.95))
        r['k_fve99'] = int(np.searchsorted(r['fve_in'], 0.99))
        res[v] = r

    # ── Part B ──────────────────────────────────────────────────────────────
    sens_files = sorted(OUT_DIR.glob(SENS_GLOB))
    partB = None
    if not sens_files:
        print('\n[Part B skipped] run compute_hysics_re_sensitivity.py first.')
    else:
        sens = dict(np.load(sens_files[-1]))
        re_med = np.median(data['re_um'], axis=0)
        lwc_med = np.median(data['lwc_gm3'], axis=0)
        K_lnre, K_lntau, R0 = build_jacobian(re_med, lwc_med, sens)

        mu, lam, Vt = pca(Y['re'])
        M = min(N_PC_STATE, int(np.sum(lam > 1e-12 * lam[0])))
        scores = (Y['re'] - mu) @ Vt[:M].T                    # (n, M)
        ln_tau = np.log(data['tau_c'])
        cov_pc_lntau = np.array([np.cov(scores[:, k], ln_tau)[0, 1]
                                 for k in range(M)])
        partB = {}
        for sfm in SIGMA_FM_LIST:
            partB[sfm] = oe_information(K_lnre, K_lntau, R0, Vt[:M], lam[:M],
                                        cov_pc_lntau, float(np.var(ln_tau, ddof=1)),
                                        sfm, FM_BAND_CORR)
        partB_meta = dict(n_band=R0.size, sens_file=sens_files[-1].name, M=M)

    # ── Report ──────────────────────────────────────────────────────────────
    lines = []
    P = lines.append
    P(f'PC truncation analysis — {DATE}')
    P(f'In-situ file: {MAT_FILE.name}')
    P(f'Filters: {TAU_C_MIN} ≤ τ_c ≤ {TAU_C_MAX} ({TAU_C_SOURCE}); N = {n} '
      f'profiles; L = {L}; random seed (parallel analysis) = {RANDOM_SEED}')
    P('')
    for v, name in (('re', 'log r_e'), ('lwc', 'log LWC')):
        r = res[v]
        P(f'=== {name} ===')
        P(f' k | var frac | cum (in-sample, log) | cum LOO (log) | cum LOO (phys) '
          f'| LOO RMSE [{units[v]}] | p90 per-profile RMSE | North-separated from k+1')
        for k in range(1, K_MAX_REPORT + 1):
            P(f'{k:2d} | {r["lam"][k-1] / r["lam"].sum():8.4f} | '
              f'{r["fve_in"][k]:20.4f} | {r["fve_loo_log"][k]:13.4f} | '
              f'{r["fve_loo_phys"][k]:14.4f} | {r["rmse_phys"][k]:16.4f} | '
              f'{r["rmse_phys_p90"][k]:20.4f} | {"yes" if r["north_resolved"][k-1] else "NO"}')
        P(f' k=0 (mean profile only): LOO RMSE = {r["rmse_phys"][0]:.4f} {units[v]}')
        P(f' Rules → broken stick: {r["k_bstick"]}; parallel analysis (95%): '
          f'{r["k_pa"]}; Gavish–Donoho: {r["k_gd"]}; North (leading separated '
          f'PCs): {r["k_north"]}; FVE ≥ 90/95/99 %: {r["k_fve90"]}/'
          f'{r["k_fve95"]}/{r["k_fve99"]}')
        P('')

    if partB is not None:
        P('=== Part B — retrievability of log r_e PCs (linear OE, HySICS) ===')
        P(f'Bands used: {partB_meta["n_band"]}; sensitivity file: '
          f'{partB_meta["sens_file"]}; state = {partB_meta["M"]} PCs + ln τ; '
          f'σ_rad = {SIGMA_RAD:.3%}; band corr of FM error = {FM_BAND_CORR}')
        hdr = ' σ_fm  | DFS(r_e) | ' + ' '.join(f'A{k:<4d}' for k in range(1, 9)) + \
            '| #PC with A_kk ≥ 0.5'
        P(hdr)
        for sfm, b in partB.items():
            P(f'{sfm:6.3f} | {b["dfs_re"]:8.2f} | ' +
              ' '.join(f'{a:5.2f}' for a in b['A_diag_pc'][:8]) +
              f'| {int(np.sum(b["A_diag_pc"] >= 0.5))}')
        b = partB[SIGMA_FM_HEADLINE]
        P(f'Prior-normalised SNR per PC at σ_fm = {SIGMA_FM_HEADLINE:.1%} '
          f'(τ fixed): ' + ', '.join(f'PC{k+1}={s:.2f}' for k, s in
                                     enumerate(b['snr'][:8])))
        P('')
    P('Caveats: Part B is linear, single-scene (τ = 8, sza ≈ 49°, vza ≈ 32°), treats '
      'water vapour / cloud height / α / aerosol / surface as known, and assumes '
      'band-independent noise — an UPPER bound on retrievable modes.')

    summary = '\n'.join(lines)
    print('\n' + summary)
    (OUT_DIR / f'pc_truncation_summary_{DATE}.txt').write_text(summary)

    # CSV table
    with open(OUT_DIR / f'pc_truncation_table_{DATE}.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        head = ['variable', 'k', 'var_frac', 'fve_insample_log', 'fve_loo_log',
                'fve_loo_phys', 'loo_rmse_phys', 'loo_rmse_p90_phys',
                'north_separated_from_next']
        if partB is not None:
            head += [f'A_kk_sigmafm_{s:g}' for s in SIGMA_FM_LIST]
        wr.writerow(head)
        for v in ('re', 'lwc'):
            r = res[v]
            for k in range(1, K_MAX_REPORT + 1):
                row = [v, k, r['lam'][k-1] / r['lam'].sum(), r['fve_in'][k],
                       r['fve_loo_log'][k], r['fve_loo_phys'][k],
                       r['rmse_phys'][k], r['rmse_phys_p90'][k],
                       bool(r['north_resolved'][k-1])]
                if partB is not None:
                    row += [partB[s]['A_diag_pc'][k-1] if v == 're' else ''
                            for s in SIGMA_FM_LIST]
                wr.writerow(row)

    # ── Figures ─────────────────────────────────────────────────────────────
    kk = np.arange(1, K_MAX_REPORT + 1)
    titles = {'re': 'log $r_e(z)$', 'lwc': 'log LWC$(z)$'}

    # Fig 1: scree with North error bars and thresholds
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, v in zip(axs, ('re', 'lwc')):
        r = res[v]
        frac = r['lam'] / r['lam'].sum()
        ax.errorbar(kk, frac[:K_MAX_REPORT],
                    yerr=(r['dlam'] / r['lam'].sum())[:K_MAX_REPORT],
                    fmt='o', ms=4, color=C_BLUE, capsize=2,
                    label='eigenvalue ± North et al. (1982) δλ')
        ax.plot(kk, r['bstick'][:K_MAX_REPORT], '-', color=C_ORANGE, lw=1.5,
                label='broken stick')
        ax.plot(kk, (r['pa95'] / r['lam'].sum())[:K_MAX_REPORT], '--',
                color=C_GREEN, lw=1.5, label='parallel analysis (95th pct)')
        ax.set_yscale('log')
        ax.set_xlabel('PC index k')
        ax.set_ylabel('fraction of variance')
        ax.set_title(titles[v])
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / f'fig1_scree_{DATE}.png', dpi=DPI)
    plt.close(fig)

    # Fig 2: variance retained
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                            sharey=True)
    k0 = np.arange(0, K_MAX_REPORT + 1)
    for ax, v, kc in zip(axs, ('re', 'lwc'), (K_CURRENT_RE, K_CURRENT_LWC)):
        r = res[v]
        # start at k = 1 (k = 0, the mean profile alone, explains 0 % by definition)
        ax.plot(kk, 100 * r['fve_in'][1:], '-o', ms=3, color=C_BLUE, lw=1.5,
                label='in-sample, log space')
        ax.plot(kk, 100 * r['fve_loo_log'][1:], '-s', ms=3, color=C_ORANGE, lw=1.5,
                label='leave-one-out, log space')
        ax.plot(kk, 100 * r['fve_loo_phys'][1:], '-^', ms=3, color=C_GREEN, lw=1.5,
                label='leave-one-out, physical units')
        ax.axvline(kc, color=C_GREY, lw=1, ls=':')
        ax.text(kc + 0.2, 99, f'draft 23: k = {kc}', color=C_GREY, fontsize=8,
                va='top')
        ax.set_xlabel('number of PCs retained, k')
        ax.set_title(titles[v])
        ax.set_ylim(55, 100.5)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].set_ylabel('variance explained (%)')
    axs[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.savefig(FIG_DIR / f'fig2_variance_retained_{DATE}.png', dpi=DPI)
    plt.close(fig)

    # Fig 3: LOO RMSE in physical units
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, v, kc in zip(axs, ('re', 'lwc'), (K_CURRENT_RE, K_CURRENT_LWC)):
        r = res[v]
        ax.plot(k0, r['rmse_phys'], '-o', ms=3, color=C_BLUE, lw=1.5,
                label='RMSE, all profiles')
        ax.plot(k0, r['rmse_phys_p90'], '-s', ms=3, color=C_ORANGE, lw=1.5,
                label='90th pct of per-profile RMSE')
        if v == 're':
            ax.axhline(RE_INSTRUMENT_UNC_UM, color=C_VERM, lw=1, ls='--',
                       label='in-situ $r_e$ uncertainty (~1 μm)')
        ax.axvline(kc, color=C_GREY, lw=1, ls=':')
        ax.set_xlabel('number of PCs retained, k')
        ax.set_ylabel(f'LOO reconstruction RMSE ({units[v]})')
        ax.set_title(titles[v])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / f'fig3_loo_rmse_{DATE}.png', dpi=DPI)
    plt.close(fig)

    # Fig 4: averaging-kernel diagonal per PC for several σ_fm
    if partB is not None:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        seq = plt.get_cmap('Blues')(np.linspace(0.35, 1.0, len(SIGMA_FM_LIST)))
        for c, sfm in zip(seq, SIGMA_FM_LIST):
            axs[0].plot(kk[:10], partB[sfm]['A_diag_pc'][:10], '-o', ms=3,
                        color=c, lw=1.5, label=f'σ_fm = {sfm:.1%}')
        axs[0].axhline(0.5, color=C_GREY, lw=1, ls=':')
        axs[0].set_xlabel('PC index k (log $r_e$)')
        axs[0].set_ylabel('averaging-kernel diagonal $A_{kk}$')
        axs[0].set_title('fraction of each mode set by the measurement')
        axs[0].set_ylim(0, 1.02)
        axs[0].legend(frameon=False, fontsize=7, ncol=2)
        axs[0].grid(alpha=0.25, lw=0.5)
        sf = np.array(SIGMA_FM_LIST)
        axs[1].plot(100 * sf, [partB[s]['dfs_re'] for s in SIGMA_FM_LIST],
                    '-o', ms=4, color=C_BLUE, lw=1.5,
                    label='DFS (r_e block)')
        axs[1].plot(100 * sf, [np.sum(partB[s]['A_diag_pc'] >= 0.5)
                               for s in SIGMA_FM_LIST],
                    '-s', ms=4, color=C_ORANGE, lw=1.5,
                    label='# PCs with $A_{kk}$ ≥ 0.5')
        axs[1].set_xlabel('forward-model uncertainty σ_fm (%)')
        axs[1].set_ylabel('number of modes')
        axs[1].set_title('retrievable profile modes (σ_rad = 0.3 %)')
        axs[1].legend(frameon=False, fontsize=8)
        axs[1].grid(alpha=0.25, lw=0.5)
        fig.savefig(FIG_DIR / f'fig4_averaging_kernel_{DATE}.png', dpi=DPI)
        plt.close(fig)

    print(f'\nFigures → {FIG_DIR}\nTables  → {OUT_DIR}')


if __name__ == '__main__':
    main()
