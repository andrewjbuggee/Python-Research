"""
Synthetic in-situ profile generator (joint r_e + LWC + alpha_param + geometry).

Fits a functional-PCA + multivariate-normal generative model in LOG-SPACE
to a set of in-situ profiles from the ORACLES campaign, then samples N
new synthetic profiles. Each synthetic sample carries:

    re        : (N_FIXED_LEVELS,) effective radius profile  (μm, top → base)
    lwc       : (N_FIXED_LEVELS,) liquid water content        (g/m³,  top → base)
    z         : (N_FIXED_LEVELS,) altitude grid              (km,   top → base)
    alpha     : scalar mean libRadtran shape parameter
    z_top     : scalar cloud-top altitude     (km)
    z_base    : scalar cloud-base altitude    (km)
    thickness : scalar cloud geometric depth  (km)
    tau_c     : scalar cloud optical depth    (derived from re + LWC)
    LWP       : scalar liquid water path      (g/m², derived from LWC)

Method
======
1. Load every .mat file in MAT_DIR. Keep only files that
     (a) contain 'alpha_param'  → this filters out VOCALS-REx, since only
         ORACLES profiles store alpha_param,
     (b) have tau_c ≥ TAU_C_MIN   (default 3, per request),
     (c) are not duplicate profile fingerprints.
2. Project raw re and lwc onto a common normalized-altitude grid of length
   L_COMMON (≥ 60), then take logs:
       Y_re  = log(re_common)              (always positive, no offset)
       Y_lwc = log(lwc_common + LWC_EPS)   (small floor for near-base zeros)
   Per-profile mean alpha is computed by trapezoid-weighted average over
   the raw altitude axis and stored as log(mean_alpha) for the joint MVN.
3. PCA via SVD on Y_re and Y_lwc separately. Keep K_RE / K_LWC modes
   chosen so cumulative variance ≥ VAR_TARGET (default 0.99).
4. Single joint multivariate normal over the (K_RE + K_LWC + 3)-dim feature
       [re_scores; lwc_scores; log(mean_alpha); z_base; log(thickness)]
   The cross-covariance terms encode the physical coupling between r_e
   and LWC shape, between cloud thickness and shape, etc.
5. Draw N samples from this MVN. Inverse-PCA gives Y_re and Y_lwc on the
   common grid; exp() recovers re and lwc as guaranteed-positive log-normal
   variates at each level. Resample to N_FIXED_LEVELS evenly-spaced
   altitudes from sampled (z_top, z_base). alpha is broadcast as a scalar.
6. Derive per-sample LWP and tau_c from the re + lwc profile:
       LWP[g/m²] = 1000 · ∫ LWC dz   (LWC g/m³, dz km)
       tau_c     = 1500 · ∫ LWC / r_e dz   (LWC g/m³, r_e μm, dz km)
   These are physically self-consistent with the sampled re + lwc; they
   are NOT independently sampled.
7. Save to .npz; write a 3 x 3 diagnostic figure at DPI 500.

Profile orientation throughout: index 0 = cloud top, index -1 = cloud base.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
MAT_DIR          = Path('/Volumes/My Passport/neural_network_training_data/saz0_allProfiles/')

TAU_C_MIN        = 3.0          # only use clouds with cloud optical depth ≥ this
L_COMMON         = 80           # length of the common normalized-altitude grid (≥ 60)
N_FIXED_LEVELS   = 7            # every synthetic profile is on this fixed grid
N_SAMPLES        = 300000

VAR_TARGET       = 0.99         # cumulative variance target for picking K_RE, K_LWC, K_T, K_VAPOR
K_RE_MAX         = 6           # hard cap on number of re modes kept
K_LWC_MAX        = 8           # hard cap on number of lwc modes kept
K_T_MAX          = 8            # hard cap on number of ERA5 temperature modes kept
K_VAPOR_MAX      = 6            # hard cap on number of ERA5 vapor-concentration modes kept

LWC_EPS          = 1e-3         # g/m³ floor for log(lwc + eps); below typical in-cloud LWC
VAPOR_EPS        = 1.0          # molec/cm³ floor for log(vapor + eps)
RANDOM_SEED      = 0

OUT_DIR          = Path(__file__).parent / 'synthetic_profiles'
OUT_PATH         = OUT_DIR / f'synthetic_profiles_jointMVN_N{N_SAMPLES}_L{N_FIXED_LEVELS}.npz'
FIG_PATH         = OUT_DIR / f'synthetic_profiles_jointMVN_diagnostic_N{N_SAMPLES}.png'
FIG_EXAMPLES_PATH = OUT_DIR / f'synthetic_profiles_jointMVN_examples_N{N_SAMPLES}.png'
FIG_ATMOS_PATH    = OUT_DIR / f'synthetic_profiles_jointMVN_atmos_diagnostic_N{N_SAMPLES}.png'
FIGURE_DPI       = 500

# Rejection-sampling loop: keeps drawing batches until N samples have
# tau_c >= TAU_C_MIN (matching the in-situ filter). Without this, the
# tail of the joint MVN produces thinner-than-trained clouds.
REJECTION_BATCH_OVERSAMPLE = 1.6      # draw 1.6× the remaining need each batch
REJECTION_MAX_BATCHES      = 30


# ── Paper-quality matplotlib style (mirrors regenerate_kfold_figures.setup_style)
def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'figure.titlesize':  14,
        'legend.fontsize':   8,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.linewidth':    0.8,
    })


setup_style()


# ── Helpers ────────────────────────────────────────────────────────────────────
def project_to_common_grid(values_top_to_base, z_top_to_base, L):
    """Resample (values, z) (top→base) onto an ascending u-grid of length L,
    where u=0 at cloud base, u=1 at cloud top. Returned array is base→top."""
    z_top, z_base = float(z_top_to_base[0]), float(z_top_to_base[-1])
    if z_top == z_base:
        return np.full(L, np.mean(values_top_to_base))
    u_raw = (np.asarray(z_top_to_base) - z_base) / (z_top - z_base)
    order = np.argsort(u_raw)
    u_sorted = u_raw[order]
    v_sorted = np.asarray(values_top_to_base)[order]
    u_target = np.linspace(0.0, 1.0, L)
    return np.interp(u_target, u_sorted, v_sorted)


def trapz_vertical_mean(values_top_to_base, z_top_to_base):
    """Trapezoid-weighted vertical average over the (descending) altitude axis."""
    v = np.asarray(values_top_to_base)
    z = np.asarray(z_top_to_base)
    integral  = abs(np.trapezoid(v, z))
    thickness = abs(z[0] - z[-1])
    return integral / thickness if thickness > 0 else float(np.mean(v))


def compute_lwp(lwc_top_to_base, z_top_to_base):
    """LWP in g/m². LWC in g/m³, z in km → factor of 1000 for km→m."""
    return 1000.0 * abs(np.trapezoid(lwc_top_to_base, z_top_to_base))


def compute_tau_c(re_top_to_base, lwc_top_to_base, z_top_to_base):
    """Layer-integrated cloud optical depth from re (μm), LWC (g/m³), z (km).

    dτ = (3/2) · LWC[kg/m³] · dz[m] / (ρ_w[kg/m³] · r_e[m])
       = 1500 · LWC[g/m³] · dz[km] / r_e[μm]      (after unit conversions)
    """
    integrand = lwc_top_to_base / np.maximum(re_top_to_base, 1e-6)
    return 1500.0 * abs(np.trapezoid(integrand, z_top_to_base))


def load_era5_fields(d):
    """Pull (T, vapor_concentration, pressure_levels) from a loaded ORACLES .mat.

    Access path comes from convert_matFiles_to_HDF._era5_datprofiles:
        d['era5'].item()[6].item() → tuple-of-arrays with fields
            0: GP_height (m)        1: T (K)        2: pressure (hPa)
            3: q (kg/kg)            4: vapor_conc (molec/cm³)
            5: vapor_massDensity (kg/m³)
    All length 37 on the standard ERA5 pressure grid (surface → TOA).
    """
    if 'era5' not in d:
        return None
    try:
        dp = d['era5'].item()[6].item()
        T   = np.asarray(dp[1], dtype=np.float64)
        P   = np.asarray(dp[2], dtype=np.float64)
        vap = np.asarray(dp[4], dtype=np.float64)
    except (IndexError, AttributeError, ValueError):
        return None
    if not (len(T) == len(P) == len(vap)):
        return None
    return T, vap, P


# ── Load ORACLES profiles with tau_c ≥ TAU_C_MIN ───────────────────────────────
mat_files = sorted(f for f in MAT_DIR.glob('*.mat') if not f.name.startswith('._'))
print(f'Found {len(mat_files)} .mat files in {MAT_DIR}')

profiles_re   = []   # raw arrays (top → base)
profiles_lwc  = []
altitudes_raw = []
mean_alpha    = []
z_top_each    = []
z_base_each   = []
tau_c_each    = []
lwp_each      = []
n_levels_each = []
T_each        = []   # (37,) ERA5 temperature, surface → TOA
vapor_each    = []   # (37,) ERA5 vapor_concentration, surface → TOA
era5_pressure = None

seen_fingerprints = set()
n_skip_no_alpha = n_skip_low_tau = n_skip_dup = n_skip_missing = 0
n_skip_no_era5  = 0

for path in mat_files:
    d = scipy.io.loadmat(path, squeeze_me=True)

    if not {'re', 'z', 'tau', 'lwc'}.issubset(d.keys()):
        n_skip_missing += 1
        continue
    if 'alpha_param' not in d.keys():
        n_skip_no_alpha += 1
        continue

    era5_fields = load_era5_fields(d)
    if era5_fields is None:
        n_skip_no_era5 += 1
        continue
    T_prof, vapor_prof, P_grid = era5_fields
    if era5_pressure is None:
        era5_pressure = P_grid                               # snapshot once

    re    = np.asarray(d['re'][()],          dtype=np.float64)
    lwc   = np.asarray(d['lwc'][()],         dtype=np.float64)
    z     = np.asarray(d['z'][()],           dtype=np.float64)
    alpha = np.asarray(d['alpha_param'][()], dtype=np.float64)
    tau_c = float(np.asarray(d['tau'][()]).max())

    if tau_c < TAU_C_MIN:
        n_skip_low_tau += 1
        continue

    fp = tuple(np.round(re[:5], 4))
    if fp in seen_fingerprints:
        n_skip_dup += 1
        continue
    seen_fingerprints.add(fp)

    # Defensive: trim to common length if any of re / lwc / alpha drift in size
    n_use = min(len(re), len(lwc), len(z), len(alpha))
    re, lwc, z, alpha = re[:n_use], lwc[:n_use], z[:n_use], alpha[:n_use]

    profiles_re.append(re)
    profiles_lwc.append(lwc)
    altitudes_raw.append(z)
    mean_alpha.append(trapz_vertical_mean(alpha, z))
    z_top_each.append(float(z[0]))
    z_base_each.append(float(z[-1]))
    tau_c_each.append(tau_c)
    lwp_each.append(compute_lwp(lwc, z))
    n_levels_each.append(n_use)
    T_each.append(T_prof)
    vapor_each.append(vapor_prof)

n_train         = len(profiles_re)
mean_alpha      = np.asarray(mean_alpha,   dtype=np.float64)
z_top_each      = np.asarray(z_top_each,   dtype=np.float64)
z_base_each     = np.asarray(z_base_each,  dtype=np.float64)
tau_c_each      = np.asarray(tau_c_each,   dtype=np.float64)
lwp_each        = np.asarray(lwp_each,     dtype=np.float64)
n_levels_each   = np.asarray(n_levels_each, dtype=int)
thickness_each  = z_top_each - z_base_each
T_arr           = np.asarray(T_each,     dtype=np.float64)   # (n_train, 37)
vapor_arr       = np.asarray(vapor_each, dtype=np.float64)   # (n_train, 37)
N_ERA5          = T_arr.shape[1]

print(f'  skipped: {n_skip_no_alpha} no alpha_param (likely VOCALS-REx), '
      f'{n_skip_low_tau} tau_c<{TAU_C_MIN}, '
      f'{n_skip_dup} duplicates, {n_skip_missing} missing keys, '
      f'{n_skip_no_era5} no era5')
print(f'  kept   : {n_train} unique ORACLES profiles')
print(f'  n_lev   : min={n_levels_each.min()}, max={n_levels_each.max()}, '
      f'mean={n_levels_each.mean():.1f}')
print(f'  tau_c   : [{tau_c_each.min():.2f}, {tau_c_each.max():.2f}], '
      f'median={np.median(tau_c_each):.2f}')
print(f'  thick km: [{thickness_each.min():.3f}, {thickness_each.max():.3f}]')
print(f'  alpha   : [{mean_alpha.min():.2f}, {mean_alpha.max():.2f}], '
      f'median={np.median(mean_alpha):.2f}')
print(f'  ERA5    : {N_ERA5} pressure levels '
      f'({era5_pressure[0]:.0f}–{era5_pressure[-1]:.0f} hPa), '
      f'T [{T_arr.min():.1f}, {T_arr.max():.1f}] K, '
      f'vapor [{vapor_arr.min():.2e}, {vapor_arr.max():.2e}] molec/cm³')

# Hard physical clip bounds (used after every reconstruction).
re_min_raw    = float(min(p.min() for p in profiles_re))
re_max_raw    = float(max(p.max() for p in profiles_re))
T_min, T_max  = float(T_arr.min()),     float(T_arr.max())
v_min, v_max  = float(vapor_arr.min()), float(vapor_arr.max())


# ── Project onto common grid and take logs ─────────────────────────────────────
re_common  = np.empty((n_train, L_COMMON))   # base → top
lwc_common = np.empty((n_train, L_COMMON))

for i in range(n_train):
    re_common[i]  = project_to_common_grid(profiles_re[i],  altitudes_raw[i], L_COMMON)
    lwc_common[i] = project_to_common_grid(profiles_lwc[i], altitudes_raw[i], L_COMMON)

Y_re  = np.log(np.maximum(re_common,  LWC_EPS))           # re always >> 0
Y_lwc = np.log(np.maximum(lwc_common, LWC_EPS))           # add floor for log

# Quick sanity check on the interpolation error vs. raw at L_COMMON
re_lin_rmse = []
for i in range(n_train):
    re_back = np.interp(altitudes_raw[i],
                        np.linspace(z_top_each[i], z_base_each[i], L_COMMON)[::-1],
                        re_common[i])
    re_lin_rmse.append(np.sqrt(np.mean((re_back - profiles_re[i]) ** 2)))
re_lin_rmse = np.asarray(re_lin_rmse)
print(f'\nCommon-grid interp RMSE (re raw vs round-trip): '
      f'mean={re_lin_rmse.mean():.4f} μm, max={re_lin_rmse.max():.4f} μm, '
      f'(L_COMMON={L_COMMON})')


# ── PCA on log-re and log-lwc ──────────────────────────────────────────────────
def fit_pca(Y, k_max, var_target, name):
    mu  = Y.mean(axis=0)
    Yc  = Y - mu
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    var_ratio = (S ** 2) / np.sum(S ** 2)
    cum = np.cumsum(var_ratio)
    K = int(np.searchsorted(cum, var_target) + 1)
    K = max(1, min(K, k_max, len(S)))
    print(f'  {name}: K={K} (cum var={cum[K - 1]:.4f}); '
          f'variance per mode = '
          f'{", ".join(f"{v:.3f}" for v in var_ratio[:K])}')
    scores = (U * S)[:, :K]
    components = Vt[:K, :]
    return mu, components, scores, var_ratio

Y_T     = np.log(np.maximum(T_arr,     1e-3))
Y_vapor = np.log(np.maximum(vapor_arr, VAPOR_EPS))

print(f'\nPCA fits:')
mu_re,    comps_re,    scores_re,    var_re    = fit_pca(Y_re,    K_RE_MAX,    VAR_TARGET, 'log re   ')
mu_lwc,   comps_lwc,   scores_lwc,   var_lwc   = fit_pca(Y_lwc,   K_LWC_MAX,   VAR_TARGET, 'log lwc  ')
mu_T,     comps_T,     scores_T,     var_T     = fit_pca(Y_T,     K_T_MAX,     VAR_TARGET, 'log T    ')
mu_vapor, comps_vapor, scores_vapor, var_vapor = fit_pca(Y_vapor, K_VAPOR_MAX, VAR_TARGET, 'log vapor')
K_re,  K_lwc            = scores_re.shape[1],   scores_lwc.shape[1]
K_T,   K_vapor          = scores_T.shape[1],    scores_vapor.shape[1]


# ── Joint MVN ──────────────────────────────────────────────────────────────────
log_mean_alpha = np.log(np.maximum(mean_alpha, 1e-3))
log_thickness  = np.log(thickness_each)

features = np.hstack([
    scores_re,
    scores_lwc,
    scores_T,
    scores_vapor,
    log_mean_alpha[:, None],
    z_base_each[:,    None],
    log_thickness[:,  None],
])
mu_f  = features.mean(axis=0)
cov_f = np.cov(features, rowvar=False)
print(f'\nJoint MVN feature dimension: {features.shape[1]} '
      f'(K_re={K_re} + K_lwc={K_lwc} + K_T={K_T} + K_vapor={K_vapor} '
      f'+ log(α) + z_base + log(thickness))')


# ── Sample N synthetic profiles (with rejection on tau_c) ─────────────────────
slc_re    = slice(0,                                 K_re)
slc_lwc   = slice(K_re,                              K_re + K_lwc)
slc_T     = slice(K_re + K_lwc,                      K_re + K_lwc + K_T)
slc_vapor = slice(K_re + K_lwc + K_T,                K_re + K_lwc + K_T + K_vapor)
idx_alpha = K_re + K_lwc + K_T + K_vapor
idx_zbase = idx_alpha + 1
idx_thick = idx_alpha + 2
u_grid_b2t = np.linspace(0.0, 1.0, L_COMMON)


def reconstruct_batch(samples_batch):
    """Map a batch of MVN draws → physical synthetic profiles + derived bulk values.

    Returns a dict whose arrays all share the leading dimension n_batch.
    """
    n = samples_batch.shape[0]

    new_scores_re    = samples_batch[:, slc_re]
    new_scores_lwc   = samples_batch[:, slc_lwc]
    new_scores_T     = samples_batch[:, slc_T]
    new_scores_vapor = samples_batch[:, slc_vapor]
    new_log_alpha    = samples_batch[:, idx_alpha]
    new_z_base       = samples_batch[:, idx_zbase]
    new_log_thick    = samples_batch[:, idx_thick]

    new_z_base    = np.clip(new_z_base,
                            z_base_each.min(), z_base_each.max())
    new_thickness = np.clip(np.exp(new_log_thick),
                            thickness_each.min(), thickness_each.max())
    new_z_top     = new_z_base + new_thickness
    new_alpha     = np.clip(np.exp(new_log_alpha),
                            mean_alpha.min(), mean_alpha.max())

    # Inverse PCA in log-space, then exp() back to physical units.
    log_re_recon    = mu_re    + new_scores_re    @ comps_re
    log_lwc_recon   = mu_lwc   + new_scores_lwc   @ comps_lwc
    log_T_recon     = mu_T     + new_scores_T     @ comps_T
    log_vapor_recon = mu_vapor + new_scores_vapor @ comps_vapor
    re_recon    = np.exp(log_re_recon)
    lwc_recon   = np.exp(log_lwc_recon) - LWC_EPS
    T_recon     = np.exp(log_T_recon)
    vapor_recon = np.exp(log_vapor_recon) - VAPOR_EPS
    np.clip(lwc_recon,   0.0, None, out=lwc_recon)
    np.clip(vapor_recon, 0.0, None, out=vapor_recon)

    re_out    = np.empty((n, N_FIXED_LEVELS), dtype=np.float32)
    lwc_out   = np.empty((n, N_FIXED_LEVELS), dtype=np.float32)
    z_out     = np.empty((n, N_FIXED_LEVELS), dtype=np.float32)
    alpha_out = np.empty((n, N_FIXED_LEVELS), dtype=np.float32)
    for i in range(n):
        z_i = np.linspace(new_z_top[i], new_z_base[i], N_FIXED_LEVELS)
        u_i = (z_i - new_z_base[i]) / (new_z_top[i] - new_z_base[i])
        re_out[i]    = np.interp(u_i, u_grid_b2t, re_recon[i]).astype(np.float32)
        lwc_out[i]   = np.interp(u_i, u_grid_b2t, lwc_recon[i]).astype(np.float32)
        z_out[i]     = z_i.astype(np.float32)
        alpha_out[i] = new_alpha[i]

    np.clip(re_out,  re_min_raw, re_max_raw, out=re_out)
    np.clip(lwc_out, 0.0,        None,       out=lwc_out)
    # T and vapor live on the ERA5 pressure grid (length N_ERA5); no resampling.
    T_out     = np.clip(T_recon,     T_min, T_max).astype(np.float32)
    vapor_out = np.clip(vapor_recon, v_min, v_max).astype(np.float32)

    lwp   = np.empty(n, dtype=np.float32)
    tau_c = np.empty(n, dtype=np.float32)
    for i in range(n):
        lwp[i]   = compute_lwp(lwc_out[i],            z_out[i])
        tau_c[i] = compute_tau_c(re_out[i], lwc_out[i], z_out[i])

    return {
        're':         re_out,
        'lwc':        lwc_out,
        'z':          z_out,
        'alpha':      alpha_out,
        'T':          T_out,
        'vapor':      vapor_out,
        'z_top':      new_z_top.astype(np.float32),
        'z_base':     new_z_base.astype(np.float32),
        'thickness':  new_thickness.astype(np.float32),
        'mean_alpha': new_alpha.astype(np.float32),
        'tau_c':      tau_c,
        'lwp':        lwp,
    }


rng = np.random.default_rng(RANDOM_SEED)
print(f'\nSampling {N_SAMPLES} synthetic profiles with rejection on tau_c >= {TAU_C_MIN}...')

accepted = {k: [] for k in
            ('re', 'lwc', 'z', 'alpha', 'T', 'vapor',
             'z_top', 'z_base', 'thickness', 'mean_alpha',
             'tau_c', 'lwp')}
n_accepted   = 0
n_drawn_total = 0

for batch_idx in range(REJECTION_MAX_BATCHES):
    n_need = N_SAMPLES - n_accepted
    if n_need <= 0:
        break
    n_draw = max(int(REJECTION_BATCH_OVERSAMPLE * n_need), 2000)
    samples_batch = rng.multivariate_normal(mu_f, cov_f, size=n_draw)
    n_drawn_total += n_draw

    out = reconstruct_batch(samples_batch)
    keep = out['tau_c'] >= TAU_C_MIN
    take = min(int(keep.sum()), n_need)
    if take == 0:
        continue
    keep_idx = np.flatnonzero(keep)[:take]
    for k in accepted:
        accepted[k].append(out[k][keep_idx])
    n_accepted += take
    print(f'  batch {batch_idx + 1:2d}: drew {n_draw}, '
          f'{int(keep.sum())} passed (tau≥{TAU_C_MIN}), '
          f'cumulative {n_accepted}/{N_SAMPLES}')

if n_accepted < N_SAMPLES:
    raise RuntimeError(
        f'Only accumulated {n_accepted}/{N_SAMPLES} samples with '
        f'tau_c >= {TAU_C_MIN} after {REJECTION_MAX_BATCHES} batches. '
        f'Increase REJECTION_MAX_BATCHES or relax TAU_C_MIN.')

re_out      = np.concatenate(accepted['re'])
lwc_out     = np.concatenate(accepted['lwc'])
z_out       = np.concatenate(accepted['z'])
alpha_out   = np.concatenate(accepted['alpha'])
T_out       = np.concatenate(accepted['T'])
vapor_out   = np.concatenate(accepted['vapor'])
new_z_top   = np.concatenate(accepted['z_top'])
new_z_base  = np.concatenate(accepted['z_base'])
new_thickness = np.concatenate(accepted['thickness'])
new_alpha   = np.concatenate(accepted['mean_alpha'])
new_tau_c   = np.concatenate(accepted['tau_c'])
new_lwp     = np.concatenate(accepted['lwp'])

acceptance_rate = n_accepted / n_drawn_total
print(f'  done. acceptance rate = {acceptance_rate:.3f} '
      f'(drew {n_drawn_total} to keep {n_accepted})')

# For the diagnostic envelope plot we want the FULL common-grid recon of the
# kept samples. We rebuild it from the saved scores indirectly: just regenerate
# a small diagnostic batch from the same rng for plotting purposes.
diag_samples = rng.multivariate_normal(mu_f, cov_f, size=min(2000, N_SAMPLES))
diag_re_recon    = np.exp(mu_re    + diag_samples[:, slc_re]    @ comps_re)
diag_lwc_recon   = np.exp(mu_lwc   + diag_samples[:, slc_lwc]   @ comps_lwc) - LWC_EPS
diag_T_recon     = np.exp(mu_T     + diag_samples[:, slc_T]     @ comps_T)
diag_vapor_recon = np.exp(mu_vapor + diag_samples[:, slc_vapor] @ comps_vapor) - VAPOR_EPS
np.clip(diag_lwc_recon,   0.0, None, out=diag_lwc_recon)
np.clip(diag_vapor_recon, 0.0, None, out=diag_vapor_recon)

print(f'  re   range: [{re_out.min():.3f}, {re_out.max():.3f}] μm  '
      f'(in-situ raw [{re_min_raw:.3f}, {re_max_raw:.3f}])')
print(f'  lwc  range: [{lwc_out.min():.3f}, {lwc_out.max():.3f}] g/m³')
print(f'  alpha range: [{new_alpha.min():.2f}, {new_alpha.max():.2f}]')
print(f'  tau_c range: [{new_tau_c.min():.2f}, {new_tau_c.max():.2f}]   '
      f'(in-situ [{tau_c_each.min():.2f}, {tau_c_each.max():.2f}])')
print(f'  LWP  range: [{new_lwp.min():.1f}, {new_lwp.max():.1f}] g/m²   '
      f'(in-situ [{lwp_each.min():.1f}, {lwp_each.max():.1f}])')


# ── Save ───────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT_PATH,
    # Per-sample physical fields
    re               = re_out,                              # (N, 7) μm, top → base
    lwc              = lwc_out,                             # (N, 7) g/m³, top → base
    z                = z_out,                               # (N, 7) km, top → base
    alpha            = alpha_out,                           # (N, 7) constant per profile
    T                = T_out,                               # (N, 37) ERA5 K, surface → TOA
    vapor            = vapor_out,                           # (N, 37) ERA5 molec/cm³
    era5_pressure    = era5_pressure.astype(np.float32),    # (37,) hPa
    # Per-sample scalars
    z_top            = new_z_top.astype(np.float32),
    z_base           = new_z_base.astype(np.float32),
    thickness        = new_thickness.astype(np.float32),
    mean_alpha       = new_alpha.astype(np.float32),
    tau_c            = new_tau_c,
    LWP_g_per_m2     = new_lwp,
    # Generative model parameters (for resampling more without refitting)
    pca_re_mean        = mu_re.astype(np.float32),
    pca_re_components  = comps_re.astype(np.float32),
    pca_re_var_ratio   = var_re.astype(np.float32),
    pca_lwc_mean       = mu_lwc.astype(np.float32),
    pca_lwc_components = comps_lwc.astype(np.float32),
    pca_lwc_var_ratio  = var_lwc.astype(np.float32),
    pca_T_mean         = mu_T.astype(np.float32),
    pca_T_components   = comps_T.astype(np.float32),
    pca_T_var_ratio    = var_T.astype(np.float32),
    pca_vapor_mean       = mu_vapor.astype(np.float32),
    pca_vapor_components = comps_vapor.astype(np.float32),
    pca_vapor_var_ratio  = var_vapor.astype(np.float32),
    mvn_mean         = mu_f.astype(np.float32),
    mvn_cov          = cov_f.astype(np.float32),
    # Provenance
    L_COMMON         = np.int32(L_COMMON),
    K_re             = np.int32(K_re),
    K_lwc            = np.int32(K_lwc),
    K_T              = np.int32(K_T),
    K_vapor          = np.int32(K_vapor),
    N_FIXED_LEVELS   = np.int32(N_FIXED_LEVELS),
    N_ERA5           = np.int32(N_ERA5),
    TAU_C_MIN        = np.float32(TAU_C_MIN),
    n_train_profiles = np.int32(n_train),
)
print(f'\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)')


# ── Diagnostic figure (3 x 3) ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

u   = u_grid_b2t                                # 0 = base, 1 = top
pct = (5, 50, 95)

# (a) Scree plots — re and lwc
ax = axes[0, 0]
ax.plot(np.arange(1, len(var_re) + 1),  np.cumsum(var_re),  'o-',
        color='steelblue', label=f'log re  (kept K={K_re})')
ax.plot(np.arange(1, len(var_lwc) + 1), np.cumsum(var_lwc), 's-',
        color='darkorange', label=f'log lwc (kept K={K_lwc})')
ax.axhline(VAR_TARGET, color='0.5', linestyle=':', linewidth=1)
ax.set_xlabel('Principal component')
ax.set_ylabel('Cumulative variance')
ax.set_title('PCA scree (log-space)')
ax.set_xlim(0, max(K_re, K_lwc) + 5)
ax.set_ylim(0.0, 1.01)
ax.legend()
ax.grid(alpha=0.3)

# (b) Mean log re profile + first 3 PCs
ax = axes[0, 1]
ax.plot(np.exp(mu_re), u, color='black', linewidth=2, label='Mean (exp)')
for k in range(min(3, K_re)):
    sigma = scores_re[:, k].std()
    ax.plot(np.exp(mu_re + 2 * sigma * comps_re[k]), u,
            linestyle='--', alpha=0.8, label=f'+2σ PC{k + 1}')
    ax.plot(np.exp(mu_re - 2 * sigma * comps_re[k]), u,
            linestyle=':',  alpha=0.8, label=f'−2σ PC{k + 1}')
ax.set_xscale('log')
ax.set_xlabel(r'$r_e$ ($\mu$m, log)')
ax.set_ylabel('Normalized altitude (0 = base)')
ax.set_title(r'Dominant log-$r_e$ shape modes')
ax.legend(ncol=2)
ax.grid(alpha=0.3, which='both')

# (c) Mean log lwc profile + first 3 PCs
ax = axes[0, 2]
ax.plot(np.exp(mu_lwc) - LWC_EPS, u, color='black', linewidth=2, label='Mean')
for k in range(min(3, K_lwc)):
    sigma = scores_lwc[:, k].std()
    ax.plot(np.exp(mu_lwc + 2 * sigma * comps_lwc[k]) - LWC_EPS, u,
            linestyle='--', alpha=0.8, label=f'+2σ PC{k + 1}')
    ax.plot(np.exp(mu_lwc - 2 * sigma * comps_lwc[k]) - LWC_EPS, u,
            linestyle=':',  alpha=0.8, label=f'−2σ PC{k + 1}')
ax.set_xscale('log')
ax.set_xlabel(r'LWC (g/m$^3$, log)')
ax.set_ylabel('Normalized altitude (0 = base)')
ax.set_title('Dominant log-LWC shape modes')
ax.legend(ncol=2)
ax.grid(alpha=0.3, which='both')

# (d) Per-level re envelope: synthetic vs in-situ (5/50/95 percentiles)
ax = axes[1, 0]
re_train_pct = np.percentile(re_common,  pct, axis=0)
re_syn_pct   = np.percentile(diag_re_recon,   pct, axis=0)
ax.fill_betweenx(u, re_train_pct[0], re_train_pct[2], alpha=0.30,
                 color='firebrick', label='In-situ 5–95%')
ax.plot(re_train_pct[1], u, color='firebrick', linewidth=1.5, label='In-situ median')
ax.fill_betweenx(u, re_syn_pct[0], re_syn_pct[2], alpha=0.25,
                 color='steelblue', label='Synthetic 5–95%')
ax.plot(re_syn_pct[1], u, color='steelblue', linewidth=1.5, label='Synthetic median')
ax.set_xscale('log')
ax.set_xlabel(r'$r_e$ ($\mu$m, log)')
ax.set_ylabel('Normalized altitude (0 = base)')
ax.set_title('Per-level $r_e$ envelope (log)')
ax.legend()
ax.grid(alpha=0.3, which='both')

# (e) Per-level lwc envelope
ax = axes[1, 1]
lwc_train_pct = np.percentile(lwc_common, pct, axis=0)
lwc_syn_pct   = np.percentile(diag_lwc_recon,  pct, axis=0)
ax.fill_betweenx(u, lwc_train_pct[0], lwc_train_pct[2], alpha=0.30,
                 color='firebrick', label='In-situ 5–95%')
ax.plot(lwc_train_pct[1], u, color='firebrick', linewidth=1.5, label='In-situ median')
ax.fill_betweenx(u, lwc_syn_pct[0], lwc_syn_pct[2], alpha=0.25,
                 color='steelblue', label='Synthetic 5–95%')
ax.plot(lwc_syn_pct[1], u, color='steelblue', linewidth=1.5, label='Synthetic median')
ax.set_xscale('log')
ax.set_xlabel(r'LWC (g/m$^3$, log)')
ax.set_ylabel('Normalized altitude (0 = base)')
ax.set_title('Per-level LWC envelope (log)')
ax.legend()
ax.grid(alpha=0.3, which='both')

# (f) Cloud thickness distribution
ax = axes[1, 2]
bins = np.linspace(min(thickness_each.min(), new_thickness.min()),
                   max(thickness_each.max(), new_thickness.max()), 40)
ax.hist(thickness_each, bins=bins, density=True, alpha=0.55,
        color='firebrick', label=f'In-situ (n={n_train})')
ax.hist(new_thickness,  bins=bins, density=True, alpha=0.55,
        color='steelblue', label=f'Synthetic (n={N_SAMPLES})')
ax.set_xlabel('Cloud thickness (km)')
ax.set_ylabel('Density')
ax.set_title('Cloud thickness')
ax.legend()
ax.grid(alpha=0.3)

# (g) tau_c — derived synthetic vs in-situ
ax = axes[2, 0]
bins = np.logspace(np.log10(min(tau_c_each.min(), max(new_tau_c.min(), 1e-3))),
                   np.log10(max(tau_c_each.max(), new_tau_c.max())), 50)
ax.hist(tau_c_each, bins=bins, density=True, alpha=0.55,
        color='firebrick', label='In-situ')
ax.hist(new_tau_c,  bins=bins, density=True, alpha=0.55,
        color='steelblue', label=f'Synthetic (derived)')
ax.set_xscale('log')
ax.set_xlabel(r'$\tau_c$')
ax.set_ylabel('Density')
ax.set_title(r'Cloud optical depth $\tau_c$')
ax.legend()
ax.grid(alpha=0.3, which='both')

# (h) LWP — derived synthetic vs in-situ
ax = axes[2, 1]
bins = np.logspace(np.log10(max(min(lwp_each.min(), new_lwp.min()), 1e-3)),
                   np.log10(max(lwp_each.max(), new_lwp.max())), 50)
ax.hist(lwp_each, bins=bins, density=True, alpha=0.55,
        color='firebrick', label='In-situ')
ax.hist(new_lwp,  bins=bins, density=True, alpha=0.55,
        color='steelblue', label='Synthetic (derived)')
ax.set_xscale('log')
ax.set_xlabel(r'LWP (g/m$^2$)')
ax.set_ylabel('Density')
ax.set_title('Liquid water path')
ax.legend()
ax.grid(alpha=0.3, which='both')

# (i) Mean alpha distribution
ax = axes[2, 2]
bins = np.linspace(min(mean_alpha.min(), new_alpha.min()),
                   max(mean_alpha.max(), new_alpha.max()), 40)
ax.hist(mean_alpha, bins=bins, density=True, alpha=0.55,
        color='firebrick', label='In-situ')
ax.hist(new_alpha,  bins=bins, density=True, alpha=0.55,
        color='steelblue', label='Synthetic')
ax.set_xlabel(r'Vertical-mean $\alpha$ (libRadtran shape)')
ax.set_ylabel('Density')
ax.set_title(r'Mean $\alpha_{\rm param}$')
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(
    f'Joint MVN generator: ORACLES, '
    f'$\\tau_c \\geq {TAU_C_MIN:g}$,  N$_\\text{{train}}$={n_train},  '
    f'L={L_COMMON},  K$_{{re}}$={K_re},  K$_{{lwc}}$={K_lwc},  '
    f'N$_\\text{{sample}}$={N_SAMPLES}',
    y=1.005,
)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Diagnostic figure → {FIG_PATH}  (dpi={FIGURE_DPI})')


# ── Atmospheric diagnostic figure (2 x 3) — ERA5 T and vapor ──────────────────
fig_a, axes_a = plt.subplots(2, 3, figsize=(15, 9))
P = era5_pressure                                        # (37,) hPa, surface → TOA

# (a) Scree — log T, log vapor
ax = axes_a[0, 0]
ax.plot(np.arange(1, len(var_T) + 1), np.cumsum(var_T), 'o-',
        color='steelblue', label=f'log T     (kept K={K_T})')
ax.plot(np.arange(1, len(var_vapor) + 1), np.cumsum(var_vapor), 's-',
        color='darkorange', label=f'log vapor (kept K={K_vapor})')
ax.axhline(VAR_TARGET, color='0.5', linestyle=':', linewidth=1)
ax.set_xlabel('Principal component')
ax.set_ylabel('Cumulative variance')
ax.set_title('PCA scree (atmosphere, log-space)')
ax.set_ylim(0.0, 1.01)
ax.legend()
ax.grid(alpha=0.3)

# (b) Mean T profile + first 3 PCs (linear T axis; T variation is small)
ax = axes_a[0, 1]
ax.plot(np.exp(mu_T), P, color='black', linewidth=2, label='Mean')
for k in range(min(3, K_T)):
    sigma = scores_T[:, k].std()
    ax.plot(np.exp(mu_T + 2 * sigma * comps_T[k]), P,
            linestyle='--', alpha=0.8, label=f'+2σ PC{k + 1}')
    ax.plot(np.exp(mu_T - 2 * sigma * comps_T[k]), P,
            linestyle=':',  alpha=0.8, label=f'−2σ PC{k + 1}')
ax.invert_yaxis()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Dominant log-T shape modes')
ax.legend(ncol=2)
ax.grid(alpha=0.3)

# (c) Mean vapor profile + first 3 PCs (log axis)
ax = axes_a[0, 2]
ax.plot(np.exp(mu_vapor) - VAPOR_EPS, P, color='black', linewidth=2, label='Mean')
for k in range(min(3, K_vapor)):
    sigma = scores_vapor[:, k].std()
    ax.plot(np.exp(mu_vapor + 2 * sigma * comps_vapor[k]) - VAPOR_EPS, P,
            linestyle='--', alpha=0.8, label=f'+2σ PC{k + 1}')
    ax.plot(np.exp(mu_vapor - 2 * sigma * comps_vapor[k]) - VAPOR_EPS, P,
            linestyle=':',  alpha=0.8, label=f'−2σ PC{k + 1}')
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel(r'Vapor concentration (molec/cm$^3$, log)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Dominant log-vapor shape modes')
ax.legend(ncol=2)
ax.grid(alpha=0.3, which='both')

# (d) Per-level T envelope: synthetic vs in-situ
ax = axes_a[1, 0]
T_train_pct = np.percentile(T_arr,         pct, axis=0)
T_syn_pct   = np.percentile(diag_T_recon,  pct, axis=0)
ax.fill_betweenx(P, T_train_pct[0], T_train_pct[2], alpha=0.30,
                 color='firebrick', label='In-situ 5–95%')
ax.plot(T_train_pct[1], P, color='firebrick', linewidth=1.5, label='In-situ median')
ax.fill_betweenx(P, T_syn_pct[0], T_syn_pct[2], alpha=0.25,
                 color='steelblue', label='Synthetic 5–95%')
ax.plot(T_syn_pct[1], P, color='steelblue', linewidth=1.5, label='Synthetic median')
ax.invert_yaxis()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Per-level T envelope')
ax.legend()
ax.grid(alpha=0.3)

# (e) Per-level vapor envelope: synthetic vs in-situ (log)
ax = axes_a[1, 1]
v_train_pct = np.percentile(vapor_arr,        pct, axis=0)
v_syn_pct   = np.percentile(diag_vapor_recon, pct, axis=0)
ax.fill_betweenx(P, v_train_pct[0], v_train_pct[2], alpha=0.30,
                 color='firebrick', label='In-situ 5–95%')
ax.plot(v_train_pct[1], P, color='firebrick', linewidth=1.5, label='In-situ median')
ax.fill_betweenx(P, v_syn_pct[0], v_syn_pct[2], alpha=0.25,
                 color='steelblue', label='Synthetic 5–95%')
ax.plot(v_syn_pct[1], P, color='steelblue', linewidth=1.5, label='Synthetic median')
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel(r'Vapor concentration (molec/cm$^3$, log)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Per-level vapor envelope')
ax.legend()
ax.grid(alpha=0.3, which='both')

# (f) Distribution: column-integrated water vapor (molec/cm²)
# Convert hPa to height via hypsometric approximation isn't critical here;
# trapezoid in log-pressure produces a column metric on the same units for
# both populations, so ratios match. For interpretability, integrate vapor
# over pressure-equivalent height (hPa → ~ -dz) so units come out consistent.
def column_vapor(vap_arr_2d, P_hPa):
    # crude proxy: integrate vapor against pressure, units are molec/cm³ * hPa
    return np.trapezoid(vap_arr_2d, P_hPa, axis=1)
cv_train = np.abs(column_vapor(vapor_arr,        P))
cv_syn   = np.abs(column_vapor(diag_vapor_recon, P))
ax = axes_a[1, 2]
bins = np.logspace(np.log10(min(cv_train.min(), cv_syn.min())),
                   np.log10(max(cv_train.max(), cv_syn.max())), 40)
ax.hist(cv_train, bins=bins, density=True, alpha=0.55,
        color='firebrick', label='In-situ')
ax.hist(cv_syn,   bins=bins, density=True, alpha=0.55,
        color='steelblue', label='Synthetic')
ax.set_xscale('log')
ax.set_xlabel(r'$\int$ vapor $\,dP$  (molec/cm$^3$ · hPa, log)')
ax.set_ylabel('Density')
ax.set_title('Column-integrated water vapor proxy')
ax.legend()
ax.grid(alpha=0.3, which='both')

fig_a.suptitle(
    f'Atmosphere diagnostic: ERA5 T and vapor  '
    f'(K$_T$={K_T}, K$_{{vapor}}$={K_vapor}, N$_{{train}}$={n_train})',
    y=1.005,
)
fig_a.tight_layout()
fig_a.savefig(FIG_ATMOS_PATH, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Atmosphere figure → {FIG_ATMOS_PATH}  (dpi={FIGURE_DPI})')


# ── Random-examples figure (3 x 3, synthetic only) ────────────────────────────
# Fresh RNG (no seed) so the picks change every run — useful for eyeballing
# the smoothness of synthesized profiles and how it scales with K_re / K_lwc.
example_rng = np.random.default_rng()
syn_idx = example_rng.choice(N_SAMPLES, size=9, replace=False)

fig_ex, axes_ex = plt.subplots(3, 3, figsize=(14, 12))
axes_ex = axes_ex.flatten()

for ax, sidx in zip(axes_ex, syn_idx):
    ax.plot(re_out[sidx], z_out[sidx],
            color='steelblue', linewidth=1.6,
            marker='o', markersize=5,
            label=r'$r_e$ ($\mu$m)')
    ax2 = ax.twiny()
    ax2.plot(lwc_out[sidx], z_out[sidx],
             color='darkorange', linewidth=1.6, linestyle='--',
             marker='s', markersize=4,
             label=r'LWC (g/m$^3$)')
    ax.set_xlabel(r'$r_e$ ($\mu$m)', color='steelblue')
    ax2.set_xlabel(r'LWC (g/m$^3$)',  color='darkorange')
    ax.set_ylabel('Altitude (km)')
    ax.tick_params(axis='x', colors='steelblue')
    ax2.tick_params(axis='x', colors='darkorange')
    ax.set_title(
        fr'$\tau_c$={new_tau_c[sidx]:.1f},  '
        fr'LWP={new_lwp[sidx]:.0f} g/m$^2$,  '
        fr'$\alpha$={new_alpha[sidx]:.1f}',
        fontsize=10,
    )
    ax.grid(alpha=0.3)

fig_ex.suptitle(
    f'9 random synthetic profiles  '
    f'(K$_{{re}}$={K_re}, K$_{{lwc}}$={K_lwc})',
    y=1.005,
)
fig_ex.tight_layout()
fig_ex.savefig(FIG_EXAMPLES_PATH, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Examples figure   → {FIG_EXAMPLES_PATH}  (dpi={FIGURE_DPI})')

plt.show()
