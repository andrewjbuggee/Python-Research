"""
Synthetic in-situ profile generator (joint r_e + LWC + alpha_param + geometry).

Fits a functional-PCA + multivariate-normal generative model in LOG-SPACE
to a set of in-situ profiles from the ORACLES and VOCALS-REx field campaign,
then samples N new synthetic profiles. Each synthetic sample carries:

    re        : (N_FIXED_LEVELS,) effective radius profile  (μm, top → base)
    lwc       : (N_FIXED_LEVELS,) liquid water content        (g/m³,  top → base)
    z         : (N_FIXED_LEVELS,) altitude grid              (km,   top → base)
    alpha     : scalar mean libRadtran distribution shape parameter
    z_top     : scalar cloud-top altitude     (km)
    z_base    : scalar cloud-base altitude    (km)
    thickness : scalar cloud geometric depth  (km)
    tau_c     : scalar cloud optical depth    (derived from re + LWC)
    LWP       : scalar liquid water path      (g/m², derived from LWC)

Method
======
1. Load the in-situ profiles from MAT_DIR, which may be either
     (a) the consolidated .mat file built by
         build_synthetic_profile_inputs_from_insitu_and_ERA5.m, holding both
         campaigns with alpha for every profile (preferred), or
     (b) the legacy directory of one .mat file per profile, where only the
         ORACLES files store 'alpha_param'.
   Keep only profiles with TAU_C_MIN ≤ tau_c ≤ TAU_C_MAX (defaults 3 and 40)
   that are not duplicate profile fingerprints.
2. Project raw re and lwc onto a common normalized-altitude grid of length
   L_COMMON (≥ 60), then take logs:
       Y_re  = log(re_common)              (always positive, no offset)
       Y_lwc = log(lwc_common + LWC_EPS)   (small floor for near-base zeros)
   Per-profile mean alpha is computed by trapezoid-weighted average over
   the raw altitude axis and stored as log(mean_alpha) for the joint MVN.
3. PCA via SVD on Y_re and Y_lwc separately. Keep K_RE / K_LWC modes
   chosen so cumulative variance ≥ VAR_TARGET (default 0.99).
4. Single joint multivariate normal over the feature
       [re_scores; lwc_scores; T_scores; vapor_scores;
        log(mean_alpha); z_base; log(thickness)]
   log(mean_alpha) is included whenever every kept profile carries an alpha,
   which is always true for the consolidated input. With the legacy input,
   where only ORACLES files store alpha_param, it is dropped from the joint
   feature and sampled from a 1-D log-normal on the ORACLES subset instead.
   The cross-covariance terms encode the physical coupling between r_e
   and LWC shape, between cloud thickness and shape, etc.
5. Draw N samples from this MVN. Inverse-PCA gives Y_re and Y_lwc on the
   common grid; exp() recovers re and lwc as guaranteed-positive log-normal
   variates at each level. Resample to N_FIXED_LEVELS evenly-spaced
   altitudes from sampled (z_top, z_base). alpha is broadcast as a scalar.
   Draws are rejected (and replaced by new draws, until N are accepted) if
   tau_c falls outside [TAU_C_MIN, TAU_C_MAX] or, with LWC_MAX_REJECTION, if
   LWC at any of the N_FIXED_LEVELS exceeds the in-situ 99th-percentile LWC at
   that normalised height (see "LWC ceiling" below).
6. Derive per-sample LWP and tau_c from the re + lwc profile:
       LWP[g/m²] = 1000 · ∫ LWC dz   (LWC g/m³, dz km)
       tau_c     = 750 · ∫ Q̄_ext(r_e, α) · LWC / r_e dz   at 550 nm
                   (LWC g/m³, r_e μm, dz km; Hansen & Travis 1974 eqs. 2.48,
                   2.53, 2.58), with Q̄_ext from the libRadtran custom Mie
                   table nearest the draw's α. Both by trapezoid over the levels.
   These are physically self-consistent with the sampled re + lwc + alpha;
   they are NOT independently sampled.
7. Save to .npz; write a 3 x 3 diagnostic figure at DPI 500.

Profile orientation throughout: index 0 = cloud top, index -1 = cloud base.
"""

import datetime
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
# MAT_DIR accepts either input layout (see "Input loading" below):
#   - the consolidated .mat file, or the directory holding it (preferred), or
#   - the legacy directory of one .mat file per profile.
# The layout is detected from what the path actually points at.
MAT_DIR          = Path('/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/'
               'Hyperspectral_Cloud_Retrievals/Neural_Network/'
               'Training_data_set/synthetic_profile_inputs/'
               'insitu_era5_synthetic_inputs_combined_08-Nov-2026.mat')

# Legacy per-profile data set (VOCALS-REx profiles there carry no alpha):
# MAT_DIR = Path('/Volumes/My Passport/CU_Boulder/neural_network_training_data/saz0_allProfiles/')

TAU_C_MIN        = 3.0          # only use clouds with cloud optical depth ≥ this
TAU_C_MAX        = 40.0         # ... and ≤ this. Both limits apply twice: to the
                                # in-situ profiles before the MVN is fit (so the
                                # thickest clouds don't shape the PCA modes and
                                # covariance) and to every synthetic draw (the
                                # log-space MVN tail would otherwise exceed the cap)

# Which optical thickness the TAU_C_MIN / TAU_C_MAX filter uses (consolidated input only):
#   'stored'   - max of the in-situ tau profile, as recorded by the campaign code
#   'computed' - recomputed in build_synthetic_profile_inputs_from_insitu_and_ERA5.m
#                from the measured r_e and LWC with the OLD constant Q_ext = 2.32
#                (not the Mie Q̄_ext now used for the synthetic draws)
# A handful of stored ORACLES tau profiles are corrupt (τ ~ 1e-4 for clouds with
# LWP > 50 g/m²); 'computed' rescues those but runs ~20% lower than 'stored' on
# the profiles where both are sound, so don't mix the two within one run.
TAU_C_SOURCE     = 'stored'

# ── OUTLIERS ───────────────────────────────────────────────────────────────────
# In-situ profiles left out of the joint MVN fit (consolidated input only). Each
# entry is (campaign, date_of_flight, profile_idx), as written per profile by
# build_synthetic_profile_inputs_from_insitu_and_ERA5.m; profile_idx is the
# profile's position in that campaign's saved ensemble.
#
# ORACLES, 30 Aug 2017, profiles 111 and 112 (rows 202 and 203 of the
# 08-Nov-2026 combined file):
#
#   What they are. Median LWC of 1.9 and 2.3 g/m³ (max 2.6 and 3.1) through
#   cloud layers only 122 and 125 m deep, at 17–18 °C. An adiabatic parcel at
#   that temperature and pressure gains about 2.4 g/m³ per km of ascent, i.e.
#   about 0.3 g/m³ over 125 m. These layers hold roughly ten times the water
#   that condensation inside them could supply: f_ad = LWP_obs / LWP_ad is 9.3
#   and 10.4 (Wood 2005 definition, same cloud-base rule and ERA5 Γ_ad as
#   plot_per_cloud_rmse_kfold.wood_adiabaticity).
#
#   How far out they are. Every other profile in the training set
#   (TAU_C_MIN ≤ τ_c ≤ TAU_C_MAX) has f_ad ≤ 2.75; the next highest is the
#   VOCALS-REx RF07 drizzle shaft at 2.75. That is a gap of more than a factor
#   of three, which is why only these two are listed.
#
#   Why they distort the MVN. The generator fits ONE Gaussian to the joint
#   log-space LWC PCA scores, log-thickness and the other features. Two points
#   this far off the LWP–thickness relation inflate its covariance and loosen
#   the coupling between LWP and h, which widens the synthetic f_ad
#   distribution. Generating 100,000 profiles with and without them (same code,
#   seed RANDOM_SEED = 0 and 08-Nov-2026 inputs, 11 Nov 2026), each compared with
#   the 290 remaining in-situ profiles resampled to 7 levels:
#                                    f_ad > 1   p90 f_ad   corr(log h, log LWP)
#       in-situ, 7 levels             14.8%      1.06        0.80
#       synthetic, outliers kept      22.2%      1.33        0.73
#       synthetic, outliers dropped   19.5%      1.24        0.77
#   KS test on f_ad against the in-situ profiles: D = 0.099 (p = 0.007) kept,
#   D = 0.073 (p = 0.084) dropped. The medians agree either way (synthetic minus
#   in-situ +0.02 and +0.01, both within the 95% bootstrap interval of zero).
#   Dropping them narrows the tail, but the synthetic distribution is still
#   somewhat wider than the in-situ one.
#   (adiabaticity_insitu_vs_synthetic.ipynb reproduces the comparison).
#
#   What they physically are is uncertain. My inference: a convective cell or
#   rain shaft rather than stratocumulus, or a probe artifact. Either way they
#   are not the stratiform population the generator is meant to reproduce.
#
# Dropping them is a judgment call; empty this tuple to restore them.
OUTLIERS = (
    ('oracles', '2017-08-30', 111),
    ('oracles', '2017-08-30', 112),
)

L_COMMON         = 80           # length of the common normalized-altitude grid (≥ 60)
N_FIXED_LEVELS   = 7            # every synthetic profile is on this fixed grid
N_SAMPLES        = 100000

# Set var_target = 1.0 to keep exactly K_RE_MAX / K_LWC_MAX modes
VAR_TARGET       = 1.0          # cumulative variance target for picking K_RE, K_LWC, K_T, K_VAPOR
K_RE_MAX         = 3             # hard cap on number of re modes kept
K_LWC_MAX        = 5             # hard cap on number of lwc modes kept
K_T_MAX          = 10            # hard cap on number of ERA5 temperature modes kept
K_VAPOR_MAX      = 8            # hard cap on number of ERA5 vapor-concentration modes kept

LWC_EPS          = 1e-3         # g/m³ floor for log(lwc + eps); below typical in-cloud LWC
VAPOR_EPS        = 1.0          # molec/cm³ floor for log(vapor + eps)
# τ_c of every synthetic draw uses the size-distribution-averaged Mie
# extinction efficiency Q̄_ext(r_e, α) at TAU_WAVELENGTH_UM, read from the same
# libRadtran custom Mie tables the synthetic RT uses (see compute_tau_c).
# The 1–50 μm tables match hysics_refl_from_synthetic_NN_inputs.m
# (use_35_or_50 = 50); where they overlap, the 1–35 μm tables hold identical values.
MIE_TABLE_DIR         = Path('/Users/andrewbuggee/Documents/libRadtran-2.0.6/'
                             'data/wc/custom_mieTables')
MIE_TABLE_REFF_MAX_UM = 50      # which table family: 35 or 50 (μm upper r_e limit)
TAU_WAVELENGTH_UM     = 0.55    # wavelength τ_c is defined at (μm); 550 nm as in
                                # find_verticalProfiles_VOCALS_REx_ver2.m
RANDOM_SEED      = 0

RUN_DATE         = datetime.date.today().strftime('%Y-%m-%d')

OUT_DIR          = Path(__file__).parent / 'synthetic_profiles'
OUT_PATH         = OUT_DIR / f'synthetic_profiles_jointMVN_N{N_SAMPLES}_L{N_FIXED_LEVELS}_{RUN_DATE}.npz'
FIG_PATH         = OUT_DIR / f'synthetic_profiles_jointMVN_diagnostic_N{N_SAMPLES}_{RUN_DATE}.png'
FIG_EXAMPLES_PATH = OUT_DIR / f'synthetic_profiles_jointMVN_examples_N{N_SAMPLES}_{RUN_DATE}.png'
FIG_ATMOS_PATH    = OUT_DIR / f'synthetic_profiles_jointMVN_atmos_diagnostic_N{N_SAMPLES}_{RUN_DATE}.png'
FIGURE_DPI       = 500


def _versioned_path(path):
    """Append '_ver2', '_ver3', ... to path's stem if a file already exists there."""
    if not path.exists():
        return path
    ver = 2
    while True:
        candidate = path.with_name(f'{path.stem}_ver{ver}{path.suffix}')
        if not candidate.exists():
            return candidate
        ver += 1

# Rejection-sampling loop: keeps drawing batches until N samples have
# TAU_C_MIN <= tau_c <= TAU_C_MAX (matching the in-situ filter). Without this,
# the tails of the joint MVN produce thinner- and thicker-than-trained clouds.
REJECTION_BATCH_OVERSAMPLE = 1.6      # draw 1.6× the remaining need each batch
REJECTION_MAX_BATCHES      = 30

# LWC ceiling (added 13 Nov 2026). The joint MVN is Gaussian in log LWC, so at
# cloud top, where the in-situ values have a long tail toward tiny LWC and a
# hard upper limit (topmost measured level ≤ 0.63 g/m³), the symmetric
# log-normal puts mass far above anything measured: with no ceiling, 1.9 % of
# synthetic top levels exceeded 1 g/m³ and the largest reached 8.97 g/m³
# (08b_compare_lwc_insitu_vs_synthetic.py, 12-Nov-2026 bundle). With
# LWC_MAX_REJECTION, a draw is rejected when LWC at any output level exceeds
# LWC_MAX_FACTOR × the LWC_CEILING_PERCENTILE-th percentile of the in-situ LWC at
# that normalised height: the percentile across training profiles of
# lwc_common (the 80-level grid), read off at the N_FIXED_LEVELS output
# heights. The 99th percentile rather than the maximum (13 Nov 2026): the
# maximum at a height is set by a single profile and can be probe noise (e.g.
# the 7.84 g/m³ VOCALS-REx 2008-10-25 profile-24 spike). Rejecting rather than
# clipping avoids a pile-up at the ceiling.
LWC_MAX_REJECTION      = True
LWC_CEILING_PERCENTILE = 100.0     # 100 = in-situ maximum
LWC_MAX_FACTOR         = 1.0


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


def load_mie_ext_tables(table_dir: Path, reff_max_um: int,
                        wavelength_um: float) -> tuple[np.ndarray, dict]:
    """Read ext(r_e) at one wavelength from every libRadtran custom Mie table.

    The tables were made with libRadtran's Mie tool for its gamma size
    distribution, n(r) ∝ r^α · exp(−(α+3) r / r_e). That is Hansen & Travis
    (1974) eq. (2.56) with a = r_eff and b = v_eff = 1/(α+3), since
    (1 − 3b)/b = α (H&T eq. 2.57). There is one file per α:
        wc_mieTable_gamma_rEff_1-<reff_max_um>microns_gammaDist_alpha_<α:03d>.cdf

    The variable 'ext' (nlam, nreff) is the extinction coefficient per unit
    liquid water content, β/LWC, in km⁻¹ / (g m⁻³). The files are netCDF-4
    (HDF5), so h5py reads them; no netCDF4 dependency is needed.

    Returns
    -------
    alphas : (n_tables,) sorted α values for which a table exists
    tables : {α: (reff_um (nreff,), ext_per_lwc (nreff,))} with ext already
             interpolated (linearly) to `wavelength_um`
    """
    import h5py   # in requirements.txt; imported here so a missing install
                  # fails only when the tables are actually needed

    pattern = f'wc_mieTable_gamma_rEff_1-{reff_max_um}microns_gammaDist_alpha_*.cdf'
    files = sorted(f for f in table_dir.glob(pattern) if not f.name.startswith('._'))
    if not files:
        raise SystemExit(f'No Mie tables matching {pattern} in {table_dir}')

    tables = {}
    for f in files:
        alpha = int(f.stem.rsplit('_', 1)[-1])          # '..._alpha_007' -> 7
        with h5py.File(f, 'r') as ds:
            wl_um   = ds['wavelen'][:].astype(np.float64)        # (nlam,) μm
            reff_um = ds['reff'][:].astype(np.float64)           # (nreff,) μm
            ext     = ds['ext'][:].astype(np.float64)            # (nlam, nreff)
        if not (wl_um[0] <= wavelength_um <= wl_um[-1]):
            raise SystemExit(f'{f.name}: {wavelength_um} μm is outside the table '
                             f'wavelengths [{wl_um[0]}, {wl_um[-1]}] μm')
        # Linear interpolation in wavelength for every r_e column:
        # (nlam, nreff) -> (nreff,). 0.55 μm is a grid point of these tables,
        # so at the default wavelength this is an exact lookup.
        ext_wl = np.array([np.interp(wavelength_um, wl_um, ext[:, j])
                           for j in range(reff_um.size)])
        tables[alpha] = (reff_um, ext_wl)

    alphas = np.array(sorted(tables), dtype=np.float64)
    return alphas, tables


def nearest_mie_alpha(alpha):
    """α of the Mie table closest to `alpha` (scalar or array).

    Same rule as find_custom_mieTable_closest_to_alpha_profile.m, which picks
    the table libRadtran uses for each synthetic cloud in
    hysics_refl_from_synthetic_NN_inputs.m.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    idx = np.argmin(np.abs(alpha[..., None] - MIE_ALPHAS), axis=-1)
    return MIE_ALPHAS[idx].astype(int)


def compute_tau_c(re_top_to_base, lwc_top_to_base, z_top_to_base, alpha):
    """Cloud optical depth at TAU_WAVELENGTH_UM from r_e (μm), LWC (g/m³), z (km).

    Physics (Hansen & Travis 1974, Space Sci. Rev. 16, 527–610):

      Extinction coefficient of a polydispersion, eq. (2.48):
          β = ∫ π r² Q_ext(r) n(r) dr
      Effective radius, eq. (2.53), and the size-distribution-averaged
      efficiency, eq. (2.58) (written there for scattering; identical form):
          r_e = ⟨r³⟩/⟨r²⟩,   Q̄_ext = β / G,   G = ∫ π r² n(r) dr
      With LWC = ρ_w (4/3) π ⟨r³⟩ N, these give EXACTLY
          β = (3/4) · Q̄_ext(r_e, α, λ) · LWC / (ρ_w · r_e)
      i.e. in mixed units  β [km⁻¹] = 750 · Q̄_ext · LWC[g m⁻³] / r_e[μm].
      This pairs Q̄_ext with LWC/r_e (∝ G), which is where H&T define it. The
      only assumption is that n(r) is libRadtran's gamma with this α.

      The Mie tables store ext = β/LWC = 750 · Q̄_ext / r_e, so
          β(z) = ext(r_e(z); α, λ) · LWC(z)
      with ext linearly interpolated in r_e (libRadtran's 'interpolate' does
      the same; checked against uvspec's verbose per-layer τ to 1e-4 in
      Testing/test_wc_geometry_optical_depth.m).

      τ_c = ∫ β dz, by the trapezoid rule over the profile levels, i.e. over
      z_base … z_top. This is the cloud the profile and the LWP describe.

    This replaces the old constant Q_ext = 2.32. At 550 nm Q̄_ext is 2.06–2.23
    for these clouds (median ≈ 2.11), so the constant overstated τ by ≈ 10%.
    The remaining ~2.32/2.11 came from the libRadtran wc_file geometry
    (write_wc_file_from_in_situ.m places the top level's values in a layer
    above z_top), which is a property of the RT input, not of the cloud.

    Parameters
    ----------
    re_top_to_base, lwc_top_to_base, z_top_to_base : (n_levels,) arrays
    alpha : scalar gamma shape parameter of this profile; the table with the
            nearest α is used

    Returns
    -------
    τ_c (float), dimensionless, at TAU_WAVELENGTH_UM
    """
    re_um = np.asarray(re_top_to_base, dtype=np.float64)
    reff_tab_um, ext_tab = MIE_TABLES[int(nearest_mie_alpha(alpha))]

    # np.interp clamps silently outside the table; refuse instead, since
    # libRadtran would also fail on these
    if re_um.min() < reff_tab_um[0] or re_um.max() > reff_tab_um[-1]:
        raise ValueError(f'r_e range [{re_um.min():.2f}, {re_um.max():.2f}] μm is '
                         f'outside the Mie table [{reff_tab_um[0]}, {reff_tab_um[-1]}] μm')

    beta_per_km = np.interp(re_um, reff_tab_um, ext_tab) * lwc_top_to_base   # (n_levels,) km⁻¹
    return float(abs(np.trapezoid(beta_per_km, z_top_to_base)))


def compute_qbar_ext(re_um, alpha):
    """Q̄_ext(r_e, α) at TAU_WAVELENGTH_UM, from ext = 750 · Q̄_ext / r_e. For diagnostics."""
    reff_tab_um, ext_tab = MIE_TABLES[int(nearest_mie_alpha(alpha))]
    re_um = np.asarray(re_um, dtype=np.float64)
    return np.interp(re_um, reff_tab_um, ext_tab) * re_um / 750.0


# Load the Mie tables once; compute_tau_c reads them from these globals
MIE_ALPHAS, MIE_TABLES = load_mie_ext_tables(MIE_TABLE_DIR, MIE_TABLE_REFF_MAX_UM,
                                             TAU_WAVELENGTH_UM)
print(f'Mie tables: {len(MIE_ALPHAS)} α values '
      f'[{MIE_ALPHAS.min():g} … {MIE_ALPHAS.max():g}] from {MIE_TABLE_DIR}, '
      f'τ at {TAU_WAVELENGTH_UM} μm')


# ── Input loading ──────────────────────────────────────────────────────────────
# Two input layouts are supported.
#
# 1. Consolidated (preferred). A single .mat file written by
#      Matlab-Research/Hyperspectral_Cloud_Retrievals/Neural_Network/
#      build_synthetic_profile_inputs_from_insitu_and_ERA5.m
#    holding every VOCALS-REx and ORACLES in-situ profile together with its
#    paired ERA5 profile. Both campaigns carry a per-level gamma shape
#    parameter, so alpha is available for every profile.
#
# 2. Per-profile (legacy). A directory of one .mat file per profile from the
#    neural-network training data set (saz0_allProfiles). Each file holds
#    're', 'lwc', 'z', 'tau', an 'era5' struct, reflectances, and — for ORACLES
#    only — 'alpha_param'; VOCALS-REx profiles there have no alpha.
#
# The layout is detected from what MAT_DIR points at rather than from the
# literal path, so either drive or folder works.

CONSOLIDATED_GLOB = 'insitu_era5_synthetic_inputs_*.mat'


def resolve_input_path(path: Path) -> tuple[str, Path]:
    """Work out which input layout `path` refers to.

    Returns ('consolidated', <.mat file>) or ('per_profile', <directory>).
    A directory is treated as consolidated when it holds at least one
    CONSOLIDATED_GLOB file; the 'combined' one (both campaigns) wins, and the
    most recently modified breaks any remaining tie.
    """
    if path.is_file():
        if path.suffix != '.mat':
            raise SystemExit(f'MAT_DIR is a file but not a .mat file: {path}')
        return 'consolidated', path

    if not path.is_dir():
        raise SystemExit(
            f'MAT_DIR does not exist: {path} '
            f'(is the external drive mounted?)'
        )

    candidates = [f for f in path.glob(CONSOLIDATED_GLOB)
                  if not f.name.startswith('._')]

    if candidates:
        combined = [f for f in candidates if 'combined' in f.name]
        pick = max(combined or candidates, key=lambda f: f.stat().st_mtime)
        return 'consolidated', pick

    return 'per_profile', path


def load_profiles_consolidated(path: Path) -> dict:
    """Load every profile from the consolidated .mat file.

    In-cloud vectors are stored cloud top → base and ERA5 profiles surface →
    TOA, which is the orientation the rest of this script assumes.
    """
    d = scipy.io.loadmat(path, squeeze_me=True)

    required = {'re', 'lwc', 'z', 'alpha', 'tau_c', 'era5_T_K',
                'era5_vapor_concentration_cm3', 'era5_pressure_hPa'}
    missing = required - set(d.keys())
    if missing:
        raise SystemExit(
            f'{path.name} is missing expected variables: {sorted(missing)}'
        )

    if TAU_C_SOURCE not in ('stored', 'computed'):
        raise SystemExit(f"TAU_C_SOURCE must be 'stored' or 'computed', "
                         f'got {TAU_C_SOURCE!r}')

    tau_c_stored   = np.atleast_1d(np.asarray(d['tau_c'],          dtype=np.float64))
    tau_c_computed = np.atleast_1d(np.asarray(d['tau_c_computed'], dtype=np.float64))
    tau_c_all      = tau_c_stored if TAU_C_SOURCE == 'stored' else tau_c_computed

    # Campaign label per profile; the single-campaign files carry one label for
    # the whole file rather than a per-profile vector.
    n_file = len(tau_c_all)
    if 'campaign_each' in d:
        campaign_all = np.atleast_1d(np.asarray(d['campaign_each'], dtype=object))
    else:
        campaign_all = np.asarray([str(d['campaign'])] * n_file, dtype=object)

    # Identifiers used to match OUTLIERS: (campaign, date_of_flight, profile_idx)
    if OUTLIERS and not {'date_of_flight', 'profile_idx'}.issubset(d.keys()):
        raise SystemExit(f'{path.name} has no date_of_flight / profile_idx, so '
                         f'OUTLIERS cannot be matched. Empty OUTLIERS or rebuild the file.')
    outlier_set = {(str(c), str(t), int(k)) for c, t, k in OUTLIERS}
    outliers_found = set()

    out = {k: [] for k in ('re', 'lwc', 'z', 'mean_alpha', 'campaign',
                           'z_top', 'z_base', 'tau_c', 'lwp', 'n_levels',
                           'T', 'vapor', 'source_index')}
    skips = dict(low_tau=0, high_tau=0, outlier=0, dup=0, missing=0, no_era5=0)
    seen_fingerprints = set()

    for i in range(n_file):

        if outlier_set:
            key = (str(campaign_all[i]), str(d['date_of_flight'][i]), int(d['profile_idx'][i]))
            if key in outlier_set:
                skips['outlier'] += 1
                outliers_found.add(key)
                continue

        re    = np.atleast_1d(np.asarray(d['re'][i],    dtype=np.float64))
        lwc   = np.atleast_1d(np.asarray(d['lwc'][i],   dtype=np.float64))
        z     = np.atleast_1d(np.asarray(d['z'][i],     dtype=np.float64))
        alpha = np.atleast_1d(np.asarray(d['alpha'][i], dtype=np.float64))

        if tau_c_all[i] < TAU_C_MIN:
            skips['low_tau'] += 1
            continue
        if tau_c_all[i] > TAU_C_MAX:
            skips['high_tau'] += 1
            continue

        fp = tuple(np.round(re[:5], 4))
        if fp in seen_fingerprints:
            skips['dup'] += 1
            continue
        seen_fingerprints.add(fp)

        out['re'].append(re)
        out['lwc'].append(lwc)
        out['z'].append(z)
        out['mean_alpha'].append(trapz_vertical_mean(alpha, z))
        out['campaign'].append(str(campaign_all[i]))
        out['z_top'].append(float(z[0]))
        out['z_base'].append(float(z[-1]))
        out['tau_c'].append(float(tau_c_all[i]))
        out['lwp'].append(compute_lwp(lwc, z))
        out['n_levels'].append(len(re))
        out['T'].append(np.asarray(d['era5_T_K'][i], dtype=np.float64))
        out['vapor'].append(
            np.asarray(d['era5_vapor_concentration_cm3'][i], dtype=np.float64))
        out['source_index'].append(i)     # row in the input file, saved for provenance

    # Every listed outlier should have been found; a miss means the input file
    # changed and the list needs checking
    missing_outliers = outlier_set - outliers_found
    if missing_outliers:
        print(f'  WARNING: OUTLIERS not found in {path.name}: {sorted(missing_outliers)}')

    out['era5_pressure'] = np.asarray(d['era5_pressure_hPa'][0], dtype=np.float64)
    out['skips'] = skips

    # Say how much the other tau_c choice would change the filter, since the two
    # are not interchangeable.
    other = tau_c_computed if TAU_C_SOURCE == 'stored' else tau_c_stored
    in_range = lambda t: (t >= TAU_C_MIN) & (t <= TAU_C_MAX)
    n_other = int(in_range(other).sum() - in_range(tau_c_all).sum())
    print(f"  tau_c source: '{TAU_C_SOURCE}' "
          f"(the other choice would keep {n_other:+d} profiles)")

    return out


def load_profiles_per_profile(mat_dir: Path) -> dict:
    """Load the legacy data set: one .mat file per profile.

    VOCALS-REx files there have no 'alpha_param', so their mean alpha is NaN and
    they contribute nothing to the alpha part of the joint model.
    """
    mat_files = sorted(f for f in mat_dir.glob('*.mat')
                       if not f.name.startswith('._'))
    print(f'Found {len(mat_files)} .mat files in {mat_dir}')

    if not mat_files:
        raise SystemExit(f'No .mat files found in {mat_dir}.')

    out = {k: [] for k in ('re', 'lwc', 'z', 'mean_alpha', 'campaign',
                           'z_top', 'z_base', 'tau_c', 'lwp', 'n_levels',
                           'T', 'vapor', 'source_index')}
    # OUTLIERS cannot be matched here (the per-profile files carry no profile_idx)
    skips = dict(low_tau=0, high_tau=0, outlier=0, dup=0, missing=0, no_era5=0)
    seen_fingerprints = set()
    era5_pressure = None

    for path in mat_files:
        d = scipy.io.loadmat(path, squeeze_me=True)

        if not {'re', 'z', 'tau', 'lwc'}.issubset(d.keys()):
            skips['missing'] += 1
            continue

        era5_fields = load_era5_fields(d)
        if era5_fields is None:
            skips['no_era5'] += 1
            continue
        T_prof, vapor_prof, P_grid = era5_fields
        if era5_pressure is None:
            era5_pressure = P_grid                               # snapshot once

        re    = np.asarray(d['re'][()],  dtype=np.float64)
        lwc   = np.asarray(d['lwc'][()], dtype=np.float64)
        z     = np.asarray(d['z'][()],   dtype=np.float64)
        tau_c = float(np.asarray(d['tau'][()]).max())

        has_alpha = 'alpha_param' in d.keys()
        if has_alpha:
            alpha = np.asarray(d['alpha_param'][()], dtype=np.float64)

        if tau_c < TAU_C_MIN:
            skips['low_tau'] += 1
            continue
        if tau_c > TAU_C_MAX:
            skips['high_tau'] += 1
            continue

        fp = tuple(np.round(re[:5], 4))
        if fp in seen_fingerprints:
            skips['dup'] += 1
            continue
        seen_fingerprints.add(fp)

        # Defensive: trim to common length if any of re / lwc / alpha drift in size
        if has_alpha:
            n_use = min(len(re), len(lwc), len(z), len(alpha))
            alpha = alpha[:n_use]
        else:
            n_use = min(len(re), len(lwc), len(z))
        re, lwc, z = re[:n_use], lwc[:n_use], z[:n_use]

        out['re'].append(re)
        out['lwc'].append(lwc)
        out['z'].append(z)
        out['mean_alpha'].append(
            trapz_vertical_mean(alpha, z) if has_alpha else np.nan)
        out['campaign'].append('oracles' if has_alpha else 'vocals-rex')
        out['z_top'].append(float(z[0]))
        out['z_base'].append(float(z[-1]))
        out['tau_c'].append(tau_c)
        out['lwp'].append(compute_lwp(lwc, z))
        out['n_levels'].append(n_use)
        out['T'].append(T_prof)
        out['vapor'].append(vapor_prof)
        out['source_index'].append(mat_files.index(path))   # file position, for provenance

    out['era5_pressure'] = era5_pressure
    out['skips'] = skips
    return out


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


# ── Load the in-situ + ERA5 profiles with TAU_C_MIN ≤ tau_c ≤ TAU_C_MAX ────────
input_layout, input_path = resolve_input_path(MAT_DIR)
print(f'Input layout: {input_layout}  →  {input_path}')

if input_layout == 'consolidated':
    data = load_profiles_consolidated(input_path)
else:
    data = load_profiles_per_profile(input_path)

profiles_re   = data['re']          # raw arrays (top → base)
profiles_lwc  = data['lwc']
altitudes_raw = data['z']
mean_alpha    = data['mean_alpha']  # per-profile vertical-mean alpha
campaign_each = data['campaign']    # 'oracles' or 'vocals-rex' per profile
z_top_each    = data['z_top']
z_base_each   = data['z_base']
tau_c_each    = data['tau_c']
lwp_each      = data['lwp']
n_levels_each = data['n_levels']
T_each        = data['T']           # (37,) ERA5 temperature, surface → TOA
vapor_each    = data['vapor']       # (37,) ERA5 vapor_concentration, surface → TOA
era5_pressure = data['era5_pressure']

n_skip_low_tau  = data['skips']['low_tau']
n_skip_high_tau = data['skips']['high_tau']
n_skip_outlier  = data['skips']['outlier']
source_index    = np.asarray(data['source_index'], dtype=np.int32)   # (n_train,) input rows used
n_skip_dup      = data['skips']['dup']
n_skip_missing  = data['skips']['missing']
n_skip_no_era5  = data['skips']['no_era5']

if not profiles_re:
    raise SystemExit(f'No usable profiles found in {input_path}')

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

campaign_each = np.asarray(campaign_each)
n_oracles = int((campaign_each == 'oracles').sum())
n_vocals  = int((campaign_each == 'vocals-rex').sum())

print(f'  skipped: {n_skip_low_tau} tau_c<{TAU_C_MIN}, {n_skip_high_tau} tau_c>{TAU_C_MAX}, '
      f'{n_skip_outlier} OUTLIERS, {n_skip_dup} duplicates, {n_skip_missing} missing keys, '
      f'{n_skip_no_era5} no era5')
print(f'  kept   : {n_train} unique profiles  '
      f'(ORACLES = {n_oracles}, VOCALS-REx = {n_vocals})')
print(f'  n_lev   : min={n_levels_each.min()}, max={n_levels_each.max()}, '
      f'mean={n_levels_each.mean():.1f}')
print(f'  tau_c   : [{tau_c_each.min():.2f}, {tau_c_each.max():.2f}], '
      f'median={np.median(tau_c_each):.2f}')
print(f'  thick km: [{thickness_each.min():.3f}, {thickness_each.max():.3f}]')

# The consolidated file carries alpha for both campaigns. In the legacy layout
# only the ORACLES files store alpha_param, so VOCALS-REx alphas are NaN.
mask_alpha = ~np.isnan(mean_alpha)
alpha_obs  = mean_alpha[mask_alpha]
alpha_note = '' if mask_alpha.all() else ' (ORACLES only)'

# Which campaigns actually contribute an alpha, for labelling prints and figures.
ALPHA_SOURCE_LABEL = ' + '.join(sorted(set(campaign_each[mask_alpha])))
print(f'  alpha   : measured for {mask_alpha.sum()}/{n_train} profiles'
      f'{alpha_note}; '
      f'[{alpha_obs.min():.2f}, {alpha_obs.max():.2f}], '
      f'median={np.median(alpha_obs):.2f}')
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


# ── Joint MVN over the cross-campaign feature ─────────────────────────────────
# log(mean_alpha) joins the joint feature vector whenever every kept profile
# carries an alpha, so the MVN's cross-covariance captures the coupling between
# the gamma shape parameter and the r_e / LWC / thickness structure. The legacy
# per-profile layout stores alpha for ORACLES only; rather than impute a quarter
# of the rows there, alpha falls back to an independent 1-D log-normal fit on
# the profiles that do have one.
ALPHA_IN_MVN = bool(mask_alpha.all())

log_thickness = np.log(thickness_each)

# 1-D log-normal statistics for alpha. These set the sampling clip bounds in
# both modes, and are the sampling distribution itself when ALPHA_IN_MVN is False.
log_alpha_obs    = np.log(np.maximum(alpha_obs, 1e-3))
mu_alpha_log     = float(log_alpha_obs.mean())
sigma_alpha_log  = float(log_alpha_obs.std(ddof=1)) if len(log_alpha_obs) > 1 else 0.0
alpha_obs_min, alpha_obs_max = float(alpha_obs.min()), float(alpha_obs.max())

feature_blocks = [scores_re, scores_lwc, scores_T, scores_vapor]
feature_names  = [f'K_re={K_re}', f'K_lwc={K_lwc}', f'K_T={K_T}', f'K_vapor={K_vapor}']

if ALPHA_IN_MVN:
    feature_blocks.append(np.log(np.maximum(mean_alpha, 1e-3))[:, None])
    feature_names.append('log(alpha)')

feature_blocks += [z_base_each[:, None], log_thickness[:, None]]
feature_names  += ['z_base', 'log(thickness)']

features = np.hstack(feature_blocks)
mu_f  = features.mean(axis=0)
cov_f = np.cov(features, rowvar=False)
print(f'\nJoint MVN feature dimension: {features.shape[1]} '
      f'({" + ".join(feature_names)})')

if ALPHA_IN_MVN:
    print(f'Alpha ({ALPHA_SOURCE_LABEL}) is in the joint MVN; '
          f'clip [{alpha_obs_min:.2f}, {alpha_obs_max:.2f}]')
else:
    print(f'Alpha ({ALPHA_SOURCE_LABEL}) sampled independently: '
          f'log-N(mu={mu_alpha_log:.3f}, sigma={sigma_alpha_log:.3f})  '
          f'clip [{alpha_obs_min:.2f}, {alpha_obs_max:.2f}]')


# ── Sample N synthetic profiles (with rejection on tau_c) ─────────────────────
slc_re    = slice(0,                                 K_re)
slc_lwc   = slice(K_re,                              K_re + K_lwc)
slc_T     = slice(K_re + K_lwc,                      K_re + K_lwc + K_T)
slc_vapor = slice(K_re + K_lwc + K_T,                K_re + K_lwc + K_T + K_vapor)
idx_alpha = K_re + K_lwc + K_T + K_vapor if ALPHA_IN_MVN else None
idx_zbase = K_re + K_lwc + K_T + K_vapor + (1 if ALPHA_IN_MVN else 0)
idx_thick = idx_zbase + 1
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
    new_z_base       = samples_batch[:, idx_zbase]
    new_log_thick    = samples_batch[:, idx_thick]

    new_z_base    = np.clip(new_z_base,
                            z_base_each.min(), z_base_each.max())
    new_thickness = np.clip(np.exp(new_log_thick),
                            thickness_each.min(), thickness_each.max())
    new_z_top     = new_z_base + new_thickness

    # Alpha comes out of the joint MVN when every training profile carried one,
    # so it keeps its correlation with the r_e / LWC / thickness features. With
    # the legacy layout (alpha for ORACLES only) it is drawn from the standalone
    # 1-D log-normal instead. Either way it is clipped to the observed envelope
    # so we never extrapolate alpha past the measured range.
    if ALPHA_IN_MVN:
        new_log_alpha = samples_batch[:, idx_alpha]
    else:
        new_log_alpha = rng.normal(mu_alpha_log, sigma_alpha_log, size=n)
    new_alpha = np.clip(np.exp(new_log_alpha), alpha_obs_min, alpha_obs_max)

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
        # Mie τ with the table nearest this draw's α (same α libRadtran gets)
        tau_c[i] = compute_tau_c(re_out[i], lwc_out[i], z_out[i], new_alpha[i])

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


# ── LWC ceiling at the output heights ─────────────────────────────────────────
# Every synthetic cloud has its N_FIXED_LEVELS levels at the same normalised
# heights u = 1 (top) … 0 (base), so one ceiling vector serves every draw.
# lwc_common is the in-situ LWC on the 80-level grid, ordered base → top.
u_out_t2b      = np.linspace(1.0, 0.0, N_FIXED_LEVELS)          # (L_out,) top → base
# percentile across the training profiles at each of the L_COMMON heights
lwc_pct_common = np.percentile(lwc_common, LWC_CEILING_PERCENTILE, axis=0)  # (L_COMMON,) g/m³, base → top
lwc_ceiling    = LWC_MAX_FACTOR * np.interp(u_out_t2b, u_grid_b2t, lwc_pct_common)
if LWC_MAX_REJECTION:
    # How many TRAINING profiles would themselves fail this ceiling, judged
    # the same way (their common-grid LWC read at the output heights)?
    insitu_at_out = np.array([np.interp(u_out_t2b, u_grid_b2t, row) for row in lwc_common])
    n_insitu_fail = int((insitu_at_out > lwc_ceiling[None, :]).any(axis=1).sum())
    print(f'\nLWC ceiling (in-situ {LWC_CEILING_PERCENTILE:g}th percentile × '
          f'{LWC_MAX_FACTOR:g}) at output levels, top → base [g/m³]: '
          + ', '.join(f'{v:.3f}' for v in lwc_ceiling))
    print(f'  {n_insitu_fail} of {n_train} in-situ profiles exceed it at one or more levels')

rng = np.random.default_rng(RANDOM_SEED)
print(f'\nSampling {N_SAMPLES} synthetic profiles with rejection on '
      f'{TAU_C_MIN} <= tau_c <= {TAU_C_MAX}'
      + (f' and LWC <= in-situ {LWC_CEILING_PERCENTILE:g}th percentile at each level'
         if LWC_MAX_REJECTION else '')
      + '...')

accepted = {k: [] for k in
            ('re', 'lwc', 'z', 'alpha', 'T', 'vapor',
             'z_top', 'z_base', 'thickness', 'mean_alpha',
             'tau_c', 'lwp')}
n_accepted   = 0
n_drawn_total = 0
n_passed_total = 0             # draws inside the tau_c limits, used or not
n_rejected_low_total  = 0      # draws with tau_c < TAU_C_MIN
n_rejected_high_total = 0      # draws with tau_c > TAU_C_MAX
n_rejected_wet_total  = 0      # draws with LWC above the ceiling at some level
                               # (can overlap with the tau_c rejections)

for batch_idx in range(REJECTION_MAX_BATCHES):
    n_need = N_SAMPLES - n_accepted
    if n_need <= 0:
        break
    n_draw = max(int(REJECTION_BATCH_OVERSAMPLE * n_need), 2000)
    samples_batch = rng.multivariate_normal(mu_f, cov_f, size=n_draw)
    n_drawn_total += n_draw

    out = reconstruct_batch(samples_batch)
    too_thin  = out['tau_c'] < TAU_C_MIN
    too_thick = out['tau_c'] > TAU_C_MAX
    if LWC_MAX_REJECTION:
        too_wet = (out['lwc'] > lwc_ceiling[None, :]).any(axis=1)   # (n_draw,)
    else:
        too_wet = np.zeros(n_draw, dtype=bool)
    n_rejected_low_total  += int(too_thin.sum())
    n_rejected_high_total += int(too_thick.sum())
    n_rejected_wet_total  += int(too_wet.sum())
    keep = ~too_thin & ~too_thick & ~too_wet
    n_passed_total += int(keep.sum())
    take = min(int(keep.sum()), n_need)
    if take == 0:
        continue
    keep_idx = np.flatnonzero(keep)[:take]
    for k in accepted:
        accepted[k].append(out[k][keep_idx])
    n_accepted += take
    print(f'  batch {batch_idx + 1:2d}: drew {n_draw}, '
          f'{int(keep.sum())} passed ({TAU_C_MIN}≤tau≤{TAU_C_MAX}'
          f'{", LWC ceiling" if LWC_MAX_REJECTION else ""}), '
          f'cumulative {n_accepted}/{N_SAMPLES}')

if n_accepted < N_SAMPLES:
    raise RuntimeError(
        f'Only accumulated {n_accepted}/{N_SAMPLES} samples with '
        f'{TAU_C_MIN} <= tau_c <= {TAU_C_MAX} (and the LWC ceiling) after '
        f'{REJECTION_MAX_BATCHES} batches. Increase REJECTION_MAX_BATCHES or relax '
        f'TAU_C_MIN / TAU_C_MAX / LWC_MAX_FACTOR.')

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

# Acceptance rate = fraction of ALL draws that pass the tau_c limits. Each batch
# draws at least 2000, so more can pass than are needed; dividing kept by drawn
# would mix that oversampling into the rate and understate it.
acceptance_rate = n_passed_total / n_drawn_total
print(f'  done. acceptance rate = {acceptance_rate:.3f} '
      f'({n_passed_total} of {n_drawn_total} draws passed; kept the first {n_accepted})')
# If a large share of draws exceeds the cap, the MVN tail is heavy and the cut
# distorts the joint LWC-thickness distribution near TAU_C_MAX
print(f'  rejected: {n_rejected_low_total} below TAU_C_MIN '
      f'({n_rejected_low_total / n_drawn_total:.1%} of draws), '
      f'{n_rejected_high_total} above TAU_C_MAX '
      f'({n_rejected_high_total / n_drawn_total:.1%} of draws), '
      f'{n_rejected_wet_total} above the LWC ceiling '
      f'({n_rejected_wet_total / n_drawn_total:.1%} of draws; overlaps allowed)')

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
    TAU_C_MAX        = np.float32(TAU_C_MAX),
    n_train_profiles = np.int32(n_train),
    # Which in-situ profiles trained the MVN: their rows in the input file, the
    # file itself, and the OUTLIERS that were left out
    train_source_index = source_index,
    input_path       = np.array(str(input_path)),
    outliers         = np.array([f'{c} {t} {k}' for c, t, k in OUTLIERS]),
    # How tau_c was computed (Mie Q̄_ext(r_e, α), trapezoid over the levels)
    tau_c_method     = np.array('mie_qbar_trapz'),
    tau_wavelength_um = np.float32(TAU_WAVELENGTH_UM),
    mie_alpha_table  = nearest_mie_alpha(new_alpha).astype(np.int32),   # (N,) α of the table used
    # LWC ceiling used for rejection (13 Nov 2026); see LWC_MAX_REJECTION
    lwc_max_rejection = np.bool_(LWC_MAX_REJECTION),
    lwc_max_factor    = np.float32(LWC_MAX_FACTOR),
    lwc_ceiling_percentile = np.float32(LWC_CEILING_PERCENTILE),
    lwc_ceiling       = lwc_ceiling.astype(np.float32),     # (N_FIXED_LEVELS,) g/m³, top → base
    acceptance_rate   = np.float32(acceptance_rate),
    mie_table_dir    = np.array(str(MIE_TABLE_DIR)),
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
bins = np.linspace(min(alpha_obs.min(), new_alpha.min()),
                   max(alpha_obs.max(), new_alpha.max()), 40)
ax.hist(alpha_obs, bins=bins, density=True, alpha=0.55,
        color='firebrick',
        label=f'In-situ ({ALPHA_SOURCE_LABEL}, n={alpha_obs.size})')
ax.hist(new_alpha, bins=bins, density=True, alpha=0.55,
        color='steelblue', label=f'Synthetic (n={N_SAMPLES})')
ax.set_xlabel(r'Vertical-mean $\alpha$ (libRadtran shape)')
ax.set_ylabel('Density')
ax.set_title(r'Mean $\alpha_{\rm param}$')
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(
    f'Joint MVN generator: ORACLES, '
    f'${TAU_C_MIN:g} \\leq \\tau_c \\leq {TAU_C_MAX:g}$,  N$_\\text{{train}}$={n_train},  '
    f'L={L_COMMON},  K$_{{re}}$={K_re},  K$_{{lwc}}$={K_lwc},  '
    f'N$_\\text{{sample}}$={N_SAMPLES}',
    y=1.005,
)
fig.tight_layout()
fig_path = _versioned_path(FIG_PATH)
fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Diagnostic figure → {fig_path}  (dpi={FIGURE_DPI})')


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
fig_atmos_path = _versioned_path(FIG_ATMOS_PATH)
fig_a.savefig(fig_atmos_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Atmosphere figure → {fig_atmos_path}  (dpi={FIGURE_DPI})')


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
fig_examples_path = _versioned_path(FIG_EXAMPLES_PATH)
fig_ex.savefig(fig_examples_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Examples figure   → {fig_examples_path}  (dpi={FIGURE_DPI})')

plt.show()
