"""
convert_matFiles_to_HDF.py

Converts individual libRadtran .MAT simulation output files into a single
HDF5 training file compatible with LibRadtranDataset in data.py.

Structure of each .MAT file
----------------------------
  Refl_model_allStateVectors : (636, 128) float64
      Raw TOA reflectance without noise. Noise is added programmatically:
      - HySICS: 0.3% Gaussian noise
      - EMIT: 4.0% Gaussian noise

  changing_variables_allStateVectors : (636 * 128, N_cols) float64
      One row per (wavelength by geometry) RT simulation.
      N_cols is 6 or 7 depending on which .mat file version generated the data.
      Column  0     = VZA (deg), Column 1 = VAZ (deg), Column 2 = SAZ (deg).
      Column -3     = band lower bound (nm)   (3rd from last)
      Column -2     = band upper bound (nm)   (2nd from last)
      Column -1     = band index (1-based)    (last column)
      Some 7-column files have an unused zero placeholder at column 3.
      Band center wavelength = mean(col -3, col -2).
      Extract geometry for each of the 128 viewing configs with [::636, :3].

  re  : (n_insitu,) float64   — in-situ r_e profile, cloud top → base (μm)
  z   : (n_insitu,) float64   — altitude, cloud top → base (km, decreasing)
  tau : (n_insitu,) float64   — cumulative optical depth; tau[-1] = tau_c
  lwc : (n_insitu,) float64   — liquid water content (not used here)

SZA varies per .mat file and is parsed from the filename (pattern: _sza_<value>_).

Output HDF5 structure (matches LibRadtranDataset in data.py)
--------------------------------------------------------------
  /reflectances_hysics  (n_total, 636)       — noisy TOA reflectance (0.3% noise)
  /reflectances_emit    (n_total, 636)       — noisy TOA reflectance (4% noise)
  /reflectances_uncertainty_hysics (n_total, 636) — per-channel 1-sigma uncertainty
  /reflectances_uncertainty_emit   (n_total, 636) — per-channel 1-sigma uncertainty
  /profiles           (n_total, N_LEVELS)  — r_e interpolated to fixed grid (top→base)
  /profiles_raw       (n_total, max_raw)    — raw in-situ r_e (top→base), NaN-padded
  /profiles_raw_z     (n_total, max_raw)    — raw altitude (km, top→base), NaN-padded
  /profiles_raw_tau   (n_total, max_raw)    — raw optical depth (top=0 → base=τ_c),
                                              NaN-padded.  Same ordering and valid
                                              length (per /profile_n_levels) as
                                              profiles_raw and profiles_raw_z.
  /profile_n_levels   (n_total,)            — valid entries in each raw row
  /tau_c              (n_total,)            — scalar column optical depth (max of tau profile)
  /vza           (n_total,)           — viewing zenith angle (deg)
  /vaz           (n_total,)           — viewing azimuth angle (deg)
  /saz           (n_total,)           — solar azimuth angle (deg)
  /sza           (n_total,)           — solar zenith angle (deg), parsed from filename
  /wavelengths   (636,)               — HySICS band center wavelengths (nm)

  Attributes on the root group:
    n_mat_files, n_geometries,
    re_global_min, re_global_max, tau_global_min, tau_global_max

n_total = n_mat_files by 128

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import re
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MAT_DIR  = Path('/Volumes/My Passport/neural_network_training_data/combined_vocals_oracles_training_data_17_April_2026/')
OUT_PATH = Path('/Volumes/My Passport/neural_network_training_data/'
                'combined_vocals_oracles_training_data_50-evenZ-levels_deSpiked_28_April_2026.h5')

N_LEVELS      = 50     # target vertical levels in output profile
N_GEOMETRIES  = 128   # viewing geometry configs per .mat file (8 VZA × 4 VAZ × 4 SAZ)
N_WAVELENGTHS = 636   # HySICS spectral channels

# ------------------------------------------------------------------
# Training-target sampling mode
# ------------------------------------------------------------------
# 'even_z'        : N_LEVELS evenly spaced in geometric altitude (km).
#                   Only re_raw and z_raw are used by the interpolator.
# 'tau_weighted'  : N_LEVELS spaced non-uniformly in cumulative optical
#                   depth, with TAU_TOP_FRAC of the levels placed in the
#                   top TAU_SPLIT fraction of optical depth (denser near
#                   cloud top, where shortwave reflectance is most
#                   sensitive).  Requires a valid tau_raw vector.
SAMPLING_MODE   = 'even_z'   # 'even_z' or 'tau_weighted'
TAU_TOP_FRAC    = 0.6              # fraction of N_LEVELS placed above TAU_SPLIT
TAU_SPLIT       = 0.5              # normalized tau (0=top, 1=base) at the split

# Minimum physically-plausible cloud optical depth.
# Files with tau_c below this value are skipped — a near-zero tau_c indicates
# either a clear-sky profile or a corrupted measurement, neither of which
# should be in the cloud retrieval training set.
TAU_C_MIN = 3.0

# Keys that must be present in every .mat file to be included
# Now we generate noise from the raw reflectances, so we only need raw data
REQUIRED_KEYS = {
    'Refl_model_allStateVectors',
    'changing_variables_allStateVectors',
    're', 'z', 'tau', 'era5',
}

# Noise levels for each instrument (fraction of signal, applied as Gaussian)
NOISE_HYSICS = 0.003  # 0.3% Gaussian noise
NOISE_EMIT   = 0.02   # 4.0% Gaussian noise
NOISE_FM     = 0      # 0.7% Forward model noise assumption 

_SZA_RE = re.compile(r'_sza_(\d+(?:\.\d+)?)_')

def parse_sza_from_filename(filename: str) -> float:
    """Extract the SZA value from a filename containing the pattern _sza_<value>_."""
    m = _SZA_RE.search(filename)
    if m is None:
        raise ValueError(f'Cannot parse SZA from filename: {filename}')
    return float(m.group(1))


def add_gaussian_noise(refl: np.ndarray, noise_level: float) -> tuple:
    """
    Add Gaussian noise to reflectance data.

    Parameters
    ----------
    refl : (n_wavelengths, n_geometries) float64
        Raw reflectance without noise.
    noise_level : float
        Noise as a fraction of signal (e.g., 0.003 for 0.3%).

    Returns
    -------
    refl_noisy : (n_wavelengths, n_geometries) float32
        Reflectance with added noise.
    uncert : (n_wavelengths, n_geometries) float32
        Per-channel 1-sigma measurement uncertainty.
    """
    # Generate random noise: same shape, mean=0, std=1
    random_noise = np.random.randn(*refl.shape)

    # Add noise: noisy = raw + (noise_level * raw) * randn
    refl_noisy = refl + (noise_level * refl) * random_noise

    # Uncertainty is the standard deviation of the noise: sigma = noise_level * raw
    uncert = np.abs(noise_level * refl)

    return refl_noisy.astype(np.float32), uncert.astype(np.float32)


# ============================================================
# Helper: targeted single-profile despike
# ============================================================
#
# Background.  The K=5 50-level profile-only sweep (April 2026) showed a
# systematic L13 (idx 12) RMSE spike of ~4 μm in every top-10 config,
# while neighboring levels were ~1.0–1.6 μm.  Investigation (see the
# session notes / sweep results discussion) traced this to ONE bad in-situ
# data point in profile pid 204:
#
#     raw k=7 (z = 1.040 km):  r_e = 20.90 μm
#     local 3-pt median:        7.43 μm
#     deviation:              +13.5 μm    ← the offender
#
# All neighboring raw points in pid 204 are reasonable (~7–10 μm), and the
# rest of the 290-profile catalog is clean (median worst-spike per profile
# is 0.07 μm; the next-worst profile has a 3.0 μm spike, well below the
# threshold here).  So we apply the despike *only* to pid 204 to keep the
# rest of the dataset bit-identical to the previous HDF5.
#
# Fingerprint format matches data.compute_profile_ids():
#   tuple(np.round(re_raw[:5], 4))
#
# If we later decide to despike the whole catalog, change the conditional
# below to `if True:` and pick a threshold that's conservative enough to
# leave the borderline pid 104/177/174 spikes (~2.5–3 μm) alone.

PID_204_FINGERPRINT = (9.5434, 9.1513, 9.5306, 9.5567, 7.7079)


def despike_re(re_raw: np.ndarray,
               threshold_um: float = 5.0,
               window: int = 3) -> np.ndarray:
    """
    Replace any in-situ r_e sample whose deviation from the local
    `window`-point median exceeds `threshold_um` with that median.

    Conservative single-pass median filter — only patches obvious
    instrument/classification spikes.  Clean profiles pass through
    bit-identical (because no point exceeds the threshold).

    Parameters
    ----------
    re_raw       : (n,) float64 — raw effective-radius profile (μm),
                                  index 0 = top, index -1 = base
    threshold_um : float        — μm deviation that flags a sample as a spike
    window       : odd int      — width of the local median window

    Returns
    -------
    out : (n,) float64 — despiked copy
    """
    if window % 2 != 1:
        raise ValueError("despike_re: window must be odd")
    out = re_raw.copy()
    half = window // 2
    n = len(re_raw)
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        med = np.median(re_raw[lo:hi])
        if abs(re_raw[k] - med) > threshold_um:
            out[k] = med
    return out


# ============================================================
# Helper: profile interpolation
# ============================================================

def interpolate_profile(re_raw: np.ndarray,
                        z_raw:  np.ndarray,
                        n_levels_out: int) -> np.ndarray:
    """
    Interpolate an in-situ r_e profile to a fixed number of evenly-spaced
    levels using actual altitude coordinates.

    Parameters
    ----------
    re_raw : (n_insitu,)
        Effective radius, ordered cloud top → cloud base (index 0 = top).
    z_raw : (n_insitu,)
        Altitude in km, same ordering (decreasing values: top has highest z).
    n_levels_out : int
        Number of output levels.

    Returns
    -------
    (n_levels_out,) float32
        Interpolated r_e, index 0 = cloud top, index -1 = cloud base.
    """
    # np.interp requires ascending x-coordinates; flip so z goes base → top
    z_asc  = z_raw[::-1]       # now increasing: cloud base to top
    re_asc = re_raw[::-1]

    # Evenly-spaced target grid from cloud base to cloud top (in km)
    z_target_asc = np.linspace(z_asc[0], z_asc[-1], n_levels_out)
    re_interp_asc = np.interp(z_target_asc, z_asc, re_asc)

    # Flip back so index 0 = cloud top
    return re_interp_asc[::-1].astype(np.float32)


def interpolate_profile_tau_weighted(
    re_raw:   np.ndarray,
    tau_raw:  np.ndarray,
    z_raw:    np.ndarray | None = None,
    n_top:    int   = 6,
    n_bot:    int   = 4,
    split_tau: float = 0.4,
) -> tuple:
    """
    Interpolate an in-situ r_e profile onto a non-uniformly spaced
    optical-depth grid that places more levels near cloud top, where
    reflected radiance sensitivity is highest.

    The normalized optical depth τ̃ = τ / τ_c runs from 0 (cloud top)
    to 1 (cloud base).  n_top points are placed evenly over [0, split_tau]
    and n_bot points over (split_tau, 1], giving n_top + n_bot levels total.

    Default parameters (n_top=6, n_bot=4, split_tau=0.4) produce a 10-level
    grid that matches N_LEVELS while concentrating half the levels in the
    top 40 % of optical depth.

    Parameters
    ----------
    re_raw    : (n,) float  — effective radius, cloud top → base (μm)
    tau_raw   : (n,) float  — cumulative optical depth; may be ascending
                              (descending flight) or descending (ascending
                              flight) — handled automatically via min/max norm
    z_raw     : (n,) float or None  — altitude in km, cloud top → base
                              (decreasing).  When supplied, z at each
                              sampled τ̃ point is also returned.
    n_top     : int   — levels in [0, split_tau]       (default 6)
    n_bot     : int   — levels in (split_tau, 1]        (default 4)
    split_tau : float — normalized τ split point        (default 0.4)

    Returns
    -------
    re_interp : (n_top + n_bot,) float32
        Effective radius at each sampled level (cloud top → base order).
    tau_grid  : (n_top + n_bot,) float64
        Normalized optical depth τ̃ ∈ [0, 1] at each level.
        Use as the y-axis when plotting in τ-space (invert axis so 0 is at top).
    z_interp  : (n_top + n_bot,) float64  or  None
        Altitude in km at each sampled τ̃ level.
        Use as the y-axis when plotting in z-space.
    """
    # Coerce inputs to 1-D float64 arrays so indexing always works.
    tau_raw = np.atleast_1d(np.asarray(tau_raw, dtype=np.float64))
    re_raw  = np.atleast_1d(np.asarray(re_raw,  dtype=np.float64))

    # Normalize to [0, 1] using global min/max so the result is correct
    # regardless of whether tau was stored cloud-top→base (ascending) or
    # cloud-base→top (descending), and robust to slight non-monotonicities
    # that can appear in processed data.
    tau_min, tau_max = tau_raw.min(), tau_raw.max()
    tau_norm = (tau_raw - tau_min) / (tau_max - tau_min)

    # Sort everything by ascending τ̃ so np.interp receives a valid xp array.
    # For profiles already in top→base (ascending τ) order this is a no-op.
    order      = np.argsort(tau_norm)
    tau_sorted = tau_norm[order]
    re_sorted  = re_raw[order]
    z_sorted   = np.asarray(z_raw, dtype=np.float64)[order] if z_raw is not None else None

    # Build the non-uniform τ̃ sampling grid:
    #   n_top evenly-spaced points over [0, split_tau]  — denser near cloud top
    #   n_bot evenly-spaced points over (split_tau, 1]  — sparser toward base
    tau_top  = np.linspace(0.0, split_tau, n_top)
    tau_bot  = np.linspace(split_tau, 1.0, n_bot + 1)[1:]   # drop shared endpoint
    tau_grid = np.concatenate([tau_top, tau_bot])            # (n_top + n_bot,)

    re_interp = np.interp(tau_grid, tau_sorted, re_sorted).astype(np.float32)
    z_interp  = np.interp(tau_grid, tau_sorted, z_sorted) if z_sorted is not None else None

    return re_interp, tau_grid, z_interp


# ============================================================
# ERA5 water vapour extraction
# ============================================================

def _era5_datprofiles(d: dict) -> tuple:
    """
    Return the (gp_height_m, vapor_conc) arrays from a loaded .mat dict.

    Access path (scipy.io structured arrays):
        d['era5'].item()[6]  →  datProfiles void array
        .item()              →  tuple with dtype fields:
            0: GP_height          (37,) float  metres, surface→TOA
            1: T                  (37,) float  K
            2: p                  (37,) uint16 hPa  (1000 → 1)
            3: q                  (37,) float  specific humidity (kg/kg)
            4: vapor_concentration (37,) float  molecules/cm³
            5: vapor_massDensity  (37,) float  kg/m³
    """
    dp = d['era5'].item()[6].item()
    gph = np.asarray(dp[0], dtype=np.float64)   # metres
    vap = np.asarray(dp[4], dtype=np.float64)   # molecules/cm³
    return gph, vap


def extract_era5_vapor(d: dict, z_raw: np.ndarray) -> tuple:
    """
    Compute above-cloud and in-cloud water vapour column densities.

    Cloud top and bottom are derived from the in-situ altitude array ``z_raw``
    stored in each .mat file as max(z_raw) and min(z_raw) respectively.
    Using max/min rather than index 0/-1 is robust to files where the altitude
    vector is not stored in a guaranteed top-to-bottom order.

    Because ``z_raw`` is the actual flight-segment altitude measured by the
    aircraft, these are the exact boundaries used in the libRadtran RT
    calculations.  The cloud-top/base altitudes are interpolated onto the
    ERA5 geopotential-height grid (37 standard pressure levels, surface→TOA)
    so the trapezoidal integrals use physically consistent altitude nodes.

    Integration:
        wv_above_cloud = ∫[z_top → TOA]    n(z) dz     (molec/cm²)
        wv_in_cloud    = ∫[z_base → z_top] n(z) dz     (molec/cm²)
    where n(z) is vapor_concentration (molec/cm³) and dz is in cm.

    Parameters
    ----------
    d : dict
        Output of scipy.io.loadmat(..., squeeze_me=True) for one .mat file.
    z_raw : (n,) float
        In-situ cloud altitude in km, ordered cloud top → cloud base
        (z_raw[0] is the highest altitude).

    Returns
    -------
    wv_above_cloud       : float   — column above cloud top  (molec/cm²)
    wv_in_cloud          : float   — column within cloud     (molec/cm²)
    vapor_conc_profile   : (37,) float32 — full ERA5 profile (molec/cm³)
    """
    gph, vap = _era5_datprofiles(d)

    # Cloud boundaries: km → metres.
    # Use max/min rather than index 0/-1 — the altitude array is not guaranteed
    # to be stored in any particular order across all .mat files.
    z_top_m  = float(z_raw.max()) * 1000.0
    z_base_m = float(z_raw.min()) * 1000.0

    # Guard: ERA5 must span at least from cloud base to above cloud top.
    # In practice this is always satisfied for low marine clouds.
    z_top_m  = np.clip(z_top_m,  gph[0], gph[-1])
    z_base_m = np.clip(z_base_m, gph[0], z_top_m)

    # Interpolate vapour concentration at the exact cloud boundaries.
    vap_at_top  = float(np.interp(z_top_m,  gph, vap))
    vap_at_base = float(np.interp(z_base_m, gph, vap))

    # ---- Above-cloud column ----------------------------------------
    # ERA5 levels strictly above cloud top, prepended with cloud-top node.
    above_mask = gph > z_top_m
    gph_above  = np.concatenate([[z_top_m],    gph[above_mask]])
    vap_above  = np.concatenate([[vap_at_top], vap[above_mask]])
    # Integrate: metres → cm (×100), result in molec/cm²
    wv_above = float(np.trapezoid(vap_above, gph_above * 100.0))

    # ---- In-cloud column -------------------------------------------
    # ERA5 levels strictly between cloud base and cloud top, bracketed by
    # the interpolated boundary values.  The sort guard handles the rare
    # case where cloud base coincides exactly with an ERA5 level.
    in_mask = (gph > z_base_m) & (gph < z_top_m)
    gph_in  = np.concatenate([[z_base_m],    gph[in_mask],  [z_top_m]])
    vap_in  = np.concatenate([[vap_at_base], vap[in_mask],  [vap_at_top]])
    sort_i  = np.argsort(gph_in)
    wv_in   = float(np.trapezoid(vap_in[sort_i], gph_in[sort_i] * 100.0))

    return wv_above, wv_in, vap.astype(np.float32)


# ============================================================
# Collect .mat files (exclude macOS metadata files starting with ._)
# ============================================================

mat_files = sorted(f for f in MAT_DIR.glob('*.mat')
                   if not f.name.startswith('._'))
n_files  = len(mat_files)
n_total  = n_files * N_GEOMETRIES
print(f'Found {n_files} .mat files → {n_total} total training samples')


# ============================================================
# Pass 1: scan r_e, tau_c bounds and max profile length
# Needed to:
#   (a) set RE_MAX and TAU_MAX in data.py and models.py
#   (b) pre-allocate the padded raw-profile dataset in HDF5
# ============================================================

print('\nPass 1: scanning profile bounds and lengths across all files...')
re_global_min   =  np.inf
re_global_max   = -np.inf
tau_global_min  =  np.inf
tau_global_max  = -np.inf
max_raw_levels  = 0
tau_c_all       = []
valid_mat_files = []
skipped_files   = []

for i, path in enumerate(mat_files):
    d = scipy.io.loadmat(path, squeeze_me=True)

    # Validate that all required keys are present before doing anything else
    missing = REQUIRED_KEYS - set(d.keys())
    if missing:
        print(f'\n  SKIPPING {path.name} — missing keys: {missing}')
        skipped_files.append((path, missing))
        continue

    # re, z, tau are stored as 1x1 cell arrays; squeeze_me=True gives a 0-d
    # object array, so use [()] to extract the inner numpy array.
    re    = d['re'][()]
    tau_c = float(d['tau'][()].max())

    # Skip files whose cloud optical depth is below the physical minimum.
    # A near-zero tau_c means no real cloud is present (clear sky or bad data).
    if tau_c < TAU_C_MIN:
        print(f'\n  SKIPPING {path.name} — tau_c = {tau_c:.4f} < TAU_C_MIN ({TAU_C_MIN})')
        skipped_files.append((path, f'tau_c={tau_c:.4f} < TAU_C_MIN'))
        continue

    re_global_min  = min(re_global_min,  float(re.min()))
    re_global_max  = max(re_global_max,  float(re.max()))
    tau_global_min = min(tau_global_min, tau_c)
    tau_global_max = max(tau_global_max, tau_c)
    max_raw_levels = max(max_raw_levels, len(re))
    tau_c_all.append(tau_c)
    valid_mat_files.append(path)

    if (i + 1) % 10 == 0 or i == n_files - 1:
        print(f'  {i+1}/{n_files}', end='\r')

print(f'\n')
if skipped_files:
    print(f'Skipped {len(skipped_files)} file(s) due to missing keys:')
    for p, missing in skipped_files:
        print(f'  {p.name}: {missing}')
    print()

# Update counts to reflect only valid files
n_files = len(valid_mat_files)
n_total = n_files * N_GEOMETRIES
print(f'Valid files for Pass 2: {n_files} → {n_total} total training samples')
print(f'r_e  bounds:      [{re_global_min:.2f}, {re_global_max:.2f}] μm')
print(f'tau_c bounds:     [{tau_global_min:.2f}, {tau_global_max:.2f}]')
print(f'max profile levels: {max_raw_levels}')
print()
# Suggest round numbers slightly above the observed max
re_max_suggested  = float(np.ceil(re_global_max  / 5) * 5)
tau_max_suggested = float(np.ceil(tau_global_max / 5) * 5)
print(f'Suggested normalization bounds — update RE_MAX and TAU_MAX in data.py and models.py:')
print(f'  RE_MIN  = {re_global_min:.1f}')
print(f'  RE_MAX  = {re_max_suggested:.0f}')
print(f'  TAU_MIN = {tau_global_min:.1f}')
print(f'  TAU_MAX = {tau_max_suggested:.0f}')

# ---- Count unique profiles after tau_c filtering ----
# Fingerprint by rounding the first 5 values of each raw r_e profile.
# This matches the logic in data.compute_profile_ids().
unique_fingerprints = set()
for path in valid_mat_files:
    d = scipy.io.loadmat(path, squeeze_me=True)
    re_raw = d['re'][()].astype(np.float64)
    fp = tuple(np.round(re_raw[:5], 4))
    unique_fingerprints.add(fp)
n_unique_profiles = len(unique_fingerprints)
print(f'\nUnique cloud droplet profiles after τ_c ≥ {TAU_C_MIN} filtering: {n_unique_profiles}')
print(f'  (Each profile is simulated under {N_GEOMETRIES} viewing geometries)')
print(f'  Suggested profile-held-out split for data.py / emulator.yaml:')
n_test_suggest  = max(1, round(n_unique_profiles * 0.07))
n_val_suggest   = max(1, round(n_unique_profiles * 0.07))
n_train_suggest = n_unique_profiles - n_val_suggest - n_test_suggest
print(f'    n_val_profiles:  {n_val_suggest}')
print(f'    n_test_profiles: {n_test_suggest}')
print(f'    Training profiles: ~{n_train_suggest}')

FIGURES_DIR = Path('/Users/andrewbuggee/Documents/VS_CODE/Python-Research/lasp-CU-paper-3/Figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots()
ax.hist(tau_c_all, bins=30)
ax.set_xlabel('tau_c')
ax.set_ylabel('Count')
ax.set_title('Distribution of tau_c across all files')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_tau_c.png', dpi=400, bbox_inches='tight')
plt.show()


# ============================================================
# Extract ERA5 pressure levels from the first file
# (standard 37-level grid: 1000 → 1 hPa — identical across all files)
# ============================================================

d_era5_ref = scipy.io.loadmat(valid_mat_files[0], squeeze_me=True)
dp_ref     = d_era5_ref['era5'].item()[6].item()
era5_pressure_levels = np.asarray(dp_ref[2], dtype=np.float32)   # hPa, (37,)
N_ERA5_LEVELS        = len(era5_pressure_levels)
print(f'ERA5 pressure levels: {N_ERA5_LEVELS} levels  '
      f'({era5_pressure_levels[0]:.0f}–{era5_pressure_levels[-1]:.0f} hPa)')


# ============================================================
# Extract wavelength grid from the first file
# Band center = mean of lower and upper bounds.
# These are always in the 3rd-to-last (-3) and 2nd-to-last (-2) columns,
# with the last column (-1) always being the band index.
# (Some .mat files have 6 columns, others 7 — using relative indices
#  makes this robust to both layouts.)
# ============================================================

d_ref = scipy.io.loadmat(valid_mat_files[0], squeeze_me=True)
cv_ref = d_ref['changing_variables_allStateVectors']
wavelengths = ((cv_ref[:N_WAVELENGTHS, -3] + cv_ref[:N_WAVELENGTHS, -2]) / 2
               ).astype(np.float32)
assert len(wavelengths) == N_WAVELENGTHS
print(f'\nWavelength grid: {N_WAVELENGTHS} bands, '
      f'{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} nm')


# ============================================================
# Pass 2: write HDF5
# ============================================================

print(f'\nPass 2: writing HDF5 → {OUT_PATH}')

with h5py.File(OUT_PATH, 'w') as f:
    ds_refl_hysics   = f.create_dataset('reflectances_hysics',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_refl_emit     = f.create_dataset('reflectances_emit',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_uncert_hysics = f.create_dataset('reflectances_uncertainty_hysics',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_uncert_emit   = f.create_dataset('reflectances_uncertainty_emit',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')

    # Training target: r_e interpolated to N_LEVELS evenly-spaced altitude levels
    ds_prof   = f.create_dataset('profiles',
                                 shape=(n_total, N_LEVELS), dtype='f4')

    # Raw in-situ profiles at original resolution, padded with NaN to max_raw_levels.
    # Not used during training; kept for validation plots and diagnostics.
    # profiles_raw_tau stores the full measured τ(z) profile, in the same ordering
    # (cloud top → base, index 0 = top, so τ[0] ≈ 0 and τ[n_lev-1] = τ_c) and the
    # same valid length (via /profile_n_levels) as profiles_raw and profiles_raw_z.
    raw_fill = np.full((n_total, max_raw_levels), np.nan, dtype=np.float32)
    ds_prof_raw = f.create_dataset('profiles_raw',     data=raw_fill)
    ds_z_raw    = f.create_dataset('profiles_raw_z',   data=raw_fill.copy())
    ds_tau_raw  = f.create_dataset('profiles_raw_tau', data=raw_fill.copy())
    del raw_fill   # free memory before the main loop
    ds_n_levels = f.create_dataset('profile_n_levels', shape=(n_total,), dtype='i4')

    ds_tau  = f.create_dataset('tau_c', shape=(n_total,), dtype='f4')
    ds_vza  = f.create_dataset('vza',   shape=(n_total,), dtype='f4')
    ds_vaz  = f.create_dataset('vaz',   shape=(n_total,), dtype='f4')
    ds_saz  = f.create_dataset('saz',   shape=(n_total,), dtype='f4')
    ds_sza  = f.create_dataset('sza',   shape=(n_total,), dtype='f4')

    # ERA5 water vapour datasets
    # wv_above_cloud / wv_in_cloud: integrated column densities (molec/cm²)
    # era5_vapor_concentration: full 37-level profile (molec/cm³)
    ds_wv_above = f.create_dataset('wv_above_cloud',
                                   shape=(n_total,), dtype='f8')
    ds_wv_in    = f.create_dataset('wv_in_cloud',
                                   shape=(n_total,), dtype='f8')
    ds_era5_vap = f.create_dataset('era5_vapor_concentration',
                                   shape=(n_total, N_ERA5_LEVELS), dtype='f4')
    # Pressure levels are the same for every file — store once
    f.create_dataset('era5_pressure_levels', data=era5_pressure_levels)

    wv_above_all = []   # collect one value per file for the summary plots
    wv_in_all    = []
    era5_vap_all = []   # collect per-file vapor profiles for bounds summary

    for i, path in enumerate(valid_mat_files):
        d = scipy.io.loadmat(path, squeeze_me=True)

        # Load raw reflectances and generate noisy versions
        refl_raw = d['Refl_model_allStateVectors'].astype(np.float64)  # (636, 128)

        # Generate HySICS (0.3% noise) and EMIT (4% noise)
        refl_hysics, uncert_hysics = add_gaussian_noise(refl_raw, NOISE_HYSICS+NOISE_FM)
        refl_emit, uncert_emit     = add_gaussian_noise(refl_raw, NOISE_EMIT+NOISE_FM)

        # Transpose: (636, 128) → (128, 636)
        refl_hysics   = refl_hysics.T
        refl_emit     = refl_emit.T
        uncert_hysics = uncert_hysics.T
        uncert_emit   = uncert_emit.T

        # re, z, tau are 1x1 cell arrays; use [()] to extract the inner array.
        # Convention (confirmed across re, z, tau, lwc in the source .mat files):
        # index 0 = cloud top, index -1 = cloud base.
        re_raw  = d['re'][()].astype(np.float64)
        z_raw   = d['z'][()].astype(np.float64)
        tau_raw = np.atleast_1d(d['tau'][()]).astype(np.float64)
        n_lev   = len(re_raw)

        # Targeted despike: profile pid 204 has one bad in-situ r_e sample at
        # k=7 (~21 μm vs ~7 μm at neighbors).  Patch ONLY this profile, leaving
        # every other file bit-identical to previous HDF5 builds.  See the
        # comment block above PID_204_FINGERPRINT for the diagnostic trail.
        fp = tuple(round(float(v), 4) for v in re_raw[:5])
        if fp == PID_204_FINGERPRINT:
            re_raw_orig = re_raw.copy()
            re_raw = despike_re(re_raw, threshold_um=5.0, window=3)
            n_patched = int(np.sum(re_raw_orig != re_raw))
            patched_k = np.where(re_raw_orig != re_raw)[0].tolist()
            print(f"  [despike] {path.name}: matched pid 204 fingerprint; "
                  f"patched {n_patched} sample(s) at k={patched_k} "
                  f"(was {[round(float(re_raw_orig[k]),3) for k in patched_k]} "
                  f"-> {[round(float(re_raw[k]),3) for k in patched_k]} μm).")

        # Sanity check: tau profile should match re/z in length.
        #
        # A known quirk in some source .mat files: the tau vector contains
        # one or more duplicate values (sometimes a trailing duplicate of
        # the last entry, sometimes duplicated values deeper in the cloud).
        # These were supposed to have been cleaned up upstream but a few
        # cases slipped through.
        #
        # Some of those "duplicates" are not bit-exact — for example,
        # tau[-1]=9.159509165643216 vs tau[-2]=9.159509165643220 differ at
        # the 16th decimal place (float64 rounding noise from whatever
        # computation produced them).  We therefore dedupe at a relative
        # tolerance of 1e-6 (≈ 6 significant figures), which is orders of
        # magnitude looser than machine epsilon but far tighter than any
        # physically meaningful difference in optical depth.
        #
        # When dedup produces exactly n_lev unique values, we proceed.
        # Any other mismatch is a real data-quality problem; fail loudly.
        def _unique_with_tol(a, rtol=1e-6):
            """
            Return the values of `a` with near-duplicates removed, preserving
            the order of first occurrence.  Two values v and w are considered
            the same if  |v - w| <= rtol * max(|v|, |w|, 1).
            """
            kept = []
            for v in a:
                is_dup = False
                for k in kept:
                    if abs(v - k) <= rtol * max(abs(v), abs(k), 1.0):
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(v)
            return np.array(kept, dtype=a.dtype)

        if len(tau_raw) > n_lev:
            tau_unique = _unique_with_tol(tau_raw, rtol=1e-6)
            if len(tau_unique) == n_lev:
                n_dropped = len(tau_raw) - n_lev
                print(f"  [note] {path.name}: dropped {n_dropped} duplicate "
                      f"tau value(s) within 1e-6 relative tolerance "
                      f"(len(tau) {len(tau_raw)} → {n_lev}).")
                tau_raw = tau_unique
            else:
                raise ValueError(
                    f"{path.name}: len(tau)={len(tau_raw)} > len(re)={n_lev}, "
                    f"but tau has {len(tau_unique)} unique values at rtol=1e-6 "
                    f"(expected {n_lev}).  Inspect this .mat file manually."
                )
        elif len(tau_raw) < n_lev:
            raise ValueError(
                f"{path.name}: len(tau)={len(tau_raw)} < len(re)={n_lev}. "
                f"Cannot infer missing tau values.  Inspect this .mat file."
            )

        # Training target: interpolate to N_LEVELS using the configured mode.
        # Both branches return a (N_LEVELS,) float32 array ordered top→base.
        if SAMPLING_MODE == 'even_z':
            profile_fixed = interpolate_profile(re_raw, z_raw, N_LEVELS)
        elif SAMPLING_MODE == 'tau_weighted':
            # int() cast is required: np.linspace(..., num=...) rejects floats.
            n_top = int(round(TAU_TOP_FRAC * N_LEVELS))
            n_bot = N_LEVELS - n_top
            profile_fixed, _tau_grid, _z_interp = interpolate_profile_tau_weighted(
                re_raw, tau_raw, z_raw,
                n_top=n_top, n_bot=n_bot, split_tau=TAU_SPLIT,
            )  # profile_fixed: (N_LEVELS,) float32
        else:
            raise ValueError(
                f"SAMPLING_MODE must be 'even_z' or 'tau_weighted', "
                f"got {SAMPLING_MODE!r}"
            )

        # Total (column) optical depth — kept as a separate scalar for backward
        # compatibility with existing consumers; the full τ(z) profile is now
        # also saved to /profiles_raw_tau.
        tau_c = float(tau_raw.max())

        # Geometry: extract one row per geometry block (every 636 rows)
        # Columns: [VZA, VAZ, SAZ]
        cv   = d['changing_variables_allStateVectors']
        geom = cv[::N_WAVELENGTHS, :3].astype(np.float32)   # (128, 3)
        vza  = geom[:, 0]
        vaz  = geom[:, 1]
        saz  = geom[:, 2]

        # Write all 128 samples for this file
        row_start = i * N_GEOMETRIES
        row_end   = row_start + N_GEOMETRIES

        ds_refl_hysics[row_start:row_end]   = refl_hysics
        ds_refl_emit[row_start:row_end]     = refl_emit
        ds_uncert_hysics[row_start:row_end] = uncert_hysics
        ds_uncert_emit[row_start:row_end]   = uncert_emit

        # Same profile and tau_c broadcast to all 128 geometry samples
        ds_prof[row_start:row_end]     = profile_fixed[np.newaxis, :]
        ds_tau[row_start:row_end]      = tau_c

        # Raw profile: store actual levels; columns beyond n_lev remain NaN
        ds_prof_raw[row_start:row_end, :n_lev]  = re_raw.astype(np.float32)[np.newaxis, :]
        ds_z_raw[row_start:row_end,    :n_lev]  = z_raw.astype(np.float32)[np.newaxis, :]
        ds_tau_raw[row_start:row_end,  :n_lev]  = tau_raw.astype(np.float32)[np.newaxis, :]
        ds_n_levels[row_start:row_end]           = n_lev

        sza = parse_sza_from_filename(path.name)

        ds_vza[row_start:row_end]    = vza
        ds_vaz[row_start:row_end]    = vaz
        ds_saz[row_start:row_end]    = saz
        ds_sza[row_start:row_end]    = sza

        # ERA5 water vapour — same profile for all 128 geometries in this file
        wv_above, wv_in, vap_profile = extract_era5_vapor(d, z_raw)
        ds_wv_above[row_start:row_end] = wv_above
        ds_wv_in[row_start:row_end]    = wv_in
        ds_era5_vap[row_start:row_end] = vap_profile[np.newaxis, :]
        wv_above_all.append(wv_above)
        wv_in_all.append(wv_in)
        era5_vap_all.append(vap_profile)

        if (i + 1) % 10 == 0 or i == n_files - 1:
            print(f'  {i+1}/{n_files}  ({row_end} samples written)')

    # Wavelength grid (stored once)
    f.create_dataset('wavelengths', data=wavelengths)

    # Root-level metadata
    f.attrs['n_mat_files']     = n_files
    f.attrs['n_geometries']    = N_GEOMETRIES
    f.attrs['re_global_min']   = re_global_min
    f.attrs['re_global_max']   = re_global_max
    f.attrs['tau_global_min']  = tau_global_min
    f.attrs['tau_global_max']  = tau_global_max
    f.attrs['max_raw_levels']  = max_raw_levels

print(f'\nDone.')
print(f'  Output: {OUT_PATH}')
print(f'  Total samples: {n_total}  ({n_files} profiles × {N_GEOMETRIES} geometries)')


# ============================================================
# Above-cloud water vapour histogram
# ============================================================

AVOGADRO = 6.02214076e23   # molecules / mol
M_H2O    = 18.015e-3       # kg / mol
MOLEC_CM2_TO_KG_M2 = M_H2O / AVOGADRO * 1e4   # ≈ 2.99e-22

wv_above_arr   = np.array(wv_above_all)
wv_above_kg_m2 = wv_above_arr * MOLEC_CM2_TO_KG_M2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(wv_above_kg_m2, bins=30, color='steelblue', edgecolor='white',
             linewidth=0.5)
axes[0].set_xlabel('Above-cloud WV column  (kg m⁻²)', fontsize=11)
axes[0].set_ylabel('Number of profiles', fontsize=11)
axes[0].set_title('Above-cloud water vapour — linear scale', fontsize=12)

log_above_kg = np.log10(wv_above_kg_m2)
axes[1].hist(log_above_kg, bins=30, color='steelblue', edgecolor='white',
             linewidth=0.5)
axes[1].set_xlabel('log₁₀  [above-cloud WV column  (kg m⁻²)]', fontsize=11)
axes[1].set_ylabel('Number of profiles', fontsize=11)
axes[1].set_title('Above-cloud water vapour — log₁₀ scale', fontsize=12)

plt.suptitle(
    f'ERA5 above-cloud water vapour  ({n_files} unique profiles)\n'
    f'Range: {wv_above_kg_m2.min():.2f} – {wv_above_kg_m2.max():.2f}  kg m⁻²',
    fontsize=11,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_wv_above_cloud.png', dpi=400, bbox_inches='tight')
plt.show()


# ============================================================
# In-cloud water vapour histogram
# ============================================================

wv_in_arr   = np.array(wv_in_all)
wv_in_kg_m2 = wv_in_arr * MOLEC_CM2_TO_KG_M2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(wv_in_kg_m2, bins=30, color='darkorange', edgecolor='white',
             linewidth=0.5)
axes[0].set_xlabel('In-cloud WV column  (kg m⁻²)', fontsize=11)
axes[0].set_ylabel('Number of profiles', fontsize=11)
axes[0].set_title('In-cloud water vapour — linear scale', fontsize=12)

log_in_kg = np.log10(wv_in_kg_m2)
axes[1].hist(log_in_kg, bins=30, color='darkorange', edgecolor='white',
             linewidth=0.5)
axes[1].set_xlabel('log₁₀  [in-cloud WV column  (kg m⁻²)]', fontsize=11)
axes[1].set_ylabel('Number of profiles', fontsize=11)
axes[1].set_title('In-cloud water vapour — log₁₀ scale', fontsize=12)

plt.suptitle(
    f'ERA5 in-cloud water vapour  ({n_files} unique profiles)\n'
    f'Range: {wv_in_kg_m2.min():.4f} – {wv_in_kg_m2.max():.4f}  kg m⁻²',
    fontsize=11,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_wv_in_cloud.png', dpi=400, bbox_inches='tight')
plt.show()


# ============================================================
# Water vapour bounds summary for data.py
# ============================================================

era5_vap_arr = np.array(era5_vap_all)  # (n_files, N_ERA5_LEVELS), molec/cm³

log10_wv_above = np.log10(wv_above_arr)
log10_wv_in    = np.log10(wv_in_arr)
log10_era5_vap = np.log10(era5_vap_arr[era5_vap_arr > 0])  # exclude zeros

print('\n' + '=' * 60)
print('WATER VAPOUR BOUNDS — update these in data.py')
print('=' * 60)
print(f'\n  Above-cloud WV column (molec/cm²):')
print(f'    min = {wv_above_arr.min():.4e}   log10 = {log10_wv_above.min():.2f}')
print(f'    max = {wv_above_arr.max():.4e}   log10 = {log10_wv_above.max():.2f}')
print(f'  In-cloud WV column (molec/cm²):')
print(f'    min = {wv_in_arr.min():.4e}   log10 = {log10_wv_in.min():.2f}')
print(f'    max = {wv_in_arr.max():.4e}   log10 = {log10_wv_in.max():.2f}')
print(f'  ERA5 vapor concentration (molec/cm³):')
print(f'    surface max = {era5_vap_arr.max():.4e}   log10 = {np.log10(era5_vap_arr.max()):.2f}')

# Suggest bounds with a small margin
wv_above_log10_min = float(np.floor(log10_wv_above.min()))
wv_above_log10_max = float(np.ceil(log10_wv_above.max()))
wv_in_log10_min    = float(np.floor(log10_wv_in.min()))
wv_in_log10_max    = float(np.ceil(log10_wv_in.max()))
era5_vap_log10_max = float(np.ceil(np.log10(era5_vap_arr.max())))

print(f'\n  Suggested bounds for data.py:')
print(f'    WV_ABOVE_LOG10_MIN = {wv_above_log10_min:.1f}')
print(f'    WV_ABOVE_LOG10_MAX = {wv_above_log10_max:.1f}')
print(f'    WV_IN_LOG10_MIN    = {wv_in_log10_min:.1f}')
print(f'    WV_IN_LOG10_MAX    = {wv_in_log10_max:.1f}')
print(f'    ERA5_VAP_LOG10_MAX = {era5_vap_log10_max:.1f}')
