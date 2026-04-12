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
  /profiles      (n_total, N_LEVELS)  — r_e interpolated to fixed grid (top→base)
  /tau_c         (n_total,)           — cloud optical depth
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

MAT_DIR  = Path('/Volumes/My Passport/neural_network_training_data/combined_vocals_oracles_training_data_12_April_2026/')
OUT_PATH = Path('/Volumes/My Passport/neural_network_training_data/'
                'combined_vocals_oracles_training_data_12_April_2026.h5')

N_LEVELS      = 10    # target vertical levels in output profile
N_GEOMETRIES  = 128   # viewing geometry configs per .mat file (8 VZA × 4 VAZ × 4 SAZ)
N_WAVELENGTHS = 636   # HySICS spectral channels

# Keys that must be present in every .mat file to be included
# Now we generate noise from the raw reflectances, so we only need raw data
REQUIRED_KEYS = {
    'Refl_model_allStateVectors',
    'changing_variables_allStateVectors',
    're', 'z', 'tau',
}

# Noise levels for each instrument (fraction of signal, applied as Gaussian)
NOISE_HYSICS = 0.003  # 0.3% Gaussian noise
NOISE_EMIT   = 0.04   # 4.0% Gaussian noise

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

fig, ax = plt.subplots()
ax.hist(tau_c_all, bins=30)
ax.set_xlabel('tau_c')
ax.set_ylabel('Count')
ax.set_title('Distribution of tau_c across all files')
plt.tight_layout()
plt.show()


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
    raw_fill = np.full((n_total, max_raw_levels), np.nan, dtype=np.float32)
    ds_prof_raw = f.create_dataset('profiles_raw',    data=raw_fill)
    ds_z_raw    = f.create_dataset('profiles_raw_z',  data=raw_fill.copy())
    del raw_fill   # free memory before the main loop
    ds_n_levels = f.create_dataset('profile_n_levels', shape=(n_total,), dtype='i4')

    ds_tau  = f.create_dataset('tau_c', shape=(n_total,), dtype='f4')
    ds_vza  = f.create_dataset('vza',   shape=(n_total,), dtype='f4')
    ds_vaz  = f.create_dataset('vaz',   shape=(n_total,), dtype='f4')
    ds_saz  = f.create_dataset('saz',   shape=(n_total,), dtype='f4')
    ds_sza  = f.create_dataset('sza',   shape=(n_total,), dtype='f4')

    for i, path in enumerate(valid_mat_files):
        d = scipy.io.loadmat(path, squeeze_me=True)

        # Load raw reflectances and generate noisy versions
        refl_raw = d['Refl_model_allStateVectors'].astype(np.float64)  # (636, 128)

        # Generate HySICS (0.3% noise) and EMIT (4% noise)
        refl_hysics, uncert_hysics = add_gaussian_noise(refl_raw, NOISE_HYSICS)
        refl_emit, uncert_emit     = add_gaussian_noise(refl_raw, NOISE_EMIT)

        # Transpose: (636, 128) → (128, 636)
        refl_hysics   = refl_hysics.T
        refl_emit     = refl_emit.T
        uncert_hysics = uncert_hysics.T
        uncert_emit   = uncert_emit.T

        # re, z, tau are 1x1 cell arrays; use [()] to extract the inner array
        re_raw = d['re'][()].astype(np.float64)
        z_raw  = d['z'][()].astype(np.float64)
        n_lev  = len(re_raw)

        # Training target: interpolate to N_LEVELS
        profile_fixed = interpolate_profile(re_raw, z_raw, N_LEVELS)  # (N_LEVELS,)

        # Total optical depth
        tau_c = float(d['tau'][()].max())

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
        ds_n_levels[row_start:row_end]           = n_lev

        sza = parse_sza_from_filename(path.name)

        ds_vza[row_start:row_end]    = vza
        ds_vaz[row_start:row_end]    = vaz
        ds_saz[row_start:row_end]    = saz
        ds_sza[row_start:row_end]    = sza

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
