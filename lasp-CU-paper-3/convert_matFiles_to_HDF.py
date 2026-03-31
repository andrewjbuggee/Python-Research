"""
convert_matFiles_to_HDF.py

Converts individual libRadtran .MAT simulation output files into a single
HDF5 training file compatible with LibRadtranDataset in data.py.

Structure of each .MAT file
----------------------------
  Refl_model_with_noise_allStateVectors : (636, 128) float64
      TOA reflectance with 0.3% Gaussian noise added; one spectrum per
      viewing geometry (columns). This is what the network is trained on.

  Refl_model_uncert_allStateVectors : (636, 128) float64
      Per-channel 1-sigma measurement uncertainty (same shape). Stored in
      the HDF5 for potential use in the Stage 2 emulator data-fidelity loss.

  changing_variables_allStateVectors : (636 * 128, 6) float64
      One row per (wavelength by geometry) RT simulation.
      Column 0 = VZA (deg), Column 1 = VAZ (deg), Column 2 = SAZ (deg).
      Columns 3, 4, 5 = band lower bound (nm), upper bound (nm), band index.
      Extract geometry for each of the 128 viewing configs with [::636, :3].

  re  : (n_insitu,) float64   — in-situ r_e profile, cloud top → base (μm)
  z   : (n_insitu,) float64   — altitude, cloud top → base (km, decreasing)
  tau : (n_insitu,) float64   — cumulative optical depth; tau[-1] = tau_c
  lwc : (n_insitu,) float64   — liquid water content (not used here)

SZA is fixed at 0° for all simulations in this dataset (overhead sun).

Output HDF5 structure (matches LibRadtranDataset in data.py)
--------------------------------------------------------------
  /reflectances  (n_total, 636)       — noisy TOA reflectance
  /reflectances_uncertainty (n_total, 636) — per-channel 1-sigma uncertainty
  /profiles      (n_total, N_LEVELS)  — r_e interpolated to fixed grid (top→base)
  /tau_c         (n_total,)           — cloud optical depth
  /vza           (n_total,)           — viewing zenith angle (deg)
  /vaz           (n_total,)           — viewing azimuth angle (deg)
  /saz           (n_total,)           — solar azimuth angle (deg)
  /sza           (n_total,)           — solar zenith angle (deg) = 0 here
  /wavelengths   (636,)               — HySICS band center wavelengths (nm)

  Attributes on the root group:
    n_mat_files, n_geometries, sza_fixed_deg,
    re_global_min, re_global_max, tau_global_min, tau_global_max

n_total = n_mat_files by 128

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import numpy as np
import h5py
import scipy.io
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MAT_DIR  = Path('/Volumes/My Passport/neural_network_training_data/'
                'Combined_data_from_13_March_and_30_March_2026_sza0_23_33_and_41/')
OUT_PATH = Path('/Volumes/My Passport/neural_network_training_data/'
                'training_data_VR_sza_0_23_33_41.h5')

N_LEVELS      = 10    # target vertical levels in output profile
N_GEOMETRIES  = 128   # viewing geometry configs per .mat file (8 VZA × 4 VAZ × 4 SAZ)
N_WAVELENGTHS = 636   # HySICS spectral channels

SZA_FIXED = 0.0       # solar zenith angle is fixed at 0° for this dataset


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

for i, path in enumerate(mat_files):
    d     = scipy.io.loadmat(path, squeeze_me=True)
    re    = d['re']
    tau_c = float(d['tau'][-1])

    re_global_min  = min(re_global_min,  float(re.min()))
    re_global_max  = max(re_global_max,  float(re.max()))
    tau_global_min = min(tau_global_min, tau_c)
    tau_global_max = max(tau_global_max, tau_c)
    max_raw_levels = max(max_raw_levels, len(re))

    if (i + 1) % 10 == 0 or i == n_files - 1:
        print(f'  {i+1}/{n_files}', end='\r')

print(f'\n')
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


# ============================================================
# Extract wavelength grid from the first file
# (band center = mean of lower and upper bound in columns 3 and 4)
# ============================================================

d_ref = scipy.io.loadmat(mat_files[0], squeeze_me=True)
cv_ref = d_ref['changing_variables_allStateVectors']
wavelengths = ((cv_ref[:N_WAVELENGTHS, 3] + cv_ref[:N_WAVELENGTHS, 4]) / 2
               ).astype(np.float32)
assert len(wavelengths) == N_WAVELENGTHS
print(f'\nWavelength grid: {N_WAVELENGTHS} bands, '
      f'{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} nm')


# ============================================================
# Pass 2: write HDF5
# ============================================================

print(f'\nPass 2: writing HDF5 → {OUT_PATH}')

with h5py.File(OUT_PATH, 'w') as f:
    ds_refl   = f.create_dataset('reflectances',
                                 shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_uncert = f.create_dataset('reflectances_uncertainty',
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

    for i, path in enumerate(mat_files):
        d = scipy.io.loadmat(path, squeeze_me=True)

        # Reflectances: (636, 128) → transpose → (128, 636)
        refl   = d['Refl_model_with_noise_allStateVectors'].astype(np.float32).T
        uncert = d['Refl_model_uncert_allStateVectors'].astype(np.float32).T

        re_raw = d['re'].astype(np.float64)
        z_raw  = d['z'].astype(np.float64)
        n_lev  = len(re_raw)

        # Training target: interpolate to N_LEVELS
        profile_fixed = interpolate_profile(re_raw, z_raw, N_LEVELS)  # (N_LEVELS,)

        # Total optical depth
        tau_c = float(d['tau'][-1])

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

        ds_refl[row_start:row_end]   = refl
        ds_uncert[row_start:row_end] = uncert

        # Same profile and tau_c broadcast to all 128 geometry samples
        ds_prof[row_start:row_end]     = profile_fixed[np.newaxis, :]
        ds_tau[row_start:row_end]      = tau_c

        # Raw profile: store actual levels; columns beyond n_lev remain NaN
        ds_prof_raw[row_start:row_end, :n_lev]  = re_raw.astype(np.float32)[np.newaxis, :]
        ds_z_raw[row_start:row_end,    :n_lev]  = z_raw.astype(np.float32)[np.newaxis, :]
        ds_n_levels[row_start:row_end]           = n_lev

        ds_vza[row_start:row_end]    = vza
        ds_vaz[row_start:row_end]    = vaz
        ds_saz[row_start:row_end]    = saz
        ds_sza[row_start:row_end]    = SZA_FIXED

        if (i + 1) % 10 == 0 or i == n_files - 1:
            print(f'  {i+1}/{n_files}  ({row_end} samples written)')

    # Wavelength grid (stored once)
    f.create_dataset('wavelengths', data=wavelengths)

    # Root-level metadata
    f.attrs['n_mat_files']     = n_files
    f.attrs['n_geometries']    = N_GEOMETRIES
    f.attrs['sza_fixed_deg']   = SZA_FIXED
    f.attrs['re_global_min']   = re_global_min
    f.attrs['re_global_max']   = re_global_max
    f.attrs['tau_global_min']  = tau_global_min
    f.attrs['tau_global_max']  = tau_global_max
    f.attrs['max_raw_levels']  = max_raw_levels

print(f'\nDone.')
print(f'  Output: {OUT_PATH}')
print(f'  Total samples: {n_total}  ({n_files} profiles × {N_GEOMETRIES} geometries)')
