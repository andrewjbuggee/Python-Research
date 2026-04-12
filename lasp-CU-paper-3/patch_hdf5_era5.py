"""
patch_hdf5_era5.py

Patches an existing HDF5 training file to add ERA5 water vapour datasets
without re-running the full libRadtran conversion.

Three new datasets are added:
    /wv_above_cloud           (n_total,)        float64  molec/cm²
    /wv_in_cloud              (n_total,)        float64  molec/cm²
    /era5_vapor_concentration (n_total, 37)     float32  molec/cm³
    /era5_pressure_levels     (37,)             float32  hPa  (stored once)

Cloud top and base
------------------
Cloud top  = max(z)   (highest altitude in the in-situ array, km)
Cloud base = min(z)   (lowest  altitude in the in-situ array, km)

These are the actual measured cloud boundaries used in each libRadtran
simulation and are read from the same .mat file as the reflectances.

Integrals
---------
    wv_above_cloud = ∫[z_top → TOA]    n(z) dz   (molec/cm²)
    wv_in_cloud    = ∫[z_base → z_top] n(z) dz   (molec/cm²)

n(z) = ERA5 vapor_concentration (molec/cm³); altitude in cm.

Usage
-----
    python patch_hdf5_era5.py
    python patch_hdf5_era5.py /path/to/mat_dir /path/to/file.h5
"""

import sys
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------ #
# Configuration — edit these paths as needed                         #
# ------------------------------------------------------------------ #

MAT_DIR   = Path('/Volumes/My Passport/neural_network_training_data/'
                 'combined_vocals_oracles_training_data_12_April_2026/')
HDF5_PATH = Path('/Volumes/My Passport/neural_network_training_data/'
                 'combined_vocals_oracles_training_data_12_April_2026.h5')

N_GEOMETRIES = 128    # viewing geometry configs per .mat file
REQUIRED_KEYS = {'era5', 'z', 'Refl_model_allStateVectors'}


# ------------------------------------------------------------------ #
# Helpers (mirrors convert_matFiles_to_HDF.py exactly)               #
# ------------------------------------------------------------------ #

def _era5_datprofiles(d: dict) -> tuple:
    """Return (gp_height_m, vapor_conc) from a loaded .mat dict."""
    dp  = d['era5'].item()[6].item()
    gph = np.asarray(dp[0], dtype=np.float64)   # metres, ascending
    vap = np.asarray(dp[4], dtype=np.float64)   # molecules/cm³
    return gph, vap


def extract_era5_vapor(d: dict, z_raw: np.ndarray) -> tuple:
    """
    Compute above-cloud and in-cloud water vapour column densities.

    Parameters
    ----------
    d     : loadmat dict for one .mat file
    z_raw : (n,) float  cloud altitude in km (any ordering; max/min used)

    Returns
    -------
    wv_above_cloud     : float       molec/cm²
    wv_in_cloud        : float       molec/cm²
    vapor_conc_profile : (37,) f32   full ERA5 profile, molec/cm³
    """
    gph, vap = _era5_datprofiles(d)

    # Use max/min rather than index 0/-1 — the altitude array is not guaranteed
    # to be stored in any particular order across all .mat files.
    z_top_m  = float(z_raw.max()) * 1000.0   # km → m
    z_base_m = float(z_raw.min()) * 1000.0

    z_top_m  = np.clip(z_top_m,  gph[0], gph[-1])
    z_base_m = np.clip(z_base_m, gph[0], z_top_m)

    vap_at_top  = float(np.interp(z_top_m,  gph, vap))
    vap_at_base = float(np.interp(z_base_m, gph, vap))

    # Above-cloud column
    above_mask = gph > z_top_m
    gph_above  = np.concatenate([[z_top_m],    gph[above_mask]])
    vap_above  = np.concatenate([[vap_at_top], vap[above_mask]])
    wv_above   = float(np.trapz(vap_above, gph_above * 100.0))

    # In-cloud column
    in_mask = (gph > z_base_m) & (gph < z_top_m)
    gph_in  = np.concatenate([[z_base_m],    gph[in_mask],  [z_top_m]])
    vap_in  = np.concatenate([[vap_at_base], vap[in_mask],  [vap_at_top]])
    sort_i  = np.argsort(gph_in)
    wv_in   = float(np.trapz(vap_in[sort_i], gph_in[sort_i] * 100.0))

    return wv_above, wv_in, vap.astype(np.float32)


# ------------------------------------------------------------------ #
# Parse command-line arguments                                        #
# ------------------------------------------------------------------ #

if len(sys.argv) == 3:
    mat_dir   = Path(sys.argv[1])
    hdf5_path = Path(sys.argv[2])
else:
    mat_dir   = MAT_DIR
    hdf5_path = HDF5_PATH

print(f'MAT directory : {mat_dir}')
print(f'HDF5 file     : {hdf5_path}')

if not hdf5_path.exists():
    raise FileNotFoundError(f'HDF5 file not found: {hdf5_path}')

# ------------------------------------------------------------------ #
# Collect and sort .mat files (same logic as convert_matFiles_to_HDF) #
# ------------------------------------------------------------------ #

mat_files = sorted(f for f in mat_dir.glob('*.mat')
                   if not f.name.startswith('._'))
print(f'\nFound {len(mat_files)} .mat files in {mat_dir.name}')

# Filter to valid files (must have required keys)
valid_mat_files = []
skipped = []
print('Scanning for required keys...')
for path in mat_files:
    d = scipy.io.loadmat(path, squeeze_me=True)
    missing = REQUIRED_KEYS - set(d.keys())
    if missing:
        skipped.append((path.name, missing))
    else:
        valid_mat_files.append(path)

if skipped:
    print(f'  Skipping {len(skipped)} file(s) missing required keys:')
    for name, miss in skipped:
        print(f'    {name}: {miss}')

n_files = len(valid_mat_files)
n_total = n_files * N_GEOMETRIES
print(f'Valid files: {n_files}  →  {n_total} total samples\n')

# ------------------------------------------------------------------ #
# Verify HDF5 sample count matches                                    #
# ------------------------------------------------------------------ #

with h5py.File(hdf5_path, 'r') as f:
    hdf5_n = f['reflectances_hysics'].shape[0]

if hdf5_n != n_total:
    raise ValueError(
        f'HDF5 has {hdf5_n} samples but {n_files} valid .mat files × '
        f'{N_GEOMETRIES} = {n_total}.  '
        'Ensure MAT_DIR matches the directory used to create the HDF5 file.'
    )
print(f'Sample count verified: {n_total}')

# ------------------------------------------------------------------ #
# Extract ERA5 pressure levels from the first file (stored once)     #
# ------------------------------------------------------------------ #

d_ref = scipy.io.loadmat(valid_mat_files[0], squeeze_me=True)
dp_ref = d_ref['era5'].item()[6].item()
era5_pressure_levels = np.asarray(dp_ref[2], dtype=np.float32)
N_ERA5_LEVELS = len(era5_pressure_levels)
print(f'ERA5 pressure levels: {N_ERA5_LEVELS} levels  '
      f'({era5_pressure_levels[0]:.0f}–{era5_pressure_levels[-1]:.0f} hPa)\n')

# ------------------------------------------------------------------ #
# Check which datasets already exist                                  #
# ------------------------------------------------------------------ #

with h5py.File(hdf5_path, 'r') as f:
    already_have = [k for k in
                    ('wv_above_cloud', 'wv_in_cloud',
                     'era5_vapor_concentration', 'era5_pressure_levels')
                    if k in f]

if already_have:
    print(f'WARNING: the following datasets already exist in the HDF5:')
    for k in already_have:
        print(f'  /{k}')
    ans = input('Overwrite them? [y/N] ').strip().lower()
    if ans != 'y':
        print('Aborted.')
        sys.exit(0)
    # Delete existing datasets so we can recreate them
    with h5py.File(hdf5_path, 'r+') as f:
        for k in already_have:
            del f[k]
    print('Existing datasets deleted.\n')

# ------------------------------------------------------------------ #
# Pre-allocate new datasets                                           #
# ------------------------------------------------------------------ #

print('Pre-allocating new datasets...')
with h5py.File(hdf5_path, 'r+') as f:
    f.create_dataset('wv_above_cloud',
                     shape=(n_total,), dtype='f8')
    f.create_dataset('wv_in_cloud',
                     shape=(n_total,), dtype='f8')
    f.create_dataset('era5_vapor_concentration',
                     shape=(n_total, N_ERA5_LEVELS), dtype='f4')
    f.create_dataset('era5_pressure_levels',
                     data=era5_pressure_levels)
print('  Done.\n')

# ------------------------------------------------------------------ #
# Fill datasets from .mat files                                       #
# ------------------------------------------------------------------ #

wv_above_all = []
wv_in_all    = []

print(f'Extracting ERA5 water vapour from {n_files} files...')
with h5py.File(hdf5_path, 'r+') as f:
    ds_wv_above = f['wv_above_cloud']
    ds_wv_in    = f['wv_in_cloud']
    ds_era5_vap = f['era5_vapor_concentration']

    for i, path in enumerate(valid_mat_files):
        d = scipy.io.loadmat(path, squeeze_me=True)
        z_raw = d['z'][()].astype(np.float64)

        wv_above, wv_in, vap_profile = extract_era5_vapor(d, z_raw)

        row_start = i * N_GEOMETRIES
        row_end   = row_start + N_GEOMETRIES

        ds_wv_above[row_start:row_end] = wv_above
        ds_wv_in[row_start:row_end]    = wv_in
        ds_era5_vap[row_start:row_end] = vap_profile[np.newaxis, :]

        wv_above_all.append(wv_above)
        wv_in_all.append(wv_in)

        if (i + 1) % 50 == 0 or i == n_files - 1:
            print(f'  {i+1}/{n_files}')

# ------------------------------------------------------------------ #
# Summary statistics                                                  #
# ------------------------------------------------------------------ #

wv_above_arr = np.array(wv_above_all)
wv_in_arr    = np.array(wv_in_all)

print('\nColumn water vapour statistics (unique profiles, molec/cm²):')
print(f'  wv_above_cloud: '
      f'min={wv_above_arr.min():.3e}  max={wv_above_arr.max():.3e}  '
      f'log10 range=[{np.log10(wv_above_arr.min()):.2f}, '
      f'{np.log10(wv_above_arr.max()):.2f}]')
print(f'  wv_in_cloud:    '
      f'min={wv_in_arr.min():.3e}  max={wv_in_arr.max():.3e}  '
      f'log10 range=[{np.log10(wv_in_arr.min()):.2f}, '
      f'{np.log10(wv_in_arr.max()):.2f}]')
print('\nUse these log10 ranges to set WV_ABOVE_LOG10_MIN/MAX and '
      'WV_IN_LOG10_MIN/MAX in data.py.')

# ------------------------------------------------------------------ #
# Histogram of above-cloud water vapour                               #
# One value per unique profile (n_files points), not per sample.     #
# ------------------------------------------------------------------ #

# Convert from molecules/cm² to kg/m²:
#   ÷ Avogadro (molec/mol)  × M_H2O (kg/mol)  × 1e4 (cm²/m²)
AVOGADRO  = 6.02214076e23   # molecules / mol
M_H2O     = 18.015e-3       # kg / mol
MOLEC_CM2_TO_KG_M2 = M_H2O / AVOGADRO * 1e4

wv_above_kg_m2 = wv_above_arr * MOLEC_CM2_TO_KG_M2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear scale (kg/m²)
axes[0].hist(wv_above_kg_m2, bins=30, color='steelblue', edgecolor='white',
             linewidth=0.5)
axes[0].set_xlabel('Above-cloud WV column  (kg m⁻²)', fontsize=11)
axes[0].set_ylabel('Number of profiles', fontsize=11)
axes[0].set_title('Above-cloud water vapour — linear scale', fontsize=12)

# Log10 scale (log10 kg/m²) — easier to spot outliers and check bounds
log_above_kg = np.log10(wv_above_kg_m2)
axes[1].hist(log_above_kg, bins=30, color='steelblue', edgecolor='white',
             linewidth=0.5)
axes[1].set_xlabel('log₁₀  [above-cloud WV column  (kg m⁻²)]', fontsize=11)
axes[1].set_ylabel('Number of profiles', fontsize=11)
axes[1].set_title('Above-cloud water vapour — log₁₀ scale', fontsize=12)

# Mark the current normalization bounds from data.py (stored in molec/cm²
# log10 space; convert to kg/m² log10 for the plot)
try:
    import importlib, sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    _data = importlib.import_module('data')
    lo_kg = _data.WV_ABOVE_LOG10_MIN + np.log10(MOLEC_CM2_TO_KG_M2)
    hi_kg = _data.WV_ABOVE_LOG10_MAX + np.log10(MOLEC_CM2_TO_KG_M2)
    axes[1].axvline(lo_kg, color='tomato', linestyle='--', linewidth=1.5,
                    label=f'norm MIN  ({lo_kg:.2f})')
    axes[1].axvline(hi_kg, color='tomato', linestyle='-',  linewidth=1.5,
                    label=f'norm MAX  ({hi_kg:.2f})')
    axes[1].legend(fontsize=9)
except Exception:
    pass   # data.py not importable — skip bound lines

plt.suptitle(
    f'ERA5 above-cloud water vapour  ({n_files} unique profiles)\n'
    f'Range: {wv_above_kg_m2.min():.2f} – {wv_above_kg_m2.max():.2f}  kg m⁻²',
    fontsize=11,
)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------ #
# Verify                                                              #
# ------------------------------------------------------------------ #

print('\nVerifying patch...')
with h5py.File(hdf5_path, 'r') as f:
    print(f'  /wv_above_cloud          shape: {f["wv_above_cloud"].shape}  '
          f'sample[0]: {f["wv_above_cloud"][0]:.4e}')
    print(f'  /wv_in_cloud             shape: {f["wv_in_cloud"].shape}  '
          f'sample[0]: {f["wv_in_cloud"][0]:.4e}')
    print(f'  /era5_vapor_concentration shape: {f["era5_vapor_concentration"].shape}')
    print(f'  /era5_pressure_levels    shape: {f["era5_pressure_levels"].shape}  '
          f'{f["era5_pressure_levels"][:3].tolist()} … '
          f'{f["era5_pressure_levels"][-1]:.0f} hPa')

print('\nPatch complete.')
