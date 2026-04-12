"""
patch_hdf5_wavelengths.py

Patches the /wavelengths dataset in an existing HDF5 training file to use
the correct HySICS band-center wavelengths (~352–2297 nm) instead of the
erroneous values (166–1138 nm) that were written by the original
convert_matFiles_to_HDF.py, which accidentally used column 3 (an unused
zero placeholder) instead of the correct column 4 (band lower bound).

Correct extraction:
  band_center = mean(col 4, col 5)   →  (lower_nm + upper_nm) / 2

This script reads the corrected wavelengths from a single reference .mat
file and overwrites /wavelengths in the HDF5 file — no reflectances or
profiles are touched.

Usage
-----
  python patch_hdf5_wavelengths.py

Set MAT_REF_PATH and HDF5_PATH below (or pass them on the command line).
"""

import sys
import numpy as np
import h5py
import scipy.io
from pathlib import Path


# ------------------------------------------------------------------ #
# Configuration — edit these paths as needed                         #
# ------------------------------------------------------------------ #

# Any one of the original .mat files (used only to read wavelength bounds).
MAT_REF_PATH = Path(
    '/Volumes/My Passport/neural_network_training_data/'
    'combined_vocals_oracles_training_data_12_April_2026/'
).glob('*.mat')

# The HDF5 file whose /wavelengths dataset will be patched.
HDF5_PATH = Path(
    '/Volumes/My Passport/neural_network_training_data/'
    'vocalsRex_training_data_created_on_31_March_2026.h5'
)

N_WAVELENGTHS = 636


# ------------------------------------------------------------------ #
# Resolve the reference .mat file                                     #
# ------------------------------------------------------------------ #

if len(sys.argv) == 3:
    mat_ref_path = Path(sys.argv[1])
    hdf5_path    = Path(sys.argv[2])
else:
    # Default: pick the first non-metadata mat file from MAT_REF_PATH glob
    candidates = sorted(
        f for f in MAT_REF_PATH
        if not f.name.startswith('._')
    )
    if not candidates:
        raise FileNotFoundError(
            f'No .mat files found. '
            f'Edit MAT_REF_PATH in {__file__} or pass paths as arguments.'
        )
    mat_ref_path = candidates[0]
    hdf5_path    = HDF5_PATH


# ------------------------------------------------------------------ #
# Read correct wavelengths from the .mat reference file               #
# ------------------------------------------------------------------ #

print(f'Reading wavelength bounds from: {mat_ref_path.name}')
d = scipy.io.loadmat(mat_ref_path, squeeze_me=True)
cv = d['changing_variables_allStateVectors']

# Column layout (relative, 0-indexed from the right):
#   col -1 = band index (1-based)
#   col -2 = band upper bound (nm)
#   col -3 = band lower bound (nm)
#   col  0 = VZA,  col 1 = VAZ,  col 2 = SAZ
# Some .mat files have 6 columns, others 7 (extra unused zero at col 3).
# Using relative indices [-3, -2] handles both layouts correctly.
n_cols = cv.shape[1]
print(f'  changing_variables shape: {cv.shape}  ({n_cols} columns)')
assert n_cols in (6, 7), (
    f'Unexpected column count {n_cols}; expected 6 or 7. '
    'Check the .mat file format.'
)

wavelengths_new = ((cv[:N_WAVELENGTHS, -3] + cv[:N_WAVELENGTHS, -2]) / 2
                   ).astype(np.float32)
assert len(wavelengths_new) == N_WAVELENGTHS

print(f'  Correct wavelength range: '
      f'{wavelengths_new[0]:.2f} – {wavelengths_new[-1]:.2f} nm')


# ------------------------------------------------------------------ #
# Patch the HDF5 file                                                  #
# ------------------------------------------------------------------ #

print(f'\nPatching HDF5 file: {hdf5_path}')

with h5py.File(hdf5_path, 'r+') as f:
    if 'wavelengths' not in f:
        raise KeyError(
            "'/wavelengths' dataset not found in HDF5 file. "
            "Has the file been created by convert_matFiles_to_HDF.py?"
        )

    old_wav = f['wavelengths'][:]
    print(f'  Old wavelength range: {old_wav[0]:.2f} – {old_wav[-1]:.2f} nm')
    print(f'  New wavelength range: {wavelengths_new[0]:.2f} – {wavelengths_new[-1]:.2f} nm')

    if old_wav.shape != wavelengths_new.shape:
        raise ValueError(
            f'Shape mismatch: HDF5 has {old_wav.shape}, '
            f'new array has {wavelengths_new.shape}.'
        )

    # Overwrite in-place
    f['wavelengths'][:] = wavelengths_new
    f['wavelengths'].attrs['patched_by']   = 'patch_hdf5_wavelengths.py'
    f['wavelengths'].attrs['patch_note']   = (
        'Corrected from wrong col[3,4] (166-1138 nm) to correct col[-3,-2] '
        '(~352-2297 nm). Band center = mean(lower_bound, upper_bound). '
        'Relative column indices used to handle both 6- and 7-column .mat layouts.'
    )

print('\nPatch complete. Verifying...')
with h5py.File(hdf5_path, 'r') as f:
    wav_check = f['wavelengths'][:]
    print(f'  Stored wavelength range: {wav_check[0]:.2f} – {wav_check[-1]:.2f} nm')
    print(f'  First 5 channels (nm): {wav_check[:5].tolist()}')
    print(f'  Last  5 channels (nm): {wav_check[-5:].tolist()}')

print('\nDone.')
