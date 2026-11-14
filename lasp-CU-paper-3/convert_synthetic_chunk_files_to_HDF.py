"""
convert_synthetic_chunk_files_to_HDF.py
=======================================
Build the training HDF5 from the per-CHUNK files written by
run_synthetic_spectra_chunk.m (one .mat per Slurm array task, each holding
many clouds), instead of the one-.mat-per-cloud files read by
convert_synthetic_matFiles_to_HDF.py.

The HDF5 layout is identical to convert_synthetic_matFiles_to_HDF.py, so
data.py reads it unchanged:

  /reflectances_hysics, /reflectances_emit                     (n, 636) f4
  /reflectances_uncertainty_hysics, /reflectances_uncertainty_emit (n, 636) f4
  /profiles, /profiles_raw, /profiles_raw_z, /profiles_raw_tau (n, 7)   f4
  /profile_n_levels (n,) i4     /lwc (n, 7) f4     /alpha, /tau_c (n,) f4
  /sza, /saz, /vza, /vaz (n,) f4
  /wv_above_cloud, /wv_in_cloud (n,) f8
  /era5_vapor_concentration, /era5_temperature (n, 37) f4
  /era5_pressure_levels (37,) f4      /wavelengths (636,) f4
plus, new here: /cloud_idx (n,) i4 (row of the training-input .nc) and
/band_center_wavelength (636,) f4 (Gaussian centre of each 6-nm-FWHM band).

Differences from the per-cloud converter
  * Reads a handful of chunk files (n_clouds x 636 arrays) instead of
    ~100,000 small files.
  * Truth fields come straight from the chunk file (they were read from the
    training-input .nc by the MATLAB driver).
  * /wavelengths is the midpoint of each band's response window, exactly as
    the per-cloud converter derives it from changing_variables.
  * Clouds whose RT failed (done = False in the chunk file) are skipped and
    listed.

Noise: added here from the clean reflectance, as in the per-cloud converter,
with a fixed seed (RANDOM_SEED, stored as HDF5 attribute 'noise_random_seed')
so the output is reproducible.

Usage
    python convert_synthetic_chunk_files_to_HDF.py <chunk_dir> <out.h5>

Author: Claude (assistant) for Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io

N_LEVELS = 7
N_WAVELENGTHS = 636
TAU_C_MIN = 3.0          # defence in depth; 06 already enforces it

# Noise levels — same values as convert_synthetic_matFiles_to_HDF.py
NOISE_HYSICS = 0.003     # 0.3 % Gaussian
NOISE_EMIT = 0.015       # 1.5 % Gaussian
NOISE_FM = 0.0           # forward-model noise budget
RANDOM_SEED = 20261114   # fixed so the noisy reflectances are reproducible

# chunk-file variables holding one value per cloud (all others hold a row per cloud)
SCALAR_KEYS = {'cloud_idx', 'done', 'sza', 'saz', 'vza', 'vaz', 'tau_c', 'alpha'}

R_DRY = 287.05           # J/(kg K), dry-air gas constant
G0 = 9.80665             # m/s²


# ── helpers copied from convert_synthetic_matFiles_to_HDF.py ──────────────────
def hypsometric_gph(T_K: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Geopotential height (m) from T(p) by hypsometric integration, z = 0 at
    the highest pressure: dz = (R_d / g) · T̄ · ln(p_lower / p_upper).
    Virtual-temperature correction omitted (q is not stored), as in the
    per-cloud converter."""
    p = np.asarray(p_hPa, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    order = np.argsort(p)[::-1]
    p_s, T_s = p[order], T[order]
    z_s = np.zeros_like(p_s)
    for i in range(1, len(p_s)):
        T_bar = 0.5 * (T_s[i - 1] + T_s[i])
        z_s[i] = z_s[i - 1] + (R_DRY / G0) * T_bar * np.log(p_s[i - 1] / p_s[i])
    z = np.empty_like(z_s)
    z[order] = z_s
    return z


def wv_columns(T_K: np.ndarray, p_hPa: np.ndarray, vap_cm3: np.ndarray,
               z_cloud_km: np.ndarray) -> tuple[float, float]:
    """(above-cloud, in-cloud) water-vapour columns in molecules/cm², the same
    integration as extract_synthetic_era5_vapor in the per-cloud converter."""
    gph = hypsometric_gph(T_K, p_hPa)
    z_top_m = np.clip(float(np.max(z_cloud_km)) * 1000.0, gph.min(), gph.max())
    z_base_m = np.clip(float(np.min(z_cloud_km)) * 1000.0, gph.min(), z_top_m)
    o = np.argsort(gph)
    gph_s, vap_s = gph[o], np.asarray(vap_cm3, dtype=np.float64)[o]
    vap_top = float(np.interp(z_top_m, gph_s, vap_s))
    vap_base = float(np.interp(z_base_m, gph_s, vap_s))
    above = gph_s > z_top_m
    wv_above = float(np.trapezoid(np.concatenate([[vap_top], vap_s[above]]),
                                  np.concatenate([[z_top_m], gph_s[above]]) * 100.0))
    inside = (gph_s > z_base_m) & (gph_s < z_top_m)
    g_in = np.concatenate([[z_base_m], gph_s[inside], [z_top_m]])
    v_in = np.concatenate([[vap_base], vap_s[inside], [vap_top]])
    si = np.argsort(g_in)
    wv_in = float(np.trapezoid(v_in[si], g_in[si] * 100.0))
    return wv_above, wv_in


def add_gaussian_noise(refl: np.ndarray, level: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """refl + level·refl·N(0,1) and the 1-sigma uncertainty level·|refl|."""
    noisy = refl + (level * refl) * rng.standard_normal(refl.shape)
    return noisy.astype(np.float32), np.abs(level * refl).astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('usage: python convert_synthetic_chunk_files_to_HDF.py <chunk_dir> <out.h5>')
    chunk_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(f for f in chunk_dir.glob('synthetic_spectra_HySICS_636bands_clouds*.mat')
                   if not f.name.endswith('_partial.mat'))
    if not files:
        raise SystemExit(f'No chunk files in {chunk_dir}')
    print(f'{len(files)} chunk files in {chunk_dir}')
    print(f'Noise random seed: {RANDOM_SEED}')

    keys = ['cloud_idx', 'done', 'Refl_model', 'sza', 'saz', 'vza', 'vaz', 're_um',
            'lwc_g_per_m3', 'z_km', 'tau_c', 'alpha', 'T_K', 'vapor_cm3']
    parts = {k: [] for k in keys}
    pressure = window = centre = None
    n_failed, seen = [], set()
    for f in files:
        d = scipy.io.loadmat(f, squeeze_me=False)
        done = np.asarray(d['done']).ravel().astype(bool)
        idx = np.asarray(d['cloud_idx']).ravel().astype(int)
        n_failed += [int(i) for i in idx[~done]]
        keep = done & (np.asarray(d['tau_c']).ravel() >= TAU_C_MIN)
        dup = np.isin(idx, list(seen))
        if (keep & dup).any():
            print(f'  {f.name}: {int((keep & dup).sum())} clouds already seen in another file; skipped')
        keep &= ~dup
        seen.update(idx[keep].tolist())
        for k in keys:
            a = np.asarray(d[k])
            if k in SCALAR_KEYS:          # one value per cloud: (n, 1) -> (n,)
                parts[k].append(a.ravel()[keep])
            else:                         # one row per cloud: (n, m), also when n = 1
                parts[k].append(a.reshape(len(idx), -1)[keep, :])
        pressure = np.asarray(d['pressure_hPa'], dtype=np.float32).ravel()
        window = np.asarray(d['band_window_nm'], dtype=np.float64)
        centre = np.asarray(d['band_center_wavelength_nm'], dtype=np.float32).ravel()

    A = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    order = np.argsort(A['cloud_idx'])
    A = {k: v[order] for k, v in A.items()}
    n = len(A['cloud_idx'])
    print(f'{n} clouds with spectra; {len(n_failed)} failed/not run'
          + (f' (first: {n_failed[:10]})' if n_failed else ''))

    rng = np.random.default_rng(RANDOM_SEED)
    refl = A['Refl_model'].astype(np.float64)                 # (n, 636)
    refl_hysics, unc_hysics = add_gaussian_noise(refl, NOISE_HYSICS + NOISE_FM, rng)
    refl_emit, unc_emit = add_gaussian_noise(refl, NOISE_EMIT + NOISE_FM, rng)

    wv_above = np.empty(n)
    wv_in = np.empty(n)
    for i in range(n):
        wv_above[i], wv_in[i] = wv_columns(A['T_K'][i], pressure, A['vapor_cm3'][i], A['z_km'][i])

    wavelengths = ((window[:, 0] + window[:, 1]) / 2).astype(np.float32)   # (636,)
    re = A['re_um'].astype(np.float32)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('reflectances_hysics', data=refl_hysics)
        f.create_dataset('reflectances_emit', data=refl_emit)
        f.create_dataset('reflectances_uncertainty_hysics', data=unc_hysics)
        f.create_dataset('reflectances_uncertainty_emit', data=unc_emit)
        f.create_dataset('profiles', data=re)
        f.create_dataset('profiles_raw', data=re)
        f.create_dataset('profiles_raw_z', data=A['z_km'].astype(np.float32))
        f.create_dataset('profiles_raw_tau', data=np.full_like(re, np.nan))
        f.create_dataset('profile_n_levels', data=np.full(n, N_LEVELS, dtype=np.int32))
        f.create_dataset('lwc', data=A['lwc_g_per_m3'].astype(np.float32))
        f.create_dataset('alpha', data=A['alpha'].astype(np.float32))
        f.create_dataset('tau_c', data=A['tau_c'].astype(np.float32))
        for g in ('sza', 'saz', 'vza', 'vaz'):
            f.create_dataset(g, data=A[g].astype(np.float32))
        f.create_dataset('wv_above_cloud', data=wv_above)
        f.create_dataset('wv_in_cloud', data=wv_in)
        f.create_dataset('era5_vapor_concentration', data=A['vapor_cm3'].astype(np.float32))
        f.create_dataset('era5_temperature', data=A['T_K'].astype(np.float32))
        f.create_dataset('era5_pressure_levels', data=pressure)
        f.create_dataset('wavelengths', data=wavelengths)
        f.create_dataset('band_center_wavelength', data=centre)
        f.create_dataset('cloud_idx', data=A['cloud_idx'].astype(np.int32))
        f.attrs['n_chunk_files'] = len(files)
        f.attrs['n_geometries'] = 1
        f.attrs['re_global_min'] = float(re.min())
        f.attrs['re_global_max'] = float(re.max())
        f.attrs['tau_global_min'] = float(A['tau_c'].min())
        f.attrs['tau_global_max'] = float(A['tau_c'].max())
        f.attrs['max_raw_levels'] = N_LEVELS
        f.attrs['source'] = 'synthetic-cloud RT runs (run_synthetic_spectra_chunk.m)'
        f.attrs['noise_random_seed'] = RANDOM_SEED
    print(f'wrote {out_path} ({n} clouds)')


if __name__ == '__main__':
    main()
