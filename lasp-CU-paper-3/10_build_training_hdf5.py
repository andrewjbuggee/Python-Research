"""
10_build_training_hdf5.py — STEP 10 of the paper-3 pipeline
===========================================================
Build the training HDF5 from the per-chunk spectra files written on Alpine by
run_synthetic_spectra_chunk.m (one .mat per Slurm array task, many clouds
each). Formerly convert_synthetic_chunk_files_to_HDF.py (14 Nov 2026).

Pipeline position
  06  synthetic profiles (06_synthetic_profile_generator.py)
  07  training-input NetCDF + geometry (07_build_training_inputs.py)
  —   HySICS spectra on Alpine (MATLAB, synthetic_12_Nov_2026.sh)
  10  THIS SCRIPT: spectra + truth → training HDF5
  11  check the HDF5 (11_check_training_hdf5.py)

HDF5 layout
-----------
Identical to convert_synthetic_matFiles_to_HDF.py, so data.py and the
trainers read it unchanged:

  /reflectances_hysics, /reflectances_emit                         (n, 636) f4
  /reflectances_uncertainty_hysics, /reflectances_uncertainty_emit (n, 636) f4
  /profiles, /profiles_raw (r_e, μm), /profiles_raw_z (km), /profiles_raw_tau (NaN)  (n, 7)
  /profile_n_levels (n,) i4    /lwc (n, 7) g m⁻³    /alpha, /tau_c (n,)
  /sza, /saz, /vza, /vaz (n,) deg
  /wv_above_cloud, /wv_in_cloud (n,) molec cm⁻²
  /era5_vapor_concentration (n, 37) cm⁻³, /era5_temperature (n, 37) K
  /era5_pressure_levels (37,) hPa      /wavelengths (636,) nm

plus, new in this pipeline:
  /cloud_idx (n,) i4                 row of the training-input .nc
  /band_center_wavelength (636,) nm  Gaussian centre of each 6-nm-FWHM band
  /lwp_g_per_m2 (n,)                 truth LWP = 1000·∫LWC dz (g m⁻²)
  /adiabaticity (n,)                 Wood (2005) f_ad, 7-Nov-2026 cloud-base rule
  /z_cloud_base_wood_km (n,)         extrapolated cloud base used for f_ad (km)
  /gamma_ad_g_m3_per_m (n,)          adiabatic LWC lapse rate (g m⁻³ per m)
  /adiabaticity_used_fallback (n,) i1  1 if the fallback base was used

Adiabaticity is computed with paper3_physics.wood_adiabaticity_details, a
verbatim copy of plot_per_cloud_rmse_kfold.py (identical results, checked
14 Nov 2026), which matches the MATLAB in-situ pipeline. So in-situ and
synthetic f_ad use the same rule and the same total LWC.

The τ_c label is the Mie optical depth at 550 nm from the 06 generator
(ext from the custom Mie tables, trapezoid over the 7 levels); its definition
is stored as an attribute because the LWP step (15_) must use the same Mie
extinction (see paper3_physics.py).

Noise: measurement noise is added here to the clean reflectance with a fixed
seed (RANDOM_SEED, stored as attribute 'noise_random_seed').

Usage
    python 10_build_training_hdf5.py <chunk_dir> [<chunk_dir> ...] --out <file.h5>
           [--n-expected 100000]
    python 10_build_training_hdf5.py --stamp-tau-bounds <existing.h5>   # files built before 15 Nov 2026

Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from paper3_physics import wood_adiabaticity_details

N_LEVELS = 7
N_WAVELENGTHS = 636
TAU_C_MIN = 3.0          # defence in depth; 06 already enforces it

# τ_c normalization pair for the network, stored as HDF5 attrs
# tau_c_norm_min / tau_c_norm_max and read by data.resolve_tau_bounds.
# = the generator's τ_c rejection limits (06_synthetic_profile_generator.py
# TAU_C_MIN / TAU_C_MAX), so every file built from 06 satisfies them by
# construction.  Deliberately NOT the observed min/max of this file: the pair
# defines what the τ head's sigmoid output means, so it must not change from
# one data file to the next.  Change it only together with 06's limits.
TAU_NORM_BOUNDS = (3.0, 40.0)

# Noise levels — same values as convert_synthetic_matFiles_to_HDF.py
NOISE_HYSICS = 0.003     # 0.3 % Gaussian
NOISE_EMIT = 0.015       # 1.5 % Gaussian
NOISE_FM = 0.0           # forward-model noise budget
RANDOM_SEED = 20261114   # fixed so the noisy reflectances are reproducible

# chunk-file variables holding one value per cloud (all others hold a row per cloud)
SCALAR_KEYS = {'cloud_idx', 'done', 'sza', 'saz', 'vza', 'vaz', 'tau_c', 'alpha',
               'LWP_g_per_m2'}

R_DRY = 287.05           # J/(kg K), dry-air gas constant (WV columns, as before)
G0 = 9.80665             # m/s²


# ── water-vapour columns: unchanged from convert_synthetic_matFiles_to_HDF.py ──
def hypsometric_gph(T_K: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Geopotential height (m) from T(p) by hypsometric integration, z = 0 at
    the highest pressure: dz = (R_d / g) · T̄ · ln(p_lower / p_upper)."""
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


def wv_columns(T_K, p_hPa, vap_cm3, z_cloud_km) -> tuple[float, float]:
    """(above-cloud, in-cloud) water-vapour columns in molecules cm⁻², same
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


def add_gaussian_noise(refl, level, rng):
    """refl + level·refl·N(0,1) and the 1-sigma uncertainty level·|refl|."""
    noisy = refl + (level * refl) * rng.standard_normal(refl.shape)
    return noisy.astype(np.float32), np.abs(level * refl).astype(np.float32)


def load_chunks(chunk_dirs: list[Path]) -> tuple[dict, dict]:
    """Concatenate every finished cloud from all chunk files (sorted by cloud
    index, duplicates dropped). Returns (arrays, shared metadata)."""
    files = sorted(f for d in chunk_dirs
                   for f in d.glob('synthetic_spectra_HySICS_636bands_clouds*.mat')
                   if not f.name.endswith('_partial.mat'))
    if not files:
        raise SystemExit(f'No chunk files in {[str(d) for d in chunk_dirs]}')
    print(f'{len(files)} chunk files')
    keys = ['cloud_idx', 'done', 'Refl_model', 'sza', 'saz', 'vza', 'vaz', 're_um',
            'lwc_g_per_m3', 'z_km', 'tau_c', 'alpha', 'LWP_g_per_m2', 'T_K', 'vapor_cm3']
    parts = {k: [] for k in keys}
    meta, failed, seen = {}, [], set()
    for f in files:
        d = scipy.io.loadmat(f, squeeze_me=False)
        idx = np.asarray(d['cloud_idx']).ravel().astype(int)
        done = np.asarray(d['done']).ravel().astype(bool)
        failed += idx[~done].tolist()
        keep = done & (np.asarray(d['tau_c']).ravel() >= TAU_C_MIN)
        dup = np.isin(idx, list(seen))
        if (keep & dup).any():
            print(f'  {f.name}: {int((keep & dup).sum())} clouds already seen; skipped')
        keep &= ~dup
        seen.update(idx[keep].tolist())
        for k in keys:
            a = np.asarray(d[k])
            parts[k].append(a.ravel()[keep] if k in SCALAR_KEYS
                            else a.reshape(len(idx), -1)[keep, :])
        meta = dict(pressure=np.asarray(d['pressure_hPa'], dtype=np.float32).ravel(),
                    window=np.asarray(d['band_window_nm'], dtype=np.float64),
                    centre=np.asarray(d['band_center_wavelength_nm'], dtype=np.float32).ravel(),
                    n_files=len(files), failed=sorted(set(failed) - seen))
    A = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    order = np.argsort(A['cloud_idx'])
    return {k: v[order] for k, v in A.items()}, meta


TAU_NORM_NOTE = ('network tau normalization pair = 06_synthetic_profile_generator '
                 'TAU_C_MIN/TAU_C_MAX rejection limits (fixed by design, not scanned)')


def stamp_tau_bounds(h5_path: Path) -> None:
    """Add the τ normalization attrs to an existing step-10 file (attrs only)."""
    with h5py.File(h5_path, 'a') as f:
        tau = f['tau_c'][:]
        lo, hi = TAU_NORM_BOUNDS
        n_out = int(np.sum((tau < lo) | (tau > hi)))
        if n_out:
            raise SystemExit(f'{n_out} τ_c values outside {lo}–{hi}: not stamping {h5_path}')
        f.attrs['tau_c_norm_min'], f.attrs['tau_c_norm_max'] = TAU_NORM_BOUNDS
        f.attrs['tau_c_norm_definition'] = TAU_NORM_NOTE
    print(f'stamped {h5_path}: tau_c_norm_min/max = {TAU_NORM_BOUNDS} '
          f'(τ_c in file {tau.min():.3f}–{tau.max():.3f})')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('chunk_dirs', nargs='*', type=Path, help='folder(s) holding chunk .mat files')
    p.add_argument('--out', type=Path, help='output HDF5 path')
    p.add_argument('--stamp-tau-bounds', type=Path, default=None, metavar='H5',
                   help='only add the tau_c_norm_min/max attrs to an HDF5 built by this '
                        'script before Nov-15-2026, then exit (no chunk files needed)')
    p.add_argument('--n-expected', type=int, default=None,
                   help='number of clouds in the training-input .nc (e.g. 100000): '
                        'reports any cloud index with no spectrum')
    args = p.parse_args()

    if args.stamp_tau_bounds is not None:
        stamp_tau_bounds(args.stamp_tau_bounds)
        return
    if not args.chunk_dirs or args.out is None:
        p.error('chunk_dirs and --out are required (unless --stamp-tau-bounds)')

    A, meta = load_chunks(args.chunk_dirs)
    n = len(A['cloud_idx'])
    print(f'{n} clouds with spectra; {len(meta["failed"])} failed in RT'
          + (f' (first: {meta["failed"][:10]})' if meta['failed'] else ''))
    if args.n_expected:
        missing = np.setdiff1d(np.arange(1, args.n_expected + 1), A['cloud_idx'])
        print(f'cloud indices with no spectrum: {missing.size} of {args.n_expected}'
              + (f' (first: {missing[:10].tolist()})' if missing.size else ''))

    # ---- noise (seeded) ------------------------------------------------------
    print(f'Noise random seed: {RANDOM_SEED}')
    rng = np.random.default_rng(RANDOM_SEED)
    refl = A['Refl_model'].astype(np.float64)                         # (n, 636)
    refl_hysics, unc_hysics = add_gaussian_noise(refl, NOISE_HYSICS + NOISE_FM, rng)
    refl_emit, unc_emit = add_gaussian_noise(refl, NOISE_EMIT + NOISE_FM, rng)

    # ---- per-cloud derived quantities: WV columns and Wood (2005) adiabaticity ----
    pressure = meta['pressure']                                       # (37,) hPa
    wv_above, wv_in = np.empty(n), np.empty(n)
    f_ad, z_cb_km, gamma_ad, fallback = (np.empty(n), np.empty(n), np.empty(n),
                                         np.zeros(n, dtype=np.int8))
    for i in range(n):
        wv_above[i], wv_in[i] = wv_columns(A['T_K'][i], pressure, A['vapor_cm3'][i], A['z_km'][i])
        w = wood_adiabaticity_details(A['lwc_g_per_m3'][i], A['z_km'][i], A['T_K'][i], pressure)
        f_ad[i], gamma_ad[i], fallback[i] = w['f_ad'], w['gamma_ad_g_m3_per_m'], w['used_fallback']
        z_cb_km[i] = w['z_cb_m'] / 1000.0
        if (i + 1) % 10000 == 0:
            print(f'  derived quantities: {i + 1}/{n}')
    print(f'adiabaticity: median {np.nanmedian(f_ad):.2f}, f_ad > 1: '
          f'{100 * np.nanmean(f_ad > 1):.1f} %, fallback base: {100 * fallback.mean():.1f} %')

    wavelengths = ((meta['window'][:, 0] + meta['window'][:, 1]) / 2).astype(np.float32)
    re = A['re_um'].astype(np.float32)

    with h5py.File(args.out, 'w') as f:
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
        f.create_dataset('band_center_wavelength', data=meta['centre'])
        f.create_dataset('cloud_idx', data=A['cloud_idx'].astype(np.int32))
        f.create_dataset('lwp_g_per_m2', data=A['LWP_g_per_m2'].astype(np.float32))
        f.create_dataset('adiabaticity', data=f_ad.astype(np.float32))
        f.create_dataset('z_cloud_base_wood_km', data=z_cb_km.astype(np.float32))
        f.create_dataset('gamma_ad_g_m3_per_m', data=gamma_ad.astype(np.float32))
        f.create_dataset('adiabaticity_used_fallback', data=fallback)
        f.attrs['n_chunk_files'] = meta['n_files']
        f.attrs['n_geometries'] = 1
        f.attrs['re_global_min'] = float(re.min())
        f.attrs['re_global_max'] = float(re.max())
        f.attrs['tau_global_min'] = float(A['tau_c'].min())
        f.attrs['tau_global_max'] = float(A['tau_c'].max())
        f.attrs['tau_c_norm_min'], f.attrs['tau_c_norm_max'] = TAU_NORM_BOUNDS
        f.attrs['tau_c_norm_definition'] = TAU_NORM_NOTE
        f.attrs['max_raw_levels'] = N_LEVELS
        f.attrs['source'] = 'synthetic-cloud RT runs (run_synthetic_spectra_chunk.m)'
        f.attrs['noise_random_seed'] = RANDOM_SEED
        f.attrs['tau_c_definition'] = ('Mie optical depth at 550 nm: trapz over levels of '
                                       'ext(r_e; alpha) * LWC, custom 1-50 um Mie tables '
                                       '(06_synthetic_profile_generator.compute_tau_c)')
        f.attrs['adiabaticity_definition'] = ('Wood (2005) f_ad = LWP_obs/LWP_ad; cloud base from a '
                                              'line fit to the bottom third of LWC(z), fallback '
                                              'z_lowest - 0.05/Gamma_ad (paper3_physics.py)')
    print(f'wrote {args.out} ({n} clouds)')


if __name__ == '__main__':
    main()
