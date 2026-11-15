"""
11_check_training_hdf5.py — STEP 11 of the paper-3 pipeline
===========================================================
Sanity checks on the training HDF5 from step 10, before any GPU time is spent.

Checks (printed; the script exits with status 1 if a hard check fails)
  hard  every dataset finite where it must be, matching sample counts,
        unique cloud_idx, reflectance in [0, 1.5], τ_c in [3, 40], r_e > 0
  info  value ranges and medians of r_e, LWC, τ_c, LWP, f_ad, geometry,
        water-vapour columns; how many f_ad used the fallback base
  opt.  --check-tau N : recompute τ_c for N random clouds from the Mie tables
        (paper3_physics.tau_from_profile_mie) and compare with the label

Figure (2 x 3) <out-dir>/<h5 stem>_check.png
  (a) HySICS reflectance 5/50/95 % vs wavelength   (b) r_e profile envelope
  (c) τ_c histogram (log)   (d) LWP histogram (log)
  (e) Wood (2005) f_ad histogram with in-situ medians marked
  (f) solar / viewing zenith angles

In-situ f_ad medians for reference (Wood 2005 with the same base rule,
VOCALS-REx 0.77, ORACLES 0.67; MATLAB compute_adiabaticity_Wood2005.m, 7 Nov
2026, from Andrew's adiabaticity analysis).

Usage
    python 11_check_training_hdf5.py <file.h5> [--out-dir DIR] [--check-tau 200]
           [--mie-dir DIR]

Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

INSITU_FAD_MEDIAN = {'VOCALS-REx': 0.77, 'ORACLES': 0.67}
RANDOM_SEED = 0                  # for the --check-tau subsample
C_BLUE, C_ORANGE, C_GREEN, C_GREY = '#0072B2', '#E69F00', '#009E73', '#7f7f7f'


def summarize(name, x, units=''):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    print(f'  {name:<26} [{x.min():10.4g}, {x.max():10.4g}]  median {np.median(x):10.4g}  {units}')


def main() -> int:
    p = argparse.ArgumentParser(description='Check a training HDF5 from step 10')
    p.add_argument('h5', type=Path)
    p.add_argument('--out-dir', type=Path, default=None, help='figure folder (default: next to the HDF5)')
    p.add_argument('--check-tau', type=int, default=0, help='recompute τ for this many random clouds')
    p.add_argument('--mie-dir', type=str, default=None, help='custom Mie table folder (for --check-tau)')
    args = p.parse_args()
    out_dir = args.out_dir or args.h5.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, 'r') as f:
        D = {k: f[k][:] for k in f.keys()}
        attrs = dict(f.attrs)
    n = D['tau_c'].shape[0]
    print(f'{args.h5.name}: {n} samples, {len(D)} datasets')
    for k in ('noise_random_seed', 'tau_c_definition', 'adiabaticity_definition'):
        if k in attrs:
            print(f'  attr {k}: {attrs[k]}')

    # ── hard checks ──────────────────────────────────────────────────────────
    fails = []
    per_sample = [k for k, v in D.items() if v.ndim >= 1 and v.shape[0] == n]
    for k in ('reflectances_hysics', 'profiles', 'lwc', 'tau_c', 'sza', 'vza', 'saz', 'vaz',
              'wv_above_cloud', 'wv_in_cloud', 'era5_temperature', 'era5_vapor_concentration'):
        if k not in D:
            fails.append(f'missing dataset {k}')
        elif not np.isfinite(D[k]).all():
            fails.append(f'{k}: {int((~np.isfinite(D[k])).sum())} non-finite values')
    other_n = {k: D[k].shape[0] for k in D if D[k].ndim >= 1 and k not in per_sample
               and k not in ('wavelengths', 'band_center_wavelength', 'era5_pressure_levels')}
    if other_n:
        fails.append(f'datasets with a different sample count: {other_n}')
    if 'cloud_idx' in D and np.unique(D['cloud_idx']).size != n:
        fails.append('duplicate cloud_idx')
    r = D['reflectances_hysics']
    if r.min() < 0 or r.max() > 1.5:
        fails.append(f'reflectance outside [0, 1.5]: [{r.min():.3g}, {r.max():.3g}]')
    if D['tau_c'].min() < 3 - 1e-3 or D['tau_c'].max() > 40 + 1e-3:
        fails.append(f'tau_c outside [3, 40]: [{D["tau_c"].min():.3g}, {D["tau_c"].max():.3g}]')
    if D['profiles'].min() <= 0:
        fails.append('r_e <= 0 somewhere')

    # ── informational ranges ─────────────────────────────────────────────────
    print('\nRanges:')
    summarize('r_e', D['profiles'], 'μm')
    summarize('LWC', D['lwc'], 'g m⁻³')
    summarize('tau_c', D['tau_c'])
    if 'lwp_g_per_m2' in D:
        summarize('LWP', D['lwp_g_per_m2'], 'g m⁻²')
    summarize('alpha', D['alpha'])
    summarize('SZA / VZA', np.r_[D['sza'], D['vza']], 'deg')
    summarize('WV above cloud', D['wv_above_cloud'] * 2.991e-22, 'mm')
    summarize('reflectance (HySICS, noisy)', r)
    if 'adiabaticity' in D:
        fad = D['adiabaticity']
        summarize('Wood f_ad', fad)
        print(f'  f_ad > 1: {100 * np.nanmean(fad > 1):.1f} %   fallback base: '
              f'{100 * D["adiabaticity_used_fallback"].mean():.1f} %   NaN: {int(np.isnan(fad).sum())}')
    else:
        print('  (no adiabaticity dataset — HDF5 not built by step 10)')

    # ── optional τ re-computation from the Mie tables ────────────────────────
    if args.check_tau > 0:
        from paper3_physics import DEFAULT_MIE_TABLE_DIR, tau_from_profile_mie
        mie_dir = args.mie_dir or str(DEFAULT_MIE_TABLE_DIR)
        rng = np.random.default_rng(RANDOM_SEED)
        pick = rng.choice(n, size=min(args.check_tau, n), replace=False)
        # float() so the ratio is formed in float64: the label is stored as
        # float32, and float64/float32 in NumPy 2 rounds to float32 resolution
        rel = np.array([tau_from_profile_mie(D['profiles'][i], D['lwc'][i], D['profiles_raw_z'][i],
                                             D['alpha'][i], mie_dir) / float(D['tau_c'][i]) - 1
                        for i in pick])
        print(f'\nτ_c recomputed from Mie tables for {pick.size} clouds (seed {RANDOM_SEED}): '
              f'max |relative difference| {np.abs(rel).max():.2e}')
        if np.abs(rel).max() > 1e-3:
            fails.append(f'τ_c label differs from the Mie recomputation by up to {np.abs(rel).max():.2e}')

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    wl = D.get('band_center_wavelength', D['wavelengths'])
    pr = np.percentile(r, [5, 50, 95], axis=0)
    ax[0, 0].fill_between(wl, pr[0], pr[2], color=C_BLUE, alpha=0.3, label='5–95 %')
    ax[0, 0].plot(wl, pr[1], color=C_BLUE, label='median')
    ax[0, 0].set(xlabel='Wavelength (nm)', ylabel='Reflectance', title='(a) HySICS reflectance')
    ax[0, 0].legend(frameon=False)
    lev = np.arange(D['profiles'].shape[1], 0, -1)          # 7 = top … 1 = base
    pp = np.percentile(D['profiles'], [5, 50, 95], axis=0)
    ax[0, 1].fill_betweenx(lev, pp[0], pp[2], color=C_GREEN, alpha=0.3, label='5–95 %')
    ax[0, 1].plot(pp[1], lev, color=C_GREEN, label='median')
    ax[0, 1].set(xlabel='r$_e$ (μm)', ylabel='Level (1 = base, 7 = top)', title='(b) r$_e$ profile')
    ax[0, 1].legend(frameon=False)
    ax[0, 2].hist(D['tau_c'], bins=np.logspace(np.log10(3), np.log10(40), 50), color=C_BLUE)
    ax[0, 2].set(xscale='log', xlabel='τ$_c$', ylabel='Count', title='(c) τ$_c$ label')
    if 'lwp_g_per_m2' in D:
        lw = D['lwp_g_per_m2']
        ax[1, 0].hist(lw, bins=np.logspace(np.log10(lw.min()), np.log10(lw.max()), 50), color=C_ORANGE)
        ax[1, 0].set(xscale='log', xlabel='LWP (g m$^{-2}$)', ylabel='Count', title='(d) LWP (∫LWC dz)')
    if 'adiabaticity' in D:
        fad = D['adiabaticity'][np.isfinite(D['adiabaticity'])]
        ax[1, 1].hist(fad, bins=np.linspace(0, min(3, fad.max()), 60), color=C_GREEN)
        ax[1, 1].axvline(np.median(fad), color='k', lw=1.5, label=f'synthetic median {np.median(fad):.2f}')
        for (name, med), ls in zip(INSITU_FAD_MEDIAN.items(), ('--', ':')):
            ax[1, 1].axvline(med, color=C_GREY, ls=ls, lw=1.5, label=f'{name} median {med:.2f}')
        ax[1, 1].set(xlabel='f$_{ad}$ (Wood 2005)', ylabel='Count', title='(e) Adiabaticity')
        ax[1, 1].legend(frameon=False, fontsize=8)
    ax[1, 2].hist(D['sza'], bins=40, alpha=0.6, color=C_BLUE, label='SZA')
    ax[1, 2].hist(D['vza'], bins=40, alpha=0.6, color=C_ORANGE, label='VZA')
    ax[1, 2].set(xlabel='Angle (deg)', ylabel='Count', title='(f) Geometry')
    ax[1, 2].legend(frameon=False)
    fig.suptitle(f'{args.h5.name}: {n} samples')
    fig_path = out_dir / f'{args.h5.stem}_check.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nfigure → {fig_path}')

    if fails:
        print('\nHARD CHECK FAILURES:')
        for m in fails:
            print('  -', m)
        return 1
    print('\nAll hard checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
