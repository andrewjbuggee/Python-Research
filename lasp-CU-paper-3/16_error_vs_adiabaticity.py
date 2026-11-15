"""
16_error_vs_adiabaticity.py — STEP 16 of the paper-3 pipeline
=============================================================
Do the retrieval errors depend on how adiabatic the cloud is?

For every test cloud of one trained model (pred_cache.npz from step 14):
  profile RMSE (μm, mean over the 7 levels), τ_c error (retrieved − true),
  LWP error (%, step 15's operational Mie formula, median α)
against Wood (2005) adiabaticity f_ad (7-Nov-2026 cloud-base rule; read from
/adiabaticity in step-10 HDF5 files, recomputed with paper3_physics for older
files). Clouds whose base came from the fallback rule are shown separately,
since their f_ad rests on an assumed base.

Reported: Pearson r and Spearman ρ of each error with f_ad, and median errors
in f_ad bins (< 0.5, 0.5–0.8, 0.8–1.0, 1.0–1.5, > 1.5). The in-situ medians
(VOCALS-REx 0.77, ORACLES 0.67) are marked for reference.

Outputs (in --out-dir, default the run dir)
  error_vs_adiabaticity.png (1 x 3), error_vs_adiabaticity_stats.json

Usage
    python 16_error_vs_adiabaticity.py --tag PT_2026-11-12 [--h5-path local.h5]
    python 16_error_vs_adiabaticity.py --run-dir <run dir>

Nothing here is stochastic.
Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from paper3_physics import (DEFAULT_MIE_TABLE_DIR, lwp_constant_N_mie,  # noqa: E402
                            lwp_from_lwc_g_m2, wood_adiabaticity_details)
from plot_percentile_profiles_synthetic import _resolve_h5_path       # noqa: E402

FAD_BINS = [0, 0.5, 0.8, 1.0, 1.5, np.inf]
INSITU_FAD_MEDIAN = {'VOCALS-REx': 0.77, 'ORACLES': 0.67}
C_MAIN, C_FALLBACK, C_MED = '#0072B2', '#E69F00', '#000000'


def selected_run_dir(tag, run_dir) -> Path:
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    return (REPO / json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))['run_dir']).resolve()


def read_rows(f, name, idx):
    """f[name][idx] for an unsorted index array (h5py needs sorted indices)."""
    order = np.argsort(idx)
    return f[name][idx[order]][np.argsort(order)]


def main() -> None:
    p = argparse.ArgumentParser(description='Retrieval error vs Wood (2005) adiabaticity')
    p.add_argument('--tag', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--h5-path', default=None)
    p.add_argument('--mie-dir', default=str(DEFAULT_MIE_TABLE_DIR))
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    run_dir = selected_run_dir(args.tag, args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    c = np.load(run_dir / 'pred_cache.npz')
    s = json.load(open(run_dir / 'summary.json'))
    h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)
    idx = c['h5_idx']
    with h5py.File(h5_path, 'r') as f:
        alpha_med = float(np.median(f['alpha'][:]))
        lwc = read_rows(f, 'lwc', idx)
        z = read_rows(f, 'profiles_raw_z', idx)
        if 'adiabaticity' in f:
            f_ad = read_rows(f, 'adiabaticity', idx).astype(np.float64)
            fallback = read_rows(f, 'adiabaticity_used_fallback', idx).astype(bool)
            lwp_true = read_rows(f, 'lwp_g_per_m2', idx).astype(np.float64)
            source = 'HDF5 /adiabaticity (step 10)'
        else:                                         # older HDF5: compute now, same rule
            T = read_rows(f, 'era5_temperature', idx)
            P = f['era5_pressure_levels'][:]
            det = [wood_adiabaticity_details(lwc[i], z[i], T[i], P) for i in range(len(idx))]
            f_ad = np.array([d['f_ad'] for d in det])
            fallback = np.array([d['used_fallback'] for d in det])
            lwp_true = lwp_from_lwc_g_m2(lwc, z)
            source = 'recomputed with paper3_physics (pre-step-10 HDF5)'
    print(f'{run_dir.name}: {len(idx)} test clouds; f_ad from {source}; '
          f'median {np.nanmedian(f_ad):.2f}, fallback base {100 * fallback.mean():.1f} %')

    prof_rmse = np.sqrt(np.mean((c['pred'] - c['true']) ** 2, axis=1))            # μm
    tau_err = (c['tau_pred'] - c['tau_true']).astype(np.float64)
    lwp_ret = lwp_constant_N_mie(c['pred'], c['tau_pred'], alpha_med, args.mie_dir)
    lwp_err_pct = 100 * (lwp_ret - lwp_true) / lwp_true
    errors = {'profile RMSE (μm)': prof_rmse, 'τ_c error (retrieved − true)': tau_err,
              'LWP error (%)': lwp_err_pct}
    axis_label = {'profile RMSE (μm)': 'profile RMSE (μm)',
                  'τ_c error (retrieved − true)': 'τ$_c$ error (retrieved − true)',
                  'LWP error (%)': 'LWP error (%)'}

    ok = np.isfinite(f_ad)
    stats = {'n_test': int(len(idx)), 'f_ad_source': source, 'f_ad_median': float(np.nanmedian(f_ad)),
             'fallback_fraction': float(fallback.mean()), 'bins': FAD_BINS[:-1] + ['inf'], 'errors': {}}
    print(f'\n{"error":<32}{"Pearson r":>10}{"Spearman ρ":>12}   median per f_ad bin {FAD_BINS[1:-1]}')
    for name, e in errors.items():
        r, _ = pearsonr(f_ad[ok], e[ok])
        rho, _ = spearmanr(f_ad[ok], e[ok])
        b = np.digitize(f_ad[ok], FAD_BINS[1:-1])
        med = [float(np.median(e[ok][b == k])) if np.any(b == k) else None for k in range(len(FAD_BINS) - 1)]
        cnt = [int(np.sum(b == k)) for k in range(len(FAD_BINS) - 1)]
        stats['errors'][name] = dict(pearson_r=float(r), spearman_rho=float(rho), bin_median=med, bin_count=cnt)
        print(f'{name:<32}{r:>+10.3f}{rho:>+12.3f}   '
              + '  '.join('—' if m is None else f'{m:+.3g}' for m in med))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    x_hi = float(np.nanpercentile(f_ad, 99.5))
    for a, (name, e) in zip(ax, errors.items()):
        a.scatter(f_ad[ok & ~fallback], e[ok & ~fallback], s=4, alpha=0.35, color=C_MAIN,
                  edgecolors='none', label='fitted cloud base')
        a.scatter(f_ad[ok & fallback], e[ok & fallback], s=6, alpha=0.6, color=C_FALLBACK,
                  edgecolors='none', label='fallback cloud base')
        b = np.digitize(f_ad[ok], FAD_BINS[1:-1])
        centres = [np.median(f_ad[ok][b == k]) for k in range(len(FAD_BINS) - 1) if np.any(b == k)]
        meds = [np.median(e[ok][b == k]) for k in range(len(FAD_BINS) - 1) if np.any(b == k)]
        a.plot(centres, meds, 'o-', color=C_MED, lw=1.8, ms=5, label='bin median')
        for (lbl, med), ls in zip(INSITU_FAD_MEDIAN.items(), ('--', ':')):
            a.axvline(med, color='0.5', ls=ls, lw=1, label=f'{lbl} median f$_{{ad}}$')
        a.axhline(0, color='k', lw=0.6)
        st = stats['errors'][name]
        a.set(xlim=(0, x_hi), xlabel='Wood (2005) adiabaticity f$_{ad}$', ylabel=axis_label[name],
              title=f'{axis_label[name]}\nPearson r = {st["pearson_r"]:+.2f}, Spearman ρ = {st["spearman_rho"]:+.2f}')
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Retrieval error vs adiabaticity — {run_dir.name} ({len(idx)} test clouds)')
    fig.savefig(out_dir / 'error_vs_adiabaticity.png', dpi=200)
    plt.close(fig)
    json.dump(stats, open(out_dir / 'error_vs_adiabaticity_stats.json', 'w'), indent=2)
    print(f'\nwrote {out_dir / "error_vs_adiabaticity.png"} and error_vs_adiabaticity_stats.json')


if __name__ == '__main__':
    main()
