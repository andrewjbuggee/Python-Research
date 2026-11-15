"""
15_lwp_from_retrievals.py — STEP 15 of the paper-3 pipeline
===========================================================
Liquid water path from the retrieved r_e profile and τ_c, with the Mie
extinction that defines the τ_c label (replaces plot_lwp_analysis_profile_tau.py,
which used Paper-2 Eq. 10 with Q_ext = 2).

Why the formula changed (details in paper3_physics.py)
  The τ_c label is now ∫ ext(r_e; α) · LWC dz with ext from libRadtran's custom
  Mie tables at 550 nm, and the RT sees exactly that τ_c (mixed-layer water-cloud
  files). With N and α constant with height (Paper-2's assumption),
      LWP = 1000 · τ_c · ∫ r_e³ dz / ∫ ext(r_e; α) r_e³ dz      [g m⁻²]
  (Claude's derivation; reduces to Eq. 10 when Q̄_ext = 2). Eq. 10 with Q = 2 is
  biased high by ≈ Q̄_ext/2 ≈ 1.05 on the new labels; on the pre-Nov-2026 labels
  (Q = 2.32) it was ≈ 1.16.

Methods (each applied to TRUE inputs — true r_e profile and τ_c label, which
isolates the formula's own bias — and to RETRIEVED inputs)
  mie_alpha_median : the formula above with α = the HDF5's median α
                     (operational: the network does not retrieve α)
  mie_alpha_true   : the same with each cloud's true α (diagnostic upper bound)
  eq10_q2          : Paper-2 Eq. 10, Q = 2 (the previous method, for reference)
Reference truth: LWP = 1000·∫LWC dz from the HDF5 (/lwp_g_per_m2, or from
/lwc and /profiles_raw_z for older files).

Error convention: error = estimate − truth, bias % = 100·(estimate − truth)/truth
(positive = overestimate; Paper 2's "average percent difference" is −mean(bias %)).
Error attribution for the operational method: τ-only (true profile, retrieved τ)
and profile-only (retrieved profile, true τ).

Outputs (in --out-dir, default the run dir)
  lwp_mie_stats.json, lwp_mie_diagnostics.png (2 x 2)

Usage
    python 15_lwp_from_retrievals.py --tag PT_2026-11-12 [--h5-path local.h5] [--mie-dir DIR]
    python 15_lwp_from_retrievals.py --run-dir <run dir>

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

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from paper3_physics import (DEFAULT_MIE_TABLE_DIR, lwp_constant_N_mie,  # noqa: E402
                            lwp_eq10_q2_g_m2, lwp_from_lwc_g_m2)
from plot_percentile_profiles_synthetic import _resolve_h5_path       # noqa: E402

C = {'mie_alpha_median': '#0072B2', 'mie_alpha_true': '#009E73', 'eq10_q2': '#D55E00'}
LABEL = {'mie_alpha_median': 'Mie, median α (operational)',
         'mie_alpha_true': 'Mie, true α (diagnostic)',
         'eq10_q2': 'Eq. 10, Q = 2 (previous)'}


def selected_run_dir(tag, run_dir) -> Path:
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    return (REPO / json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))['run_dir']).resolve()


def stats(bias_pct: np.ndarray, err: np.ndarray) -> dict:
    """Summary of percent bias and absolute error (g m⁻²)."""
    return dict(mean_bias_pct=float(np.mean(bias_pct)), median_bias_pct=float(np.median(bias_pct)),
                p05_bias_pct=float(np.percentile(bias_pct, 5)), p95_bias_pct=float(np.percentile(bias_pct, 95)),
                mean_err_g_m2=float(np.mean(err)), rms_err_g_m2=float(np.sqrt(np.mean(err ** 2))))


def main() -> None:
    p = argparse.ArgumentParser(description='LWP from retrieved profiles with the Mie-consistent formula')
    p.add_argument('--tag', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--h5-path', default=None)
    p.add_argument('--mie-dir', default=str(DEFAULT_MIE_TABLE_DIR))
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    run_dir = selected_run_dir(args.tag, args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    c = np.load(run_dir / 'pred_cache.npz')                   # written by step 14
    s = json.load(open(run_dir / 'summary.json'))
    h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)
    idx = c['h5_idx']
    order = np.argsort(idx)                                   # h5py needs sorted indices
    with h5py.File(h5_path, 'r') as f:
        alpha_all = f['alpha'][:]
        alpha = f['alpha'][idx[order]][np.argsort(order)]
        if 'lwp_g_per_m2' in f:
            lwp_true = f['lwp_g_per_m2'][idx[order]][np.argsort(order)].astype(np.float64)
        else:                                                 # files from before step 10
            lwc = f['lwc'][idx[order]][np.argsort(order)]
            z = f['profiles_raw_z'][idx[order]][np.argsort(order)]
            lwp_true = lwp_from_lwc_g_m2(lwc, z)
        tau_def = f.attrs.get('tau_c_definition', 'not recorded (pre-step-10 file)')
    alpha_med = float(np.median(alpha_all))
    print(f'{run_dir.name}: {len(idx)} test clouds; HDF5 {h5_path.name}; median α = {alpha_med:.2f}')
    print(f'τ_c label definition: {tau_def}')

    re_t, re_r = c['true'].astype(np.float64), c['pred'].astype(np.float64)       # (n, 7) μm
    tau_t, tau_r = c['tau_true'].astype(np.float64), c['tau_pred'].astype(np.float64)

    def lwp(method, re, tau):
        if method == 'mie_alpha_median':
            return lwp_constant_N_mie(re, tau, alpha_med, args.mie_dir)
        if method == 'mie_alpha_true':
            return lwp_constant_N_mie(re, tau, alpha, args.mie_dir)
        return lwp_eq10_q2_g_m2(re, tau)

    res, out = {}, {'run_dir': str(run_dir), 'h5_path': str(h5_path), 'n_test': int(len(idx)),
                    'alpha_median_used': alpha_med, 'tau_c_definition': str(tau_def)}
    print(f'\n{"method":<28}{"inputs":<11}{"mean bias %":>12}{"median %":>10}{"RMS g/m²":>10}')
    for m in LABEL:
        for inp, (re, tau) in {'true': (re_t, tau_t), 'retrieved': (re_r, tau_r)}.items():
            est = lwp(m, re, tau)
            st = stats(100 * (est - lwp_true) / lwp_true, est - lwp_true)
            res[(m, inp)] = est
            out[f'{m}_{inp}_inputs'] = st
            print(f'{LABEL[m]:<28}{inp:<11}{st["mean_bias_pct"]:>+12.2f}{st["median_bias_pct"]:>+10.2f}'
                  f'{st["rms_err_g_m2"]:>10.2f}')

    # error attribution for the operational method (one retrieval error at a time)
    tau_only = lwp('mie_alpha_median', re_t, tau_r)
    prof_only = lwp('mie_alpha_median', re_r, tau_t)
    base = res[('mie_alpha_median', 'true')]
    out['attribution_mie_alpha_median'] = dict(
        tau_only=stats(100 * (tau_only - base) / base, tau_only - base),
        profile_only=stats(100 * (prof_only - base) / base, prof_only - base))
    print(f'\nattribution (operational, relative to true-input LWP): RMS τ-only '
          f'{out["attribution_mie_alpha_median"]["tau_only"]["rms_err_g_m2"]:.2f}, profile-only '
          f'{out["attribution_mie_alpha_median"]["profile_only"]["rms_err_g_m2"]:.2f} g m⁻²')

    json.dump(out, open(out_dir / 'lwp_mie_stats.json', 'w'), indent=2)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    est = res[('mie_alpha_median', 'retrieved')]
    lim = [min(lwp_true.min(), est.min()) * 0.9, max(lwp_true.max(), est.max()) * 1.1]
    ax[0, 0].scatter(lwp_true, est, s=4, alpha=0.4, color=C['mie_alpha_median'], edgecolors='none')
    ax[0, 0].plot(lim, lim, 'k-', lw=0.8)
    ax[0, 0].set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='True LWP (∫LWC dz, g m$^{-2}$)',
                 ylabel='Retrieved LWP (g m$^{-2}$)', title='(a) Retrieved LWP, Mie formula, median α')
    bins = np.linspace(-40, 40, 81)
    for k, inp in enumerate(('true', 'retrieved')):
        a = ax[0, 1] if inp == 'true' else ax[1, 0]
        for m in LABEL:
            b = 100 * (res[(m, inp)] - lwp_true) / lwp_true
            a.hist(np.clip(b, bins[0], bins[-1]), bins=bins, histtype='step', lw=1.6, color=C[m],
                   label=f'{LABEL[m]} (median {np.median(b):+.1f} %)')
        a.axvline(0, color='k', lw=0.8)
        a.set(xlabel='LWP bias (%)', ylabel='Clouds',
              title=('(b) Formula bias: TRUE r$_e$ and τ$_c$' if inp == 'true'
                     else '(c) Total error: RETRIEVED r$_e$ and τ$_c$'))
        a.legend(frameon=False, fontsize=8, loc='upper left')
    for lbl, arr, col in (('τ$_c$ error only', tau_only, '#CC79A7'), ('profile error only', prof_only, '#56B4E9')):
        b = 100 * (arr - base) / base
        ax[1, 1].hist(np.clip(b, bins[0], bins[-1]), bins=bins, histtype='step', lw=1.6, color=col,
                      label=f'{lbl} (RMS {np.sqrt(np.mean((arr - base) ** 2)):.1f} g m$^{{-2}}$)')
    ax[1, 1].axvline(0, color='k', lw=0.8)
    ax[1, 1].set(xlabel='LWP change vs true-input LWP (%)', ylabel='Clouds',
                 title='(d) Error attribution (operational method)')
    ax[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'LWP from retrieved profiles — {run_dir.name} ({len(idx)} test clouds)')
    fig.savefig(out_dir / 'lwp_mie_diagnostics.png', dpi=200)
    plt.close(fig)
    print(f'\nwrote {out_dir / "lwp_mie_stats.json"} and {out_dir / "lwp_mie_diagnostics.png"}')


if __name__ == '__main__':
    main()
