"""
diagnose_lwp_error_cancellation.py — why the LWP error from RETRIEVED inputs is
smaller than the LWP formula's own error on TRUE inputs.

Context (step 18, run_047 of PT_sweep_2026-11-15)
-------------------------------------------------
The operational LWP of step 15,

    LWP = 1000 · τ_c · ∫ r_e³ dz / ∫ ext(r_e; α_med) r_e³ dz        [g m⁻²]

(paper3_physics.lwp_constant_N_mie; constant N and one α for every cloud; the
formula is Claude's derivation, see paper3_physics.py) has a mean error of
+1.4 % (RMSE 2.0 g m⁻²) when applied to the TRUE r_e profile and the TRUE τ_c,
but only +0.35 % (RMSE 1.6 g m⁻²) when applied to the RETRIEVED ones. This
script finds out why.

What it does (the analysis design is Claude's, Nov 2026; not from a paper)
--------------------------------------------------------------------------
1. Exact log-error decomposition. With W(r; α) = ∫ r³ dz / ∫ ext(r; α) r³ dz
   (so LWP_formula = 1000 · τ · W) and W_true = LWP_true / (1000 · τ_true),

     ln(LWP_formula(ret) / LWP_true) =  ln(τ_ret / τ_true)                  τ term
                                      + ln(W(r_ret; α_med) / W(r_true; α_med))   profile term
                                      + ln(W(r_true; α_med) / W(r_true; α_true)) α term
                                      + ln(W(r_true; α_true) / W_true)       N(z) term

   The last two terms together are the formula-only error. The N(z) term is
   non-zero only because N ∝ LWC / r_e³ varies with height (at fixed α).
   The covariance matrix of the four terms shows which retrieval term cancels
   which formula term.

2. Partial correlation. Does the profile term still anti-correlate with the
   N(z) term after removing everything a flexible function of the TRUE r_e
   profile (plus α and τ) can explain? If yes, the network's profile error
   depends on N(z) — which is never a training label — so the information can
   only have reached the network through the spectrum.

3. What the radiative transfer sees. The synthetic spectra are computed from
   write_wc_file_mixed_layers.m, which turns the 7 sampled levels into 6
   homogeneous slabs, each the well-mixed contents of its two bounding levels:
       LWC_mix = (LWC_k + LWC_k+1) / 2
       r_e,mix = LWC_mix / [ (LWC_k / r_e,k + LWC_k+1 / r_e,k+1) / 2 ]
   (mixing rule from that file's header — Claude's derivation from the
   Hansen & Travis 1974 definitions, eqs. 2.48 and 2.53). The script applies
   the same constant-N formula to these slab r_e values and compares N's
   within-cloud variation at level and slab scale.

4. Per level, regresses the network's error (r_ret − r_true) on
   (r_seen − r_true), where r_seen is the cross-section-weighted mean of the
   slab mixtures next to that level (Claude's construction), controlling for
   shrinkage toward the test-set mean profile.

Inputs: <run>/pred_cache.npz (step 14), the run's HDF5 (lwc, profiles_raw_z,
alpha, tau_c) and the libRadtran custom Mie tables.

Outputs (in --out-dir, default the run directory)
    lwp_error_cancellation.png         2 × 2 figure
    lwp_error_cancellation_stats.json  every number printed

Usage
    python diagnose_lwp_error_cancellation.py \\
        --run-dir training_runs/PT_sweep_2026-11-15/results/run_047

Nothing here is stochastic.
Author: Andrew J. Buggee (with Claude), 21 Nov 2026
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
from paper3_physics import (DEFAULT_MIE_TABLE_DIR, load_mie_ext_tables,  # noqa: E402
                            lwp_constant_N_mie)
from plot_percentile_profiles_synthetic import _resolve_h5_path          # noqa: E402
from plot_run004_figures import setup_style_paper                         # noqa: E402

DEFAULT_RUN_DIR = 'training_runs/PT_sweep_2026-11-15/results/run_047'
DPI = 500

# Okabe-Ito colours; every series also has its own line style / marker so
# identity is not colour-alone.
C_LEVEL = '#D55E00'   # vermillion — true level values
C_SLAB = '#0072B2'    # blue       — slab mixtures the RT sees
C_RET = '#009E73'     # green      — network retrieval
C_SCAT = '#0072B2'

# Example cloud for panel (d): the one at this percentile of the N(z) term
# (a large but not extreme formula error; the rule is fixed, not hand-picked).
EXAMPLE_PERCENTILE = 98.0


# ─── Mie extinction, vectorised ───────────────────────────────────────────
def ext_per_lwc_many(r_e_um: np.ndarray, alpha, mie_dir: str) -> np.ndarray:
    """ext(r_e; α) [km⁻¹ per g m⁻³] for many profiles at once.

    Same rule as paper3_physics.ext_per_lwc (table with the nearest α,
    linear in r_e, r_e clipped to the table range), vectorised over clouds.
    r_e_um (n, L); alpha scalar or (n,). Returns (n, L).
    """
    alphas, tables = load_mie_ext_tables(mie_dir)
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (r_e_um.shape[0],))
    nearest = alphas[np.abs(a[:, None] - alphas[None, :]).argmin(axis=1)].astype(int)
    out = np.empty_like(r_e_um)
    for a_tab in np.unique(nearest):
        rows = nearest == a_tab
        reff_um, ext = tables[int(a_tab)]
        out[rows] = np.interp(np.clip(r_e_um[rows], reff_um[0], reff_um[-1]), reff_um, ext)
    return out


def w_formula_km(r_e_um: np.ndarray, alpha, mie_dir: str, layers: bool = False) -> np.ndarray:
    """W = ∫ r³ dz / ∫ ext(r; α) r³ dz, so LWP_formula = 1000 · τ_c · W.

    layers=False: 7 levels, trapezoid on unit spacing (lwp_constant_N_mie).
    layers=True : equal-thickness slabs, plain sums (each slab is homogeneous).
    Units: km (ext is per km per g m⁻³), the 1000 converts km → m.
    """
    ext = ext_per_lwc_many(r_e_um, alpha, mie_dir)
    if layers:
        return (r_e_um ** 3).sum(1) / (ext * r_e_um ** 3).sum(1)
    return np.trapezoid(r_e_um ** 3, axis=1) / np.trapezoid(ext * r_e_um ** 3, axis=1)


def slab_mixtures(r_e_um: np.ndarray, lwc_g_m3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(LWC_mix, r_e,mix) of the N−1 slabs write_wc_file_mixed_layers.m builds
    from N levels; arrays (n, N−1)."""
    xs = lwc_g_m3 / r_e_um                                  # ∝ droplet cross-section
    lwc_mix = 0.5 * (lwc_g_m3[:, :-1] + lwc_g_m3[:, 1:])
    r_mix = lwc_mix / (0.5 * (xs[:, :-1] + xs[:, 1:]))
    return lwc_mix, r_mix


def signed(value: float, fmt: str) -> str:
    """'+0.76' / '−0.76' with a typographic minus (U+2212) for figure text."""
    return format(value, '+' + fmt).replace('-', '−')


def pct_stats(e: np.ndarray) -> dict:
    """Mean / sd of 100·(log ratio) — ≈ percent error for small errors."""
    return {'mean_pct': float(100 * e.mean()), 'sd_pct': float(100 * e.std())}


def gm2_stats(err_g_m2: np.ndarray) -> dict:
    return {'mean_g_m2': float(err_g_m2.mean()), 'rmse_g_m2': float(np.sqrt(np.mean(err_g_m2 ** 2)))}


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description='Why the retrieved-input LWP error is smaller than the formula error')
    p.add_argument('--run-dir', default=DEFAULT_RUN_DIR)
    p.add_argument('--h5-path', default=None)
    p.add_argument('--mie-dir', default=str(DEFAULT_MIE_TABLE_DIR))
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir = (run_dir if run_dir.is_absolute() else REPO / run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test-set predictions and truth ──────────────────────────────
    c = np.load(run_dir / 'pred_cache.npz')
    r_t = c['true'].astype(np.float64)            # (n, 7) μm, top → base
    r_r = c['pred'].astype(np.float64)            # (n, 7) μm
    tau_t = c['tau_true'].astype(np.float64)      # (n,)
    tau_r = c['tau_pred'].astype(np.float64)      # (n,)
    h5_idx = c['h5_idx'].astype(np.int64)
    n = h5_idx.size
    summary = json.load(open(run_dir / 'summary.json'))
    h5_path = _resolve_h5_path(summary['data']['h5_path'], args.h5_path)
    order, unsort = np.argsort(h5_idx), None
    unsort = np.argsort(order)
    with h5py.File(h5_path, 'r') as f:
        lwc = f['lwc'][h5_idx[order]].astype(np.float64)[unsort]            # (n, 7) g m⁻³
        z_km = f['profiles_raw_z'][h5_idx[order]].astype(np.float64)[unsort]
        alpha_t = f['alpha'][h5_idx[order]].astype(np.float64)[unsort]      # (n,)
        tau_label = f['tau_c'][h5_idx[order]].astype(np.float64)[unsort]
        alpha_med = float(np.median(f['alpha'][:]))                        # operational α
    if not np.allclose(tau_t, tau_label, rtol=1e-5):
        raise SystemExit("pred_cache tau_true ≠ HDF5 tau_c — wrong HDF5?")
    lwp_true = 1000.0 * np.abs(np.trapezoid(lwc, z_km, axis=1))           # g m⁻²
    print(f'Run {run_dir.name}: {n:,} test clouds, median α = {alpha_med:.2f}')

    # The label τ must be the Mie trapezoid of the true profile (step 10's
    # definition); otherwise the decomposition below is not exact.
    tau_chk = np.abs(np.trapezoid(ext_per_lwc_many(r_t, alpha_t, args.mie_dir) * lwc, z_km, axis=1))
    print(f'τ label = ∫ext·LWC dz: max relative difference {np.max(np.abs(tau_chk / tau_t - 1)):.1e}')

    # ── 1. exact log-error decomposition ─────────────────────────────────
    W_true = lwp_true / (1000.0 * tau_t)
    W_tm = w_formula_km(r_t, alpha_med, args.mie_dir)    # true profile, operational α
    W_tt = w_formula_km(r_t, alpha_t, args.mie_dir)      # true profile, true α
    W_rm = w_formula_km(r_r, alpha_med, args.mie_dir)    # retrieved profile, operational α
    terms = {
        'tau':     np.log(tau_r / tau_t),
        'profile': np.log(W_rm / W_tm),
        'alpha':   np.log(W_tm / W_tt),
        'N_z':     np.log(W_tt / W_true),
    }
    total = np.log(1000.0 * tau_r * W_rm / lwp_true)
    assert np.allclose(total, sum(terms.values()), atol=1e-10)
    # cross-check against the step-15 implementation on a subset
    chk = lwp_constant_N_mie(r_r[:200], tau_r[:200], alpha_med, args.mie_dir)
    assert np.allclose(chk, 1000.0 * tau_r[:200] * W_rm[:200], rtol=1e-10)

    keys = list(terms)
    M = 100.0 * np.array([terms[k] for k in keys])        # (4, n) in ~%
    cov, corr = np.cov(M), np.corrcoef(M)
    print('\n1. Log-error terms, 100·ln(ratio) ≈ %:')
    print(f'   {"term":<9}{"mean":>9}{"sd":>8}')
    for k in keys:
        print(f'   {k:<9}{100 * terms[k].mean():>+9.3f}{100 * terms[k].std():>8.3f}')
    formula = terms['alpha'] + terms['N_z']
    print(f'   {"formula":<9}{100 * formula.mean():>+9.3f}{100 * formula.std():>8.3f}   (α + N_z)')
    print(f'   {"total":<9}{100 * total.mean():>+9.3f}{100 * total.std():>8.3f}')
    print('   correlation matrix:')
    print('   ' + ' ' * 9 + ''.join(f'{k:>9}' for k in keys))
    for i, k in enumerate(keys):
        print(f'   {k:<9}' + ''.join(f'{corr[i, j]:>+9.3f}' for j in range(len(keys))))
    slope = float(np.polyfit(terms['N_z'], terms['profile'], 1)[0])
    print(f'   OLS slope of profile term on N_z term: {slope:+.3f}')

    # ── 2. partial correlation given the true profile ────────────────────
    lr = np.log(r_t)
    cols = [np.ones(n)]
    for i in range(7):
        cols += [lr[:, i], lr[:, i] ** 2, lr[:, i] ** 3]
        cols += [lr[:, i] * lr[:, j] for j in range(i + 1, 7)]
    cols += [np.log(alpha_t), np.log(tau_t), np.log(tau_t) ** 2]
    X = np.column_stack(cols)                                 # (n, 46)

    def resid(y: np.ndarray) -> np.ndarray:
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    pcorr = float(np.corrcoef(resid(terms['profile']), resid(terms['N_z']))[0, 1])
    r2_N = float(1 - resid(terms['N_z']).var() / terms['N_z'].var())
    print(f'\n2. corr(profile, N_z) = {corr[1, 3]:+.3f}; partial corr given the true r_e profile '
          f'({X.shape[1]}-term basis) = {pcorr:+.3f}  [basis explains {r2_N:.0%} of N_z variance]')

    # ── 3. what the RT sees: the constant-N formula on the slab mixtures ─
    lwc_mix, r_mix = slab_mixtures(r_t, lwc)
    e_slab = np.log(w_formula_km(r_mix, alpha_t, args.mie_dir, layers=True) / W_true)
    e_ret_true_tau = terms['N_z'] + terms['alpha'] + terms['profile']     # retrieved r_e, true τ
    N_lev = lwc / r_t ** 3                                                 # ∝ N at fixed α
    N_slab = lwc_mix / r_mix ** 3
    cv = lambda N: N.std(1) / N.mean(1)                                    # noqa: E731
    lwp_err = {
        'formula_true_levels':     1000 * tau_t * W_tm - lwp_true,
        'formula_slab_mixtures':   1000 * tau_t * w_formula_km(r_mix, alpha_med, args.mie_dir, True) - lwp_true,
        'formula_retrieved_true_tau': 1000 * tau_t * W_rm - lwp_true,
        'total_retrieved':         1000 * tau_r * W_rm - lwp_true,
    }
    print('\n3. Constant-N formula vs ∫LWC dz, by which r_e profile it is given:')
    print(f'   true LEVEL values (α true)     : {100 * terms["N_z"].mean():+.3f} % ± {100 * terms["N_z"].std():.3f}')
    print(f'   SLAB mixtures the RT sees      : {100 * e_slab.mean():+.3f} % ± {100 * e_slab.std():.3f}')
    print(f'   RETRIEVED profile, true τ      : {100 * e_ret_true_tau.mean():+.3f} % ± {100 * e_ret_true_tau.std():.3f}')
    print('   in g m⁻² (operational α):')
    for k, v in lwp_err.items():
        s = gm2_stats(v)
        print(f'     {k:<28}: mean {s["mean_g_m2"]:+.3f}  RMSE {s["rmse_g_m2"]:.3f}')
    edge = lambda N, a, b: float(np.median(N[:, a] / N[:, b].mean(1)))    # noqa: E731
    print(f'   within-cloud CV of N: levels {np.median(cv(N_lev)):.3f}, slabs {np.median(cv(N_slab)):.3f}; '
          f'N(edge)/N(interior): top level {edge(N_lev, 0, slice(1, 6)):.2f} → top slab '
          f'{edge(N_slab, 0, slice(1, 5)):.2f}, base level {edge(N_lev, 6, slice(1, 6)):.2f} → base slab '
          f'{edge(N_slab, 5, slice(1, 5)):.2f}')

    # ── 4. per level: does the network report what the slabs show? ───────
    xs_slab = 0.5 * (lwc[:, :-1] / r_t[:, :-1] + lwc[:, 1:] / r_t[:, 1:])  # slab cross-section
    seen = np.empty_like(r_t)
    seen[:, 0], seen[:, -1] = r_mix[:, 0], r_mix[:, -1]
    for L in range(1, 6):
        w1, w2 = xs_slab[:, L - 1], xs_slab[:, L]
        seen[:, L] = (w1 * r_mix[:, L - 1] + w2 * r_mix[:, L]) / (w1 + w2)
    r_pop = r_t.mean(0)
    per_level = []
    print('\n4. Per level: (r_ret − r_true) = a·(r_seen − r_true) + b·(r_pop − r_true) + c')
    print('   level      a        b      R²')
    for L in range(7):
        y = r_r[:, L] - r_t[:, L]
        XL = np.column_stack([seen[:, L] - r_t[:, L], r_pop[L] - r_t[:, L], np.ones(n)])
        co = np.linalg.lstsq(XL, y, rcond=None)[0]
        r2 = float(1 - np.var(y - XL @ co) / np.var(y))
        per_level.append({'level': L + 1, 'a_seen': float(co[0]), 'b_shrink': float(co[1]), 'r2': r2})
        print(f'   {L + 1:>3}   {co[0]:+7.3f}  {co[1]:+7.3f}  {r2:6.3f}')

    # ── Figure ────────────────────────────────────────────────────────────
    setup_style_paper()
    fig, ax = plt.subplots(2, 2, figsize=(6.9, 6.2))

    # (a) profile term vs N(z) term
    a = ax[0, 0]
    a.scatter(100 * terms['N_z'], 100 * terms['profile'], s=4, alpha=0.25, color=C_SCAT,
              edgecolors='none', rasterized=True)
    lim = np.percentile(np.abs(100 * np.r_[terms['N_z'], terms['profile']]), 99.8)
    xx = np.array([-lim, lim])
    a.plot(xx, -xx, color='0.4', lw=0.8, ls='--', label='full cancellation (slope −1)')
    b0 = np.polyfit(100 * terms['N_z'], 100 * terms['profile'], 1)
    a.plot(xx, np.polyval(b0, xx), color='k', lw=1.0, label=f'OLS fit (slope {signed(b0[0], ".2f")})')
    a.set(xlim=xx, ylim=xx, xlabel='formula error from N(z), true profile (%)',
          ylabel='change from retrieved profile (%)')
    a.set_title(f'(a)  correlation {signed(corr[1, 3], ".2f")}  '
                f'(partial, given r$_e$: {signed(pcorr, ".2f")})', fontsize=8.5)
    a.legend(loc='upper right', frameon=False, fontsize=6.5)
    a.grid(alpha=0.3)

    # (b) formula error by which profile it is given
    a = ax[0, 1]
    bins = np.linspace(-10, 15, 76)
    for e, col, ls, lab in ((terms['N_z'], C_LEVEL, '-', 'true level values'),
                            (e_slab, C_SLAB, ':', 'slab mixtures (what the RT sees)'),
                            (e_ret_true_tau, C_RET, '--', 'retrieved profile (true τ$_c$)')):
        v = np.clip(100 * e, bins[0], bins[-1])
        a.hist(v, bins=bins, histtype='step', lw=1.3, color=col, ls=ls,
               label=f'{lab}\n   mean {100 * e.mean():+.2f} %, sd {100 * e.std():.2f} %')
    a.axvline(0, color='k', lw=0.6)
    a.set(xlabel='constant-N formula vs ∫LWC dz (%)', ylabel='clouds')
    a.set_ylim(0, a.get_ylim()[1] * 1.45)
    a.set_title('(b)  formula error, by input profile', fontsize=8.5)
    a.legend(loc='upper right', frameon=False, fontsize=6.0)
    a.grid(alpha=0.3)

    # (c) N(z) relative to the cloud mean: levels vs slabs
    a = ax[1, 0]
    lev_pos, slab_pos = np.arange(1, 8), np.arange(1, 7) + 0.5
    for N, pos, col, mk, lab in ((N_lev, lev_pos, C_LEVEL, 'o-', 'levels'),
                                 (N_slab, slab_pos, C_SLAB, 's--', 'slabs')):
        Nn = N / N.mean(1, keepdims=True)
        lo, med, hi = np.percentile(Nn, [10, 50, 90], axis=0)
        a.fill_betweenx(pos, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(med, pos, mk, color=col, ms=4, lw=1.2, label=lab)
    a.axvline(1, color='k', lw=0.6)
    a.invert_yaxis()
    a.set_yticks(lev_pos)
    # Extra room on the right keeps the legend clear of the 90th-percentile band.
    a.set_xlim(0.0, 2.5)
    a.set(xlabel='N / cloud-mean N  (∝ LWC / r$_e^3$)', ylabel='level (1 = cloud top)')
    a.set_title('(c)  N(z): median and 10–90 %, levels vs slabs', fontsize=8.5)
    a.legend(loc='upper right', frameon=False, fontsize=6.5)
    a.grid(alpha=0.3)

    # (d) example cloud
    a = ax[1, 1]
    k = int(np.argmin(np.abs(terms['N_z'] - np.percentile(terms['N_z'], EXAMPLE_PERCENTILE))))
    a.plot(r_t[k], lev_pos, 'o-', color=C_LEVEL, ms=4, lw=1.2, label='true level values')
    for s in range(6):
        a.plot([r_mix[k, s]] * 2, [s + 1, s + 2], color=C_SLAB, lw=2.2, ls='-',
               label='slab mixtures (RT)' if s == 0 else None)
    a.plot(r_r[k], lev_pos, 'd--', color=C_RET, ms=4, lw=1.2, label='retrieved')
    a.invert_yaxis()
    a.set_yticks(lev_pos)
    a.set(xlabel=r'$r_e$ ($\mu$m)', ylabel='level (1 = cloud top)')
    a.set_title(f'(d)  example cloud ({EXAMPLE_PERCENTILE:.0f}th pct of N(z) error)\n'
                f'formula: true {100 * terms["N_z"][k]:+.1f} %, slabs {100 * e_slab[k]:+.1f} %, '
                f'retrieved {100 * e_ret_true_tau[k]:+.1f} %', fontsize=8)
    a.legend(loc='best', frameon=True, framealpha=1.0, edgecolor='none', fontsize=6.5)
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / 'lwp_error_cancellation.png'
    fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure -> {fig_path}')

    stats = {
        'run_dir': str(run_dir), 'n_test': int(n), 'alpha_median_used': alpha_med,
        'convention': '100*ln(estimate/truth) (~%); g m-2 errors = estimate - truth',
        'terms': {k: pct_stats(v) for k, v in terms.items()},
        'formula_only': pct_stats(formula), 'total': pct_stats(total),
        'correlation_matrix': {'order': keys, 'values': corr.tolist()},
        'covariance_matrix_pct2': {'order': keys, 'values': cov.tolist()},
        'slope_profile_on_N_z': slope,
        'partial_corr_profile_N_z_given_true_profile': pcorr,
        'r2_N_z_explained_by_true_profile_basis': r2_N,
        'formula_on_true_levels': pct_stats(terms['N_z']),
        'formula_on_slab_mixtures': pct_stats(e_slab),
        'formula_on_retrieved_true_tau': pct_stats(e_ret_true_tau),
        'lwp_errors_g_m2': {k: gm2_stats(v) for k, v in lwp_err.items()},
        'N_within_cloud_cv_median': {'levels': float(np.median(cv(N_lev))),
                                     'slabs': float(np.median(cv(N_slab)))},
        'per_level_seen_regression': per_level,
        'example_cloud_index_in_cache': k,
    }
    with (out_dir / 'lwp_error_cancellation_stats.json').open('w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats  -> {out_dir / "lwp_error_cancellation_stats.json"}')


if __name__ == '__main__':
    main()
