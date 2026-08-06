"""
plot_lwp_analysis_profile_tau.py — liquid-water-path analysis for a
PROFILE + τ_c sweep run.

LWP is computed with Eq. (10) of Buggee & Pilewskie (Paper 2, Remote Sens.
2026): assuming constant droplet number concentration with altitude and
geometric optics (Q_ext = 2),

    LWP = (2/3) · ρ_w · τ_c · ∫ r_e(z)^3 dz / ∫ r_e(z)^2 dz

With r_e in μm and ρ_w = 1000 kg m⁻³ this reduces to

    LWP [g m⁻²] = (2/3) · τ_c · ( ∫r³dz / ∫r²dz  in μm ).

Integration note
----------------
Both integrals are evaluated with TRAPEZOIDAL quadrature.  The 7 retrieval
levels are evenly spaced in altitude between each profile's cloud top and
base (verified on the 8 May HDF5: per-profile spacing spread < 4e-7 km,
i.e. float32 rounding), so with spacing h

    ∫r³dz ≈ h·(½r₀³ + r₁³ + … + ½r₆³),   ∫r²dz ≈ h·(½r₀² + r₁² + … + ½r₆²)

and the SCALAR h multiplies both weighted sums, cancelling in the ratio —
the trapezoidal weights themselves do not cancel and are retained
(np.trapezoid with unit spacing).  This holds per profile for any r_e(z) and
any cloud thickness, but only because each profile's grid is uniform.
Verified numerically against the per-profile z grids (profiles_raw_z):
max |Δ ratio| ≈ 1e-5 μm across the dataset.

Definitions
-----------
    true LWP        : Eq. 10 applied to the TRUE 7-level profile + TRUE τ_c
    retrieved LWP   : Eq. 10 applied to the RETRIEVED profile + RETRIEVED τ_c
    LWC-integrated LWP : 1000 · |∫ LWC(z) dz|  (LWC in g m⁻³, z in km) — the
        model-truth analogue of Paper 2's in situ LWP; free of Eq. 10's
        constant-N assumption
    retrieval error = true − retrieved   (positive ⇒ retrieval underestimates)
    average percent difference = mean of 100·(true − retrieved)/true —
        the Paper-2 Table-3 metric (there: in situ − retrieval)

Outputs (written into the run directory)
----------------------------------------
    lwp_retrieval_diagnostics.png    — 2×2: retrieved-vs-true, error histogram
                                        (mean/median marked), error vs true
                                        LWP, relative-error histogram
    lwp_error_vs_predictors_6panel.png — same 3×2 predictor layout as the
                                        published per-profile-RMSE 6-panel
                                        (τ_c, WV above, Wood adiabaticity,
                                        std r_e, drizzle top/bottom), y = LWP
                                        retrieval error
    lwp_stats.json                   — all printed statistics, for reporting

The console also prints an error attribution: LWP recomputed with only the
τ error active (true profile + retrieved τ) and only the profile error active
(retrieved profile + true τ), showing which retrieval drives the LWP error.

Usage:
    python plot_lwp_analysis_profile_tau.py \\
        --run-dir hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004

Requires pred_cache.npz in the run dir (written by
plot_run_figures_profile_tau.py); runs inference itself if missing.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'hyper_parameter_sweep'))

from plot_percentile_profiles_synthetic import _resolve_h5_path, _pearson_r
from plot_per_cloud_rmse_kfold import compute_predictors
from plot_run004_figures import setup_style_paper, _spearman_rho

DPI = 500

# Same CVD-safe palette / panel order as the published per-profile 6-panel
COLOR_TAUC        = '#0072B2'
COLOR_WV_ABOVE    = '#CC79A7'
COLOR_WOOD        = '#117733'
COLOR_RE_STD      = '#332288'
COLOR_DRIZZLE_TOP = '#56B4E9'
COLOR_DRIZZLE_BOT = '#D55E00'
COLOR_SCATTER     = '#0072B2'
COLOR_HIST        = '#009E73'
COLOR_ERR         = '#D55E00'


# ─────────────────────────────────────────────────────────────────────────────
# Eq. 10
# ─────────────────────────────────────────────────────────────────────────────
def lwp_eq10_g_m2(r_e_um: np.ndarray, tau_c: np.ndarray) -> np.ndarray:
    """LWP (g m⁻²) from Eq. 10 with constant-N assumption and Q_ext = 2.

    Parameters
    ----------
    r_e_um : (n, n_levels) droplet effective radius profile, μm,
             on EVENLY z-spaced levels (spacing cancels in the ratio).
    tau_c  : (n,) cloud optical thickness.
    """
    num = np.trapezoid(r_e_um ** 3, axis=1)     # ∝ ∫ r³ dz
    den = np.trapezoid(r_e_um ** 2, axis=1)     # ∝ ∫ r² dz
    return (2.0 / 3.0) * tau_c * (num / den)


def lwp_from_lwc_g_m2(h5_path: Path, h5_idx: np.ndarray) -> np.ndarray:
    """LWC-integrated LWP (g m⁻²) for the given HDF5 sample indices.

    LWP = 1000 · |∫ LWC dz|  with LWC in g m⁻³ on the 7-level grid and z in
    km (profiles_raw_z, cloud top → base).  Same convention as
    plot_per_cloud_rmse_kfold.wood_adiabaticity's LWP_obs.
    """
    sort_order = np.argsort(h5_idx)
    unsort     = np.argsort(sort_order)
    sorted_idx = h5_idx[sort_order]
    with h5py.File(h5_path, 'r') as f:
        lwc  = f['lwc'][sorted_idx].astype(np.float64)             # (n, 7)
        z_km = f['profiles_raw_z'][sorted_idx].astype(np.float64)  # (n, 7)
    lwc, z_km = lwc[unsort], z_km[unsort]
    return 1000.0 * np.abs(np.trapezoid(lwc, z_km, axis=1))


def _stats(x: np.ndarray) -> dict:
    return {
        'mean':   float(np.mean(x)),
        'median': float(np.median(x)),
        'std':    float(np.std(x)),
        'p05':    float(np.percentile(x, 5)),
        'p25':    float(np.percentile(x, 25)),
        'p75':    float(np.percentile(x, 75)),
        'p95':    float(np.percentile(x, 95)),
        'rms':    float(np.sqrt(np.mean(x ** 2))),
        'mean_abs':   float(np.mean(np.abs(x))),
        'median_abs': float(np.median(np.abs(x))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def plot_lwp_diagnostics(lwp_true, lwp_ret, err, lwp_lwc, out_path: Path) -> None:
    """2×3 diagnostics: Eq.10 retrieval error (left/mid columns) plus the
    LWC-integrated-truth comparison (right column)."""
    setup_style_paper()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))

    rel_pct = 100.0 * err / np.maximum(lwp_true, 1e-9)
    rms  = float(np.sqrt(np.mean(err ** 2)))
    var  = float(np.var(lwp_true))
    r2   = 1.0 - float(np.mean(err ** 2)) / var if var > 0 else float('nan')

    # (a) retrieved vs true (log-log)
    ax = axes[0, 0]
    ax.scatter(lwp_true, lwp_ret, s=6, alpha=0.35, color=COLOR_SCATTER,
               edgecolors='none')
    lims = (min(lwp_true.min(), lwp_ret.min()) * 0.9,
            max(lwp_true.max(), lwp_ret.max()) * 1.1)
    ax.plot(lims, lims, color='k', lw=0.8, ls='--', label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims);    ax.set_ylim(lims)
    ax.set_xlabel(r'true LWP (g m$^{-2}$)')
    ax.set_ylabel(r'retrieved LWP (g m$^{-2}$)')
    ax.set_title(fr'(a)  RMS error = {rms:.1f} g m$^{{-2}}$,  $R^2$ = {r2:.3f}',
                 fontsize=8.5)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.3, which='both')

    # (b) signed-error histogram, mean/median marked
    ax = axes[0, 1]
    clip = np.clip(err, *np.percentile(err, [0.5, 99.5]))
    ax.hist(clip, bins=60, color=COLOR_HIST, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axvline(np.mean(err), color=COLOR_ERR, lw=1.2,
               label=fr'mean = {np.mean(err):+.2f}')
    ax.axvline(np.median(err), color=COLOR_ERR, lw=1.2, ls=':',
               label=fr'median = {np.median(err):+.2f}')
    ax.set_xlabel(r'LWP retrieval error, true $-$ retrieved (g m$^{-2}$)')
    ax.set_ylabel('count')
    ax.set_title('(b)  signed retrieval error', fontsize=8.5)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(alpha=0.3)

    # (c) Eq.10 truth vs LWC-integrated truth — the constant-N assumption bias
    ax = axes[0, 2]
    ab_err  = lwp_lwc - lwp_true                 # LWC truth − Eq.10 truth
    ab_pct  = 100.0 * ab_err / np.maximum(lwp_lwc, 1e-9)
    ax.scatter(lwp_lwc, lwp_true, s=6, alpha=0.35, color=COLOR_WOOD,
               edgecolors='none')
    lims = (min(lwp_lwc.min(), lwp_true.min()) * 0.9,
            max(lwp_lwc.max(), lwp_true.max()) * 1.1)
    ax.plot(lims, lims, color='k', lw=0.8, ls='--', label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims);    ax.set_ylim(lims)
    ax.set_xlabel(r'LWC-integrated LWP (g m$^{-2}$)')
    ax.set_ylabel(r'Eq. 10 true LWP (g m$^{-2}$)')
    ax.set_title(fr'(c)  constant-$N$ assumption bias'  '\n'
                 fr'      avg % diff = {np.mean(ab_pct):+.1f} %  '
                 fr'({np.mean(ab_err):+.1f} g m$^{{-2}}$)', fontsize=8.5)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.3, which='both')

    # (d) error vs true LWP
    ax = axes[1, 0]
    ax.scatter(lwp_true, err, s=6, alpha=0.35, color=COLOR_ERR,
               edgecolors='none')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'true LWP (g m$^{-2}$)')
    ax.set_ylabel(r'retrieval error (g m$^{-2}$)')
    r   = _pearson_r(lwp_true, err)
    rho = _spearman_rho(lwp_true, err)
    ax.set_title(fr'(d)  Pearson $r$ = {r:+.3f}'  '\n'
                 fr'      Spearman $\rho$ = {rho:+.3f}', fontsize=8.5)
    ax.grid(alpha=0.3, which='both')

    # (e) relative-error histogram
    ax = axes[1, 1]
    clip = np.clip(rel_pct, *np.percentile(rel_pct, [0.5, 99.5]))
    ax.hist(clip, bins=60, color=COLOR_RE_STD, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    med = float(np.median(rel_pct))
    p05, p95 = np.percentile(rel_pct, [5, 95])
    ax.set_xlabel('relative LWP retrieval error (%)')
    ax.set_ylabel('count')
    ax.set_title(fr'(e)  median = {med:+.1f} %'  '\n'
                 fr'      5–95 % : [{p05:+.1f}, {p95:+.1f}] %', fontsize=8.5)
    ax.grid(alpha=0.3)

    # (f) error vs LWC-integrated truth — Paper-2-comparable total error
    ax = axes[1, 2]
    err_lwc = lwp_lwc - lwp_ret
    rel_lwc = 100.0 * err_lwc / np.maximum(lwp_lwc, 1e-9)
    clip = np.clip(err_lwc, *np.percentile(err_lwc, [0.5, 99.5]))
    ax.hist(clip, bins=60, color=COLOR_DRIZZLE_BOT, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axvline(np.mean(err_lwc), color='k', lw=1.2,
               label=fr'mean = {np.mean(err_lwc):+.2f}')
    ax.axvline(np.median(err_lwc), color='k', lw=1.2, ls=':',
               label=fr'median = {np.median(err_lwc):+.2f}')
    ax.set_xlabel(r'LWC truth $-$ retrieved (g m$^{-2}$)')
    ax.set_ylabel('count')
    ax.set_title(fr'(f)  vs LWC-integrated truth'  '\n'
                 fr'      avg % diff = {np.mean(rel_lwc):+.1f} %',
                 fontsize=8.5)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(alpha=0.3)

    fig.suptitle(fr'LWP retrieval diagnostics — Eq. 10, test set '
                 fr'({len(lwp_true):,} profiles)', fontsize=9.5, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_lwp_error_6panel(err: np.ndarray, predictors: dict,
                          out_path: Path) -> None:
    """Same predictor layout as the published per-profile-RMSE 6-panel,
    with y = signed LWP retrieval error."""
    setup_style_paper()
    fig, axes = plt.subplots(3, 2, figsize=(6.9, 6.5))

    def _scatter(ax, x, name, color, label, xlog=False):
        valid = np.isfinite(x) & np.isfinite(err)
        ax.scatter(x[valid], err[valid], s=6, alpha=0.55,
                   color=color, edgecolors='none')
        ax.axhline(0, color='k', lw=0.6, ls='--')
        if xlog:
            ax.set_xscale('log')
        r   = _pearson_r(x, err)
        rho = _spearman_rho(x, err)
        ax.set_xlabel(name)
        ax.set_ylabel(r'LWP error (g m$^{-2}$)')
        ax.set_title(fr'({label})  Pearson $r$ = {r:+.3f}'  '\n'
                     fr'      Spearman $\rho$ = {rho:+.3f}',
                     fontsize=8.5)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    _scatter(axes[0, 0], predictors['tau_c'],
             r'$\tau_c$', COLOR_TAUC, 'a', xlog=True)
    _scatter(axes[0, 1], predictors['wv_above_mm'],
             'above-cloud column water vapor (mm)', COLOR_WV_ABOVE, 'b')
    _scatter(axes[1, 0], predictors['wood_adiab'],
             r'adiabaticity:  '
             r'$\mathrm{LWP}_{\mathrm{obs}}/\mathrm{LWP}_{\mathrm{ad}}$',
             COLOR_WOOD, 'c')
    _scatter(axes[1, 1], predictors['re_std'],
             r'std of $r_e$ profile ($\mu$m)', COLOR_RE_STD, 'd')
    _scatter(axes[2, 0], predictors['drizzle_proxy_top'],
             r'max $r_e$ in upper 30 % ($\mu$m)', COLOR_DRIZZLE_TOP, 'e')
    _scatter(axes[2, 1], predictors['drizzle_proxy'],
             r'max $r_e$ in lower 30 % ($\mu$m)', COLOR_DRIZZLE_BOT, 'f')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-dir', required=True,
                   help='PT-sweep run directory (e.g. .../run_004).')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f'run-dir does not exist: {run_dir}')
    cache_path = run_dir / 'pred_cache.npz'

    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)

    if not cache_path.exists() or args.refresh:
        print('pred_cache.npz missing/refresh — running inference...')
        from plot_run_figures_profile_tau import _run_inference
        out = _run_inference(run_dir, args.h5_path, args.device, args.seed)
        np.savez_compressed(cache_path,
                            pred=out['pred'], pred_std=out['pred_std'],
                            true=out['true'], tau_pred=out['tau_pred'],
                            tau_pred_std=out['tau_pred_std'],
                            tau_true=out['tau_true'], h5_idx=out['h5_idx'])
    c = np.load(cache_path)
    if 'tau_pred' not in c.files:
        raise RuntimeError('pred_cache.npz has no τ arrays — re-run '
                           'plot_run_figures_profile_tau.py with --refresh.')
    prof_pred, prof_true = c['pred'], c['true']
    tau_pred,  tau_true  = c['tau_pred'], c['tau_true']
    h5_idx = c['h5_idx']
    h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)

    n = prof_true.shape[0]
    print(f'Test set: {n:,} profiles × {prof_true.shape[1]} levels')

    # ── Eq. 10 LWP, true and retrieved; LWC-integrated truth ───────────
    lwp_true = lwp_eq10_g_m2(prof_true, tau_true)
    lwp_ret  = lwp_eq10_g_m2(prof_pred, tau_pred)
    err      = lwp_true - lwp_ret                # true − retrieved
    rel_pct  = 100.0 * err / np.maximum(lwp_true, 1e-9)

    lwp_lwc  = lwp_from_lwc_g_m2(h5_path, h5_idx)   # assumption-free truth
    err_lwc  = lwp_lwc - lwp_ret                    # LWC truth − retrieved
    rel_lwc  = 100.0 * err_lwc / np.maximum(lwp_lwc, 1e-9)
    assum    = lwp_lwc - lwp_true                   # constant-N assumption bias
    rel_assum = 100.0 * assum / np.maximum(lwp_lwc, 1e-9)

    # Error attribution: one retrieval error active at a time
    lwp_tau_only  = lwp_eq10_g_m2(prof_true, tau_pred)   # τ error only
    lwp_prof_only = lwp_eq10_g_m2(prof_pred, tau_true)   # profile error only
    err_tau  = lwp_true - lwp_tau_only
    err_prof = lwp_true - lwp_prof_only

    st_true, st_ret = _stats(lwp_true), _stats(lwp_ret)
    st_err, st_rel  = _stats(err), _stats(rel_pct)
    st_etau, st_eprof = _stats(err_tau), _stats(err_prof)
    st_lwc, st_errlwc = _stats(lwp_lwc), _stats(err_lwc)
    st_rellwc, st_assum, st_relassum = (_stats(rel_lwc), _stats(assum),
                                        _stats(rel_assum))

    print(f'\nLWP (Eq. 10, g m⁻²):')
    print(f'  true      : mean {st_true["mean"]:7.1f}  median {st_true["median"]:7.1f}  '
          f'p05–p95 [{st_true["p05"]:.1f}, {st_true["p95"]:.1f}]')
    print(f'  retrieved : mean {st_ret["mean"]:7.1f}  median {st_ret["median"]:7.1f}  '
          f'p05–p95 [{st_ret["p05"]:.1f}, {st_ret["p95"]:.1f}]')
    print(f'\nLWP retrieval error (true − retrieved, g m⁻²):')
    print(f'  mean      : {st_err["mean"]:+8.2f}')
    print(f'  median    : {st_err["median"]:+8.2f}')
    print(f'  std       : {st_err["std"]:8.2f}   RMS: {st_err["rms"]:.2f}')
    print(f'  IQR       : [{st_err["p25"]:+.2f}, {st_err["p75"]:+.2f}]')
    print(f'  5–95 %    : [{st_err["p05"]:+.2f}, {st_err["p95"]:+.2f}]')
    print(f'  mean |err|: {st_err["mean_abs"]:.2f}   median |err|: {st_err["median_abs"]:.2f}')
    print(f'\nRelative error (%): mean {st_rel["mean"]:+.2f}, '
          f'median {st_rel["median"]:+.2f}, '
          f'5–95 % [{st_rel["p05"]:+.1f}, {st_rel["p95"]:+.1f}]')

    # ── Paper-2-style average percent difference ───────────────────────
    print(f'\nAverage percent difference, mean of 100·(truth − retrieved)/truth')
    print(f'  vs Eq. 10 truth (pure retrieval error) : '
          f'{st_rel["mean"]:+6.1f} %  ({st_err["mean"]:+.2f} g m⁻²)')
    print(f'  vs LWC-integrated truth (incl. const-N): '
          f'{st_rellwc["mean"]:+6.1f} %  ({st_errlwc["mean"]:+.2f} g m⁻²)'
          f'   [median {st_errlwc["median"]:+.2f} g m⁻²]')
    print(f'  constant-N assumption alone (LWC − Eq10 truth): '
          f'{st_relassum["mean"]:+6.1f} %  ({st_assum["mean"]:+.2f} g m⁻²)')
    print(f'  Paper 2, Table 3 (N=69, in situ truth) : '
          f'GNOE −26.9 % (−11.6) | TBLUT −45.2 % (−20.3) | '
          f'TBLUT-WH −21.1 % (−8.4)')

    print(f'\nLWC-integrated LWP: mean {st_lwc["mean"]:.1f}, '
          f'median {st_lwc["median"]:.1f} g m⁻²')
    print(f'\nError attribution (one retrieval error active at a time):')
    print(f'  τ error only       : RMS {st_etau["rms"]:6.2f}  '
          f'mean {st_etau["mean"]:+6.2f}  median {st_etau["median"]:+6.2f}')
    print(f'  profile error only : RMS {st_eprof["rms"]:6.2f}  '
          f'mean {st_eprof["mean"]:+6.2f}  median {st_eprof["median"]:+6.2f}')
    print('  (components interact nonlinearly; they need not sum to the total)')

    # ── Predictors + correlation table ─────────────────────────────────
    print('\nComputing predictors (τ_c, WV, Wood adiab, ...)...')
    predictors = compute_predictors(h5_idx, h5_path)

    corr_rows = [
        ('tau_c',             r'cloud optical thickness τ_c'),
        ('adiab_score',       'adiabaticity (Pearson r of r_e³ vs z)'),
        ('wood_adiab',        'Wood (2005) adiabaticity'),
        ('re_std',            'std of r_e profile'),
        ('drizzle_proxy_top', 'max r_e upper 30 %'),
        ('drizzle_proxy',     'max r_e lower 30 %'),
        ('wv_above_mm',       'above-cloud WV (mm)'),
        ('wv_in_mm',          'in-cloud WV (mm)'),
    ]
    print('\nLWP retrieval error vs predictors:')
    print(f'  {"predictor":<40s} {"Pearson r":>10s} {"Spearman ρ":>11s}')
    print('  ' + '-' * 63)
    corr_out = {}
    for key, name in corr_rows:
        r   = _pearson_r(predictors[key], err)
        rho = _spearman_rho(predictors[key], err)
        corr_out[key] = {'pearson_r': r, 'spearman_rho': rho}
        print(f'  {name:<40s} {r:>+10.3f} {rho:>+11.3f}')

    # ── Figures ─────────────────────────────────────────────────────────
    out1 = run_dir / 'lwp_retrieval_diagnostics.png'
    plot_lwp_diagnostics(lwp_true, lwp_ret, err, lwp_lwc, out1)
    print(f'\n(1) LWP diagnostics        -> {out1}')

    out2 = run_dir / 'lwp_error_vs_predictors_6panel.png'
    plot_lwp_error_6panel(err, predictors, out2)
    print(f'(2) LWP error vs predictors -> {out2}')

    # ── Stats JSON for paper reporting ─────────────────────────────────
    stats = {
        'run_id':        s['run_id'],
        'n_test':        int(n),
        'equation':      'LWP = (2/3) rho_w tau_c int(r^3 dz)/int(r^2 dz)  '
                         '[Paper-2 Eq. 10; Q_ext=2, constant N]',
        'sign_convention': 'error = true - retrieved',
        'avg_percent_difference': {
            'definition': 'mean of 100*(truth - retrieved)/truth, per sample '
                          '(Paper-2 Table-3 metric)',
            'vs_eq10_truth_pct': st_rel['mean'],
            'vs_lwc_integrated_truth_pct': st_rellwc['mean'],
            'constant_N_assumption_only_pct': st_relassum['mean'],
            'paper2_table3_reference': {
                'GNOE': {'pct': -26.9, 'g_m2': -11.6},
                'TBLUT': {'pct': -45.2, 'g_m2': -20.3},
                'TBLUT_WoodHartmann': {'pct': -21.1, 'g_m2': -8.4},
                'note': 'N=69, truth = in situ LWC-integrated LWP',
            },
        },
        'lwp_true_g_m2':      st_true,
        'lwp_retrieved_g_m2': st_ret,
        'lwp_lwc_integrated_g_m2': st_lwc,
        'error_g_m2':         st_err,
        'error_relative_pct': st_rel,
        'error_vs_lwc_truth_g_m2':       st_errlwc,
        'error_vs_lwc_truth_relative_pct': st_rellwc,
        'constant_N_assumption_bias_g_m2': st_assum,
        'constant_N_assumption_bias_pct':  st_relassum,
        'error_tau_only_g_m2':     st_etau,
        'error_profile_only_g_m2': st_eprof,
        'correlations_vs_error':   corr_out,
    }
    with (run_dir / 'lwp_stats.json').open('w') as f:
        json.dump(stats, f, indent=2)
    print(f'(3) stats                  -> {run_dir / "lwp_stats.json"}')


if __name__ == '__main__':
    main()
