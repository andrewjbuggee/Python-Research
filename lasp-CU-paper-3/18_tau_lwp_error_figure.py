"""
18_tau_lwp_error_figure.py — STEP 18 of the paper-3 pipeline
============================================================
Manuscript figure: retrieval error in cloud optical depth and in liquid water
path, each plotted against its true value, for one trained profile + τ model.
2 x 1 figure, IEEE single column (same style as step 17's per-level r_e figure).

  (a) τ_c error vs true τ_c, log x. The same quantities as panel (b) of
      tau_retrieval_diagnostics.png (plot_run_figures_profile_tau.py, written
      by step 14); the title gives the median τ_c error instead of the
      correlation coefficients.
  (b) LWP error (g m⁻²) vs true LWP, log x; the title gives the median LWP
      error.

Two versions are written, identical except for how the error is defined
(same pair as step 17's per-level figure):
  tau_lwp_error_2panel_signed.png  —  retrieved − true   (the sign convention
                                      of steps 15–17; positive = overestimate)
  tau_lwp_error_2panel_abs.png     — |retrieved − true|  (magnitude only)
Each title's median is of the error as plotted in that version, so the
|error| median is NOT the absolute value of the signed median.

LWP definitions
---------------
True LWP: the integral of the cloud's LWC profile over its vertical extent,

    LWP_true = 1000 · |∫ LWC dz|        [g m⁻²; LWC in g m⁻³, z in km]

trapezoid over the 7 levels of /lwc and /profiles_raw_z
(paper3_physics.lwp_from_lwc_g_m2, the generator's definition). Where the HDF5
also stores /lwp_g_per_m2, the two are checked to agree.

Retrieved LWP: the operational method of step 15 ('mie_alpha_median'), from the
retrieved r_e profile and the retrieved τ_c with the Mie extinction that defines
the τ_c label:

    LWP = 1000 · τ_c · ∫ r_e³ dz / ∫ ext(r_e; α) r_e³ dz       [g m⁻²]

(paper3_physics.lwp_constant_N_mie). It assumes the droplet number concentration
N and the gamma shape parameter α are constant with height, and uses one α for
every cloud: the median α of the whole HDF5, because the network does not
retrieve α. The formula is Claude's derivation (Paper-2's constant-N assumption
combined with the label's Mie extinction), not from a paper; see the
paper3_physics.py module docstring. ext(r_e; α) comes from libRadtran's custom
Mie tables at 550 nm.

Panel (b) is therefore the TOTAL LWP error: τ_c retrieval error + r_e-profile
retrieval error + the formula's own error (constant N and α). The console and
the JSON also report the formula-only part (the same formula applied to the TRUE
r_e profile and the TRUE τ_c label), so the two can be separated in the text.

Input: <run>/pred_cache.npz from step 14 (true, pred, tau_true, tau_pred,
h5_idx), the run's summary.json for the HDF5 path, and the libRadtran Mie tables.

Outputs (in --out-dir, default the run directory)
    tau_lwp_error_2panel_signed.png
    tau_lwp_error_2panel_abs.png
    tau_lwp_error_2panel_stats.json   (statistics of both error definitions)

Usage
    python 18_tau_lwp_error_figure.py --tag PT_sweep_2026-11-15
    python 18_tau_lwp_error_figure.py --run-dir <run dir> [--h5-path local.h5] [--mie-dir DIR]

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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from paper3_physics import (DEFAULT_MIE_TABLE_DIR, lwp_constant_N_mie,   # noqa: E402
                            lwp_from_lwc_g_m2)
# Same IEEE single-column style as step 17, so the manuscript figures match.
from plot_per_level_error_percentiles import setup_style                 # noqa: E402
# HDF5 path resolution shared with steps 15 and 16 (Alpine path → local
# training_data/ fallback).
from plot_percentile_profiles_synthetic import _resolve_h5_path           # noqa: E402

# ─── Figure configuration ─────────────────────────────────────────────────
FIG_WIDTH_IN  = 3.25     # IEEE single column (step 17)
FIG_HEIGHT_IN = 5.4      # two stacked scatter panels
DPI           = 500

# Okabe-Ito / Wong (2011, Nat. Methods 8, 441) vermillion: retrieval error, as in
# panel (b) of tau_retrieval_diagnostics.png and panel (a) of step 17's figure.
COLOR_ERROR = '#D55E00'
# Marker size / opacity of the original panel (b); 10,000 overlapping points
# need the transparency to show where they pile up.
MARKER_SIZE  = 6
MARKER_ALPHA = 0.35

# Labelled ticks on the log x axes. Matplotlib's default labels only the
# decades, which leaves the τ_c axis (≈ 3–40) with a single label (10¹).
# Ticks outside a panel's axis limits are dropped automatically.
TAU_TICKS     = (3, 5, 10, 20, 40)
LWP_TICKS_G_M2 = (5, 10, 20, 50, 100, 200)


# ─── Run directory ────────────────────────────────────────────────────────
def selected_run_dir(tag: str | None, run_dir: str | None) -> Path:
    """--run-dir if given, else the run step 13 selected for --tag."""
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    sel = json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))
    return (REPO / sel['run_dir']).resolve()


# ─── Statistics ───────────────────────────────────────────────────────────
def error_stats(err: np.ndarray, truth: np.ndarray) -> dict:
    """Summary of an error array (estimate − truth) in its own units, plus the
    relative error 100·err/truth in %."""
    rel_pct = 100.0 * err / truth
    return {
        'bias':            float(np.mean(err)),
        'median_error':    float(np.median(err)),
        # magnitude-only statistics (the |error| version of the figure)
        'mean_abs_error':   float(np.mean(np.abs(err))),
        'median_abs_error': float(np.median(np.abs(err))),
        'rmse':            float(np.sqrt(np.mean(err ** 2))),
        'p05_error':       float(np.percentile(err, 5)),
        'p95_error':       float(np.percentile(err, 95)),
        'mean_bias_pct':   float(np.mean(rel_pct)),
        'median_bias_pct': float(np.median(rel_pct)),
        'mape_pct':        float(np.mean(np.abs(rel_pct))),
    }


# ─── Figure ───────────────────────────────────────────────────────────────
def _signed(value: float, fmt: str) -> str:
    """'+0.03' / '−0.001' with a typographic minus (U+2212), not a hyphen."""
    return format(value, '+' + fmt).replace('-', '−')


def _error_panel(ax, truth: np.ndarray, err: np.ndarray, xlabel: str,
                 ylabel: str, title: str, x_ticks: tuple[float, ...],
                 error_kind: str) -> None:
    """One error-vs-truth scatter panel, styled like the original panel (b).

    error_kind 'signed' → dashed zero line so the sign of the bias is visible;
               'abs'    → no zero line, y axis starts at 0 (|error| ≥ 0).
    """
    ax.scatter(truth, err, s=MARKER_SIZE, alpha=MARKER_ALPHA, color=COLOR_ERROR,
               edgecolors='none', rasterized=True)   # rasterised: 10k points stay light in PDF
    if error_kind == 'signed':
        ax.axhline(0, color='k', lw=0.8, ls='--')
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xscale('log')
    # Plain-number labels at the x_ticks inside the (autoscaled, log-margin)
    # axis limits; minor ticks keep their grid lines but no labels.
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in x_ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.3, which='both')


def plot_two_panel(tau_true: np.ndarray, tau_err: np.ndarray,
                   lwp_true_g_m2: np.ndarray, lwp_err_g_m2: np.ndarray,
                   out_path: Path, error_kind: str = 'signed') -> None:
    """2 x 1 figure: (a) τ_c error vs true τ_c, (b) LWP error vs true LWP.

    tau_err and lwp_err_g_m2 are always the SIGNED errors (retrieved − true);
    error_kind 'signed' plots them as they are,
               'abs'    plots |retrieved − true|.
    Each title carries the panel letter and the median of the error as
    plotted, in place of the correlation coefficients of the original panel (b).
    """
    if error_kind == 'signed':
        tau_plot, lwp_plot_g_m2 = tau_err, lwp_err_g_m2
        tau_word, lwp_word = r'$\tau_c$ error', 'LWP error'
        tau_ylabel = r'$\tau_c$ error (retrieved $-$ true)'
        lwp_ylabel = 'LWP error (retrieved $-$ true)\n(g m$^{-2}$)'
        fmt_median = _signed                     # '+0.03' / '−0.001'
    elif error_kind == 'abs':
        tau_plot, lwp_plot_g_m2 = np.abs(tau_err), np.abs(lwp_err_g_m2)
        tau_word, lwp_word = r'$|\tau_c$ error$|$', '|LWP error|'
        tau_ylabel = r'$\tau_c$ $|$retrieved $-$ true$|$'
        lwp_ylabel = 'LWP $|$retrieved $-$ true$|$\n(g m$^{-2}$)'
        fmt_median = format                      # magnitudes: no sign
    else:
        raise ValueError(f'error_kind must be signed or abs, got {error_kind!r}')

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    med_tau = float(np.median(tau_plot))              # dimensionless
    med_lwp_g_m2 = float(np.median(lwp_plot_g_m2))    # g m⁻²
    _error_panel(ax_a, tau_true, tau_plot,
                 xlabel=r'True $\tau_c$',
                 ylabel=tau_ylabel,
                 title=fr'(a)  Median {tau_word} = {fmt_median(med_tau, ".3f")}',
                 x_ticks=TAU_TICKS, error_kind=error_kind)
    _error_panel(ax_b, lwp_true_g_m2, lwp_plot_g_m2,
                 xlabel=r'True LWP (g m$^{-2}$)',
                 ylabel=lwp_ylabel,
                 title=fr'(b)  Median {lwp_word} = {fmt_median(med_lwp_g_m2, ".2f")} '
                       r'g m$^{-2}$',
                 x_ticks=LWP_TICKS_G_M2, error_kind=error_kind)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description='τ_c and LWP retrieval-error figure (2 x 1)')
    p.add_argument('--tag', default=None, help='training_runs/<tag> (step 13 selection)')
    p.add_argument('--run-dir', default=None, help='run directory (overrides --tag)')
    p.add_argument('--h5-path', default=None,
                   help='local HDF5 if the path in summary.json is not reachable')
    p.add_argument('--mie-dir', default=str(DEFAULT_MIE_TABLE_DIR),
                   help='libRadtran custom Mie-table directory')
    p.add_argument('--out-dir', default=None, help='default: the run directory')
    args = p.parse_args()

    run_dir = selected_run_dir(args.tag, args.run_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Test-set predictions from step 14 ────────────────────────────────
    cache_path = run_dir / 'pred_cache.npz'
    if not cache_path.exists():
        raise SystemExit(f'{cache_path} not found — run step 14 first')
    c = np.load(cache_path)
    re_true_um = c['true'].astype(np.float64)          # (n, 7) μm, top → base
    re_ret_um = c['pred'].astype(np.float64)           # (n, 7) μm
    tau_true = c['tau_true'].astype(np.float64)        # (n,)
    tau_ret = c['tau_pred'].astype(np.float64)         # (n,)
    h5_idx = c['h5_idx'].astype(np.int64)              # (n,) rows of the HDF5
    n = h5_idx.size

    # ── Truth from the HDF5 ──────────────────────────────────────────────
    summary = json.load(open(run_dir / 'summary.json'))
    h5_path = _resolve_h5_path(summary['data']['h5_path'], args.h5_path)
    order = np.argsort(h5_idx)            # h5py fancy indexing needs increasing indices
    unsort = np.argsort(order)            # puts rows back in pred_cache order
    with h5py.File(h5_path, 'r') as f:
        lwc_g_m3 = f['lwc'][h5_idx[order]].astype(np.float64)[unsort]          # (n, 7)
        z_km = f['profiles_raw_z'][h5_idx[order]].astype(np.float64)[unsort]   # (n, 7)
        tau_label = f['tau_c'][h5_idx[order]].astype(np.float64)[unsort]       # (n,)
        alpha_all = f['alpha'][:].astype(np.float64)                           # every cloud
        lwp_stored = (f['lwp_g_per_m2'][h5_idx[order]].astype(np.float64)[unsort]
                      if 'lwp_g_per_m2' in f else None)

    print(f'Run      : {run_dir}')
    print(f'HDF5     : {h5_path}')
    print(f'Test set : {n:,} clouds')

    # The cache must describe the same clouds as these HDF5 rows.
    if not np.allclose(tau_true, tau_label, rtol=1e-5):
        raise SystemExit("pred_cache 'tau_true' ≠ HDF5 /tau_c for h5_idx — wrong HDF5?")

    # True LWP: integral of the LWC profile over the cloud's vertical extent.
    lwp_true_g_m2 = lwp_from_lwc_g_m2(lwc_g_m3, z_km)                         # (n,)
    if lwp_stored is not None:
        max_rel = float(np.max(np.abs(lwp_stored - lwp_true_g_m2) / lwp_true_g_m2))
        print(f'True LWP : 1000·∫LWC dz; max relative difference from /lwp_g_per_m2 '
              f'= {max_rel:.1e}')
        if max_rel > 1e-3:
            raise SystemExit('∫LWC dz disagrees with /lwp_g_per_m2 by > 0.1 % — check the HDF5')

    # ── Retrieved LWP, operational method of step 15 ─────────────────────
    alpha_median = float(np.median(alpha_all))      # one α for every cloud (not retrieved)
    lwp_ret_g_m2 = lwp_constant_N_mie(re_ret_um, tau_ret, alpha_median, args.mie_dir)
    # Same formula on TRUE inputs: the formula's own error (constant N and α),
    # with no retrieval error in it. Reported, not plotted.
    lwp_formula_g_m2 = lwp_constant_N_mie(re_true_um, tau_true, alpha_median, args.mie_dir)
    print(f'Retrieved LWP: Mie constant-N formula, median α = {alpha_median:.2f}')

    tau_err = tau_ret - tau_true                       # (n,)
    lwp_err_g_m2 = lwp_ret_g_m2 - lwp_true_g_m2        # (n,) g m⁻²

    stats = {
        'run_dir': str(run_dir),
        'h5_path': str(h5_path),
        'n_test': int(n),
        'sign_convention': 'error = retrieved - true (positive = overestimate)',
        'alpha_median_used': alpha_median,
        'tau_c': error_stats(tau_err, tau_true),
        'lwp_g_m2_total': error_stats(lwp_err_g_m2, lwp_true_g_m2),
        'lwp_g_m2_formula_only_true_inputs': error_stats(lwp_formula_g_m2 - lwp_true_g_m2,
                                                         lwp_true_g_m2),
        'tau_true_range': [float(tau_true.min()), float(tau_true.max())],
        'lwp_true_range_g_m2': [float(lwp_true_g_m2.min()), float(lwp_true_g_m2.max())],
    }

    print(f'\n{"":<34}{"bias":>9}{"median":>9}{"med|e|":>9}{"RMSE":>9}{"p05":>9}{"p95":>9}'
          f'{"mean %":>9}{"MAPE %":>9}')
    for key, name in (('tau_c', 'τ_c'),
                      ('lwp_g_m2_total', 'LWP total (g m⁻²)'),
                      ('lwp_g_m2_formula_only_true_inputs', 'LWP formula only (g m⁻²)')):
        s = stats[key]
        print(f'{name:<34}{s["bias"]:>+9.3f}{s["median_error"]:>+9.3f}'
              f'{s["median_abs_error"]:>9.3f}{s["rmse"]:>9.3f}{s["p05_error"]:>+9.3f}'
              f'{s["p95_error"]:>+9.3f}{s["mean_bias_pct"]:>+9.2f}{s["mape_pct"]:>9.2f}')

    # ── Figures (signed and |error| versions) + JSON ─────────────────────
    setup_style()
    print()
    for kind in ('signed', 'abs'):
        fig_path = out_dir / f'tau_lwp_error_2panel_{kind}.png'
        plot_two_panel(tau_true, tau_err, lwp_true_g_m2, lwp_err_g_m2, fig_path,
                       error_kind=kind)
        print(f'Figure   -> {fig_path}')

    stats_path = out_dir / 'tau_lwp_error_2panel_stats.json'
    with stats_path.open('w') as f:
        json.dump(stats, f, indent=2)
    print(f'Summary  -> {stats_path}')


if __name__ == '__main__':
    main()
