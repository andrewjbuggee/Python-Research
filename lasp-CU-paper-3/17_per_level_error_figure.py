"""
17_per_level_error_figure.py — STEP 17 of the paper-3 pipeline
==============================================================
Manuscript figure: per-level r_e retrieval error and predicted uncertainty
for one trained model, as a 2 x 1 figure.

  (a) Retrieval error at each vertical level over the held-out test set:
      the median (line + markers) and the 5th–95th percentile range (shading).
  (b) Mean (or median) predicted aleatoric σ at each level over the test set.
      The network outputs a Gaussian for r_e at each level given the
      spectrum, p(r_e(l) | R) = N(μ(l), σ(l)²); σ is that Gaussian's standard
      deviation, written σ(r_e(l) | R) on the axis.
      A dotted line marks the smallest σ the network can output, set by the
      run's log-σ clamp: exp(log_std_min) × (re_max − re_min), 0.888 μm for the
      legacy −4. A σ curve lying on that line is the clamp, not an estimate
      (17 Nov 2026: 92–98 % of run_004's mid-cloud predictions sit on it).
      --no-floor-line drops it.

Two versions are written, identical except for panel (a):
  per_level_error_2panel_abs.png     — |retrieved − true|
  per_level_error_2panel_signed.png  —  retrieved − true   (the same sign
                                        convention as steps 15 and 16)

Input: <run>/pred_cache.npz from step 14 (pred, pred_std, true; μm) and the
run's summary.json (or config.json) for its log-σ clamp. No HDF5 needed.

Usage
    python 17_per_level_error_figure.py --tag PT_2026-11-14
    python 17_per_level_error_figure.py --run-dir <run dir> [--sigma-stat median]

Nothing here is stochastic.
Author: Andrew J. Buggee, 17 Nov 2026 (panel (b) simplified 19 Nov 2026)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
# Same IEEE single-column style as the earlier per-level figure
# (plot_per_level_error_percentiles.py), so the manuscript figures match.
from plot_per_level_error_percentiles import setup_style   # noqa: E402

# ─── Figure configuration ─────────────────────────────────────────────────
FIG_WIDTH_IN  = 3.25     # IEEE single column
FIG_HEIGHT_IN = 6.0      # two stacked panels
DPI           = 500

# Okabe-Ito / Wong (2011, Nat. Methods 8, 441) colour-blind-safe colours.
COLOR_ERROR  = '#D55E00'  # vermillion — retrieval error (panel a)
COLOR_SIGMA  = '#0072B2'  # blue — predicted aleatoric σ (panel b)
BAND_ALPHA   = 0.25

PERCENTILES = (5, 50, 95)

# σ of the network's conditional Gaussian for r_e at level l given spectrum R.
LABEL_SIGMA = r'Aleatoric $\sigma(r_e(l)\,|\,\mathbf{R})$ ($\mu$m)'

# r_e normalization range of the network (models.RetrievalConfig re_max − re_min);
# the r_e σ floor is exp(log_std_min) times this.
RE_RANGE_UM = 50.0 - 1.5
LEGACY_LOG_STD_MIN = -4.0   # = models.LEGACY_LOG_STD_BOUNDS[0]


# ─── Run directory / metadata ─────────────────────────────────────────────
def selected_run_dir(tag: str | None, run_dir: str | None) -> Path:
    """--run-dir if given, else the run step 13 selected for --tag."""
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    sel = json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))
    return (REPO / sel['run_dir']).resolve()


def run_log_std_min(run_dir: Path) -> float:
    """log_std_min the run was trained with (legacy −4 if not recorded).

    Same rule as models.log_std_bounds_for_run, read here without importing
    torch so the figure can be made on any machine.
    """
    for name in ('summary.json', 'config.json'):
        p = run_dir / name
        if p.exists():
            hp = json.load(open(p)).get('hyperparams', {})
            return float(hp.get('log_std_min', LEGACY_LOG_STD_MIN))
    return LEGACY_LOG_STD_MIN


# Shared x-axis range and tick locations for both panels (μm).
X_RANGE_UM = (0.0, 4.0)
X_TICKS_UM = np.arange(0, 5, 1)                          # 0, 1, 2, 3, 4 μm


# ─── Figure ───────────────────────────────────────────────────────────────
def plot_two_panel(pred: np.ndarray, true: np.ndarray, pred_std: np.ndarray,
                   out_path: Path, error_kind: str = 'abs',
                   sigma_stat: str = 'mean',
                   sigma_floor_um: float | None = None) -> dict:
    """2 x 1 figure. pred, true, pred_std: (n_samples, n_levels) in μm.

    error_kind 'abs'    → panel (a) shows |pred − true|
               'signed' → panel (a) shows  pred − true
    sigma_floor_um      → if given, a dotted line at the clamp floor in (b)
    """
    err = pred - true                                    # (n_samp, n_lev), μm
    if error_kind == 'abs':
        err = np.abs(err)
        err_word = '|error|'
        xlabel_a = r'Per-level $|r_e^{\mathrm{ret}} - r_e^{\mathrm{true}}|$ ($\mu$m)'
    elif error_kind == 'signed':
        err_word = 'error'
        xlabel_a = r'Per-level $r_e^{\mathrm{ret}} - r_e^{\mathrm{true}}$ ($\mu$m)'
    else:
        raise ValueError(f'error_kind must be abs or signed, got {error_kind!r}')

    n_levels = pred.shape[1]
    levels   = np.arange(1, n_levels + 1)                # 1 = cloud top
    p05, p50, p95 = np.percentile(err, PERCENTILES, axis=0)  # each (n_lev,)

    # Central value of the predicted σ over the test set at each level.
    if sigma_stat == 'mean':
        sigma_c = pred_std.mean(axis=0)                  # (n_lev,)
        sigma_word = 'Mean'
    elif sigma_stat == 'median':
        sigma_c = np.median(pred_std, axis=0)            # (n_lev,)
        sigma_word = 'Median'
    else:
        raise ValueError(f'sigma_stat must be mean or median, got {sigma_stat!r}')

    # The fixed x range can clip data; warn rather than silently hide it.
    if error_kind == 'abs' and p95.max() > X_RANGE_UM[1]:
        print(f'WARNING: panel (a) 95th percentile reaches {p95.max():.2f} μm, '
              f'beyond the {X_RANGE_UM[1]:.0f} μm axis limit')
    if sigma_c.max() > X_RANGE_UM[1]:
        print(f'WARNING: panel (b) σ reaches {sigma_c.max():.2f} μm, '
              f'beyond the {X_RANGE_UM[1]:.0f} μm axis limit')

    # Two lines: on one line the label is taller than a 3.25"-wide panel.
    level_label = f'Vertical level\n(1 = cloud top, {n_levels} = cloud base)'
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # ── (a) error ─────────────────────────────────────────────────────────
    ax_a.fill_betweenx(levels, p05, p95, color=COLOR_ERROR, alpha=BAND_ALPHA,
                       linewidth=0, label=f'5–95% of {err_word}')
    ax_a.plot(p50, levels, 'o-', color=COLOR_ERROR, linewidth=1.5,
              markersize=4.5, label=f'Median {err_word}')
    if error_kind == 'abs':
        # Fixed 0–4 μm range with a tick label at every micron, matching (b).
        ax_a.set_xlim(*X_RANGE_UM)
        ax_a.set_xticks(X_TICKS_UM)
    else:
        # Zero line so the sign of the bias is visible.
        ax_a.axvline(0, color='k', linewidth=0.6, zorder=0)
    ax_a.set_xlabel(xlabel_a)
    ax_a.set_ylabel(level_label)
    ax_a.set_yticks(levels)
    ax_a.invert_yaxis()
    ax_a.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    # Legend above the axes: inside, it covers the band in the signed
    # version (the band spans most of the x range at levels 1 and 7).
    ax_a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                frameon=False, handlelength=1.6, columnspacing=1.0,
                handletextpad=0.5, borderaxespad=0.2)

    # ── (b) predicted aleatoric σ ─────────────────────────────────────────
    ax_b.plot(sigma_c, levels, 'd--', color=COLOR_SIGMA, linewidth=1.2,
              markersize=4, label=fr'{sigma_word} predicted $\sigma$')
    if sigma_floor_um is not None:
        ax_b.axvline(sigma_floor_um, color='0.35', linestyle=':', linewidth=1.0,
                     zorder=0, label=r'Smallest $\sigma$ allowed (log-$\sigma$ clamp)')
    ax_b.set_xlabel(LABEL_SIGMA)
    ax_b.set_ylabel(level_label)
    ax_b.set_yticks(levels)
    ax_b.invert_yaxis()
    # Fixed 0–4 μm range with a tick label at every micron, matching (a).
    ax_b.set_xlim(*X_RANGE_UM)
    ax_b.set_xticks(X_TICKS_UM)
    ax_b.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    # Same placement as panel (a): above the axes, clear of the data.
    ax_b.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1,
                frameon=False, handlelength=1.6, labelspacing=0.3,
                handletextpad=0.5, borderaxespad=0.2)

    # Panel letters outside the top-left corner of each axes, so they never
    # sit on data.
    for ax, letter in ((ax_a, '(a)'), (ax_b, '(b)')):
        ax.text(-0.22, 1.0, letter, transform=ax.transAxes, ha='left',
                va='bottom', fontweight='bold', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'levels':                  levels.tolist(),
        'error_kind':              error_kind,
        'error_p05_um':            p05.tolist(),
        'error_median_um':         p50.tolist(),
        'error_p95_um':            p95.tolist(),
        'sigma_stat':              sigma_stat,
        'predicted_sigma_um':      sigma_c.tolist(),
        'predicted_sigma_overall_mean_um': float(pred_std.mean()),
        'n_test_samples':          int(pred.shape[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description='Per-level r_e error + σ figure (2 x 1)')
    p.add_argument('--tag', default=None, help='training_runs/<tag> (step 13 selection)')
    p.add_argument('--run-dir', default=None, help='run directory (overrides --tag)')
    p.add_argument('--sigma-stat', choices=['mean', 'median'], default='mean',
                   help='test-set statistic of the predicted σ per level (default mean)')
    p.add_argument('--no-floor-line', action='store_true',
                   help='do not mark the log-σ clamp floor in panel (b)')
    p.add_argument('--out-dir', default=None, help='default: the run directory')
    args = p.parse_args()

    run_dir = selected_run_dir(args.tag, args.run_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = run_dir / 'pred_cache.npz'
    if not cache_path.exists():
        raise SystemExit(f'{cache_path} not found — run step 14 first')
    c = np.load(cache_path)
    pred, pred_std, true = c['pred'], c['pred_std'], c['true']  # (n, n_lev), μm
    print(f'Run      : {run_dir}')
    print(f'Test set : {pred.shape[0]} samples x {pred.shape[1]} levels')

    lsm = run_log_std_min(run_dir)
    floor_um = float(np.exp(lsm) * RE_RANGE_UM)
    # Share of predictions on the floor, per level (σ within 1e-4 relative).
    on_floor = np.isclose(pred_std, floor_um, rtol=1e-4, atol=0).mean(axis=0)  # (n_lev,)
    print(f'log-σ clamp min {lsm:g} → σ floor {floor_um:.4f} μm; share on floor per level: '
          + ' '.join(f'{f:.2f}' for f in on_floor))

    setup_style()
    all_stats = {'run_dir': str(run_dir), 'log_std_min': lsm,
                 'sigma_floor_um': floor_um,
                 'share_sigma_on_floor_per_level': on_floor.tolist()}
    for kind in ('abs', 'signed'):
        out = out_dir / f'per_level_error_2panel_{kind}.png'
        all_stats[kind] = plot_two_panel(
            pred, true, pred_std, out, error_kind=kind, sigma_stat=args.sigma_stat,
            sigma_floor_um=None if args.no_floor_line else floor_um)
        print(f'Figure   -> {out}')

    # Table for the manuscript text.
    s_abs, s_sgn = all_stats['abs'], all_stats['signed']
    print('\nPer level (μm): median|err|  5–95%|err|   median err  5–95% err   '
          f'{args.sigma_stat} σ   on floor')
    for i, L in enumerate(s_abs['levels']):
        print(f'  {L:>2d}  {s_abs["error_median_um"][i]:9.3f}  '
              f'{s_abs["error_p05_um"][i]:5.2f}–{s_abs["error_p95_um"][i]:5.2f}  '
              f'{s_sgn["error_median_um"][i]:+9.3f}  '
              f'{s_sgn["error_p05_um"][i]:+5.2f}–{s_sgn["error_p95_um"][i]:+5.2f}  '
              f'{s_abs["predicted_sigma_um"][i]:7.3f}  {on_floor[i]:7.2f}')
    print(f'  mean predicted σ over all samples and levels: '
          f'{s_abs["predicted_sigma_overall_mean_um"]:.3f} μm')

    stats_path = out_dir / 'per_level_error_2panel_stats.json'
    with stats_path.open('w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'Summary  -> {stats_path}')


if __name__ == '__main__':
    main()
