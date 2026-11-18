"""
17_per_level_error_figure.py — STEP 17 of the paper-3 pipeline
==============================================================
Manuscript figure: per-level r_e retrieval error and predicted uncertainty
for one trained model, as a 2 x 1 figure.

  (a) Retrieval error at each vertical level over the held-out test set:
      the median (line + markers) and the 5th–95th percentile range (shading).
  (b) Two quantities, each on its own y-axis:
        left axis  — mean predicted aleatoric σ at each level, i.e. the
                     width of the network's predictive distribution
                     p(r_e(l) | R), averaged over the test set;
        right axis — standard deviation of the TRAINING-set true r_e at each
                     level, i.e. the spread of the data the network learned
                     from.

Two versions are written, identical except for panel (a):
  per_level_error_2panel_abs.png     — |retrieved − true|
  per_level_error_2panel_signed.png  —  retrieved − true   (the same sign
                                        convention as steps 15 and 16)

Inputs
  * <run>/pred_cache.npz from step 14 (pred, pred_std, true, h5_idx; μm).
  * The training HDF5 (only for the training-set σ of r_e). The training
    split is rebuilt with data.create_profile_aware_splits using the run's
    own seed and val/test sizes (same call as plot_run_figures_profile_tau's
    _run_inference). The script checks that the rebuilt TEST indices equal
    the cached h5_idx, so the training split is known to be the one the
    model was trained on. The per-level training σ is then cached in
    <run>/train_re_std_cache.npz so later replots don't need the HDF5.

Orientation of panel (b)
  --panel-b vertical (default): vertical level on the y-axis like panel (a);
      predicted σ on the bottom x-axis, training-set σ on the top x-axis
      (matplotlib twiny).
  --panel-b horizontal: vertical level on the x-axis, σ on the left y-axis and
      training-set σ on the right y-axis (twinx).

Usage
    python 17_per_level_error_figure.py --tag PT_2026-11-14 [--h5-path local.h5]
    python 17_per_level_error_figure.py --run-dir <run dir> [--sigma-stat median]

Nothing here is stochastic (the split is a deterministic function of the
run's seed, which is printed and saved in the stats JSON).
Author: Andrew J. Buggee, 17 Nov 2026
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
from plot_percentile_profiles_synthetic import _resolve_h5_path  # noqa: E402

# ─── Figure configuration ─────────────────────────────────────────────────
FIG_WIDTH_IN  = 3.25     # IEEE single column
FIG_HEIGHT_IN = 6.4      # two stacked panels
DPI           = 500

# Okabe-Ito / Wong (2011, Nat. Methods 8, 441) colour-blind-safe colours.
COLOR_ERROR  = '#D55E00'  # vermillion — retrieval error (panel a); same as before
COLOR_SIGMA  = '#0072B2'  # blue — predicted aleatoric σ (panel b, left axis)
COLOR_TRAIN  = '#009E73'  # bluish green — training-set σ of r_e (panel b, right axis)
BAND_ALPHA   = 0.25

PERCENTILES = (5, 50, 95)

# Axis labels requested for panel (b). Units (μm) added; both are r_e
# standard deviations.
LABEL_SIGMA = r'Aleatoric $\sigma$ (width of $p(r_e(l)\,|\,\mathbf{R})$) ($\mu$m)'
LABEL_TRAIN = r'$\sigma$ of training-data $r_e(l)$ ($\mu$m)'


# ─── Run directory / metadata ─────────────────────────────────────────────
def selected_run_dir(tag: str | None, run_dir: str | None) -> Path:
    """--run-dir if given, else the run step 13 selected for --tag."""
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    sel = json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))
    return (REPO / sel['run_dir']).resolve()


def run_split_settings(run_dir: Path) -> dict:
    """HDF5 path, split seed and val/test sizes the run was trained with.

    Profile + τ runs write summary.json (h5 path under 'data'); the older
    profile-only runs write config.json (h5 path at top level). The split
    seed is 'seed' in either file; 42 was the trainers' default.
    """
    for name in ('summary.json', 'config.json'):
        p = run_dir / name
        if p.exists():
            s = json.load(open(p))
            h5 = s.get('data', {}).get('h5_path') or s.get('h5_path')
            return {
                'h5_path':         h5,
                'seed':            int(s.get('seed', 42)),
                'n_val_profiles':  int(s['n_val_profiles']),
                'n_test_profiles': int(s['n_test_profiles']),
                'source':          name,
            }
    raise FileNotFoundError(f'no summary.json or config.json in {run_dir}')


# ─── Training-set σ of r_e per level ──────────────────────────────────────
def training_re_std(run_dir: Path, h5_override: str | None,
                    test_idx_cached: np.ndarray, refresh: bool) -> dict:
    """Per-level standard deviation of the true r_e over the training split.

    Returns {'std_um': (n_lev,), 'mean_um': (n_lev,), 'n_train': int, ...}.
    Cached in <run>/train_re_std_cache.npz.
    """
    cache = run_dir / 'train_re_std_cache.npz'
    if cache.exists() and not refresh:
        c = np.load(cache)
        print(f'Loaded training-set r_e σ from {cache}')
        return {k: c[k] for k in c.files}

    import h5py
    from data import create_profile_aware_splits  # imports torch

    st = run_split_settings(run_dir)
    h5_path = _resolve_h5_path(st['h5_path'], h5_override)
    print(f'Rebuilding the training split from {h5_path}')
    print(f'  seed = {st["seed"]}, n_val = {st["n_val_profiles"]}, '
          f'n_test = {st["n_test_profiles"]} (from {st["source"]})')

    train_idx, _, test_idx = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=st['n_val_profiles'],
        n_test_profiles=st['n_test_profiles'],
        seed=st['seed'],
    )
    # Hard check: the rebuilt test split must be the one the predictions
    # were made on. If not, the training split is not the model's either.
    if not np.array_equal(np.sort(test_idx), np.sort(test_idx_cached)):
        raise RuntimeError('Rebuilt test indices differ from pred_cache.npz '
                           'h5_idx; wrong HDF5 or split settings.')

    with h5py.File(h5_path, 'r') as f:
        # h5py fancy indexing needs increasing indices; train_idx from
        # np.where is already sorted.
        re_train_um = f['profiles'][train_idx].astype(np.float64)  # (n_train, n_lev), μm

    # Population std (ddof=0) over training SAMPLES. When a profile is
    # simulated at several geometries it is counted once per sample, i.e.
    # weighted the way the network saw it in training.
    out = {
        'std_um':  re_train_um.std(axis=0),           # (n_lev,)
        'mean_um': re_train_um.mean(axis=0),          # (n_lev,)
        'n_train': np.int64(len(train_idx)),
        'seed':    np.int64(st['seed']),
        'h5_path': np.str_(str(h5_path)),
    }
    np.savez(cache, **out)
    print(f'Cached training-set r_e σ -> {cache}')
    return out


# ─── Figure ───────────────────────────────────────────────────────────────
def plot_two_panel(pred: np.ndarray, true: np.ndarray, pred_std: np.ndarray,
                   train_std_um: np.ndarray, out_path: Path,
                   error_kind: str = 'abs', sigma_stat: str = 'mean',
                   panel_b: str = 'vertical') -> dict:
    """2 x 1 figure. pred, true, pred_std: (n_samples, n_levels) in μm.

    error_kind 'abs'    → panel (a) shows |pred − true|
               'signed' → panel (a) shows  pred − true
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

    # Two lines: on one line the label is taller than a 3.25"-wide panel.
    level_label = f'Vertical level\n(1 = cloud top, {n_levels} = cloud base)'
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # ── (a) error: same layout as the original figure, no σ curve ─────────
    ax_a.fill_betweenx(levels, p05, p95, color=COLOR_ERROR, alpha=BAND_ALPHA,
                       linewidth=0, label=f'5–95% of {err_word}')
    ax_a.plot(p50, levels, 'o-', color=COLOR_ERROR, linewidth=1.5,
              markersize=4.5, label=f'Median {err_word}')
    if error_kind == 'abs':
        ax_a.set_xlim(left=0)
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

    # ── (b) predicted σ and training-set σ on separate axes ───────────────
    if panel_b == 'horizontal':
        # Level on x; σ on left y, training σ on right y (twinx).
        ax_t = ax_b.twinx()
        h1, = ax_b.plot(levels, sigma_c, 'd--', color=COLOR_SIGMA,
                        linewidth=1.2, markersize=4,
                        label=fr'{sigma_word} predicted $\sigma$')
        h2, = ax_t.plot(levels, train_std_um, 's-', color=COLOR_TRAIN,
                        linewidth=1.2, markersize=3.5,
                        label=r'Training-set $\sigma$')
        ax_b.set_xlabel(level_label)
        ax_b.set_xticks(levels)
        ax_b.set_ylabel(LABEL_SIGMA, color=COLOR_SIGMA)
        ax_t.set_ylabel(LABEL_TRAIN, color=COLOR_TRAIN)
        ax_b.tick_params(axis='y', colors=COLOR_SIGMA)
        ax_t.tick_params(axis='y', colors=COLOR_TRAIN)
        ax_b.set_ylim(bottom=0)
        ax_t.set_ylim(bottom=0)
    elif panel_b == 'vertical':
        # Level on y like panel (a); σ on bottom x, training σ on top x (twiny).
        ax_t = ax_b.twiny()
        h1, = ax_b.plot(sigma_c, levels, 'd--', color=COLOR_SIGMA,
                        linewidth=1.2, markersize=4,
                        label=fr'{sigma_word} predicted $\sigma$')
        h2, = ax_t.plot(train_std_um, levels, 's-', color=COLOR_TRAIN,
                        linewidth=1.2, markersize=3.5,
                        label=r'Training-set $\sigma$')
        ax_b.set_ylabel(level_label)
        ax_b.set_yticks(levels)
        ax_b.invert_yaxis()
        ax_b.set_xlabel(LABEL_SIGMA, color=COLOR_SIGMA)
        ax_t.set_xlabel(LABEL_TRAIN, color=COLOR_TRAIN)
        ax_b.tick_params(axis='x', colors=COLOR_SIGMA)
        ax_t.tick_params(axis='x', colors=COLOR_TRAIN)
        ax_b.set_xlim(left=0)
        ax_t.set_xlim(left=0)
    else:
        raise ValueError(f'panel_b must be horizontal or vertical, got {panel_b!r}')
    ax_b.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    # Fixed location: loc='best' only sees ax_b's line, not the twin axis's,
    # and put the legend on the training-σ curve. Both σ curves are smallest
    # in mid-cloud, so the empty region is low (horizontal) / lower left
    # (vertical). Labels are kept short so the legend fits there.
    legend_loc = 'lower center' if panel_b == 'horizontal' else 'lower left'
    ax_b.legend(handles=[h1, h2], loc=legend_loc, frameon=True, framealpha=0.9,
                handlelength=1.6, labelspacing=0.35, handletextpad=0.5,
                borderaxespad=0.3)

    # Panel letters outside the top-left corner of each axes, so they never
    # sit on data (inside, '(b)' covered the level-1 marker).
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
        'train_re_std_um':         np.asarray(train_std_um).tolist(),
        'n_test_samples':          int(pred.shape[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description='Per-level r_e error + σ figure (2 x 1)')
    p.add_argument('--tag', default=None, help='training_runs/<tag> (step 13 selection)')
    p.add_argument('--run-dir', default=None, help='run directory (overrides --tag)')
    p.add_argument('--h5-path', default=None,
                   help='local HDF5 if the path stored with the run is not reachable')
    p.add_argument('--sigma-stat', choices=['mean', 'median'], default='mean',
                   help='test-set statistic of the predicted σ per level (default mean)')
    p.add_argument('--panel-b', choices=['vertical', 'horizontal'], default='vertical',
                   help='orientation of panel (b); see module docstring')
    p.add_argument('--refresh', action='store_true',
                   help='recompute the training-set σ instead of using its cache')
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
    if 'h5_idx' not in c.files:
        raise SystemExit('pred_cache.npz has no h5_idx; rerun step 14 with --refresh')
    print(f'Run      : {run_dir}')
    print(f'Test set : {pred.shape[0]} samples x {pred.shape[1]} levels')

    tr = training_re_std(run_dir, args.h5_path, c['h5_idx'], args.refresh)
    print(f'Training : {int(tr["n_train"])} samples')

    setup_style()
    all_stats = {'run_dir': str(run_dir), 'split_seed': int(tr['seed']),
                 'n_train_samples': int(tr['n_train']),
                 'train_re_mean_um': np.asarray(tr['mean_um']).tolist()}
    for kind in ('abs', 'signed'):
        out = out_dir / f'per_level_error_2panel_{kind}.png'
        all_stats[kind] = plot_two_panel(pred, true, pred_std, tr['std_um'], out,
                                         error_kind=kind, sigma_stat=args.sigma_stat,
                                         panel_b=args.panel_b)
        print(f'Figure   -> {out}')

    # Table for the manuscript text.
    s_abs, s_sgn = all_stats['abs'], all_stats['signed']
    print('\nPer level (μm): median|err|  5–95%|err|   median err  5–95% err   '
          f'{args.sigma_stat} σ   train σ')
    for i, L in enumerate(s_abs['levels']):
        print(f'  {L:>2d}  {s_abs["error_median_um"][i]:9.3f}  '
              f'{s_abs["error_p05_um"][i]:5.2f}–{s_abs["error_p95_um"][i]:5.2f}  '
              f'{s_sgn["error_median_um"][i]:+9.3f}  '
              f'{s_sgn["error_p05_um"][i]:+5.2f}–{s_sgn["error_p95_um"][i]:+5.2f}  '
              f'{s_abs["predicted_sigma_um"][i]:7.3f}  {s_abs["train_re_std_um"][i]:7.3f}')
    print(f'  mean predicted σ over all samples and levels: '
          f'{s_abs["predicted_sigma_overall_mean_um"]:.3f} μm')

    stats_path = out_dir / 'per_level_error_2panel_stats.json'
    with stats_path.open('w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'Summary  -> {stats_path}')


if __name__ == '__main__':
    main()
