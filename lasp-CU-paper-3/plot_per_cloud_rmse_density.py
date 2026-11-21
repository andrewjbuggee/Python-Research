"""
plot_per_cloud_rmse_density.py — density-coloured versions of the per-profile
RMSE paper figures (per_cloud_rmse_paper_{4,6}panel.png).

The original figures (plot_run004_figures.plot_paper_4panel / _6panel) draw
10,000 test profiles in one colour per panel. With that many points the core
of each cloud saturates and the reader cannot see where most profiles fall.
Here every point is coloured by the local density of points around it, so the
core of the distribution stands out from the sparse tail — the convention of
"density-coloured scatter plots" common in remote-sensing comparisons.

Outputs (written next to the originals, which are left untouched):
    <run>/per_cloud_rmse_paper_4panel_density.png
    <run>/per_cloud_rmse_paper_6panel_density.png

How the density is computed (my own choices, not from a cited source):
  * A Gaussian kernel density estimate (scipy.stats.gaussian_kde, Scott's-rule
    bandwidth; Scott 1992, "Multivariate Density Estimation", Wiley) is fitted
    to the (x, RMSE) points of each panel and evaluated at every point.
  * The KDE is computed in the coordinates the panel is DRAWN in: log10(x) for
    the log-scaled τ_c axis, linear otherwise. Otherwise a point's colour would
    not match the visual crowding the reader sees.
  * gaussian_kde whitens the data with its covariance, so the very different
    units of the x and y axes do not matter.
  * Each panel's density is divided by its own maximum, so the colour scale
    reads 0–1 ("fraction of the panel's peak density") and one shared colour
    bar serves all panels. Absolute densities are NOT comparable between panels
    (the x variables differ), so nothing is lost by this.
  * Points are drawn sparsest-first, densest-last, so the core is not hidden
    under tail points.

The Pearson r and Spearman ρ in each title are computed exactly as in the
original figures (same helper functions), so the numbers are unchanged.

Colour map: black → purple → blue → cyan → green → yellow → orange → red,
modelled on the "pixel size" legend of the example figure the user supplied.
Black is the LOW-density end.

Usage
    python plot_per_cloud_rmse_density.py \\
        --run-dir training_runs/PT_sweep_2026-11-15/results/run_047 [--h5-path local.h5]

Needs <run>/pred_cache.npz (written by step 14 / plot_run_figures_profile_tau.py).

Author: Andrew J. Buggee (with Claude), 20 Nov 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'hyper_parameter_sweep'))

from plot_percentile_profiles_synthetic import _resolve_h5_path, _pearson_r
from plot_run004_figures import (
    setup_style_paper, _spearman_rho,
    FIG_W_PAPER, FIG_H_PAPER, FIG_W_PAPER_6, FIG_H_PAPER_6, DPI,
)

# ─── Colour map ──────────────────────────────────────────────────────────────
# Stops read off the example figure's "pixel size" legend (0.5 km black,
# 5 km purple, 10–15 km blues, 20 km cyan, 30–40 km greens, 50 km yellow-green,
# 60 km yellow-orange, 70 km red). Hex values are my approximations of those
# legend colours, not taken from the source paper.
_DENSITY_STOPS = [
    (0.00, '#000000'),   # black       (lowest density)
    (0.14, '#4B1A7A'),   # purple
    (0.28, '#3B4FB8'),   # blue
    (0.42, '#4A90D9'),   # light blue
    (0.54, '#6CCFDF'),   # cyan
    (0.66, '#5DBB63'),   # green
    (0.78, '#B5D334'),   # yellow-green
    (0.88, '#F2C230'),   # yellow-orange
    (1.00, '#E8412C'),   # red         (highest density)
]
RAINBOW_CMAP = LinearSegmentedColormap.from_list(
    'black_purple_blue_green_yellow_red', _DENSITY_STOPS, N=256)

# Default ("pale") colour map: matplotlib's 'magma', reversed so LOW density is
# a pale peach that recedes into the white background and HIGH density is a
# near-black purple with the strongest contrast against white, which draws the
# eye to the core. magma is perceptually uniform (lightness rises monotonically)
# and designed to stay readable for the common colour-vision deficiencies and in
# greyscale (van der Walt & Smith 2015, "mpl colormaps", SciPy talk;
# https://bids.github.io/colormap/). The truncation to 0.08–0.95 is my choice:
# it drops magma_r's almost-white first step, which would make the sparsest
# points invisible, and its pure-black last step, so the core keeps some hue.
_MAGMA_R = plt.get_cmap('magma_r')
PALE_CMAP = LinearSegmentedColormap.from_list(
    'magma_r_trunc', _MAGMA_R(np.linspace(0.08, 0.95, 256)), N=256)

DENSITY_CMAP = PALE_CMAP          # set by --cmap in main()

# RMSE axis limit (μm). Points above it are still used for the density and the
# correlations; they are only off-screen, and counted in a text box.
Y_MAX_UM = 3.0                    # set by --ymax in main(); <= 0 → no limit
Y_TICK_STEP_UM = 0.5

MARKER_SIZE = 4          # points², a little smaller than the original s=6
                         # because colour now carries the density information


def point_density(x: np.ndarray, y: np.ndarray, xlog: bool) -> np.ndarray:
    """Gaussian-KDE density at each (x, y) point, normalised to max = 1.

    Computed in the drawn coordinates (log10 x for log axes). Returns an array
    aligned with x and y. Assumes x, y are already finite (and x > 0 if xlog).
    """
    xx = np.log10(x) if xlog else x
    xy = np.vstack([xx, y])                      # shape (2, N)
    dens = gaussian_kde(xy)(xy)                  # shape (N,)  — O(N²), ~seconds for 10k
    return dens / dens.max()


def _density_scatter(ax, x, rmse, name, label, xlog=False, show_cut_note=False):
    """One panel: density-coloured scatter + the same r / ρ title as before."""
    valid = np.isfinite(x) & np.isfinite(rmse)
    if xlog:
        valid &= x > 0
    xv, yv = x[valid], rmse[valid]
    dens = point_density(xv, yv, xlog)
    order = np.argsort(dens)                     # draw densest points last (on top)
    sc = ax.scatter(xv[order], yv[order], c=dens[order], s=MARKER_SIZE,
                    cmap=DENSITY_CMAP, norm=Normalize(0.0, 1.0),
                    edgecolors='none', rasterized=True)
    if xlog:
        ax.set_xscale('log')
    # Correlations on ALL valid pairs, identical to the original figures.
    r   = _pearson_r(x, rmse)
    rho = _spearman_rho(x, rmse)
    ax.set_xlabel(name)
    ax.set_ylabel('Per-profile RMSE (μm)')
    ax.set_title(fr'({label})  Pearson $r$ = {r:+.3f}'  '\n'
                 fr'      Spearman $\rho$ = {rho:+.3f}', fontsize=8.5)
    ax.grid(alpha=0.3, which='both' if xlog else 'major')
    # Matplotlib's default draws grid lines ABOVE scatter collections, which
    # cut visible pale lines through the dense core; put them underneath.
    ax.set_axisbelow(True)

    # Optional RMSE-axis limit: the few large-error profiles stretch the axis
    # to ~7 μm and squash the core into the bottom of the panel. Clip the view
    # (not the data) and say how many points are off-screen.
    if Y_MAX_UM and Y_MAX_UM > 0:
        ax.set_ylim(0.0, Y_MAX_UM)
        ax.set_yticks(np.arange(0.0, Y_MAX_UM + 1e-9, Y_TICK_STEP_UM))
        n_cut = int(np.sum(yv > Y_MAX_UM))
        # The count is the same in every panel (same profiles, same RMSE), so
        # the note is drawn once, in panel (a) only.
        if n_cut > 0 and show_cut_note:
            # Put the note in whichever upper corner hides fewer points: draw
            # it in each corner, count the points inside its actual box, keep
            # the better one. Keeps it off the tail of a rising trend (d, f).
            # get_xlim() forces matplotlib to apply its pending autoscale;
            # without it transData still holds the default 0–1 x limits here.
            ax.get_xlim()
            pts_ax = ax.transAxes.inverted().transform(
                ax.transData.transform(np.column_stack([xv, yv])))   # (N, 2)
            renderer = ax.figure.canvas.get_renderer()
            best = None
            for x_txt, ha in ((0.98, 'right'), (0.02, 'left')):
                t = ax.text(x_txt, 0.95,
                            # two lines → a narrow box that fits in a corner
                            f'{n_cut} point{"s" if n_cut != 1 else ""} beyond\n'
                            f'{Y_MAX_UM:g} μm cut off',
                            transform=ax.transAxes, ha=ha, va='top', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                      edgecolor='0.6', linewidth=0.5))
                bb = t.get_window_extent(renderer).transformed(
                    ax.transAxes.inverted())
                n_hidden = int(np.sum((pts_ax[:, 0] >= bb.x0) & (pts_ax[:, 0] <= bb.x1)
                                      & (pts_ax[:, 1] >= bb.y0) & (pts_ax[:, 1] <= bb.y1)))
                if best is None or n_hidden < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (n_hidden, t)
                else:
                    t.remove()
    return sc


def _finish(fig, axes, sc, out_path: Path) -> None:
    """Shared colour bar on the right, then save."""
    plt.tight_layout(rect=(0, 0, 0.92, 1))       # leave room for the colour bar
    cax = fig.add_axes((0.935, 0.12, 0.015, 0.76))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('point density (fraction of panel peak)', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_paper_4panel_density(per_sample_rmse, predictors, out_path: Path) -> None:
    """Same 2×2 layout and labels as plot_run004_figures.plot_paper_4panel."""
    setup_style_paper()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_PAPER, FIG_H_PAPER))
    _density_scatter(axes[0, 0], predictors['tau_c'], per_sample_rmse,
                     r'$\tau_c$', 'a', xlog=True,
                     show_cut_note=True)
    _density_scatter(axes[0, 1], predictors['wv_above_mm'], per_sample_rmse,
                     'above-cloud column water vapor (mm)', 'b')
    _density_scatter(axes[1, 0], predictors['wood_adiab'], per_sample_rmse,
                     r'adiabaticity:  '
                     r'$\mathrm{LWP}_{\mathrm{obs}}/\mathrm{LWP}_{\mathrm{ad}}$', 'c')
    sc = _density_scatter(axes[1, 1], predictors['re_max'], per_sample_rmse,
                          r'max $r_e$ anywhere in profile ($\mu$m)', 'd')
    _finish(fig, axes, sc, out_path)


def plot_paper_6panel_density(per_sample_rmse, predictors, out_path: Path) -> None:
    """Same 3×2 layout and labels as plot_run004_figures.plot_paper_6panel."""
    setup_style_paper()
    fig, axes = plt.subplots(3, 2, figsize=(FIG_W_PAPER_6, FIG_H_PAPER_6))
    _density_scatter(axes[0, 0], predictors['tau_c'], per_sample_rmse,
                     r'$\tau_c$', 'a', xlog=True,
                     show_cut_note=True)
    _density_scatter(axes[0, 1], predictors['wv_above_mm'], per_sample_rmse,
                     'above-cloud column water vapor (mm)', 'b')
    _density_scatter(axes[1, 0], predictors['wood_adiab'], per_sample_rmse,
                     r'adiabaticity:  '
                     r'$\mathrm{LWP}_{\mathrm{obs}}/\mathrm{LWP}_{\mathrm{ad}}$', 'c')
    _density_scatter(axes[1, 1], predictors['re_std'], per_sample_rmse,
                     r'std of $r_e$ profile ($\mu$m)', 'd')
    _density_scatter(axes[2, 0], predictors['drizzle_proxy_top'], per_sample_rmse,
                     r'max $r_e$ in upper 30 % ($\mu$m)', 'e')
    sc = _density_scatter(axes[2, 1], predictors['drizzle_proxy'], per_sample_rmse,
                          r'max $r_e$ in lower 30 % ($\mu$m)', 'f')
    _finish(fig, axes, sc, out_path)


def main() -> int:
    p = argparse.ArgumentParser(description='Density-coloured per-profile RMSE figures')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--h5-path', default=None,
                   help='local HDF5 if the path in summary.json is not reachable')
    p.add_argument('--cmap', choices=['pale', 'rainbow'], default='pale',
                   help="'pale' = truncated magma_r (CVD-safe, default); "
                        "'rainbow' = black→purple→…→red of the first version")
    p.add_argument('--ymax', type=float, default=3.0,
                   help='upper RMSE-axis limit in μm (<= 0 for no limit)')
    args = p.parse_args()

    global DENSITY_CMAP, Y_MAX_UM
    DENSITY_CMAP = PALE_CMAP if args.cmap == 'pale' else RAINBOW_CMAP
    Y_MAX_UM = args.ymax

    run_dir = Path(args.run_dir).resolve()
    cache = np.load(run_dir / 'pred_cache.npz')
    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)
    h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)

    # Per-profile RMSE across the 7 levels — same definition as
    # plot_run_figures_profile_tau.main (μm).
    err_um = cache['pred'] - cache['true']                       # (N, 7)
    per_sample_rmse_um = np.sqrt(np.mean(err_um ** 2, axis=1))   # (N,)

    from plot_per_cloud_rmse_kfold import compute_predictors
    predictors = compute_predictors(cache['h5_idx'], h5_path)

    out4 = run_dir / 'per_cloud_rmse_paper_4panel_density.png'
    plot_paper_4panel_density(per_sample_rmse_um, predictors, out4)
    print(f'4-panel density -> {out4}')
    out6 = run_dir / 'per_cloud_rmse_paper_6panel_density.png'
    plot_paper_6panel_density(per_sample_rmse_um, predictors, out6)
    print(f'6-panel density -> {out6}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
