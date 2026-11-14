"""
08b_compare_lwc_insitu_vs_synthetic.py
======================================
Compare liquid water content (LWC) in the synthetic profiles with the in-situ
profiles they were generated from, to see whether the large synthetic LWC
values (up to ~9 g/m³) are inherited from the in-situ data or created by the
generator.

Data
----
* Synthetic: the `lwc` array of a bundle written by
  06_synthetic_profile_generator.py, shape (N, 7), g/m³, index 0 = cloud top,
  index -1 = cloud base.
* In-situ: the consolidated VOCALS-REx + ORACLES .mat file the bundle names
  (`input_path`), restricted to the rows the generator was fitted to
  (`train_source_index`): the τ window, the OUTLIERS list and de-duplication
  are therefore identical. Each in-situ profile keeps its measured vertical
  levels (16–157 per profile), ordered cloud top → base.

Figures (semi-transparent overlaid histograms, density-normalised so the two
sets are comparable despite different sample sizes)
------------------------------------------------------------------------------
  1. all levels      every LWC value in every profile
  2. cloud top       synthetic: top level (index 0)
                     in-situ:   upper 10 % of each profile's levels
  3. cloud base      synthetic: bottom level (index -1)
                     in-situ:   lower 10 % of each profile's levels
Each figure has two panels showing the same data: linear LWC with a log
density axis (makes the high-LWC tail visible) and log LWC (shows the bulk).
LWC values <= 0 cannot sit on a log axis; their count is printed and written
in the legend.

"Upper / lower 10 % of levels" means the first / last ceil(0.10 · n_levels)
measured levels of each profile (at least one), counted by level index, not
by altitude. Profiles with more levels contribute more values to the in-situ
histograms; set INSITU_EQUAL_PROFILE_WEIGHT = True to weight every profile
equally instead.

Also prints percentile tables and lists the in-situ profiles whose maximum
LWC exceeds HIGH_LWC_G_M3, so the source of the tail can be traced.

Usage
    python 08b_compare_lwc_insitu_vs_synthetic.py [path/to/bundle.npz]

Author: Claude (assistant) for Andrew J. Buggee, 13 Nov 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')            # write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import scipy.io                  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
NPZ_PATH = (SCRIPT_DIR / 'synthetic_profiles' /
            'synthetic_profiles_jointMVN_N100000_L7_2026-11-12.npz')

EDGE_FRACTION = 0.10                 # in-situ: top / bottom 10 % of levels
HIGH_LWC_G_M3 = 2.0                  # report in-situ profiles with max LWC above this
INSITU_EQUAL_PROFILE_WEIGHT = False  # True → each in-situ profile has equal total weight
N_BINS = 60
FIG_DPI = 300

# Okabe & Ito colour-blind-safe pair, same as 06b: vermillion = in-situ, blue = synthetic.
C_INSITU, C_SYN = '#D55E00', '#0072B2'
ALPHA = 0.5


# ── Data loading ───────────────────────────────────────────────────────────────
def load_synthetic(npz_path: Path) -> tuple[np.ndarray, dict]:
    """Synthetic LWC (N, L) in g/m³, top → base, plus the opened bundle."""
    d = np.load(npz_path)
    lwc = np.asarray(d['lwc'], dtype=np.float64)
    return lwc, d


def load_insitu(bundle) -> dict:
    """Per-profile in-situ LWC arrays (g/m³, top → base) for the generator's rows.

    Returns dict with:
      lwc      list of 1-D arrays, one per profile
      label    list of 'campaign date profile_idx' strings (as in OUTLIERS)
      z        list of 1-D altitude arrays (km), top → base
    """
    mat_path = Path(str(bundle['input_path']))
    rows = np.asarray(bundle['train_source_index'], dtype=int)
    m = scipy.io.loadmat(mat_path, squeeze_me=True)
    out = {'lwc': [], 'label': [], 'z': []}
    for i in rows:
        out['lwc'].append(np.atleast_1d(np.asarray(m['lwc'][i], dtype=np.float64)))
        out['z'].append(np.atleast_1d(np.asarray(m['z'][i], dtype=np.float64)))
        out['label'].append(f"{m['campaign_each'][i]} {m['date_of_flight'][i]} "
                            f"{int(m['profile_idx'][i])}")
    print(f'In-situ: {rows.size} profiles from {mat_path.name}')
    return out


# ── Selecting levels ───────────────────────────────────────────────────────────
def n_edge(n_levels: int) -> int:
    """Number of levels in the top / bottom EDGE_FRACTION of a profile (>= 1)."""
    return max(1, int(np.ceil(EDGE_FRACTION * n_levels)))


def insitu_values(profiles: list[np.ndarray], which: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate in-situ LWC values and matching histogram weights.

    which = 'all' | 'top' | 'base'. Profiles are ordered top → base, so 'top'
    takes the first n_edge levels and 'base' the last n_edge levels.
    Weights are 1 per value, or 1/len(selection) per value when
    INSITU_EQUAL_PROFILE_WEIGHT is True (each profile then sums to 1).
    """
    vals, wts = [], []
    for p in profiles:
        if which == 'all':
            sel = p
        elif which == 'top':
            sel = p[:n_edge(p.size)]
        elif which == 'base':
            sel = p[-n_edge(p.size):]
        else:
            raise ValueError(which)
        vals.append(sel)
        w = 1.0 / sel.size if INSITU_EQUAL_PROFILE_WEIGHT else 1.0
        wts.append(np.full(sel.size, w))
    return np.concatenate(vals), np.concatenate(wts)


def synthetic_values(lwc: np.ndarray, which: str) -> np.ndarray:
    """Synthetic LWC values: all levels, the top level (0) or the base level (-1)."""
    if which == 'all':
        return lwc.ravel()
    if which == 'top':
        return lwc[:, 0]
    if which == 'base':
        return lwc[:, -1]
    raise ValueError(which)


# ── Statistics ─────────────────────────────────────────────────────────────────
def summary_row(name: str, x: np.ndarray, w: np.ndarray | None = None) -> str:
    """One line of percentiles and tail fractions (weighted if w is given)."""
    x = np.asarray(x)
    w = np.ones_like(x) if w is None else np.asarray(w)
    order = np.argsort(x)
    xs, cw = x[order], np.cumsum(w[order]) / w.sum()
    pct = {p: xs[np.searchsorted(cw, p / 100.0)] for p in (1, 50, 99)}
    frac = lambda t: 100.0 * w[x > t].sum() / w.sum()
    return (f'  {name:<26} n={x.size:>7}  p1={pct[1]:6.3f}  p50={pct[50]:6.3f}  '
            f'p99={pct[99]:6.3f}  max={x.max():6.3f}  '
            f'>1:{frac(1.0):5.2f}%  >2:{frac(2.0):5.2f}%  <=0: {int((x <= 0).sum())}')


# ── Plotting ───────────────────────────────────────────────────────────────────
def overlay_figure(ins_x, ins_w, syn_x, title: str, ins_desc: str, syn_desc: str,
                   out_path: Path) -> None:
    """Two-panel overlaid histogram: linear LWC (log density) and log LWC."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # (a) linear LWC, log density — the high-LWC tail
    hi = max(ins_x.max(), syn_x.max())
    bins_lin = np.linspace(0.0, hi, N_BINS + 1)
    ax = axes[0]
    ax.hist(ins_x, bins=bins_lin, weights=ins_w, density=True, alpha=ALPHA,
            color=C_INSITU, label=f'In-situ ({ins_desc})')
    ax.hist(syn_x, bins=bins_lin, density=True, alpha=ALPHA,
            color=C_SYN, label=f'Synthetic ({syn_desc})')
    ax.set_yscale('log')
    ax.set_xlabel('LWC (g m$^{-3}$)')
    ax.set_ylabel('Density (log)')
    ax.set_title('(a) linear LWC: tail')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    # (b) log LWC — the bulk. Histogram log10(LWC) with density=True, so the
    # density is per unit log10(LWC) and equal areas mean equal fractions of
    # the data. (Histogramming LWC on log-spaced bins with density=True instead
    # divides by the LINEAR bin width, which inflates the narrow low-LWC bins
    # and turns a handful of tiny values into tall spikes.) Values <= 0 have no
    # logarithm; they are counted in the legend.
    pos_i, pos_s = ins_x > 0, syn_x > 0
    li, ls = np.log10(ins_x[pos_i]), np.log10(syn_x[pos_s])
    bins_log = np.linspace(min(li.min(), ls.min()), max(li.max(), ls.max()), N_BINS + 1)
    ax = axes[1]
    ax.hist(li, bins=bins_log, weights=ins_w[pos_i], density=True,
            alpha=ALPHA, color=C_INSITU,
            label=f'In-situ ({(~pos_i).sum()} values <= 0 not shown)')
    ax.hist(ls, bins=bins_log, density=True, alpha=ALPHA,
            color=C_SYN, label=f'Synthetic ({(~pos_s).sum()} values <= 0 not shown)')
    ticks = np.arange(np.floor(bins_log[0]), np.ceil(bins_log[-1]) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'$10^{{{int(t)}}}$' for t in ticks])
    ax.set_xlabel('LWC (g m$^{-3}$, log scale)')
    ax.set_ylabel('Density per unit log$_{10}$(LWC)')
    ax.set_title('(b) log LWC: bulk')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'  figure → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NPZ_PATH
    tag = npz_path.stem.split('jointMVN_', 1)[-1]          # e.g. N100000_L7_2026-11-12
    out_dir = npz_path.parent

    syn_lwc, bundle = load_synthetic(npz_path)
    ins = load_insitu(bundle)
    n_lev = np.array([p.size for p in ins['lwc']])
    print(f'Synthetic: {syn_lwc.shape[0]} profiles × {syn_lwc.shape[1]} levels')
    print(f'In-situ levels per profile: {n_lev.min()}–{n_lev.max()} '
          f'(median {int(np.median(n_lev))}); top/base 10 % = '
          f'{n_edge(n_lev.min())}–{n_edge(n_lev.max())} levels')
    weight_note = ('each in-situ profile weighted equally' if INSITU_EQUAL_PROFILE_WEIGHT
                   else 'in-situ: every measured level counts once')

    cases = [
        ('all', 'LWC at all levels', 'all measured levels', 'all 7 levels'),
        ('top', 'LWC at cloud top', f'upper {EDGE_FRACTION:.0%} of levels', 'top level'),
        ('base', 'LWC at cloud base', f'lower {EDGE_FRACTION:.0%} of levels', 'base level'),
    ]
    print(f'\nPercentiles (g/m³); {weight_note}')
    for which, title, ins_desc, syn_desc in cases:
        ins_x, ins_w = insitu_values(ins['lwc'], which)
        syn_x = synthetic_values(syn_lwc, which)
        print(f' {title}:')
        print(summary_row('in-situ', ins_x, ins_w))
        print(summary_row('synthetic', syn_x))
        if which in ('top', 'base'):
            # Like-for-like with the synthetic single edge level: the in-situ
            # topmost (or lowest) measured level only.
            k = 0 if which == 'top' else -1
            edge = np.array([p[k] for p in ins['lwc']])
            print(summary_row(f'in-situ {which}most level only', edge))
        overlay_figure(ins_x, ins_w, syn_x,
                       f'{title}: in-situ (N = {len(ins["lwc"])} profiles) vs '
                       f'synthetic (N = {syn_lwc.shape[0]})',
                       ins_desc, syn_desc,
                       out_dir / f'lwc_insitu_vs_synthetic_{which}_{tag}.png')

    # Which in-situ profiles carry the high-LWC tail, and where in the cloud?
    print(f'\nIn-situ profiles with max LWC > {HIGH_LWC_G_M3} g/m³ '
          f'(campaign, date, profile_idx — the OUTLIERS key):')
    hits = 0
    for lab, p, z in zip(ins['label'], ins['lwc'], ins['z']):
        if p.max() > HIGH_LWC_G_M3:
            k = int(np.argmax(p))
            u = (z[k] - z[-1]) / (z[0] - z[-1]) if z[0] != z[-1] else np.nan
            nxt = p[k + 1] if k + 1 < p.size else np.nan
            prv = p[k - 1] if k > 0 else np.nan
            print(f'  {lab:<28} max={p.max():5.2f} at level {k + 1}/{p.size} '
                  f'(u = {u:.2f}, 1 = top); neighbours {prv:.2f} / {nxt:.2f}; '
                  f'profile median {np.median(p):.2f}')
            hits += 1
    if hits == 0:
        print('  none')

    syn_max = syn_lwc.max(axis=1)
    lev_of_max = np.argmax(syn_lwc, axis=1)
    big = syn_max > HIGH_LWC_G_M3
    print(f'\nSynthetic clouds with max LWC > {HIGH_LWC_G_M3} g/m³: {big.sum()} '
          f'({100 * big.mean():.2f}%); level of the maximum (1 = top): '
          + ', '.join(f'{k + 1}: {(lev_of_max[big] == k).sum()}'
                      for k in range(syn_lwc.shape[1])))


if __name__ == '__main__':
    main()
