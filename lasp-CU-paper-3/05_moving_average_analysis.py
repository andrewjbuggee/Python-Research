"""
Moving-average smoothing analysis of raw in-situ effective radius profiles.

For every UNIQUE droplet profile in MAT_DIR, this script:
  1. Loads the raw in-situ r_e profile (cloud top → cloud base).
  2. Computes a centered moving average for window sizes 5, 10, 15 points.
     Near the profile edges the window shrinks so output length matches the
     input length and no padding is invented.
  3. Reports the root-mean-square error between the raw profile and each
     smoothed version (per-profile values + summary statistics).
  4. Renders a 2 x 3 grid of example profiles spanning a variety of raw
     profile lengths, with raw + three smoothed versions overlaid.

Profile convention (matches notebook 04): index 0 = cloud top, index -1 = cloud base.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
MAT_DIR        = Path('/Volumes/My Passport/neural_network_training_data/saz0_allProfiles/')

# Each window spec is either:
#   • int   — a fixed window length in points (e.g. 5  → average over 5 points)
#   • str   — a percentage of the profile length (e.g. '10%' → window = round(0.10 * len))
#             The percentage is evaluated per profile, so a 10% spec gives
#             window=10 for a 100-point profile and window=2 for a 20-point profile.
#   • float — a fraction in (0, 1] (e.g. 0.10 is equivalent to '10%')
# WINDOW_SPECS   = (5, 10, 15, '10%')
# WINDOW_SPECS   = (10, '25%')
# WINDOW_SPECS   = ('25%', '35%', '45%')
WINDOW_SPECS   = ('25%', '45%', '65%')
MIN_WINDOW     = 1                       # floor for percentage-based windows
N_PANELS       = 6                       # 2 rows x 3 cols
FIGURE_DIR     = Path(__file__).parent / 'Figures'
FIGURE_NAME    = 'moving_average_smoothing.png'
FIGURE_NAME_REDUCED = 'moving_average_smoothing_7_levels.png'
FIGURE_DPI     = 500
TAU_C_MIN      = 1.0                     # skip files with no real cloud, mirrors converter
N_REDUCED_LEVELS = 7                     # block-average target length for the second analysis


# ── Paper-quality matplotlib style (mirrors regenerate_kfold_figures.setup_style)
def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    12,
        'axes.titlesize':    13,
        'figure.titlesize':  14,
        'legend.fontsize':   7,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'axes.linewidth':    0.8,
    })


setup_style()


# ── Smoothing ──────────────────────────────────────────────────────────────────
def resolve_window(spec, n):
    """Translate a window spec into an integer window length for a profile of length n.

    spec : int (fixed) | float in (0,1] (fraction) | str ending in '%' (percentage)
    """
    if isinstance(spec, str):
        s = spec.strip()
        if not s.endswith('%'):
            raise ValueError(f"String window spec must end with '%': got {spec!r}")
        frac = float(s[:-1]) / 100.0
        w = int(round(frac * n))
    elif isinstance(spec, float):
        if not 0.0 < spec <= 1.0:
            raise ValueError(f'Float window spec must be in (0, 1]: got {spec}')
        w = int(round(spec * n))
    elif isinstance(spec, (int, np.integer)):
        w = int(spec)
    else:
        raise TypeError(f'Unsupported window spec type: {type(spec).__name__}')
    return max(MIN_WINDOW, min(w, n))


def spec_label(spec):
    """Short legend-friendly label for a window spec."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, float):
        return f'{spec * 100:g}%'
    return f'{int(spec)} pts'


def centered_moving_average(x, window):
    """Centered moving average; near edges the window shrinks rather than padding.

    Output length equals input length. For short profiles this avoids
    fabricating boundary values that would inflate the error metric.
    """
    n = len(x)
    half = window // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = x[lo:hi].mean()
    return out


def equal_altitude_bin_average(re, z, m=N_REDUCED_LEVELS):
    """Reduce a (re, z) profile to m points sampled at evenly spaced altitudes.

    Lays down m+1 bin edges at altitudes evenly spaced from z[0] (cloud top)
    to z[-1] (cloud base) and takes the mean of raw r_e within each bin. The
    output altitude grid is the bin centers, which are evenly spaced by
    construction and inherit the cloud-top → cloud-base orientation of the
    input. Direction is handled by working in a normalized coordinate
    u = (z - z_top) / (z_base - z_top) so both increasing and decreasing z
    are treated identically.

    Empty bins (only possible with very sparse altitude sampling) are filled
    by linear interpolation across the populated bin centers.
    """
    re = np.asarray(re, dtype=np.float64)
    z  = np.asarray(z,  dtype=np.float64)

    z_top, z_base = float(z[0]), float(z[-1])
    edges   = np.linspace(z_top, z_base, m + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if z_top == z_base:
        return np.full(m, re.mean()), centers

    u = (z - z_top) / (z_base - z_top)
    bin_idx = np.clip(np.floor(u * m).astype(int), 0, m - 1)

    re_out = np.full(m, np.nan)
    for k in range(m):
        mask = bin_idx == k
        if mask.any():
            re_out[k] = re[mask].mean()

    valid = ~np.isnan(re_out)
    if not valid.all():
        re_out[~valid] = np.interp(centers[~valid], centers[valid], re_out[valid])

    return re_out, centers


# ── Load unique profiles ───────────────────────────────────────────────────────
mat_files = sorted(f for f in MAT_DIR.glob('*.mat') if not f.name.startswith('._'))
print(f'Found {len(mat_files)} .mat files in {MAT_DIR}')

profiles_raw  = []
altitudes_raw = []
seen_fingerprints = set()
n_skipped_tau = 0
n_skipped_dup = 0

for path in mat_files:
    d = scipy.io.loadmat(path, squeeze_me=True)
    if not {'re', 'z', 'tau'}.issubset(d.keys()):
        continue

    re = np.asarray(d['re'][()], dtype=np.float64)
    z  = np.asarray(d['z'][()],  dtype=np.float64)
    tau_c = float(np.asarray(d['tau'][()]).max())

    if tau_c < TAU_C_MIN:
        n_skipped_tau += 1
        continue

    fp = tuple(np.round(re[:5], 4))
    if fp in seen_fingerprints:
        n_skipped_dup += 1
        continue
    seen_fingerprints.add(fp)

    profiles_raw.append(re)
    altitudes_raw.append(z)

n_profiles = len(profiles_raw)
n_levels_each = np.array([len(r) for r in profiles_raw])

print(f'Skipped {n_skipped_tau} file(s) for tau_c < {TAU_C_MIN}')
print(f'Skipped {n_skipped_dup} duplicate profile(s)')
print(f'Unique profiles loaded: {n_profiles}')
print(f'Profile lengths: min={n_levels_each.min()}, max={n_levels_each.max()}, '
      f'mean={n_levels_each.mean():.1f}')


# ── Compute smoothed versions and per-profile RMSE ─────────────────────────────
# Per-spec storage. Window length resolves per-profile so percentage specs
# produce a different integer window for every profile.
smoothed       = {spec: [] for spec in WINDOW_SPECS}
rmse           = {spec: np.zeros(n_profiles) for spec in WINDOW_SPECS}
windows_used   = {spec: np.zeros(n_profiles, dtype=int) for spec in WINDOW_SPECS}

for i, re_raw in enumerate(profiles_raw):
    for spec in WINDOW_SPECS:
        w = resolve_window(spec, len(re_raw))
        re_smooth = centered_moving_average(re_raw, w)
        smoothed[spec].append(re_smooth)
        rmse[spec][i] = np.sqrt(np.mean((re_smooth - re_raw) ** 2))
        windows_used[spec][i] = w

print()
print(f'Moving-average RMSE vs. raw profile (μm):')
print(f'  {"Spec":>10} {"window pts (min/median/max)":>30} '
      f'{"RMSE mean":>10} {"median":>10} {"max":>10} {"min":>10}')
print(f'  {"-"*84}')
for spec in WINDOW_SPECS:
    r  = rmse[spec]
    ws = windows_used[spec]
    win_summary = f'{ws.min()}/{int(np.median(ws))}/{ws.max()}'
    print(f'  {spec_label(spec):>10} {win_summary:>30} '
          f'{r.mean():>10.4f} {np.median(r):>10.4f} '
          f'{r.max():>10.4f} {r.min():>10.4f}')


# ── Pick N_PANELS profiles that span the range of raw profile lengths ──────────
sort_idx = np.argsort(n_levels_each)
panel_idx = sort_idx[np.linspace(0, n_profiles - 1, N_PANELS, dtype=int)]


# ── 2 x 3 figure ───────────────────────────────────────────────────────────────
nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))
axes = axes.flatten()

palette = ['steelblue', 'darkorange', 'seagreen', 'firebrick',
           'mediumpurple', 'goldenrod']
spec_color = {spec: palette[k % len(palette)]
              for k, spec in enumerate(WINDOW_SPECS)}

for ax, idx in zip(axes, panel_idx):
    re_raw = profiles_raw[idx]
    z_raw  = altitudes_raw[idx]

    ax.plot(re_raw, z_raw, color='0.4', linewidth=1.2,
            marker='o', markersize=2.8, label='Raw in-situ')

    for spec in WINDOW_SPECS:
        w_used = windows_used[spec][idx]
        ax.plot(smoothed[spec][idx], z_raw, color=spec_color[spec],
                linewidth=1.6, alpha=0.9,
                label=f'{spec_label(spec)} ({w_used} pts, '
                      f'RMSE={rmse[spec][idx]:.3f} $\\mu$m)')

    ax.set_title(f'Profile {idx + 1}  |  {len(re_raw)} measurement levels')
    ax.set_xlabel(r'$r_e$ ($\mu$m)')
    ax.set_ylabel('Altitude (km)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

fig.suptitle(
    'Moving average of in-situ $r_e$ profiles  '
    f'(specs: {", ".join(spec_label(s) for s in WINDOW_SPECS)})',
    y=1.005,
)
fig.tight_layout()

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIGURE_DIR / FIGURE_NAME
fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'\nFigure saved to: {fig_path}  (dpi={FIGURE_DPI})')


# ── 7-level equal-altitude reduction ──────────────────────────────────────────
# Each profile is collapsed to exactly N_REDUCED_LEVELS points whose altitudes
# are evenly spaced between cloud top and cloud base. Within each altitude bin
# the raw r_e values are averaged.
re_reduced   = []
z_reduced    = []
rmse_reduced = np.zeros(n_profiles)

for i, (re_raw, z_raw) in enumerate(zip(profiles_raw, altitudes_raw)):
    re_red, z_red = equal_altitude_bin_average(re_raw, z_raw, N_REDUCED_LEVELS)
    re_reduced.append(re_red)
    z_reduced.append(z_red)

    # Compare on the raw altitude grid: profiles run cloud-top → cloud-base, so
    # z may be monotonically decreasing. np.interp requires increasing xp, hence
    # the argsort.
    order = np.argsort(z_red)
    re_red_on_raw = np.interp(z_raw, z_red[order], re_red[order])
    rmse_reduced[i] = np.sqrt(np.mean((re_red_on_raw - re_raw) ** 2))

print()
print(f'{N_REDUCED_LEVELS}-level equal-altitude RMSE vs. raw profile (μm):')
print(f'  mean={rmse_reduced.mean():.4f}  median={np.median(rmse_reduced):.4f}  '
      f'max={rmse_reduced.max():.4f}  min={rmse_reduced.min():.4f}')


# ── 2 x 3 figure for the 7-level reduction ────────────────────────────────────
# Use the same panel_idx selection as the first figure so the comparison is
# apples-to-apples across raw profile lengths.
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(13, 8))
axes2 = axes2.flatten()

reduced_color = 'crimson'

for ax, idx in zip(axes2, panel_idx):
    re_raw = profiles_raw[idx]
    z_raw  = altitudes_raw[idx]

    ax.plot(re_raw, z_raw, color='0.4', linewidth=1.2,
            marker='o', markersize=2.8, label='Raw in-situ')
    ax.plot(re_reduced[idx], z_reduced[idx], color=reduced_color,
            linewidth=1.8, marker='s', markersize=5,
            label=f'{N_REDUCED_LEVELS}-level mean '
                  f'(RMSE={rmse_reduced[idx]:.3f} $\\mu$m)')

    ax.set_title(f'Profile {idx + 1}  |  {len(re_raw)} measurement levels '
                 f'→ {N_REDUCED_LEVELS}')
    ax.set_xlabel(r'$r_e$ ($\mu$m)')
    ax.set_ylabel('Altitude (km)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

fig2.suptitle(
    f'In-situ $r_e$ profiles reduced to {N_REDUCED_LEVELS} evenly spaced '
    'altitudes (per-bin average)',
    y=1.005,
)
fig2.tight_layout()

fig2_path = FIGURE_DIR / FIGURE_NAME_REDUCED
fig2.savefig(fig2_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f'Figure saved to: {fig2_path}  (dpi={FIGURE_DPI})')

plt.show()
