"""
Quick-look verification of the training-inputs NetCDF written by 07_build_training_inputs.

Reads the .nc and produces three paper-quality figures:

  Fig 1 (2x2)  — geometry histograms: SZA, SAZ, VZA, VAZ
  Fig 2 (2x3)  — bulk cloud scalars: alpha, tau_c, LWP, z_top, z_base, thickness
  Fig 3 (2x2)  — per-level envelopes (5/50/95): r_e(z), LWC(z), T(P), vapor(P)

Also prints a one-screen summary of every variable so unexpected values jump out,
and runs the checks the MATLAB driver (hysics_refl_from_synthetic_NN_inputs.m)
depends on. The script exits with status 1 if a check fails, so it can gate a
SLURM submission:  python 08_verify_training_inputs.py <file.nc> && sbatch ...

Usage
    python 08_verify_training_inputs.py                 # newest training_inputs_*.nc
    python 08_verify_training_inputs.py path/to/file.nc

Changes on 12 Nov 2026: NC path from the command line (default = newest file),
figure names carry the .nc stem so runs never overwrite each other, driver
pre-flight checks, a Mie-table α panel, and no blocking plt.show().
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')            # write files only; never open windows
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc             # noqa: E402
import numpy as np               # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
IN_DIR = SCRIPT_DIR / 'training_inputs'
if len(sys.argv) > 1:
    NC_PATH = Path(sys.argv[1])
else:
    # newest training-inputs file by modification time
    NC_PATH = max(IN_DIR.glob('training_inputs_*.nc'), key=lambda f: f.stat().st_mtime)
OUT_DIR = NC_PATH.parent
FIG_TAG = NC_PATH.stem.replace('training_inputs_', '')   # e.g. jointMVN_N100000_L7_2026-11-12
FIG_DPI = 500

# Limits the MATLAB driver / libRadtran set-up depends on (same as 07).
RE_MIN_UM, RE_MAX_UM = 0.1, 50.0
ALPHA_RANGE = (1.0, 150.0)
SZA_MAX_DEG = 50.0


def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'figure.titlesize':  14,
        'legend.fontsize':   8,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.linewidth':    0.8,
    })


setup_style()


# ── Load ──────────────────────────────────────────────────────────────────────
print(f'Reading {NC_PATH}')
with nc.Dataset(NC_PATH, 'r') as ds:
    ds.set_auto_mask(False)    # plain ndarrays (07 writes no fill values)
    re_um         = ds.variables['re_um'][:]                  # (N, 7) μm
    lwc_g         = ds.variables['lwc_g_per_m3'][:]           # (N, 7) g/m³
    z_km          = ds.variables['z_km'][:]                   # (N, 7) km
    alpha         = ds.variables['alpha'][:]                  # (N,)
    tau_c         = ds.variables['tau_c'][:]                  # (N,)
    LWP           = ds.variables['LWP_g_per_m2'][:]           # (N,)
    sza           = ds.variables['sza_deg'][:]                # (N,)
    saz           = ds.variables['saz_deg'][:]                # (N,)
    vza           = ds.variables['vza_deg'][:]                # (N,)
    vaz           = ds.variables['vaz_deg'][:]                # (N,)
    T_K           = ds.variables['T_K'][:]                    # (N, 37)
    vapor         = ds.variables['vapor_molec_per_cm3'][:]    # (N, 37)
    pressure_hPa  = ds.variables['pressure_hPa'][:]           # (37,)
    mie_alpha_tab = (ds.variables['mie_alpha_table'][:]       # (N,) α of Mie table
                     if 'mie_alpha_table' in ds.variables else None)
    attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    n_cloud, n_lev_cld = re_um.shape
    n_lev_atm = T_K.shape[1]

z_top     = z_km[:, 0]      # index 0 = top
z_base    = z_km[:, -1]
thickness = z_top - z_base


# ── Console summary ───────────────────────────────────────────────────────────
def stats(name, x, units=''):
    x = np.asarray(x).ravel()
    print(f'  {name:>22}  [{x.min():>10.3f}, {x.max():>10.3f}]  '
          f'median={np.median(x):>9.3f}  mean={x.mean():>9.3f}  {units}')

print(f'\nN_CLOUDS = {n_cloud}, level_cloud = {n_lev_cld}, level_atmos = {n_lev_atm}')
print(f'Pressure grid (hPa): [{pressure_hPa.min():.1f}, {pressure_hPa.max():.1f}]\n')
print('Per-variable summary (over all clouds and levels where applicable):')
stats('re',         re_um,     'μm')
stats('lwc',        lwc_g,     'g/m³')
stats('z',          z_km,      'km')
stats('z_top',      z_top,     'km')
stats('z_base',     z_base,    'km')
stats('thickness',  thickness, 'km')
stats('alpha',      alpha)
stats('tau_c',      tau_c)
stats('LWP',        LWP,       'g/m²')
stats('sza',        sza,       'deg')
stats('saz',        saz,       'deg')
stats('vza',        vza,       'deg')
stats('vaz',        vaz,       'deg')
stats('T',          T_K,       'K')
stats('vapor',      vapor,     'molec/cm³')

print('\nGlobal attributes:')
for k in ('source_npz', 'source_insitu_mat', 'K_re', 'K_lwc', 'K_T', 'K_vapor',
          'tau_c_method', 'geometry_random_seed', 'date_created'):
    if k in attrs:
        print(f'  {k:>22} = {attrs[k]}')


# ── Pre-flight checks for the MATLAB driver ───────────────────────────────────
# Hard failures stop a SLURM job from being submitted; warnings are listed with
# their 1-based cloud index (the idx the MATLAB driver takes).
def _ids(mask):
    ids = (np.flatnonzero(mask)[:10] + 1).tolist()
    return f'  (MATLAB idx: {ids}{" …" if mask.sum() > 10 else ""})' if mask.any() else ''

re_np, lwc_np, z_np = (np.asarray(a, dtype=np.float64) for a in (re_um, lwc_g, z_km))
hard = {
    'masked/non-finite values': ~(np.isfinite(re_np).all(1) & np.isfinite(lwc_np).all(1)
                                  & np.isfinite(z_np).all(1)
                                  & np.isfinite(np.asarray(T_K)).all(1)
                                  & np.isfinite(np.asarray(vapor)).all(1)),
    f'r_e >= {RE_MAX_UM} μm': (re_np >= RE_MAX_UM).any(1),
    'z not strictly decreasing (top → base)': ~(np.diff(z_np, axis=1) < 0).all(1),
    f'alpha outside {ALPHA_RANGE}': (np.asarray(alpha) < ALPHA_RANGE[0])
                                    | (np.asarray(alpha) > ALPHA_RANGE[1]),
    f'SZA outside [0, {SZA_MAX_DEG})': (np.asarray(sza) < 0) | (np.asarray(sza) >= SZA_MAX_DEG),
}
soft = {
    f'r_e <= {RE_MIN_UM} μm at a level': (re_np <= RE_MIN_UM).any(1),
    'LWC <= 0 at a level': (lwc_np <= 0).any(1),
}
print('\nPre-flight checks:')
n_hard = 0
for label, m in hard.items():
    print(f'  {"FAIL" if m.any() else "ok  "}  {label}: {int(m.sum())}{_ids(m)}')
    n_hard += int(m.sum())
for label, m in soft.items():
    print(f'  {"WARN" if m.any() else "ok  "}  {label}: {int(m.sum())}{_ids(m)}')


# ── Figure 1: geometry histograms (2 x 2) ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

def hist_panel(ax, x, xlabel, title, color, expected_range=None):
    ax.hist(x, bins=60, density=True, color=color, edgecolor='white', linewidth=0.4)
    if expected_range is not None:
        for v in expected_range:
            ax.axvline(v, color='0.4', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(alpha=0.3)

hist_panel(axes[0, 0], sza, 'SZA (deg)',
           f'Solar zenith angle  (n={n_cloud})', 'steelblue',
           expected_range=(0, 50))
hist_panel(axes[0, 1], saz, 'SAZ (deg)',
           'Solar azimuth angle', 'darkorange',
           expected_range=(0, 360))
hist_panel(axes[1, 0], vza, 'VZA (deg)',
           'Viewing zenith angle  (Uniform[0,45])', 'seagreen',
           expected_range=(0, 45))
hist_panel(axes[1, 1], vaz, 'VAZ (deg)',
           'Viewing azimuth angle  (Uniform[0,180])', 'firebrick',
           expected_range=(0, 180))

fig.suptitle('Geometry distributions sampled per synthetic cloud', y=1.005)
fig.tight_layout()
out1 = OUT_DIR / f'verify_geometry_histograms_{FIG_TAG}.png'
fig.savefig(out1, dpi=FIG_DPI, bbox_inches='tight')
print(f'\nFig 1 → {out1}  (dpi={FIG_DPI})')


# ── Figure 2: bulk cloud scalars (2 x 3) ──────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

def hist_log(ax, x, xlabel, title, color):
    x = np.asarray(x)
    x_pos = x[x > 0]
    bins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), 50)
    ax.hist(x_pos, bins=bins, density=True, color=color, edgecolor='white', linewidth=0.4)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')

hist_panel(axes[0, 0], alpha,    r'$\alpha$',           r'libRadtran shape $\alpha$', 'steelblue')
hist_log  (axes[0, 1], tau_c,    r'$\tau_c$',           r'Cloud optical depth $\tau_c$ (log)', 'darkorange')
hist_log  (axes[0, 2], LWP,      r'LWP (g/m$^2$)',      'Liquid water path (log)', 'seagreen')
hist_panel(axes[1, 0], z_top,    'z_top (km)',          'Cloud top altitude', 'firebrick')
hist_panel(axes[1, 1], z_base,   'z_base (km)',         'Cloud base altitude', 'mediumpurple')
hist_panel(axes[1, 2], thickness,'thickness (km)',      'Cloud geometric depth', 'goldenrod')
# Mie-table α: the generator labelled tau_c with the table nearest each α; the
# driver picks its table with the same nearest-α rule.
if mie_alpha_tab is not None:
    vals, counts = np.unique(np.asarray(mie_alpha_tab), return_counts=True)
    axes[0, 3].bar(np.arange(vals.size), counts / counts.sum(), color='steelblue')
    step = max(1, vals.size // 12)
    axes[0, 3].set_xticks(np.arange(vals.size)[::step])
    axes[0, 3].set_xticklabels(vals[::step])
    axes[0, 3].set_xlabel(r'$\alpha$ of Mie table used')
    axes[0, 3].set_ylabel('Fraction of clouds')
    axes[0, 3].set_title('Custom Mie table per cloud')
    axes[0, 3].grid(alpha=0.3)
else:
    axes[0, 3].set_visible(False)
# relative azimuth |SAZ − VAZ| folded to [0, 180]
rel_az = np.abs((np.asarray(saz) - np.asarray(vaz) + 180.0) % 360.0 - 180.0)
hist_panel(axes[1, 3], rel_az, '|SAZ − VAZ| (deg, folded to 0–180)',
           'Relative azimuth', 'slategray', expected_range=(0, 180))

fig.suptitle('Bulk cloud scalars (per synthetic cloud)', y=1.005)
fig.tight_layout()
out2 = OUT_DIR / f'verify_cloud_scalars_{FIG_TAG}.png'
fig.savefig(out2, dpi=FIG_DPI, bbox_inches='tight')
print(f'Fig 2 → {out2}  (dpi={FIG_DPI})')


# ── Figure 3: per-level envelopes (2 x 2) ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
pct = (5, 50, 95)

# Cloud profiles indexed top→base; for plotting use a normalized in-cloud axis
# u = 1 at top, 0 at base, identical for every cloud.
u_cloud = np.linspace(1.0, 0.0, n_lev_cld)

def envelope(ax, mat_2d, y_axis, xlabel, ylabel, title, log_x=False):
    p = np.percentile(mat_2d, pct, axis=0)
    ax.fill_betweenx(y_axis, p[0], p[2], alpha=0.30,
                     color='steelblue', label='5–95%')
    ax.plot(p[1], y_axis, color='steelblue', linewidth=1.6, label='median')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_x:
        ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both' if log_x else 'major')

envelope(axes[0, 0], re_um,  u_cloud,
         r'$r_e$ ($\mu$m)', 'Normalized in-cloud altitude (1 = top)',
         r'Per-level $r_e$ envelope')
envelope(axes[0, 1], lwc_g,  u_cloud,
         r'LWC (g/m$^3$, log)', 'Normalized in-cloud altitude (1 = top)',
         'Per-level LWC envelope', log_x=True)

# Atmospheric profiles on the ERA5 pressure grid (37 levels, surface→TOA)
ax = axes[1, 0]
p_T = np.percentile(T_K, pct, axis=0)
ax.fill_betweenx(pressure_hPa, p_T[0], p_T[2], alpha=0.30,
                 color='steelblue', label='5–95%')
ax.plot(p_T[1], pressure_hPa, color='steelblue', linewidth=1.6, label='median')
ax.invert_yaxis()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Per-level T envelope')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
p_v = np.percentile(vapor, pct, axis=0)
ax.fill_betweenx(pressure_hPa, p_v[0], p_v[2], alpha=0.30,
                 color='steelblue', label='5–95%')
ax.plot(p_v[1], pressure_hPa, color='steelblue', linewidth=1.6, label='median')
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel(r'Vapor (molec/cm$^3$, log)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Per-level vapor envelope')
ax.legend()
ax.grid(alpha=0.3, which='both')

fig.suptitle(f'Per-level envelopes across {n_cloud} synthetic clouds', y=1.005)
fig.tight_layout()
out3 = OUT_DIR / f'verify_profile_envelopes_{FIG_TAG}.png'
fig.savefig(out3, dpi=FIG_DPI, bbox_inches='tight')
print(f'Fig 3 → {out3}  (dpi={FIG_DPI})')

plt.close('all')

if n_hard:
    print(f'\n{n_hard} hard check failure(s): do NOT submit this file.')
    sys.exit(1)
print('\nAll hard checks passed.')
