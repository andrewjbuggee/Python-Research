"""
Quick-look verification of the training-inputs NetCDF written by 07_build_training_inputs.

Reads the .nc and produces three paper-quality figures:

  Fig 1 (2x2)  — geometry histograms: SZA, SAZ, VZA, VAZ
  Fig 2 (2x3)  — bulk cloud scalars: alpha, tau_c, LWP, z_top, z_base, thickness
  Fig 3 (2x2)  — per-level envelopes (5/50/95): r_e(z), LWC(z), T(P), vapor(P)

Also prints a one-screen summary of every variable so unexpected values jump out.
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
NC_PATH     = SCRIPT_DIR / 'training_inputs' / 'training_inputs_jointMVN_N300000_L7.nc'
OUT_DIR     = NC_PATH.parent
FIG_DPI     = 500


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
out1 = OUT_DIR / 'verify_geometry_histograms.png'
fig.savefig(out1, dpi=FIG_DPI, bbox_inches='tight')
print(f'\nFig 1 → {out1}  (dpi={FIG_DPI})')


# ── Figure 2: bulk cloud scalars (2 x 3) ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

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

fig.suptitle('Bulk cloud scalars (per synthetic cloud)', y=1.005)
fig.tight_layout()
out2 = OUT_DIR / 'verify_cloud_scalars.png'
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
out3 = OUT_DIR / 'verify_profile_envelopes.png'
fig.savefig(out3, dpi=FIG_DPI, bbox_inches='tight')
print(f'Fig 3 → {out3}  (dpi={FIG_DPI})')

plt.show()
