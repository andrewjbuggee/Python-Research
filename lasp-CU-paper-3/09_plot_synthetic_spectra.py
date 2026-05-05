"""
Quick-look plot of up to 3 simulated HySICS reflectance spectra produced by
hysics_refl_from_synthetic_NN_inputs.m on the supercomputer.

Picks SPECTRA_TO_PLOT random .mat files from SPECTRA_DIR (capped at 3),
extracts the noise-free reflectance + cloud truth (tau_c and the vertical-mean
r_e), and plots them on one axes with a legend describing each.

Usage:  python 09_plot_synthetic_spectra.py
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
SPECTRA_DIR     = Path('/Volumes/My Passport/neural_network_training_data/synthetic_training_data_5_May_2026/')
SPECTRA_TO_PLOT = 3                       # capped at the number of .mat files found
PLOT_NOISY      = False                   # True → use Refl_model_with_noise_*_hysics
RANDOM_SEED     = None                    # None → fresh pick each run
OUT_PNG         = SPECTRA_DIR / 'verify_synthetic_spectra.png'
FIG_DPI         = 500


# Paper-quality matplotlib style (mirrors regenerate_kfold_figures.setup_style)
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                          'Times New Roman'],
    'mathtext.fontset':  'cm',
    'mathtext.rm':       'serif',
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'figure.titlesize':  14,
    'legend.fontsize':   9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.linewidth':    0.8,
})


def load_one(path):
    """Pull (wavelength_nm, reflectance, tau_c, mean_re_um, sza, vza, saz, vaz)
    from one .mat file produced by hysics_refl_from_synthetic_NN_inputs.m."""
    d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    refl_key = 'Refl_model_with_noise_allStateVectors_hysics' if PLOT_NOISY \
               else 'Refl_model_allStateVectors'
    refl = np.asarray(d[refl_key]).ravel()

    # changing_variables: cols [sza, vza, vaz, saz, wl_lo, wl_hi, band_idx]
    cv = np.asarray(d['changing_variables_allStateVectors'])
    wl_center = 0.5 * (cv[:, 4] + cv[:, 5])

    # Sort by wavelength (channel order is already monotonic, but defensively)
    order = np.argsort(wl_center)
    wl_center = wl_center[order]
    refl      = refl[order]

    re_arr  = np.atleast_1d(np.asarray(d['re'])).astype(float).ravel()
    tau_val = float(np.atleast_1d(np.asarray(d['tau'])).ravel()[0])

    sza = float(cv[0, 0])
    vza = float(cv[0, 1])
    vaz = float(cv[0, 2])
    saz = float(cv[0, 3])

    return {
        'path':       path,
        'wavelength': wl_center,
        'reflectance': refl,
        'tau_c':      tau_val,
        're_mean':    float(re_arr.mean()),
        're_top':     float(re_arr[0]),
        're_base':    float(re_arr[-1]),
        'sza':        sza,
        'vza':        vza,
        'saz':        saz,
        'vaz':        vaz,
    }


# ── Pick files ────────────────────────────────────────────────────────────────
mat_files = sorted(SPECTRA_DIR.glob('*.mat'))
if not mat_files:
    raise FileNotFoundError(f'No .mat files found in {SPECTRA_DIR}')

n_show = min(SPECTRA_TO_PLOT, len(mat_files))
rng = np.random.default_rng(RANDOM_SEED)
picks = list(rng.choice(len(mat_files), size=n_show, replace=False))
chosen_paths = [mat_files[i] for i in picks]

print(f'Found {len(mat_files)} .mat file(s) in {SPECTRA_DIR.name}; plotting {n_show}')
for p in chosen_paths:
    print(f'  - {p.name}')


# ── Load all chosen spectra once ──────────────────────────────────────────────
spectra = [load_one(p) for p in chosen_paths]


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

palette = ['steelblue', 'darkorange', 'seagreen', 'firebrick',
           'mediumpurple', 'goldenrod']

for k, s in enumerate(spectra):
    label = (
        fr'$\tau_c$={s["tau_c"]:.2f},  '
        fr'$\langle r_e \rangle$={s["re_mean"]:.2f} $\mu$m  '
        fr'(top={s["re_top"]:.1f}, base={s["re_base"]:.1f})  |  '
        fr'SZA={s["sza"]:.0f}$^\circ$, VZA={s["vza"]:.0f}$^\circ$'
    )
    ax.plot(s['wavelength'], s['reflectance'],
            color=palette[k % len(palette)], linewidth=1.0,
            label=label)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title(
    f'Synthetic HySICS reflectance spectra '
    f'({"with HySICS noise" if PLOT_NOISY else "noise-free"}),  N = {n_show}'
)
ax.set_xlim(min(s['wavelength'][0]  for s in spectra),
            max(s['wavelength'][-1] for s in spectra))
ax.grid(alpha=0.3)
ax.legend(loc='best', frameon=True, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=FIG_DPI, bbox_inches='tight')
print(f'\nSaved → {OUT_PNG}  (dpi={FIG_DPI})')

plt.show()
