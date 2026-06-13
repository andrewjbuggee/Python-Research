"""
diagnose_uv_importance_run004.py — Why does the 351.95 nm channel (input idx 0)
get the highest permutation importance for the droplet-profile retrieval, when
King & Vaughan (2012) show little r_t/r_b information below ~0.5 um and the
single-scattering albedo is ~1 there (no droplet-absorption signal)?

Runs five cheap diagnostics on the run_004 model + training HDF5 to separate a
normalization/edge artifact from a real (if indirect) physical anchor:

  T1  Per-channel SNR & dynamic range      -> is ch0 a low-SNR / low-range edge?
  T2  Adjacent-channel correlation         -> is ch0's reliance non-redundant?
  T3  First-layer weight-column norm        -> did training put special weight on ch0?
  T4  Physical correlations of raw refl     -> what does ch0 encode (tau / height / wv)?
  T5  UV importance falloff vs Rayleigh λ^-4 -> is the spectral shape physical?

Nothing here retrains; the definitive causal test (retrain with the UV channels
removed) is proposed separately.

Run:
    /opt/anaconda3/bin/python3 diagnose_uv_importance_run004.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
RUN_DIR = (REPO / 'hyper_parameter_sweep'
           / 'sweep_results_profile_only_synthetic_M0_rev2' / 'run_004')
H5 = REPO / 'training_data' / 'synthetic_training_data_7-levels_8_May_2026.h5'
OUT_DIR = REPO / 'feature_importance_run004'


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    with h5py.File(H5, 'r') as f:
        refl = f['reflectances_hysics'][:].astype(np.float64)      # (N,636)
        uncert = f['reflectances_uncertainty_hysics'][:].astype(np.float64)
        wl_nm = f['wavelengths'][:].astype(np.float64)
        tau_c = f['tau_c'][:].astype(np.float64)
        lwc = f['lwc'][:].astype(np.float64)                       # (N,7) g/m^3
        z_km = f['profiles_raw_z'][:].astype(np.float64)           # (N,7) top->base
        wv_above = f['wv_above_cloud'][:].astype(np.float64)
        re_prof = f['profiles_raw'][:].astype(np.float64)          # (N,7) um, top->base
    N, n_ch = refl.shape
    wl_um = wl_nm / 1000.0

    # derived physical scalars
    re_top, re_base, re_mean = re_prof[:, 0], re_prof[:, -1], re_prof.mean(1)
    z_top = z_km[:, 0]                                             # cloud-top height
    thickness = z_km[:, 0] - z_km[:, -1]
    # LWP ~ integral of lwc dz  (sign-safe; units g/m^2 up to a constant)
    dz_m = np.abs(np.gradient(z_km, axis=1)) * 1000.0
    lwp = (lwc * dz_m).sum(1)

    # importance (100-rep permutation) for context
    fi = np.load(OUT_DIR / 'feature_importance_run004.npz', allow_pickle=True)
    imp = fi['perm_mean_um'][:n_ch]

    # ── T1: SNR & dynamic range per channel ──────────────────────────────────
    rng_ch = refl.max(0) - refl.min(0)
    std_ch = refl.std(0)
    snr_ch = std_ch / np.maximum(uncert.mean(0), 1e-12)
    def pr(name, arr, c):  # percentile rank of channel c within arr (0=low,100=high)
        return float((arr < arr[c]).mean() * 100)
    print('=== T1: SNR & dynamic range (is ch0 a low-SNR/low-range edge?) ===')
    print(f'{"idx":>3} {"λ(nm)":>8} {"range":>8} {"std":>8} {"uncert":>9} '
          f'{"SNR":>7} {"SNR %ile":>9} {"imp(μm)":>8}')
    for c in [0, 1, 2, 5, 10]:
        print(f'{c:>3} {wl_nm[c]:>8.1f} {rng_ch[c]:>8.4f} {std_ch[c]:>8.4f} '
              f'{uncert.mean(0)[c]:>9.5f} {snr_ch[c]:>7.1f} '
              f'{pr("snr",snr_ch,c):>8.1f}% {imp[c]:>8.3f}')
    print(f'  spectrum SNR: min={snr_ch.min():.1f} median={np.median(snr_ch):.1f} '
          f'max={snr_ch.max():.1f}')
    print(f'  ch0 range percentile among 636 channels: '
          f'{pr("rng",rng_ch,0):.1f}%  (100% = largest range)')

    # ── T2: adjacent-channel correlation (redundant or not?) ─────────────────
    print('\n=== T2: correlation of ch0 with nearby channels ===')
    for c in [1, 2, 3, 5, 10, 20]:
        print(f'  corr(ch0 @352nm, ch{c} @{wl_nm[c]:.0f}nm) = '
              f'{pearson(refl[:, 0], refl[:, c]):+.4f}')
    print('  (if ~0.99 yet single-channel shuffle of ch0 still costs '
          f'{imp[0]:.2f} μm, the model relies on ch0 NON-redundantly)')

    # ── T3: first-layer weight-column norm ───────────────────────────────────
    ck = torch.load(RUN_DIR / 'best_model.pt', map_location='cpu',
                    weights_only=False)
    state = ck['model_state_dict'] if (isinstance(ck, dict)
                                       and 'model_state_dict' in ck) else ck
    W0 = state['encoder.0.weight'].numpy()                        # (512, 643)
    wnorm = np.linalg.norm(W0[:, :n_ch], axis=0)                  # per input channel
    print('\n=== T3: first-layer weight-column L2 norm (did training weight ch0?) ===')
    print(f'  ch0 ||W|| = {wnorm[0]:.3f}  (percentile {pr("w",wnorm,0):.1f}%, '
          f'median {np.median(wnorm):.3f}, max {wnorm.max():.3f} '
          f'at {wl_nm[int(wnorm.argmax())]:.0f}nm)')

    # ── T4: physical correlations of raw reflectance ─────────────────────────
    targets = {'tau_c': tau_c, 'LWP': lwp, 'cloud_top_z': z_top,
               'thickness': thickness, 'wv_above': wv_above,
               're_mean': re_mean, 're_top': re_top, 're_base': re_base}
    # reference channels: VIS, NIR, two SWIR absorption peaks
    refs = {c_idx: wl_nm[c_idx] for c_idx in
            [0, int(np.argmin(abs(wl_nm-550))), int(np.argmin(abs(wl_nm-860))),
             int(np.argmin(abs(wl_nm-1600))), 532]}
    print('\n=== T4: Pearson r between RAW reflectance and physical state ===')
    hdr = '  '.join(f'{k:>11}' for k in targets)
    print(f'{"channel":>14}  {hdr}')
    for c_idx, lam in refs.items():
        row = '  '.join(f'{pearson(refl[:, c_idx], v):>+11.3f}'
                        for v in targets.values())
        print(f'{f"{lam:.0f}nm(#{c_idx})":>14}  {row}')

    # ── T5: UV importance falloff vs Rayleigh λ^-4 ───────────────────────────
    print('\n=== T5: UV importance falloff vs Rayleigh (∝ λ^-4) ===')
    ray = (wl_nm[0] / wl_nm[:6]) ** 4          # normalized to ch0
    print(f'{"idx":>3} {"λ(nm)":>8} {"imp/imp0":>9} {"Rayleigh λ^-4 (norm)":>21}')
    for c in range(6):
        print(f'{c:>3} {wl_nm[c]:>8.1f} {imp[c]/imp[0]:>9.3f} {ray[c]:>21.3f}')

    # ── figure: SNR, weight norm, and best physical scatter ──────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm'})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    axs[0].plot(wl_um, snr_ch, lw=0.8, color='#444')
    axs[0].scatter([wl_um[0]], [snr_ch[0]], color='#CC2A1E', zorder=5,
                   label='352 nm (ch0)')
    axs[0].set_xlabel('Wavelength (μm)'); axs[0].set_ylabel('SNR = std / uncertainty')
    axs[0].set_title('(a)  Per-channel SNR'); axs[0].legend(); axs[0].grid(alpha=0.3)

    axs[1].plot(wl_um, wnorm, lw=0.8, color='#1F5FC2')
    axs[1].scatter([wl_um[0]], [wnorm[0]], color='#CC2A1E', zorder=5,
                   label='352 nm (ch0)')
    axs[1].set_xlabel('Wavelength (μm)')
    axs[1].set_ylabel('First-layer ‖W[:, ch]‖')
    axs[1].set_title('(b)  Learned input weight'); axs[1].legend(); axs[1].grid(alpha=0.3)

    # best-correlated physical driver for ch0
    best_key = max(targets, key=lambda k: abs(pearson(refl[:, 0], targets[k])))
    axs[2].scatter(targets[best_key], refl[:, 0], s=3, alpha=0.25, color='#117733')
    axs[2].set_xlabel(best_key)
    axs[2].set_ylabel('352 nm reflectance')
    axs[2].set_title(f'(c)  ch0 vs {best_key}  '
                     f'(r = {pearson(refl[:, 0], targets[best_key]):+.2f})')
    axs[2].grid(alpha=0.3)

    fig.suptitle('Diagnostics for the 352 nm (idx 0) feature-importance peak '
                 '(run_004)', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / 'fig5_uv_importance_diagnostics.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out.name}')


if __name__ == '__main__':
    main()
