"""
compute_hysics_re_sensitivity.py
================================
Per-band sensitivity of HySICS top-of-atmosphere reflectance to a vertically
uniform change in droplet effective radius, dR_lambda / d ln(r_e), at fixed
cloud optical thickness.

Why this is needed
------------------
The PC-truncation analysis (pc_truncation_analysis.py) builds a linear
forward model for the r_e(tau) profile following Platnick (2000):

    R_lambda  ~  R_lambda( r_e*_lambda ),   r_e*_lambda = ∫ w_lambda(tau) r_e(tau) dtau

where w_lambda(tau) is the (unit-integral) vertical weighting function.
Differentiating gives the Jacobian of reflectance with respect to the r_e
profile:

    dR_lambda / d r_e(tau)  =  (dR_lambda / d r_e)|_homogeneous  *  w_lambda(tau)

so the Jacobian factorises into (i) the weighting function, which is already
on disk (the same DISORT run behind Fig. 5 of the paper), and (ii) a single
number per band, the homogeneous-cloud sensitivity dR/dr_e. This script
computes (ii). Non-absorbing (visible) bands have dR/dr_e ~ 0, which is
exactly why they carry no droplet-size information even though their
weighting functions are well defined.

Method
------
* libRadtran uvspec + DISORT (16 streams), REPTRAN coarse gas absorption,
  output_quantity reflectivity, 1-nm output from 350 to 2300 nm.
* Plane-parallel, vertically homogeneous water cloud from 0.75 to 1.25 km
  (the same geometry as the weighting-function run), tau(550 nm) fixed at
  TAU_C with `wc_modify tau550 set`, so r_e is varied at constant tau.
* Central difference in ln(r_e):  r_e = R_E0 * exp(±DLN).
* The 1-nm reflectivity spectra are convolved with the HySICS spectral
  response functions stored in the weighting-function .mat file.
* Geometry and atmosphere match the weighting-function run
  (sza / vza / azimuths read from its file, albedo 0.04 lambertian,
  boundary-layer aerosol tau = 0.1, CO2 = 416 ppm, afglus atmosphere).

Outputs
-------
hysics_re_sensitivity_<date>.npz with
    wl_nm        (636,)  HySICS band centre wavelength
    R0           (636,)  reflectivity at R_E0
    dR_dlnre     (636,)  central-difference dR/d ln r_e   (tau fixed)
    dR_dlntau    (636,)  central-difference dR/d ln tau   (r_e fixed)
    re0_um, tau_c, dln, sza_deg   (scalars, the settings used)

Usage
-----
    python compute_hysics_re_sensitivity.py            # ~30 s on an M-series Mac

Author: Claude (assistant) for Andrew J. Buggee, Nov 2026
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import numpy as np
import scipy.io

# ── Configuration ──────────────────────────────────────────────────────────────
UVSPEC = Path('/usr/local/bin/uvspec')     # arm64 build (the copy inside
                                           # libRadtran-2.0.4/bin is x86_64 and
                                           # no longer links on this machine)
# Two partial libRadtran data trees exist on this laptop: the one installed with
# the arm64 build has the aerosol/REPTRAN tables but no Mie water-cloud tables;
# the old 2.0.4 tree has the Mie tables (wc/mie/wc.sol.mie.cdf, gamma alpha = 7)
# but no aerosol tables. main() assembles a combined tree from symlinks.
LRT_DATA_MAIN = Path('/usr/local/share/libRadtran/data')
LRT_DATA_WC = Path('/Users/andrewbuggee/Documents/CU-Boulder-ATOC/'
                   'Hyperspectral-Cloud-Droplet-Retrieval/LibRadTran/'
                   'libRadtran-2.0.4/data/wc')
WF_FILE = Path('/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/'
               'Hyperspectral_Cloud_Retrievals/HySICS/weighting_functions/'
               'disort_HySICS_weightingFunctions_platnick2000-profile_'
               'cloud-top-view_allBands_sim-ran-on-27-Jun-2025_rev1.mat')
# NB: the sibling '..._near-nadir-view_...' file (sza 10°, vza 0°, Cox–Munk
# ocean) is sun-glint contaminated: R(τ) DEcreases with τ for τ < 1 and its
# weighting functions are negative near cloud top. Do not use it.

RE0_UM = 8.0          # nominal effective radius (μm); ~ r_m of the 12 → 5 μm
                      # adiabatic profile used for the weighting functions
DLN = 0.05            # ± 5 % perturbation in ln(r_e)
TAU_C = 8.0           # cloud optical thickness at 550 nm (matches the WF run)
Z_TOP_KM, Z_BOT_KM = 1.25, 0.75
# Sun / view geometry are read from the weighting-function file in main() so
# the two calculations always match (sza ≈ 49.5°, vza ≈ 31.8°, relative
# azimuth from phi0 = 0°, vaz = 210°).
LWC_DUMMY_GM3 = 0.3   # any positive LWC; tau is overwritten by tau550 set

OUT_DIR = Path(__file__).parent / 'outputs'
WORK_DIR = OUT_DIR / 'uvspec_work'
LRT_DATA = WORK_DIR / 'lrt_data'          # combined tree (symlinks), see above


def build_combined_data_tree() -> None:
    """Symlink every sub-folder of LRT_DATA_MAIN into LRT_DATA, except wc/,
    which points at the tree that holds the Mie water-cloud tables."""
    LRT_DATA.mkdir(parents=True, exist_ok=True)
    for sub in LRT_DATA_MAIN.iterdir():
        link = LRT_DATA / sub.name
        target = LRT_DATA_WC if sub.name == 'wc' else sub
        if not link.exists():
            link.symlink_to(target)


def write_wc_file(path: Path, re_um: float) -> None:
    """Vertically homogeneous libRadtran 1D water-cloud file.

    Columns: altitude (km), LWC (g/m³), r_eff (μm), from cloud top downward.
    Because the cloud is homogeneous, libRadtran's level/layer convention does
    not matter here; total tau is then fixed by `wc_modify tau550 set`.
    """
    z = np.linspace(Z_TOP_KM, Z_BOT_KM, 6)
    with open(path, 'w') as f:
        f.write('# z(km)  LWC(g/m3)  reff(um)\n')
        for zz in z:
            f.write(f'{zz:.4f}  {LWC_DUMMY_GM3:.4f}  {re_um:.4f}\n')


def run_uvspec(re_um: float, tag: str, geom: dict,
               tau_c: float = TAU_C) -> tuple[np.ndarray, np.ndarray]:
    """Run uvspec once for a homogeneous cloud; return (wl_nm, reflectivity)."""
    wc_path = WORK_DIR / f'wc_{tag}.DAT'
    write_wc_file(wc_path, re_um)
    inp = f"""rte_solver disort
number_of_streams 16
disort_intcor phase
data_files_path {LRT_DATA}/
mol_abs_param reptran coarse
atmosphere_file {LRT_DATA}/atmmod/afglus.dat
source solar {LRT_DATA}/solar_flux/kurudz_1.0nm.dat
day_of_year 316
albedo 0.04
wc_file 1D {wc_path}
wc_properties mie interpolate
wc_modify tau550 set {tau_c}
wavelength 350 2300
crs_model rayleigh Bodhaine29
mixing_ratio CO2 416
aerosol_default
aerosol_haze 4
aerosol_modify tau set 0.1
zout toa
sza {geom['sza']}
phi0 {geom['phi0']}
umu {geom['umu']}
phi {geom['phi']}
output_quantity reflectivity
output_user lambda uu
quiet
"""
    (WORK_DIR / f'{tag}.INP').write_text(inp)
    res = subprocess.run([str(UVSPEC)], input=inp, capture_output=True,
                         text=True, check=False)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f'uvspec failed for {tag}:\n{res.stderr[-2000:]}')
    arr = np.loadtxt(res.stdout.splitlines())
    return arr[:, 0], arr[:, 1]


def convolve_to_hysics(wl_nm: np.ndarray, refl: np.ndarray,
                       srf_wl: np.ndarray, srf_val: np.ndarray) -> np.ndarray:
    """Band-average a 1-nm spectrum with each HySICS spectral response function.

    srf_wl, srf_val : (n_bands, n_samples) wavelength (nm) and response.
    Returns (n_bands,) response-weighted mean reflectivity.
    """
    out = np.empty(srf_wl.shape[0])
    for b in range(srf_wl.shape[0]):
        r_interp = np.interp(srf_wl[b], wl_nm, refl)        # (n_samples,)
        out[b] = np.trapezoid(srf_val[b] * r_interp, srf_wl[b]) / \
            np.trapezoid(srf_val[b], srf_wl[b])
    return out


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    build_combined_data_tree()

    wf = scipy.io.loadmat(WF_FILE, squeeze_me=True, struct_as_record=False)
    srf_wl = np.asarray(wf['spec_response'].wavelength, dtype=float)   # (636, 400)
    srf_val = np.asarray(wf['spec_response'].value, dtype=float)      # (636, 400)
    wl_center_nm = srf_wl.mean(axis=1)
    RT = wf['inputs'].RT
    geom = dict(sza=float(RT.sza), phi0=float(RT.phi0),
                umu=float(np.cos(np.deg2rad(RT.vza))), phi=float(RT.vaz))
    print(f'geometry from WF file: {geom}')

    # r_e is perturbed at fixed tau, and tau at fixed r_e. The tau derivative
    # is needed because tau is retrieved jointly with the profile: at fixed tau
    # even visible bands respond to r_e (through the asymmetry parameter), but
    # that signal is largely degenerate with a change in tau, so the
    # information analysis must carry ln(tau) as an extra unknown.
    runs = [('minus',     RE0_UM * np.exp(-DLN), TAU_C),
            ('zero',      RE0_UM,                TAU_C),
            ('plus',      RE0_UM * np.exp(+DLN), TAU_C),
            ('tau_minus', RE0_UM,                TAU_C * np.exp(-DLN)),
            ('tau_plus',  RE0_UM,                TAU_C * np.exp(+DLN))]
    spectra = {}
    for tag, re_um, tau in runs:
        print(f'uvspec: r_e = {re_um:.3f} μm, tau = {tau:.3f} ({tag}) ...',
              flush=True)
        wl, r = run_uvspec(re_um, tag, geom, tau)
        spectra[tag] = convolve_to_hysics(wl, r, srf_wl, srf_val)

    # Central differences in ln r_e and ln tau (second-order accurate in DLN).
    dR_dlnre = (spectra['plus'] - spectra['minus']) / (2.0 * DLN)
    dR_dlntau = (spectra['tau_plus'] - spectra['tau_minus']) / (2.0 * DLN)

    date = datetime.date.today().isoformat()
    out = OUT_DIR / f'hysics_re_sensitivity_{date}.npz'
    np.savez(out, wl_nm=wl_center_nm, R0=spectra['zero'], dR_dlnre=dR_dlnre,
             dR_dlntau=dR_dlntau,
             R_minus=spectra['minus'], R_plus=spectra['plus'],
             re0_um=RE0_UM, tau_c=TAU_C, dln=DLN, sza_deg=geom['sza'],
             umu=geom['umu'])
    print(f'saved {out}')

    # Quick sanity print at a few familiar wavelengths.
    for target in (645, 860, 1240, 1640, 2130):
        b = int(np.argmin(abs(wl_center_nm - target)))
        print(f'  {wl_center_nm[b]:7.1f} nm   R = {spectra["zero"][b]:.4f}   '
              f'dlnR/dlnre = {dR_dlnre[b] / spectra["zero"][b]:+.3f}   '
              f'dlnR/dlntau = {dR_dlntau[b] / spectra["zero"][b]:+.3f}')


if __name__ == '__main__':
    main()
