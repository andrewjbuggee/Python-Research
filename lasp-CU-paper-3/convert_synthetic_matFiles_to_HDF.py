"""
Convert synthetic-cloud .mat files (one per cloud, one geometry per file)
into a single HDF5 with the SAME schema the existing data.py training
pipeline already understands.

Sibling of convert_matFiles_to_HDF.py. The differences from the VOCALS+ORACLES
converter are localized to (a) the per-file read block — synthetic .mat files
have a single (SZA, SAZ, VZA, VAZ) per cloud and profiles are already on the
7-level grid by construction, so no interpolation step is needed — and (b) the
ERA5 access pattern, since the synthetic save writes era5.datProfiles as a named
struct rather than the field-indexed layout used in the original ORACLES files.

HDF5 schema (identical to the existing converter's output):

  /reflectances_hysics              (n_total, 636)         f4
  /reflectances_emit                (n_total, 636)         f4
  /reflectances_uncertainty_hysics  (n_total, 636)         f4
  /reflectances_uncertainty_emit    (n_total, 636)         f4
  /profiles                         (n_total, N_LEVELS)    f4   r_e profile, top→base
  /profiles_raw                     (n_total, max_raw)     f4   same as /profiles for synthetic
  /profiles_raw_z                   (n_total, max_raw)     f4   per-cloud z grid (km)
  /profiles_raw_tau                 (n_total, max_raw)     f4   NaN — synthetic stores tau_c only
  /profile_n_levels                 (n_total,)             i4   = N_LEVELS for every row
  /tau_c                            (n_total,)             f4
  /lwc                              (n_total, N_LEVELS)    f4   *new: synthetic carries it
  /alpha                            (n_total,)             f4   *new: synthetic carries it
  /sza, /saz, /vza, /vaz            (n_total,)             f4   read from changing_variables[0,:4]
  /wv_above_cloud, /wv_in_cloud     (n_total,)             f8   integrated WV columns (molec/cm²)
  /era5_vapor_concentration         (n_total, 37)          f4
  /era5_pressure_levels             (37,)                  f4
  /era5_temperature                 (n_total, 37)          f4   *new: synthetic carries it
  /wavelengths                      (636,)                 f4

For each .mat file there is one synthetic cloud sample, so n_total = n_files
(N_GEOMETRIES = 1).

Notes
-----
- The synthetic generator (06_synthetic_profile_generator.py) saves
  era5.datProfiles.GP_height = NaN. We compute the geopotential-height grid
  on the fly from the stored T(p) profile via the hypsometric equation,
  using the surface (highest p) as the z = 0 reference. That gives us the
  altitude grid needed for the wv_above_cloud / wv_in_cloud integrations.
"""

import re as _re
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MAT_DIR  = Path('/Volumes/My Passport/neural_network_training_data/'
                'synthetic_training_data_5_May_2026/')
OUT_PATH = Path('/Volumes/My Passport/neural_network_training_data/'
                'synthetic_training_data_7-levels_5_May_2026.h5')
FIGURES_DIR = Path(__file__).parent / 'Figures' / 'synthetic_dataset_summary'

N_LEVELS      = 7      # cloud profile target grid (already 7 in the synthetic .mat)
N_GEOMETRIES  = 1      # one (SZA, SAZ, VZA, VAZ) per cloud
N_WAVELENGTHS = 636    # HySICS spectral channels

# Files with tau_c below this are skipped — should already be filtered by 06.
TAU_C_MIN = 0.0        # set non-zero only if you want defense-in-depth

REQUIRED_KEYS = {
    'Refl_model_allStateVectors',
    'changing_variables_allStateVectors',
    're', 'z', 'lwc', 'tau', 'era5',
}

# Noise: applied to the raw reflectances saved by the MATLAB driver, mirroring
# the existing pipeline. The synthetic MATLAB also pre-applied noise but we
# regenerate from the clean Refl_model_allStateVectors so the HDF5 has
# consistent statistics with the original training data.
NOISE_HYSICS = 0.003   # 0.3% Gaussian
NOISE_EMIT   = 0.02    # 2% Gaussian (was 4% in some legacy runs; matches recent)
NOISE_FM     = 0.0     # forward-model noise budget

R_DRY = 287.05         # J/(kg·K), dry-air specific gas constant
G0    = 9.80665        # m/s²


# ============================================================
# Helpers
# ============================================================

def add_gaussian_noise(refl: np.ndarray, noise_level: float) -> tuple:
    """Add Gaussian noise to reflectance. Returns (noisy, uncert), both float32.

    The uncertainty is the per-channel 1-sigma value (= noise_level * |refl|);
    the noisy reflectance is refl + noise_level * refl * randn().
    """
    rnd        = np.random.randn(*refl.shape)
    refl_noisy = refl + (noise_level * refl) * rnd
    uncert     = np.abs(noise_level * refl)
    return refl_noisy.astype(np.float32), uncert.astype(np.float32)


def _unwrap_struct(x):
    """0-d ndarray → inner scalar/struct. Pass through structured arrays."""
    while isinstance(x, np.ndarray) and x.dtype.names is None and x.size == 1:
        x = x.item()
    return x


def _struct_field(s, name):
    """Field access that works on both numpy structured arrays and mat_struct."""
    if hasattr(s, 'dtype') and s.dtype.names is not None and name in s.dtype.names:
        return s[name]
    return getattr(s, name)


def _flat_numerical_field(s, name):
    """Get a numerical-array field from a MATLAB struct as a flat float64 array.

    Necessary because scipy.io.loadmat wraps nested-struct fields inside
    0-d object ndarrays (e.g. the (37,) T profile arrives as
    `array(array([...]), dtype=object)` and np.asarray with a float dtype
    chokes on that double wrapping). We peel object-array wrappers until we
    hit the actual numerical payload, then flatten.
    """
    val = _struct_field(s, name)
    while isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
        val = val.item()
    return np.asarray(val, dtype=np.float64).flatten()


def load_synthetic_era5(d):
    """Pull (T, p, vapor_concentration) profiles from the synthetic era5 struct.

    Each is length 37 on the standard ERA5 pressure grid.
    """
    era5 = _unwrap_struct(d['era5'])
    dp   = _unwrap_struct(_struct_field(era5, 'datProfiles'))
    T    = _flat_numerical_field(dp, 'T')
    p    = _flat_numerical_field(dp, 'p')
    vap  = _flat_numerical_field(dp, 'vapor_concentration')
    if not (len(T) == len(p) == len(vap)):
        raise ValueError(f'era5 profiles disagree in length: T={len(T)}, '
                         f'p={len(p)}, vap={len(vap)}')
    return T, p, vap


def hypsometric_gph(T_K: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Geopotential height (m) from T(p), hypsometric integration.

    Reference: surface (highest pressure) at z = 0. Layers are integrated
    bottom-up using the layer-mean temperature. Virtual-temperature
    correction is omitted because the synthetic .mat does not store q
    (specific humidity); the resulting altitude error is < 1% across the
    troposphere, which is well within the precision needed to bracket the
    cloud for the WV-column integrals.

        dz = (R_d / g) · T̄ · ln(p_lower / p_upper)
    """
    p = np.asarray(p_hPa, dtype=np.float64)
    T = np.asarray(T_K,   dtype=np.float64)
    order = np.argsort(p)[::-1]              # descending p → ascending z
    p_s, T_s = p[order], T[order]
    z_s = np.zeros_like(p_s)
    for i in range(1, len(p_s)):
        T_bar = 0.5 * (T_s[i - 1] + T_s[i])
        z_s[i] = z_s[i - 1] + (R_DRY / G0) * T_bar * np.log(p_s[i - 1] / p_s[i])
    z = np.empty_like(z_s)
    z[order] = z_s
    return z


def extract_synthetic_era5_vapor(d, z_cloud_km):
    """Compute (wv_above_cloud, wv_in_cloud, vap_profile) for a synthetic file.

    Mirrors extract_era5_vapor in the original converter, but builds the
    geopotential-height grid from T(p) since synthetic .mat files store
    GP_height as NaN.
    """
    T, p, vap = load_synthetic_era5(d)
    gph = hypsometric_gph(T, p)            # metres, ascending with descending p

    z_top_m  = float(np.max(z_cloud_km)) * 1000.0
    z_base_m = float(np.min(z_cloud_km)) * 1000.0
    z_top_m  = np.clip(z_top_m,  gph.min(), gph.max())
    z_base_m = np.clip(z_base_m, gph.min(), z_top_m)

    # gph and vap need to be sorted ascending in z for np.interp + np.trapezoid.
    o = np.argsort(gph)
    gph_s = gph[o]
    vap_s = vap[o]

    vap_top  = float(np.interp(z_top_m,  gph_s, vap_s))
    vap_base = float(np.interp(z_base_m, gph_s, vap_s))

    above_mask = gph_s > z_top_m
    gph_above  = np.concatenate([[z_top_m],   gph_s[above_mask]])
    vap_above  = np.concatenate([[vap_top],   vap_s[above_mask]])
    wv_above   = float(np.trapezoid(vap_above, gph_above * 100.0))   # molec/cm²

    in_mask = (gph_s > z_base_m) & (gph_s < z_top_m)
    gph_in  = np.concatenate([[z_base_m],  gph_s[in_mask],  [z_top_m]])
    vap_in  = np.concatenate([[vap_base],  vap_s[in_mask],  [vap_top]])
    si      = np.argsort(gph_in)
    wv_in   = float(np.trapezoid(vap_in[si], gph_in[si] * 100.0))

    return wv_above, wv_in, vap.astype(np.float32), T.astype(np.float32)


def read_cell(d, key):
    """re/lwc/z/tau are saved as 1×1 cells in the synthetic .mat. Unwrap."""
    val = d[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
        val = val.item()
    return np.atleast_1d(np.asarray(val, dtype=np.float64)).flatten()


# ============================================================
# Pass 1: enumerate, validate, collect global bounds
# ============================================================

mat_files = sorted(p for p in MAT_DIR.glob('*.mat') if not p.name.startswith('._'))
n_files_found = len(mat_files)
print(f'Found {n_files_found} .mat files in {MAT_DIR}')

re_global_min, re_global_max   = np.inf, -np.inf
tau_global_min, tau_global_max = np.inf, -np.inf
max_raw_levels = 0
tau_c_all      = []
valid_mat_files = []
skipped_files   = []

print('\nPass 1: validating files and collecting global bounds...')
for i, path in enumerate(mat_files):
    d = scipy.io.loadmat(path, squeeze_me=True)

    missing = REQUIRED_KEYS - set(d.keys())
    if missing:
        skipped_files.append((path, f'missing keys: {missing}'))
        continue

    re_arr  = read_cell(d, 're')
    tau_arr = read_cell(d, 'tau')
    tau_c   = float(tau_arr.max())

    if tau_c < TAU_C_MIN:
        skipped_files.append((path, f'tau_c={tau_c:.4f} < {TAU_C_MIN}'))
        continue

    re_global_min  = min(re_global_min,  float(re_arr.min()))
    re_global_max  = max(re_global_max,  float(re_arr.max()))
    tau_global_min = min(tau_global_min, tau_c)
    tau_global_max = max(tau_global_max, tau_c)
    max_raw_levels = max(max_raw_levels, len(re_arr))
    tau_c_all.append(tau_c)
    valid_mat_files.append(path)

    if (i + 1) % 500 == 0 or i == n_files_found - 1:
        print(f'  {i + 1}/{n_files_found}', end='\r')
print()

if skipped_files:
    print(f'\nSkipped {len(skipped_files)} file(s):')
    for p, why in skipped_files[:20]:
        print(f'  {p.name}: {why}')
    if len(skipped_files) > 20:
        print(f'  ... and {len(skipped_files) - 20} more')

n_files = len(valid_mat_files)
n_total = n_files * N_GEOMETRIES
print(f'\nValid files: {n_files} → {n_total} total training samples '
      f'({N_GEOMETRIES} geometry per file)')
print(f'r_e bounds:   [{re_global_min:.2f}, {re_global_max:.2f}] μm')
print(f'tau_c bounds: [{tau_global_min:.2f}, {tau_global_max:.2f}]')
print(f'max profile levels: {max_raw_levels}  (should equal N_LEVELS={N_LEVELS})')


# ============================================================
# tau_c histogram (sanity)
# ============================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
tau_c_all_arr = np.array(tau_c_all)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(tau_c_all_arr, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
ax.set_xlabel(r'$\tau_c$', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Synthetic τ_c across {n_files} clouds')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_tau_c.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Reference reads from the first valid file (pressure grid + wavelengths)
# ============================================================

d_ref = scipy.io.loadmat(valid_mat_files[0], squeeze_me=True)
_, p_ref, _ = load_synthetic_era5(d_ref)
era5_pressure_levels = p_ref.astype(np.float32)
N_ERA5_LEVELS = len(era5_pressure_levels)
print(f'\nERA5 pressure levels: {N_ERA5_LEVELS} levels '
      f'({era5_pressure_levels[0]:.0f}–{era5_pressure_levels[-1]:.0f} hPa)')

# Wavelengths: each row of changing_variables corresponds to one spectral
# channel for the synthetic data. Columns at -3 / -2 are wl_lo / wl_hi.
cv_ref = d_ref['changing_variables_allStateVectors']
if cv_ref.ndim == 1:
    cv_ref = cv_ref.reshape(1, -1)
wavelengths = ((cv_ref[:N_WAVELENGTHS, -3] + cv_ref[:N_WAVELENGTHS, -2]) / 2
               ).astype(np.float32)
assert len(wavelengths) == N_WAVELENGTHS, (
    f'Expected {N_WAVELENGTHS} bands, got {len(wavelengths)}'
)
print(f'Wavelength grid: {N_WAVELENGTHS} bands, '
      f'{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} nm')


# ============================================================
# Pass 2: write HDF5
# ============================================================

print(f'\nPass 2: writing HDF5 → {OUT_PATH}')

wv_above_all, wv_in_all, era5_vap_all = [], [], []

with h5py.File(OUT_PATH, 'w') as f:
    # --- Reflectance datasets ---
    ds_refl_hysics   = f.create_dataset('reflectances_hysics',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_refl_emit     = f.create_dataset('reflectances_emit',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_uncert_hysics = f.create_dataset('reflectances_uncertainty_hysics',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')
    ds_uncert_emit   = f.create_dataset('reflectances_uncertainty_emit',
                                        shape=(n_total, N_WAVELENGTHS), dtype='f4')

    # --- Cloud profile datasets ---
    ds_prof = f.create_dataset('profiles',
                               shape=(n_total, N_LEVELS), dtype='f4')

    # /profiles_raw, /profiles_raw_z, /profiles_raw_tau retain the same shape as
    # the existing converter so downstream code (data.py) doesn't have to
    # branch on dataset shape. For synthetic data /profiles_raw is identical to
    # /profiles, /profiles_raw_z carries the per-cloud z grid (km), and
    # /profiles_raw_tau is filled with NaN since synthetic .mat files store
    # tau_c as a scalar rather than a τ(z) profile.
    raw_fill = np.full((n_total, max_raw_levels), np.nan, dtype=np.float32)
    ds_prof_raw = f.create_dataset('profiles_raw',     data=raw_fill)
    ds_z_raw    = f.create_dataset('profiles_raw_z',   data=raw_fill.copy())
    ds_tau_raw  = f.create_dataset('profiles_raw_tau', data=raw_fill.copy())
    del raw_fill
    ds_n_levels = f.create_dataset('profile_n_levels', shape=(n_total,), dtype='i4')

    # New per-cloud datasets the synthetic dataset carries explicitly
    ds_lwc     = f.create_dataset('lwc',     shape=(n_total, N_LEVELS), dtype='f4')
    ds_alpha   = f.create_dataset('alpha',   shape=(n_total,),          dtype='f4')

    ds_tau  = f.create_dataset('tau_c', shape=(n_total,), dtype='f4')
    ds_vza  = f.create_dataset('vza',   shape=(n_total,), dtype='f4')
    ds_vaz  = f.create_dataset('vaz',   shape=(n_total,), dtype='f4')
    ds_saz  = f.create_dataset('saz',   shape=(n_total,), dtype='f4')
    ds_sza  = f.create_dataset('sza',   shape=(n_total,), dtype='f4')

    # --- ERA5 datasets ---
    ds_wv_above = f.create_dataset('wv_above_cloud',  shape=(n_total,), dtype='f8')
    ds_wv_in    = f.create_dataset('wv_in_cloud',     shape=(n_total,), dtype='f8')
    ds_era5_vap = f.create_dataset('era5_vapor_concentration',
                                   shape=(n_total, N_ERA5_LEVELS), dtype='f4')
    ds_era5_T   = f.create_dataset('era5_temperature',
                                   shape=(n_total, N_ERA5_LEVELS), dtype='f4')
    f.create_dataset('era5_pressure_levels', data=era5_pressure_levels)

    for i, path in enumerate(valid_mat_files):
        d = scipy.io.loadmat(path, squeeze_me=True)

        # --- Reflectances: synthetic shape is (636, 1) or (636,) ----------
        refl_raw = np.asarray(d['Refl_model_allStateVectors'], dtype=np.float64)
        refl_raw = np.atleast_2d(refl_raw).reshape(N_WAVELENGTHS, -1)
        if refl_raw.shape[1] != N_GEOMETRIES:
            raise ValueError(
                f'{path.name}: Refl_model_allStateVectors has shape '
                f'{refl_raw.shape}, expected (636, {N_GEOMETRIES})'
            )

        refl_hysics, uncert_hysics = add_gaussian_noise(refl_raw, NOISE_HYSICS + NOISE_FM)
        refl_emit,   uncert_emit   = add_gaussian_noise(refl_raw, NOISE_EMIT   + NOISE_FM)

        # (636, 1) → (1, 636)
        refl_hysics   = refl_hysics.T
        refl_emit     = refl_emit.T
        uncert_hysics = uncert_hysics.T
        uncert_emit   = uncert_emit.T

        # --- Profiles ----------------------------------------------------
        re_um  = read_cell(d, 're')
        z_km   = read_cell(d, 'z')
        lwc_g  = read_cell(d, 'lwc')
        tau_c  = float(read_cell(d, 'tau').max())

        if len(re_um) != N_LEVELS:
            raise ValueError(
                f'{path.name}: profile length {len(re_um)} ≠ N_LEVELS={N_LEVELS}'
            )

        # --- Geometry: synthetic stores one row per spectral channel; all
        # rows in a single .mat share the same (sza, vza, vaz, saz) so any
        # row works. Columns 0..3 are [sza, vza, vaz, saz]. -----------
        cv = np.atleast_2d(d['changing_variables_allStateVectors'])
        sza = float(cv[0, 0])
        vza = float(cv[0, 1])
        vaz = float(cv[0, 2])
        saz = float(cv[0, 3])

        # --- Alpha: per-cloud scalar; the MATLAB function broadcasts it
        # across the 7 levels and saves under inputs.RT.distribution_var. We
        # take the mean so /alpha is always a scalar regardless of how
        # alpha_out was stored. -----------------------------------------
        alpha_scalar = float('nan')
        if 'inputs' in d:
            inputs_struct = _unwrap_struct(d['inputs'])
            try:
                rt_struct = _unwrap_struct(_struct_field(inputs_struct, 'RT'))
                mean_dv = _struct_field(rt_struct, 'mean_distribution_var')
                alpha_scalar = float(np.asarray(mean_dv).flatten()[0])
            except (KeyError, AttributeError):
                pass

        # --- ERA5 -------------------------------------------------------
        wv_above, wv_in, vap_profile, T_profile = extract_synthetic_era5_vapor(d, z_km)

        # --- Write row i ------------------------------------------------
        row = i  # n_total = n_files since N_GEOMETRIES = 1
        ds_refl_hysics[row]   = refl_hysics[0]
        ds_refl_emit[row]     = refl_emit[0]
        ds_uncert_hysics[row] = uncert_hysics[0]
        ds_uncert_emit[row]   = uncert_emit[0]

        ds_prof[row]      = re_um.astype(np.float32)
        ds_lwc[row]       = lwc_g.astype(np.float32)
        ds_tau[row]       = tau_c
        ds_alpha[row]     = alpha_scalar

        # /profiles_raw* — synthetic data has all 7 levels; pad with NaN if
        # max_raw_levels happens to exceed 7 (it shouldn't).
        ds_prof_raw[row, :len(re_um)] = re_um.astype(np.float32)
        ds_z_raw[row,    :len(z_km)]  = z_km.astype(np.float32)
        # ds_tau_raw stays NaN (only tau_c is known per cloud)
        ds_n_levels[row] = len(re_um)

        ds_sza[row] = sza
        ds_vza[row] = vza
        ds_vaz[row] = vaz
        ds_saz[row] = saz

        ds_wv_above[row] = wv_above
        ds_wv_in[row]    = wv_in
        ds_era5_vap[row] = vap_profile
        ds_era5_T[row]   = T_profile

        wv_above_all.append(wv_above)
        wv_in_all.append(wv_in)
        era5_vap_all.append(vap_profile)

        if (i + 1) % 500 == 0 or i == n_files - 1:
            print(f'  {i + 1}/{n_files}  ({row + 1} samples written)')

    f.create_dataset('wavelengths', data=wavelengths)

    f.attrs['n_mat_files']     = n_files
    f.attrs['n_geometries']    = N_GEOMETRIES
    f.attrs['re_global_min']   = re_global_min
    f.attrs['re_global_max']   = re_global_max
    f.attrs['tau_global_min']  = tau_global_min
    f.attrs['tau_global_max']  = tau_global_max
    f.attrs['max_raw_levels']  = max_raw_levels
    f.attrs['source']          = 'synthetic-cloud RT runs'

print(f'\nDone.')
print(f'  Output: {OUT_PATH}')
print(f'  Total samples: {n_total} ({n_files} clouds × {N_GEOMETRIES} geometry)')


# ============================================================
# WV column histograms + bounds summary (mirror existing converter)
# ============================================================

AVOGADRO = 6.02214076e23
M_H2O    = 18.015e-3
MOLEC_CM2_TO_KG_M2 = M_H2O / AVOGADRO * 1e4   # ≈ 2.99e-22

wv_above_arr   = np.array(wv_above_all)
wv_above_kg_m2 = wv_above_arr * MOLEC_CM2_TO_KG_M2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(wv_above_kg_m2, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].set_xlabel(r'Above-cloud WV column  (kg m$^{-2}$)')
axes[0].set_ylabel('Number of clouds')
axes[0].set_title('Above-cloud water vapour — linear scale')
axes[1].hist(np.log10(wv_above_kg_m2), bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel(r'$\log_{10}$ above-cloud WV column  (kg m$^{-2}$)')
axes[1].set_ylabel('Number of clouds')
axes[1].set_title('Above-cloud water vapour — log scale')
plt.suptitle(
    f'ERA5 above-cloud water vapour  ({n_files} synthetic clouds)\n'
    f'Range: {wv_above_kg_m2.min():.2f} – {wv_above_kg_m2.max():.2f} kg m$^{{-2}}$'
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_wv_above_cloud.png', dpi=400, bbox_inches='tight')
plt.close(fig)

wv_in_arr   = np.array(wv_in_all)
wv_in_kg_m2 = wv_in_arr * MOLEC_CM2_TO_KG_M2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(wv_in_kg_m2, bins=30, color='darkorange', edgecolor='white', linewidth=0.5)
axes[0].set_xlabel(r'In-cloud WV column  (kg m$^{-2}$)')
axes[0].set_ylabel('Number of clouds')
axes[0].set_title('In-cloud water vapour — linear scale')
axes[1].hist(np.log10(wv_in_kg_m2), bins=30, color='darkorange', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel(r'$\log_{10}$ in-cloud WV column  (kg m$^{-2}$)')
axes[1].set_ylabel('Number of clouds')
axes[1].set_title('In-cloud water vapour — log scale')
plt.suptitle(
    f'ERA5 in-cloud water vapour  ({n_files} synthetic clouds)\n'
    f'Range: {wv_in_kg_m2.min():.4f} – {wv_in_kg_m2.max():.4f} kg m$^{{-2}}$'
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'distribution_wv_in_cloud.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# ============================================================
# WV bounds summary (for data.py update)
# ============================================================

era5_vap_arr   = np.array(era5_vap_all)
log10_wv_above = np.log10(wv_above_arr)
log10_wv_in    = np.log10(wv_in_arr)

print('\n' + '=' * 60)
print('WATER VAPOUR BOUNDS — update these in data.py if needed')
print('=' * 60)
print(f'  Above-cloud WV column (molec/cm²):')
print(f'    min = {wv_above_arr.min():.4e}   log10 = {log10_wv_above.min():.2f}')
print(f'    max = {wv_above_arr.max():.4e}   log10 = {log10_wv_above.max():.2f}')
print(f'  In-cloud WV column (molec/cm²):')
print(f'    min = {wv_in_arr.min():.4e}   log10 = {log10_wv_in.min():.2f}')
print(f'    max = {wv_in_arr.max():.4e}   log10 = {log10_wv_in.max():.2f}')
print(f'  ERA5 vapor concentration (molec/cm³):')
print(f'    surface max = {era5_vap_arr.max():.4e}   '
      f'log10 = {np.log10(era5_vap_arr.max()):.2f}')

print(f'\n  Suggested bounds for data.py:')
print(f'    WV_ABOVE_LOG10_MIN = {float(np.floor(log10_wv_above.min())):.1f}')
print(f'    WV_ABOVE_LOG10_MAX = {float(np.ceil(log10_wv_above.max())):.1f}')
print(f'    WV_IN_LOG10_MIN    = {float(np.floor(log10_wv_in.min())):.1f}')
print(f'    WV_IN_LOG10_MAX    = {float(np.ceil(log10_wv_in.max())):.1f}')
print(f'    ERA5_VAP_LOG10_MAX = {float(np.ceil(np.log10(era5_vap_arr.max()))):.1f}')
