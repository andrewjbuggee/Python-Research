"""
Build the NetCDF input file for the synthetic-cloud RT training set.

The file written here is read, one cloud per call, by the MATLAB driver
    Matlab-Research/Hyperspectral_Cloud_Retrievals/Neural_Network/
        hysics_refl_from_synthetic_NN_inputs.m
which computes the HySICS TOA reflectance spectrum for that cloud with
libRadtran. The variable names, shapes and orientations below are exactly the
ones that function reads with ncread(); do not rename them without editing it.

Pipeline (run end-to-end with `python 07_build_training_inputs.py [npz]`):

  Cell 1 — Load the synthetic profile bundle produced by
           06_synthetic_profile_generator.py.
  Cell 2 — Read the (lat, lon, UTC time) of the in-situ profiles the generator
           was fitted to. These come from the consolidated in-situ + ERA5 .mat
           file named in the bundle (`input_path`), restricted to the rows the
           generator actually used (`train_source_index`), so the solar-geometry
           pool matches the training population exactly (τ window, outliers
           removed, duplicates dropped).
  Cell 3 — Compute the solar zenith and azimuth angles of every in-situ profile
           with pysolar, and keep those with SZA < SZA_MAX_DEG.
  Cell 4 — For each synthetic cloud:
             • bootstrap one (SZA, SAZ) pair from the filtered empirical set,
             • sample VZA ~ Uniform(VZA_RANGE_DEG),
             • sample VAZ ~ Uniform(VAZ_RANGE_DEG).
  Cell 5 — Check every cloud against the limits the MATLAB driver and
           libRadtran impose (fail loudly here rather than on CURC).
  Cell 6 — Write a NetCDF4 file with one row per cloud.

Changes on 12 Nov 2026
----------------------
* Geometry now comes from the consolidated .mat file rather than the legacy
  per-profile directory on the external drive (no longer needed, and the
  legacy set included profiles the generator now excludes).
* The profile time is the in-situ profile time (`time_utc_hours`), not the
  hour of the nearest ERA5 file, so SZA is that of the actual measurement.
* Output name carries the bundle's tag (N, L, date) so earlier training sets
  are never overwritten; provenance (K_re, K_lwc, seeds, source files) is
  stored as global attributes.
* Solar azimuth converted from pysolar's clockwise-from-north value to
  libRadtran's phi0 convention (0° = sun in the south): phi0 = az − 180 mod 360.
  Earlier versions wrote the pysolar value directly, rotating every scene's
  relative azimuth by 180°. See sza_saz() for the sources.
* `mie_alpha_table` (α of the custom Mie table the generator used for the τ_c
  label) is written as an extra variable. The driver ignores it; it lets a
  later check confirm the driver picked the same table.

Output schema (NetCDF):

  dimensions:
    cloud         : N  (synthetic clouds)
    level_cloud   : 7  (top → base, evenly spaced between z_top and z_base)
    level_atmos   : 37 (ERA5 standard pressure grid, surface → TOA)

  variables (read by the MATLAB driver):
    re_um               (cloud, level_cloud)  effective radius profile (μm)
    lwc_g_per_m3        (cloud, level_cloud)  liquid water content (g/m³)
    z_km                (cloud, level_cloud)  altitude grid (km)
    alpha               (cloud,)              libRadtran gamma shape parameter
                                              (vertical mean; constant per cloud)
    tau_c               (cloud,)              cloud optical depth at 550 nm
    LWP_g_per_m2        (cloud,)              liquid water path (g/m²)
    sza_deg, saz_deg, vza_deg, vaz_deg (cloud,)  geometry (degrees)
    T_K                 (cloud, level_atmos)  ERA5-like temperature (K)
    vapor_molec_per_cm3 (cloud, level_atmos)  water-vapour number density (cm⁻³)
    pressure_hPa        (level_atmos,)        shared pressure grid (hPa)
  extra (not read by the driver):
    mie_alpha_table     (cloud,)              α of the Mie table behind tau_c
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import scipy.io

# pysolar is a pure-Python solar-position package (already in the environment)
from pysolar import solar


# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

# Default bundle; override with the first command-line argument.
SYNTHETIC_NPZ = (SCRIPT_DIR / 'synthetic_profiles' /
                 'synthetic_profiles_jointMVN_N100000_L7_2026-11-12.npz')

OUT_DIR = SCRIPT_DIR / 'training_inputs'
SZA_MAX_DEG = 50.0                 # draft §III.D: retrievals degrade above 50°
VZA_RANGE_DEG = (0.0, 45.0)        # uniform
VAZ_RANGE_DEG = (0.0, 180.0)       # uniform
RANDOM_SEED = 0                    # geometry sampling; written to the .nc

# Limits enforced by the MATLAB driver / libRadtran set-up (Cell 5).
RE_MIN_UM = 0.1      # driver drops levels with r_e <= 0.1 μm
RE_MAX_UM = 50.0     # driver errors for r_e >= 50 μm (custom Mie tables span 1–50 μm)
ALPHA_RANGE = (1.0, 150.0)   # custom Mie tables exist for α = 1 … 150


def _versioned_path(path: Path) -> Path:
    """Append '_ver2', '_ver3', ... to the stem if `path` already exists."""
    if not path.exists():
        return path
    ver = 2
    while True:
        candidate = path.with_name(f'{path.stem}_ver{ver}{path.suffix}')
        if not candidate.exists():
            return candidate
        ver += 1


def bundle_tag(npz_path: Path) -> str:
    """'synthetic_profiles_jointMVN_N100000_L7_2026-11-12' → 'N100000_L7_2026-11-12'."""
    stem = npz_path.stem
    return stem.split('jointMVN_', 1)[1] if 'jointMVN_' in stem else stem


# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Load synthetic cloud bundle
# ─────────────────────────────────────────────────────────────────────────────
npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SYNTHETIC_NPZ
print(f'\n[Cell 1] Loading synthetic profiles from {npz_path}')
syn = np.load(npz_path)
N_CLOUDS = syn['re'].shape[0]
N_LEVELS = int(syn['N_FIXED_LEVELS'])
N_ERA5 = int(syn['N_ERA5'])
print(f'         N_CLOUDS = {N_CLOUDS}, levels = {N_LEVELS}, ERA5 levels = {N_ERA5}')
print(f'         K_re = {int(syn["K_re"])}, K_lwc = {int(syn["K_lwc"])}, '
      f'K_T = {int(syn["K_T"])}, K_vapor = {int(syn["K_vapor"])}')
print(f'         tau_c range : [{syn["tau_c"].min():.2f}, {syn["tau_c"].max():.2f}]')
print(f'         LWP range   : [{syn["LWP_g_per_m2"].min():.1f}, '
      f'{syn["LWP_g_per_m2"].max():.1f}] g/m²')

# α is stored per level by the generator but is constant within a cloud (one
# draw per cloud). The driver reads one α per cloud, so write the vertical
# mean, and confirm nothing is lost by doing so.
alpha_per_cloud = np.asarray(syn['mean_alpha'], dtype=np.float64)       # (N,)
if not np.allclose(syn['alpha'], syn['alpha'][:, :1]):
    raise SystemExit('alpha varies within a cloud; the driver reads a single '
                     'alpha per cloud. Update the driver before using this bundle.')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — (lat, lon, UTC time) of the in-situ profiles behind the generator
# ─────────────────────────────────────────────────────────────────────────────
insitu_path = Path(str(syn['input_path']))
rows = np.asarray(syn['train_source_index'], dtype=int)    # rows the generator used
print(f'\n[Cell 2] Reading in-situ geolocation from {insitu_path.name}')
print(f'         using the {rows.size} rows the generator was fitted to')
mat = scipy.io.loadmat(insitu_path, squeeze_me=True)

lats = np.asarray(mat['latitude'], dtype=np.float64)[rows]         # degrees N
lons = np.asarray(mat['longitude'], dtype=np.float64)[rows]        # degrees E
flight_dates = np.asarray(mat['date_of_flight'], dtype=object)[rows]   # 'YYYY-MM-DD'
time_utc_h = np.asarray(mat['time_utc_hours'], dtype=np.float64)[rows]  # hours UTC
campaigns = np.asarray(mat['campaign_each'], dtype=object)[rows]

# The profile time is decimal hours after 00 UTC on the flight date. Values
# ≥ 24 h (a flight that crossed midnight UTC) roll into the next day through
# the timedelta, so no special case is needed.
times = [datetime.strptime(str(d), '%Y-%m-%d').replace(tzinfo=timezone.utc)
         + timedelta(hours=float(h))
         for d, h in zip(flight_dates, time_utc_h)]

bad = ~(np.isfinite(lats) & np.isfinite(lons) & np.isfinite(time_utc_h))
if bad.any():
    raise SystemExit(f'{bad.sum()} training rows have missing lat/lon/time.')
print(f'         ORACLES = {(campaigns == "oracles").sum()}, '
      f'VOCALS-REx = {(campaigns == "vocals-rex").sum()}')
print(f'         lat range : [{lats.min():.2f}, {lats.max():.2f}]')
print(f'         lon range : [{lons.min():.2f}, {lons.max():.2f}]')
print(f'         time span : {min(times)}  →  {max(times)}')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — Solar geometry of each in-situ profile; keep SZA < SZA_MAX_DEG
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[Cell 3] Computing solar geometry (pysolar) and filtering SZA < {SZA_MAX_DEG}°')


def sza_saz(lat_deg: float, lon_deg: float, dt_utc: datetime) -> tuple[float, float]:
    """Solar zenith angle and solar azimuth in the libRadtran convention (degrees).

    Zenith: pysolar returns the solar altitude above the horizon, so
    SZA = 90° − altitude.

    Azimuth: the value returned here is written to the .nc as `saz_deg`, and
    the MATLAB driver passes it unchanged to write_INP_file.m as libRadtran's
    `phi0`. The two conventions differ by 180°:
      * pysolar get_azimuth: degrees clockwise from north (sun due south =
        180°). Verified numerically, not from pysolar's docs: Boulder (40.0 N,
        105.25 W) on 21 Dec 2025 gives 180.2° at local noon, 138° at 09 local
        and 222° at 15 local.
      * libRadtran phi0: "Sun in the South: 0 degrees, Sun in the West: 90,
        Sun in the North: 180, Sun in the East: 270" (libRadtran 2.0.6 user
        manual, §6.1, option phi0).
    Hence phi0 = (azimuth_from_north − 180) mod 360, the same conversion used by
    LibRadTran-wrapper/solar_position_from_lat_lon_time.m for the in-situ
    spectra. Until 12 Nov 2026 this function returned the pysolar value
    unchanged, which rotated every scene's relative azimuth (phi − phi0) by 180°.

    The viewing azimuth `vaz_deg` needs no conversion: it is sampled directly
    as libRadtran's `phi` (sensor position, "Sensor in the North (looking
    South): 0 deg", same manual section, option phi). For 1-D solvers only
    phi − phi0 matters (manual, options phi and phi0).
    """
    alt = solar.get_altitude(lat_deg, lon_deg, dt_utc)
    az_from_north = solar.get_azimuth(lat_deg, lon_deg, dt_utc)
    phi0 = (az_from_north - 180.0) % 360.0
    return 90.0 - alt, phi0


geom = np.array([sza_saz(la, lo, t) for la, lo, t in zip(lats, lons, times)])
sza_all, saz_all = geom[:, 0], geom[:, 1]

mask = sza_all < SZA_MAX_DEG
sza_pool = sza_all[mask]
saz_pool = saz_all[mask]
print(f'         all profiles : SZA ∈ [{sza_all.min():.1f}, {sza_all.max():.1f}], '
      f'median = {np.median(sza_all):.1f}')
print(f'         SZA < {SZA_MAX_DEG}°: {mask.sum()} of {len(sza_all)} '
      f'({100 * mask.sum() / len(sza_all):.1f}%)')
if mask.sum() == 0:
    raise RuntimeError(
        f'No in-situ profiles satisfy SZA < {SZA_MAX_DEG}°. If every SZA is near '
        '90°, the times are probably being parsed as midnight UTC.')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Pair each synthetic cloud with sampled (SZA, SAZ, VZA, VAZ)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[Cell 4] Sampling geometry per synthetic cloud (N = {N_CLOUDS}, '
      f'seed = {RANDOM_SEED})')
rng = np.random.default_rng(RANDOM_SEED)

idx_pool = rng.integers(0, len(sza_pool), size=N_CLOUDS)   # bootstrap (SZA, SAZ) pairs
sza_per_cloud = sza_pool[idx_pool]
saz_per_cloud = saz_pool[idx_pool]
vza_per_cloud = rng.uniform(*VZA_RANGE_DEG, size=N_CLOUDS)
vaz_per_cloud = rng.uniform(*VAZ_RANGE_DEG, size=N_CLOUDS)

for name, v in (('SZA', sza_per_cloud), ('SAZ', saz_per_cloud),
                ('VZA', vza_per_cloud), ('VAZ', vaz_per_cloud)):
    print(f'         {name} per cloud : [{v.min():.1f}, {v.max():.1f}]')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Check every cloud against what the MATLAB driver can run
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Cell 5] Checking clouds against driver / libRadtran limits')
re_um = np.asarray(syn['re'], dtype=np.float64)          # (N, L) top → base
lwc = np.asarray(syn['lwc'], dtype=np.float64)           # (N, L) g/m³
z_km = np.asarray(syn['z'], dtype=np.float64)            # (N, L) km, top → base

hard_fail = {
    'non-finite value in re/lwc/z/T/vapor':
        ~(np.isfinite(re_um).all(1) & np.isfinite(lwc).all(1) & np.isfinite(z_km).all(1)
          & np.isfinite(syn['T']).all(1) & np.isfinite(syn['vapor']).all(1)),
    f'r_e >= {RE_MAX_UM} μm (driver raises an error)': (re_um >= RE_MAX_UM).any(1),
    'z not strictly decreasing top → base': ~(np.diff(z_km, axis=1) < 0).all(1),
    f'alpha outside the Mie-table range {ALPHA_RANGE}':
        (alpha_per_cloud < ALPHA_RANGE[0]) | (alpha_per_cloud > ALPHA_RANGE[1]),
}
warn = {
    f'r_e <= {RE_MIN_UM} μm at some level (driver deletes that level)':
        (re_um <= RE_MIN_UM).any(1),
    'LWC <= 0 at some level (zero-water layer inside the cloud)': (lwc <= 0).any(1),
}
n_fail = 0
for label, m in hard_fail.items():
    print(f'         {"FAIL" if m.any() else "ok  "}  {label}: {m.sum()} clouds')
    n_fail += int(m.sum())
for label, m in warn.items():
    flag = 'WARN' if m.any() else 'ok  '
    ids = np.flatnonzero(m)[:10] + 1                      # 1-based, as MATLAB idx
    extra = f'  (MATLAB idx: {ids.tolist()}{" …" if m.sum() > 10 else ""})' if m.any() else ''
    print(f'         {flag}  {label}: {m.sum()} clouds{extra}')
if n_fail:
    raise SystemExit('Hard check(s) failed; not writing the NetCDF file.')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Save to NetCDF4 (single file, one row per cloud)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NC = _versioned_path(OUT_DIR / f'training_inputs_jointMVN_{bundle_tag(npz_path)}.nc')
print(f'\n[Cell 6] Writing NetCDF → {OUT_NC}')

with nc.Dataset(OUT_NC, 'w', format='NETCDF4') as ds:
    # The driver reads the 'title' attribute, so keep it.
    ds.title = 'Synthetic in-situ cloud profiles + paired solar/viewing geometry'
    ds.created_by = '07_build_training_inputs.py'
    ds.date_created = datetime.now(timezone.utc).isoformat(timespec='seconds')
    ds.source_npz = npz_path.name
    ds.source_insitu_mat = insitu_path.name
    ds.n_insitu_profiles_geometry_pool = int(mask.sum())
    ds.K_re = int(syn['K_re'])
    ds.K_lwc = int(syn['K_lwc'])
    ds.K_T = int(syn['K_T'])
    ds.K_vapor = int(syn['K_vapor'])
    ds.tau_c_method = str(syn['tau_c_method'])
    ds.tau_wavelength_um = float(syn['tau_wavelength_um'])
    ds.geometry_random_seed = RANDOM_SEED
    ds.sza_filter = (f'(SZA, SAZ) bootstrapped from the generator\'s in-situ profiles '
                     f'with SZA < {SZA_MAX_DEG}°')
    ds.azimuth_convention = ('saz_deg = libRadtran phi0 (sun azimuth from north - 180); '
                             'vaz_deg = libRadtran phi (sensor azimuth from north); '
                             'see libRadtran 2.0.6 manual sec. 6.1, phi0 and phi')
    ds.cloud_orientation = 'index 0 = cloud top, index -1 = cloud base'
    ds.atmos_orientation = 'surface → TOA on the standard ERA5 pressure grid'

    ds.createDimension('cloud', N_CLOUDS)
    ds.createDimension('level_cloud', N_LEVELS)
    ds.createDimension('level_atmos', N_ERA5)

    def _v(name, dims, data, units, long_name, dtype='f4'):
        v = ds.createVariable(name, dtype, dims, zlib=True, complevel=4)
        v[:] = np.asarray(data, dtype=np.dtype(dtype))
        v.units = units
        v.long_name = long_name
        return v

    _v('re_um', ('cloud', 'level_cloud'), re_um,
       'um', 'Effective radius profile (cloud top → base)')
    _v('lwc_g_per_m3', ('cloud', 'level_cloud'), lwc,
       'g m-3', 'Liquid water content profile (cloud top → base)')
    _v('z_km', ('cloud', 'level_cloud'), z_km,
       'km', 'Altitude grid for cloud profile (cloud top → base)')
    _v('alpha', ('cloud',), alpha_per_cloud,
       '1', 'libRadtran gamma-distribution shape parameter (vertical mean)')
    _v('tau_c', ('cloud',), syn['tau_c'],
       '1', f'Cloud optical depth at {float(syn["tau_wavelength_um"])} um '
            f'({str(syn["tau_c_method"])})')
    _v('LWP_g_per_m2', ('cloud',), syn['LWP_g_per_m2'],
       'g m-2', 'Liquid water path (derived from lwc)')

    _v('sza_deg', ('cloud',), sza_per_cloud,
       'degree', f'Solar zenith angle (bootstrapped from in-situ SZA < {SZA_MAX_DEG})')
    _v('saz_deg', ('cloud',), saz_per_cloud,
       'degree', 'Solar azimuth angle, libRadtran phi0 convention '
                 '(0 = sun in the south, 90 = west; = azimuth from north - 180)')
    _v('vza_deg', ('cloud',), vza_per_cloud,
       'degree', f'Viewing zenith angle (Uniform{list(VZA_RANGE_DEG)})')
    _v('vaz_deg', ('cloud',), vaz_per_cloud,
       'degree', f'Viewing azimuth angle, libRadtran phi convention '
                 f'(0 = sensor in the north; Uniform{list(VAZ_RANGE_DEG)})')

    _v('T_K', ('cloud', 'level_atmos'), syn['T'],
       'K', 'Temperature profile (surface → TOA)')
    _v('vapor_molec_per_cm3', ('cloud', 'level_atmos'), syn['vapor'],
       'cm-3', 'Water vapour number density (surface → TOA)')
    _v('pressure_hPa', ('level_atmos',), syn['era5_pressure'],
       'hPa', 'Shared ERA5 pressure grid')

    _v('mie_alpha_table', ('cloud',), syn['mie_alpha_table'],
       '1', 'alpha of the custom Mie table used for the tau_c label '
            '(not read by the MATLAB driver)', dtype='i4')

print(f'         file size: {OUT_NC.stat().st_size / 1e6:.1f} MB')
print('Done.')
