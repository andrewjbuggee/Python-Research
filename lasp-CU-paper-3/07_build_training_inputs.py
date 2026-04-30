"""
Build a NetCDF input file for the synthetic-cloud RT training set.

Pipeline (organized as discrete cells; run end-to-end with `python 07_build_training_inputs.py`):

  Cell 1 — Load the synthetic profile bundle produced by 06_synthetic_profile_generator.
  Cell 2 — Walk ORACLES .mat files; pull per-profile (lat, lon, UTC datetime) from
           era5.geo.latitude, era5.geo.longitude, era5.metadata.era5_utcTime.
  Cell 3 — Compute the corresponding solar zenith and azimuth angles via pysolar.
           Filter the empirical distribution to SZA < 50°.
  Cell 4 — For each synthetic cloud:
             • bootstrap one (SZA, SAZ) pair from the filtered empirical set,
             • sample VZA ~ Uniform(0, 45),
             • sample VAZ ~ Uniform(0, 180).
  Cell 5 — Write a NetCDF4 file with one row per cloud. Variables match the names the
           MATLAB driver reads via ncread.

Output schema (NetCDF):

  dimensions:
    cloud         : N  (synthetic clouds)
    level_cloud   : 7  (top → base, evenly spaced between z_top and z_base)
    level_atmos   : 37 (ERA5 standard pressure grid, surface → TOA)

  variables:
    re_um            (cloud, level_cloud)   effective radius profile (μm)
    lwc_g_per_m3     (cloud, level_cloud)   liquid water content (g/m³)
    z_km             (cloud, level_cloud)   altitude grid (km)
    alpha            (cloud,)               libRadtran shape parameter
    tau_c            (cloud,)               cloud optical depth
    LWP_g_per_m2     (cloud,)               liquid water path
    sza_deg          (cloud,)               solar zenith angle
    saz_deg          (cloud,)               solar azimuth angle
    vza_deg          (cloud,)               viewing zenith angle (Uniform[0,45])
    vaz_deg          (cloud,)               viewing azimuth angle (Uniform[0,180])
    T_K              (cloud, level_atmos)   ERA5 temperature
    vapor_molec_per_cm3 (cloud, level_atmos) ERA5 water vapor number density
    pressure_hPa     (level_atmos,)         shared ERA5 pressure grid

The `.nc` is what the MATLAB function reads via ncread(input_file, var, [start..], [count..]).
"""

import re as _re
import numpy as np
import scipy.io
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timezone

# pysolar is a pure-Python solar geometry package
from pysolar import solar


# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).parent
SYNTHETIC_NPZ     = SCRIPT_DIR / 'synthetic_profiles' / 'synthetic_profiles_jointMVN_N300000_L7.npz'
MAT_DIR           = Path('/Volumes/My Passport/neural_network_training_data/saz0_allProfiles/')

OUT_DIR           = SCRIPT_DIR / 'training_inputs'
SZA_MAX_DEG       = 50.0
VZA_RANGE_DEG     = (0.0, 45.0)        # uniform
VAZ_RANGE_DEG     = (0.0, 180.0)       # uniform
RANDOM_SEED       = 0


# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Load synthetic cloud bundle
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[Cell 1] Loading synthetic profiles from {SYNTHETIC_NPZ}')
syn = np.load(SYNTHETIC_NPZ)
N_CLOUDS = syn['re'].shape[0]
print(f'         N_CLOUDS = {N_CLOUDS}')
print(f'         re shape = {syn["re"].shape}, T shape = {syn["T"].shape}')
print(f'         tau_c range : [{syn["tau_c"].min():.2f}, {syn["tau_c"].max():.2f}]')
print(f'         LWP range   : [{syn["LWP_g_per_m2"].min():.1f}, {syn["LWP_g_per_m2"].max():.1f}] g/m²')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Walk ORACLES .mat files; extract (lat, lon, UTC datetime)
# ─────────────────────────────────────────────────────────────────────────────
# MATLAB structs survive scipy.io.loadmat as either numpy void/structured arrays
# or scipy.io.matlab.mat_struct instances depending on the loadmat call. These
# helpers smooth over that.
def _struct_fields(s):
    if s is None:
        return ()
    if hasattr(s, 'dtype') and s.dtype is not None and s.dtype.names is not None:
        return s.dtype.names
    if hasattr(s, '_fieldnames'):
        return tuple(s._fieldnames)
    return ()


def _struct_get(s, name):
    if s is None:
        return None
    try:
        return s[name]
    except Exception:
        return getattr(s, name, None)


def _unwrap(x):
    """Recursively unwrap 0-d ndarrays / 1-element wrappers down to the inner value."""
    while isinstance(x, np.ndarray):
        if x.dtype.names is not None:
            # structured array: stop here, callers will field-access
            return x
        if x.size == 1:
            x = x.item()
            continue
        return x
    return x


def _scalar(arr):
    if arr is None:
        return None
    v = _unwrap(arr)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        return v.flatten()[0]
    return v


def _to_str(v):
    v = _unwrap(v)
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        return str(v.flatten()[0])
    return None


# Filename pattern: era5_2018_10_02_1000UTC.nc  →  YYYY, MM, DD, HHMM
_ERA5_FNAME_RE = _re.compile(
    r'era5[_-](\d{4})[_-](\d{2})[_-](\d{2})[_-]?(\d{2})(\d{2})\s*UTC',
    _re.IGNORECASE,
)


def _parse_era5_filename(fname):
    """Parse era5 filename (e.g. 'era5_2018_10_02_1000UTC.nc') → datetime UTC."""
    if not fname:
        return None
    m = _ERA5_FNAME_RE.search(fname)
    if not m:
        return None
    y, mo, d, hh, mm = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_start_ymd(meta):
    """Fallback: build a datetime from start_year/start_month/start_day cells."""
    try:
        y  = int(_scalar(_struct_get(meta, 'start_year')))
        mo = int(_scalar(_struct_get(meta, 'start_month')))
        d  = int(_scalar(_struct_get(meta, 'start_day')))
        return datetime(y, mo, d, 12, tzinfo=timezone.utc)   # noon UTC default
    except Exception:
        return None


def extract_lat_lon_dt(d, debug=False):
    """Pull (lat, lon, datetime_utc) from the era5 substruct. Returns None on miss.

    Datetime priority:
      1. parse era5.metadata.era5_filename (most reliable; embeds Y M D H MM)
      2. era5.metadata.start_year/_month/_day cells (no time-of-day → noon)
    """
    if 'era5' not in d:
        return None
    era5_struct = _unwrap(d['era5'])

    geo  = _unwrap(_struct_get(era5_struct, 'geo'))
    meta = _unwrap(_struct_get(era5_struct, 'metadata'))

    if debug:
        print(f'    era5 fields  : {_struct_fields(era5_struct)}')
        print(f'    geo  fields  : {_struct_fields(geo)}')
        print(f'    meta fields  : {_struct_fields(meta)}')
        ef = _to_str(_struct_get(meta, 'era5_filename'))
        print(f'    era5_filename: {ef!r}')

    # MATLAB convention: capitalized
    lat = _scalar(_struct_get(geo, 'Latitude')  or _struct_get(geo, 'latitude'))
    lon = _scalar(_struct_get(geo, 'Longitude') or _struct_get(geo, 'longitude'))

    fname = _to_str(_struct_get(meta, 'era5_filename'))
    dt    = _parse_era5_filename(fname)
    if dt is None:
        dt = _parse_start_ymd(meta)

    if lat is None or lon is None or dt is None:
        return None
    return float(lat), float(lon), dt


print(f'\n[Cell 2] Walking ORACLES .mat files in {MAT_DIR}')
mat_files = sorted(f for f in MAT_DIR.glob('*.mat') if not f.name.startswith('._'))
print(f'         found {len(mat_files)} files')

geo_records = []   # list of (lat, lon, datetime_utc, source_path)
seen_fingerprints = set()
n_ok = n_skip_no_alpha = n_skip_no_geo = n_skip_dup = 0
debugged_one = False

for path in mat_files:
    d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if 'alpha_param' not in d.keys():           # ORACLES filter (same as 06)
        n_skip_no_alpha += 1
        continue

    re_arr = np.asarray(d.get('re', np.array([0]))[()], dtype=np.float64).flatten()
    fp = tuple(np.round(re_arr[:5], 4))
    if fp in seen_fingerprints:
        n_skip_dup += 1
        continue
    seen_fingerprints.add(fp)

    # Print era5 structure once on the first ORACLES file to make any future
    # struct/key drift easy to diagnose.
    debug_now = not debugged_one
    if debug_now:
        print(f'    [debug] inspecting {path.name}')
    geo = extract_lat_lon_dt(d, debug=debug_now)
    debugged_one = True

    if geo is None:
        n_skip_no_geo += 1
        continue
    lat, lon, dt = geo
    geo_records.append((lat, lon, dt, path.name))
    n_ok += 1

print(f'         kept: {n_ok}  | '
      f'skipped: {n_skip_no_alpha} no alpha, {n_skip_no_geo} no geo, '
      f'{n_skip_dup} duplicate')

if n_ok == 0:
    raise RuntimeError(
        'No (lat, lon, datetime) records extracted. Inspect a .mat file with '
        'scipy.io.loadmat and confirm era5.geo.* and era5.metadata.era5_utcTime '
        'access paths in extract_lat_lon_dt().'
    )

lats   = np.array([r[0] for r in geo_records])
lons   = np.array([r[1] for r in geo_records])
times  = [r[2] for r in geo_records]
print(f'         lat range : [{lats.min():.2f}, {lats.max():.2f}]')
print(f'         lon range : [{lons.min():.2f}, {lons.max():.2f}]')
print(f'         time span : {min(times)}  →  {max(times)}')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — Compute (SZA, SAZ) for each in-situ measurement; filter SZA<50
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[Cell 3] Computing solar geometry (pysolar) and filtering SZA < {SZA_MAX_DEG}°')

def sza_saz(lat, lon, dt_utc):
    """Solar zenith and azimuth angles in degrees.

    pysolar returns altitude (degrees above horizon) and azimuth (degrees
    east of north). We convert altitude → zenith = 90 − altitude.
    pysolar's azimuth follows the libRadtran solar-azimuth convention
    (clockwise from north) closely enough for our sampling needs.
    """
    alt = solar.get_altitude(lat, lon, dt_utc)
    az  = solar.get_azimuth(lat, lon, dt_utc)
    return 90.0 - alt, az % 360.0

sza_all = np.array([sza_saz(la, lo, dt)[0] for la, lo, dt in zip(lats, lons, times)])
saz_all = np.array([sza_saz(la, lo, dt)[1] for la, lo, dt in zip(lats, lons, times)])

mask = sza_all < SZA_MAX_DEG
sza_pool = sza_all[mask]
saz_pool = saz_all[mask]
print(f'         all measurements : SZA ∈ [{sza_all.min():.1f}, {sza_all.max():.1f}], '
      f'median = {np.median(sza_all):.1f}')
print(f'         SZA < {SZA_MAX_DEG}°  : {mask.sum()} of {len(sza_all)} '
      f'({100 * mask.sum() / len(sza_all):.1f}%)')
if mask.sum() == 0:
    raise RuntimeError(
        f'No in-situ measurements satisfy SZA < {SZA_MAX_DEG}°. '
        'Check datetime parsing — if all SZAs cluster near 90° the times may '
        'be parsing as midnight UTC.'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Pair each synthetic cloud with sampled (SZA, SAZ, VZA, VAZ)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[Cell 4] Sampling geometry per synthetic cloud (N = {N_CLOUDS})')
rng = np.random.default_rng(RANDOM_SEED)

idx_pool = rng.integers(0, len(sza_pool), size=N_CLOUDS)
sza_per_cloud = sza_pool[idx_pool]
saz_per_cloud = saz_pool[idx_pool]
vza_per_cloud = rng.uniform(VZA_RANGE_DEG[0], VZA_RANGE_DEG[1], size=N_CLOUDS)
vaz_per_cloud = rng.uniform(VAZ_RANGE_DEG[0], VAZ_RANGE_DEG[1], size=N_CLOUDS)

print(f'         SZA per cloud : [{sza_per_cloud.min():.1f}, {sza_per_cloud.max():.1f}]')
print(f'         SAZ per cloud : [{saz_per_cloud.min():.1f}, {saz_per_cloud.max():.1f}]')
print(f'         VZA per cloud : [{vza_per_cloud.min():.1f}, {vza_per_cloud.max():.1f}]')
print(f'         VAZ per cloud : [{vaz_per_cloud.min():.1f}, {vaz_per_cloud.max():.1f}]')


# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Save to NetCDF4 (single file, one row per cloud)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NC = OUT_DIR / f'training_inputs_jointMVN_N{N_CLOUDS}_L{syn["N_FIXED_LEVELS"]}.nc'

print(f'\n[Cell 5] Writing NetCDF → {OUT_NC}')

with nc.Dataset(OUT_NC, 'w', format='NETCDF4') as ds:
    ds.title       = 'Synthetic in-situ cloud profiles + paired solar/viewing geometry'
    ds.created_by  = '07_build_training_inputs.py'
    ds.source_npz  = str(SYNTHETIC_NPZ.name)
    ds.sza_filter  = f'sampled (SZA, SAZ) restricted to in-situ data with SZA < {SZA_MAX_DEG}°'
    ds.cloud_orientation = 'index 0 = cloud top, index -1 = cloud base'
    ds.atmos_orientation = 'surface → TOA on the standard ERA5 pressure grid'

    ds.createDimension('cloud',       N_CLOUDS)
    ds.createDimension('level_cloud', int(syn['N_FIXED_LEVELS']))
    ds.createDimension('level_atmos', int(syn['N_ERA5']))

    def _v(name, dims, data, units, long_name):
        v = ds.createVariable(name, 'f4', dims, zlib=True, complevel=4)
        v[:] = np.asarray(data, dtype=np.float32)
        v.units = units
        v.long_name = long_name
        return v

    _v('re_um',            ('cloud', 'level_cloud'), syn['re'],
       'um',           'Effective radius profile (cloud top → base)')
    _v('lwc_g_per_m3',     ('cloud', 'level_cloud'), syn['lwc'],
       'g m-3',        'Liquid water content profile')
    _v('z_km',             ('cloud', 'level_cloud'), syn['z'],
       'km',           'Altitude grid for cloud profile')
    _v('alpha',            ('cloud',),               syn['mean_alpha'],
       '1',            'libRadtran gamma-distribution shape parameter (vertical mean)')
    _v('tau_c',            ('cloud',),               syn['tau_c'],
       '1',            'Cloud optical depth (derived from re and lwc)')
    _v('LWP_g_per_m2',     ('cloud',),               syn['LWP_g_per_m2'],
       'g m-2',        'Liquid water path (derived from lwc)')

    _v('sza_deg',          ('cloud',),               sza_per_cloud,
       'degree',       'Solar zenith angle (bootstrapped from in-situ SZA<50)')
    _v('saz_deg',          ('cloud',),               saz_per_cloud,
       'degree',       'Solar azimuth angle (clockwise from north)')
    _v('vza_deg',          ('cloud',),               vza_per_cloud,
       'degree',       'Viewing zenith angle (Uniform[0,45])')
    _v('vaz_deg',          ('cloud',),               vaz_per_cloud,
       'degree',       'Viewing azimuth angle (Uniform[0,180])')

    _v('T_K',              ('cloud', 'level_atmos'), syn['T'],
       'K',            'ERA5 temperature profile (surface → TOA)')
    _v('vapor_molec_per_cm3', ('cloud', 'level_atmos'), syn['vapor'],
       'cm-3',         'ERA5 water vapor number density (surface → TOA)')
    _v('pressure_hPa',     ('level_atmos',),         syn['era5_pressure'],
       'hPa',          'Shared ERA5 pressure grid')

print(f'         file size: {OUT_NC.stat().st_size / 1e6:.1f} MB')
print('Done.')
