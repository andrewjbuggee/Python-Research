#!/usr/bin/env python3
"""Regression guard: do the g m-2 CLI thresholds mean what they say?

Checks the masks each script builds against a reference computed straight from
the native kg m-2 arrays, so a mislabelled unit shows up as a count mismatch.
"""
import glob, sys, warnings, numpy as np, xarray as xr, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import analyze_cloud_liquid_frequency as acl
from cloud_classification import cloud_phase_masks

ds = xr.open_mfdataset(sorted(glob.glob("data/barrow/*.nc")), combine="by_coords",
                       join="outer", compat="no_conflicts")[["tcc", "tclw", "tciw"]]
t = pd.DatetimeIndex(ds["valid_time"].values)
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
ds = ds.isel(valid_time=np.isin(t.year, (YEAR,))).compute()
tclw_kg = ds["tclw"].values          # native
tciw_kg = ds["tciw"].values
tcc = ds["tcc"].values

fails = []
print(f"sample: {tclw_kg.size:,} cell-hours ({YEAR})\n")
print(f"{'g m-2 threshold':>16}{'script mask':>14}{'kg reference':>14}{'':>4}")
for thr_g in (0.0, 0.03, 1.0, 5.0, 10.0, 25.0, 50.0):
    thr_kg = thr_g / 1000.0
    # what the script does, given a g m-2 CLI value
    cloudy, liquid, liquid_any, valid = acl.build_masks(
        ds, thr_g, 0.0, 1.0, ocean_only=False)
    # the truth, computed only from native kg m-2 values
    ref_valid = np.isfinite(tcc) & np.isfinite(tclw_kg) & np.isfinite(tciw_kg)
    ref_cloudy = ref_valid & (tcc >= 1.0) & ((tclw_kg > thr_kg) | (tciw_kg > 0.0))
    ref_liquid = ref_cloudy & (tclw_kg > thr_kg)
    ok = (cloudy == ref_cloudy).all() and (liquid == ref_liquid).all()
    if not ok:
        fails.append(("build_masks", thr_g))
    print(f"{thr_g:>16g}{int(liquid.sum()):>14,}{int(ref_liquid.sum()):>14,}"
          f"{'  ok' if ok else '  MISMATCH':>4}")

# cloud_classification takes g m-2 arrays; confirm the same threshold in g on a
# g-converted array equals the kg threshold on the native array.
print()
for thr_g in (0.03, 30.0):
    ph = cloud_phase_masks(tclw_kg * 1000.0, tciw_kg * 1000.0,
                           lwp_min_g=thr_g, iwp_min_g=thr_g,
                           lwp_max_ice_g=0.001, iwp_max_liquid_g=0.001)
    ref = (tclw_kg > thr_g / 1000.0) & (tciw_kg > thr_g / 1000.0)
    ok = (ph["mixed"] == (ref & np.isfinite(tclw_kg) & np.isfinite(tciw_kg))).all()
    if not ok:
        fails.append(("cloud_phase_masks", thr_g))
    print(f"cloud_phase_masks mixed @ {thr_g:g} g m-2: "
          f"{int(ph['mixed'].sum()):,} vs kg reference {int(ref.sum()):,}"
          f"{'  ok' if ok else '  MISMATCH'}")

print("\nRESULT:", "all unit conversions consistent" if not fails else f"FAILURES {fails}")
sys.exit(1 if fails else 0)
