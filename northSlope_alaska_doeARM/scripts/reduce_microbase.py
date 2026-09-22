#!/usr/bin/env python3
"""Download-reduce-delete sampler for the MICROBASE VAP (nsamicrobaseC1.c1).

MICROBASE (Dunn, Johnson & Jensen 2011, DOE/SC-ARM/TR-095) is the ARM
"Continuous Baseline Microphysical Retrieval": LWC, IWC, liquid and ice
effective radius on the ARSCL time-height grid (4 s x 30 m, 596 gates). At
NSA the files are ~670 MB/day, so a two-season record (~485 days) would be
~325 GB -- not storable here. This script therefore fetches one day at a time,
keeps a compact 1-min product, and deletes the raw file, on a day stride
(--every N) so that a representative sample of both cold seasons is kept.

What is kept per day (data/processed/microbase_1min/nsamicrobase_1min.<date>.nc):
  profiles, 1-min means, (time, height) with height in m AGL, zlib-compressed
    lwc_g_m3, iwc_g_m3, re_liq_um, re_ice_um
  column quantities, (time)
    lwp_mb_g_m2        integral of LWC over height          [g m-2]
    iwp_mb_g_m2        integral of IWC over height          [g m-2]
    re_liq_lwcw_um     LWC-weighted mean liquid r_e          [um]
    re_ice_iwcw_um     IWC-weighted mean ice r_e             [um]
    re_liq_base_um     liquid r_e at the lowest liquid gate  [um]
    z_liq_base_m, z_liq_top_m   lowest / highest gate with LWC > 0 [m AGL]
    z_ice_base_m, z_ice_top_m   same for IWC
    n_liq_gates, n_ice_gates    number of 30-m gates with LWC / IWC > 0
    mwr_scale_factor   MWR LWP / integrated Z-based LWC (the scaling applied)
    precip_flag, clear_cloud_flag (max over the minute)
    n_valid_gates      gates with a usable retrieval (retrieval_flag 0-3)

QC applied before averaging: gates whose qc_ variable has a "Bad" bit set are
NaN (arm_nsa.qc.apply_qc); "Indeterminate" bits (possible clutter, out of
range, questionable MWR LWP, liquid precipitation) are kept but the column
count of such gates could be added if needed.

IMPORTANT for interpretation (verified against the file attributes and the
MICROBASE report): the liquid r_e is NOT an independent size measurement. It
is derived from the retrieved LWC assuming a fixed droplet number
concentration and a log-normal width (file comment: "For sigma = 0.35,
Re = 1.358 Rm"), so r_e_liq ~ LWC^(1/3). The ice r_e follows Ivanova et al.
(2001), a function of temperature only. Both are parameterizations; the
notebook treats them as such.

Usage (from the repo root; re-runnable, days already reduced are skipped):
    python scripts/reduce_microbase.py --start 2023-09-01 --end 2024-04-30 --every 5
    python scripts/reduce_microbase.py --start 2024-09-01 --end 2025-04-30 --every 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402
from arm_nsa.credentials import get_credentials  # noqa: E402
from arm_nsa.download import download_file, query_files  # noqa: E402
from arm_nsa.qc import apply_qc  # noqa: E402

ARM_NAME = "nsamicrobaseC1.c1"
OUT_DIR = config.PROCESSED_DATA_DIR / "microbase_1min"
GATE_DZ_M = 30.0  # ARSCL/KAZR range-gate spacing at NSA [m]; verified from height_bounds
VALID_RETRIEVAL_FLAGS = (0, 1, 2, 3)  # 10 = no reflectivity data at all


def _weighted_mean(x: xr.DataArray, w: xr.DataArray, dim: str) -> xr.DataArray:
    """Weighted mean of x with weights w along dim, NaN where the weights sum to 0."""
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    num = (x * w).where(ok).sum(dim)
    den = w.where(ok).sum(dim)
    return (num / den).where(den > 0)


def _edge_height(mask: xr.DataArray, height: xr.DataArray, which: str) -> xr.DataArray:
    """Lowest ('base') or highest ('top') height where mask is True, NaN if none."""
    h = height.broadcast_like(mask).where(mask)
    return h.min("height") if which == "base" else h.max("height")


def reduce_day(raw_path: Path) -> xr.Dataset:
    """Collapse one raw MICROBASE day to the compact 1-min product."""
    ds = xr.open_dataset(raw_path)
    # --- QC: drop Bad-flagged gates, keep Indeterminate ones -------------------
    lwc = apply_qc(ds, "liquid_water_content")
    iwc = apply_qc(ds, "ice_water_content")
    re_l = apply_qc(ds, "liquid_effective_radius")
    re_i = apply_qc(ds, "ice_effective_radius")
    valid = ds["retrieval_flag"].isin(VALID_RETRIEVAL_FLAGS)

    # --- 1-min means of the profiles (4-s native) --------------------------------
    prof = xr.Dataset(
        {
            "lwc_g_m3": lwc,
            "iwc_g_m3": iwc,
            "re_liq_um": re_l,
            "re_ice_um": re_i,
        }
    ).resample(time="1min").mean()
    n_valid = valid.astype("int16").resample(time="1min").max().sum("height")

    # --- column quantities from the 1-min profiles ------------------------------
    lwc1 = prof["lwc_g_m3"]
    iwc1 = prof["iwc_g_m3"]
    liq = lwc1 > 0
    ice = iwc1 > 0
    col = xr.Dataset()
    # A column with no usable gate at all is NaN, not 0 (radar outage), whereas a
    # valid clear column integrates to 0 -- that distinction matters for regressions.
    col_ok = n_valid > 0
    col["lwp_mb_g_m2"] = (lwc1.fillna(0.0) * GATE_DZ_M).sum("height").where(col_ok)
    col["iwp_mb_g_m2"] = (iwc1.fillna(0.0) * GATE_DZ_M).sum("height").where(col_ok)
    col["re_liq_lwcw_um"] = _weighted_mean(prof["re_liq_um"], lwc1, "height")
    col["re_ice_iwcw_um"] = _weighted_mean(prof["re_ice_um"], iwc1, "height")
    hgt = prof["height"]
    col["z_liq_base_m"] = _edge_height(liq, hgt, "base")
    col["z_liq_top_m"] = _edge_height(liq, hgt, "top")
    col["z_ice_base_m"] = _edge_height(ice, hgt, "base")
    col["z_ice_top_m"] = _edge_height(ice, hgt, "top")
    # r_e at the lowest liquid gate: pick by height index of the base
    base_idx = liq.argmax("height")  # first True along height (0 if none)
    col["re_liq_base_um"] = prof["re_liq_um"].isel(height=base_idx).where(liq.any("height"))
    col["n_liq_gates"] = liq.sum("height").astype("int16")
    col["n_ice_gates"] = ice.sum("height").astype("int16")
    col["n_valid_gates"] = n_valid.astype("int16")
    col["mwr_scale_factor"] = ds["mwr_scale_factor"].resample(time="1min").mean()
    col["precip_flag"] = ds["precip_flag"].resample(time="1min").max().astype("int8")
    col["clear_cloud_flag"] = ds["clear_cloud_flag"].resample(time="1min").max().astype("int8")
    col = col.drop_vars("height", errors="ignore")

    out = xr.merge([prof, col])
    units = {
        "lwc_g_m3": "g m-3", "iwc_g_m3": "g m-3", "re_liq_um": "um", "re_ice_um": "um",
        "lwp_mb_g_m2": "g m-2", "iwp_mb_g_m2": "g m-2", "re_liq_lwcw_um": "um",
        "re_ice_iwcw_um": "um", "re_liq_base_um": "um", "z_liq_base_m": "m",
        "z_liq_top_m": "m", "z_ice_base_m": "m", "z_ice_top_m": "m",
    }
    for k, u in units.items():
        out[k].attrs["units"] = u
    out["height"].attrs.update({"units": "m", "long_name": "height above ground level"})
    out.attrs = {
        "title": "MICROBASE 1-min reduction (scripts/reduce_microbase.py)",
        "source_file": raw_path.name,
        "source_process_version": str(ds.attrs.get("process_version", "")),
        "source_dod_version": str(ds.attrs.get("dod_version", "")),
        "lwc_comment": str(ds["liquid_water_content"].attrs.get("comment", "")),
        "iwc_comment": str(ds["ice_water_content"].attrs.get("comment", "")),
        "re_liq_comment": str(ds["liquid_effective_radius"].attrs.get("comment", "")),
        "re_ice_comment": str(ds["ice_effective_radius"].attrs.get("comment", "")),
        "qc_policy": "gates with any Bad qc bit set to NaN before 1-min averaging",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    ds.close()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--every", type=int, default=5, help="day stride (1 = every day)")
    p.add_argument("--keep-raw", action="store_true", help="do not delete the raw 670 MB file")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)
    days = [start + dt.timedelta(days=k) for k in range(0, (end - start).days + 1, a.every)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    creds = get_credentials()
    raw_dir = config.raw_dir_for(ARM_NAME)
    print(f"{len(days)} day(s) requested, stride {a.every}; output -> {OUT_DIR}")

    for day in days:
        tag = day.strftime("%Y%m%d")
        out_path = OUT_DIR / f"nsamicrobase_1min.{tag}.nc"
        if out_path.exists():
            print(f"{tag}: already reduced, skipping")
            continue
        files = query_files(ARM_NAME, day.isoformat(), day.isoformat(), creds)
        if not files:
            print(f"{tag}: not in archive")
            continue
        if a.dry_run:
            print(f"{tag}: would fetch {files}")
            continue
        t0 = time.time()
        raw = download_file(files[0], raw_dir, creds)
        t1 = time.time()
        try:
            red = reduce_day(raw)
            enc = {v: {"zlib": True, "complevel": 4} for v in red.data_vars if red[v].ndim == 2}
            # atomic write: a reader (the notebook's merge step) must never see a half-written file
            tmp_path = out_path.with_suffix(".nc.tmp")
            red.to_netcdf(tmp_path, encoding=enc)
            tmp_path.rename(out_path)
        finally:
            if not a.keep_raw:
                raw.unlink(missing_ok=True)
        print(
            f"{tag}: download {t1 - t0:5.0f} s ({raw.stat().st_size / 1e6 if raw.exists() else 670:.0f} MB), "
            f"reduce {time.time() - t1:4.0f} s -> {out_path.name} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
