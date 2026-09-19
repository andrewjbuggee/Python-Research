#!/usr/bin/env python3
"""Check the S3 backend against the locally downloaded CDS archive.

Two levels:

1. ``variables`` -- open the same box and hours from S3 and from
   ``data/<region>/``, and compare every variable cell by cell: NaN pattern
   agreement, max |difference|, and for ``tp`` (synthesised from ``mtpr`` on
   S3) the agreement of the precipitation-filter mask at the threshold the
   Ocean Visions figures use.

2. ``analysis`` -- run ``plot_lwp_histogram_by_surface_class.prepare`` with
   ``storage="local"`` and ``storage="aws"`` for one season and compare the
   reduced outputs the figures are drawn from (``A.col``), so the whole
   streaming path is exercised, not just the reader.

Examples
--------
    python verify_against_local.py variables --region barrow --start 2024-10-01 --end 2024-10-04
    python verify_against_local.py variables --region barrow --start 2024-10-01 --end 2024-10-01 --pressure
    python verify_against_local.py analysis  --region barrow --season-year 2024 --season-end 10-31

Results on 2026-09-18 (Barrow, 1-4 Oct 2024): every analysis and flux
variable bit-for-bit (max |diff| = 0, NaN patterns identical); ``istl1``
within 6.1e-5 K (two float32 ULP -- the two GRIB-to-netCDF converters round
differently); ``tp`` within 1.5e-6 m per hour, i.e. GRIB packing precision,
which flips the 0.05 mm/hr precipitation mask on ~1% of the cell-hours it
removes; pressure-level ``u``/``v`` on 16 levels bit-for-bit; land-sea mask
bit-for-bit.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path

import numpy as np

SEB_DIR = Path(__file__).resolve().parent.parent
if str(SEB_DIR) not in sys.path:
    sys.path.insert(0, str(SEB_DIR))

from aws_pipeline import era5_s3, s3_storage  # noqa: E402


def _local_files(region_dir: Path, start: str, end: str) -> list[str]:
    """Local chunk files overlapping start..end, by name (same rule as the archive)."""
    from download_era5_seb import days_covered_by_file  # noqa: E402

    d0 = np.datetime64(start, "D")
    d1 = np.datetime64(end, "D")
    days = {np.datetime64(d, "D").astype(object) for d in np.arange(d0, d1 + 1)}
    out = []
    for f in sorted(glob.glob(str(region_dir / "*.nc"))):
        if days_covered_by_file(Path(f)) & days:
            out.append(f)
    return out


def verify_variables(args) -> int:
    import xarray as xr
    from seb_analysis_common import resolve_data_root  # noqa: E402

    n, w, s, e = s3_storage.region_box(args.region)
    root = resolve_data_root(args.local_storage, None)
    region = args.region + ("_pressure_wind" if args.pressure else "")
    files = _local_files(root / region, args.start, args.end)
    if not files:
        print(f"no local files for {region} in {args.start}..{args.end} under {root}")
        return 2
    loc = xr.open_mfdataset(files, combine="nested", concat_dim="valid_time",
                            join="outer", compat="override", coords="minimal",
                            data_vars="minimal")
    loc = loc.sel(valid_time=slice(args.start, args.end + "T23")).sortby("valid_time").load()
    if args.pressure:
        variables = [v for v in ("u", "v") if v in loc.data_vars]
        levels = loc["pressure_level"].values.tolist()
        ds = era5_s3.open_dataset(n, w, s, e, [(args.start, args.end)], variables=variables,
                                  levels_hpa=levels, month_align=False)
    else:
        variables = [v for v in era5_s3.SEB_STANDARD if v in loc.data_vars]
        ds = era5_s3.open_dataset(n, w, s, e, [(args.start, args.end)], variables=variables,
                                  month_align=False)
    t0 = time.time()
    sub = ds.load()
    print(f"  loaded {len(variables)} variables x {sub.sizes['valid_time']} h in {time.time() - t0:.0f} s")

    ok = True
    ok &= np.array_equal(sub.valid_time.values, loc.valid_time.values)
    print(f"  valid_time identical : {np.array_equal(sub.valid_time.values, loc.valid_time.values)}")
    print(f"  latitude identical   : {np.allclose(sub.latitude.values, loc.latitude.values)}")
    print(f"  longitude identical  : {np.allclose(sub.longitude.values, loc.longitude.values)}")
    if args.pressure:
        print(f"  levels identical     : {np.array_equal(sub.pressure_level.values, loc.pressure_level.values)}")
    print(f"\n  {'variable':<11}{'nan agree':>10}{'nan% S3':>9}{'nan% CDS':>9}{'max|diff|':>12}"
          f"{'max rel':>10}{'CDS mean':>12}  verdict")
    for v in variables:
        a, b = sub[v].values, loc[v].values
        na, nb = np.isnan(a), np.isnan(b)
        both = ~na & ~nb
        d = np.abs(a[both].astype("float64") - b[both].astype("float64")) if both.any() else np.array([0.0])
        rel = d / np.maximum(np.abs(b[both].astype("float64")), 1e-30) if both.any() else np.array([0.0])
        agree = float(np.mean(na == nb))
        verdict = "bit-for-bit" if (d.max() == 0 and agree == 1.0) else (
            "float32 rounding" if rel.max() < 1e-5 and agree == 1.0 else "DIFFERS")
        if v == "tp":
            verdict = "packing (see mask)"
        print(f"  {v:<11}{agree * 100:>9.3f}%{na.mean() * 100:>8.1f}%{nb.mean() * 100:>8.1f}%"
              f"{d.max():>12.3e}{rel.max():>10.1e}{np.nanmean(b):>12.4g}  {verdict}")
        if verdict == "DIFFERS":
            ok = False
        if v == "tp":
            thr_m = args.precip_rate_max / 1000.0        # mm/hr -> m/hr
            ma, mb = (a >= thr_m), (b >= thr_m)
            flips = int(np.sum(ma != mb))
            print(f"      precipitation mask (tp >= {args.precip_rate_max} mm/hr): "
                  f"{flips:,} of {int(mb.sum()):,} CDS-flagged cell-hours differ "
                  f"({100 * flips / max(1, int(mb.sum())):.2f}%); "
                  f"{int(np.sum(ma & ~mb))} S3-only, {int(np.sum(mb & ~ma))} CDS-only")
            q = np.abs(a[both] - b[both])
            print(f"      |tp diff| quantiles: 50% {np.quantile(q, 0.5):.2e}  99% {np.quantile(q, 0.99):.2e}  "
                  f"max {q.max():.2e} m/hr")
    if not args.pressure:
        lsm = era5_s3.open_land_sea_mask(n, w, s, e)
        try:
            from surface_classification import load_land_sea_mask  # noqa: E402

            loc_lsm = load_land_sea_mask(args.region, root, None)
            print(f"\n  land-sea mask max|diff| vs local: {float(np.nanmax(np.abs(lsm.values - loc_lsm.values))):.3e}")
        except FileNotFoundError as exc:
            print(f"\n  (no local land-sea mask to compare: {exc})")
    era5_s3.shutdown_pool()
    print("\nRESULT:", "OK" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


def _walk(a, b, path, report):
    worst = 0.0
    if isinstance(a, dict):
        for k in a:
            if k in b:
                worst = max(worst, _walk(a[k], b[k], f"{path}.{k}", report))
    elif isinstance(a, (np.ndarray, list, tuple)) and np.asarray(a).dtype.kind in "fiub":
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if aa.shape != bb.shape:
            report.append(f"  shape {path}: local {aa.shape} vs aws {bb.shape}")
            return 0.0
        d = float(np.nanmax(np.abs(aa - bb))) if aa.size else 0.0
        scale = float(np.nanmax(np.abs(aa))) if aa.size else 0.0
        if d > 0:
            report.append(f"  {path:<40} max|d| {d:9.3e}  scale {scale:9.3e}  rel {d / max(scale, 1e-30):8.1e}")
        worst = max(worst, d / max(scale, 1e-30))
    elif isinstance(a, (int, float, np.floating, np.integer)):
        d = abs(float(a) - float(b))
        if d > 0:
            report.append(f"  {path:<40} {a} vs {b}")
        worst = max(worst, d / max(abs(float(a)), 1e-30))
    return worst


def verify_analysis(args) -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402

    m1, d1 = (int(x) for x in args.season_end.split("-"))
    m0, d0 = (int(x) for x in args.season_start.split("-"))
    common = dict(region=args.region, years=(args.season_year,), season_start=(m0, d0),
                  season_end=(m1, d1), phase_mode="fraction", liquid_fraction_min=0.90,
                  ice_fraction_min=0.90, min_lwp=0.01, min_iwp=0.01, min_cloud_fraction=0.95,
                  lsm_tol=0.01, dpi=100, no_precip=not args.all_sky, precip_var="rate",
                  precip_rate_max=args.precip_rate_max)
    t0 = time.time()
    A_loc = lwph.prepare(**common, storage=args.local_storage)
    t_loc = time.time() - t0
    t0 = time.time()
    A_aws = lwph.prepare(**common, storage="aws")
    t_aws = time.time() - t0
    report: list[str] = []
    worst = _walk(A_loc.col, A_aws.col, "col", report)
    print(f"\n=== lwph.prepare: local {t_loc:.0f} s, aws {t_aws:.0f} s "
          f"(aws again would be ~free from the chunk cache)")
    print(f"    seasons used: local {A_loc.used}  aws {A_aws.used}")
    print("    A.col differences (nothing listed = identical):")
    print("\n".join(report) if report else "      none")
    print(f"    worst relative difference in A.col: {worst:.2e}")
    if "precip_removed" in A_loc.col:
        pr_l, pr_a = A_loc.col["precip_removed"], A_aws.col["precip_removed"]
        print(f"    precipitating cell-hours removed: local {pr_l:,.0f}  aws {pr_a:,.0f}  "
              f"({100 * (pr_a - pr_l) / max(pr_l, 1):+.2f}%)")
    era5_s3.shutdown_pool()
    return 0 if worst < 1e-2 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variables", help="cell-by-cell comparison of every variable")
    v.add_argument("--region", default="barrow")
    v.add_argument("--start", default="2024-10-01")
    v.add_argument("--end", default="2024-10-04")
    v.add_argument("--pressure", action="store_true", help="compare the <region>_pressure_wind archive (u, v)")
    v.add_argument("--local-storage", default="local", choices=["local", "external"])
    v.add_argument("--precip-rate-max", type=float, default=0.05, help="mm/hr, the OV filter threshold")
    a = sub.add_parser("analysis", help="lwph.prepare local vs aws for one season")
    a.add_argument("--region", default="barrow")
    a.add_argument("--season-year", type=int, default=2024)
    a.add_argument("--season-start", default="10-01")
    a.add_argument("--season-end", default="03-31")
    a.add_argument("--all-sky", action="store_true", help="no precipitation filter")
    a.add_argument("--local-storage", default="local", choices=["local", "external"])
    a.add_argument("--precip-rate-max", type=float, default=0.05)
    args = p.parse_args(argv)
    return verify_variables(args) if args.cmd == "variables" else verify_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
