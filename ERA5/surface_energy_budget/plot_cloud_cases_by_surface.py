#!/usr/bin/env python3
"""Rachel's cloud cases A-F in ERA5, broken out by surface class.

Runs the ported case detection in ``cloud_vertical_cases.py`` over the ERA5
pressure-level archive, applies the same quality filters, and reports how often
each case occurs -- for the five surface classes and for the single grid cell
holding the DOE ARM facility at Utqiagvik, so the ERA5 result can be set beside
the ground-based statistics.

WHAT THIS NUMBER IS COMPARABLE TO
=================================
Every statistic here is a fraction of CELL-HOURS. Rachel's pipeline produces two
different columns, and only one of them is the right comparison:

    "Hours passing column filters"   <- compare against THIS
    "Hours in periods >= 30 min"     <- do NOT compare against this

Her second column exists because ARM samples every 30 s, so events have to be
assembled from many samples, gaps bridged and short events dropped. ERA5 samples
hourly: one sample already exceeds her 30-minute minimum and her 90-second gap
bridge is far below the sampling interval. Neither post-processing step can be
applied, so what comes out here is the raw per-sample count -- her first column.

Everything else about the detection is the same code path, verified column by
column against her scalar implementation.

USAGE
=====
    python plot_cloud_cases_by_surface.py --region barrow --storage local
    python plot_cloud_cases_by_surface.py --region barrow --basis levels
"""

from __future__ import annotations

import argparse
import calendar
import os
import glob
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cloud_vertical_cases import (
    CASE_COLORS,
    CASE_DESC,
    CASE_LABELS,
    CASE_ORDER,
    DEFAULT_PHASE_RULE,
    DEFAULT_PHASE_THRESHOLD_G_M3,
    LWP_THRESHOLD_G_M2,
    PRECIP_THRESHOLD_MM_HR,
    detect_columns,
    profile_fields,
)
from download_era5_seb import STORAGE_ROOTS, days_covered_by_file
from plot_surface_class_timeseries import (
    SITE_COLOR,
    SITE_LABEL,
    site_cell_mask,
)
from seb_analysis_common import area_weights
from surface_classification import (
    CLASS_CODES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    align_lsm_to_grid,
    classify_cells,
    load_land_sea_mask,
)

DEFAULT_MIN_CLOUD_FRACTION = 0.99
SINGLE_VARS = ("tcc", "sp", "tp", "siconc")

# Filters in the order they are reported, each a (key, label) pair. The order is
# the order Rachel lists them, so the two breakdowns can be read side by side.
FILTER_ORDER = [
    ("cloudy", "Total cloud cover >= threshold"),
    ("has_liquid", "Has liquid in the column"),
    ("clear_below", "Clear sky below cloud base"),
    ("no_drizzle", "No drizzle/rain in column"),
    ("no_snow", "No snow in column"),
    ("precip", "Precipitation < 0.01 mm/hr"),
    ("lwp", "LWP > 2 g/m2"),
]


class Analysis(SimpleNamespace):
    """Everything the figures need, computed in one streaming pass."""


def _classify_one(task):
    """Classify one pressure-level file. Runs in a worker process.

    Takes plain arrays rather than xarray objects so the payload pickles
    cheaply: the single-level slice for one file is a few MB, against the
    hundreds of MB an open dataset would cost. Returns partial accumulators
    that the parent simply adds up -- every reduction here is a sum, so the
    split is exact and the result does not depend on how many workers run.
    """
    import xarray as xr

    (path, cfg, sl_arrays, lsm, w2d, site_mask, months) = task
    pl = xr.open_dataset(path).load()
    phase, lwp, iwp, extras = profile_fields(
        pl, sl_arrays["sp"], cfg["phase_threshold"], cfg["phase_rule"])
    dims = extras["dims"]
    shape = tuple(pl.sizes[d] for d in dims)

    res = detect_columns(phase.reshape(-1, phase.shape[-1]),
                         lwp.reshape(-1, lwp.shape[-1]),
                         iwp.reshape(-1, iwp.shape[-1]), basis=cfg["basis"])
    case = res["case"].reshape(shape)
    has_liq = (res["liq_count"] > 0).reshape(shape)
    clear_below = res["clear_below"].reshape(shape)
    pl.close()

    f_cloudy = sl_arrays["tcc"] >= cfg["min_cloud_fraction"]
    f_precip = sl_arrays["tp"] * 1000.0 < PRECIP_THRESHOLD_MM_HR
    f_lwp = extras["lwp_column"] > LWP_THRESHOLD_G_M2
    f_dry, f_snow = extras["no_drizzle"], extras["no_snow"]
    keep = (f_cloudy & has_liq & clear_below & f_dry & f_snow
            & f_precip & f_lwp)

    classes = classify_cells(lsm, sl_arrays["siconc"], cfg["lsm_tol"],
                             cfg["open_ocean_max_siconc"],
                             cfg["sea_ice_min_siconc"], cfg["land_max_siconc"])
    w = np.broadcast_to(w2d, shape)

    n_series = len(CLASS_ORDER) + 1
    site_code = len(CLASS_ORDER)
    n_case = 7
    counts = np.zeros((n_case, n_series))
    valid_w = np.zeros(n_series)
    month_parts: dict[int, np.ndarray] = {}

    selectors = [(CLASS_CODES[n], classes == CLASS_CODES[n]) for n in CLASS_ORDER]
    selectors.append((site_code, np.broadcast_to(site_mask, shape)))
    m3 = np.broadcast_to(months[:, None, None], shape)
    uniq = np.unique(months)

    for code, sel in selectors:
        wsel = w * sel
        valid_w[code] += float(wsel.sum())
        for c in range(n_case):
            hit = wsel * (((case == c) & keep) if c else ~keep)
            counts[c, code] += float(hit.sum())
        for m in uniq:
            mm = month_parts.setdefault(int(m), np.zeros((n_case, n_series)))
            sel_m = wsel * (m3 == m)
            for c in range(1, n_case):
                mm[c, code] += float((sel_m * ((case == c) & keep)).sum())
            mm[0, code] += float(sel_m.sum())

    filt = {}
    for key, flag in (("cloudy", f_cloudy), ("has_liquid", has_liq),
                      ("clear_below", clear_below), ("no_drizzle", f_dry),
                      ("no_snow", f_snow), ("precip", f_precip), ("lwp", f_lwp)):
        filt[key] = float((w * flag).sum())
    return (counts, valid_w, month_parts, filt, float(w.sum()), shape[0])


def prepare(argv=None, args=None, **overrides):
    """Stream the pressure-level archive and count cases by surface class.

    Slow: it reads every pressure-level file once. A notebook should call this
    once and keep the result.
    """
    import xarray as xr

    warnings.filterwarnings("ignore", category=FutureWarning)
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    root = args.data_root or STORAGE_ROOTS[args.storage]
    pl_dir = Path(root) / f"{args.region}_pressure"
    single_dir = Path(root) / args.region

    pl_files = sorted(glob.glob(str(pl_dir / "*.nc")))
    if not pl_files:
        raise FileNotFoundError(f"no pressure-level files in {pl_dir}")
    single_files = sorted(glob.glob(str(single_dir / "*.nc")))
    if not single_files:
        raise FileNotFoundError(f"no single-level files in {single_dir}")

    print("=" * 78)
    print("Cloud cases A-F by surface class")
    print("=" * 78)
    print(f"  Pressure   : {len(pl_files)} file(s) from {pl_dir}")
    print(f"  Basis      : {args.basis}-weighted phase fractions")
    print(f"  Phase thr  : {args.phase_threshold:g} g m-3   rule: {args.phase_rule}")
    print(f"  Cloudy when: tcc >= {args.min_cloud_fraction:g}")

    # Open ONLY the single-level files whose days the pressure archive actually
    # covers. This is the single largest cost in the whole pipeline: the
    # single-level archive spans 2000-2026 in 654 files (8.5 GB), the pressure
    # archive covers a few months, and open_mfdataset over the whole lot reads
    # every one of them to build the time index -- 25x more files and 19x more
    # bytes than needed, taking longer than all the classification put together.
    # Filtering on filename dates first is free and turns minutes into seconds.
    pl_days = set()
    for f in pl_files:
        pl_days |= days_covered_by_file(Path(f))
    needed = [f for f in single_files
              if days_covered_by_file(Path(f)) & pl_days]
    if not needed:
        raise FileNotFoundError(
            "no single-level file overlaps the pressure archive's dates "
            f"({min(pl_days)} .. {max(pl_days)})")
    print(f"  Single lvl : {len(needed)} of {len(single_files)} file(s) overlap "
          f"the pressure dates")
    single = xr.open_mfdataset(needed, combine="by_coords", join="outer",
                               compat="no_conflicts")[list(SINGLE_VARS)].load()
    print(f"  Loaded single-level subset: "
          f"{single.sizes['valid_time']:,} time steps")
    lsm = align_lsm_to_grid(
        load_land_sea_mask(args.region, Path(root), args.mask_grid), single)

    n_series = len(CLASS_ORDER) + 1
    site_code = len(CLASS_ORDER)
    n_case = 7                                    # 0 = none, 1..6 = cases
    counts = np.zeros((n_case, n_series))
    valid_w = np.zeros(n_series)
    months_seen: dict[int, np.ndarray] = {}
    filt_pass = {k: 0.0 for k, _ in FILTER_ORDER}
    filt_total = 0.0
    n_steps = 0

    cfg = {k: getattr(args, k) for k in
           ("phase_threshold", "phase_rule", "basis", "min_cloud_fraction",
            "lsm_tol", "open_ocean_max_siconc", "sea_ice_min_siconc",
            "land_max_siconc")}
    probe = xr.open_dataset(pl_files[0])
    smask, site_lat, site_lon = site_cell_mask(probe)
    w2d = np.broadcast_to(area_weights(probe).values[:, None],
                          (probe.sizes["latitude"], probe.sizes["longitude"])
                          ).copy()
    probe.close()
    lsm_a = np.asarray(lsm)

    def build_task(f):
        pl_t = xr.open_dataset(f)["valid_time"].values
        pl_t = pl_t[np.isin(pl_t, single["valid_time"].values)]
        if pl_t.size == 0:
            return None
        sl = single.sel(valid_time=pl_t)
        arrays = {v: sl[v].transpose("valid_time", "latitude",
                                     "longitude").values.astype(float)
                  for v in SINGLE_VARS}
        months = pd.DatetimeIndex(pl_t).month.values
        return (f, cfg, arrays, lsm_a, w2d, smask, months)

    def fold(part):
        nonlocal filt_total, n_steps
        c, vw, mp, fl, wt, nt = part
        counts[:] += c
        valid_w[:] += vw
        for m, mm in mp.items():
            months_seen.setdefault(m, np.zeros_like(mm))
            months_seen[m] += mm
        for k, v in fl.items():
            filt_pass[k] += v
        filt_total += wt
        n_steps += nt

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(pl_files)))
    print(f"  Workers    : {jobs}")
    if jobs == 1:
        for i, f in enumerate(pl_files, 1):
            task = build_task(f)
            if task is None:
                print(f"  !! {Path(f).name}: no overlapping single-level data",
                      file=sys.stderr)
                continue
            fold(_classify_one(task))
            if i % 10 == 0 or i == len(pl_files):
                print(f"    [{i}/{len(pl_files)}] {Path(f).name}")
    else:
        # Files are independent and every reduction is a sum, so the result is
        # identical to the serial path regardless of worker count or ordering.
        #
        # macOS spawns rather than forks, so the worker re-imports __main__. That
        # is fine from a script or a notebook, but fails outright when the parent
        # has no importable __main__ (a python -c or a heredoc). Rather than let
        # that surface as a BrokenProcessPool halfway through, it falls back to
        # the serial path, which gives the same answer more slowly.
        tasks = []
        for f in pl_files:
            task = build_task(f)
            if task is None:
                print(f"  !! {Path(f).name}: no overlapping single-level data",
                      file=sys.stderr)
                continue
            tasks.append(task)
        try:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(_classify_one, t) for t in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    fold(fut.result())
                    if i % 10 == 0 or i == len(futures):
                        print(f"    [{i}/{len(futures)}] done")
        except BrokenProcessPool:
            print("  !! worker pool unusable here (no importable __main__); "
                  "falling back to serial.", file=sys.stderr)
            counts[:] = 0.0
            valid_w[:] = 0.0
            months_seen.clear()
            for k in filt_pass:
                filt_pass[k] = 0.0
            filt_total = 0.0
            n_steps = 0
            for i, t in enumerate(tasks, 1):
                fold(_classify_one(t))
                if i % 10 == 0 or i == len(tasks):
                    print(f"    [{i}/{len(tasks)}] done")

    labels = [CLASS_LABELS[n] for n in CLASS_ORDER] + [SITE_LABEL]
    colors = [CLASS_COLORS[n] for n in CLASS_ORDER] + [SITE_COLOR]
    return Analysis(
        args=args, counts=counts, valid_w=valid_w, months=months_seen,
        filt_pass=filt_pass, filt_total=filt_total, n_steps=n_steps,
        labels=labels, colors=colors, site_code=site_code,
        site_lat=site_lat, site_lon=site_lon, n_files=len(pl_files),
    )


def case_percent(A, series=None):
    """Percent of cell-hours in each case, for one series or the whole domain."""
    if series is None:
        num = A.counts[:, :len(CLASS_ORDER)].sum(axis=1)
        den = A.valid_w[:len(CLASS_ORDER)].sum()
    else:
        num, den = A.counts[:, series], A.valid_w[series]
    return 100.0 * num / den if den else np.zeros_like(num)


def print_report(A):
    """Filter breakdown and case frequencies, laid out like Rachel's summary."""
    print("\n" + "-" * 78)
    print("  FILTER BREAKDOWN (fraction of all cell-hours, each filter alone)")
    print("-" * 78)
    for key, label in FILTER_ORDER:
        pct = 100.0 * A.filt_pass[key] / A.filt_total if A.filt_total else 0.0
        note = "   (vacuous by construction)" if key == "clear_below" else ""
        print(f"    {label:<36}{pct:6.1f}%{note}")

    print("\n" + "-" * 78)
    print("  CASE FREQUENCY, whole domain")
    print("-" * 78)
    pct = case_percent(A)
    print(f"    {'case':<6}{'description':<52}{'% of cell-hours':>16}")
    for c in CASE_ORDER:
        print(f"    {CASE_LABELS[c]:<6}{CASE_DESC[CASE_LABELS[c]]:<52}"
              f"{pct[c]:>15.3f}%")
    print(f"    {'total':<6}{'':<52}{pct[CASE_ORDER].sum():>15.3f}%")

    print("\n" + "-" * 78)
    print("  CASE FREQUENCY BY SURFACE CLASS  [% of that class's cell-hours]")
    print("-" * 78)
    hdr = f"    {'case':<6}" + "".join(f"{l[:13]:>15}" for l in A.labels)
    print(hdr)
    for c in CASE_ORDER:
        row = [case_percent(A, s)[c] for s in range(len(A.labels))]
        print(f"    {CASE_LABELS[c]:<6}" + "".join(f"{v:>15.3f}" for v in row))
    tot = [case_percent(A, s)[CASE_ORDER].sum() for s in range(len(A.labels))]
    print(f"    {'total':<6}" + "".join(f"{v:>15.3f}" for v in tot))
    print(f"\n  ARM cell centre {A.site_lat:.3f} N, {A.site_lon:.3f} E")
    print("  Compare against Rachel's 'Hours passing column filters' column,")
    print("  NOT her '>= 30 min' column -- see the module docstring.")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _emit(fig, A, out_dir, stem, dpi=None):
    if out_dir is None:
        return fig
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{A.args.region}_cloudcase_{stem}.png"
    fig.savefig(p, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {p}")
    return fig


def fig_case_by_surface(A, out_dir=None, dpi=None):
    """Grouped bars: each case's frequency, per surface class and the ARM cell."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    x = np.arange(len(CASE_ORDER))
    n = len(A.labels)
    width = 0.84 / n
    for s, (lab, color) in enumerate(zip(A.labels, A.colors)):
        pct = case_percent(A, s)
        ax.bar(x + (s - (n - 1) / 2) * width, pct[CASE_ORDER], width=width * 0.9,
               color=color, label=lab, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CASE_LABELS[c]}" for c in CASE_ORDER], fontsize=12)
    ax.set_xlabel("Radiative case", fontsize=10)
    ax.set_ylabel("% of that class's cell-hours", fontsize=10)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Cloud radiative cases by surface class — {A.args.region}\n"
                 f"{A.args.basis}-weighted fractions   |   phase threshold "
                 f"{A.args.phase_threshold:g} g m$^{{-3}}$   |   "
                 f"tcc $\\geq$ {A.args.min_cloud_fraction:g}", fontsize=12, pad=10)
    return _emit(fig, A, out_dir, f"by_surface_{A.args.basis}", dpi)


def fig_case_monthly(A, out_dir=None, dpi=None):
    """Each case's frequency by month, for the whole domain and the ARM cell."""
    import matplotlib.pyplot as plt

    months = sorted(A.months)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, c in zip(axes.ravel(), CASE_ORDER):
        lab = CASE_LABELS[c]
        dom, site = [], []
        for m in months:
            mm = A.months[m]
            d_num = mm[c, :len(CLASS_ORDER)].sum()
            d_den = mm[0, :len(CLASS_ORDER)].sum()
            dom.append(100.0 * d_num / d_den if d_den else np.nan)
            s_den = mm[0, A.site_code]
            site.append(100.0 * mm[c, A.site_code] / s_den if s_den else np.nan)
        xs = np.arange(len(months))
        ax.bar(xs - 0.2, dom, width=0.4, color=CASE_COLORS[lab], label="domain")
        ax.bar(xs + 0.2, site, width=0.4, color=SITE_COLOR, label="ARM cell")
        ax.set_title(f"{lab} — {CASE_DESC[lab]}", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([calendar.month_abbr[m] for m in months])
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[0].legend(fontsize=8, framealpha=0.9)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of cell-hours", fontsize=9)
    fig.suptitle(f"Cloud radiative cases by month — {A.args.region}   |   "
                 f"{A.args.basis}-weighted fractions", fontsize=12)
    return _emit(fig, A, out_dir, f"monthly_{A.args.basis}", dpi)


def fig_filter_breakdown(A, out_dir=None, dpi=None):
    """How much each quality filter passes on its own."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    keys = [k for k, _ in FILTER_ORDER]
    vals = [100.0 * A.filt_pass[k] / A.filt_total for k in keys]
    labs = [l for _, l in FILTER_ORDER]
    cols = ["#9aa5ad" if k == "clear_below" else "#1B7FBD" for k in keys]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of all cell-hours passing this filter alone", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Quality filters — {A.args.region}\n"
                 "grey = passes everything by construction, in this code and "
                 "in the ARM original", fontsize=11, pad=10)
    return _emit(fig, A, out_dir, "filters", dpi)


ALL_FIGURES = (fig_case_by_surface, fig_case_monthly, fig_filter_breakdown)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--storage", choices=sorted(STORAGE_ROOTS), default="local")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--region", default="barrow")
    p.add_argument("--basis", choices=("mass", "levels"), default="mass",
                   help="How phase fractions are measured. 'mass' is "
                        "resolution-independent and the default; 'levels' is "
                        "the literal ARM rule and is degenerate at 23 levels.")
    p.add_argument("--phase-threshold", type=float,
                   default=DEFAULT_PHASE_THRESHOLD_G_M3, metavar="G_M3")
    p.add_argument("--phase-rule", choices=("fraction", "absolute"),
                   default=DEFAULT_PHASE_RULE,
                   help="How each level's phase is decided. 'fraction' applies "
                        "the same 90/10 purity rule used at column level and is "
                        "the recommended setting; 'absolute' tests each species "
                        "against the threshold independently.")
    p.add_argument("--min-cloud-fraction", type=float,
                   default=DEFAULT_MIN_CLOUD_FRACTION)
    p.add_argument("--lsm-tol", type=float, default=1e-4)
    p.add_argument("--open-ocean-max-siconc", type=float, default=0.05)
    p.add_argument("--sea-ice-min-siconc", type=float, default=0.95)
    p.add_argument("--land-max-siconc", type=float, default=0.001)
    p.add_argument("--mask-grid", type=float, default=None,
                   help="Read the regridded mask for this spacing. "
                        "Default None = the native-grid file, which "
                        "is what is on disk.")
    p.add_argument("--jobs", type=int, default=0, metavar="N",
                   help="Worker processes for the per-file "
                        "classification. 0 = one per CPU core. "
                        "Files are independent and the reduction "
                        "is a sum, so the result does not depend "
                        "on this.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        A = prepare(args=args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    print_report(A)
    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    print()
    for fn in ALL_FIGURES:
        fn(A, out_dir=out_dir)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
