"""Sensitivity of the sky-state DLR statistics to the ERA5 time-stamp offset.

WHY THIS EXISTS
---------------
At an ERA5 valid time T the fields used to define the sky state (tcc, tclw,
tciw, siconc) are instantaneous analysis snapshots AT T (GRIB stepType
"instant"), while the precipitation filter field (tp, stepType "accum") and the
downwelling longwave (msdwlwrf / msdwlwrfcs, stepType "avg") are integrated over
the hour (T-1 h, T] (ECMWF, "ERA5: data documentation", section "Mean
rates/fluxes and accumulations",
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).
The delivered products (make_era5_hour_count_csvs.py,
make_era5_hourly_values_csvs.py) pair them on the same valid time, so the flux
window is centred 30 min BEFORE the cloud snapshot it is binned by.

This script re-does the sky-state classification and DLR binning four ways and
compares them. The four pairings are my own construction for this test, not a
published method:

  A  as delivered   state from snapshot X(T); tp(T); DLR(T) = mean over (T-1, T]
  B  centred flux   state from X(T); tp and DLR averaged over hours ending at T
                    and T+1, i.e. a 2-h window (T-1, T+1] centred on the snapshot
  C  centred state  state from the mean of the snapshots X(T-1) and X(T)
                    (trapezoid estimate of the state over the flux hour);
                    tp(T); DLR(T)
  D  persistent     state must be the same at BOTH snapshots T-1 and T;
                    tp(T); DLR(T)  (a subset of A)

The thresholds, the floors and the phase split are exactly the delivered
SETTINGS (make_era5_utqiagvik_filtered_netcdf.SETTINGS), and the surface class
of every variant is taken from siconc at T. Statistics are pooled over
(grid cell, hour) pairs; for the UTQ slot (one cell) that is simply hours.

Output: era5_time_alignment_sensitivity/summary.csv and a readME.txt, plus a
printed table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

# ov_config first: thread limits and the ERA5/surface_energy_budget sys.path.
import ov_config as ov

import numpy as np
import pandas as pd

from make_era5_utqiagvik_filtered_netcdf import HERE, SETTINGS, build_args, git_describe

from cloud_classification import fraction_phase_masks  # noqa: E402
from plot_dlr_by_phase import DEFAULT_CLEAR_TCC_MAX  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    align_lsm_to_grid,
    classify_cells,
    load_land_sea_mask,
)
import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402

OUT_PRODUCT = "era5_time_alignment_sensitivity"

# Spatial slots: the ARM cell and all five surface classes.
SLOTS: tuple[str, ...] = ("UTQ", "Land", "Coastal", "OpenOcean", "MarginalSeaIce", "PackIce")
SLOT_CLASS = {"Land": "land", "Coastal": "coastal", "OpenOcean": "open_ocean",
              "MarginalSeaIce": "marginal_ice", "PackIce": "sea_ice"}
STATES: tuple[str, ...] = ("liquid_containing", "liquid_only", "ice_only", "clear_sky")
VARIANTS: tuple[str, ...] = ("A", "B", "C", "D")
VARIANT_LABEL = {
    "A": "as delivered: state X(T), tp(T), DLR(T-1,T]",
    "B": "centred flux: state X(T), tp & DLR over (T-1,T+1]",
    "C": "centred state: state from mean of X(T-1), X(T); tp(T), DLR(T-1,T]",
    "D": "persistent: same state at T-1 and T; tp(T), DLR(T-1,T]",
}

# Histogram bins for medians. 0.25 W m-2 bins: a median is good to +-0.125 W m-2.
DLR_EDGES_W_M2 = np.arange(0.0, 500.0 + 0.25, 0.25)
CRE_EDGES_W_M2 = np.arange(-100.0, 200.0 + 0.25, 0.25)


def sky_states(tcc: np.ndarray, lwp_g_m2: np.ndarray, iwp_g_m2: np.ndarray,
               rate_mm_hr: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Sky-state masks, copied rule for rule from make_era5_hourly_values_csvs.accumulate.

    All inputs share one shape (time, lat, lon). Returns boolean masks.
    """
    valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
    # Precipitation filter: an hour AT or above the threshold is dropped.
    raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
    # Liquid only / mixed / ice only from the floored paths (same helper as delivered).
    f = fraction_phase_masks(lwp_g_m2, iwp_g_m2,
                             args.liquid_fraction_min, args.ice_fraction_min,
                             args.min_lwp, args.min_iwp)
    has_cloud = valid & (f["cwp_g"] > 0.0)
    kept = valid & (tcc >= args.min_cloud_fraction) & has_cloud & ~raining
    return {
        "liquid_containing": kept & (f["liquid"] | f["mixed"]),
        "liquid_only": kept & f["liquid"],
        "ice_only": kept & f["ice"],
        "clear_sky": valid & ~has_cloud & (tcc < DEFAULT_CLEAR_TCC_MAX) & ~raining,
    }


def hist_median(counts: np.ndarray, edges: np.ndarray) -> float:
    """Median from a histogram, by linear interpolation inside the median bin."""
    n = counts.sum()
    if n == 0:
        return np.nan
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, 0.5 * n))
    below = cum[k - 1] if k > 0 else 0
    frac = (0.5 * n - below) / counts[k] if counts[k] else 0.5
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))


def accumulate(args: argparse.Namespace, block_hours: int) -> dict:
    """Stream the record and tally counts, sums and histograms per slot/state/variant."""
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(args.region,
                           lwph.resolve_data_root(args.storage, args.data_root),
                           args.mask_grid), ds)

    # Same season bookkeeping as the delivered products.
    layout = season_layout(ds, args)
    keep_idx, used, _ = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]   # (time,)

    times = np.asarray(ds["valid_time"].values)
    n_time = times.size
    # Neighbour availability: T-1 / T+1 exist only if the archive step is exactly 1 h.
    one_h = np.timedelta64(1, "h")
    has_prev = np.zeros(n_time, dtype=bool)
    has_prev[1:] = (times[1:] - times[:-1]) == one_h
    has_next = np.zeros(n_time, dtype=bool)
    has_next[:-1] = (times[1:] - times[:-1]) == one_h

    site_mask, _, _ = site_cell_mask(ds)
    read_vars = ["tcc", args.liquid_var, "tciw", "tp", "siconc", "msdwlwrf", "msdwlwrfcs"]

    nd, nc = DLR_EDGES_W_M2.size - 1, CRE_EDGES_W_M2.size - 1
    key = [(sl, st, v) for sl in SLOTS for st in STATES for v in VARIANTS]
    n = {k: 0 for k in key}                    # cell-hours in state
    n_both_A = {k: 0 for k in key}             # ... that are also in that state under A
    s_dlr = {k: 0.0 for k in key}              # sum of DLR, W m-2
    s_cre = {k: 0.0 for k in key}              # sum of DLR - DLR_clear, W m-2
    h_dlr = {k: np.zeros(nd, dtype=np.int64) for k in key}
    h_cre = {k: np.zeros(nc, dtype=np.int64) for k in key}
    # Hours-only comparison restricted to the hours where every variant is defined
    # (needs T-1 and T+1), so A..D are compared on identical hour sets.
    n_common_hours = 0

    print(f"  Streaming {int(use_step.sum()):,} in-window hours of {n_time:,} ...", flush=True)
    for i0 in range(0, n_time, block_hours):
        i1 = min(i0 + block_hours, n_time)
        if not use_step[i0:i1].any():
            continue
        # Read one extra hour on each side so T-1 and T+1 exist at block edges.
        r0, r1 = max(i0 - 1, 0), min(i1 + 1, n_time)
        blk = ds[read_vars].isel(valid_time=slice(r0, r1)).load()
        # Positions inside blk of the block's own hours T, and of T-1 / T+1.
        idx_T = np.arange(i0, i1)
        # Keep only hours that are in the window AND have both neighbours, so all
        # four variants are evaluated on exactly the same hours.
        ok = use_step[idx_T] & has_prev[idx_T] & has_next[idx_T]
        if not ok.any():
            continue
        tT = idx_T[ok] - r0
        tM = tT - 1                                               # T-1
        tP = tT + 1                                               # T+1
        n_common_hours += tT.size

        def g(name: str, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
            return blk[name].values[t].astype(np.float64) * scale   # (t, lat, lon)

        # Snapshots at T and T-1 (paths kg m-2 -> g m-2).
        tcc_T, tcc_M = g("tcc", tT), g("tcc", tM)
        lwp_T, lwp_M = g(args.liquid_var, tT, 1e3), g(args.liquid_var, tM, 1e3)
        iwp_T, iwp_M = g("tciw", tT, 1e3), g("tciw", tM, 1e3)
        # tp is metres per hour-ending-at-stamp: x1000 -> mm hr-1.
        tp_T, tp_P = g("tp", tT, 1e3), g("tp", tP, 1e3)
        dlr_T, dlr_P = g("msdwlwrf", tT), g("msdwlwrf", tP)
        cs_T, cs_P = g("msdwlwrfcs", tT), g("msdwlwrfcs", tP)

        # Surface class from siconc at T for every variant.
        classes = classify_cells(lsm, blk["siconc"].values[tT], args.lsm_tol,
                                 args.open_ocean_max_siconc, args.sea_ice_min_siconc,
                                 args.land_max_siconc)
        slot_mask = {"UTQ": np.broadcast_to(site_mask, classes.shape)}
        for sl in SLOTS[1:]:
            slot_mask[sl] = classes == CLASS_CODES[SLOT_CLASS[sl]]

        # --- the four pairings -------------------------------------------------
        st_A = sky_states(tcc_T, lwp_T, iwp_T, tp_T, args)
        tp_c = 0.5 * (tp_T + tp_P)                     # mean rate over (T-1, T+1]
        st_B = sky_states(tcc_T, lwp_T, iwp_T, tp_c, args)
        st_C = sky_states(0.5 * (tcc_T + tcc_M), 0.5 * (lwp_T + lwp_M),
                          0.5 * (iwp_T + iwp_M), tp_T, args)
        st_M = sky_states(tcc_M, lwp_M, iwp_M, tp_T, args)   # state at T-1 (tp(T) as in A)
        st_D = {s: st_A[s] & st_M[s] for s in STATES}
        states = {"A": st_A, "B": st_B, "C": st_C, "D": st_D}
        dlr = {"A": dlr_T, "B": 0.5 * (dlr_T + dlr_P), "C": dlr_T, "D": dlr_T}
        cre = {"A": dlr_T - cs_T, "B": 0.5 * ((dlr_T - cs_T) + (dlr_P - cs_P)),
               "C": dlr_T - cs_T, "D": dlr_T - cs_T}

        for sl in SLOTS:
            sm = slot_mask[sl]
            for st in STATES:
                a_sel = st_A[st] & sm
                for v in VARIANTS:
                    sel = states[v][st] & sm & np.isfinite(dlr[v]) & np.isfinite(cre[v])
                    k = (sl, st, v)
                    n[k] += int(sel.sum())
                    n_both_A[k] += int((sel & a_sel).sum())
                    x, c = dlr[v][sel], cre[v][sel]
                    s_dlr[k] += float(x.sum())
                    s_cre[k] += float(c.sum())
                    h_dlr[k] += np.histogram(x, DLR_EDGES_W_M2)[0]
                    h_cre[k] += np.histogram(c, CRE_EDGES_W_M2)[0]
        print(f"    {str(times[i0])[:10]} .. {str(times[i1 - 1])[:10]}", flush=True)

    rows = []
    for (sl, st, v) in key:
        k = (sl, st, v)
        kA = (sl, st, "A")
        rows.append(dict(
            slot=sl, state=st, variant=v,
            cell_hours=n[k],
            cell_hours_change_vs_A_pct=100.0 * (n[k] - n[kA]) / n[kA] if n[kA] else np.nan,
            # Of the cell-hours in this state under A, the share also in it under v.
            pct_of_A_retained=100.0 * n_both_A[k] / n[kA] if n[kA] else np.nan,
            mean_dlr_W_m2=s_dlr[k] / n[k] if n[k] else np.nan,
            median_dlr_W_m2=hist_median(h_dlr[k], DLR_EDGES_W_M2),
            mean_lw_cre_W_m2=s_cre[k] / n[k] if n[k] else np.nan,
            median_lw_cre_W_m2=hist_median(h_cre[k], CRE_EDGES_W_M2),
        ))
    df = pd.DataFrame(rows)
    # Differences from A, the delivered pairing.
    base = df[df.variant == "A"].set_index(["slot", "state"])
    for col in ("mean_dlr_W_m2", "median_dlr_W_m2", "mean_lw_cre_W_m2", "median_lw_cre_W_m2"):
        df[f"d_{col}_vs_A"] = [getattr(r, col) - base.loc[(r.slot, r.state), col]
                               for r in df.itertuples()]
    return {"df": df, "seasons": list(used), "n_common_hours": n_common_hours,
            "n_window_hours": int(use_step.sum())}


def write_readme(R: dict, args: argparse.Namespace, out_dir: Path) -> Path:
    lines = [
        "ERA5 TIME-ALIGNMENT SENSITIVITY OF SKY-STATE DLR STATISTICS",
        "=" * 59, "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by "
        f"era5_time_alignment_sensitivity.py (repo {git_describe()})", "",
        "Cloud-state fields (tcc, tclw, tciw, siconc) are snapshots at the valid",
        "time T; tp and msdwlwrf/msdwlwrfcs are integrated over (T-1 h, T].",
        "Variants (my own construction for this test):", "",
    ]
    lines += [f"  {v}  {VARIANT_LABEL[v]}" for v in VARIANTS]
    lines += [
        "",
        f"Seasons {R['seasons'][0]} .. {R['seasons'][-1]}, Oct 1 - Mar 31. All variants use",
        f"the same {R['n_common_hours']:,} hours (of {R['n_window_hours']:,} in window): those with",
        "an archive hour on each side. Thresholds as delivered: tcc >= "
        f"{args.min_cloud_fraction}, floors {args.min_lwp}/{args.min_iwp} g m-2,",
        f"tp < {args.precip_rate_max} mm hr-1, phase share {args.liquid_fraction_min}.",
        "Statistics pool (cell, hour) pairs, unweighted. LW CRE = msdwlwrf - msdwlwrfcs.",
        "Medians from 0.25 W m-2 histograms (+-0.125 W m-2).", "",
        "Columns: cell_hours; cell_hours_change_vs_A_pct; pct_of_A_retained (share of",
        "A's cell-hours in that state also in it under the variant); mean/median DLR",
        "and LW CRE; d_*_vs_A = variant minus A.",
    ]
    p = out_dir / "readME.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--block-hours", type=int, default=720,
                   help="hours read per block (default 720)")
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    out_dir = opts.out_dir or (HERE / OUT_PRODUCT if run.is_baseline
                               else run.out_dir(OUT_PRODUCT))
    out_dir.mkdir(parents=True, exist_ok=True)
    args = build_args({**SETTINGS, **run.settings()})
    print(run.report())
    R = accumulate(args, opts.block_hours)
    df = R["df"]
    df.to_csv(out_dir / "summary.csv", index=False, float_format="%.3f")
    write_readme(R, args, out_dir)
    show = df[["slot", "state", "variant", "cell_hours", "cell_hours_change_vs_A_pct",
               "pct_of_A_retained", "median_dlr_W_m2", "d_median_dlr_W_m2_vs_A",
               "d_mean_dlr_W_m2_vs_A", "d_median_lw_cre_W_m2_vs_A"]]
    with pd.option_context("display.width", 220, "display.max_rows", 500,
                           "display.float_format", "{:.2f}".format):
        print(show.to_string(index=False))
    print(f"\n  -> {out_dir}")


if __name__ == "__main__":
    main()
