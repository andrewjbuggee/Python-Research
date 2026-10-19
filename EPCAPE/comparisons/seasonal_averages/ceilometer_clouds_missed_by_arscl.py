#!/usr/bin/env python
"""How often does the Pier ceilometer see a cloud that the KAZR-ARSCL layer product misses, and what are those clouds?

Why: Section 3d of check_seasonal_averages.ipynb applies "CTH < 2 km" with the KAZR-ARSCL cloud tops. ARSCL's
cloud_layer_top_height has no layer (flag -1, "clear_sky") at about a third of the times the ceilometer sees low
cloud, so how those samples are treated moves the GOES-like LWP by 20-35 g m-2. This script measures how often it
happens and describes the clouds involved.

Data (EPCAPE products in config.yaml; 15 Feb 2023 - 14 Feb 2024, UTC):
  cbh_ceil_M1     Pier ceilometer epcceilM1.b1, 16 s: first_cbh (m above ground, lowest base) with qc_first_cbh, and
                  detection_status (0 no cloud, 1-3 one to three bases, 4 obscured without a base, 5 some
                  obscuration judged transparent). Its bases reach at most 7,620 m in this record.
  cloud_arscl_M1  KAZR-ARSCL epcarsclkazr1kolliasM1.c1, 4 s: cloud_layer_base_height and cloud_layer_top_height
                  (time, layer; up to 10 hydrometeor layers from the radar + micropulse-lidar mask; -1 = no layer;
                  no qc_ fields) and cloud_base_best_estimate (ceilometer + lidar best-estimate base).
  lwp_mwrret2_M1  MWRRET v2 phys_lwp (3-channel MWR), qc_phys_lwp = 0 and phys_qc_flag = 0, ~1.3 s.
  rain_ld_M1      Pier Parsivel precip_rate (QC = 0), 1 min: whether a sample is within ±RAIN_WINDOW_MIN of rain.

Definitions (one row per ceilometer sample):
  ceilometer cloud   detection_status 1-3 and a QC = 0 first_cbh (at or below --max-cbh-m, if given).
  ARSCL sample       the nearest ARSCL sample within ±ARSCL_TOLERANCE_S (its 4-s sampling interval).
  detected           an ARSCL layer spans the ceilometer base: layer base - BASE_TOLERANCE_M <= first_cbh <=
                     layer top + BASE_TOLERANCE_M, for any of the 10 layers.
  missed             no such layer. Two kinds:
                       "no ARSCL layer at all"        every layer is -1;
                       "ARSCL layer elsewhere only"   layers exist, but none at the ceilometer's base (e.g. only an
                                                      upper layer the radar saw through a thin low cloud).
Characteristics compared for missed vs detected samples:
  (a) monthly share missed, split into its two kinds;
  (b) share missed by hour of the day (local standard time, UTC-8; daylight time is ignored);
  (c) share missed by ceilometer base height (P(missed | base));
  (d) distribution of the ceilometer base height;
  (e) distribution of the MWR LWP at the same time (nearest MWRRET v2 sample within ±MWR_TOLERANCE_S);
  (f) distribution of the hour's ceilometer cloud fraction (any base height): broken vs overcast cloud.
The share near rain (±RAIN_WINDOW_MIN of Pier Parsivel rain) and ARSCL's own cloud_base_best_estimate at the missed
samples are printed and written to the summary CSV.

Background (for reading the plots; not a result of this script): KAZR is a 35-GHz (Ka-band) cloud radar. In the
Rayleigh regime the radar reflectivity factor is Z = sum of N(D) D^6, so thin clouds of small droplets without drizzle
can fall below its sensitivity, while the micropulse lidar is extinguished within the first few hundred metres of a
cloud. ARSCL's radar range-gate centres start at 160 m above ground (file height coordinate), so the lowest bases sit
at the bottom of the radar's range. No reference for KAZR's sensitivity at EPCAPE was looked up: treat this
explanation of the misses as an inference to be tested by the panels, not as a sourced fact.

Outputs (in arscl_missed_clouds/, next to this script; --out chooses another folder; <tag> is empty for all base
heights, or _cbh<H>m with --max-cbh-m H):
  ceilometer_clouds_missed_by_arscl<tag>.png / .pdf    the 6-panel figure
  ceilometer_clouds_missed_by_arscl_summary<tag>.csv   overall and seasonal shares
  ceilometer_clouds_missed_by_arscl_by_month<tag>.csv, _by_base_height<tag>.csv, _by_local_hour<tag>.csv

Examples (from the EPCAPE folder)
  python comparisons/seasonal_averages/ceilometer_clouds_missed_by_arscl.py
  python comparisons/seasonal_averages/ceilometer_clouds_missed_by_arscl.py --max-cbh-m 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402

from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.plotting import GRID, INK_2, INK_3, use_style  # noqa: E402

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
ARSCL_TOLERANCE_S = 4.0  # nearest ARSCL sample within ± this (ARSCL samples every 4 s)
MWR_TOLERANCE_S = 2.0  # nearest MWRRET v2 sample within ± this (~1.3-s samples)
BASE_TOLERANCE_M = 200.0  # an ARSCL layer "spans" the ceilometer base if within ± this of the layer's base..top
RAIN_WINDOW_MIN = 30.0  # "near rain" = Pier Parsivel rain within ± this many minutes (the LWP rain screen)
LOCAL_OFFSET_H = -8  # local standard time at the Pier (PST = UTC-8)
CBH_BINS_M = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7700]  # panel (c)
MISSED_COLOR = "#4a3aa7"  # violet: missed by ARSCL (the focus group)
MISSED_COLOR_LIGHT = "#9085e9"  # lighter violet: the "layer elsewhere only" part of the missed share
DETECTED_COLOR = INK_3  # neutral grey: detected by ARSCL (the reference group)
AXIS_LABEL_FONTSIZE = 12


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def nearest_index(t_ref: np.ndarray, t_src: np.ndarray, tol_s: float) -> np.ndarray:
    """Index into sorted t_src of the sample nearest each t_ref (both datetime64[ns]); -1 if none within ±tol_s."""
    ref = t_ref.astype("datetime64[ns]").astype(np.int64)
    src = t_src.astype("datetime64[ns]").astype(np.int64)
    k = np.searchsorted(src, ref)
    k0, k1 = np.clip(k - 1, 0, src.size - 1), np.clip(k, 0, src.size - 1)
    pick = np.where(np.abs(src[k1] - ref) < np.abs(src[k0] - ref), k1, k0)
    return np.where(np.abs(src[pick] - ref) <= tol_s * 1e9, pick, -1)


def ceilometer_samples(max_cbh_m: Optional[float]) -> pd.DataFrame:
    """Every ceilometer sample with a QC'd cloud base (columns cbh_m, status), plus the hour's cloud fraction."""
    ceil = load_product("cbh_ceil_M1")
    status = ceil["detection_status"].values
    cbh = Q.qc0_values(ceil, "first_cbh")[0].values  # m above ground; NaN where qc_first_cbh != 0
    times = pd.DatetimeIndex(ceil["time"].values)
    # the hour's cloud fraction (any base height): fraction of its known samples with a base (Q.low_cloud_fraction)
    any_cloud = Q.ceilometer_low_cloud_state(ceil, max_cbh_m=np.inf)
    hourly_cf = Q.low_cloud_fraction(any_cloud, window_min=60.0, min_coverage=0.5)["low_cloud_fraction"]
    cloud = np.isin(status, [1, 2, 3]) & np.isfinite(cbh)
    if max_cbh_m is not None:
        cloud &= cbh <= max_cbh_m
    out = pd.DataFrame({"cbh_m": cbh[cloud], "status": status[cloud]}, index=times[cloud])
    out["hour_cloud_fraction"] = hourly_cf.reindex(out.index.floor("60min")).to_numpy()
    n_known = int((np.isin(status, [0, 5]) | (np.isin(status, [1, 2, 3]) & np.isfinite(cbh))).sum())
    print(f"Ceilometer: {len(times):,} samples, {n_known:,} with a known state, {len(out):,} with a QC'd cloud base"
          + (f" at or below {max_cbh_m:g} m" if max_cbh_m is not None else ""))
    return out


def attach_arscl(samples: pd.DataFrame) -> pd.DataFrame:
    """ARSCL at each ceilometer cloud sample: detected / missed (and how), its best-estimate base."""
    arscl = load_product("cloud_arscl_M1")
    idx = nearest_index(samples.index.values, arscl["time"].values, ARSCL_TOLERANCE_S)
    has = idx >= 0
    base2 = arscl["cloud_layer_base_height"].values[idx[has]]  # (n, 10) m above ground; -1 = no layer
    top2 = arscl["cloud_layer_top_height"].values[idx[has]]
    cbbe = arscl["cloud_base_best_estimate"].values[idx[has]]
    del arscl  # ~725 MB
    cbh = samples["cbh_m"].to_numpy()[has][:, None]
    layer = (base2 >= 0) & (top2 >= 0)
    spans = layer & (base2 - BASE_TOLERANCE_M <= cbh) & (cbh <= top2 + BASE_TOLERANCE_M)
    kind = np.full(len(samples), "no ARSCL sample", dtype=object)
    kind_has = np.where(spans.any(axis=1), "detected",
                        np.where(layer.any(axis=1), "ARSCL layer elsewhere only", "no ARSCL layer at all"))
    kind[has] = kind_has
    out = samples.copy()
    out["arscl"] = kind
    out["missed"] = np.isin(kind, ["ARSCL layer elsewhere only", "no ARSCL layer at all"])
    best = np.full(len(samples), np.nan)
    best[has] = cbbe
    out["arscl_best_estimate_base_m"] = best  # ARSCL's ceilometer + lidar base (-1 = clear)
    return out


def attach_mwr_and_rain(samples: pd.DataFrame) -> pd.DataFrame:
    """MWRRET v2 LWP (QC'd, nearest sample) and whether Pier Parsivel rain is within ±RAIN_WINDOW_MIN."""
    ds = load_product("lwp_mwrret2_M1")
    lwp, _ = Q.lwp_qc_rain_free_gm2(ds, pd.Series(dtype=float), rain_window_min=0)  # QC only; no rain screen
    del ds
    idx = nearest_index(samples.index.values, lwp.index.values, MWR_TOLERANCE_S)
    vals = np.full(len(samples), np.nan)
    vals[idx >= 0] = lwp.to_numpy()[idx[idx >= 0]]
    out = samples.copy()
    out["lwp_gm2"] = vals
    rain = Q.to_series(Q.qc0_values(load_product("rain_ld_M1"), "precip_rate")[0])
    out["near_rain"] = Q.near_rain(rain, out.index, window_min=RAIN_WINDOW_MIN)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def share_table(d: pd.DataFrame, key) -> pd.DataFrame:
    """Per group: n cloud samples with an ARSCL sample, share missed (%), and its two kinds (%)."""
    d = d[d["arscl"] != "no ARSCL sample"]
    g = d.groupby(key, observed=True)
    out = pd.DataFrame({
        "n_cloud_samples": g.size(),
        "missed_pct": 100 * g["missed"].mean(),
        "no_layer_at_all_pct": 100 * g["arscl"].apply(lambda s: (s == "no ARSCL layer at all").mean()),
        "layer_elsewhere_only_pct": 100 * g["arscl"].apply(lambda s: (s == "ARSCL layer elsewhere only").mean()),
    })
    return out


def summary_table(d: pd.DataFrame) -> pd.DataFrame:
    """Overall and seasonal shares, and the medians of each characteristic for missed vs detected samples."""
    d = d[d["arscl"] != "no ARSCL sample"]
    rows = {}
    for season in S.SEASON_ORDER:
        s = d[S.season_mask(d.index, season)]
        m, det = s[s["missed"]], s[~s["missed"]]
        rows[season] = {
            "ceilometer cloud samples (with an ARSCL sample)": len(s),
            "missed by ARSCL (%)": 100 * s["missed"].mean(),
            "  no ARSCL layer at all (%)": 100 * (s["arscl"] == "no ARSCL layer at all").mean(),
            "  ARSCL layer elsewhere only (%)": 100 * (s["arscl"] == "ARSCL layer elsewhere only").mean(),
            "missed: ARSCL best-estimate base present (%)": 100 * (m["arscl_best_estimate_base_m"] >= 0).mean(),
            "median base, missed / detected (m)": f"{m['cbh_m'].median():.0f} / {det['cbh_m'].median():.0f}",
            "median MWR LWP, missed / detected (g m-2)": f"{m['lwp_gm2'].median():.1f} / {det['lwp_gm2'].median():.1f}",
            "median hourly cloud fraction, missed / detected": (f"{m['hour_cloud_fraction'].median():.2f} / "
                                                                f"{det['hour_cloud_fraction'].median():.2f}"),
            f"near rain (±{RAIN_WINDOW_MIN:g} min), missed / detected (%)": (f"{100 * m['near_rain'].mean():.1f} / "
                                                                            f"{100 * det['near_rain'].mean():.1f}"),
        }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def step_hist(ax, values, bins, color, label, lw=1.6):
    """Normalized histogram (fraction of samples per bin) as a step line; the median is marked."""
    v = values[np.isfinite(values)]
    counts, edges = np.histogram(np.clip(v, bins[0], bins[-1]), bins=bins)  # values beyond the range go to the end bins
    ax.stairs(counts / max(v.size, 1), edges, color=color, lw=lw, label=f"{label} (n = {v.size:,})")
    ax.axvline(np.median(v), color=color, lw=1.0, ls="--")


def plot(d: pd.DataFrame, monthly: pd.DataFrame, by_hour: pd.DataFrame, by_cbh: pd.DataFrame, title_note: str):
    use_style()
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    d = d[d["arscl"] != "no ARSCL sample"]
    missed, detected = d[d["missed"]], d[~d["missed"]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12.5))
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axes

    # (a) monthly share missed, stacked by kind
    x = np.arange(len(monthly))
    ax_a.bar(x, monthly["no_layer_at_all_pct"], color=MISSED_COLOR, width=0.75, label="no ARSCL layer at all")
    ax_a.bar(x, monthly["layer_elsewhere_only_pct"], bottom=monthly["no_layer_at_all_pct"], color=MISSED_COLOR_LIGHT,
             width=0.75, label="ARSCL layer elsewhere only")
    ax_a.set_xticks(x, [t.strftime("%b\n%Y") if (t.month in (1, 2) or i == 0) else t.strftime("%b")
                        for i, t in enumerate(monthly.index)], fontsize=8.5)
    ax_a.set_ylabel("ceilometer clouds missed by ARSCL (%)")
    ax_a.set_title("(a) By month", loc="left")
    ax_a.legend(loc="upper left")
    ax_a.set_ylim(0, max(100, monthly["missed_pct"].max() * 1.1))

    # (b) by local hour
    ax_b.plot(by_hour.index, by_hour["missed_pct"], color=MISSED_COLOR, marker="o", ms=5)
    ax_b.set_xticks(range(0, 24, 3))
    ax_b.set_xlim(-0.5, 23.5)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("hour of day (local standard time, UTC−8)")
    ax_b.set_ylabel("missed by ARSCL (%)")
    ax_b.set_title("(b) By time of day", loc="left")

    # (c) P(missed | base height)
    xc = np.arange(len(by_cbh))
    ax_c.bar(xc, by_cbh["missed_pct"], color=MISSED_COLOR, width=0.75)
    for xi, (pct, n) in enumerate(zip(by_cbh["missed_pct"], by_cbh["n_cloud_samples"])):
        ax_c.text(xi, pct + 1.5, f"{n / 1000:.0f}k" if n >= 1000 else f"{n}", ha="center", va="bottom",
                  fontsize=7.5, color=INK_2)
    ax_c.set_xticks(xc, [f"{int(iv.left)}–{int(iv.right)}" for iv in by_cbh.index], rotation=45, ha="right", fontsize=8.5)
    ax_c.set_ylim(0, 105)
    ax_c.set_xlabel("ceilometer cloud base (m above ground)")
    ax_c.set_ylabel("missed by ARSCL (%)")
    ax_c.set_title("(c) By cloud-base height (labels: samples)", loc="left")

    # (d) base-height distributions
    bins_d = np.arange(0, 3001, 50)
    step_hist(ax_d, detected["cbh_m"].to_numpy(), bins_d, DETECTED_COLOR, "detected by ARSCL")
    step_hist(ax_d, missed["cbh_m"].to_numpy(), bins_d, MISSED_COLOR, "missed by ARSCL")
    ax_d.set_xlabel("ceilometer cloud base (m above ground; > 3 km in the last bin)")
    ax_d.set_ylabel("fraction of samples per 50 m")
    ax_d.set_title("(d) Cloud-base height (dashed: medians)", loc="left")
    ax_d.legend(loc="upper right")

    # (e) MWR LWP distributions
    bins_e = np.arange(-20, 401, 5)
    step_hist(ax_e, detected["lwp_gm2"].to_numpy(), bins_e, DETECTED_COLOR, "detected by ARSCL")
    step_hist(ax_e, missed["lwp_gm2"].to_numpy(), bins_e, MISSED_COLOR, "missed by ARSCL")
    ax_e.set_xlabel("MWRRET v2 LWP (g m⁻²; beyond −20 or 400 in the end bins)")
    ax_e.set_ylabel("fraction of samples per 5 g m⁻²")
    ax_e.set_title("(e) MWR liquid water path (dashed: medians)", loc="left")
    ax_e.legend(loc="upper right")

    # (f) the hour's cloud fraction
    bins_f = np.linspace(0, 1, 21)
    step_hist(ax_f, detected["hour_cloud_fraction"].to_numpy(), bins_f, DETECTED_COLOR, "detected by ARSCL")
    step_hist(ax_f, missed["hour_cloud_fraction"].to_numpy(), bins_f, MISSED_COLOR, "missed by ARSCL")
    ax_f.set_xlabel("ceilometer cloud fraction of the hour (any base height)")
    ax_f.set_ylabel("fraction of samples per 0.05")
    ax_f.set_title("(f) How broken the cloud is (dashed: medians)", loc="left")
    ax_f.legend(loc="upper left")

    n_all = len(d)
    fig.suptitle(f"Ceilometer clouds that the KAZR-ARSCL layer product misses: {100 * d['missed'].mean():.0f} % of "
                 f"{n_all:,} ceilometer cloud samples{title_note}", fontsize=12.5, fontweight="semibold", y=0.995)
    fig.text(0.01, 0.003,
             f"Missed = no ARSCL layer (cloud_layer_base/top_height) spans the ceilometer's first_cbh within "
             f"±{BASE_TOLERANCE_M:g} m (nearest 4-s ARSCL sample). Ceilometer: detection_status 1-3, qc_first_cbh = 0. "
             f"MWR: MWRRET v2 phys_lwp, QC = 0 and phys_qc_flag = 0, nearest sample within ±{MWR_TOLERANCE_S:g} s. "
             f"15 Feb 2023 - 14 Feb 2024.", fontsize=8, color=INK_2, ha="left", va="bottom", wrap=True)
    for ax in axes.flat:
        ax.grid(True, color=GRID)
        ax.tick_params(colors=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-cbh-m", type=float, default=None,
                        help="only ceilometer bases at or below this height (default: every height)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "arscl_missed_clouds",
                        help="output folder (default: arscl_missed_clouds/, next to this script)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_cbh{args.max_cbh_m:g}m" if args.max_cbh_m is not None else ""
    title_note = f" (bases ≤ {args.max_cbh_m:g} m)" if args.max_cbh_m is not None else ""

    d = attach_mwr_and_rain(attach_arscl(ceilometer_samples(args.max_cbh_m)))
    print(f"  no ARSCL sample within ±{ARSCL_TOLERANCE_S:g} s: {(d['arscl'] == 'no ARSCL sample').sum():,}")

    monthly = share_table(d, d.index.to_period("M").to_timestamp())
    local_hour = (d.index + pd.Timedelta(hours=LOCAL_OFFSET_H)).hour
    by_hour = share_table(d, local_hour)
    by_cbh = share_table(d, pd.cut(d["cbh_m"], CBH_BINS_M))
    summary = summary_table(d)

    fig = plot(d, monthly, by_hour, by_cbh, title_note)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"ceilometer_clouds_missed_by_arscl{tag}.{ext}", dpi=200)
    plt.close(fig)
    summary.to_csv(args.out / f"ceilometer_clouds_missed_by_arscl_summary{tag}.csv")
    monthly.to_csv(args.out / f"ceilometer_clouds_missed_by_arscl_by_month{tag}.csv", float_format="%.2f")
    by_cbh.to_csv(args.out / f"ceilometer_clouds_missed_by_arscl_by_base_height{tag}.csv", float_format="%.2f")
    by_hour.to_csv(args.out / f"ceilometer_clouds_missed_by_arscl_by_local_hour{tag}.csv", float_format="%.2f")
    print(f"Wrote ceilometer_clouds_missed_by_arscl{tag}.png / .pdf and CSVs to {args.out}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print("\nSummary:")
    print(summary.round(1).to_string())
    print("\nBy cloud-base height:")
    print(by_cbh.round(1).to_string())
    print("\nBy month:")
    print(monthly.round(1).to_string())


if __name__ == "__main__":
    main()
