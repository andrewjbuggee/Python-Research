"""Figures for the MFRSR / sunphotometer / MWR cloud-property comparison.

Conventions (EPCAPE.plotting): instrument colours are fixed (MFRSR blue,
SPHOT orange, MWR aqua). Paired scatter points are neutral, or coloured by
a third variable on a single-hue sequential scale. Text stays in neutral ink.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from EPCAPE.plotting import (
    COLORS,
    GRID,
    INK,
    INK_2,
    INK_3,
    concise_dates,
    legend_above,
    note,
    one_to_one,
    plain_log_axis,
)
from EPCAPE.analysis_tools.stats import format_stats, paired_stats

# Single-hue sequential ramp (light -> dark blue) for colouring points by a magnitude.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)


# -- coverage ------------------------------------------------------------------
def plot_daily_coverage(counts: Mapping[str, pd.Series], units: Mapping[str, str]):
    """Small multiples: valid samples per day for each product (own y-axis each,
    because 20 s and 10 min sampling are not comparable on one scale)."""
    fig, axes = plt.subplots(len(counts), 1, figsize=(10, 1.6 * len(counts) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, counts.items()):
        ax.bar(s.index, s.values, width=0.8, color=COLORS[name], linewidth=0)
        ax.set_ylabel(units.get(name, "samples / day"), fontsize=8.5)
        ax.set_title(name, loc="left", fontsize=10)
        days_with = int((s > 0).sum())
        ax.set_title(
            f"{days_with} of {s.size} days have valid data",
            loc="right",
            fontsize=8.5,
            color=INK_2,
            fontweight="normal",
        )
    concise_dates(axes[-1])
    fig.tight_layout()
    return fig


# -- example day -----------------------------------------------------------------
def _valid_and_rejected(ax, t, all_values, valid_values, color, label, ms=2.5):
    """Rejected samples as small grey dots, valid ones in the instrument colour."""
    rejected = np.isfinite(all_values) & ~np.isfinite(valid_values)
    ax.plot(
        t[rejected],
        all_values[rejected],
        ".",
        color=INK_3,
        ms=ms,
        alpha=0.5,
        zorder=1,
        label=f"{label} (rejected)",
    )
    ax.plot(t, valid_values, ".", color=color, ms=ms, zorder=2, label=label)


def plot_example_day(day: str, mfrsr, mfrsr_tau_ok, mfrsr_re_ok, sphot, sphot_ok, mwr, mwr_ok, matched=None):
    """Four stacked panels for one UTC day: tau, r_e, LWP, cloud fraction.

    Rejected samples are drawn in grey so the effect of each product's
    filtering is visible."""
    t0, t1 = np.datetime64(day), np.datetime64(day) + np.timedelta64(1, "D")
    m = mfrsr.sel(time=slice(t0, t1))
    s = sphot.sel(time=slice(t0, t1))
    w = mwr.sel(time=slice(t0, t1))
    m_tau_ok, m_re_ok = mfrsr_tau_ok.sel(time=slice(t0, t1)), mfrsr_re_ok.sel(time=slice(t0, t1))
    s_ok, w_ok = sphot_ok.sel(time=slice(t0, t1)), mwr_ok.sel(time=slice(t0, t1))

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 2.2, 3, 1.3]}
    )
    tm, ts, tw = m["time"].values, s["time"].values, w["time"].values

    # (a) optical depth
    ax = axes[0]
    _valid_and_rejected(
        ax, tm, m["tau"].values, m["tau"].where(m_tau_ok).values, COLORS["MFRSR"], "MFRSR 20 s"
    )
    sv = s["tau"].where(s_ok)
    ax.errorbar(
        ts,
        sv.values,
        yerr=s["tau_std"].where(s_ok).values if "tau_std" in s else None,
        fmt="o",
        ms=6,
        color=COLORS["SPHOT"],
        mec="white",
        mew=1,
        elinewidth=1,
        capsize=0,
        zorder=4,
        label="SPHOT (± std)",
    )
    if matched is not None:
        mm = matched.sel(time=slice(t0, t1))
        ax.plot(
            mm["time"].values,
            mm["mfrsr_tau"].values,
            "s",
            ms=5,
            mfc="white",
            mec=COLORS["MFRSR"],
            mew=1.5,
            zorder=3,
            label="MFRSR window mean",
        )
    ax.set_ylabel("cloud optical depth τ")
    tau_all = np.concatenate([m["tau"].values.ravel(), s["tau"].values.ravel()])
    if np.any(np.isfinite(tau_all) & (tau_all > 0)):  # a log axis needs at least one positive value
        ax.set_yscale("log")
        plain_log_axis(ax, "y")
    legend_above(ax, ncol=5, fontsize=8)

    # (b) effective radius
    ax = axes[1]
    _valid_and_rejected(
        ax, tm, m["r_e_um"].values, m["r_e_um"].where(m_re_ok).values, COLORS["MFRSR"], "MFRSR (τ + MWR LWP)"
    )
    ax.errorbar(
        ts,
        s["r_e_um"].where(s_ok).values,
        yerr=s["r_e_std_um"].where(s_ok).values if "r_e_std_um" in s else None,
        fmt="o",
        ms=6,
        color=COLORS["SPHOT"],
        mec="white",
        mew=1,
        elinewidth=1,
        zorder=4,
        label="SPHOT (± std)",
    )
    ax.set_ylabel("r_e (µm)")
    # Limit the axis to the valid values: rejected retrievals (e.g. thin cloud
    # with tiny tau) can reach tens of um and would flatten everything else.
    valid_re = np.concatenate([m["r_e_um"].where(m_re_ok).values, s["r_e_um"].where(s_ok).values])
    top = np.nanpercentile(valid_re, 99) * 1.3 if np.isfinite(valid_re).any() else 25.0
    ax.set_ylim(0, max(20.0, top))
    legend_above(ax, ncol=3, fontsize=8)

    # (c) liquid water path
    ax = axes[2]
    _valid_and_rejected(
        ax, tw, w["lwp_gm2"].values, w["lwp_gm2"].where(w_ok).values, COLORS["MWR"], "MWR", ms=1.5
    )
    if "lwp_gm2" in m:
        ax.plot(
            tm,
            m["lwp_gm2"].where(m_tau_ok & ~m["r_e_from_mwr"]).values,
            ".",
            ms=2.5,
            color=COLORS["MFRSR"],
            label="MFRSR (τ × assumed r_e = 8 µm)",
        )
    ax.errorbar(
        ts,
        s["lwp_gm2"].where(s_ok).values,
        yerr=s["lwp_std_gm2"].where(s_ok).values if "lwp_std_gm2" in s else None,
        fmt="o",
        ms=6,
        color=COLORS["SPHOT"],
        mec="white",
        mew=1,
        elinewidth=1,
        zorder=4,
        label="SPHOT (2/3 ρ τ r_e)",
    )
    ax.axhspan(-30, 30, color=GRID, zorder=0, lw=0)
    ax.text(
        0.005, 0.04, "shaded: ±30 g m⁻² MWR retrieval RMS", transform=ax.transAxes, fontsize=7.5, color=INK_2
    )
    ax.set_ylabel("LWP (g m⁻²)")
    legend_above(ax, ncol=4, fontsize=8)

    # (d) cloud fraction (MFRSR ancillary)
    ax = axes[3]
    if "cloud_fraction" in m:
        ax.plot(tm, m["cloud_fraction"].values, "-", lw=1, color=INK_2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("cloud\nfraction")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("time (UTC)")
    fig.suptitle(f"{day} (UTC)", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig


# -- paired comparison -------------------------------------------------------------
def scatter_compare(
    ax,
    x,
    y,
    *,
    xerr=None,
    yerr=None,
    xlabel="",
    ylabel="",
    log=False,
    units="",
    color_by=None,
    color_label="",
    vmin=None,
    vmax=None,
    lims=None,
    max_points: Optional[int] = None,
    seed: int = 0,
):
    """y against reference x with 1:1 line, RMA fit and statistics box.

    Statistics always use every valid pair. With max_points, only a random
    subset (numpy Generator seeded with `seed`) is drawn, so tens of
    thousands of 20 s pairs do not hide each other.

    Returns the statistics dict (EPCAPE.analysis_tools.stats.paired_stats)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log:
        ok &= (x > 0) & (y > 0)
    st = paired_stats(x, y, log=log)
    if ok.sum() == 0:
        ax.text(0.5, 0.5, "no matched pairs", transform=ax.transAxes, ha="center", color=INK_2)
        return st
    if max_points is not None and ok.sum() > max_points:
        idx = np.flatnonzero(ok)
        keep = np.random.default_rng(seed).choice(idx, size=max_points, replace=False)
        ok = np.zeros_like(ok)
        ok[keep] = True
        st["plotted_subset"] = f"{max_points:,} of {st['n']:,} pairs drawn (seed {seed})"
    xe = None if xerr is None else np.asarray(xerr, float)[ok]
    ye = None if yerr is None else np.asarray(yerr, float)[ok]
    if xe is not None or ye is not None:
        ax.errorbar(x[ok], y[ok], xerr=xe, yerr=ye, fmt="none", ecolor=GRID, elinewidth=0.8, zorder=1)
    if color_by is not None:
        c = np.asarray(color_by, float)[ok]
        sc = ax.scatter(
            x[ok],
            y[ok],
            c=c,
            cmap=SEQ_BLUE,
            vmin=vmin,
            vmax=vmax,
            s=22,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(color_label, fontsize=8.5)
        cb.outline.set_visible(False)
    else:
        ax.scatter(x[ok], y[ok], s=22, color=INK_2, edgecolors="white", linewidths=0.6, zorder=3)
    if lims is None:
        vals = np.concatenate([x[ok], y[ok]])
        lo, hi = (
            (np.nanpercentile(vals, 0.5) / 1.3, np.nanpercentile(vals, 99.5) * 1.3)
            if log
            else (min(0.0, np.nanmin(vals)), np.nanpercentile(vals, 99.5) * 1.1)
        )
    else:
        lo, hi = lims
    one_to_one(ax, lo, hi, log=log)
    if np.isfinite(st["rma_slope"]):
        xx = np.geomspace(lo, hi, 50) if log else np.linspace(lo, hi, 50)
        yy = (
            10 ** (st["rma_intercept"] + st["rma_slope"] * np.log10(xx))
            if log
            else st["rma_intercept"] + st["rma_slope"] * xx
        )
        ax.plot(xx, yy, color=INK, lw=1.2, zorder=2, label="RMA fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    text = format_stats(st, units)
    if "plotted_subset" in st:
        text += f"\n({st['plotted_subset']})"
    note(ax, text)
    ax.legend(loc="lower right", fontsize=8)
    return st


def binned_difference(ax, driver, diff, *, bins, xlabel="", ylabel="", color=INK_2, logx=False):
    """Difference against a possible cause: raw points (faint) plus the binned
    median and interquartile range. Bins with < 5 points are not summarised."""
    driver = np.asarray(driver, float)
    diff = np.asarray(diff, float)
    ok = np.isfinite(driver) & np.isfinite(diff)
    ax.scatter(driver[ok], diff[ok], s=10, color=INK_3, alpha=0.45, linewidths=0, zorder=1)
    centers, med, q25, q75 = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = ok & (driver >= a) & (driver < b)
        if sel.sum() >= 5:
            centers.append(np.sqrt(a * b) if logx else 0.5 * (a + b))
            q = np.percentile(diff[sel], [25, 50, 75])
            q25.append(q[0])
            med.append(q[1])
            q75.append(q[2])
    if centers:
        ax.fill_between(
            centers, q25, q75, color=color, alpha=0.18, lw=0, zorder=2, label="interquartile range"
        )
        ax.plot(centers, med, "-o", color=color, ms=5, lw=2, zorder=3, label="median")
    ax.axhline(0, color=INK_3, lw=1, zorder=0)
    if not ok.any():
        ax.text(0.5, 0.5, "no matched pairs", transform=ax.transAxes, ha="center", color=INK_2)
    elif logx:
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ok.any():
        legend_above(ax, ncol=2, fontsize=8)


def plot_distributions(ax, samples: Sequence[dict], *, bins, xlabel="", log_x=False):
    """Normalised step histograms.

    samples: list of dicts with keys values, label, instrument and optional
    linestyle. Colour follows the instrument; linestyle separates subsets of
    the same instrument (e.g. all samples vs samples at SPHOT times)."""
    drawn = 0
    for s in samples:
        v = np.asarray(s["values"], float)
        v = v[np.isfinite(v)]
        if log_x:
            v = v[v > 0]
        if v.size == 0:
            continue
        drawn += 1
        ax.hist(
            v,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=COLORS[s["instrument"]],
            ls=s.get("linestyle", "-"),
            label=f"{s['label']} (N={v.size:,}, median {np.median(v):.3g})",
        )
    if log_x and drawn:
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
    if not drawn:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=INK_2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")
    # Legend below the x-axis label so the step lines are never covered
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1)


def stats_table(rows: Mapping[str, dict]) -> pd.DataFrame:
    """Statistics dicts -> one tidy table (rows = comparisons)."""
    df = pd.DataFrame(rows).T
    cols = ["n", "mean_x", "mean_y", "bias", "median_bias", "rel_bias_pct", "rmsd", "r", "rma_slope"]
    df = df[[c for c in cols if c in df.columns]]
    df["n"] = df["n"].astype(int)
    return df
