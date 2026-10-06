"""Quicklook figures for SPHOTCOD (see sphotcod.py for the variables)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from EPCAPE.analysis_tools import filters
from EPCAPE.plotting import COLORS, INK_3, concise_dates


def quicklook_day(std, crit, day: str):
    """tau, r_e and LWP retrievals (± reported std) for one UTC day. Grey = rejected."""
    t0 = np.datetime64(day)
    d = std.sel(time=slice(t0, t0 + np.timedelta64(1, "D")))
    ok = filters.combine(crit).sel(time=d["time"])
    t = d["time"].values
    panels = [
        ("tau", "tau_std", "τ"),
        ("r_e_um", "r_e_std_um", "r_e (µm)"),
        ("lwp_gm2", "lwp_std_gm2", "LWP (g m⁻²)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for ax, (name, err, label) in zip(axes, panels):
        ax.plot(t, d[name].where(~ok).values, "o", ms=4, color=INK_3, label="rejected")
        ax.errorbar(
            t,
            d[name].where(ok).values,
            yerr=d[err].where(ok).values if err in d else None,
            fmt="o",
            ms=5,
            color=COLORS["SPHOT"],
            mec="white",
            mew=0.8,
            label="valid (± std)",
        )
        ax.set_ylabel(label)
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"SPHOTCOD, {day} (UTC)", loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return fig


# -- the three gains -------------------------------------------------------------
# All three are sunphotometer retrievals, so they share the SPHOT colour (orange)
# and are told apart by marker shape and fill, which also survives greyscale.
GAIN_STYLE = {
    0: dict(marker="^", mfc="white", ms=6, label="A (aureole gain)"),
    1: dict(marker="s", mfc="white", ms=5, label="K (sky gain)"),
    2: dict(marker="o", mfc=None, ms=4.5, label="mean of A and K (default)"),
}
_GAIN_PANELS = [("tau", "τ"), ("r_e_um", "r_e (µm)"), ("lwp_gm2", "LWP (g m⁻²)")]


def _per_gain(raw, crit):
    """{gain: (standardized dataset, valid mask)} using the same criteria for each gain."""
    from EPCAPE.instruments.sunphotometer import sphotcod

    out = {}
    for g in (0, 1, 2):
        std = sphotcod.standardize(raw, gain=g)
        out[g] = (std, filters.combine(sphotcod.lwp_criteria(std, crit)))
    return out


def gain_timeseries(raw, crit, start, end=None):
    """tau, r_e and LWP from the A, K and mean-gain retrievals, for [start, end).

    raw : combined SPHOTCOD dataset (sphotcod.load), all gains
    crit : sphotcod.Criteria applied identically to every gain
    start, end : e.g. "2023-07-01" (end defaults to start + 1 day)
    Only samples that pass the criteria for that gain are drawn; the legend
    gives how many passed."""
    t0 = np.datetime64(start)
    t1 = np.datetime64(end) if end is not None else t0 + np.timedelta64(1, "D")
    gains = _per_gain(raw.sel(time=slice(t0, t1)), crit)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (name, label) in zip(axes, _GAIN_PANELS):
        for g, (std, ok) in gains.items():
            st = dict(GAIN_STYLE[g])
            mfc = st.pop("mfc") or COLORS["SPHOT"]
            lab = st.pop("label")
            v = std[name].where(ok).values
            ax.plot(
                std["time"].values,
                v,
                ls="none",
                color=COLORS["SPHOT"],
                mfc=mfc,
                mew=1.1,
                label=f"{lab}, N={int(np.isfinite(v).sum())}",
                **st,
            )
        ax.set_ylabel(label)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    fig.suptitle(
        f"Sunphotometer retrievals by gain, {str(t0)[:10]} to {str(t1)[:10]} (UTC)",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    if (t1 - t0) <= np.timedelta64(1, "D"):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel("time (UTC)")
    else:
        concise_dates(axes[-1])
    fig.tight_layout()
    return fig


def gain_differences(raw, crit):
    """Campaign view: (A - mean)/mean and (K - mean)/mean in percent for tau, r_e
    and LWP, at samples where all three gains pass the criteria.

    Returns (figure, summary table of median and 5th/95th percentiles)."""
    import pandas as pd

    gains = _per_gain(raw, crit)
    both = gains[0][1] & gains[1][1] & gains[2][1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    rows = []
    for ax, (name, label) in zip(axes, _GAIN_PANELS):
        ref = gains[2][0][name].where(both)
        for g in (0, 1):
            rel = (100 * (gains[g][0][name].where(both) - ref) / ref).values
            st = dict(GAIN_STYLE[g])
            st.pop("label")
            mfc = st.pop("mfc")
            st["ms"] = 3.5
            ax.plot(
                gains[g][0]["time"].values,
                rel,
                ls="none",
                color=COLORS["SPHOT"],
                mfc=mfc,
                mew=0.8,
                alpha=0.8,
                label=GAIN_STYLE[g]["label"],
                **st,
            )
            ok = np.isfinite(rel)
            if ok.any():
                p5, p50, p95 = np.percentile(rel[ok], [5, 50, 95])
                rows.append(
                    {
                        "quantity": name,
                        "gain": GAIN_STYLE[g]["label"],
                        "N": int(ok.sum()),
                        "median_pct": p50,
                        "p5_pct": p5,
                        "p95_pct": p95,
                    }
                )
        ax.axhline(0, color=INK_3, lw=1, zorder=0)
        # A few large outliers would flatten the bulk: limit the axis to the 99th
        # percentile of |difference| and say how many points fall outside.
        allrel = np.concatenate([100 * ((gains[g][0][name].where(both) - ref) / ref).values for g in (0, 1)])
        allrel = allrel[np.isfinite(allrel)]
        if allrel.size:
            lim = max(10.0, 1.2 * np.percentile(np.abs(allrel), 99))
            ax.set_ylim(-lim, lim)
            n_off = int((np.abs(allrel) > lim).sum())
            if n_off:
                ax.text(
                    0.995,
                    0.03,
                    f"{n_off} points beyond ±{lim:.0f}% not shown",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=INK_3,
                )
        ax.set_ylabel(f"{label}\n(gain − mean) / mean (%)", fontsize=9)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.suptitle(
        "Sunphotometer: A- and K-gain retrievals relative to the mean-gain retrieval",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    concise_dates(axes[-1])
    fig.tight_layout()
    return fig, pd.DataFrame(rows)
