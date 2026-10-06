"""Quicklook figures for MWRLOS (see mwrlos.py for the variables)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from EPCAPE import filters
from EPCAPE.plotting import COLORS, GRID, INK_2, INK_3


def quicklook_day(std, crit, day: str):
    """LWP, PWV, brightness temperatures and wet-window flag for one UTC day."""
    t0 = np.datetime64(day)
    d = std.sel(time=slice(t0, t0 + np.timedelta64(1, "D")))
    ok = filters.combine(crit).sel(time=d["time"])
    t = d["time"].values
    n = 3 + ("wet_window" in d)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    ax = axes[0]
    ax.axhspan(-30, 30, color=GRID, lw=0, zorder=0)
    ax.plot(t, d["lwp_gm2"].values, ".", ms=1.5, color=INK_3, alpha=0.5, label="rejected")
    ax.plot(t, d["lwp_gm2"].where(ok).values, ".", ms=1.5, color=COLORS["MWR"], label="valid")
    ax.set_ylabel("LWP (g m⁻²)")
    ax.legend(loc="upper left", markerscale=4)
    ax.set_title(f"MWRLOS, {day} (UTC); shaded ±30 g m⁻² = retrieval RMS", loc="left")
    axes[1].plot(t, d["pwv_cm"].values, "-", lw=1, color=INK_2)
    axes[1].set_ylabel("PWV (cm)")
    if "tb23_K" in d:
        axes[2].plot(t, d["tb23_K"].values, "-", lw=1, color=INK_2, label="23.8 GHz")
    if "tb31_K" in d:
        axes[2].plot(t, d["tb31_K"].values, "-", lw=1, color=INK_3, label="31.4 GHz")
    axes[2].set_ylabel("Tb (K)")
    axes[2].legend(loc="upper left")
    if "wet_window" in d:
        axes[3].step(t, d["wet_window"].values, where="post", color=INK_2, lw=1)
        axes[3].set_ylabel("wet window")
        axes[3].set_yticks([0, 1])
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return fig
