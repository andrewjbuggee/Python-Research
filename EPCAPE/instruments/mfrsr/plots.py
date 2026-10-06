"""Quicklook figures for MFRSRCLDOD (see mfrsrcldod.py for the variables)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from EPCAPE.analysis_tools import filters
from EPCAPE.plotting import COLORS, INK_2, INK_3


def quicklook_day(std, tau_crit, re_crit, day: str):
    """tau, r_e, LWP and cloud fraction for one UTC day. Grey = rejected by the criteria."""
    t0 = np.datetime64(day)
    d = std.sel(time=slice(t0, t0 + np.timedelta64(1, "D")))
    tau_ok = filters.combine(tau_crit).sel(time=d["time"])
    re_ok = filters.combine(re_crit).sel(time=d["time"])
    t = d["time"].values
    panels = [("tau", tau_ok, "τ (415 nm)", True), ("r_e_um", re_ok, "r_e (µm)", False)]
    if "lwp_gm2" in d:
        panels.append(("lwp_gm2", tau_ok, "VAP LWP (g m⁻²)", False))
    if "cloud_fraction" in d:
        panels.append(("cloud_fraction", None, "cloud fraction", False))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)), sharex=True)
    for ax, (name, ok, label, logy) in zip(np.atleast_1d(axes), panels):
        v = d[name].values
        if ok is None:
            ax.plot(t, v, "-", lw=1, color=INK_2)
        else:
            good = d[name].where(ok).values
            ax.plot(t, v, ".", ms=2, color=INK_3, alpha=0.5, label="rejected")
            ax.plot(t, good, ".", ms=2.5, color=COLORS["MFRSR"], label="valid")
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(label)
    np.atleast_1d(axes)[0].legend(loc="upper left", markerscale=3)
    np.atleast_1d(axes)[0].set_title(f"MFRSRCLDOD, {day} (UTC)", loc="left")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return fig
