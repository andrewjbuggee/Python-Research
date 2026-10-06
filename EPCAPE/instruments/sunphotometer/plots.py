"""Quicklook figures for SPHOTCOD (see sphotcod.py for the variables)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from epcape import filters
from epcape.plotting import COLORS, INK_3


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
