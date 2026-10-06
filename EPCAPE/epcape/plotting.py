"""Shared matplotlib conventions for EPCAPE figures.

Each instrument keeps one colour in every figure, so a reader who learns
"MFRSR is blue" never has to relearn it. The three hues are the first three
slots of a categorical palette that was checked for colour-vision-deficiency
separation as a set (all pairs). Text and axes stay in neutral greys; colour
marks identity only.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import ticker

# Fixed identity colours: never reassign or cycle these.
COLORS = {
    "MFRSR": "#2a78d6",  # blue   - multifilter rotating shadowband radiometer (MFRSRCLDOD)
    "SPHOT": "#eb6834",  # orange - Cimel sunphotometer cloud mode (SPHOTCOD)
    "MWR": "#1baf7a",  # aqua   - 2-channel microwave radiometer (MWRLOS)
}
# Secondary (non-identity) ink
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e4e3df"

LABELS = {
    "MFRSR": "MFRSR (415 nm diffuse, hemispheric)",
    "SPHOT": "Sunphotometer (440/870/1640 nm zenith radiance, 1.2° FOV)",
    "MWR": "MWR (23.8/31.4 GHz, ~5° FOV)",
}


def use_style() -> None:
    """Thin marks, hairline recessive grid, no top/right spines."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.edgecolor": INK_3,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,  # gridlines behind bars and markers
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def one_to_one(ax, lo: float, hi: float, log: bool = False) -> None:
    """Draw the 1:1 line in neutral ink and set equal limits."""
    ax.plot([lo, hi], [lo, hi], color=INK_3, lw=1, zorder=1, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axis(ax)
    ax.set_aspect("equal", adjustable="box")


def note(ax, text: str, loc: str = "upper left") -> None:
    """Statistics box in neutral ink."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6),
    )


def savefig(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plain_log_axis(ax, which: str = "both") -> None:
    """Log axis labelled with plain numbers (5, 10, 20, 50 ...) instead of 2x10^1."""
    fmt = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    axes = [ax.xaxis, ax.yaxis] if which == "both" else [ax.xaxis if which == "x" else ax.yaxis]
    for axis in axes:
        axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(ticker.NullFormatter())


def concise_dates(ax) -> None:
    """Date axis with matplotlib's concise formatter (no repeated month/year labels)."""
    loc = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def legend_above(ax, ncol: int = 4, **kw) -> None:
    """Legend in a row above the axes, so it never covers data."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=ncol,
        borderaxespad=0.3,
        handletextpad=0.4,
        columnspacing=1.2,
        **kw,
    )
