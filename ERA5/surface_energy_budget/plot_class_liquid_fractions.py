#!/usr/bin/env python3
"""Liquid-containing cloud fraction per season, by surface class.

THE AVERAGING SCHEME, AND WHY IT IS THE ONE IT IS
=================================================
The quantity each bar shows is

    fraction(season, class) = liquid-containing cell-hours in the class
                              --------------------------------------------
                              valid cell-hours the class actually occupied

as a percentage. It is a RATIO OF SUMS over cell-hours -- pool first, divide
once -- not a mean of per-cell fractions, and the distinction is not cosmetic.

WHY A RATIO OF SUMS AND NOT A MEAN OF PER-CELL RATIOS
-----------------------------------------------------
Three of the five classes are defined by sea ice concentration, so CLASS
MEMBERSHIP MOVES: one cell is open ocean in October and sea ice in February. A
per-cell fraction would need a denominator of "the hours that cell spent in this
class", and averaging those fractions across cells would give a cell present for
20 hours the same weight as one present for 2,000. Pooling the cell-hours first
weights each cell by how long it was actually in the class, which is the only
construction that stays correct while the ice edge migrates.

This is the same normalisation the rest of the project uses, and the error it
avoids is large rather than subtle: multiplying a monthly occupancy fraction by
the calendar month instead reached a 1,595 h discrepancy on this archive -- a
third of a season. See the note in ``season_phase_hours``.

WHAT "A TYPICAL GRID CELL" MEANS -- THE WEIGHTING CHOICE
--------------------------------------------------------
Two defensible weights, answering different questions:

    area (cos latitude)   per unit AREA of the class. A randomly chosen square
                          kilometre-hour of pack ice.
    uniform              per GRID CELL of the class. A randomly chosen ERA5
                          cell-hour of pack ice.

They are NOT interchangeable here, because the classes are latitude-structured.
A 0.25 deg cell is about 264 km2 at 70 N and 134 km2 at 80 N, so uniform
weighting counts a northern cell as fully as a southern one that covers twice
the ground -- and pack ice lives at the northern end of this domain. The default
is ``area``: "what fraction of the pack-ice REGION carried liquid cloud" is the
question with a physical referent, and a grid-cell count is an artefact of
ERA5's grid rather than a property of the Arctic.

The size of the disagreement is an empirical matter, not something to assert, so
:func:`compare_weightings` measures it and the notebook reports it. Where the two
agree the choice is immaterial and should be said to be; where they differ, the
area-weighted number is the one to quote.

THE DENOMINATOR IS ALL SEASON TIME, NOT CLOUDY TIME
---------------------------------------------------
The bars are the fraction of the WHOLE season window, matching
``map_liquid_containing_hours.ipynb``. The conditional quantity -- liquid
containing as a share of cloudy time -- is a different number and is drawn as a
faint outline behind each bar (the overcast fraction) plus reported separately,
since "within a cloudy scene" can be read either way.
"""

from __future__ import annotations

import numpy as np

from plot_lwp_histogram_by_surface_class import (
    CLASS_ORDER,
    SITE_COLOR,
    panel_order,
    season_phase_binary,
    window_label,
)
from map_liquid_hours import precip_banner, precip_suffix
from surface_classification import CLASS_COLORS


def class_fractions(A):
    """(labels, fraction %, cloudy %, panel spec) for every class and the site.

    ``fraction`` and ``cloudy`` are shaped ``(n_season, n_panel)``. Both use the
    class's own valid cell-hours as the denominator, so they are directly
    comparable and the first can never exceed the second.
    """
    labels, liquid, ice, _clear, season_h = season_phase_binary(A)
    season_h = np.asarray(season_h, dtype=float)[:, None]
    cloudy_h = liquid + ice
    panels = panel_order(A.col["site_code"])
    idx = [code for code, _lab, _is_site in panels]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = 100.0 * liquid[:, idx] / season_h
        cloud = 100.0 * cloudy_h[:, idx] / season_h
    return labels, frac, cloud, panels


def compare_weightings(A_area, A_uniform) -> None:
    """Measure how much the cos(latitude) choice actually changes the answer.

    Prints the record-mean fraction per class under both weightings. This is the
    evidence for the default rather than an argument for it.
    """
    _l, f_a, _c, panels = class_fractions(A_area)
    _l2, f_u, _c2, _p2 = class_fractions(A_uniform)
    print(f"{'panel':<22}{'area-wtd':>10}{'uniform':>10}{'diff':>9}"
          f"{'rel':>9}")
    for k, (_code, lab, _is_site) in enumerate(panels):
        a, u = np.nanmean(f_a[:, k]), np.nanmean(f_u[:, k])
        rel = 100.0 * (u - a) / a if a > 0 else np.nan
        print(f"{lab:<22}{a:>9.2f}%{u:>9.2f}%{u - a:>+8.2f}{rel:>+8.1f}%")
    print("\n  A single grid cell has no interior cells to weight, so the ARM "
          "row must read\n  exactly 0.00 -- it is the control on this table.")


def fig_class_season_fractions(A, A_sc=None, out_dir=None, dpi=None,
                               n_cols: int = 3, show_cloudy: bool = True,
                               label_fontsize: float = 11.5,
                               tick_fontsize: float = 9.5):
    """Six panels -- five surface classes plus the ARM cell -- one bar a season.

    Bar height is the fraction of the season window with a liquid-containing
    cloud overhead. The faint outline behind each bar is the OVERCAST fraction,
    so the gap between them reads as "cloudy but not liquid containing" and the
    subset relation is visible rather than asserted.

    ``A_sc`` is an otherwise identical run with ``liquid_var="tcslw"``. When
    given, each panel's header carries the record-mean supercooled fraction
    beside the all-liquid one.
    """
    import matplotlib.pyplot as plt

    labels, frac, cloud, panels = class_fractions(A)
    frac_sc = None
    if A_sc is not None:
        _check_comparable(A, A_sc)
        _l, frac_sc, _c, _p = class_fractions(A_sc)

    n_p = len(panels)
    n_rows = int(np.ceil(n_p / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True, sharex=True,
                            figsize=(4.6 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(labels))
    top = float(np.nanmax(cloud)) * 1.30

    for k, (_code, lab, is_site) in enumerate(panels):
        ax = axes[k]
        color = SITE_COLOR if is_site else CLASS_COLORS[CLASS_ORDER[k]]
        if show_cloudy:
            ax.bar(x, cloud[:, k], width=0.72, facecolor="none",
                   edgecolor="0.55", linewidth=0.9, linestyle="--",
                   label="overcast" if k == 0 else None)
        ax.bar(x, frac[:, k], width=0.72, color=color, edgecolor="white",
               linewidth=0.5, label="liquid containing" if k == 0 else None)

        m = np.nanmean(frac[:, k])
        ax.axhline(m, color="0.2", lw=1.1, ls=":")
        head = (f"{lab}\nmean {m:.1f}% of the season"
                f"   |   {np.nanmean(frac[:, k] / cloud[:, k] * 100):.0f}% of cloudy time")
        if frac_sc is not None:
            head += f"\nsupercooled: {np.nanmean(frac_sc[:, k]):.1f}% of the season"
        ax.set_title(head, fontsize=label_fontsize - 1.5, linespacing=1.35)
        ax.set_ylim(0, top)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if k % n_cols == 0:
            ax.set_ylabel("fraction of the season [%]", fontsize=label_fontsize)
    for ax in axes[n_p:]:
        ax.set_visible(False)
    for ax in axes[:n_p]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right",
                           fontsize=tick_fontsize - 0.5)
    axes[0].legend(fontsize=tick_fontsize, framealpha=0.9, loc="upper left")

    a = A.args
    wt = ("area weighted, cos(latitude)" if getattr(a, "cell_weighting", "area")
          == "area" else "uniform per grid cell")
    sc_note = ("   |   supercooled figure from an identical tcslw run"
               if frac_sc is not None else "")
    fig.suptitle(
        f"{precip_banner(a)}\n"
        f"Liquid-containing cloud time by surface class — {a.region}\n"
        f"tcc $\\geq$ {a.min_cloud_fraction:g}   |   liquid containing: "
        f"LWP/(LWP+IWP) > {100 * (1 - A.phase_kw['ice_fraction_min']):g}%   |   "
        f"min LWP/IWP {A.phase_kw['min_lwp_g']:g}/{A.phase_kw['min_iwp_g']:g} "
        f"g m$^{{-2}}$\nclass average: {wt}   |   season window "
        f"{window_label(A.col['season_hours'])}   |   dotted line: record mean"
        f"{sc_note}",
        fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, A, out_dir, "class_liquid_fraction_by_season", dpi)


def _check_comparable(A, A_sc) -> None:
    """The supercooled run must differ from the main one ONLY in liquid_var."""
    if A_sc.args.liquid_var == A.args.liquid_var:
        raise ValueError("A_sc must be a tcslw run; it uses the same "
                         f"liquid_var as A ({A.args.liquid_var})")
    for name in ("region", "season_start", "season_end", "min_cloud_fraction",
                 "min_lwp", "min_iwp", "liquid_fraction_min",
                 "ice_fraction_min", "no_precip", "cell_weighting"):
        if getattr(A.args, name, None) != getattr(A_sc.args, name, None):
            raise ValueError(
                f"the supercooled run differs in {name!r}, so its fraction "
                f"would not be attributable to the liquid definition")
    if list(A.used) != list(A_sc.used):
        raise ValueError("the two runs cover different seasons")


def _save(fig, A, out_dir, stem, dpi):
    from pathlib import Path
    if out_dir is None:
        return fig
    path = Path(out_dir) / (f"{A.args.region}_{stem}_"
                            f"{precip_suffix(A.args)}_{A.tag}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def print_class_report(A, A_sc=None) -> None:
    """Per-class, per-season fractions, and the record means."""
    print(f"*** {precip_banner(A.args, mathtext=False)} ***")
    labels, frac, cloud, panels = class_fractions(A)
    frac_sc = class_fractions(A_sc)[1] if A_sc is not None else None
    names = [lab for _c, lab, _s in panels]
    print(f"{'season':<10}" + "".join(f"{n[:13]:>15}" for n in names))
    for i, lab in enumerate(labels):
        print(f"{lab:<10}" + "".join(f"{frac[i, k]:>14.1f}%"
                                     for k in range(len(names))))
    print(f"\n{'mean':<10}" + "".join(f"{np.nanmean(frac[:, k]):>14.1f}%"
                                      for k in range(len(names))))
    print(f"{'of cloudy':<10}" + "".join(
        f"{np.nanmean(frac[:, k] / cloud[:, k]) * 100:>14.1f}%"
        for k in range(len(names))))
    if frac_sc is not None:
        print(f"{'supercool':<10}" + "".join(
            f"{np.nanmean(frac_sc[:, k]):>14.1f}%" for k in range(len(names))))
        print(f"{'sc/all':<10}" + "".join(
            f"{100 * np.nanmean(frac_sc[:, k]) / np.nanmean(frac[:, k]):>14.1f}%"
            for k in range(len(names))))
