#!/usr/bin/env python3
"""Downwelling longwave by cloud state and cloud phase, within each surface class.

Produces two bar charts and the tables behind them:

    1. DLR by CLOUD STATE   (clear / cloudy / overcast / all-sky)
    2. DLR by CLOUD PHASE   (liquid / ice / mixed), within overcast scenes

Each chart carries six groups: the whole region first, then the five surface
classes from ``surface_classification.py`` (land, coastal, open ocean, marginal
ice zone, sea ice). The whole-region group is separated by a rule, because it is
a different kind of number -- an average over the other five weighted by how much
area each happened to occupy.

Reading the cloud-state chart
=============================
all-sky means NO cloud filter -- the actual atmosphere, clouds and all. It is the
standard counterpart to clear-sky, not a synonym for "cloudy"; see
``cloud_classification.py``. Its bar therefore sits between the clear and
overcast bars roughly in proportion to how often the sky is covered, and
overcast-minus-clear is the quantity that isolates the cloud effect.

The surface classes are NOT independent of cloud state. Open water in autumn is
both warmer and cloudier than the pack ice a few hundred kilometres north, so a
difference between the open-ocean and sea-ice bars mixes a surface effect with a
cloud-regime effect. Comparing within a cloud state -- reading down a single
colour across groups -- is what separates them, and is the reason this chart is
grouped the way it is.

Sample sizes
============
Cell-hour counts are printed in the table and annotated on each bar. They matter
here more than in a time series: clear skies over the autumn Arctic Ocean are
uncommon, and an ice-only overcast scene over open water rarer still. A bar
backed by fewer than ``--min-samples`` cell-hours is drawn hatched and its mean
should not be read as a regional statistic.

Error bars are the area-weighted standard deviation of the population, not a
standard error. They describe how variable DLR is within that bin, which is the
useful number; with hundreds of thousands of correlated cell-hours a standard
error would be misleadingly tiny.

Weighting
=========
Every mean is cos(latitude)-weighted, matching the rest of the analysis. Means
pool all cell-hours in the selected seasons rather than averaging per-season
means, so a season with partial coverage contributes in proportion to what it
actually covers.

Examples
========
  ./plot_dlr_by_cloud_state.py --region barrow
  ./plot_dlr_by_cloud_state.py --region barrow --years 2021
  ./plot_dlr_by_cloud_state.py --region barrow --years 2019-2025 --cloudy-threshold 0.7

  # if your phase thresholds were meant in kg m-2 rather than g m-2
  ./plot_dlr_by_cloud_state.py --region barrow --lwp-min 30 --iwp-min 30 \
      --lwp-max-ice 1 --iwp-max-liquid 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cloud_classification import (
    PHASE_COLORS,
    PHASE_LABELS,
    PHASE_ORDER,
    STATE_COLORS,
    STATE_LABELS,
    STATE_ORDER,
    add_cloud_phase_args,
    add_cloud_state_args,
    cloud_phase_masks,
    cloud_state_masks,
)
from plot_surface_class_timeseries import (
    parse_month_day,
    parse_years,
    season_layout,
    select_seasons,
)
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_LABELS,
    CLASS_ORDER,
    DEFAULT_BLOCK_HOURS,
    add_classification_args,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

DLR_VAR = "msdwlwrf"
DLR_LABEL = "Downwelling longwave"
DLR_UNITS = "W m$^{-2}$"

REQUIRED_VARS = (DLR_VAR, "tclw", "tciw", "tcc", "siconc")

# Row 0 of every accumulator is the whole region; rows 1..5 are the classes.
GROUP_ORDER: tuple[str, ...] = ("all",) + CLASS_ORDER
GROUP_LABELS: dict[str, str] = {"all": "Whole region", **CLASS_LABELS}

# A bar backed by fewer cell-hours than this is drawn hatched.
DEFAULT_MIN_SAMPLES = 500


class Accumulator:
    """Streaming weighted mean and standard deviation per (group, category).

    Holds the four running sums a weighted mean and variance need, so the record
    can be reduced in blocks without ever materialising it. Keeping the raw count
    alongside the weight sum matters: the weight sum answers "how much area-time"
    and the count answers "how many samples", and only the latter tells you
    whether a bar is backed by enough data to read.
    """

    def __init__(self, n_group: int, n_category: int):
        self.w = np.zeros((n_group, n_category))       # sum of weights
        self.vw = np.zeros((n_group, n_category))      # sum of w * value
        self.vw2 = np.zeros((n_group, n_category))     # sum of w * value^2
        self.n = np.zeros((n_group, n_category), dtype=np.int64)

    def add(self, g: int, c: int, mask: np.ndarray, weights: np.ndarray,
            value: np.ndarray) -> None:
        """Fold one block's contribution to bin ``(g, c)``."""
        wm = mask * weights
        self.w[g, c] += wm.sum()
        self.vw[g, c] += (wm * value).sum()
        self.vw2[g, c] += (wm * value * value).sum()
        self.n[g, c] += int(mask.sum())

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.w > 0, self.vw / np.where(self.w > 0, self.w, 1.0),
                            np.nan)

    def std(self) -> np.ndarray:
        """Area-weighted population standard deviation within each bin."""
        m = self.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(
                self.w > 0,
                self.vw2 / np.where(self.w > 0, self.w, 1.0) - m * m,
                np.nan,
            )
        # Round-off can push a near-zero variance slightly negative.
        return np.sqrt(np.clip(var, 0.0, None))


def build_stats(ds, lsm: np.ndarray, args, layout: dict, keep_idx: list[int]):
    """One streaming pass returning the state and phase accumulators."""
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    n_group = len(GROUP_ORDER)
    state_acc = Accumulator(n_group, len(STATE_ORDER))
    phase_acc = Accumulator(n_group, len(PHASE_ORDER))

    weights = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])
    n_phase_unclassified = 0
    n_phase_total = 0

    for i0, block in iter_time_blocks(ds, list(REQUIRED_VARS), args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        keep = use_step[i0 : i0 + n_t]
        if not keep.all():
            block = block.isel(valid_time=np.nonzero(keep)[0])
            if block.sizes["valid_time"] == 0:
                continue

        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )
        dlr = block[DLR_VAR].values
        tcc = block["tcc"].values
        lwp_g = block["tclw"].values * 1000.0   # kg m-2 -> g m-2
        iwp_g = block["tciw"].values * 1000.0

        dlr_ok = np.isfinite(dlr)
        states = cloud_state_masks(tcc, args.tcc_tol, args.cloudy_threshold)
        phases = cloud_phase_masks(
            lwp_g, iwp_g, args.lwp_min, args.iwp_min,
            args.lwp_max_ice, args.iwp_max_liquid,
        )

        # Phase is evaluated only within the chosen cloud state (overcast by
        # default): a phase label on a half-clear scene would describe whatever
        # cloud happens to be in the column, not the scene.
        phase_base = states[args.phase_state] & dlr_ok
        any_phase = phases["liquid"] | phases["ice"] | phases["mixed"]
        n_phase_unclassified += int((phase_base & ~any_phase).sum())
        n_phase_total += int(phase_base.sum())

        for g, group in enumerate(GROUP_ORDER):
            in_group = (np.ones_like(classes, dtype=bool) if group == "all"
                        else classes == CLASS_CODES[group])
            for c, state in enumerate(STATE_ORDER):
                state_acc.add(g, c, in_group & states[state] & dlr_ok,
                              weights, np.nan_to_num(dlr))
            for c, phase in enumerate(PHASE_ORDER):
                phase_acc.add(g, c, in_group & phase_base & phases[phase],
                              weights, np.nan_to_num(dlr))

    return state_acc, phase_acc, n_phase_unclassified, n_phase_total


def print_table(acc: Accumulator, categories: tuple[str, ...],
                labels: dict[str, str], title: str) -> None:
    """Mean +/- sd and sample count for every (group, category) bin."""
    print()
    print(f"  {title}")
    print(f"  {'group':<20}" + "".join(f"{labels[c]:>22}" for c in categories))
    print("  " + "-" * (20 + 22 * len(categories)))
    m, sd = acc.mean(), acc.std()
    for g, group in enumerate(GROUP_ORDER):
        cells = []
        for c in range(len(categories)):
            if acc.n[g, c] == 0:
                cells.append(f"{'--':>22}")
            else:
                cells.append(f"{m[g, c]:>11.1f} +/-{sd[g, c]:6.1f}")
        print(f"  {GROUP_LABELS[group]:<20}" + "".join(cells))
    # Counts per GROUP, not just for the whole region: a bin can be well
    # sampled overall and nearly empty for one surface class, and the mean of
    # that bin is then not a statement about the region.
    print()
    print(f"  {'sample cell-hours':<20}"
          + "".join(f"{labels[c]:>22}" for c in categories))
    for g, group in enumerate(GROUP_ORDER):
        print(f"  {GROUP_LABELS[group]:<20}"
              + "".join(f"{acc.n[g, c]:>22,}" for c in range(len(categories))))


def make_bar_chart(
    acc: Accumulator,
    categories: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    subtitle: str,
    args,
    output_path: Path,
):
    """Grouped bar chart: six groups on x, one bar per category."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    m, sd = acc.mean(), acc.std()
    n_group, n_cat = len(GROUP_ORDER), len(categories)
    width = 0.8 / n_cat
    x = np.arange(n_group, dtype=float)
    # Push the five surface classes right, leaving a lane for the divider that
    # separates them from the whole-region group.
    x[1:] += 0.35

    fig, ax = plt.subplots(figsize=(13.5, 7.0))

    for c, name in enumerate(categories):
        offset = (c - (n_cat - 1) / 2) * width
        heights = m[:, c]
        errs = sd[:, c]
        thin = acc.n[:, c] < args.min_samples
        bars = ax.bar(
            x + offset, np.nan_to_num(heights), width * 0.92,
            yerr=np.where(np.isfinite(errs), errs, 0.0),
            color=colors[name], label=labels[name],
            edgecolor="black", linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"},
        )
        # Hatch the bars that are too thinly sampled to read as a statistic.
        for b, is_thin, n in zip(bars, thin, acc.n[:, c]):
            if n == 0:
                b.set_visible(False)
            elif is_thin:
                b.set_hatch("///")
                b.set_alpha(0.55)

        # Sample count above each bar, small and rotated so it never collides.
        for xi, h, e, n in zip(x + offset, heights, errs, acc.n[:, c]):
            if n == 0:
                continue
            top = (h + (e if np.isfinite(e) else 0.0))
            ax.annotate(f"{n:,}", xy=(xi, top), xytext=(0, 3),
                        textcoords="offset points", rotation=90,
                        ha="center", va="bottom", fontsize=6.5, color="#444444")

    ax.axvline((x[0] + x[1]) / 2, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel(f"{DLR_LABEL} [{DLR_UNITS}]")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    handles, hlabels = ax.get_legend_handles_labels()
    if (acc.n < args.min_samples).any():
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                             alpha=0.55))
        hlabels.append(f"n < {args.min_samples:,} cell-hours")
    ax.legend(handles, hlabels, loc="upper right", frameon=True, framealpha=0.92,
              fontsize=9, ncol=2)

    # Headroom for the rotated sample-count annotations.
    finite = m[np.isfinite(m)]
    if finite.size:
        top = np.nanmax(m + np.nan_to_num(sd))
        ax.set_ylim(0, top * 1.18)

    fig.suptitle(title, fontsize=13, y=0.975)
    ax.set_title(subtitle, fontsize=10, color="#444444", pad=8)
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.07, right=0.985)
    fig.savefig(output_path, dpi=args.dpi)
    print(f"  Wrote {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    add_cloud_state_args(parser)
    add_cloud_phase_args(parser)
    parser.add_argument("--phase-state", choices=STATE_ORDER, default="overcast",
                        help="Which cloud state the phase breakdown is computed "
                             "within (default overcast).")
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD", help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 03-31).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to pool, by the year each season STARTS in: "
                             "'2019', '2019-2025'. Default: every season meeting "
                             "--min-season-coverage.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6,
                        metavar="F",
                        help="Exclude seasons covering less than this fraction of "
                             "the window (default 0.6).")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        metavar="N",
                        help="Bars backed by fewer cell-hours than this are drawn "
                             f"hatched (default {DEFAULT_MIN_SAMPLES:,}).")
    parser.add_argument("--report-tcc", action="store_true",
                        help="Also print the total cloud cover histogram, to check "
                             "how much the --cloudy-threshold choice actually moves.")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Downwelling longwave by cloud state and phase, within surface class")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, None, None, region_dir.parent)
        lsm_da = load_land_sea_mask(
            args.region, resolve_data_root(args.storage, args.data_root),
            args.mask_grid,
        )
        lsm = align_lsm_to_grid(lsm_da, ds)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = sorted(set(REQUIRED_VARS) - set(ds.data_vars))
    if missing:
        print(f"  Error: dataset is missing {missing}.", file=sys.stderr)
        return 1

    print(f"  Source      : {region_dir}")
    print(f"  Grid        : {ds.sizes['latitude']} x {ds.sizes['longitude']} cells, "
          f"{ds.sizes['valid_time']:,} time steps")
    print(f"  Cloud state : clear tcc <= {args.tcc_tol:g} | cloudy tcc > "
          f"{args.cloudy_threshold:g} | overcast tcc >= {1 - args.tcc_tol:g} | "
          f"all-sky = no filter")
    print(f"  Cloud phase : liquid LWP > {args.lwp_min:g} & IWP < "
          f"{args.iwp_max_liquid:g} | ice IWP > {args.iwp_min:g} & LWP < "
          f"{args.lwp_max_ice:g} | mixed both > min   [g m-2]")
    print(f"  Phase within: {args.phase_state}")

    try:
        layout = season_layout(ds, args)
        keep_idx, used, mode_label = select_seasons(layout, args)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    print(f"\n  Reading {len(used)} season(s): {used}")

    state_acc, phase_acc, n_unphased, n_phase_total = build_stats(
        ds, lsm, args, layout, keep_idx
    )

    print_table(state_acc, STATE_ORDER, STATE_LABELS,
                "Mean DLR [W m-2] by cloud state (area-weighted, +/- weighted sd)")
    print_table(phase_acc, PHASE_ORDER, PHASE_LABELS,
                f"Mean DLR [W m-2] by cloud phase, {args.phase_state} scenes only")

    if n_phase_total:
        print(f"\n  Phase-unclassified {args.phase_state} cell-hours: "
              f"{n_unphased:,} of {n_phase_total:,} "
              f"({100.0 * n_unphased / n_phase_total:.1f}%)")
        print("    Condensate between the 'essentially none' ceiling and the "
              "'present' floor belongs to no phase; see cloud_classification.py.")

    # The cloud-effect number the state chart exists to show.
    m = state_acc.mean()
    print("\n  Overcast minus clear [W m-2]:")
    for g, group in enumerate(GROUP_ORDER):
        d = m[g, STATE_ORDER.index("overcast")] - m[g, STATE_ORDER.index("clear")]
        if np.isfinite(d):
            print(f"    {GROUP_LABELS[group]:<20}{d:>8.1f}")

    if args.report_tcc:
        print("\n  Total cloud cover histogram is not accumulated in this pass; "
              "use analyze_cloud_liquid_frequency.py for the full distribution.")

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    tag = f"season{used[0]}" if len(used) == 1 else f"pooled{used[0]}-{used[-1]}"
    season_note = (f"{args.season_start[0]:02d}-{args.season_start[1]:02d} to "
                   f"{args.season_end[0]:02d}-{args.season_end[1]:02d}")
    print()

    make_bar_chart(
        state_acc, STATE_ORDER, STATE_LABELS, STATE_COLORS,
        f"{DLR_LABEL} by cloud state and surface class — {args.region}",
        f"{mode_label}   |   {season_note}   |   cloudy = tcc > "
        f"{args.cloudy_threshold:g}, all-sky = no cloud filter   |   "
        f"error bars: area-weighted sd",
        args, out_dir / f"{args.region}_dlr_by_cloud_state_{tag}.png",
    )
    make_bar_chart(
        phase_acc, PHASE_ORDER, PHASE_LABELS, PHASE_COLORS,
        f"{DLR_LABEL} by cloud phase and surface class — {args.region}",
        f"{mode_label}   |   {season_note}   |   {args.phase_state} scenes only   |   "
        f"liquid/ice floor {args.lwp_min:g}/{args.iwp_min:g} g m$^{{-2}}$, "
        f"absence ceiling {args.lwp_max_ice:g}/{args.iwp_max_liquid:g} g m$^{{-2}}$",
        args, out_dir / f"{args.region}_dlr_by_cloud_phase_{tag}.png",
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
