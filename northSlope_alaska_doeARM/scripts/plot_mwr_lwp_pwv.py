#!/usr/bin/env python3
"""Plot MWR liquid water path and precipitable water vapor.

Produces a two-panel time series (LWP, PWV) plus an LWP histogram with the
10 g/m^2 clear-sky-ambiguity threshold marked (Hartig26 splits its analysis
at that value; about half of liquid-containing winter cases sit below it).

Example
-------
python scripts/plot_mwr_lwp_pwv.py --start 2022-01-01 --end 2022-01-31
"""

from __future__ import annotations

import argparse

from _plot_common import (
    add_data_root_argument,
    apply_data_root,
    finish_figure,
    use_headless_backend_if_needed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--no-show", action="store_true", help="save only")
    add_data_root_argument(parser)
    args = parser.parse_args()
    apply_data_root(args.data_root)
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np

    from arm_nsa import config
    from arm_nsa.mwr import read_mwr

    ds = read_mwr(args.start, args.end)
    lwp = ds["lwp_g_m2"]
    pwv = ds["pwv_cm"]
    threshold = config.LWP_CLEAR_SKY_THRESHOLD_G_M2

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 2, 1.6]}
    )

    ax = axes[0]
    ax.plot(lwp["time"], lwp, lw=0.4, color="tab:blue")
    ax.axhline(threshold, color="k", ls="--", lw=0.8,
               label=f"{threshold:.0f} g m$^{{-2}}$ clear-sky ambiguity")
    ax.set_ylabel("LWP [g m$^{-2}$]")
    ax.set_title(f"NSA C1 microwave radiometer retrievals, {args.start} to {args.end}")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(pwv["time"], pwv, lw=0.4, color="tab:green")
    ax.set_ylabel("PWV [cm]")
    ax.set_xlabel("time (UTC)")

    # LWP distribution: log-spaced bins above the threshold region show the
    # heavy tail; retrieval noise can make clear-sky LWP slightly negative.
    ax = axes[2]
    finite_lwp = lwp.values[np.isfinite(lwp.values)]
    bins = np.concatenate([[-25, 0], np.logspace(0, np.log10(500), 30)])
    ax.hist(finite_lwp, bins=bins, color="tab:blue", alpha=0.75)
    ax.axvline(threshold, color="k", ls="--", lw=0.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("LWP [g m$^{-2}$]")
    ax.set_ylabel("count")

    fig.tight_layout()
    finish_figure(
        fig, f"mwr_lwp_pwv_{args.start}_{args.end}.png", show=not args.no_show
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
