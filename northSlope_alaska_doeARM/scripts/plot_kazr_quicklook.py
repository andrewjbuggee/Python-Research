#!/usr/bin/env python3
"""KAZR time-height reflectivity quicklook, with ceilometer cloud base overlay.

Keep the window short (a day or less is typical): full-resolution KAZR is
about 0.5-2 GB per day on disk and this script loads the whole window.

Example
-------
python scripts/plot_kazr_quicklook.py --date 2022-01-05
python scripts/plot_kazr_quicklook.py --start 2022-01-05T06:00 --end 2022-01-05T18:00
"""

from __future__ import annotations

import argparse

from _plot_common import finish_figure, use_headless_backend_if_needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="single UTC day, YYYY-MM-DD")
    parser.add_argument("--start", help="window start (ISO time)")
    parser.add_argument("--end", help="window end (ISO time)")
    parser.add_argument("--zmax-km", type=float, default=8.0,
                        help="top of plotted height axis [km], default 8")
    parser.add_argument("--no-show", action="store_true", help="save only")
    args = parser.parse_args()

    if args.date:
        start, end = f"{args.date}T00:00:00", f"{args.date}T23:59:59"
        stamp = args.date
    elif args.start and args.end:
        start, end = args.start, args.end
        stamp = f"{args.start}_{args.end}".replace(":", "")
    else:
        parser.error("give either --date or both --start and --end")

    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt

    from arm_nsa.radar import read_kazr

    kazr = read_kazr(start, end, verbose=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    mesh = ax.pcolormesh(
        kazr["time"],
        kazr["height"] / 1000.0,
        kazr["reflectivity_dbz"].transpose("height", "time"),
        cmap="viridis",
        vmin=-50,
        vmax=20,
        shading="nearest",
    )
    fig.colorbar(mesh, ax=ax, label="reflectivity [dBZ]")

    # Ceilometer lowest cloud base, when those files are present locally.
    try:
        from arm_nsa.ceilometer import read_ceilometer

        ceil = read_ceilometer(start[:10], end[:10]).sel(time=slice(start, end))
        ax.plot(
            ceil["time"],
            ceil["first_cbh_m"] / 1000.0,
            ".",
            ms=1.5,
            color="red",
            label="ceilometer first cloud base",
        )
        ax.legend(loc="upper right", fontsize=8, markerscale=6)
    except FileNotFoundError:
        print("note: no local ceilometer files for this window; overlay skipped")

    ax.set_ylim(0, args.zmax_km)
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("height AGL [km]")
    ax.set_title(f"NSA C1 KAZR (general mode), {stamp}")

    fig.tight_layout()
    finish_figure(fig, f"kazr_quicklook_{stamp}.png", show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
