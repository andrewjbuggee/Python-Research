#!/usr/bin/env python3
"""Plot 2-m air temperature from the MET station.

Time series plus distribution. If the coordinated library exists you can
overlay the sonde near-surface temperatures as a cross-check (--library).

The MPCT connection: liquid-containing clouds raise the winter surface
temperature relative to clear skies through their downwelling longwave
emission; comparing this temperature record conditioned on the phase
classification (see plot_cloud_phase_stats.py) quantifies that locally.

NOTE: MET is an extension beyond the Hartig26 instrument list (see README);
download it with:  python scripts/download_nsa_data.py --datastreams met ...

Example
-------
python scripts/plot_surface_temperature.py --start 2022-01-01 --end 2022-01-31
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
    parser.add_argument("--library", default=None,
                        help="optional coordinated-library netCDF to overlay "
                             "sonde lowest-level temperatures")
    parser.add_argument("--no-show", action="store_true", help="save only")
    add_data_root_argument(parser)
    args = parser.parse_args()
    apply_data_root(args.data_root)
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np

    from arm_nsa.surface import read_met

    ds = read_met(args.start, args.end)
    t2m_c = ds["temp_2m_c"]

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2.5, 1.4]}
    )

    ax = axes[0]
    ax.plot(t2m_c["time"], t2m_c, lw=0.6, color="tab:purple", label="MET 2-m T")
    ax.axhline(0, color="k", lw=0.6, ls="--")

    if args.library:
        import xarray as xr

        library = xr.open_dataset(args.library).sel(
            launch_time=slice(args.start, f"{args.end}T23:59:59")
        )
        # Lowest gridded sonde level (~8 m) as an independent near-surface T.
        t_sonde_c = library["tdry_c"].isel(height=0)
        ax.plot(library["launch_time"], t_sonde_c, "o", ms=4,
                mfc="none", color="tab:red", label="sonde lowest level (~8 m)")

    ax.set_ylabel("air temperature [\N{DEGREE SIGN}C]")
    ax.set_xlabel("time (UTC)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"NSA C1 surface temperature, {args.start} to {args.end}")

    ax = axes[1]
    values = t2m_c.values
    values = values[np.isfinite(values)]
    if values.size:
        ax.hist(values, bins=50, color="tab:purple", alpha=0.75)
    ax.set_xlabel("2-m temperature [\N{DEGREE SIGN}C]")
    ax.set_ylabel("count")

    fig.tight_layout()
    finish_figure(
        fig, f"surface_temperature_{args.start}_{args.end}.png",
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
