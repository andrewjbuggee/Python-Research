#!/usr/bin/env python3
"""Plot all radiosonde profiles for one day, with saturated layers shaded.

One column per launch: temperature and dewpoint on the left axis logic,
RH w.r.t. liquid on a twin axis, and detected saturated layers (the sounding
proxy for liquid-containing cloud, RH >= 95% with the Hartig26 30-m rules)
shaded in blue.

Example
-------
python scripts/plot_sonde_day.py --date 2022-01-05
"""

from __future__ import annotations

import argparse

from _plot_common import finish_figure, use_headless_backend_if_needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC day, YYYY-MM-DD")
    parser.add_argument("--zmax-km", type=float, default=5.0,
                        help="top of plotted height axis [km], default 5")
    parser.add_argument("--no-show", action="store_true", help="save only")
    args = parser.parse_args()
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from arm_nsa import config
    from arm_nsa.sonde import find_saturated_layers, read_sondes

    sondes = read_sondes(args.date, args.date)
    n_launch = sondes.sizes["launch_time"]
    height_km = sondes["height"].values / 1000.0

    fig, axes = plt.subplots(
        1, n_launch, figsize=(4.2 * n_launch, 7), sharey=True, squeeze=False
    )
    for i in range(n_launch):
        ax = axes[0, i]
        profile = sondes.isel(launch_time=i)
        launch = pd.Timestamp(profile["launch_time"].values)

        ax.plot(profile["tdry_c"], height_km, color="tab:red", lw=1.2, label="T")
        ax.plot(profile["dp_c"], height_km, color="tab:red", lw=1.0, ls=":",
                label="T$_d$")
        ax.set_xlabel("temperature [\N{DEGREE SIGN}C]")
        ax.set_title(launch.strftime("%Y-%m-%d %H:%M UTC"))
        ax.grid(alpha=0.3)

        ax_rh = ax.twiny()
        ax_rh.plot(profile["rh_pct"], height_km, color="tab:blue", lw=1.0,
                   label="RH")
        ax_rh.axvline(config.SATURATION_RH_THRESHOLD_PCT, color="tab:blue",
                      lw=0.7, ls="--", alpha=0.7)
        ax_rh.set_xlim(0, 105)
        ax_rh.set_xlabel("RH w.r.t. liquid [%]", color="tab:blue")
        ax_rh.tick_params(axis="x", colors="tab:blue")

        # Shade saturated layers (Hartig26 rules: 95% threshold, 30-m gap
        # fill, 30-m minimum depth).
        layers = find_saturated_layers(
            profile["rh_pct"].values, sondes["height"].values
        )
        for layer in layers:
            ax.axhspan(layer.base_m / 1000.0, layer.top_m / 1000.0,
                       color="tab:blue", alpha=0.18)
        if i == 0:
            ax.set_ylabel("height [km MSL]")
            ax.legend(loc="upper right", fontsize=8)

        # Freezing level marker for mixed-phase context: layers between 0 and
        # -38 degC are in the heterogeneous-freezing (mixed-phase) regime.
        tdry = profile["tdry_c"].values
        if np.isfinite(tdry).any() and np.nanmin(tdry) < 0 < np.nanmax(tdry):
            zero_idx = np.nanargmin(np.abs(tdry))
            ax.axhline(height_km[zero_idx], color="gray", lw=0.7, ls="-.")

    axes[0, 0].set_ylim(0, args.zmax_km)
    fig.suptitle(f"NSA C1 radiosondes, {args.date} "
                 "(blue shading: saturated layers, RH \N{GREATER-THAN OR EQUAL TO} 95%)")
    fig.tight_layout()
    finish_figure(fig, f"sonde_profiles_{args.date}.png", show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
