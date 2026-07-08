#!/usr/bin/env python3
"""Plot downwelling surface radiation from the QCRAD best-estimate product.

Time series of downwelling longwave and shortwave irradiance, plus the
downwelling-longwave histogram. In Arctic winter the LW-down distribution is
famously bimodal (Stramler et al. 2011): a "radiatively clear" mode and an
"opaquely cloudy" mode separated by several tens of W/m^2 -- the direct,
plottable signature of the liquid-containing clouds that MPCT proposes to
thin. During polar night (roughly 18 Nov - 23 Jan at Utqiagvik) SW-down is
zero by geometry.

NOTE: QCRAD is an extension beyond the Hartig26 instrument list (see README);
download it with:  python scripts/download_nsa_data.py --datastreams qcrad ...

Example
-------
python scripts/plot_downwelling_radiation.py --start 2022-01-01 --end 2022-01-31
"""

from __future__ import annotations

import argparse

from _plot_common import finish_figure, use_headless_backend_if_needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--no-show", action="store_true", help="save only")
    args = parser.parse_args()
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np

    from arm_nsa.surface import read_qcrad

    ds = read_qcrad(args.start, args.end)

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 2, 1.6]}
    )

    ax = axes[0]
    ax.plot(ds["time"], ds["lwdn_w_m2"], lw=0.5, color="tab:red",
            label="LW down")
    ax.plot(ds["time"], ds["lwup_w_m2"], lw=0.5, color="tab:gray", alpha=0.8,
            label="LW up")
    ax.set_ylabel("longwave [W m$^{-2}$]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"NSA C1 broadband radiation (QCRAD), {args.start} to {args.end}")

    ax = axes[1]
    ax.plot(ds["time"], ds["swdn_w_m2"], lw=0.5, color="tab:orange",
            label="SW down (best estimate)")
    ax.set_ylabel("shortwave [W m$^{-2}$]")
    ax.set_xlabel("time (UTC)")
    ax.legend(loc="upper right", fontsize=8)

    # LW-down histogram: look for the two-state Arctic winter distribution.
    ax = axes[2]
    lwdn = ds["lwdn_w_m2"].values
    lwdn = lwdn[np.isfinite(lwdn)]
    if lwdn.size:
        ax.hist(lwdn, bins=60, color="tab:red", alpha=0.75)
    ax.set_xlabel("downwelling longwave [W m$^{-2}$]")
    ax.set_ylabel("count")
    ax.set_title("LW-down distribution (bimodal = clear vs. cloudy states)",
                 fontsize=9)

    fig.tight_layout()
    finish_figure(
        fig, f"downwelling_radiation_{args.start}_{args.end}.png",
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
