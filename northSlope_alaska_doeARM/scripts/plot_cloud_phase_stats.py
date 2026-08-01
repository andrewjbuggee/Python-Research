#!/usr/bin/env python3
"""Sky-state / cloud-phase occurrence statistics from the coordinated library.

Reads a library file produced by build_sonde_library.py and shows:
  (a) overall occurrence of each sky-state category (see arm_nsa/phase.py),
  (b) occurrence by month -- the MPCT question is specifically about how often
      seedable supercooled liquid is present in the cold season,
  (c) the LWP distribution split by category, as a sanity check on the
      classification thresholds.

Example
-------
python scripts/plot_cloud_phase_stats.py \
    --library data/processed/nsa_sonde_coordinated_library.20220101_20220131.nc
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
    parser.add_argument("--library", required=True,
                        help="path to a coordinated-library netCDF file")
    parser.add_argument("--no-show", action="store_true", help="save only")
    add_data_root_argument(parser)
    args = parser.parse_args()
    apply_data_root(args.data_root)
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from arm_nsa.phase import PHASE_CODES, classify_phase, phase_occurrence

    library = xr.open_dataset(args.library)
    if "phase_code" in library:
        phase_code = library["phase_code"]
    else:
        phase_code = classify_phase(library)

    stats = phase_occurrence(phase_code)
    print("Occurrence over classified soundings:")
    for name, value in stats.items():
        print(f"  {name}: {value:.1%}" if name != "n_classified"
              else f"  {name}: {value:.0f}")

    colors = {
        "clear": "lightgray",
        "ice_probable": "steelblue",
        "liquid_probable": "gold",
        "liquid_confident": "orangered",
    }
    names = [n for n in PHASE_CODES.values() if n != "missing"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) overall occurrence
    ax = axes[0]
    fractions = [stats.get(n, 0.0) for n in names]
    ax.bar(range(len(names)), fractions, color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("occurrence fraction")
    ax.set_title(f"sky state, n={stats.get('n_classified', 0):.0f} soundings")
    ax.axhline(0, color="k", lw=0.5)

    # (b) by month (liquid-containing = probable + confident)
    ax = axes[1]
    months = phase_code["launch_time"].dt.month.values
    codes = phase_code.values
    uniq_months = sorted(set(months.tolist()))
    bottoms = np.zeros(len(uniq_months))
    for code, name in PHASE_CODES.items():
        if name == "missing":
            continue
        month_fracs = []
        for m in uniq_months:
            in_month = (months == m) & (codes != 0)
            month_fracs.append(
                float((codes[in_month] == code).mean()) if in_month.any() else 0.0
            )
        ax.bar(range(len(uniq_months)), month_fracs, bottom=bottoms,
               color=colors[name], label=name)
        bottoms += np.asarray(month_fracs)
    ax.set_xticks(range(len(uniq_months)))
    ax.set_xticklabels(uniq_months)
    ax.set_xlabel("month")
    ax.set_ylabel("fraction")
    ax.set_title("sky state by month")
    ax.legend(fontsize=7, loc="lower right")

    # (c) LWP by category
    ax = axes[2]
    lwp = library["lwp_g_m2"].values
    bins = np.linspace(-20, 300, 40)
    for code, name in PHASE_CODES.items():
        if name in ("missing", "clear"):
            continue
        sel = np.isfinite(lwp) & (codes == code)
        if sel.any():
            ax.hist(lwp[sel], bins=bins, histtype="step", lw=1.4,
                    color=colors[name], label=name, density=True)
    ax.set_xlabel("LWP [g m$^{-2}$]")
    ax.set_ylabel("probability density")
    ax.set_title("LWP by sky state")
    ax.legend(fontsize=7)

    fig.tight_layout()
    stem = args.library.rsplit("/", 1)[-1].replace(".nc", "")
    finish_figure(fig, f"cloud_phase_stats_{stem}.png", show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
