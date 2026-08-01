#!/usr/bin/env python3
"""Cloud-phase occurrence and per-phase downwelling longwave radiation.

THE question this answers (first target of the MPCT project): how often are
clear skies, ice-only, mixed-phase, and liquid-only clouds present at NSA, and
what does the downwelling longwave radiation at the surface look like under
each?

Method (following Bertrand et al. 2025, Nat. Commun. 16, 9135)
--------------------------------------------------------------
1. The Shupe-Turner multi-sensor product (nsamicrobase2shupeturnC1.c1, ~1-min
   resolution, ~2004-2019) provides a vertically resolved hydrometeor phase
   code from cloud radar + lidar depolarization + MWR + sondes.
2. Each ~1-min profile is collapsed to one scene type using the Bertrand25
   column rules (see arm_nsa/shupe_turner.py): mixed-phase if a mixed volume
   exists or liquid and ice both occur in the column; liquid-only if liquid
   without ice; ice-only if ice without liquid; clear if no hydrometeors.
3. Each scene time is paired with the nearest QCRAD broadband flux sample
   (within a small tolerance -- both records are ~1-min), giving per-scene
   distributions of downwelling LW (and net LW when upwelling is present).

Interpretation notes
--------------------
* The wintertime net-LW distribution is bimodal (radiatively clear vs.
  opaquely cloudy states, Stramler et al. 2011); expect the liquid-containing
  scenes to occupy the opaque mode (net LW near 0 W/m^2, LW-down within a few
  tens of W/m^2 of LW-up) and clear/thin-ice scenes the clear mode
  (net LW around -40 to -50 W/m^2).
* Ice-only scenes span thin-to-opaque; Bertrand25 showed their opacity change
  drives most of the recent LW-down increase, so their distribution shape is
  of particular interest, not just the mean.
* Scene frequencies are reported over classified (non-missing) minutes;
  Shupe-Turner has real gaps (instrument upgrades), so also check n.

Prerequisites
-------------
    python scripts/download_nsa_data.py --datastreams shupeturn qcrad \
        --start 2015-01-01 --end 2015-03-31

Examples
--------
# One winter (Jan-Mar) at extended-winter months default:
python scripts/analyze_phase_radiation.py --start 2015-01-01 --end 2015-03-31

# Reproduce the Bertrand25 December-March season definition:
python scripts/analyze_phase_radiation.py --start 2014-12-01 --end 2015-03-31 \
    --months 12,1,2,3

Outputs
-------
figures/phase_radiation_<start>_<end>.png      4-panel summary figure
data/processed/phase_radiation_<...>.nc        scene codes + paired fluxes
data/processed/phase_radiation_stats_<...>.csv per-scene statistics table
"""

from __future__ import annotations

import argparse

from _plot_common import (
    add_data_root_argument,
    apply_data_root,
    finish_figure,
    use_headless_backend_if_needed,
)

SCENE_COLORS = {
    "clear": "lightgray",
    "ice_only": "steelblue",
    "mixed_phase": "gold",
    "liquid_only": "orangered",
    "other_hydrometeor": "plum",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--months",
        default="11,12,1,2,3",
        help="comma-separated months to keep (default 11,12,1,2,3 = extended "
        "winter per Hartig26; use 12,1,2,3 for the Bertrand25 season)",
    )
    parser.add_argument(
        "--tolerance",
        default="3min",
        help="max scene-to-flux time offset when pairing (default 3min)",
    )
    parser.add_argument("--no-show", action="store_true", help="save only")
    add_data_root_argument(parser)
    args = parser.parse_args()
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np

    from arm_nsa import config
    from arm_nsa.shupe_turner import (
        SCENE_CODES,
        classify_scene,
        pair_scene_with_flux,
        read_shupe_turner,
        restrict_to_months,
        scene_flux_stats,
        scene_occurrence,
    )
    from arm_nsa.surface import read_qcrad

    apply_data_root(args.data_root)

    months = tuple(int(m) for m in args.months.split(","))

    # ------------------------------------------------------------------ load
    print("reading Shupe-Turner phase product ...")
    st = read_shupe_turner(args.start, args.end)
    st = restrict_to_months(st, months)
    scene = classify_scene(st)

    print("reading QCRAD broadband fluxes ...")
    qcrad = read_qcrad(args.start, args.end)
    paired = pair_scene_with_flux(scene, qcrad, tolerance=args.tolerance)

    # ------------------------------------------------------- occurrence table
    occ = scene_occurrence(scene)
    print(f"\nScene occurrence, months={months}, "
          f"{args.start}..{args.end} "
          f"(n={occ.get('n_classified', 0):.0f} classified minutes):")
    for name in SCENE_CODES.values():
        if name in occ:
            print(f"  {name:<18s} {occ[name]:6.1%}")
    print(f"  {'liquid-containing':<18s} {occ.get('liquid_containing_any', 0):6.1%}"
          "   (mixed + liquid-only)")

    # Monthly breakdown, printed for the record.
    print("\nOccurrence by month:")
    month_values = scene["time"].dt.month.values
    codes = scene.values
    for m in months:
        in_month = (month_values == m) & (codes != 0)
        n_month = int(in_month.sum())
        if n_month == 0:
            print(f"  month {m:>2d}: no classified data")
            continue
        parts = [
            f"{name} {(codes[in_month] == code).mean():5.1%}"
            for code, name in SCENE_CODES.items()
            if code != 0
        ]
        print(f"  month {m:>2d} (n={n_month:>7d}): " + ", ".join(parts))

    # ------------------------------------------------------- flux statistics
    stats_lwdn = scene_flux_stats(paired, "lwdn_w_m2")
    print("\nDownwelling LW [W/m^2] by scene:")
    print(stats_lwdn.round(1).to_string())
    has_net = "lwnet_w_m2" in paired
    if has_net:
        stats_net = scene_flux_stats(paired, "lwnet_w_m2")
        print("\nNet LW (down - up) [W/m^2] by scene:")
        print(stats_net.round(1).to_string())

    # ---------------------------------------------------------------- figure
    plot_scenes = [n for n in SCENE_CODES.values() if n != "missing"]
    # Drop empty categories from the distribution panels to reduce noise.
    populated = [
        n for n in plot_scenes
        if n in stats_lwdn.index and stats_lwdn.loc[n, "n"] > 0
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) occurrence
    ax = axes[0, 0]
    fracs = [occ.get(n, 0.0) for n in plot_scenes]
    ax.bar(range(len(plot_scenes)), fracs,
           color=[SCENE_COLORS[n] for n in plot_scenes], edgecolor="k", lw=0.4)
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.01, f"{f:.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(plot_scenes)))
    ax.set_xticklabels(plot_scenes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("occurrence fraction")
    ax.set_title(
        f"scene occurrence (n={occ.get('n_classified', 0):.0f} min)", fontsize=10
    )

    # (b) LW-down distributions
    ax = axes[0, 1]
    lwdn = paired["lwdn_w_m2"].values.astype(float)
    bins = np.linspace(120, 340, 56)
    for name in populated:
        code = [c for c, n in SCENE_CODES.items() if n == name][0]
        sel = (paired["scene_code"].values == code) & np.isfinite(lwdn)
        if sel.sum() > 10:
            ax.hist(lwdn[sel], bins=bins, histtype="step", lw=1.6,
                    density=True, color=SCENE_COLORS[name], label=name)
    ax.set_xlabel("downwelling LW [W m$^{-2}$]")
    ax.set_ylabel("probability density")
    ax.set_title("LW-down by scene", fontsize=10)
    ax.legend(fontsize=7)

    # (c) LW-down summary (box-style from percentiles already computed)
    ax = axes[1, 0]
    for i, name in enumerate(populated):
        row = stats_lwdn.loc[name]
        ax.add_patch(plt.Rectangle((i - 0.3, row["p25"]), 0.6,
                                   row["p75"] - row["p25"],
                                   facecolor=SCENE_COLORS[name],
                                   edgecolor="k", lw=0.6, alpha=0.9))
        ax.plot([i - 0.3, i + 0.3], [row["median"]] * 2, "k-", lw=1.4)
        ax.plot([i, i], [row["p10"], row["p25"]], "k-", lw=0.8)
        ax.plot([i, i], [row["p75"], row["p90"]], "k-", lw=0.8)
    ax.set_xticks(range(len(populated)))
    ax.set_xticklabels(populated, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("downwelling LW [W m$^{-2}$]")
    ax.set_title("LW-down: median, 25-75% box, 10-90% whiskers", fontsize=10)

    # (d) net LW distributions (bimodal clear/opaque context), if available
    ax = axes[1, 1]
    if has_net:
        lwnet = paired["lwnet_w_m2"].values.astype(float)
        bins_net = np.linspace(-80, 20, 51)
        for name in populated:
            code = [c for c, n in SCENE_CODES.items() if n == name][0]
            sel = (paired["scene_code"].values == code) & np.isfinite(lwnet)
            if sel.sum() > 10:
                ax.hist(lwnet[sel], bins=bins_net, histtype="step", lw=1.6,
                        density=True, color=SCENE_COLORS[name], label=name)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("net LW, down $-$ up [W m$^{-2}$]")
        ax.set_ylabel("probability density")
        ax.set_title("net LW by scene (clear vs. opaque states)", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no upwelling LW available\n(net LW panel skipped)",
                ha="center", va="center", fontsize=9)

    fig.suptitle(
        f"NSA C1 cloud phase and surface longwave radiation, "
        f"{args.start} to {args.end} (months {args.months})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # ---------------------------------------------------------------- output
    stem = f"{args.start}_{args.end}_m{args.months.replace(',', '-')}"
    finish_figure(fig, f"phase_radiation_{stem}.png", show=not args.no_show)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nc_path = config.PROCESSED_DATA_DIR / f"phase_radiation_{stem}.nc"
    out = paired.copy()
    out.attrs.update(
        title="Shupe-Turner scene classification paired with QCRAD fluxes",
        methodology="Bertrand et al. 2025 scene rules; see arm_nsa/shupe_turner.py",
        months=str(months),
    )
    out.to_netcdf(nc_path)
    csv_path = config.PROCESSED_DATA_DIR / f"phase_radiation_stats_{stem}.csv"
    stats_lwdn.assign(variable="lwdn_w_m2").to_csv(csv_path)
    if has_net:
        stats_net.assign(variable="lwnet_w_m2").to_csv(
            csv_path, mode="a", header=False
        )
    print(f"\nclassified series written: {nc_path}")
    print(f"statistics table written:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
