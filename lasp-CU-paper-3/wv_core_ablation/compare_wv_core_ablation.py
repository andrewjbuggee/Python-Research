"""
compare_wv_core_ablation.py — Compare the WV-core ablation retrains against the
published run_004 baseline and the redundant-continuum control.

Reads per-run summary.json files written by train_wv_core_ablation.py plus the
published run_004 summary, then reports per-level RMSE and writes a figure. The
levels run cloud-top (L1) -> cloud-base (L7); King & Vaughan reported little
information on r_e at cloud top and base from the 1150/1900 nm bands, so L1 and
L7 are the diagnostic levels — if removing the WV cores degrades L1/L7 markedly
while the continuum control does not, the cores carried profile information.

Run:
    /opt/anaconda3/bin/python3 compare_wv_core_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# This script lives in repo_root/wv_core_ablation/; REPO is the repo root (two
# levels up). DEFAULT_ABLATION_DIR adds the wv_core_ablation/ segment back.
REPO = Path(__file__).resolve().parent.parent
DEFAULT_ABLATION_DIR = REPO / "wv_core_ablation"
DEFAULT_BASELINE = (REPO / "hyper_parameter_sweep"
                    / "sweep_results_profile_only_synthetic_M0_rev2"
                    / "run_004" / "summary.json")

# Display order and labels for the runs we know about.
RUNS = [
    ("baseline (run_004, 636 ch)", None),       # filled from DEFAULT_BASELINE
    ("full (re-baseline)", "run_full"),
    ("wv_core removed", "run_wv_core"),
    ("continuum_control removed", "run_continuum_control"),
]


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect(ablation_dir: Path, baseline_json: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for label, subdir in RUNS:
        if subdir is None:
            s = _load(baseline_json)
        else:
            s = _load(ablation_dir / subdir / "summary.json")
        if s is not None and "per_level_rmse_um" in s:
            results[label] = s
    return results


def report(results: Dict[str, Dict]) -> None:
    if not results:
        print("No runs found yet. Train at least one mask first.")
        return
    n_levels = len(next(iter(results.values()))["per_level_rmse_um"])
    base = results.get("baseline (run_004, 636 ch)")
    base_mean = base["mean_test_rmse_um"] if base else None

    hdr = ["run", "mean", "L1(top)", f"L{n_levels}(base)"] + ["Δmean"]
    print(f"{hdr[0]:<30}{hdr[1]:>8}{hdr[2]:>9}{hdr[3]:>10}{hdr[4]:>9}")
    print("-" * 66)
    for label, s in results.items():
        pl = s["per_level_rmse_um"]
        mean = s["mean_test_rmse_um"]
        dmean = "" if base_mean is None else f"{mean - base_mean:+.4f}"
        print(f"{label:<30}{mean:>8.4f}{pl[0]:>9.3f}{pl[-1]:>10.3f}{dmean:>9}")
    print("\nΔmean is vs the published run_004 baseline. Watch L1/L7: a WV-core "
          "degradation there that the continuum control does not show implicates "
          "the band cores in cloud-top/base sizing.")


def plot(results: Dict[str, Dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    n_levels = len(next(iter(results.values()))["per_level_rmse_um"])
    levels = np.arange(1, n_levels + 1)
    colors = {"baseline (run_004, 636 ch)": "k",
              "full (re-baseline)": "tab:green",
              "wv_core removed": "tab:red",
              "continuum_control removed": "tab:blue"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    for label, s in results.items():
        pl = np.asarray(s["per_level_rmse_um"])
        c = colors.get(label, None)
        style = "--" if "baseline" in label else "-"
        axL.plot(levels, pl, style, marker="o", color=c, label=label, lw=1.8)
    axL.set_xlabel("level (1 = cloud top → cloud base)")
    axL.set_ylabel("per-level RMSE [μm]")
    axL.set_title("Per-level r_e RMSE by ablation")
    axL.set_xticks(levels)
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)

    labels = list(results.keys())
    means = [results[k]["mean_test_rmse_um"] for k in labels]
    bar_colors = [colors.get(k, "gray") for k in labels]
    axR.bar(range(len(labels)), means, color=bar_colors)
    axR.set_xticks(range(len(labels)))
    axR.set_xticklabels([k.split(" (")[0].replace(" removed", "")
                         for k in labels], rotation=30, ha="right", fontsize=8)
    axR.set_ylabel("mean RMSE [μm]")
    axR.set_title("Mean RMSE")
    axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ablation-dir", type=str, default=str(DEFAULT_ABLATION_DIR))
    p.add_argument("--baseline-json", type=str, default=str(DEFAULT_BASELINE))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    results = collect(Path(args.ablation_dir), Path(args.baseline_json))
    report(results)
    if results and not args.no_plot:
        plot(results, Path(args.ablation_dir) / "wv_core_ablation_comparison.png")


if __name__ == "__main__":
    main()
