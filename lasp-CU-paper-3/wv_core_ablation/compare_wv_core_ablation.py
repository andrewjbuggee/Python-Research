"""
compare_wv_core_ablation.py — Compare the WV-core ablation retrains against the
published run_004 baseline and the redundant-continuum control.

All metrics are on the held-out TEST split. The per_level_rmse_um stored in each
summary.json is produced by predict_test() on test_loader (NOT training data),
and the per-profile RMSE used for the histograms is computed here from the same
test split.

Two figures are written into the ablation directory:

  wv_core_ablation_comparison.png
      Per-level absolute error for every run that exists (baseline / full /
      wv_core / continuum_control): the median |error| line with a 5-95%
      percentile band, matching per_level_error_percentiles.png (median |error|,
      NOT RMSE). Vertical level on the y-axis, level 1 = cloud top. Computed from
      the test-set predictions so the spread across profiles is shown.

  wv_core_ablation_test_rmse.png
      Focused full-vs-wv_core comparison from the actual test-set predictions:
        panel 1 — per-level test RMSE (vertical axis, level 1 = cloud top)
        panel 2 — per-profile RMSE histograms (semi-transparent), each
                  distribution's median marked with a dashed vertical line.
      'Full CLARREO spectrum' = run_004 (or run_full if present, 636 channels);
      'Wavelengths in water vapor absorption regions removed' = run_wv_core.
      Predictions are cached to pred_cache.npz in each run dir (built from
      best_model.pt on first call, like plot_per_level_error_percentiles.py),
      so re-runs are fast and need no torch.

Run:
    /opt/anaconda3/bin/python3 compare_wv_core_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# This script lives in repo_root/wv_core_ablation/; REPO is the repo root (two
# levels up). DEFAULT_ABLATION_DIR adds the wv_core_ablation/ segment back.
# Repo root is added to sys.path so the inference path can import data / models.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DEFAULT_ABLATION_DIR = REPO / "wv_core_ablation"
DEFAULT_BASELINE = (REPO / "hyper_parameter_sweep"
                    / "sweep_results_profile_only_synthetic_M0_rev2"
                    / "run_004" / "summary.json")
DEFAULT_H5 = REPO / "training_data" / "synthetic_training_data_7-levels_8_May_2026.h5"

# Labels requested for the focused full-vs-wv_core figure.
LABEL_FULL = "Full CLARREO spectrum"
LABEL_WV = "Wavelengths in water vapor absorption regions removed"

# Okabe-Ito CVD-safe palette (Wong 2011, Nat. Methods).
COLOR_FULL = "#0072B2"   # Wong blue — full spectrum
COLOR_WV = "#D55E00"     # vermillion — water-vapor cores removed
COLOR_CTRL = "#009E73"   # bluish green — continuum control
COLOR_BASE = "#000000"   # black — published baseline

# Display order and labels for the multi-run comparison figure.
RUNS = [
    ("baseline (run_004, 636 ch)", None),       # filled from DEFAULT_BASELINE
    ("full (re-baseline)", "run_full"),
    ("wv_core removed", "run_wv_core"),
    ("continuum_control removed", "run_continuum_control"),
]
# One consistent color per case across every figure: full = blue, WV cores
# removed = vermillion, continuum control = green.
RUN_COLORS = {
    "baseline (run_004, 636 ch)": COLOR_FULL,
    "full (re-baseline)": COLOR_FULL,
    "wv_core removed": COLOR_WV,
    "continuum_control removed": COLOR_CTRL,
}

# Pretty labels for the per-profile histogram legend.
LABEL_CONT = "Continuum control"
DISPLAY_LABELS = {
    "baseline (run_004, 636 ch)": LABEL_FULL,
    "full (re-baseline)": LABEL_FULL,
    "wv_core removed": "Water-vapor cores removed",   # concise for the value legend
    "continuum_control removed": LABEL_CONT,
}


def setup_style() -> None:
    """Compact serif style matching plot_per_level_error_percentiles.py."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
    })


# ---------------------------------------------------------------------------
# Summary-based multi-run comparison (no model load)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Dict]:
    return json.loads(path.read_text()) if path.exists() else None


def collect(ablation_dir: Path, baseline_json: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for label, subdir in RUNS:
        s = _load_json(baseline_json if subdir is None
                       else ablation_dir / subdir / "summary.json")
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

    print(f"{'run':<30}{'mean':>8}{'L1(top)':>9}{f'L{n_levels}(base)':>10}{'Δmean':>9}")
    print("-" * 66)
    for label, s in results.items():
        pl = s["per_level_rmse_um"]
        mean = s["mean_test_rmse_um"]
        dmean = "" if base_mean is None else f"{mean - base_mean:+.4f}"
        print(f"{label:<30}{mean:>8.4f}{pl[0]:>9.3f}{pl[-1]:>10.3f}{dmean:>9}")
    print("\nΔmean is vs the published run_004 baseline. Watch L1/L7: a WV-core "
          "degradation there that the continuum control does not show implicates "
          "the band cores in cloud-top/base sizing.")


def plot_comparison(runs_preds, out_png: Path, show_bands: bool = False) -> None:
    """Per-level absolute error: median |error| line per run, optional 5-95% band.

    Mirrors per_level_error_percentiles.png (median |error|, not RMSE) but
    overlays every run. `runs_preds` is a list of (label, pred[N,L], true[N,L],
    color). Vertical level on the y-axis, level 1 = cloud top.

    Bands are OFF by default: the runs are near-identical, so overlaying their
    5-95% spreads blends into one blob and stretches the x-axis out to the tail
    (~3.5 um), hiding the small inter-run median differences. Pass show_bands=True
    for the single-run-style banded view.
    """
    import matplotlib.pyplot as plt

    n_levels = runs_preds[0][1].shape[1]
    levels = np.arange(1, n_levels + 1)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for label, pred, true, color in runs_preds:
        abs_err = np.abs(pred - true)                      # (n_samples, n_levels)
        median_ae = np.median(abs_err, axis=0)
        if show_bands:
            p05, p95 = np.percentile(abs_err, (5, 95), axis=0)
            ax.fill_betweenx(levels, p05, p95, color=color, alpha=0.12, linewidth=0)
        ax.plot(median_ae, levels, "o-", color=color, lw=1.6, markersize=5,
                label=label)

    ax.set_ylabel(f"Vertical level (1 = cloud top, {n_levels} = cloud base)")
    ax.set_xlabel(r"Per-level median absolute error ($\mu$m)")
    ax.set_title("Median |error| per level"
                 + (" (shaded = 5-95%)" if show_bands else ""))
    ax.set_yticks(levels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")
    for label, pred, true, _ in runs_preds:
        ae = np.abs(pred - true)
        med = np.median(ae, axis=0)
        print(f"  {label:<32} median|err|: L1={med[0]:.3f}  "
              f"L{n_levels}={med[-1]:.3f}  (pooled median {np.median(ae):.3f} um)")


def gather_run_predictions(ablation_dir: Path, baseline_json: Path, h5_path: Path,
                           device: Optional[str], seed: int, refresh: bool):
    """Load/compute (label, pred, true, color) for every run that has outputs.

    A run is skipped (with a message) if it has neither a pred_cache.npz nor a
    best_model.pt + summary.json, or if torch is unavailable when a compute is
    required.
    """
    # run_full reproduces run_004; when it is present, use it as the single
    # full-spectrum reference and drop the duplicate published baseline so the
    # figure does not draw two identical, perfectly overlapping lines.
    run_full_dir = ablation_dir / "run_full"
    full_present = ((run_full_dir / "pred_cache.npz").exists()
                    or ((run_full_dir / "summary.json").exists()
                        and (run_full_dir / "best_model.pt").exists()))
    candidates = []
    if not full_present:
        candidates.append(("baseline (run_004, 636 ch)", baseline_json.parent))
    candidates += [
        ("full (re-baseline)", run_full_dir),
        ("wv_core removed", ablation_dir / "run_wv_core"),
        ("continuum_control removed", ablation_dir / "run_continuum_control"),
    ]
    runs = []
    for label, run_dir in candidates:
        has_cache = (run_dir / "pred_cache.npz").exists()
        has_model = ((run_dir / "summary.json").exists()
                     and (run_dir / "best_model.pt").exists())
        if not (has_cache or has_model):
            continue
        try:
            pred, true = load_or_compute_predictions(run_dir, h5_path, device,
                                                     seed, refresh)
        except ImportError as e:
            print(f"[skip] {label}: needs torch to compute predictions ({e})")
            continue
        runs.append((label, pred, true, RUN_COLORS.get(label)))
    return runs


# ---------------------------------------------------------------------------
# Prediction-based full-vs-wv_core figure (needs the trained models once)
# ---------------------------------------------------------------------------
def load_or_compute_predictions(
    run_dir: Path, h5_path: Path, device_arg: Optional[str],
    seed: int, refresh: bool, num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred[N,L], true[N,L]) in μm for a run's test split.

    Loads run_dir/pred_cache.npz if present (the published run_004 ships one);
    otherwise rebuilds the masked test split from summary.json + best_model.pt,
    runs inference, and writes the cache. Heavy torch imports are local so the
    cached path works without torch installed.
    """
    cache = run_dir / "pred_cache.npz"
    if cache.exists() and not refresh:
        d = np.load(cache)
        print(f"  {run_dir.name}: loaded cached predictions ({d['pred'].shape[0]} samples)")
        return d["pred"], d["true"]

    summary = _load_json(run_dir / "summary.json")
    ckpt_path = run_dir / "best_model.pt"
    if summary is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no pred_cache.npz and is missing summary.json or "
            f"best_model.pt — cannot produce predictions.")
    if "removed_spectral_idx" not in summary:
        raise KeyError(
            f"{run_dir}/summary.json lacks 'removed_spectral_idx'; this routine "
            f"only rebuilds predictions for train_wv_core_ablation.py runs.")

    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from data import LibRadtranDatasetExtras, create_profile_aware_splits
    from models import RetrievalConfig
    from models_profile_only_extras import ProfileOnlyNetworkExtras
    from train_standalone_profile_only_extras import predict_test

    n_spectral = 636
    n_extras = 3
    removed = np.asarray(summary["removed_spectral_idx"], dtype=int)
    keep_spectral = np.setdiff1d(np.arange(n_spectral), removed)        # into 0..635
    keep_full = np.concatenate([keep_spectral,
                                np.arange(n_spectral, n_spectral + 4 + n_extras)])
    hp = summary["hyperparams"]
    extras = summary["extras"]
    n_levels = len(summary["per_level_rmse_um"])

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  {run_dir.name}: inference on {len(keep_spectral)} kept channels "
          f"(device {device})")

    class _SpectralMasked(Dataset):
        def __init__(self, base, keep_idx):
            self.base = base
            self.keep = torch.as_tensor(keep_idx, dtype=torch.long)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, prof, tau = self.base[i]
            return x.index_select(0, self.keep), prof, tau

    full_ds = LibRadtranDatasetExtras(
        str(h5_path), normalize=True,
        instrument=summary["data"].get("instrument", "hysics"),
        zero_tau_c=extras["zero_tau_c"], zero_wv_above=extras["zero_wv_above"],
        zero_wv_in=extras["zero_wv_in"])
    _, _, test_idx = create_profile_aware_splits(
        str(h5_path), n_val_profiles=summary["n_val_profiles"],
        n_test_profiles=summary["n_test_profiles"], seed=seed)
    test_loader = DataLoader(
        _SpectralMasked(Subset(full_ds, test_idx), keep_full),
        batch_size=hp["batch_size"], shuffle=False, num_workers=num_workers)

    model_cfg = RetrievalConfig(
        n_wavelengths=len(keep_spectral), n_geometry_inputs=4, n_levels=n_levels,
        hidden_dims=tuple(hp["hidden_dims"]), dropout=hp["dropout"],
        activation=hp.get("activation", "gelu"))
    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=n_extras).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    out = predict_test(model, test_loader, device, model_cfg)
    pred, pred_std, true = out["pred"], out["pred_std"], out["true"]
    np.savez_compressed(cache, pred=pred, pred_std=pred_std, true=true,
                        h5_idx=np.asarray(test_idx))
    print(f"  {run_dir.name}: cached predictions -> {cache}")
    return pred, true


def per_level_rmse_um(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """sqrt(mean over samples of (pred-true)^2) per level → (n_levels,)."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def per_profile_rmse_um(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """sqrt(mean over levels of (pred-true)^2) per sample → (n_samples,)."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


def plot_test_rmse_and_hist(full: Tuple[np.ndarray, np.ndarray],
                            wv: Tuple[np.ndarray, np.ndarray],
                            out_png: Path) -> None:
    """Two-panel: per-level test RMSE (vertical) + per-profile RMSE histograms."""
    import matplotlib.pyplot as plt

    full_pred, full_true = full
    wv_pred, wv_true = wv

    pl_full = per_level_rmse_um(full_pred, full_true)
    pl_wv = per_level_rmse_um(wv_pred, wv_true)
    pp_full = per_profile_rmse_um(full_pred, full_true)
    pp_wv = per_profile_rmse_um(wv_pred, wv_true)
    n_levels = pl_full.shape[0]
    levels = np.arange(1, n_levels + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.0))

    # ── panel 1: per-level test RMSE, vertical axis (1 = cloud top) ──────
    ax1.plot(pl_full, levels, "o-", color=COLOR_FULL, markersize=4.5,
             label=LABEL_FULL)
    ax1.plot(pl_wv, levels, "s-", color=COLOR_WV, markersize=4.5, label=LABEL_WV)
    ax1.set_ylabel(f"Vertical level (1 = cloud top, {n_levels} = cloud base)")
    ax1.set_xlabel(r"Per-level test RMSE ($\mu$m)")
    ax1.set_title("Per-level $r_e$ test RMSE")
    ax1.set_yticks(levels)
    ax1.invert_yaxis()
    ax1.set_xlim(left=0)
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # ── panel 2: per-profile RMSE histograms, semi-transparent + medians ──
    bins = np.histogram_bin_edges(np.concatenate([pp_full, pp_wv]), bins=40)
    ax2.hist(pp_full, bins=bins, color=COLOR_FULL, alpha=0.5, label=LABEL_FULL)
    ax2.hist(pp_wv, bins=bins, color=COLOR_WV, alpha=0.5, label=LABEL_WV)
    med_full, med_wv = float(np.median(pp_full)), float(np.median(pp_wv))
    # Dashed vertical lines mark each median; the value is reported in a legend
    # (no inline text label drawn on the lines themselves).
    line_full = ax2.axvline(med_full, color=COLOR_FULL, linestyle="--",
                            linewidth=1.4, alpha=0.95,
                            label=f"Median = {med_full:.2f} $\\mu$m")
    line_wv = ax2.axvline(med_wv, color=COLOR_WV, linestyle="--",
                          linewidth=1.4, alpha=0.95,
                          label=f"Median = {med_wv:.2f} $\\mu$m")
    ax2.set_xlabel(r"Per-profile RMSE ($\mu$m)")
    ax2.set_ylabel("Number of test profiles")
    ax2.set_title("Per-profile test-RMSE distribution")
    ax2.set_xlim(left=0)
    ax2.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    # Legend defining the median values (line color matches each case's histogram).
    ax2.legend(handles=[line_full, line_wv], loc="upper right", fontsize=7.5,
               framealpha=0.9, title="Per-profile median")

    # Shared top legend identifying the two cases (applies to both panels).
    from matplotlib.patches import Patch
    case_handles = [Patch(facecolor=COLOR_FULL, alpha=0.55, label=LABEL_FULL),
                    Patch(facecolor=COLOR_WV, alpha=0.55, label=LABEL_WV)]
    fig.legend(handles=case_handles, loc="upper center", ncol=1, frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")
    print(f"  full  : mean per-level RMSE {pl_full.mean():.4f} μm, "
          f"per-profile median {med_full:.4f} μm")
    print(f"  wv_core: mean per-level RMSE {pl_wv.mean():.4f} μm, "
          f"per-profile median {med_wv:.4f} μm")


def plot_per_profile_hist(runs_preds, out_png: Path) -> None:
    """Standalone overlay of per-profile RMSE histograms for every run.

    Semi-transparent histograms + a dashed vertical line at each run's median;
    a single legend names each case and its median value. `runs_preds` is a list
    of (label, pred[N,L], true[N,L], color).
    """
    import matplotlib.pyplot as plt

    series = [(label, per_profile_rmse_um(pred, true), color)
              for label, pred, true, color in runs_preds]
    bins = np.histogram_bin_edges(
        np.concatenate([pp for _, pp, _ in series]), bins=40)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    median_handles = []
    for label, pp, color in series:
        ax.hist(pp, bins=bins, color=color, alpha=0.45)
        med = float(np.median(pp))
        line = ax.axvline(med, color=color, linestyle="--", linewidth=1.6,
                          alpha=0.95,
                          label=f"{DISPLAY_LABELS.get(label, label)} "
                                f"(median = {med:.2f} $\\mu$m)")
        median_handles.append(line)

    ax.set_xlabel(r"Per-profile RMSE ($\mu$m)")
    ax.set_ylabel("Number of test profiles")
    ax.set_title("Per-profile test-RMSE distribution by ablation")
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(handles=median_handles, loc="upper right", fontsize=8.5,
              framealpha=0.9)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")
    for label, pp, _ in series:
        print(f"  {label:<32} per-profile RMSE: median {np.median(pp):.3f}  "
              f"mean {pp.mean():.3f} um")


def resolve_full_dir(ablation_dir: Path, baseline_json: Path) -> Path:
    """Prefer a same-pipeline run_full if present; else the published run_004."""
    run_full = ablation_dir / "run_full"
    if (run_full / "pred_cache.npz").exists() or (run_full / "best_model.pt").exists():
        return run_full
    return baseline_json.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ablation-dir", type=str, default=str(DEFAULT_ABLATION_DIR))
    p.add_argument("--baseline-json", type=str, default=str(DEFAULT_BASELINE))
    p.add_argument("--h5-path", type=str, default=str(DEFAULT_H5),
                   help="HDF5 used to rebuild test predictions (local copy).")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--seed", type=int, default=42, help="Must match training (42).")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore pred_cache.npz and re-run inference.")
    p.add_argument("--bands", action="store_true",
                   help="Overlay 5-95 percent |error| bands on the comparison "
                        "figure (off by default; readable only for 1-2 runs).")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    setup_style()
    ablation_dir = Path(args.ablation_dir)
    baseline_json = Path(args.baseline_json)

    # --- quick RMSE table from the summaries ---
    report(collect(ablation_dir, baseline_json))
    if args.no_plot:
        return

    # --- test-set predictions for every available run (cached after first call) ---
    h5_path = Path(args.h5_path)
    runs_preds = gather_run_predictions(ablation_dir, baseline_json, h5_path,
                                        args.device, args.seed, args.refresh)
    if not runs_preds:
        print("\n[skip] No predictions available (need pred_cache.npz, or "
              "best_model.pt + torch). Figures not produced.")
        return

    # Q1: per-level median |error| (paper style), all runs overlaid.
    plot_comparison(runs_preds, ablation_dir / "wv_core_ablation_comparison.png",
                    show_bands=args.bands)

    # Q2: focused full-vs-wv_core RMSE + per-profile RMSE histogram.
    by_label = {lbl: (pred, true) for lbl, pred, true, _ in runs_preds}
    full_dir = resolve_full_dir(ablation_dir, baseline_json)
    full_label = ("full (re-baseline)" if full_dir.name == "run_full"
                  else "baseline (run_004, 636 ch)")
    if full_label in by_label and "wv_core removed" in by_label:
        plot_test_rmse_and_hist(by_label[full_label], by_label["wv_core removed"],
                                ablation_dir / "wv_core_ablation_test_rmse.png")
    else:
        print("\n[skip] need both the full-spectrum and wv_core runs for the "
              "full-vs-wv_core figure.")

    # Standalone 3-way per-profile RMSE histogram (full / wv_core / continuum).
    plot_per_profile_hist(runs_preds,
                          ablation_dir / "wv_core_ablation_per_profile_hist.png")


if __name__ == "__main__":
    main()
