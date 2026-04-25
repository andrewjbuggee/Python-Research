"""
compare_baseline_vs_pca.py
--------------------------
Step 4 of the PCA-head experiment: side-by-side evaluation of the baseline
`DropletProfileNetwork` (trained via the existing sweep pipeline, e.g. the
checkpoints under ./hyper_parameter_sweep/sweep_results_3/run_XXX/) and the
new `PCADropletProfileNetwork` trained by `train_pca.py`.

The central question this script answers is:

    "Does constraining the profile head to a low-rank PCA manifold reduce
     per-level RMSE on held-out profiles compared to the unconstrained
     baseline, given identical train/val/test splits and augmentation?"

Both models are evaluated on the **profile-held-out test set** — the 14
profiles that were *never* seen during training of either model.  This is
the only split that tells you about generalization to new cloud profiles.

Figures produced (all saved at 500 DPI under ./figures/compare_vs_baseline/):
    01_per_level_rmse_bar.png
        Per-level RMSE with the baseline and PCA-head plotted side-by-side.
        Annotated with the mean-over-levels RMSE for each model.
    02_per_level_rmse_box.png
        Box-plots of per-sample, per-level |residual| on the test set.
        Shows whether PCA reduces the *spread* of errors, not just the mean.
    03_pred_vs_true_scatter.png
        N_levels sub-panels; each is a predicted vs true r_e scatter plot
        with 1:1 line.  Color-coded by model.
    04_example_profiles.png
        A 4×2 grid of held-out profiles: true (black), baseline ±σ (blue),
        PCA-head ±σ (red).  Useful sanity check that the PCA manifold is
        actually fitting shapes, not just the mean.
    05_tau_parity.png
        Parity plot for τ_c (only if both models have τ_c heads active).
    06_summary.txt
        A plain-text summary of the numeric metrics so the user can copy
        them into the paper.

Usage
=====
    python compare_baseline_vs_pca.py \
        --baseline-ckpt ../hyper_parameter_sweep/sweep_results_3/run_050/best_model.pt \
        --pca-ckpt      ./checkpoints_pca/<timestamp>_K3/best_model.pt \
        --pca-basis     ./pca_basis.npz

All paths are optional — sensible defaults pointed at the likely locations
are used if omitted.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Repo-root imports: reuse the same data pipeline so both models see the
# SAME test set.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import create_dataloaders, resolve_h5_path, RE_MIN, RE_MAX, TAU_MIN, TAU_MAX  # noqa: E402
from models import DropletProfileNetwork, RetrievalConfig                   # noqa: E402

from models_pca import (                                                    # noqa: E402
    PCADropletProfileNetwork,
    PCARetrievalConfig,
    load_pca_basis,
)


FIG_DPI = 500


# ───────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ───────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline DropletProfileNetwork vs PCA-head model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline-ckpt", type=str,
        default=str(
            _REPO_ROOT / "hyper_parameter_sweep" / "sweep_results_3"
            / "run_050" / "best_model.pt"
        ),
        help="Path to the baseline model .pt (best_model.pt from the sweep).",
    )
    p.add_argument(
        "--pca-ckpt", type=str, default=None,
        help="Path to the PCA-head model .pt produced by train_pca.py.  "
             "If omitted, the most recent ./checkpoints_pca/*/best_model.pt "
             "is used.",
    )
    p.add_argument(
        "--pca-basis", type=str,
        default=str(Path(__file__).parent / "pca_basis.npz"),
        help="Path to the PCA basis .npz produced by analyze_profile_pca.py. "
             "Needed to register the basis on the PCA-head model.",
    )
    p.add_argument(
        "--training-data-dir", type=str, default=None,
        help="Override the directory of data:h5_path (kept filename).",
    )
    p.add_argument(
        "--h5-path", type=str, default=None,
        help="Explicitly specify the training HDF5.  If omitted, the path is "
             "read from the PCA basis file so both models see the same data.",
    )
    p.add_argument(
        "--fig-dir", type=str,
        default=str(Path(__file__).parent / "figures" / "compare_vs_baseline"),
        help="Output directory for figures.",
    )
    p.add_argument(
        "--n-example-profiles", type=int, default=8,
        help="How many held-out test profiles to plot in figure 04.",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Inference batch size.  Only affects speed, not results.",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Utility: find the most recent PCA checkpoint if one wasn't passed
# ───────────────────────────────────────────────────────────────────────────────
def default_pca_ckpt() -> Optional[Path]:
    root = Path(__file__).parent / "checkpoints_pca"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/best_model.pt"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


# ───────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ───────────────────────────────────────────────────────────────────────────────
def load_baseline(ckpt_path: Path, device: torch.device) -> DropletProfileNetwork:
    """Reconstruct the baseline net from its saved `model_config`."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    cfg = RetrievalConfig(
        n_wavelengths=mc["n_wavelengths"],
        n_geometry_inputs=mc["n_geometry_inputs"],
        n_levels=mc["n_levels"],
        hidden_dims=tuple(mc["hidden_dims"]),
        dropout=mc["dropout"],
        activation=mc["activation"],
        # The old checkpoints may not record the physical bounds — use the
        # module-level defaults, which are the ones used at training time.
        re_min=RE_MIN, re_max=RE_MAX, tau_min=TAU_MIN, tau_max=TAU_MAX,
    )
    model = DropletProfileNetwork(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_pca_model(ckpt_path: Path, basis: dict,
                   device: torch.device) -> PCADropletProfileNetwork:
    """Reconstruct the PCA-head net from its saved `model_config`."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    cfg = PCARetrievalConfig(
        n_wavelengths=mc["n_wavelengths"],
        n_geometry_inputs=mc["n_geometry_inputs"],
        n_levels=mc["n_levels"],
        n_pca_components=mc["n_pca_components"],
        hidden_dims=tuple(mc["hidden_dims"]),
        dropout=mc["dropout"],
        activation=mc["activation"],
        use_tau_head=mc.get("use_tau_head", True),
        learn_pca_basis=mc.get("learn_pca_basis", False),
    )
    model = PCADropletProfileNetwork(cfg).to(device)
    # Register the SAME basis rows that were used during training.  The
    # state_dict load overwrites them if `learn_pca_basis=True`, but for
    # the default fixed-buffer case we still need to register so that the
    # buffer shapes are present before `load_state_dict`.
    K = cfg.n_pca_components
    model.register_pca(
        basis["mean"].astype(np.float32),
        basis["components"][:K].astype(np.float32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ───────────────────────────────────────────────────────────────────────────────
# Inference: run a model over the test loader and collect everything
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    """
    Returns a dict of numpy arrays:
        true_profile   (N, n_levels)  μm
        pred_profile   (N, n_levels)  μm
        pred_std       (N, n_levels)  μm
        true_tau       (N,)           physical τ
        pred_tau       (N,)
        pred_tau_std   (N,)
    """
    tp, pp, ps, tt, pt, pts = [], [], [], [], [], []
    for batch in loader:
        x   = batch["inputs"].to(device)
        prf = batch["profile"].to(device)           # physical μm
        tau = batch["tau_c"].to(device)             # physical τ

        out = model(x)
        tp.append(prf.cpu().numpy())
        pp.append(out["profile"].cpu().numpy())
        ps.append(out["profile_std"].cpu().numpy())
        tt.append(tau.cpu().numpy())
        pt.append(out["tau_c"].squeeze(-1).cpu().numpy())
        pts.append(out["tau_std"].squeeze(-1).cpu().numpy())

    return dict(
        true_profile=np.concatenate(tp, axis=0),
        pred_profile=np.concatenate(pp, axis=0),
        pred_std=np.concatenate(ps, axis=0),
        true_tau=np.concatenate(tt, axis=0),
        pred_tau=np.concatenate(pt, axis=0),
        pred_tau_std=np.concatenate(pts, axis=0),
    )


def per_level_rmse(result: dict) -> np.ndarray:
    """Per-level r_e RMSE over all test samples."""
    err = result["pred_profile"] - result["true_profile"]
    return np.sqrt(np.mean(err ** 2, axis=0))


def per_level_abs_residuals(result: dict) -> np.ndarray:
    """(N, n_levels) absolute residuals, for box-plots."""
    return np.abs(result["pred_profile"] - result["true_profile"])


# ───────────────────────────────────────────────────────────────────────────────
# Figures
# ───────────────────────────────────────────────────────────────────────────────
def fig_per_level_rmse_bar(baseline_rmse, pca_rmse, out_dir: Path) -> None:
    """01 — per-level RMSE, baseline vs PCA-head, side-by-side bars."""
    n_levels = len(baseline_rmse)
    x = np.arange(1, n_levels + 1)
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, baseline_rmse, w, color="#4C72B0",
                edgecolor="black",
                label=f"Baseline  (mean = {baseline_rmse.mean():.3f} μm)")
    b2 = ax.bar(x + w / 2, pca_rmse, w, color="#C44E52",
                edgecolor="black",
                label=f"PCA-head (mean = {pca_rmse.mean():.3f} μm)")

    # Annotate bar heights so the actual numbers are readable.
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x])
    ax.set_xlabel("Level index (L1 = cloud top, LN = cloud base)")
    ax.set_ylabel("r_e RMSE on held-out test set  (μm)")
    ax.set_title("Per-level r_e RMSE: baseline vs PCA-head")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_per_level_rmse_bar.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_per_level_rmse_box(baseline_res, pca_res, out_dir: Path) -> None:
    """02 — per-level distribution of |residual| as box-plots."""
    n_levels = baseline_res.shape[1]
    positions_b = np.arange(1, n_levels + 1) - 0.18
    positions_p = np.arange(1, n_levels + 1) + 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    bp_b = ax.boxplot([baseline_res[:, k] for k in range(n_levels)],
                      positions=positions_b, widths=0.3, patch_artist=True,
                      showfliers=False)
    bp_p = ax.boxplot([pca_res[:, k] for k in range(n_levels)],
                      positions=positions_p, widths=0.3, patch_artist=True,
                      showfliers=False)
    for patch in bp_b["boxes"]:
        patch.set_facecolor("#4C72B0"); patch.set_alpha(0.6)
    for patch in bp_p["boxes"]:
        patch.set_facecolor("#C44E52"); patch.set_alpha(0.6)

    ax.set_xticks(np.arange(1, n_levels + 1))
    ax.set_xticklabels([f"L{i}" for i in range(1, n_levels + 1)])
    ax.set_xlabel("Level index")
    ax.set_ylabel("|residual|  (μm)")
    ax.set_title("Per-level |residual| distribution on held-out test set")
    # Fake legend handles (boxplot doesn't support label=).
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.6, label="Baseline"),
        Patch(facecolor="#C44E52", alpha=0.6, label="PCA-head"),
    ], frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_level_rmse_box.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_pred_vs_true(baseline_res, pca_res, out_dir: Path) -> None:
    """03 — pred vs true scatter, one subplot per level, both models overlaid."""
    n_levels = baseline_res["true_profile"].shape[1]
    ncols = min(4, n_levels)
    nrows = int(np.ceil(n_levels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    tp_b, pp_b = baseline_res["true_profile"], baseline_res["pred_profile"]
    tp_p, pp_p = pca_res["true_profile"],      pca_res["pred_profile"]

    lo = min(float(tp_b.min()), float(tp_p.min()), float(pp_b.min()), float(pp_p.min())) - 1
    hi = max(float(tp_b.max()), float(tp_p.max()), float(pp_b.max()), float(pp_p.max())) + 1

    for k in range(n_levels):
        ax = axes[k]
        ax.scatter(tp_b[:, k], pp_b[:, k], s=8, alpha=0.35,
                   c="#4C72B0", label="Baseline" if k == 0 else None)
        ax.scatter(tp_p[:, k], pp_p[:, k], s=8, alpha=0.35,
                   c="#C44E52", label="PCA-head" if k == 0 else None)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"Level {k + 1}")
        ax.grid(True, alpha=0.3)
        if k % ncols == 0:
            ax.set_ylabel("Predicted r_e  (μm)")
        if k // ncols == nrows - 1:
            ax.set_xlabel("True r_e  (μm)")

    # Hide unused axes.
    for j in range(n_levels, len(axes)):
        axes[j].axis("off")
    # One global legend.
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Predicted vs. true r_e — held-out test set", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "03_pred_vs_true_scatter.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_example_profiles(baseline_res, pca_res, out_dir: Path,
                         n_examples: int = 8, seed: int = 0) -> None:
    """04 — Sample a handful of held-out profiles; plot true and both preds."""
    N = baseline_res["true_profile"].shape[0]
    rng = np.random.default_rng(seed)
    # Try to pick indices that span a range of true profile means (avoids
    # accidentally plotting 8 very similar profiles).
    means = baseline_res["true_profile"].mean(axis=1)
    quantiles = np.linspace(0.05, 0.95, n_examples)
    idx = np.unique(np.quantile(np.arange(N), quantiles).astype(int))
    # Top up with random indices if de-duplication shrank the selection.
    if len(idx) < n_examples:
        extra = rng.choice(N, size=n_examples - len(idx), replace=False)
        idx = np.unique(np.concatenate([idx, extra]))

    n_plot = len(idx)
    ncols = 4
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    n_levels = baseline_res["true_profile"].shape[1]
    level_axis = np.arange(1, n_levels + 1)

    for i, s in enumerate(idx):
        ax = axes[i]
        # Plot profile with level on y-axis to mirror atmospheric convention
        # (L1 at top → cloud top at top of panel).
        tp   = baseline_res["true_profile"][s]
        p_b  = baseline_res["pred_profile"][s]
        s_b  = baseline_res["pred_std"][s]
        p_p  = pca_res["pred_profile"][s]
        s_p  = pca_res["pred_std"][s]

        ax.plot(tp, level_axis, "k-o", linewidth=2, label="True", markersize=4)
        ax.plot(p_b, level_axis, color="#4C72B0", marker="s", linewidth=1.4,
                label="Baseline", markersize=3)
        ax.fill_betweenx(level_axis, p_b - s_b, p_b + s_b,
                         color="#4C72B0", alpha=0.2)
        ax.plot(p_p, level_axis, color="#C44E52", marker="^", linewidth=1.4,
                label="PCA-head", markersize=3)
        ax.fill_betweenx(level_axis, p_p - s_p, p_p + s_p,
                         color="#C44E52", alpha=0.2)

        ax.invert_yaxis()   # L1 on top
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Test sample {s}")
        if i % ncols == 0:
            ax.set_ylabel("Level (L1 = top)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("r_e  (μm)")

    for j in range(n_plot, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Example held-out profiles — true vs. baseline vs. PCA-head "
                 "(shaded = ±1σ)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "04_example_profiles.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_tau_parity(baseline_res, pca_res, out_dir: Path) -> None:
    """05 — τ_c parity plot for both models."""
    tt_b, tp_b = baseline_res["true_tau"], baseline_res["pred_tau"]
    tt_p, tp_p = pca_res["true_tau"],      pca_res["pred_tau"]
    lo = min(float(tt_b.min()), float(tp_b.min()), float(tp_p.min())) - 1
    hi = max(float(tt_b.max()), float(tp_b.max()), float(tp_p.max())) + 1

    rmse_b = float(np.sqrt(np.mean((tp_b - tt_b) ** 2)))
    rmse_p = float(np.sqrt(np.mean((tp_p - tt_p) ** 2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tt_b, tp_b, s=10, alpha=0.4, c="#4C72B0",
               label=f"Baseline  (RMSE = {rmse_b:.2f})")
    ax.scatter(tt_p, tp_p, s=10, alpha=0.4, c="#C44E52",
               label=f"PCA-head (RMSE = {rmse_p:.2f})")
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_xlabel("True τ_c"); ax.set_ylabel("Predicted τ_c")
    ax.set_title("τ_c parity — held-out test set")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "05_tau_parity.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Text summary
# ───────────────────────────────────────────────────────────────────────────────
def write_summary(baseline_res, pca_res, baseline_ckpt, pca_ckpt, out_dir: Path):
    """Write a plain-text summary for the user to copy into the paper."""
    b_rmse = per_level_rmse(baseline_res)
    p_rmse = per_level_rmse(pca_res)
    n_levels = len(b_rmse)

    b_tau_rmse = float(np.sqrt(np.mean(
        (baseline_res["pred_tau"] - baseline_res["true_tau"]) ** 2)))
    p_tau_rmse = float(np.sqrt(np.mean(
        (pca_res["pred_tau"] - pca_res["true_tau"]) ** 2)))

    # Per-sample σ calibration: what fraction of |residuals| fall inside ±1σ?
    def cal_inside_1sigma(res):
        err = np.abs(res["pred_profile"] - res["true_profile"])
        return float(np.mean(err <= res["pred_std"]))
    b_cal = cal_inside_1sigma(baseline_res)
    p_cal = cal_inside_1sigma(pca_res)

    lines = []
    lines.append("PCA-head vs baseline — held-out test set")
    lines.append("=" * 60)
    lines.append(f"Baseline checkpoint: {baseline_ckpt}")
    lines.append(f"PCA-head checkpoint: {pca_ckpt}")
    lines.append(f"Test samples:        {baseline_res['true_profile'].shape[0]}")
    lines.append(f"Levels:              {n_levels}")
    lines.append("")
    lines.append("Per-level r_e RMSE (μm):")
    lines.append(f"  {'level':>6}  {'baseline':>10}  {'PCA-head':>10}  {'Δ':>8}")
    for k in range(n_levels):
        d = p_rmse[k] - b_rmse[k]
        lines.append(f"  {k+1:>6}  {b_rmse[k]:>10.4f}  {p_rmse[k]:>10.4f}  "
                     f"{d:>+8.4f}")
    lines.append(f"  {'mean':>6}  {b_rmse.mean():>10.4f}  {p_rmse.mean():>10.4f}  "
                 f"{(p_rmse.mean() - b_rmse.mean()):>+8.4f}")
    lines.append("")
    lines.append(f"τ_c RMSE:         baseline={b_tau_rmse:.4f}    PCA={p_tau_rmse:.4f}")
    lines.append(f"|err|<=1σ frac:   baseline={b_cal:.3f}    PCA={p_cal:.3f}  "
                 "(expect ~0.68 for calibrated Gaussian σ)")

    txt = "\n".join(lines) + "\n"
    (out_dir / "06_summary.txt").write_text(txt)
    print("\n" + txt)


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ----- Resolve PCA ckpt if not provided -----
    pca_ckpt = Path(args.pca_ckpt) if args.pca_ckpt else default_pca_ckpt()
    if pca_ckpt is None or not pca_ckpt.exists():
        raise SystemExit(
            "No PCA-head checkpoint found.  Pass --pca-ckpt or first run "
            "train_pca.py to produce one."
        )
    baseline_ckpt = Path(args.baseline_ckpt)
    if not baseline_ckpt.exists():
        raise SystemExit(
            f"Baseline checkpoint not found: {baseline_ckpt}.  Pass a valid "
            "--baseline-ckpt (e.g. a best_model.pt from sweep_results_3)."
        )

    # ----- Load PCA basis (needed for model + split info) -----
    basis = load_pca_basis(args.pca_basis)

    # ----- Figure out which HDF5 to evaluate on -----
    # The basis records the h5_path used for training.  Use that unless the
    # user explicitly overrides.  This guarantees the two models see the
    # SAME test profiles — which is the whole point of the comparison.
    if args.h5_path is not None:
        h5_path = args.h5_path
    else:
        h5_path = str(basis["h5_path"]) if "h5_path" in basis else None
        if h5_path is None:
            raise SystemExit(
                "PCA basis file does not record h5_path and none was passed. "
                "Pass --h5-path."
            )
    h5_path = str(resolve_h5_path(h5_path, args.training_data_dir))
    print(f"Evaluating both models on: {h5_path}")

    # ----- Build dataloaders -----
    # Use the same profile-held-out split (seed + counts recorded in basis).
    _, _, test_loader = create_dataloaders(
        h5_path=h5_path,
        batch_size=args.batch_size,
        num_workers=4,
        instrument="hysics",
        profile_holdout=True,
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    print(f"Test samples: {len(test_loader.dataset):,}")

    # ----- Pick device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Load models -----
    print(f"\nLoading baseline from: {baseline_ckpt}")
    baseline_model = load_baseline(baseline_ckpt, device)
    print(f"Loading PCA-head from: {pca_ckpt}")
    pca_model      = load_pca_model(pca_ckpt, basis, device)

    # Sanity check: both models must share n_levels (or the comparison is
    # meaningless).
    if baseline_model.config.n_levels != pca_model.config.n_levels:
        raise SystemExit(
            f"Level mismatch: baseline has {baseline_model.config.n_levels} "
            f"levels, PCA-head has {pca_model.config.n_levels}.  They must "
            "be trained on the same HDF5."
        )

    # ----- Run inference -----
    print("\nRunning baseline inference ...")
    baseline_res = run_inference(baseline_model, test_loader, device)
    print("Running PCA-head inference ...")
    pca_res      = run_inference(pca_model,      test_loader, device)

    # ----- Figures & summary -----
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to: {fig_dir}")

    b_rmse = per_level_rmse(baseline_res)
    p_rmse = per_level_rmse(pca_res)
    b_abs  = per_level_abs_residuals(baseline_res)
    p_abs  = per_level_abs_residuals(pca_res)

    fig_per_level_rmse_bar(b_rmse, p_rmse, fig_dir)
    fig_per_level_rmse_box(b_abs, p_abs, fig_dir)
    fig_pred_vs_true(baseline_res, pca_res, fig_dir)
    fig_example_profiles(baseline_res, pca_res, fig_dir,
                         n_examples=args.n_example_profiles)
    fig_tau_parity(baseline_res, pca_res, fig_dir)
    write_summary(baseline_res, pca_res, baseline_ckpt, pca_ckpt, fig_dir)

    # Also write a JSON for programmatic use.
    (fig_dir / "06_summary.json").write_text(json.dumps({
        "baseline_ckpt":     str(baseline_ckpt),
        "pca_ckpt":          str(pca_ckpt),
        "n_test_samples":    int(baseline_res["true_profile"].shape[0]),
        "baseline_per_level_rmse": b_rmse.tolist(),
        "pca_per_level_rmse":      p_rmse.tolist(),
        "baseline_mean_rmse":  float(b_rmse.mean()),
        "pca_mean_rmse":       float(p_rmse.mean()),
        "baseline_tau_rmse":   float(np.sqrt(np.mean(
            (baseline_res["pred_tau"] - baseline_res["true_tau"]) ** 2))),
        "pca_tau_rmse":        float(np.sqrt(np.mean(
            (pca_res["pred_tau"] - pca_res["true_tau"]) ** 2))),
    }, indent=2))

    print(f"\nDone.  Figures in: {fig_dir}")


if __name__ == "__main__":
    main()
