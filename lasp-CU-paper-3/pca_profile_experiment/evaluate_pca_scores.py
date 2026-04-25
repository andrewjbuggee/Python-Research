"""
evaluate_pca_scores.py
----------------------
Per-PC retrieval-quality diagnostic for the PCA-head network.

The training script `train_pca.py` reports per-LEVEL RMSE (over the 50-level
profile output of the decoder), but the network actually predicts only K
PC SCORES — the 50-level profile is just `scores @ basis + mean`.  If you
want to know *which PC modes the spectrum actually constrains*, the right
diagnostic is:

    For every test sample i:
        true_score_i  = (profile_true_norm_i  − pca_mean) @ pca_components.T
        pred_score_i  = network output  ('pc_scores')

then scatter `pred_score_i[k]` against `true_score_i[k]` for each k = 1..K.
A perfect retrieval lies on the y=x line; a useless retrieval is a flat
horizontal blob (network ignores that mode).

This script also reports the **PCA truncation floor** for the same model:
the 50-level RMSE you would get if you fed the network the *true* PC scores
instead of its predictions and decoded.  This factors the achievable RMSE
into an encoder-side error (predicted vs true scores) and a representation-
side error (truncating at K modes).

Outputs (saved into <run_dir>/figures/score_diagnostics/):
    01_pc_score_scatter.png        K subplots — predicted vs true score
    02_pc_score_rmse_per_mode.png  Per-PC RMSE in score space + R² per mode
    03_floor_decomposition.png     Stacked bar: encoder error vs PCA floor

Usage
=====
    python evaluate_pca_scores.py \\
        --run-dir checkpoints_pca/20260424_175929_K5 \\
        --pca-basis pca_basis.npz

`--run-dir` is mandatory.  `--pca-basis` defaults to ./pca_basis.npz.
The script writes figures next to the checkpoint so K=5 and K=7 stay
separated.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Repo-root imports — match train_pca.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import create_dataloaders, resolve_h5_path  # noqa: E402

# Local imports
from models_pca import (  # noqa: E402
    PCADropletProfileNetwork,
    PCARetrievalConfig,
    load_pca_basis,
)


FIG_DPI = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-PC retrieval-quality scatter plots for a trained "
                    "PCA-head model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the checkpoint directory (contains best_model.pt + "
             "config.yaml).",
    )
    p.add_argument(
        "--pca-basis", type=str,
        default=str(Path(__file__).parent / "pca_basis.npz"),
        help="Path to pca_basis.npz used to fit the model.",
    )
    p.add_argument(
        "--checkpoint-name", type=str, default="best_model.pt",
        choices=["best_model.pt", "final_model.pt"],
        help="Which checkpoint inside run-dir to evaluate.",
    )
    p.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val"],
        help="Which split to evaluate the per-PC retrieval on.",
    )
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Returns:
        pred_scores : (N, K) — network's predicted PC scores
        true_norm   : (N, L) — true normalized profiles
        pred_norm   : (N, L) — model-decoded normalized profiles
    """
    model.eval()
    pred_scores, true_norm, pred_norm = [], [], []
    for x, profile_true, _tau in loader:
        x = x.to(device)
        out = model(x)
        pred_scores.append(out["pc_scores"].cpu().numpy())
        # `profile_normalized` is the decoder's output (clamped to [0,1]); use
        # the un-clamped version so encoder error and PCA-floor error compose
        # cleanly without saturation artifacts.
        pred_norm.append(out["profile_normalized_raw"].cpu().numpy())
        true_norm.append(profile_true.numpy())
    return (np.concatenate(pred_scores, axis=0),
            np.concatenate(true_norm,  axis=0),
            np.concatenate(pred_norm,  axis=0))


def linfit_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope, intercept, R² for y = slope*x + intercept."""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def plot_pc_scatter(true_scores, pred_scores, fig_path: Path):
    """K subplots: pred vs true PC score for each of K modes."""
    K = pred_scores.shape[1]
    ncol = min(K, 3)
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for k in range(K):
        ax = axes[k]
        x = true_scores[:, k]
        y = pred_scores[:, k]
        m = linfit_metrics(x, y)

        # Hexbin densities scale poorly for ~7k points; use plain scatter.
        ax.scatter(x, y, s=4, alpha=0.30, color="#4C72B0",
                   edgecolors="none", rasterized=True)

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k-", linewidth=0.8, alpha=0.6, label="1:1")
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, m["slope"] * xx + m["intercept"],
                "--", color="#C44E52", linewidth=1.2,
                label=f"fit (slope={m['slope']:.2f})")

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"True PC{k+1} score")
        ax.set_ylabel(f"Predicted PC{k+1} score")
        ax.set_title(f"PC {k+1}   R² = {m['r2']:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for k in range(K, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Per-PC retrieval quality:  predicted vs. true PC score",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_score_rmse_per_mode(true_scores, pred_scores, fig_path: Path):
    """Per-mode score-space RMSE & R² as a side-by-side bar chart."""
    K = pred_scores.shape[1]
    diff = pred_scores - true_scores
    rmse = np.sqrt((diff ** 2).mean(axis=0))
    r2 = np.array([linfit_metrics(true_scores[:, k], pred_scores[:, k])["r2"]
                   for k in range(K)])
    # Compare against the natural scale of each mode (std of the true score).
    score_std = true_scores.std(axis=0)
    rel_rmse = rmse / np.where(score_std > 0, score_std, 1.0)

    x = np.arange(1, K + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(x - 0.18, rmse, 0.34, color="#4C72B0",
            edgecolor="black", label="RMSE (score units)")
    ax1.bar(x + 0.18, rel_rmse, 0.34, color="#DD8452",
            edgecolor="black", label="RMSE / std(true score)")
    ax1.set_xlabel("PC mode")
    ax1.set_ylabel("Score-space RMSE")
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, r2, "ko-", linewidth=1.5, markersize=7, label="R²")
    ax2.set_ylabel("R²")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")

    ax1.set_title("Per-mode score retrieval quality")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_floor_decomposition(true_norm, pred_norm,
                             true_scores, pca_mean, pca_components,
                             re_min, re_max, fig_path: Path):
    """
    Stacked-bar comparison:
        (A) Network's mean per-level RMSE
        (B) PCA truncation floor: decode TRUE scores at K modes
        (C) Encoder error (in profile-RMSE units): A − B (Pythagorean)

    All in physical μm.
    """
    re_range = re_max - re_min

    # Network achieved RMSE (per-level → mean over levels)
    diff_net = (pred_norm - true_norm) * re_range
    rmse_net = np.sqrt((diff_net ** 2).mean())  # scalar

    # PCA truncation: decode TRUE scores at K modes — done in float64 to
    # avoid the same precision blowup that affects projection.
    K = true_scores.shape[1]
    sc64 = true_scores.astype(np.float64)
    comp64 = pca_components[:K].astype(np.float64)
    mean64 = pca_mean.astype(np.float64)
    decoded_true = (sc64 @ comp64) + mean64
    diff_floor = (decoded_true - true_norm.astype(np.float64)) * re_range
    rmse_floor = np.sqrt((diff_floor ** 2).mean())

    # Encoder error: orthogonal decomposition (only exact when errors are
    # orthogonal; in practice this is a useful approximation).
    enc_sq = max(0.0, rmse_net ** 2 - rmse_floor ** 2)
    rmse_encoder = float(np.sqrt(enc_sq))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["Network achieved", "PCA truncation floor", "Encoder error\n(Pythag.)"],
        [rmse_net, rmse_floor, rmse_encoder],
        color=["#C44E52", "#4C72B0", "#DD8452"],
        edgecolor="black",
    )
    for b, v in zip(bars, [rmse_net, rmse_floor, rmse_encoder]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f" {v:.3f} μm", ha="center", va="bottom")
    ax.set_ylabel("Mean per-level r_e RMSE (μm)")
    ax.set_title(f"Floor decomposition  (K = {K})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return rmse_net, rmse_floor, rmse_encoder


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir not found: {run_dir}")

    # ----- Load training-time config from the checkpoint dir -----
    with open(run_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Anchor relative h5_path against repo root (matches train_pca.py).
    h5 = resolve_h5_path(config["data"]["h5_path"], None)
    if not h5.is_absolute():
        h5 = (_REPO_ROOT / h5).resolve()
    config["data"]["h5_path"] = str(h5)

    # ----- Load PCA basis -----
    basis = load_pca_basis(args.pca_basis)
    K = int(config["model"]["n_pca_components"])
    if K > basis["components"].shape[0]:
        raise ValueError(f"K={K} but basis only has "
                         f"{basis['components'].shape[0]} components.")

    # ----- Device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Data (same profile-aware split as training) -----
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=config["data"]["h5_path"],
        batch_size=int(config["training"]["batch_size"]),
        train_frac=config["data"].get("train_frac", 0.8),
        val_frac=config["data"].get("val_frac", 0.1),
        num_workers=config["data"].get("num_workers", 4),
        instrument=config["data"].get("instrument", "hysics"),
        profile_holdout=True,
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    loader = {"val": val_loader, "test": test_loader}[args.split]
    print(f"Evaluating on '{args.split}' split: {len(loader.dataset):,} samples")

    # ----- Build model & load weights -----
    pca_cfg = PCARetrievalConfig(
        n_wavelengths=config["model"]["n_wavelengths"],
        n_geometry_inputs=config["model"].get("n_geometry_inputs", 4),
        n_levels=int(config["model"]["n_levels"]),
        n_pca_components=K,
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=config["model"].get("dropout", 0.1),
        activation=config["model"].get("activation", "gelu"),
        re_min=config["model"].get("re_min", 1.5),
        re_max=config["model"].get("re_max", 50.0),
        tau_min=config["model"].get("tau_min", 3.0),
        tau_max=config["model"].get("tau_max", 75.0),
        use_tau_head=config["model"].get("use_tau_head", True),
        learn_pca_basis=config["model"].get("learn_pca_basis", False),
    )
    model = PCADropletProfileNetwork(pca_cfg).to(device)
    model.register_pca(
        basis["mean"].astype(np.float32),
        basis["components"][:K].astype(np.float32),
    )
    ckpt = torch.load(run_dir / args.checkpoint_name, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.checkpoint_name} from epoch {ckpt.get('epoch','?')}")

    # ----- Inference -----
    pred_scores, true_norm, pred_norm = run_inference(model, loader, device)
    print(f"  pred_scores: {pred_scores.shape}, true_norm: {true_norm.shape}")

    # ----- Compute true PC scores by projection (float64 — see fit_pca docstring) -----
    # Float32 projection blows up for K beyond the effective rank because
    # eigenvectors of near-zero eigenvalues are ill-conditioned.  Casting to
    # float64 makes the top-K reconstruction stable and matches the values
    # saved in pca_basis.npz by analyze_profile_pca.py.
    pca_mean       = basis["mean"].astype(np.float64)             # (L,)
    pca_components = basis["components"][:K].astype(np.float64)   # (K, L)
    centered = true_norm.astype(np.float64) - pca_mean
    true_scores = (centered @ pca_components.T).astype(np.float32)  # (N, K)

    # ----- Output dir -----
    out_dir = run_dir / "figures" / f"score_diagnostics_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out_dir.relative_to(run_dir)}")

    plot_pc_scatter(true_scores, pred_scores,
                    out_dir / "01_pc_score_scatter.png")
    plot_score_rmse_per_mode(true_scores, pred_scores,
                             out_dir / "02_pc_score_rmse_per_mode.png")
    rmse_net, rmse_floor, rmse_enc = plot_floor_decomposition(
        true_norm, pred_norm,
        true_scores,
        pca_mean,
        basis["components"],   # full basis for decoding (we slice [:K] inside)
        float(config["model"].get("re_min", 1.5)),
        float(config["model"].get("re_max", 50.0)),
        out_dir / "03_floor_decomposition.png",
    )

    # ----- Print a one-screen summary -----
    print("\n" + "=" * 70)
    print(f"Per-PC retrieval-quality summary (K = {K}, split = {args.split})")
    print("=" * 70)
    print(f"{'PC':>4} | {'R²':>8} | {'slope':>8} | {'RMSE':>10} | {'std(true)':>10} | {'rel':>6}")
    print("-" * 70)
    for k in range(K):
        m = linfit_metrics(true_scores[:, k], pred_scores[:, k])
        diff = pred_scores[:, k] - true_scores[:, k]
        r = float(np.sqrt((diff ** 2).mean()))
        s = float(true_scores[:, k].std())
        print(f"{k+1:>4} | {m['r2']:>8.3f} | {m['slope']:>8.3f} | "
              f"{r:>10.4f} | {s:>10.4f} | {r/max(s,1e-12):>6.2f}")
    print("-" * 70)
    print(f"  Network mean per-level RMSE:  {rmse_net:.3f} μm")
    print(f"  PCA truncation floor (K={K}):  {rmse_floor:.3f} μm  "
          f"(true scores → decoder)")
    print(f"  Implied encoder error:        {rmse_enc:.3f} μm  "
          f"(Pythagorean residual)")
    print()
    print("Figures saved.")


if __name__ == "__main__":
    main()
