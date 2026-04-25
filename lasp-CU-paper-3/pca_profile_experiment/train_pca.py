"""
train_pca.py
------------
Step 3 of the PCA-head experiment: train the `PCADropletProfileNetwork`.

This script is a near-clone of the existing `train.py` at the repo root, but
with three deliberate differences:

  1. It instantiates `PCADropletProfileNetwork` instead of
     `DropletProfileNetwork`, and registers the PCA basis produced by
     `analyze_profile_pca.py`.

  2. It writes every checkpoint into a timestamped subdirectory of
     ./checkpoints_pca/  (a NEW directory, alongside the existing
     ./checkpoints/) so the user can revert or A/B compare cleanly.

  3. It saves per-epoch training diagnostics as figures in
     ./figures/training_K{K}/ at 500 DPI.  These figures include:
          - 01_loss_curves.png      NLL loss on train vs val
          - 02_rmse_curves.png      Mean per-level RMSE on train vs val over epochs
          - 03_per_level_final.png  Final per-level RMSE bar chart
          - 04_pc_score_trajectories.png
              Mean predicted PC score distribution vs epoch (sanity check
              that the network is actually spanning the PC-score manifold
              and not collapsing to a point).

The training loop logic itself is intentionally the same as the existing
pipeline so that differences in final RMSE can be attributed to the output
parameterization, not to changes in optimization.

Config
======
A minimal YAML config is used (same schema as `train.py`).  The PCA-specific
entries are inside `model:`
    n_pca_components: 3          # K — the retained PCA components
    use_tau_head:     true
    learn_pca_basis:  false

Example:
    python train_pca.py --config config_pca.yaml --pca-basis pca_basis.npz

If --pca-basis is omitted the script looks for ./pca_basis.npz (written by
analyze_profile_pca.py).

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import matplotlib.pyplot as plt

# Repo-root imports.  This script lives in <repo>/pca_profile_experiment/; the
# existing data/loss/config modules live at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import create_dataloaders, resolve_h5_path                          # noqa: E402
from models import CombinedLoss                                               # noqa: E402

# PCA-specific imports (same directory).
from models_pca import (  # noqa: E402
    PCADropletProfileNetwork,
    PCARetrievalConfig,
    load_pca_basis,
)


FIG_DPI = 500


# ───────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ───────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PCA-head droplet profile network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent / "config_pca.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--pca-basis", type=str,
        default=str(Path(__file__).parent / "pca_basis.npz"),
        help="Path to the PCA basis .npz produced by analyze_profile_pca.py.",
    )
    parser.add_argument(
        "--training-data-dir", type=str, default=None,
        help="Override the directory of data:h5_path (filename kept).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "checkpoints_pca"),
        help="Root directory for per-run checkpoint subdirectories.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Optional label appended to the timestamped run directory.",
    )
    parser.add_argument(
        "--n-pca-components", type=int, default=None,
        help="Override model.n_pca_components in the YAML.  Useful for sweeping K "
             "back-to-back without editing the config (e.g. for K in 5 7).",
    )
    return parser.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Epoch loops — kept deliberately close to train.py to minimize differences
# ───────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std: float = 0.0):
    """One pass over the training set.  Returns the mean loss dict."""
    model.train()
    running = {}
    n_batches = 0
    for x, profile_true, tau_true in loader:
        x            = x.to(device)
        profile_true = profile_true.to(device)
        tau_true     = tau_true.to(device)

        # Optional multiplicative noise augmentation on the reflectance
        # channels (same trick used in sweep 3).
        if augment_noise_std > 0.0:
            x = x + augment_noise_std * x * torch.randn_like(x)

        optimizer.zero_grad()
        output = model(x)
        losses = criterion(output, profile_true, tau_true)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    running = {}
    n_batches = 0
    for x, profile_true, tau_true in loader:
        x            = x.to(device)
        profile_true = profile_true.to(device)
        tau_true     = tau_true.to(device)
        output = model(x)
        losses = criterion(output, profile_true, tau_true)
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def compute_level_rmse(model, loader, device):
    """Per-level r_e RMSE (μm) + scalar τ RMSE on the given loader."""
    model.eval()
    sq = None
    tau_sq = 0.0
    n_samples = 0
    re_min   = float(model.re_min.item())
    re_range = float(model.re_max.item()) - re_min
    tau_min  = float(model.tau_min.item())
    tau_range = float(model.tau_max.item()) - tau_min

    # Also collect PC-score batches for the optional trajectory plot.
    pc_scores_all = []

    for x, profile_true, tau_true in loader:
        x            = x.to(device)
        profile_true = profile_true.to(device)
        tau_true     = tau_true.to(device)
        out = model(x)

        pred_profile = out["profile"]
        true_profile = profile_true * re_range + re_min
        diffs = (pred_profile - true_profile).pow(2)
        sq = diffs.sum(dim=0) if sq is None else sq + diffs.sum(dim=0)

        pred_tau = out["tau_c"].squeeze(-1)
        true_tau = tau_true * tau_range + tau_min
        tau_sq += (pred_tau - true_tau).pow(2).sum().item()

        pc_scores_all.append(out["pc_scores"].detach().cpu().numpy())
        n_samples += x.shape[0]

    rmse     = (sq / n_samples).sqrt().cpu().numpy()
    tau_rmse = (tau_sq / n_samples) ** 0.5
    return rmse, tau_rmse, np.concatenate(pc_scores_all, axis=0)


# ───────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ───────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": config,
        "model_config": {
            "n_wavelengths":    model.config.n_wavelengths,
            "n_geometry_inputs": model.config.n_geometry_inputs,
            "n_levels":         model.config.n_levels,
            "n_pca_components": model.config.n_pca_components,
            "hidden_dims":      model.config.hidden_dims,
            "dropout":          model.config.dropout,
            "activation":       model.config.activation,
            "use_tau_head":     model.config.use_tau_head,
            "learn_pca_basis":  model.config.learn_pca_basis,
        },
    }, path)


# ───────────────────────────────────────────────────────────────────────────────
# Figures (called after training)
# ───────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(history: list[dict], out_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train  = [h["train"]["total"] for h in history]
    val    = [h["val"]["total"]   for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="Train", color="#4C72B0", linewidth=1.5)
    plt.plot(epochs, val,   label="Validation", color="#C44E52", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss (Gaussian NLL + physics)")
    plt.title("Training & validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_loss_curves.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_rmse_curves(rmse_history: list[dict], out_dir: Path) -> None:
    """Mean-over-levels r_e RMSE vs epoch, train & val."""
    if not rmse_history:
        return
    epochs = [r["epoch"] for r in rmse_history]
    tr     = [float(np.mean(r["train_rmse"])) for r in rmse_history]
    va     = [float(np.mean(r["val_rmse"]))   for r in rmse_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr, "o-", color="#4C72B0", label="Train (mean over levels)")
    plt.plot(epochs, va, "s-", color="#C44E52", label="Validation (mean over levels)")
    plt.xlabel("Epoch")
    plt.ylabel("r_e RMSE (μm)")
    plt.title("Mean per-level r_e RMSE over training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_rmse_curves.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_final_per_level_rmse(rmse_final_train, rmse_final_val, rmse_final_test,
                              out_dir: Path) -> None:
    """Bar chart of the final per-level RMSE on all three splits."""
    n_levels = len(rmse_final_train)
    x = np.arange(1, n_levels + 1)
    w = 0.28

    plt.figure(figsize=(9, 5))
    plt.bar(x - w, rmse_final_train, w, color="#4C72B0",
            edgecolor="black", label=f"Train (mean={rmse_final_train.mean():.3f})")
    plt.bar(x,     rmse_final_val,   w, color="#DD8452",
            edgecolor="black", label=f"Val (mean={rmse_final_val.mean():.3f})")
    plt.bar(x + w, rmse_final_test,  w, color="#C44E52",
            edgecolor="black", label=f"Test (mean={rmse_final_test.mean():.3f})")
    plt.xticks(x, [f"L{i}" for i in x])
    plt.xlabel("Level index (L1 = cloud top, LN = cloud base)")
    plt.ylabel("r_e RMSE (μm)")
    plt.title("Final per-level r_e RMSE — PCA-head model")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "03_per_level_final.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_pc_trajectories(pc_score_history: list[dict], out_dir: Path) -> None:
    """
    PC-score distribution evolution over training.

    For each logged epoch we plot the distribution (as violin) of PC-1 and
    PC-2 scores predicted on the validation set.  If training collapses to
    a single point the violins will shrink to a line — a useful diagnostic.
    """
    if not pc_score_history:
        return
    epochs = [r["epoch"] for r in pc_score_history]
    K = pc_score_history[0]["scores"].shape[1]
    K_show = min(K, 3)

    fig, axes = plt.subplots(K_show, 1, figsize=(10, 3 * K_show), sharex=True)
    if K_show == 1:
        axes = [axes]

    for k in range(K_show):
        ax = axes[k]
        data = [r["scores"][:, k] for r in pc_score_history]
        parts = ax.violinplot(data, positions=epochs, widths=max(1, (epochs[-1] - epochs[0]) / 40 if len(epochs) > 1 else 1), showmeans=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_edgecolor("black")
            pc.set_alpha(0.55)
        ax.set_ylabel(f"PC {k+1} score")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Distribution of predicted PC scores (val set) over epochs", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "04_pc_score_trajectories.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ----- Load config -----
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ----- Resolve H5 path (supports local-vs-cluster override) -----
    # If the configured h5_path is relative, anchor it at the repo root so the
    # script works no matter the current working directory (e.g. when run from
    # inside pca_profile_experiment/).  --training-data-dir, if provided, takes
    # precedence and replaces only the directory.
    h5_resolved = resolve_h5_path(
        config["data"]["h5_path"],
        args.training_data_dir,
    )
    if not h5_resolved.is_absolute():
        h5_resolved = (_REPO_ROOT / h5_resolved).resolve()
    config["data"]["h5_path"] = str(h5_resolved)

    # ----- Load PCA basis & sanity-check against config -----
    basis = load_pca_basis(args.pca_basis)
    if args.n_pca_components is not None:
        config["model"]["n_pca_components"] = int(args.n_pca_components)
        print(f"  CLI override: n_pca_components={args.n_pca_components}")
    K = int(config["model"]["n_pca_components"])
    if K > basis["components"].shape[0]:
        raise ValueError(
            f"Requested n_pca_components={K} but the saved basis only has "
            f"{basis['components'].shape[0]} components.  Re-run "
            "analyze_profile_pca.py with --max-components={K}."
        )
    n_levels_basis = int(basis["components"].shape[1])
    if n_levels_basis != int(config["model"]["n_levels"]):
        raise ValueError(
            f"PCA basis has {n_levels_basis} levels, config says "
            f"{config['model']['n_levels']} — they must match."
        )

    # ----- Output directory -----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"K{K}" + (f"_{args.run_name}" if args.run_name else "")
    run_dir = Path(args.output_dir) / f"{stamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).parent / "figures" / f"training_K{K}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # ----- Device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Data -----
    print(f"\nLoading data from: {config['data']['h5_path']}")
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=config["data"]["h5_path"],
        batch_size=config["training"]["batch_size"],
        train_frac=config["data"].get("train_frac", 0.8),
        val_frac=config["data"].get("val_frac", 0.1),
        num_workers=config["data"].get("num_workers", 4),
        instrument=config["data"].get("instrument", "hysics"),
        profile_holdout=True,
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")
    print(f"  Test samples:  {len(test_loader.dataset):,}")

    # ----- Model -----
    pca_cfg = PCARetrievalConfig(
        n_wavelengths=config["model"]["n_wavelengths"],
        n_geometry_inputs=config["model"].get("n_geometry_inputs", 4),
        n_levels=config["model"]["n_levels"],
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPCA-head model: K={K} components, {n_params:,} trainable parameters")

    # ----- Loss -----
    lw_list = config["loss"].get("level_weights", None)
    lw = (torch.tensor(lw_list, dtype=torch.float32) if lw_list else None)

    # Note: CombinedLoss from the existing models.py uses the same keys we
    # provide (`profile_normalized`, `tau_normalized`, etc.), so it works
    # directly with PCADropletProfileNetwork's output dict.
    # We build a minimal RetrievalConfig shim just for the loss's physics
    # bounds — the loss does not use any other field of that config.
    from models import RetrievalConfig as _BaselineCfg
    dummy_cfg = _BaselineCfg(
        n_wavelengths=pca_cfg.n_wavelengths,
        n_geometry_inputs=pca_cfg.n_geometry_inputs,
        n_levels=pca_cfg.n_levels,
        hidden_dims=pca_cfg.hidden_dims,
        dropout=pca_cfg.dropout,
        activation=pca_cfg.activation,
        re_min=pca_cfg.re_min,
        re_max=pca_cfg.re_max,
        tau_min=pca_cfg.tau_min,
        tau_max=pca_cfg.tau_max,
    )

    criterion = CombinedLoss(
        config=dummy_cfg,
        lambda_physics=config["loss"]["lambda_physics"],
        lambda_monotonicity=config["loss"].get("lambda_monotonicity", 0.01),
        lambda_adiabatic=config["loss"].get("lambda_adiabatic", 0.0),
        lambda_smoothness=config["loss"].get("lambda_smoothness", 0.0),
        level_weights=lw,
        sigma_floor=config["loss"].get("sigma_floor", 0.01),
    ).to(device)

    # Zero out the τ loss weight if the head is disabled.
    if not pca_cfg.use_tau_head:
        criterion.supervised.tau_weight = 0.0
        print("  τ head disabled (use_tau_head=False) — setting τ_weight to 0.")

    # ----- Optimizer + scheduler -----
    lr = float(config["training"]["learning_rate"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config["training"].get("scheduler_T0", 50)),
        T_mult=int(config["training"].get("scheduler_T_mult", 2)),
        eta_min=float(config["training"].get("scheduler_eta_min", 1e-6)),
    )

    # ----- Train -----
    n_epochs = int(config["training"]["n_epochs"])
    patience = int(config["training"].get("early_stopping_patience", 40))
    augment_noise_std = float(config["training"].get("augment_noise_std", 0.0))
    rmse_interval = int(config["training"].get("rmse_log_interval", 10))

    best_val = float("inf")
    pat_count = 0
    history: list[dict] = []
    rmse_history: list[dict] = []
    pc_history: list[dict] = []

    print(f"\nTraining for up to {n_epochs} epochs "
          f"(early stopping patience={patience})")
    print("=" * 80)

    for epoch in range(n_epochs):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             augment_noise_std=augment_noise_std)
        va = validate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "train": tr,
            "val":   va,
            "lr":    optimizer.param_groups[0]["lr"],
            "time":  dt,
        })
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d} | train total {tr['total']:.5f} "
                  f"| val total {va['total']:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| {dt:.1f}s")

        # Per-level RMSE logging (also grabs PC scores for the trajectory plot)
        if epoch % rmse_interval == 0 or epoch == n_epochs - 1:
            tr_rmse, tr_tau, _ = compute_level_rmse(model, train_loader, device)
            va_rmse, va_tau, pc_val = compute_level_rmse(model, val_loader, device)
            rmse_history.append({"epoch": epoch,
                                 "train_rmse": tr_rmse, "train_tau": tr_tau,
                                 "val_rmse":   va_rmse, "val_tau":   va_tau})
            # Keep a subsample of PC scores (avoid bloating disk).
            sample = pc_val[np.random.default_rng(0).choice(len(pc_val),
                            size=min(1000, len(pc_val)), replace=False)]
            pc_history.append({"epoch": epoch, "scores": sample})

            print(f"  per-level val RMSE: "
                  + " ".join(f"{v:.3f}" for v in va_rmse)
                  + f"   | τ RMSE {va_tau:.3f}")

        # Early stopping on val total loss
        if va["total"] < best_val:
            best_val = va["total"]
            pat_count = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, config,
                            run_dir / "best_model.pt")
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    # ----- Final evaluation on test set -----
    print("\nEvaluating on test set...")
    te_rmse, te_tau, _ = compute_level_rmse(model, test_loader, device)
    fi_train_rmse, _, _ = compute_level_rmse(model, train_loader, device)
    fi_val_rmse,   _, _ = compute_level_rmse(model, val_loader,   device)

    print("  Per-level r_e RMSE on test set (μm):")
    for i, r in enumerate(te_rmse):
        print(f"    L{i+1:2d}: {r:.3f}")
    print(f"  Mean: {te_rmse.mean():.3f} μm")
    print(f"  τ RMSE: {te_tau:.3f}")

    # Save final model + history
    save_checkpoint(model, optimizer, scheduler, epoch, va["total"],
                    config, run_dir / "final_model.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.savez(
        run_dir / "final_rmse.npz",
        train_rmse=fi_train_rmse, val_rmse=fi_val_rmse, test_rmse=te_rmse,
        test_tau_rmse=te_tau,
    )

    # ----- Figures -----
    print(f"\nWriting training figures (DPI={FIG_DPI}) to: "
          f"{fig_dir.relative_to(_REPO_ROOT)}")
    plot_loss_curves(history, fig_dir)
    plot_rmse_curves(rmse_history, fig_dir)
    plot_final_per_level_rmse(fi_train_rmse, fi_val_rmse, te_rmse, fig_dir)
    plot_pc_trajectories(pc_history, fig_dir)

    print(f"\nCheckpoints in: {run_dir.relative_to(_REPO_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
