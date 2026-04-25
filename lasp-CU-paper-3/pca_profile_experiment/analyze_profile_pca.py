"""
analyze_profile_pca.py
----------------------
Step 1 of the PCA-head experiment: measure the effective dimensionality of the
cloud-droplet profile manifold by running PCA on the training-set profiles.

Why this exists
===============
The current `DropletProfileNetwork` predicts 7 level outputs independently.
With only ~300 unique training profiles this is a data-inefficient output
parameterization: the 7 levels are strongly correlated along a low-dimensional
physical manifold (approximately adiabatic + small deviations), so the network
effectively has far more output degrees of freedom than the data support.

This script:
  1. Loads the training-set profiles from the same HDF5 as the retrieval
     network (respecting the profile-aware train/val/test split).
  2. Fits PCA on the *training* profiles (in normalized [0, 1] space, to match
     what the network sees).
  3. Reports how much variance is explained by the first K components for
     K = 1..7.
  4. Produces several diagnostic figures to let the user decide how many PCs
     are needed.  Figures are saved at 500 DPI in ./figures/pca_analysis/.

No neural network is trained here — this is a pure data-analysis pre-check.

Outputs (in ./figures/pca_analysis/):
  - 01_scree_plot.png          Variance explained per PC + cumulative curve
  - 02_pc_modes.png            The first 5 PC basis vectors in level space
  - 03_mean_profile.png        Training-set mean profile (the PCA centre)
  - 04_reconstruction_grid.png Random held-out profiles reconstructed with
                               K = 1, 2, 3, 5, 7 PCs
  - 05_reconstruction_rmse.png Per-level RMSE of reconstruction vs K
  - 06_score_distributions.png Distributions of PC-1..3 scores on train
                               (checks for Gaussianity / multi-modality)
  - 07_score_vs_tau.png        Do PC scores correlate with cloud optical depth?
                               (informs whether tau and profile are coupled
                               on the training manifold)

Also writes ./pca_basis.npz with the fitted PCA parameters so that the
training and evaluation scripts can load the identical basis.

Usage
=====
    python analyze_profile_pca.py \
        --h5-path /path/to/combined_vocals_oracles_training_data_7-levels_17_April_2026.h5 \
        --n-val-profiles 14 \
        --n-test-profiles 14 \
        --seed 42

The defaults below match sweep 3 (7-level data, seed 42, 14/14 held out).

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Project-root imports.  This script lives in <repo>/pca_profile_experiment/,
# and data.py / models.py live one level up.  Insert parent on sys.path.
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import (  # noqa: E402
    create_profile_aware_splits,
    compute_profile_ids,
    RE_MIN,
    RE_MAX,
    TAU_MIN,
    TAU_MAX,
)

# Default DPI for all saved figures.  The user asked for 500.
FIG_DPI = 500

# Default data file — matches sweep 3 (7-level training data).
DEFAULT_H5 = (
    _REPO_ROOT / "training_data" /
    "combined_vocals_oracles_training_data_7-levels_17_April_2026.h5"
)

# Output directory for this step's figures (relative to the script location).
FIG_DIR = Path(__file__).resolve().parent / "figures" / "pca_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Where to save the fitted PCA basis for downstream scripts.
PCA_BASIS_PATH = Path(__file__).resolve().parent / "pca_basis.npz"


# ───────────────────────────────────────────────────────────────────────────────
# PCA helpers (kept local so the script is self-contained)
# ───────────────────────────────────────────────────────────────────────────────
def fit_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA on `X` via covariance-matrix eigendecomposition.

    `X` has shape (n_samples, n_levels).  We centre by the per-level mean,
    build the (n_levels × n_levels) covariance matrix, and take its leading
    eigenvectors.

    IMPORTANT: the eigendecomposition is performed in **float64** and only
    down-cast to float32 at the very end.  If we used float32 throughout,
    eigenvectors associated with near-zero eigenvalues come back with
    catastrophic orthogonality errors (they're essentially random vectors in
    the numerical null-space).  That makes reconstruction with K below the
    true rank explode — the classic symptom is K=2 RMSE being WORSE than
    K=1, and K>=rank being ≈0.  Using float64 keeps the top eigenvectors
    well-conditioned; we still clip obvious numerical noise below.

    Parameters
    ----------
    X : (n_samples, n_levels) float
        The data matrix.  For our use the rows are normalized r_e profiles
        in [0, 1] space so that scores are numerically comparable.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    mean       : (n_levels,)                 Per-level sample mean
    components : (n_components, n_levels)    Unit-norm PC basis vectors
    explained  : (n_components,)             Fraction of variance per component
    """
    X64 = np.asarray(X, dtype=np.float64)
    mean64 = X64.mean(axis=0)
    Xc = X64 - mean64                              # centred, float64
    # Covariance matrix: (L × L).  Unbiased estimator.
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    # eigh returns eigenvalues in ASCENDING order for a symmetric matrix.
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Clip tiny negative eigenvalues that appear due to floating-point rounding
    # on a rank-deficient matrix.  These should be zero in exact arithmetic.
    # We also detect an effective-rank cliff and emit a warning so the user
    # understands what the downstream reconstruction means.
    max_eig = float(eigvals[-1]) if eigvals.size else 0.0
    tol = max_eig * max(len(cov), 1) * np.finfo(np.float64).eps * 1e3
    eigvals = np.where(eigvals < tol, 0.0, eigvals)

    # Reverse so that eigvals are descending (largest variance first).
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Effective rank: how many PCs contribute meaningfully to variance?
    eff_rank = int(np.sum(eigvals > tol))
    if eff_rank < n_components:
        print(f"  NOTE: covariance matrix has effective rank {eff_rank} "
              f"(< requested {n_components}).  Components beyond index "
              f"{eff_rank} are in the numerical null-space and their "
              f"reconstruction RMSE is not meaningful.  This usually means "
              f"your training profiles were generated from a low-parameter "
              f"model (e.g. a ≤{eff_rank}-parameter adiabatic fit) and then "
              f"evaluated at more levels than the model has degrees of "
              f"freedom.")

    # Take the top K components.  `.T` so rows = components.
    components = eigvecs[:, :n_components].T
    # Fraction of total variance (use all eigvals to get the denominator).
    total_var = eigvals.sum()
    explained = (
        eigvals[:n_components] / total_var if total_var > 0
        else np.zeros(n_components)
    )
    return (
        mean64.astype(np.float32),
        components.astype(np.float32),
        explained.astype(np.float64),
    )


def reconstruct(scores: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Invert PCA: given (N, K) scores → (N, L) reconstructed profiles in the same
    space as the input to `fit_pca`.  Computes in float64 to keep precision
    stable even when `components` rows are near the numerical null-space.
    """
    return (
        np.asarray(scores, dtype=np.float64)
        @ np.asarray(components, dtype=np.float64)
        + np.asarray(mean, dtype=np.float64)
    )


def project(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project (N, L) data onto the K PCA axes.  Returns (N, K) scores.
    Computed in float64 for numerical stability (see `fit_pca` docstring)."""
    return (
        np.asarray(X, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    ) @ np.asarray(components, dtype=np.float64).T


# ───────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────────────
def _savefig(name: str) -> None:
    """Save the current figure to the figures dir at FIG_DPI and close it."""
    path = FIG_DIR / name
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"    saved: {path.relative_to(_REPO_ROOT)}")


def plot_scree(explained: np.ndarray) -> None:
    """Variance explained per PC (bars) + cumulative curve (line)."""
    cum = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    bars = ax1.bar(x, 100 * explained, color="#4C72B0",
                   edgecolor="black", label="Individual")
    ax1.set_xlabel("Principal component index")
    ax1.set_ylabel("Variance explained (%)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(x)

    # Annotate each bar with its percent value.
    for b, v in zip(bars, explained):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, 100 * cum, "o-", color="#C44E52", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative variance (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 105)
    ax2.axhline(95, color="gray", linestyle="--", linewidth=1)
    ax2.text(x[-1], 95, " 95%", va="center", ha="left", fontsize=9, color="gray")

    plt.title("PCA scree plot — training-set droplet profiles")
    fig.tight_layout()
    _savefig("01_scree_plot.png")


def plot_pc_modes(components: np.ndarray, mean: np.ndarray, n_show: int = 5) -> None:
    """
    Plot the first `n_show` PC modes as functions of the level index.
    Mean is shown as a separate thick black line for context.
    """
    n_show = min(n_show, components.shape[0])
    n_levels = components.shape[1]
    level_idx = np.arange(1, n_levels + 1)  # level 1 = cloud top, N = cloud base

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Both panels use atmospheric convention: level on y-axis, L1 at top.
    axes[0].plot(mean, level_idx, "k-o", linewidth=2.5, markersize=6,
                 label="Mean profile")
    axes[0].invert_yaxis()
    axes[0].set_yticks(level_idx)
    axes[0].set_xlabel("r_e (normalized to [0, 1])")
    axes[0].set_ylabel("Level index (1 = cloud top, N = cloud base)")
    axes[0].set_title("Mean profile — the PCA centre")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 0.85, n_show))
    for k in range(n_show):
        axes[1].plot(components[k], level_idx, "-o", color=colors[k],
                     linewidth=1.8, markersize=5, label=f"PC {k+1}")
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set_yticks(level_idx)
    axes[1].set_xlabel("PC loading (unitless)")
    axes[1].set_ylabel("Level index (1 = cloud top, N = cloud base)")
    axes[1].set_title(f"First {n_show} PC basis vectors")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig("02_pc_modes.png")


def plot_mean_profile(mean_phys: np.ndarray) -> None:
    """Mean profile in physical units (μm) — useful sanity check."""
    n_levels = len(mean_phys)
    level_idx = np.arange(1, n_levels + 1)

    plt.figure(figsize=(5.5, 5))
    plt.plot(mean_phys, level_idx, "o-", color="#4C72B0", linewidth=2, markersize=7)
    plt.gca().invert_yaxis()  # cloud top at top
    plt.xlabel("Mean r_e (μm)")
    plt.ylabel("Level index (1 = top)")
    plt.title("Training-set mean droplet profile (physical units)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig("03_mean_profile.png")


def plot_reconstruction_grid(
    train_profiles_norm: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    K_list: list[int],
    rng: np.random.Generator,
    n_examples: int = 5,
) -> None:
    """
    For a handful of held-out example profiles, show the truth and the
    reconstruction using K = 1, 2, 3, 5, 7 PCs.
    """
    n_levels = train_profiles_norm.shape[1]
    level_idx = np.arange(1, n_levels + 1)

    # Pick random example profiles from the training set itself — we are
    # characterizing the manifold, not generalizing.  (We'll also report
    # validation-set reconstruction RMSE later, which is the true test.)
    idx = rng.choice(len(train_profiles_norm), size=n_examples, replace=False)
    examples = train_profiles_norm[idx]                                       # (N_EX, L)

    fig, axes = plt.subplots(1, n_examples, figsize=(3.2 * n_examples, 4.5),
                             sharey=True)
    if n_examples == 1:
        axes = [axes]

    # Precompute reconstructions for each K.  We use the same fitted components;
    # taking the top K just means slicing to the first K rows.
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(K_list)))

    # Convert to physical units for interpretability.
    def to_phys(p_norm: np.ndarray) -> np.ndarray:
        return p_norm * (RE_MAX - RE_MIN) + RE_MIN

    for i, ex in enumerate(examples):
        ax = axes[i]
        ax.plot(to_phys(ex), level_idx, "k-", linewidth=2.5, label="Truth")
        for k_i, K in enumerate(K_list):
            comp_k = components[:K]                                           # (K, L)
            score  = project(ex[None, :], mean, comp_k)                      # (1, K)
            recon  = reconstruct(score, mean, comp_k)[0]                     # (L,)
            ax.plot(to_phys(recon), level_idx, "--",
                    color=colors[k_i], linewidth=1.5, label=f"K={K}")
        ax.invert_yaxis()
        ax.set_xlabel("r_e (μm)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Level index (1 = top)")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Sample #{idx[i]}")

    fig.suptitle("Random training profiles: truth vs PCA reconstruction",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _savefig("04_reconstruction_grid.png")


def plot_reconstruction_rmse(
    profiles_train_norm: np.ndarray,
    profiles_val_norm: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> None:
    """
    Per-level RMSE of PCA reconstruction as a function of K.

    Two panels:
      (left)  Mean RMSE across levels as a function of K, on both train and
              validation splits.  Validation is the honest test — if
              reconstruction RMSE on val is close to training RMSE then the
              manifold generalizes; a large gap means val profiles live on a
              different manifold.
      (right) Per-level RMSE for the selected K values (train split only).

    All RMSE values are reported in *physical* μm so the numbers line up with
    the retrieval-network RMSE you care about.
    """
    n_levels = profiles_train_norm.shape[1]
    K_max = components.shape[0]
    K_values = np.arange(1, K_max + 1)

    def to_phys(p_norm: np.ndarray) -> np.ndarray:
        return p_norm * (RE_MAX - RE_MIN) + RE_MIN

    # ---- RMSE vs K ----
    train_rmse_per_K = np.zeros((K_max, n_levels))
    val_rmse_per_K   = np.zeros((K_max, n_levels))

    for K in range(1, K_max + 1):
        comp_k = components[:K]
        # Train set
        scores_train = project(profiles_train_norm, mean, comp_k)
        recon_train  = reconstruct(scores_train, mean, comp_k)
        err_train    = to_phys(profiles_train_norm) - to_phys(recon_train)
        train_rmse_per_K[K - 1] = np.sqrt((err_train ** 2).mean(axis=0))

        # Val set
        scores_val = project(profiles_val_norm, mean, comp_k)
        recon_val  = reconstruct(scores_val, mean, comp_k)
        err_val    = to_phys(profiles_val_norm) - to_phys(recon_val)
        val_rmse_per_K[K - 1] = np.sqrt((err_val ** 2).mean(axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean RMSE across levels vs K
    axes[0].plot(K_values, train_rmse_per_K.mean(axis=1), "o-",
                 color="#4C72B0", linewidth=2, markersize=8, label="Train")
    axes[0].plot(K_values, val_rmse_per_K.mean(axis=1), "s--",
                 color="#C44E52", linewidth=2, markersize=8, label="Validation")
    axes[0].set_xlabel("K = number of retained PCs")
    axes[0].set_ylabel("Mean per-level RMSE (μm)")
    axes[0].set_title("Reconstruction RMSE vs K")
    axes[0].set_xticks(K_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Right: per-level RMSE curves for a handful of K values.  Level on
    # y-axis, inverted so L1 (cloud top) is at the top of the panel.
    level_idx = np.arange(1, n_levels + 1)
    K_show = [1, 2, 3, min(5, K_max), K_max]
    K_show = sorted(set(K_show))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(K_show)))
    for k_i, K in enumerate(K_show):
        axes[1].plot(train_rmse_per_K[K - 1], level_idx, "-o",
                     color=colors[k_i], linewidth=1.8, markersize=6,
                     label=f"K={K}")
    axes[1].invert_yaxis()
    axes[1].set_yticks(level_idx)
    axes[1].set_xlabel("Train reconstruction RMSE (μm)")
    axes[1].set_ylabel("Level index (1 = cloud top, N = cloud base)")
    axes[1].set_title("Per-level reconstruction RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig("05_reconstruction_rmse.png")

    # Return tables for the caller to print.
    return train_rmse_per_K, val_rmse_per_K


def plot_score_distributions(scores: np.ndarray, n_show: int = 3) -> None:
    """
    Histograms of the first `n_show` PC scores plus 2D joint scatter of the
    first two.  This tells you whether the manifold is roughly Gaussian (good
    for linear PCA) or heavy-tailed / multimodal (hint that a nonlinear
    parameterization or a mixture would do better).
    """
    n_show = min(n_show, scores.shape[1])

    fig, axes = plt.subplots(1, n_show + 1, figsize=(3.5 * (n_show + 1), 4.5))

    for k in range(n_show):
        ax = axes[k]
        ax.hist(scores[:, k], bins=40, color="#4C72B0", edgecolor="black",
                alpha=0.85)
        ax.set_xlabel(f"PC {k+1} score")
        ax.set_ylabel("Count" if k == 0 else "")
        ax.set_title(f"PC {k+1} distribution")
        ax.grid(True, alpha=0.3)

    # Final panel: joint (PC1, PC2) scatter.
    ax = axes[-1]
    ax.scatter(scores[:, 0], scores[:, 1], s=4, alpha=0.35, color="#4C72B0")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("PC 1 score")
    ax.set_ylabel("PC 2 score")
    ax.set_title("Joint distribution (PC 1 vs PC 2)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig("06_score_distributions.png")


def plot_scores_vs_tau(scores: np.ndarray, tau_c: np.ndarray, n_show: int = 3) -> None:
    """
    PC scores vs τ_c.  If the first few PC scores are strongly correlated with
    τ_c then a joint (multi-task) output makes physical sense — τ_c and the
    profile share information.  If uncorrelated then separating the heads is
    safer.  This directly informs question Q1 from the earlier conversation.
    """
    n_show = min(n_show, scores.shape[1])

    fig, axes = plt.subplots(1, n_show, figsize=(4.2 * n_show, 4.5))
    if n_show == 1:
        axes = [axes]
    for k in range(n_show):
        ax = axes[k]
        # Pearson correlation coefficient, printed on-plot.
        r = float(np.corrcoef(scores[:, k], tau_c)[0, 1])
        ax.scatter(tau_c, scores[:, k], s=4, alpha=0.35, color="#4C72B0")
        ax.set_xlabel("τ_c (cloud optical depth)")
        ax.set_ylabel(f"PC {k+1} score")
        ax.set_title(f"PC {k+1} vs τ_c   (r = {r:.2f})")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig("07_score_vs_tau.png")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: PCA analysis of training-set droplet profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5-path", type=str, default=str(DEFAULT_H5),
        help="Path to the HDF5 training-data file.",
    )
    parser.add_argument(
        "--n-val-profiles", type=int, default=14,
        help="Number of unique profiles held out for validation.",
    )
    parser.add_argument(
        "--n-test-profiles", type=int, default=14,
        help="Number of unique profiles held out for test.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the profile-aware split (match the training run).",
    )
    parser.add_argument(
        "--max-components", type=int, default=None,
        help="Cap on the number of PCs analysed.  Default: all levels.",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    print("=" * 78)
    print("PCA analysis of cloud-droplet profile manifold")
    print("=" * 78)
    print(f"HDF5 file: {h5_path}")
    print(f"Split: val={args.n_val_profiles} profiles, "
          f"test={args.n_test_profiles} profiles, seed={args.seed}")

    # ------------------------------------------------------------------
    # 1. Load profiles + τ_c from the HDF5.  We use the same profile-aware
    #    split as the training code so that the validation reconstruction
    #    test below is fair.
    # ------------------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        profiles_raw = f["profiles"][:].astype(np.float32)   # (n_samples, n_levels) in μm
        tau_raw      = f["tau_c"][:].astype(np.float32)      # (n_samples,)

    n_levels = profiles_raw.shape[1]
    print(f"\nData loaded: {len(profiles_raw):,} samples × {n_levels} levels")
    print(f"  r_e range: {profiles_raw.min():.2f} – {profiles_raw.max():.2f} μm")
    print(f"  τ_c range: {tau_raw.min():.2f} – {tau_raw.max():.2f}")

    # Profile-aware split: get indices for train / val / test so we never mix
    # cloud profiles across splits.  The PCA is fit on the TRAIN split only.
    train_idx, val_idx, test_idx = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=args.n_val_profiles,
        n_test_profiles=args.n_test_profiles,
        seed=args.seed,
    )

    # Count unique profiles (via fingerprints) for reporting.
    profile_ids = compute_profile_ids(str(h5_path))
    n_unique_total = int(profile_ids.max()) + 1
    n_unique_train = len(np.unique(profile_ids[train_idx]))
    n_unique_val   = len(np.unique(profile_ids[val_idx]))
    n_unique_test  = len(np.unique(profile_ids[test_idx]))
    print(f"\nUnique-profile counts:")
    print(f"  total: {n_unique_total}")
    print(f"  train: {n_unique_train}")
    print(f"  val:   {n_unique_val}")
    print(f"  test:  {n_unique_test}")
    print(f"Sample-level counts:  train={len(train_idx):,}  "
          f"val={len(val_idx):,}  test={len(test_idx):,}")

    # For PCA we only need the UNIQUE profiles in the train split, not every
    # (profile × geometry) row.  Using all rows biases PCA toward profiles that
    # happen to have more geometry samples.
    train_profile_ids_unique, first_occurrence = np.unique(
        profile_ids[train_idx], return_index=True,
    )
    train_rows_for_pca = train_idx[first_occurrence]
    train_profiles_phys = profiles_raw[train_rows_for_pca]                    # (N_prof, L) μm
    print(f"\nPCA will be fit on {len(train_profiles_phys)} unique training profiles.")

    # Same trick for validation reconstruction benchmarking.
    val_profile_ids_unique, val_first_occurrence = np.unique(
        profile_ids[val_idx], return_index=True,
    )
    val_rows_for_pca = val_idx[val_first_occurrence]
    val_profiles_phys = profiles_raw[val_rows_for_pca]

    # ------------------------------------------------------------------
    # 2. Normalize to [0, 1] (matching the retrieval network's target space)
    #    and fit PCA on the training profiles.
    # ------------------------------------------------------------------
    def normalize(p):  # μm -> [0,1]
        return (p - RE_MIN) / (RE_MAX - RE_MIN)

    train_norm = normalize(train_profiles_phys)
    val_norm   = normalize(val_profiles_phys)

    K_max = args.max_components if args.max_components else n_levels
    K_max = min(K_max, n_levels)
    mean, components, explained = fit_pca(train_norm, n_components=K_max)

    cum = np.cumsum(explained)
    print("\nVariance explained by each PC:")
    for k, (e, c) in enumerate(zip(explained, cum), start=1):
        print(f"  PC {k}:  {100*e:6.2f}%     cumulative: {100*c:6.2f}%")

    # ------------------------------------------------------------------
    # 3. Plots.
    # ------------------------------------------------------------------
    print(f"\nWriting figures (DPI = {FIG_DPI}) to: {FIG_DIR.relative_to(_REPO_ROOT)}")
    plot_scree(explained)
    plot_pc_modes(components, mean, n_show=min(5, K_max))
    plot_mean_profile(mean * (RE_MAX - RE_MIN) + RE_MIN)
    rng = np.random.default_rng(args.seed)
    plot_reconstruction_grid(train_norm, mean, components,
                             K_list=[1, 2, 3, min(5, K_max), K_max],
                             rng=rng, n_examples=5)
    train_rmse_per_K, val_rmse_per_K = plot_reconstruction_rmse(
        train_norm, val_norm, mean, components,
    )
    scores_train = project(train_norm, mean, components)                      # (N_prof, K_max)
    plot_score_distributions(scores_train, n_show=min(3, K_max))

    # Use τ_c for the unique profiles used to fit PCA.
    tau_for_pca = tau_raw[train_rows_for_pca]
    plot_scores_vs_tau(scores_train, tau_for_pca, n_show=min(3, K_max))

    # ------------------------------------------------------------------
    # 4. Summary tables.
    # ------------------------------------------------------------------
    print("\nReconstruction RMSE (μm) vs K — TRAIN split:")
    print("  K     mean   L1(top)   " + "  ".join(f"L{i+1}" for i in range(1, n_levels)))
    for K in range(1, K_max + 1):
        row = train_rmse_per_K[K - 1]
        print(f"  {K}  {row.mean():6.3f}   " +
              "  ".join(f"{v:5.3f}" for v in row))

    print("\nReconstruction RMSE (μm) vs K — VALIDATION split:")
    print("  K     mean   L1(top)   " + "  ".join(f"L{i+1}" for i in range(1, n_levels)))
    for K in range(1, K_max + 1):
        row = val_rmse_per_K[K - 1]
        print(f"  {K}  {row.mean():6.3f}   " +
              "  ".join(f"{v:5.3f}" for v in row))

    # ------------------------------------------------------------------
    # 5. Save PCA basis for reuse by train_pca.py and compare script.
    # ------------------------------------------------------------------
    np.savez(
        PCA_BASIS_PATH,
        mean=mean,                                # (L,)
        components=components,                    # (K_max, L)
        explained=explained,                      # (K_max,)
        train_rmse_per_K=train_rmse_per_K,        # (K_max, L) μm
        val_rmse_per_K=val_rmse_per_K,            # (K_max, L) μm
        h5_path=str(h5_path),
        seed=args.seed,
        n_val_profiles=args.n_val_profiles,
        n_test_profiles=args.n_test_profiles,
        n_levels=n_levels,
        re_min=RE_MIN,
        re_max=RE_MAX,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
    )
    print(f"\nPCA basis saved to: {PCA_BASIS_PATH.relative_to(_REPO_ROOT)}")
    print("\nNext step:  choose K (a good rule of thumb is the smallest K whose")
    print("            validation reconstruction RMSE is comparable to the")
    print("            current network's mid-cloud RMSE — ~0.6 μm), then run")
    print("            train_pca.py --n-pca-components K")
    print("=" * 78)


if __name__ == "__main__":
    main()
