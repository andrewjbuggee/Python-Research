"""
diagnose_pc_retrievability.py
-----------------------------
Diagnostic A: ask the data, not the model — "is each PC k retrievable from
the spectrum at all, in principle?"

The PCA-head network's per-PC scatter shows that PC1 is well retrieved
(R² ≈ 0.85) and PC2..K are nearly flat (R² < 0.2).  Two possible causes:

  (i)  The spectrum encodes PC2..K but the encoder fails to extract them.
  (ii) The spectrum DOESN'T encode PC2..K — no encoder can save you.

This script answers (ii) directly by measuring the *linear* retrievability
of each PC score from the input.  If the upper bound on linear retrieval is
already low for a PC, no MLP, Conv1D, or attention model will cross it
without using nonlinearities the data can't justify.

What it computes (on the TRAINING split, with held-out validation for the
multivariate fit):

  (a) Single-channel R² — for each (PC k, spectral channel c), Pearson
      correlation² between the channel reflectance and the true PC score.
      Shows which wavelengths carry information about each PC, and gives
      an absolute lower bound on retrievability (since linear-OLS-of-all
      channels is at least as good as the best single channel).

  (b) Geometry input contribution — same single-feature R² for the four
      geometry inputs (sza, vza, saz, vaz) so we can tell whether observation
      geometry, not the spectrum, explains a PC.

  (c) Multivariate linear ceiling — OLS fit `pc_score = X @ w + b` on the
      train split using all 640 inputs, evaluated on the val split.  This is
      the maximum achievable R² for any LINEAR predictor of the PC score.
      If even this is small, the relationship must be nonlinear OR absent.

Outputs (figures at 500 DPI inside ./figures/pc_retrievability/):

  01_per_channel_r2_heatmap.png   K rows × 636 channels.  Color = R².
  02_per_channel_r2_curves.png    R² vs wavelength, one curve per PC.
  03_linear_retrievability.png    Per PC: best-single-channel R² vs OLS
                                  (train / val) R² + geometry-only R².
                                  THIS is the headline plot.
  04_top_channels_per_pc.png      Top-10 channels for each PC, with their
                                  wavelengths.

Also prints a summary table to stdout.

Usage
=====
    python diagnose_pc_retrievability.py \\
        --pca-basis pca_basis.npz \\
        --max-k 7

If --max-k is omitted, all 50 PCs are analyzed (heatmap is large but
informative).  Pick K=7 to mirror the network experiments.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Repo-root imports (data.py / RE_MIN etc.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import RE_MIN, RE_MAX, create_profile_aware_splits  # noqa: E402

# Local: load_pca_basis pulls the saved basis with metadata.
from models_pca import load_pca_basis  # noqa: E402


FIG_DPI = 500
FIG_DIR = Path(__file__).resolve().parent / "figures" / "pc_retrievability"


# ───────────────────────────────────────────────────────────────────────────────
# Math
# ───────────────────────────────────────────────────────────────────────────────
def pearson_r2_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson R² between every column of X (shape N×F) and the 1-D target y
    (shape N).  Returns an array of shape (F,).

    All math in float64 to avoid catastrophic cancellation when columns have
    nearly-zero variance (e.g. constant geometry inputs in special cases).
    """
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    N = X.shape[0]
    Xm = X.mean(axis=0)
    Xc = X - Xm
    ym = y.mean()
    yc = y - ym
    num = (Xc * yc[:, None]).sum(axis=0)               # (F,)
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-30)
    r = num / den
    return r ** 2


def ols_fit_eval(X_train, y_train, X_val, y_val,
                 ridge: float = 0.0) -> dict:
    """
    OLS (or ridge if ridge>0) closed-form fit on train, evaluated on val.

    Returns dict with train_r2, val_r2, weights, bias.

    Float64 throughout.  Adds a column of ones for the intercept and uses
    `np.linalg.lstsq` (numerically robust SVD) so we don't have to worry
    about ill-conditioning of X.T @ X for collinear spectral channels.
    """
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    X_val   = X_val.astype(np.float64)
    y_val   = y_val.astype(np.float64)

    A_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    if ridge > 0:
        # Ridge via augmenting (A, λI) and (y, 0).  Skip the bias regularization.
        F = X_train.shape[1]
        L = np.eye(F + 1) * np.sqrt(ridge)
        L[-1, -1] = 0.0  # do not penalize bias
        A_aug = np.vstack([A_train, L])
        y_aug = np.concatenate([y_train, np.zeros(F + 1)])
        coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    w, b = coef[:-1], coef[-1]

    yhat_train = A_train @ coef
    A_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    yhat_val = A_val @ coef

    def r2(y, yhat):
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "train_r2": float(r2(y_train, yhat_train)),
        "val_r2":   float(r2(y_val,   yhat_val)),
        "weights":  w,
        "bias":     float(b),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────────
def plot_r2_heatmap(r2_matrix: np.ndarray, wavelengths: np.ndarray,
                    fig_path: Path):
    """
    r2_matrix : (K, n_channels) of R² values
    """
    K = r2_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.3 * K))
    im = ax.imshow(
        r2_matrix, aspect="auto", origin="lower", cmap="magma",
        extent=[wavelengths[0], wavelengths[-1], 0.5, K + 0.5],
        vmin=0.0, vmax=max(0.10, float(r2_matrix.max())),
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("PC mode")
    ax.set_yticks(np.arange(1, K + 1))
    ax.set_title("Per-channel single-feature R²:  reflectance vs true PC score")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("R²")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_r2_curves(r2_matrix: np.ndarray, wavelengths: np.ndarray,
                   fig_path: Path):
    K = r2_matrix.shape[0]
    fig, axes = plt.subplots(K, 1, figsize=(11, 1.6 * K), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.plot(wavelengths, r2_matrix[k], color="#4C72B0", linewidth=0.9)
        # Highlight the best single channel.
        c_best = int(np.argmax(r2_matrix[k]))
        ax.axvline(wavelengths[c_best], color="#C44E52", linestyle="--",
                   linewidth=0.8, alpha=0.7)
        ax.text(wavelengths[c_best], r2_matrix[k, c_best],
                f"  λ={wavelengths[c_best]:.0f} nm\n  R²={r2_matrix[k, c_best]:.3f}",
                fontsize=8, va="bottom", ha="left",
                color="#C44E52")
        ax.set_ylabel(f"PC{k+1}\nR²")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.10, float(r2_matrix[k].max()) * 1.15))
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle("Single-channel linear R² vs wavelength, per PC", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_linear_retrievability(per_pc_summary: list[dict], fig_path: Path):
    """
    Per-PC linear-retrievability summary.

    Bars compared:
      - best single spectral channel (training Pearson R²)
      - full-feature OLS train R² (upper bound, possibly overfit)
      - ridge val R² on a within-train RANDOM holdout
        (the cleanest measure of: "does the spectrum carry this PC?")
      - full-feature OLS val R² on the PROFILE-HELD-OUT split
        (population-shift sensitive; can be negative)

    Headline interpretation: the ridge-random bar is the most reliable
    estimate of pure linear retrievability.  If it's near zero for a PC,
    no linear function of the inputs can predict that PC.
    """
    K = len(per_pc_summary)
    x = np.arange(1, K + 1)
    w = 0.20
    best_chan  = np.array([s["best_channel_r2"]    for s in per_pc_summary])
    full_train = np.array([s["full_ols_train_r2"]  for s in per_pc_summary])
    ridge_rand = np.array([s["ridge_inner_val_r2"] for s in per_pc_summary])
    full_val   = np.array([s["full_ols_val_r2"]    for s in per_pc_summary])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * w, best_chan,  w, color="#4C72B0", edgecolor="black",
           label="Best single spectral channel")
    ax.bar(x - 0.5 * w, full_train, w, color="#DD8452", edgecolor="black",
           label="Full-feature OLS (train)")
    ax.bar(x + 0.5 * w, ridge_rand, w, color="#55A868", edgecolor="black",
           label="Ridge val R² (random within-train holdout)  ← key bar")
    # Profile-held-out OLS val R²: clip the bar visually but annotate the value
    # if it goes negative.
    bars_val = ax.bar(x + 1.5 * w, np.clip(full_val, 0, None), w,
                      color="#C44E52", edgecolor="black",
                      label="Full-feature OLS val (profile-held-out)")
    for xi, v in zip(x + 1.5 * w, full_val):
        if v < 0:
            ax.annotate(f"{v:.2f}", xy=(xi, 0.0), xytext=(xi, 0.04),
                        ha="center", fontsize=7, color="#C44E52",
                        arrowprops=dict(arrowstyle="-", color="#C44E52", lw=0.5))

    ax.axhline(0.05, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.4, 0.055, "noise floor (R²≈0.05)", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("R²")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Linear retrievability per PC\n"
        "Ridge-random R² answers 'does the spectrum encode this PC?';"
        " profile-val R² answers 'does the linear map generalize across profiles?'"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_channels(r2_matrix: np.ndarray, wavelengths: np.ndarray,
                      fig_path: Path, top_n: int = 10):
    K = r2_matrix.shape[0]
    fig, axes = plt.subplots(K, 1, figsize=(9, 1.5 * K), sharex=False)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        idx = np.argsort(r2_matrix[k])[::-1][:top_n]
        ax.bar(np.arange(top_n), r2_matrix[k, idx],
               color="#4C72B0", edgecolor="black")
        ax.set_xticks(np.arange(top_n))
        ax.set_xticklabels([f"{wavelengths[i]:.0f}" for i in idx],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"PC{k+1} R²")
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].set_xlabel("Wavelength (nm) of top channels")
    fig.suptitle(f"Top-{top_n} most-correlated spectral channels per PC", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Linear-retrievability diagnostic for PC scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pca-basis", type=str,
        default=str(Path(__file__).parent / "pca_basis.npz"),
        help="Path to the saved PCA basis.",
    )
    parser.add_argument(
        "--max-k", type=int, default=7,
        help="Number of leading PCs to analyze.",
    )
    parser.add_argument(
        "--instrument", type=str, default="hysics",
        choices=["hysics", "emit"],
        help="Which simulated reflectance to use as input.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load basis & resolve metadata -----
    basis = load_pca_basis(args.pca_basis)
    h5_path = Path(str(basis["h5_path"]))
    if not h5_path.exists():
        # The basis may have stored an absolute path that is not portable.
        # Fall back to <repo_root>/training_data/<filename>.
        h5_path = _REPO_ROOT / "training_data" / h5_path.name
    print(f"HDF5: {h5_path}")
    print(f"PCA basis fit on: seed={int(basis['seed'])}, "
          f"n_val_profiles={int(basis['n_val_profiles'])}, "
          f"n_test_profiles={int(basis['n_test_profiles'])}")

    K = min(int(args.max_k), int(basis["components"].shape[0]))

    # ----- Load reflectance + geometry + profiles from HDF5 -----
    refl_key = f"reflectances_{args.instrument}"
    with h5py.File(h5_path, "r") as f:
        refl = f[refl_key][:].astype(np.float32)        # (N, 636)
        wavelengths = f["wavelengths"][:].astype(np.float64)  # (636,)
        sza = f["sza"][:].astype(np.float32)
        vza = f["vza"][:].astype(np.float32)
        saz = f["saz"][:].astype(np.float32)
        vaz = f["vaz"][:].astype(np.float32)
        profiles_um = f["profiles"][:].astype(np.float32)  # (N, 50)
    print(f"Loaded {refl.shape[0]:,} samples × {refl.shape[1]} channels"
          f" + 4 geometry inputs")

    # Normalize profiles into the [0, 1] space PCA was fit in.
    profiles_norm = (profiles_um - RE_MIN) / (RE_MAX - RE_MIN)

    # ----- Profile-aware split (same as PCA fit) -----
    train_idx, val_idx, _ = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    print(f"Train: {len(train_idx):,} samples | Val: {len(val_idx):,} samples")

    # ----- Compute true PC scores (float64) -----
    pca_mean = basis["mean"].astype(np.float64)
    pca_comp = basis["components"][:K].astype(np.float64)

    centered_train = profiles_norm[train_idx].astype(np.float64) - pca_mean
    centered_val   = profiles_norm[val_idx].astype(np.float64)   - pca_mean
    scores_train = centered_train @ pca_comp.T   # (N_train, K)
    scores_val   = centered_val   @ pca_comp.T   # (N_val,   K)

    # ----- Build feature matrices -----
    refl_train = refl[train_idx]   # (N_train, 636)
    refl_val   = refl[val_idx]
    geom_train = np.stack([sza[train_idx], vza[train_idx],
                           saz[train_idx], vaz[train_idx]], axis=1)
    geom_val   = np.stack([sza[val_idx],   vza[val_idx],
                           saz[val_idx],   vaz[val_idx]], axis=1)
    full_train = np.concatenate([refl_train, geom_train], axis=1)  # (N, 640)
    full_val   = np.concatenate([refl_val,   geom_val],   axis=1)

    # ----- Per-channel R² (training set only — diagnostic) -----
    n_chan = refl_train.shape[1]
    r2_per_channel = np.zeros((K, n_chan), dtype=np.float64)
    r2_geometry    = np.zeros((K, 4), dtype=np.float64)
    print("\nComputing per-channel R²…")
    for k in range(K):
        r2_per_channel[k] = pearson_r2_columns(refl_train, scores_train[:, k])
        r2_geometry[k]    = pearson_r2_columns(geom_train, scores_train[:, k])

    # ----- Multivariate linear ceiling -----
    # We report THREE complementary numbers per PC:
    #   (1) full-OLS train R²          — upper bound on linear retrievability,
    #                                     subject to overfitting
    #   (2) ridge val R² (random split) — within-training-pool generalization
    #                                     of the linear relationship.  Answers
    #                                     "does a regularized linear function
    #                                     of the inputs predict this PC for
    #                                     unseen samples drawn from the same
    #                                     profile population?"
    #   (3) full-OLS val R² (profile-held-out) — generalization to new
    #                                     PROFILES.  Negative here means the
    #                                     held-out profile population has
    #                                     systematically different scores than
    #                                     the train population (only 14 unique
    #                                     val profiles → small effective sample).
    # Together (2) tells us whether the SPECTRUM encodes the PC, while (3)
    # tells us whether the encoding generalizes across profile populations.
    print("\nFitting full-feature OLS + ridge (640 features each)…")

    # Ridge λ is set per-PC by leave-out variance scale: λ = 1e-2 * N_train,
    # which empirically gives a sensible balance of bias/variance for this
    # sample size and feature count.  This is NOT cross-validated — for a
    # diagnostic this is fine and keeps the script fast.
    ridge_lambda = 1e-2 * full_train.shape[0]

    # Make a random within-train holdout (10% of train, by sample) so we can
    # measure linear retrievability WITHOUT confounding from profile-pop shift.
    rng = np.random.default_rng(0)
    n_tr = full_train.shape[0]
    perm = rng.permutation(n_tr)
    n_inner_val = max(2000, n_tr // 10)
    inner_val_idx   = perm[:n_inner_val]
    inner_train_idx = perm[n_inner_val:]

    per_pc_summary = []
    for k in range(K):
        # Single-channel ceiling for this PC
        c_best = int(np.argmax(r2_per_channel[k]))
        best_chan_r2 = float(r2_per_channel[k, c_best])

        # Geometry-only OLS (4 features), profile-held-out val
        geom_fit = ols_fit_eval(geom_train, scores_train[:, k],
                                geom_val,   scores_val[:, k])

        # Full-feature OLS (640 features), profile-held-out val
        full_fit = ols_fit_eval(full_train, scores_train[:, k],
                                full_val,   scores_val[:, k])

        # Ridge on within-training random holdout (measures pure linear
        # retrievability of the spectrum, modulo regularization).
        ridge_fit = ols_fit_eval(
            full_train[inner_train_idx], scores_train[inner_train_idx, k],
            full_train[inner_val_idx],   scores_train[inner_val_idx,   k],
            ridge=ridge_lambda,
        )

        summary = {
            "pc": k + 1,
            "best_channel_idx":   c_best,
            "best_channel_lambda": float(wavelengths[c_best]),
            "best_channel_r2":    best_chan_r2,
            "geom_ols_val_r2":    geom_fit["val_r2"],
            "full_ols_train_r2":  full_fit["train_r2"],
            "full_ols_val_r2":    full_fit["val_r2"],
            "ridge_inner_val_r2": ridge_fit["val_r2"],
            "score_std":          float(scores_train[:, k].std()),
        }
        per_pc_summary.append(summary)
        print(f"  PC{k+1}: best ch λ={summary['best_channel_lambda']:7.1f} nm "
              f"(R²={best_chan_r2:.3f}) | "
              f"ridge random-val R²={ridge_fit['val_r2']:.3f} | "
              f"OLS train R²={full_fit['train_r2']:.3f} | "
              f"OLS profile-val R²={full_fit['val_r2']:.3f}")

    # ----- Save figures -----
    plot_r2_heatmap(r2_per_channel, wavelengths,
                    FIG_DIR / "01_per_channel_r2_heatmap.png")
    plot_r2_curves(r2_per_channel, wavelengths,
                   FIG_DIR / "02_per_channel_r2_curves.png")
    plot_linear_retrievability(per_pc_summary,
                               FIG_DIR / "03_linear_retrievability.png")
    plot_top_channels(r2_per_channel, wavelengths,
                      FIG_DIR / "04_top_channels_per_pc.png", top_n=10)

    # ----- Headline summary -----
    print("\n" + "=" * 86)
    print(f"Linear retrievability summary  (K = {K})")
    print("=" * 86)
    print(f"{'PC':>3} | {'best λ':>8} | {'best ch R²':>10} | "
          f"{'OLS train':>9} | {'ridge rand':>10} | {'OLS prof-val':>12}")
    print("-" * 86)
    for s in per_pc_summary:
        print(f"{s['pc']:>3} | {s['best_channel_lambda']:>7.1f}n | "
              f"{s['best_channel_r2']:>10.3f} | "
              f"{s['full_ols_train_r2']:>9.3f} | "
              f"{s['ridge_inner_val_r2']:>10.3f} | "
              f"{s['full_ols_val_r2']:>12.3f}")
    print("-" * 86)
    print(f"\nFigures written to {FIG_DIR.relative_to(_REPO_ROOT)}")
    print("\nReading guide:")
    print("  • 'best ch R²'        single-channel Pearson R² (training).")
    print("  • 'OLS train'         linear ceiling — overfits with 640 features.")
    print("  • 'ridge rand'        ← KEY: ridge val R² on random within-train.")
    print("                          High ⇒ spectrum encodes this PC.")
    print("                          ≈0  ⇒ no linear map exists; probably no")
    print("                                map at all.")
    print("  • 'OLS prof-val'      generalization to NEW PROFILES.  Negative")
    print("                          values indicate population shift between")
    print("                          train and held-out profile sets, not")
    print("                          absence of signal.")


if __name__ == "__main__":
    main()
