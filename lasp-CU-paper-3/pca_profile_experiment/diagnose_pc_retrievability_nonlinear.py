"""
diagnose_pc_retrievability_nonlinear.py
---------------------------------------
Diagnostic B: measure the NONLINEAR retrievability ceiling of each PC score.

Diagnostic A established the LINEAR ceiling (ridge regression on 640 inputs):
  PC1 ≈ 0.45 | PC2 ≈ 0.14 | PC3..7 < 0.05.

A neural network can in principle fit nonlinear features (band ratios,
products, thresholded interactions) that pure ridge cannot.  This script
asks: how much room is there above the linear ceiling?  We use gradient-
boosted decision trees (sklearn HistGradientBoostingRegressor) as a strong,
model-class-agnostic nonlinear regressor, fit per-PC on the same inputs as
diagnostic A.  GBM is one of the strongest off-the-shelf tabular regressors;
if GBM fails to do better than ridge for a PC, there is no useful nonlinear
signal in (spectrum + geometry) that any tabular model can extract — which
strongly suggests the MLP encoder won't either.

What it computes (for k = 1 .. K, on the same profile-aware split as
diagnostic A):

  (a) Linear baseline:   ridge regression
        - random within-train 10% holdout R²  (no profile-pop shift)
        - profile-held-out val R²             (generalization)

  (b) Nonlinear:         HistGradientBoostingRegressor
        - random within-train 10% holdout R²  ← NONLINEAR ceiling
        - profile-held-out val R²             (generalization)

  (c) Headroom = nonlinear - linear (random-holdout R²).
      Headroom > 0  ⇒  a nonlinear function of the inputs can do strictly
                       better than any linear function.  An MLP encoder can
                       in principle reach this ceiling.
      Headroom ≈ 0  ⇒  the input simply does not carry recoverable info
                       about this PC, linear or nonlinear.

Optionally compares to the *current* network's per-PC profile-val R², passed
in as `--nn-r2 r1 r2 r3 ...`, so you can see "current NN vs nonlinear ceiling"
in one plot.

Outputs (figures at 500 DPI inside ./figures/pc_retrievability_nonlinear/):

  01_linear_vs_nonlinear_ceiling.png   ← Headline plot.
  02_hgb_pred_vs_true_scatter.png      Per-PC HGB predicted vs true scatter
                                       on the profile-held-out val set.
  03_headroom_per_pc.png               nonlinear - linear (random-holdout R²),
                                       per PC, with current-NN gap if --nn-r2.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: set thread-limit env vars BEFORE importing numpy / sklearn.
#
# On macOS with anaconda's numpy (Accelerate / MKL) + sklearn's libomp,
# HistGradientBoostingRegressor can segfault due to multiple OpenMP runtimes
# being loaded into the same process.  Forcing single-threaded BLAS / OMP
# avoids the conflict at the cost of some speed.
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Repo-root imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import RE_MIN, RE_MAX, create_profile_aware_splits  # noqa: E402

from models_pca import load_pca_basis  # noqa: E402

# sklearn for HGB + ridge baseline + RF fallback.
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FIG_DPI = 500
FIG_DIR = Path(__file__).resolve().parent / "figures" / "pc_retrievability_nonlinear"


# ───────────────────────────────────────────────────────────────────────────────
# Math helpers
# ───────────────────────────────────────────────────────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ───────────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────────
def plot_linear_vs_nonlinear(per_pc: list[dict], fig_path: Path,
                              nn_r2: list[float] | None = None):
    """
    Headline figure: per-PC linear ceiling vs nonlinear ceiling.

    For each PC we show 4 (or 5) bars:
      - Ridge (random holdout)        ← linear ceiling
      - Ridge (profile-val)           generalization of linear map
      - HGB   (random holdout)        ← NONLINEAR ceiling
      - HGB   (profile-val)           generalization of nonlinear map
      - (optional) current NN profile-val R² — passed via --nn-r2
    """
    K = len(per_pc)
    x = np.arange(1, K + 1)
    n_bars = 5 if nn_r2 is not None else 4
    w = 0.85 / n_bars

    ridge_rand = np.array([s["ridge_rand_r2"]   for s in per_pc])
    ridge_pval = np.array([s["ridge_pval_r2"]   for s in per_pc])
    hgb_rand   = np.array([s["hgb_rand_r2"]     for s in per_pc])
    hgb_pval   = np.array([s["hgb_pval_r2"]     for s in per_pc])

    fig, ax = plt.subplots(figsize=(10, 5.2))

    # Layout the bars symmetrically around each PC tick.
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * w

    ax.bar(x + offsets[0], ridge_rand, w, color="#4C72B0", edgecolor="black",
           label="Ridge   (random holdout)  ← linear ceiling")
    ax.bar(x + offsets[1], np.clip(ridge_pval, 0, None), w,
           color="#A0C0E0", edgecolor="black",
           label="Ridge   (profile-val)")
    ax.bar(x + offsets[2], hgb_rand, w, color="#55A868", edgecolor="black",
           label="HGB     (random holdout)  ← NONLINEAR ceiling")
    ax.bar(x + offsets[3], np.clip(hgb_pval, 0, None), w,
           color="#9CCBA0", edgecolor="black",
           label="HGB     (profile-val)")

    if nn_r2 is not None:
        nn_arr = np.array(nn_r2[:K] + [np.nan] * max(0, K - len(nn_r2)))
        ax.bar(x + offsets[4], np.clip(np.nan_to_num(nn_arr, nan=0.0), 0, None),
               w, color="#C44E52", edgecolor="black",
               label="Current NN (profile-val)")

    # Annotate negative profile-val values for ridge and HGB.
    for k, s in enumerate(per_pc):
        for vname, color, off in (
            ("ridge_pval_r2", "#4C72B0", offsets[1]),
            ("hgb_pval_r2",   "#55A868", offsets[3]),
        ):
            v = s[vname]
            if v < 0:
                ax.annotate(f"{v:.2f}", xy=(x[k] + off, 0.0),
                            xytext=(x[k] + off, 0.05),
                            ha="center", fontsize=7, color=color,
                            arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axhline(0.05, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.4, 0.07, "noise floor (R²≈0.05)", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("R²")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Linear vs nonlinear retrievability ceiling per PC\n"
        "If green > blue, a nonlinear regressor unlocks signal that ridge misses."
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true_scatter(per_pc: list[dict], fig_path: Path):
    """
    K-panel grid: HGB predicted PC score vs true PC score on profile-val.

    A diagonal line is drawn for reference.  Tight scatter ⇒ good retrieval.
    """
    K = len(per_pc)
    n_cols = min(K, 4)
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for k, s in enumerate(per_pc):
        ax = axes[k // n_cols, k % n_cols]
        true = s["pval_true"]
        pred = s["pval_pred"]
        # 2-D histogram for dense scatter.
        ax.hexbin(true, pred, gridsize=40, cmap="viridis", mincnt=1, linewidths=0)
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.7)
        ax.set_xlabel("True PC score")
        ax.set_ylabel("HGB predicted")
        ax.set_title(f"PC{s['pc']}  R²(prof-val) = {s['hgb_pval_r2']:.3f}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    # Hide unused panels.
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    fig.suptitle("HGB predicted vs true PC score (profile-held-out val)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_headroom(per_pc: list[dict], fig_path: Path,
                  nn_r2: list[float] | None = None):
    """
    Bar chart: nonlinear - linear (random holdout) per PC.  Positive ⇒
    nonlinear regressor recovers signal a linear regressor misses.

    If nn_r2 is given, also plot the gap from the *current network* to the
    nonlinear ceiling on profile-val (the actionable headroom).
    """
    K = len(per_pc)
    x = np.arange(1, K + 1)
    headroom_rand = np.array([s["hgb_rand_r2"] - s["ridge_rand_r2"]
                              for s in per_pc])

    n_panels = 2 if nn_r2 is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.5 * n_panels),
                             squeeze=False)
    ax = axes[0, 0]
    bars = ax.bar(x, headroom_rand, color="#55A868", edgecolor="black")
    for xi, v in zip(x, headroom_rand):
        ax.text(xi, v + (0.005 if v >= 0 else -0.02),
                f"{v:+.3f}", ha="center", fontsize=8,
                va="bottom" if v >= 0 else "top")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("ΔR²")
    ax.set_title(
        "Nonlinear headroom over linear:  HGB R² − Ridge R²  (random holdout)\n"
        ">0 ⇒ a nonlinear function of the inputs unlocks information ridge misses."
    )
    ax.grid(True, alpha=0.3, axis="y")

    if nn_r2 is not None:
        ax2 = axes[1, 0]
        nn_arr = np.array(nn_r2[:K] + [np.nan] * max(0, K - len(nn_r2)))
        hgb_pval = np.array([s["hgb_pval_r2"] for s in per_pc])
        # Headroom from current NN to HGB ceiling on profile-val:
        actionable = hgb_pval - nn_arr
        bars2 = ax2.bar(x, actionable, color="#DD8452", edgecolor="black")
        for xi, v in zip(x, actionable):
            if np.isnan(v):
                continue
            ax2.text(xi, v + (0.005 if v >= 0 else -0.02),
                     f"{v:+.3f}", ha="center", fontsize=8,
                     va="bottom" if v >= 0 else "top")
        ax2.axhline(0.0, color="black", linewidth=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"PC{i}" for i in x])
        ax2.set_ylabel("ΔR²")
        ax2.set_title(
            "Actionable headroom:  HGB R² − Current-NN R²  (profile-val)\n"
            ">0 ⇒ the current encoder leaves nonlinear signal on the table."
        )
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Nonlinear-retrievability diagnostic for PC scores.",
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
    parser.add_argument(
        "--regressor", type=str, default="hgb",
        choices=["hgb", "rf"],
        help=("Nonlinear regressor: 'hgb' = HistGradientBoosting (fast, "
              "OpenMP-threaded — switch to 'rf' if HGB segfaults on macOS)."),
    )
    parser.add_argument(
        "--max-iter", type=int, default=400,
        help="HGB max boosting iterations (with early stopping).",
    )
    parser.add_argument(
        "--max-leaf-nodes", type=int, default=63,
        help="HGB max leaves per tree.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="HGB learning rate.",
    )
    parser.add_argument(
        "--rf-n-estimators", type=int, default=80,
        help=("RandomForest: number of trees.  80 is enough for a ceiling "
              "estimate; bump higher only if the curve is still moving."),
    )
    parser.add_argument(
        "--rf-max-depth", type=int, default=12,
        help=("RandomForest: max depth per tree.  Capping at 12 keeps trees "
              "small enough to finish in minutes on 138k×640."),
    )
    parser.add_argument(
        "--rf-min-samples-leaf", type=int, default=50,
        help=("RandomForest: minimum samples per leaf.  Larger values shrink "
              "trees and speed up fitting with little accuracy loss for a "
              "ceiling estimate."),
    )
    parser.add_argument(
        "--rf-n-jobs", type=int, default=4,
        help=("RandomForest: parallel jobs.  Bump to your CPU count if you "
              "are not also running training simultaneously."),
    )
    parser.add_argument(
        "--ridge-alpha", type=float, default=10.0,
        help=("Ridge regularization strength (sklearn convention).  Default 10 "
              "is high enough to suppress ill-conditioning on 640 standardized "
              "features."),
    )
    parser.add_argument(
        "--subsample-train", type=int, default=0,
        help="Optional: subsample N training rows for speed (0 = use all).",
    )
    parser.add_argument(
        "--nn-r2", type=float, nargs="+", default=None,
        help=("Optional: current-network per-PC profile-val R² values, "
              "in PC order. E.g. --nn-r2 0.85 0.05 0.03 0.01 0.0 0.0 0.0"),
    )
    parser.add_argument(
        "--save-summary-json", type=str, default="",
        help="Optional path to save the per-PC summary as JSON.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ───── Load basis & resolve metadata ─────
    basis = load_pca_basis(args.pca_basis)
    h5_path = Path(str(basis["h5_path"]))
    if not h5_path.exists():
        h5_path = _REPO_ROOT / "training_data" / h5_path.name
    print(f"HDF5: {h5_path}")
    print(f"PCA basis fit on: seed={int(basis['seed'])}, "
          f"n_val_profiles={int(basis['n_val_profiles'])}, "
          f"n_test_profiles={int(basis['n_test_profiles'])}")

    K = min(int(args.max_k), int(basis["components"].shape[0]))

    # ───── Load reflectance + geometry + profiles from HDF5 ─────
    refl_key = f"reflectances_{args.instrument}"
    with h5py.File(h5_path, "r") as f:
        refl = f[refl_key][:].astype(np.float32)
        sza = f["sza"][:].astype(np.float32)
        vza = f["vza"][:].astype(np.float32)
        saz = f["saz"][:].astype(np.float32)
        vaz = f["vaz"][:].astype(np.float32)
        profiles_um = f["profiles"][:].astype(np.float32)
    print(f"Loaded {refl.shape[0]:,} samples × {refl.shape[1]} channels"
          f" + 4 geometry inputs ({args.instrument})")

    profiles_norm = (profiles_um - RE_MIN) / (RE_MAX - RE_MIN)

    # ───── Profile-aware split (matches PCA fit) ─────
    train_idx, val_idx, _ = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    print(f"Train: {len(train_idx):,} samples | Val: {len(val_idx):,} samples")

    # ───── True PC scores (float64 — same fix as evaluate_pca_scores) ─────
    pca_mean = basis["mean"].astype(np.float64)
    pca_comp = basis["components"][:K].astype(np.float64)

    centered_train = profiles_norm[train_idx].astype(np.float64) - pca_mean
    centered_val   = profiles_norm[val_idx].astype(np.float64)   - pca_mean
    scores_train = (centered_train @ pca_comp.T).astype(np.float32)
    scores_val   = (centered_val   @ pca_comp.T).astype(np.float32)

    # ───── Build feature matrices (refl + geometry, 640 columns) ─────
    refl_train = refl[train_idx]
    refl_val   = refl[val_idx]
    geom_train = np.stack([sza[train_idx], vza[train_idx],
                           saz[train_idx], vaz[train_idx]], axis=1)
    geom_val   = np.stack([sza[val_idx],   vza[val_idx],
                           saz[val_idx],   vaz[val_idx]], axis=1)
    full_train = np.concatenate([refl_train, geom_train], axis=1).astype(np.float32)
    full_val   = np.concatenate([refl_val,   geom_val],   axis=1).astype(np.float32)

    # ───── Random within-train holdout (10% of train, by sample) ─────
    rng = np.random.default_rng(0)
    n_tr = full_train.shape[0]
    perm = rng.permutation(n_tr)
    n_inner_val = max(2000, n_tr // 10)
    inner_val_idx   = perm[:n_inner_val]
    inner_train_idx = perm[n_inner_val:]

    # Optional subsample for speed (applied to inner-train only).
    if args.subsample_train > 0 and args.subsample_train < len(inner_train_idx):
        inner_train_idx = rng.choice(inner_train_idx,
                                     size=args.subsample_train, replace=False)
        print(f"Subsampled inner-train to {len(inner_train_idx):,} rows for speed.")

    X_inner_tr = full_train[inner_train_idx]
    X_inner_va = full_train[inner_val_idx]
    X_full_tr  = full_train       # for the profile-val fit, train on ALL train rows
    X_pval     = full_val

    # ───── Standardize features for the ridge fits ─────
    # Ridge on raw reflectances (which span a few orders of magnitude across
    # channels) plus 4 geometry features in degrees produces a heavily
    # ill-conditioned design matrix.  Per-feature standardization on the
    # respective training set fixes the conditioning.  HGB/RF are tree-based
    # and scale-invariant, so they use the unstandardized inputs directly.
    scaler_inner = StandardScaler().fit(X_inner_tr)
    Xs_inner_tr = scaler_inner.transform(X_inner_tr)
    Xs_inner_va = scaler_inner.transform(X_inner_va)
    scaler_full  = StandardScaler().fit(X_full_tr)
    Xs_full_tr  = scaler_full.transform(X_full_tr)
    Xs_pval     = scaler_full.transform(X_pval)

    def make_nonlinear_regressor():
        """Construct a fresh nonlinear regressor per fit (no warm state)."""
        if args.regressor == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_leaf_nodes=args.max_leaf_nodes,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=0,
            )
        # RandomForest fallback (joblib process-based, no OpenMP segfault risk).
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.rf_n_jobs,
            random_state=0,
            verbose=1,
        )

    def n_iter_of(reg) -> int:
        # HGB exposes n_iter_ (number of boosting rounds actually used).
        # RF doesn't have a useful equivalent — report n_estimators.
        if hasattr(reg, "n_iter_"):
            return int(reg.n_iter_)
        if hasattr(reg, "n_estimators"):
            return int(reg.n_estimators)
        return -1

    # ───── Per-PC fit (linear + nonlinear) ─────
    per_pc = []
    print(f"\nFitting per-PC linear + nonlinear ceilings (K = {K})…")
    if args.regressor == "hgb":
        print(f"Nonlinear regressor: HGB  (max_iter={args.max_iter}, "
              f"max_leaf_nodes={args.max_leaf_nodes}, "
              f"learning_rate={args.learning_rate})")
    else:
        print(f"Nonlinear regressor: RandomForest  "
              f"(n_estimators={args.rf_n_estimators}, "
              f"max_depth={args.rf_max_depth}, n_jobs={args.rf_n_jobs})")

    for k in range(K):
        y_inner_tr = scores_train[inner_train_idx, k]
        y_inner_va = scores_train[inner_val_idx,   k]
        y_full_tr  = scores_train[:, k]
        y_pval     = scores_val[:, k]

        print(f"  ── PC{k+1} ──", flush=True)
        t0 = time.time()

        # Linear (ridge) — random holdout, on STANDARDIZED features.
        ridge1 = Ridge(alpha=args.ridge_alpha, solver="lsqr", random_state=0)
        ridge1.fit(Xs_inner_tr, y_inner_tr)
        ridge_rand_r2 = r2_score(y_inner_va, ridge1.predict(Xs_inner_va))

        # Linear (ridge) — profile-val (fit on all train rows).
        ridge2 = Ridge(alpha=args.ridge_alpha, solver="lsqr", random_state=0)
        ridge2.fit(Xs_full_tr, y_full_tr)
        ridge_pval_r2 = r2_score(y_pval, ridge2.predict(Xs_pval))

        # Nonlinear — random holdout (raw features; trees are scale-invariant).
        nl1 = make_nonlinear_regressor()
        nl1.fit(X_inner_tr, y_inner_tr)
        hgb_rand_r2 = r2_score(y_inner_va, nl1.predict(X_inner_va))

        # Nonlinear — profile-val (fit on all train rows).
        nl2 = make_nonlinear_regressor()
        nl2.fit(X_full_tr, y_full_tr)
        pval_pred = nl2.predict(X_pval)
        hgb_pval_r2 = r2_score(y_pval, pval_pred)

        dt = time.time() - t0

        s = {
            "pc":            k + 1,
            "ridge_rand_r2": float(ridge_rand_r2),
            "ridge_pval_r2": float(ridge_pval_r2),
            "hgb_rand_r2":   float(hgb_rand_r2),
            "hgb_pval_r2":   float(hgb_pval_r2),
            "hgb_n_iter":    n_iter_of(nl2),
            "score_std":     float(scores_train[:, k].std()),
            "pval_true":     y_pval.copy(),
            "pval_pred":     pval_pred.copy(),
        }
        per_pc.append(s)
        nl_label = "HGB" if args.regressor == "hgb" else "RF "
        print(
            f"  PC{k+1}: "
            f"ridge rand R²={ridge_rand_r2:+.3f} | "
            f"{nl_label}   rand R²={hgb_rand_r2:+.3f} | "
            f"ridge p-val R²={ridge_pval_r2:+.3f} | "
            f"{nl_label}   p-val R²={hgb_pval_r2:+.3f} | "
            f"trees={s['hgb_n_iter']:>3d} | "
            f"{dt:5.1f}s"
        )

    # ───── Plots ─────
    plot_linear_vs_nonlinear(
        per_pc, FIG_DIR / "01_linear_vs_nonlinear_ceiling.png",
        nn_r2=args.nn_r2,
    )
    plot_pred_vs_true_scatter(
        per_pc, FIG_DIR / "02_hgb_pred_vs_true_scatter.png"
    )
    plot_headroom(
        per_pc, FIG_DIR / "03_headroom_per_pc.png",
        nn_r2=args.nn_r2,
    )

    # ───── Summary table ─────
    print("\n" + "=" * 96)
    print(f"Linear vs nonlinear retrievability  (K = {K})")
    print("=" * 96)
    print(f"{'PC':>3} | {'ridge rand':>10} | {'HGB rand':>10} | "
          f"{'Δ (rand)':>9} | {'ridge pval':>10} | {'HGB pval':>9} | "
          f"{'trees':>5}")
    print("-" * 96)
    for s in per_pc:
        delta = s["hgb_rand_r2"] - s["ridge_rand_r2"]
        print(f"{s['pc']:>3} | "
              f"{s['ridge_rand_r2']:>+10.3f} | "
              f"{s['hgb_rand_r2']:>+10.3f} | "
              f"{delta:>+9.3f} | "
              f"{s['ridge_pval_r2']:>+10.3f} | "
              f"{s['hgb_pval_r2']:>+9.3f} | "
              f"{s['hgb_n_iter']:>5d}")
    print("-" * 96)

    print("\nReading guide:")
    print("  • 'ridge rand'  linear ceiling on a within-train random holdout.")
    print("  • 'HGB rand'    nonlinear ceiling on the same split.")
    print("  • 'Δ (rand)'    NONLINEAR HEADROOM. >0 ⇒ a nonlinear function")
    print("                  of (spectrum, geometry) unlocks signal that")
    print("                  ridge misses; an MLP encoder can in principle")
    print("                  reach this ceiling.  ≈0 ⇒ no encoder will help.")
    print("  • 'p-val'       generalization to NEW PROFILES.  Sensitive to")
    print("                  population shift (only ~14 unique val profiles).")
    print(f"\nFigures written to {FIG_DIR.relative_to(_REPO_ROOT)}")

    if args.save_summary_json:
        out = []
        for s in per_pc:
            d = {k: v for k, v in s.items()
                 if k not in ("pval_true", "pval_pred")}
            out.append(d)
        Path(args.save_summary_json).write_text(json.dumps(out, indent=2))
        print(f"Summary JSON written to {args.save_summary_json}")


if __name__ == "__main__":
    main()
