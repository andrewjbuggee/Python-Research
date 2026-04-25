"""
diagnose_pc_retrievability_kfold.py
-----------------------------------
Diagnostic C: profile-aware K-fold CV of the per-PC nonlinear retrievability
ceiling.

Why this script exists
======================
Diagnostic B (HGB on a within-train RANDOM holdout) showed suspiciously high
R² (≈0.85) for ALL PCs 3–7, but the same model gave NEGATIVE profile-val R²
on the original 14-profile held-out split.  That pattern is the signature of
profile-fingerprint MEMORIZATION: trees can identify which of the 262
training profiles a sample comes from (using ~530 samples per profile under
varied geometry/wavelengths) and emit the profile-conditional mean PC score.
The random-holdout boost was leakage, not real nonlinear signal.

This script eliminates the leakage by holding out *profiles*, not samples,
and averages across 5 different held-out folds to remove "we got unlucky with
which profiles got held out" as an explanation.

What it does
============
On the original 262-profile train pool only (the 14 val + 14 test profiles
are kept fully untouched for paper-final evaluation):

  1. Compute true PC scores via float64 PCA projection.
  2. Use sklearn GroupKFold(n_splits=5) over the 262 unique profiles to make
     5 disjoint profile folds (~52 profiles per fold).  Every profile appears
     in exactly one held-out fold, never in training when held out.
  3. For each fold and each PC k=1..K:
       - Fit Ridge (linear baseline) and HistGradientBoosting (nonlinear
         ceiling) on the OTHER 4 profile folds (~210 profiles).
       - Evaluate R² on the held-out fold (~52 profiles, fully unseen).
  4. Aggregate: report per-PC mean ± std R² across the 5 folds.

Outputs (figures at 500 DPI inside ./figures/pc_retrievability_kfold/):

  01_kfold_per_pc_r2.png         Mean ± std R² per PC, ridge vs HGB.
                                 ← THE HEADLINE PLOT.
  02_kfold_breakdown.png         Per-fold R² scatter per PC (5 dots per PC).
  03_kfold_vs_random_holdout.png Comparison: K-fold R² vs the deceptive
                                 random-holdout R² from diagnostic B.

Decision rule
=============
If a PC has mean K-fold HGB R² > 0.2 with low std → genuinely retrievable
                                                    nonlinearly.
If mean ≈ 0 or strongly negative across all 5 folds → not retrievable from
                                                      this input, switch the
                                                      basis (PLS/CCA), don't
                                                      tweak the encoder.
If mean is positive but std is large (some folds positive, others negative)
→ retrievability is profile-population-specific; needs more profiles, not
  a different model.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Set thread-limit env vars BEFORE importing numpy / sklearn (macOS OpenMP
# segfault workaround — see diagnose_pc_retrievability_nonlinear.py).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data import (RE_MIN, RE_MAX,
                  create_profile_aware_splits,
                  compute_profile_ids)  # noqa: E402

from models_pca import load_pca_basis  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


FIG_DPI = 500
FIG_DIR = Path(__file__).resolve().parent / "figures" / "pc_retrievability_kfold"


# ───────────────────────────────────────────────────────────────────────────────
# Math
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
def plot_kfold_per_pc(per_pc: list[dict], fig_path: Path,
                      nn_r2: list[float] | None = None,
                      random_holdout_hgb: list[float] | None = None):
    """
    Headline plot.  Per PC, mean ± std R² across 5 profile-held-out folds for
    ridge and HGB.  Optionally overlay the current-network profile-val R²
    and the (deceptive) random-holdout HGB R² from diagnostic B for
    contrast.
    """
    K = len(per_pc)
    x = np.arange(1, K + 1)

    ridge_mean = np.array([s["ridge_r2_mean"] for s in per_pc])
    ridge_std  = np.array([s["ridge_r2_std"]  for s in per_pc])
    hgb_mean   = np.array([s["hgb_r2_mean"]   for s in per_pc])
    hgb_std    = np.array([s["hgb_r2_std"]    for s in per_pc])

    n_bars = 2
    if nn_r2 is not None:
        n_bars += 1
    if random_holdout_hgb is not None:
        n_bars += 1
    w = 0.85 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * w
    bar_idx = 0

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    ax.bar(x + offsets[bar_idx], np.clip(ridge_mean, -0.1, None), w,
           yerr=ridge_std, color="#4C72B0", edgecolor="black",
           ecolor="black", capsize=3,
           label="Ridge      (5-fold profile CV)  ← linear ceiling")
    bar_idx += 1
    ax.bar(x + offsets[bar_idx], np.clip(hgb_mean, -0.1, None), w,
           yerr=hgb_std, color="#55A868", edgecolor="black",
           ecolor="black", capsize=3,
           label="HGB        (5-fold profile CV)  ← NONLINEAR ceiling")
    bar_idx += 1
    if random_holdout_hgb is not None:
        rh = np.array(random_holdout_hgb[:K]
                      + [np.nan] * max(0, K - len(random_holdout_hgb)))
        ax.bar(x + offsets[bar_idx], np.clip(rh, -0.1, None), w,
               color="#9CCBA0", edgecolor="black", hatch="//",
               label="HGB random-holdout (diag. B) — leakage-inflated")
        bar_idx += 1
    if nn_r2 is not None:
        nn = np.array(nn_r2[:K] + [np.nan] * max(0, K - len(nn_r2)))
        ax.bar(x + offsets[bar_idx], np.clip(nn, -0.1, None), w,
               color="#C44E52", edgecolor="black",
               label="Current NN (profile-val)")
        bar_idx += 1

    # Annotate strongly-negative bars (clipped at -0.1 visually).
    for k, s in enumerate(per_pc):
        for vname, off, color in (
            ("ridge_r2_mean", offsets[0], "#4C72B0"),
            ("hgb_r2_mean",   offsets[1], "#55A868"),
        ):
            v = s[vname]
            if v < -0.1:
                ax.annotate(f"{v:.2f}", xy=(x[k] + off, -0.1),
                            xytext=(x[k] + off, -0.05),
                            ha="center", fontsize=7, color=color,
                            arrowprops=dict(arrowstyle="-",
                                            color=color, lw=0.5))

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axhline(0.05, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.4, 0.075, "noise floor (R²≈0.05)", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("R²  (held-out profiles)")
    ax.set_ylim(-0.15, 1.05)
    ax.set_title(
        "Profile-aware 5-fold CV per-PC retrievability\n"
        "Error bars = ±1 std across 5 folds; bars are means."
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kfold_breakdown(per_pc: list[dict], fig_path: Path):
    """
    Per-fold scatter of R² per PC (5 dots per PC, one per fold).  Reveals
    whether per-PC retrievability is consistent across folds or driven by
    one or two unusual folds.
    """
    K = len(per_pc)
    x = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))

    for s in per_pc:
        ridge_folds = s["ridge_r2_per_fold"]
        hgb_folds   = s["hgb_r2_per_fold"]
        xi = s["pc"]
        ax.scatter(np.full_like(ridge_folds, xi - 0.12), ridge_folds,
                   s=40, marker="o", color="#4C72B0",
                   edgecolor="black", linewidth=0.6, zorder=3,
                   label="Ridge (per fold)" if xi == 1 else None)
        ax.scatter(np.full_like(hgb_folds, xi + 0.12), hgb_folds,
                   s=40, marker="s", color="#55A868",
                   edgecolor="black", linewidth=0.6, zorder=3,
                   label="HGB (per fold)" if xi == 1 else None)
        # Mean lines.
        ax.hlines(s["ridge_r2_mean"], xi - 0.22, xi - 0.02,
                  colors="#4C72B0", linewidth=2.5)
        ax.hlines(s["hgb_r2_mean"],   xi + 0.02, xi + 0.22,
                  colors="#55A868", linewidth=2.5)

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axhline(0.05, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("R²  (held-out profile fold)")
    ax.set_title(
        "Per-fold R² breakdown — 5 dots per PC, one per held-out fold.\n"
        "Tight cluster ⇒ retrievability is profile-population-stable."
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kfold_vs_random_holdout(per_pc: list[dict], fig_path: Path,
                                 random_holdout_hgb: list[float]):
    """
    Side-by-side: K-fold profile-CV HGB R² (honest) vs random-holdout HGB R²
    (leakage-inflated).  The gap quantifies the leakage; large gaps for
    PCs whose K-fold value is ≈0 are diagnostic of profile-fingerprint
    memorization.
    """
    K = len(per_pc)
    x = np.arange(1, K + 1)
    hgb_kfold = np.array([s["hgb_r2_mean"] for s in per_pc])
    hgb_rand  = np.array(random_holdout_hgb[:K])

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    w = 0.38
    ax.bar(x - w/2, np.clip(hgb_kfold, -0.1, None), w,
           color="#55A868", edgecolor="black",
           label="HGB R² — 5-fold profile CV  (honest)")
    ax.bar(x + w/2, np.clip(hgb_rand, -0.1, None), w,
           color="#9CCBA0", edgecolor="black", hatch="//",
           label="HGB R² — random within-train holdout  (leakage-inflated)")

    for xi, (kf, rh) in enumerate(zip(hgb_kfold, hgb_rand), start=1):
        gap = rh - kf
        ax.annotate(
            f"Δ={gap:+.2f}",
            xy=(xi, max(kf, rh) + 0.04),
            ha="center", fontsize=8, color="black",
        )

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("R²")
    ax.set_ylim(-0.15, 1.1)
    ax.set_title(
        "Random-holdout vs profile-CV HGB R²\n"
        "Δ measures profile-fingerprint LEAKAGE in the random-holdout estimate."
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Profile-aware 5-fold CV diagnostic for PC retrievability.",
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
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of profile-aware K-fold splits.",
    )
    parser.add_argument(
        "--regressor", type=str, default="hgb", choices=["hgb", "rf"],
    )
    # HGB parameters — slightly leaner than diagnostic B because we'll fit
    # n_folds × K models = 5 × 7 = 35 models.  At ~135s each, default would
    # be ~80 min; trimmed defaults bring it to ~30–40 min.
    parser.add_argument("--max-iter",       type=int,   default=300)
    parser.add_argument("--max-leaf-nodes", type=int,   default=31)
    parser.add_argument("--learning-rate",  type=float, default=0.07)
    parser.add_argument("--rf-n-estimators",     type=int, default=80)
    parser.add_argument("--rf-max-depth",        type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=50)
    parser.add_argument("--rf-n-jobs",           type=int, default=4)
    parser.add_argument(
        "--ridge-alpha", type=float, default=10.0,
    )
    parser.add_argument(
        "--nn-r2", type=float, nargs="+", default=None,
        help="Optional current-NN per-PC profile-val R² for overlay.",
    )
    parser.add_argument(
        "--random-holdout-hgb", type=float, nargs="+", default=None,
        help=("Optional HGB random-holdout R² values from diagnostic B "
              "(in PC order) for the leakage-comparison plot."),
    )
    parser.add_argument(
        "--save-summary-json", type=str, default="",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ───── Load basis & resolve metadata ─────
    basis = load_pca_basis(args.pca_basis)
    h5_path = Path(str(basis["h5_path"]))
    if not h5_path.exists():
        h5_path = _REPO_ROOT / "training_data" / h5_path.name
    print(f"HDF5: {h5_path}")
    print(f"PCA basis: seed={int(basis['seed'])}, "
          f"n_val_profiles={int(basis['n_val_profiles'])}, "
          f"n_test_profiles={int(basis['n_test_profiles'])}")

    K = min(int(args.max_k), int(basis["components"].shape[0]))

    # ───── Load all data ─────
    refl_key = f"reflectances_{args.instrument}"
    with h5py.File(h5_path, "r") as f:
        refl = f[refl_key][:].astype(np.float32)
        sza = f["sza"][:].astype(np.float32)
        vza = f["vza"][:].astype(np.float32)
        saz = f["saz"][:].astype(np.float32)
        vaz = f["vaz"][:].astype(np.float32)
        profiles_um = f["profiles"][:].astype(np.float32)
    profiles_norm = (profiles_um - RE_MIN) / (RE_MAX - RE_MIN)
    print(f"Loaded {refl.shape[0]:,} samples × {refl.shape[1]} channels"
          f" + 4 geometry inputs ({args.instrument})")

    # ───── Get the original train pool (262 profiles), keep val + test out ─────
    train_idx, _, _ = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=int(basis["n_val_profiles"]),
        n_test_profiles=int(basis["n_test_profiles"]),
        seed=int(basis["seed"]),
    )
    profile_ids_all = compute_profile_ids(str(h5_path))
    train_pool_pids = profile_ids_all[train_idx]
    n_unique_train_profiles = len(np.unique(train_pool_pids))
    print(f"Train pool: {len(train_idx):,} samples covering "
          f"{n_unique_train_profiles} unique profiles "
          f"(val + test profiles excluded)")

    # ───── True PC scores (float64) ─────
    pca_mean = basis["mean"].astype(np.float64)
    pca_comp = basis["components"][:K].astype(np.float64)
    centered = profiles_norm[train_idx].astype(np.float64) - pca_mean
    scores = (centered @ pca_comp.T).astype(np.float32)   # (N_pool, K)

    # ───── Feature matrix (refl + geometry) ─────
    refl_pool = refl[train_idx]
    geom_pool = np.stack([sza[train_idx], vza[train_idx],
                          saz[train_idx], vaz[train_idx]], axis=1)
    X_pool = np.concatenate([refl_pool, geom_pool], axis=1).astype(np.float32)

    # ───── 5-fold profile-aware CV ─────
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_indices = list(gkf.split(X_pool, groups=train_pool_pids))
    print(f"\n{args.n_folds}-fold CV by PROFILE GROUP:")
    for f_id, (fit_idx, eval_idx) in enumerate(fold_indices):
        n_fit_prof = len(np.unique(train_pool_pids[fit_idx]))
        n_eval_prof = len(np.unique(train_pool_pids[eval_idx]))
        print(f"  fold {f_id}: fit on {len(fit_idx):>6,} samples "
              f"({n_fit_prof:>3} profiles), "
              f"eval on {len(eval_idx):>5,} samples "
              f"({n_eval_prof:>3} profiles)")

    # Pre-compute per-fold standardized features for ridge.  Tree models use
    # raw inputs (scale-invariant).
    def make_nl():
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
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.rf_n_jobs,
            random_state=0,
            verbose=0,
        )

    print(f"\nFitting Ridge + {args.regressor.upper()} per (fold, PC) — "
          f"{args.n_folds} folds × {K} PCs = {args.n_folds * K} models per "
          f"regressor type…")

    per_pc = []
    for k in range(K):
        ridge_folds = []
        hgb_folds   = []
        n_iter_folds = []
        print(f"\n  ── PC{k+1} ──", flush=True)
        for f_id, (fit_idx, eval_idx) in enumerate(fold_indices):
            t0 = time.time()
            X_fit_raw  = X_pool[fit_idx]
            X_eval_raw = X_pool[eval_idx]
            y_fit  = scores[fit_idx,  k]
            y_eval = scores[eval_idx, k]

            # Ridge: standardize features fit-only.
            scaler = StandardScaler().fit(X_fit_raw)
            X_fit_s  = scaler.transform(X_fit_raw)
            X_eval_s = scaler.transform(X_eval_raw)
            ridge = Ridge(alpha=args.ridge_alpha, solver="lsqr",
                          random_state=0)
            ridge.fit(X_fit_s, y_fit)
            r_r2 = r2_score(y_eval, ridge.predict(X_eval_s))

            # Nonlinear: raw features.
            nl = make_nl()
            nl.fit(X_fit_raw, y_fit)
            n_r2 = r2_score(y_eval, nl.predict(X_eval_raw))
            n_iters = int(getattr(nl, "n_iter_",
                                  getattr(nl, "n_estimators", -1)))

            ridge_folds.append(r_r2)
            hgb_folds.append(n_r2)
            n_iter_folds.append(n_iters)
            dt = time.time() - t0
            print(f"    fold {f_id}: ridge={r_r2:+.3f} | "
                  f"{args.regressor}={n_r2:+.3f} | "
                  f"trees={n_iters:>3d} | {dt:5.1f}s", flush=True)

        ridge_folds = np.array(ridge_folds)
        hgb_folds   = np.array(hgb_folds)
        s = {
            "pc": k + 1,
            "ridge_r2_per_fold": ridge_folds.tolist(),
            "ridge_r2_mean":     float(ridge_folds.mean()),
            "ridge_r2_std":      float(ridge_folds.std()),
            "hgb_r2_per_fold":   hgb_folds.tolist(),
            "hgb_r2_mean":       float(hgb_folds.mean()),
            "hgb_r2_std":        float(hgb_folds.std()),
            "n_iter_per_fold":   n_iter_folds,
        }
        per_pc.append(s)
        print(f"    PC{k+1} summary: "
              f"ridge {s['ridge_r2_mean']:+.3f} ± {s['ridge_r2_std']:.3f} | "
              f"{args.regressor} {s['hgb_r2_mean']:+.3f} ± "
              f"{s['hgb_r2_std']:.3f}")

    # ───── Plots ─────
    plot_kfold_per_pc(
        per_pc, FIG_DIR / "01_kfold_per_pc_r2.png",
        nn_r2=args.nn_r2,
        random_holdout_hgb=args.random_holdout_hgb,
    )
    plot_kfold_breakdown(
        per_pc, FIG_DIR / "02_kfold_breakdown.png"
    )
    if args.random_holdout_hgb is not None:
        plot_kfold_vs_random_holdout(
            per_pc, FIG_DIR / "03_kfold_vs_random_holdout.png",
            random_holdout_hgb=args.random_holdout_hgb,
        )

    # ───── Summary table ─────
    print("\n" + "=" * 88)
    print(f"Profile-aware {args.n_folds}-fold CV summary  (K = {K})")
    print("=" * 88)
    print(f"{'PC':>3} | {'ridge mean ± std':>17} | "
          f"{args.regressor + ' mean ± std':>17} | "
          f"{'Δ (nl − lin)':>13}")
    print("-" * 88)
    for s in per_pc:
        delta = s["hgb_r2_mean"] - s["ridge_r2_mean"]
        print(f"{s['pc']:>3} | "
              f"{s['ridge_r2_mean']:>+8.3f} ± {s['ridge_r2_std']:>5.3f}   | "
              f"{s['hgb_r2_mean']:>+8.3f} ± {s['hgb_r2_std']:>5.3f}   | "
              f"{delta:>+13.3f}")
    print("-" * 88)

    print("\nReading guide:")
    print("  • Mean R² ≥ 0.2 with low std       → genuinely retrievable.")
    print("  • Mean ≈ 0 or negative across folds → input does not encode this")
    print("                                        PC; switch the basis.")
    print("  • Mean positive but std large       → profile-population")
    print("                                        sensitivity, not absent.")
    print(f"\nFigures written to {FIG_DIR.relative_to(_REPO_ROOT)}")

    if args.save_summary_json:
        Path(args.save_summary_json).write_text(json.dumps(per_pc, indent=2))
        print(f"Summary JSON written to {args.save_summary_json}")


if __name__ == "__main__":
    main()
