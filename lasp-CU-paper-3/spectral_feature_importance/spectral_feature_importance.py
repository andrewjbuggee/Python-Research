"""
spectral_feature_importance.py — per-profile spectral permutation importance
for the paper-3 profile + τ_c network (DropletProfileNetwork)
==========================================================================

What this script answers
------------------------
"If the trained network could no longer use the reflectance at wavelength λ,
how much worse would each retrieval in the test set get?"  It is answered for
BOTH outputs of the network:

    (a) the droplet-size profile:  Δ per-profile RMSE  [μm]
    (b) the cloud optical thickness: Δ |τ_c error|      [optical-thickness units]

and summarized across the held-out test set by the MEDIAN and the
INTERQUARTILE RANGE (25th–75th percentile) of the per-profile changes.

The technique and why it was chosen (with sources)
--------------------------------------------------
1. Permutation feature importance (PFI).  Train once; on held-out data,
   destroy the information in one input by randomly permuting that input's
   column across the test samples, re-run the frozen model, and measure the
   increase in error.  Permuting (rather than overwriting with a constant)
   keeps the input's marginal distribution, so the perturbed value is always
   one the network has seen for that channel.
     - Breiman, L. (2001), Random forests, Machine Learning, 45, 5–32,
       doi:10.1023/A:1010933404324 — origin of permutation importance.
     - Fisher, A., C. Rudin and F. Dominici (2019), All models are wrong, but
       many are useful: learning a variable's importance by studying an entire
       class of prediction models simultaneously, JMLR, 20(177), 1–81,
       https://jmlr.org/papers/v20/18-760.html — formalizes PFI as "model
       reliance" of ONE fitted model (it measures what this network relies
       on, not what the physics makes informative).
     - McGovern, A. et al. (2019), Making the black box more transparent:
       understanding the physical implications of machine learning, BAMS,
       100, 2175–2200, doi:10.1175/BAMS-D-18-0195.1 — permutation importance
       as a standard interpretation tool in atmospheric science.

2. No retraining.  Removing a channel and RETRAINING (drop-column / LOCO /
   "remove and retrain", Hooker, S. et al. 2019, A benchmark for
   interpretability methods in deep neural networks, NeurIPS 32,
   https://arxiv.org/abs/1806.10758) asks a different question: "is this
   channel's information necessary GIVEN all the others?"  With 636
   hyperspectral channels whose neighbors are nearly collinear, a
   single-channel retrain will show ~0 almost everywhere because neighbors
   substitute (Hooker et al. 2019 note that redundancy masks retrain-based
   importance).  It would also cost 636 trainings.  Retraining is the right
   tool for BAND-level questions, and it was used that way in
   ../uv_wl_ablation and ../wv_core_ablation.

3. Evaluate on the held-out test set with the model in eval mode (dropout off),
   repeat each permutation K times with a recorded seed, and report the
   spread — the "interpreting a model that generalizes poorly", "ignoring
   uncertainty" and "ignoring feature dependence" pitfalls in
   Molnar, C. et al. (2022), General pitfalls of model-agnostic interpretation
   methods for machine learning models, in xxAI – Beyond Explainable AI,
   LNCS 13200, 39–68, doi:10.1007/978-3-031-04083-2_4.

4. Per-profile (local) importance.  Instead of one test-set-aggregate ΔRMSE
   (what the spring-2026 Fig. 6 shows), keep the error change of EVERY test
   profile i, averaged over the K permutations:

       Δ_i(λ) = (1/K) Σ_k [ e_i(X with column λ permuted, draw k) − e_i(X) ]

   This is the per-observation decomposition of PFI that Casalicchio et al.
   call "individual conditional importance" (ICI), whose average over
   observations is the global PFI:
     - Casalicchio, G., C. Molnar and B. Bischl (2019), Visualizing the feature
       importance for black box models, ECML PKDD 2018, LNCS 11051, 655–670,
       https://arxiv.org/abs/1804.06620.
   The distribution of Δ_i across profiles is then summarized by its median
   and IQR (this choice of summary is the user's; robust because the
   per-profile RMSE distribution is right-skewed).

Known limitation — correlated channels (read before interpreting)
----------------------------------------------------------------
Permuting ONE channel while its neighbors keep their true values produces
spectra that never occur in nature (a single wavelength "jumps" to another
cloud's value).  The network then extrapolates off the data manifold, which
can inflate or deflate the apparent importance of strongly correlated inputs:
     - Strobl, C. et al. (2008), Conditional variable importance for random
       forests, BMC Bioinformatics, 9, 307, doi:10.1186/1471-2105-9-307.
     - Hooker, G., L. Mentch and S. Zhou (2021), Unrestricted permutation
       forces extrapolation: variable importance requires at least one more
       model, or there is no free variable importance, Stat. Comput., 31, 82,
       doi:10.1007/s11222-021-10057-z.
Mitigation offered here: --group-width-nm W permutes all channels within
±W/2 of λ TOGETHER with the same permutation (grouped PFI; Au, Q. et al.
(2022), Grouped feature importance and combined features effect plot,
Data Min. Knowl. Disc., 36, 1401–1450, doi:10.1007/s10618-022-00840-5).
That keeps the spectrum locally self-consistent.  If the single-channel and
grouped curves show the same bands, the single-channel result is not an
off-manifold artifact.  (Applying grouped PFI as a sliding window centered on
every channel is my choice, not a procedure from Au et al.)

Perturbation alternatives (--perturbation)
-------------------------------------------
    permute  (default)  — PFI as above; K seeded repeats.
    mean               — replace the channel with its test-set mean
                         (deterministic; what compute_spectral_feature_importance.py
                         did in April 2026).
    min                — set the NORMALIZED input to 0.  Inputs are min–max
                         scaled per channel over the HDF5 file (data.py,
                         LibRadtranDataset._normalize_input), so 0 means "the
                         darkest reflectance anywhere in the file at this λ",
                         not zero reflectance.
Both constant substitutions answer "how far does the output move relative to
a chosen reference spectrum?", and the answer depends on the reference
(Mamalakis, A., E. A. Barnes and I. Ebert-Uphoff (2023), Carefully choose the
baseline: lessons learned from applying XAI attribution methods for regression
tasks in geoscience, AIES, 2, e220058, doi:10.1175/AIES-D-22-0058.1).  They
are kept for comparison only.

Metric definitions
------------------
    per-profile RMSE (μm):   e_i = sqrt( (1/L) Σ_l (r̂_e,il − r_e,il)² ),  L = 7 levels
    τ error:                 a_i = |τ̂_c,i − τ_c,i|
    importance:              Δ_i = e_i(perturbed) − e_i(baseline)   (likewise for a_i)

Δ_i can be negative for an individual profile (a permuted value can by chance
move the prediction toward the truth); the median and IQR show this directly.

Confidence interval on the median (CSV only): distribution-free order-statistic
interval from the binomial distribution of the number of observations below
the median (e.g. Conover, W. J. (1999), Practical Nonparametric Statistics,
3rd ed., Wiley, the quantile-test chapter — section number not checked).  It
describes sampling uncertainty of the median over test profiles; it does NOT
include the run-to-run variability of retraining the network, which only
repeating this analysis on several seed replicates can measure (Fisher et al.
2019 make the same point about a single model's reliance).

Outputs  → <run_dir>/spectral_feature_importance/   (override with --out-dir)
    spectral_importance_<suffix>.npz   per-profile Δ matrices + summary stats
    spectral_importance_<suffix>.csv   one row per input (636 spectral + 4 geometry)
    fig_spectral_importance_<suffix>.png / .pdf   the two-panel paper figure

Usage (laptop, from lasp-CU-paper-3/)
-------------------------------------
    # model chosen by step 13 for a training round
    python spectral_feature_importance/spectral_feature_importance.py --tag PT_2026-11-14 \\
        --h5-path training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5

    # any run directory; grouped robustness check with 10-nm windows
    python spectral_feature_importance/spectral_feature_importance.py \\
        --run-dir training_runs/PT_2026-11-14/results/run_003 --group-width-nm 10

    # re-draw the figure from the saved .npz
    python spectral_feature_importance/spectral_feature_importance.py --tag PT_2026-11-14 --plot-only

Author: Andrew J. Buggee (script drafted with Claude), 15 Nov 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # file output only; never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

HERE = Path(__file__).resolve().parent  # spectral_feature_importance/
REPO = HERE.parent  # lasp-CU-paper-3/ (data.py, models.py)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hyper_parameter_sweep"))

N_SPECTRAL = 636  # HySICS reflectance channels
N_GEOMETRY = 4  # SZA, VZA, SAZ, VAZ (appended after the spectra by data.py)
N_INPUTS = N_SPECTRAL + N_GEOMETRY
GEOM_NAMES = ["SZA", "VZA", "SAZ", "VAZ"]

# Water-vapor absorption bands shaded in the figure, identical to the spring
# Fig. 6 (plot_feature_importance_run004.py).  Approximate visual windows in μm,
# not channel definitions.
WV_BANDS_UM = [(0.91, 0.98), (1.10, 1.18), (1.34, 1.44), (1.78, 1.98)]

# Colors: same blue as the spring Fig. 6 for the profile panel; Okabe–Ito
# vermillion for τ (CVD-safe pair, and the two never share an axis).
C_PROFILE = "#1F5FC2"
C_TAU = "#D55E00"
C_BAND = "#E6C98C"


# ═════════════════════════════════════════════════════════════════════════
# 1. Loading the run, the model and the held-out test set
# ═════════════════════════════════════════════════════════════════════════
def selected_run_dir(tag: str | None, run_dir: str | None) -> Path:
    """--run-dir if given, else the run step 13 selected for --tag (as step 14)."""
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit("give --run-dir or --tag")
    sel_file = REPO / "training_runs" / tag / "selected_model.json"
    with sel_file.open() as f:
        sel = json.load(f)
    return (REPO / sel["run_dir"]).resolve()


def pick_device(arg: str | None):
    """CUDA > Apple MPS > CPU unless forced."""
    import torch

    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_test_set(run_dir: Path, h5_override: str | None, device) -> dict:
    """Rebuild the trained network and the exact held-out test split.

    Mirrors plot_run_figures_profile_tau._run_inference (step 14) so the test
    profiles here are the ones in the paper's evaluation:
      * hyper-parameters, split seed and τ normalization pair come from the
        run's summary.json (no hard-coded values);
      * the split is data.create_profile_aware_splits with the run's split seed;
      * inputs are normalized by data.LibRadtranDataset exactly as in training.
    """
    import torch
    from data import LibRadtranDataset, create_profile_aware_splits, tau_bounds_for_run
    from models import DropletProfileNetwork, RetrievalConfig, log_std_bounds_for_run
    from plot_percentile_profiles_synthetic import _resolve_h5_path

    with (run_dir / "summary.json").open() as f:
        s = json.load(f)
    hp = s["hyperparams"]
    tau_transform = s.get("tau_transform", hp.get("tau_transform", "linear"))
    split_seed = int(s.get("seed", 42))  # split seed (init seed is separate since Nov 2026)
    h5_path = _resolve_h5_path(s["data"]["h5_path"], h5_override)
    tau_bounds, tau_bounds_src = tau_bounds_for_run(s)

    # ---- the test split (profile-aware: no test cloud was seen in training)
    _, _, test_idx = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=s["n_val_profiles"],
        n_test_profiles=s["n_test_profiles"],
        seed=split_seed,
    )
    ds = LibRadtranDataset(
        str(h5_path),
        normalize=True,
        instrument=s["data"].get("instrument", "hysics"),
        tau_bounds=tau_bounds,
    )
    # Vectorized version of LibRadtranDataset.__getitem__ for the test rows:
    # reflectance min–max scaled per channel, then geometry scaled by fixed
    # ranges, in the order [spectra, SZA, VZA, SAZ, VAZ].
    refl = ds.reflectances[test_idx]  # (N, 636) raw reflectance
    refl_norm = (refl - ds.refl_min) / (ds.refl_max - ds.refl_min + 1e-8)
    geom = [
        (ds.sza[test_idx] - ds.sza_range[0]) / (ds.sza_range[1] - ds.sza_range[0]),
        (ds.vza[test_idx] - ds.vza_range[0]) / (ds.vza_range[1] - ds.vza_range[0]),
        (ds.saz[test_idx] - ds.saz_range[0]) / (ds.saz_range[1] - ds.saz_range[0]),
        (ds.vaz[test_idx] - ds.vaz_range[0]) / (ds.vaz_range[1] - ds.vaz_range[0]),
    ]
    X = np.concatenate([refl_norm, np.stack(geom, axis=1)], axis=1).astype(np.float32)  # (N, 640)
    # sanity check against the dataset's own per-sample path for one row
    x0 = ds[int(test_idx[0])][0].numpy()
    if not np.allclose(x0, X[0], atol=1e-6):
        raise RuntimeError("vectorized input normalization differs from LibRadtranDataset")

    re_true_um = ds.profiles[test_idx].astype(np.float32)  # (N, L) r_e in μm, top → base
    tau_true = ds.tau_c[test_idx].astype(np.float32)  # (N,)
    with h5py.File(h5_path, "r") as f:
        wavelengths_nm = f["wavelengths"][:].astype(np.float64).ravel()  # (636,)

    # ---- the network, built exactly as the trainer built it
    cfg = RetrievalConfig(
        n_wavelengths=N_SPECTRAL,
        n_geometry_inputs=N_GEOMETRY,
        n_levels=re_true_um.shape[1],
        hidden_dims=tuple(hp["hidden_dims"]),
        dropout=hp["dropout"],
        activation=hp.get("activation", "gelu"),
        tau_min=float(tau_bounds[0]),
        tau_max=float(tau_bounds[1]),
        # the run's own log-σ clamp (legacy −4/+2 for runs before 17 Nov 2026)
        log_std_min=log_std_bounds_for_run(hp)[0],
        log_std_max=log_std_bounds_for_run(hp)[1],
    )
    model = DropletProfileNetwork(cfg).to(device)
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()  # dropout OFF: importance of the deterministic retrieval
    # the checkpoint carries its own τ pair as buffers; it must match summary.json
    if not np.allclose((float(model.tau_min), float(model.tau_max)), tau_bounds):
        raise RuntimeError(f"checkpoint τ bounds {(float(model.tau_min), float(model.tau_max))} != {tau_bounds}")

    print(f"  run       : {run_dir}")
    print(f"  HDF5      : {h5_path}")
    print(f"  test set  : {len(test_idx)} spectra (split seed {split_seed})")
    print(f"  τ bounds  : {tau_bounds[0]:g}–{tau_bounds[1]:g} ({tau_bounds_src}), transform {tau_transform}")
    return dict(
        model=model,
        X=X,
        re_true_um=re_true_um,
        tau_true=tau_true,
        wavelengths_nm=wavelengths_nm,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tau_transform=tau_transform,
        tau_bounds=tuple(float(b) for b in tau_bounds),
        summary=s,
        h5_path=str(h5_path),
        split_seed=split_seed,
    )


# ═════════════════════════════════════════════════════════════════════════
# 2. Forward pass and per-profile error metrics
# ═════════════════════════════════════════════════════════════════════════
def predict(model, X: np.ndarray, tau_transform: str, tau_bounds, device, batch: int = 8192):
    """Network output in physical units: (r_e [N, L] μm, τ_c [N])."""
    import torch
    from sweep_train_profile_tau_synthetic import tau_target_to_physical

    re_um, tau = [], []
    with torch.no_grad():
        for i0 in range(0, X.shape[0], batch):
            out = model(torch.from_numpy(X[i0 : i0 + batch]).to(device))
            re_um.append(out["profile"].float().cpu().numpy())  # μm
            # the τ head outputs the normalized target; invert the trainer's
            # transform (linear or log10) with the run's own (τ_min, τ_max)
            tn = out["tau_normalized"].squeeze(-1).float().cpu().numpy()
            tau.append(tau_target_to_physical(tn, tau_transform, tau_bounds))
    return np.concatenate(re_um), np.concatenate(tau)


def per_profile_errors(re_pred_um, tau_pred, re_true_um, tau_true):
    """(e_i, a_i): per-profile RMSE over levels [μm] and |τ error| for every test spectrum."""
    rmse_um = np.sqrt(np.mean((re_pred_um - re_true_um) ** 2, axis=1))  # (N,)
    tau_abs_err = np.abs(tau_pred - tau_true)  # (N,)
    return rmse_um.astype(np.float64), tau_abs_err.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════
# 3. The importance loop
# ═════════════════════════════════════════════════════════════════════════
def feature_groups(wavelengths_nm: np.ndarray, group_width_nm: float) -> list[np.ndarray]:
    """Input columns perturbed together for each of the 640 importance values.

    group_width_nm = 0  → every input alone (classic single-feature PFI).
    group_width_nm = W  → spectral input c is perturbed with every channel
                          within ±W/2 of λ_c (sliding window, my choice);
                          geometry inputs stay single.
    """
    groups = []
    for c in range(N_SPECTRAL):
        if group_width_nm > 0:
            groups.append(np.flatnonzero(np.abs(wavelengths_nm - wavelengths_nm[c]) <= group_width_nm / 2))
        else:
            groups.append(np.array([c]))
    groups += [np.array([N_SPECTRAL + g]) for g in range(N_GEOMETRY)]
    return groups


def run_importance(d: dict, device, perturbation: str, n_repeats: int, seed: int, group_width_nm: float) -> dict:
    """Per-profile Δ of both error metrics for every input (or input group).

    For permutation, each repeat draws ONE permutation of the N test rows and
    applies it to every column of the group, so the group keeps its own joint
    distribution (grouped PFI, Au et al. 2022).  Δ is averaged over repeats
    per profile; the per-profile sum of squares is kept so the Monte-Carlo
    standard error of each Δ_i is available (diagnostic of whether K is enough).
    """
    model, X = d["model"], d["X"]
    n = X.shape[0]
    groups = feature_groups(d["wavelengths_nm"], group_width_nm)
    if perturbation != "permute":
        n_repeats = 1  # constant substitution is deterministic

    re_base_um, tau_base = predict(model, X, d["tau_transform"], d["tau_bounds"], device)
    e0, a0 = per_profile_errors(re_base_um, tau_base, d["re_true_um"], d["tau_true"])

    # accumulators, shape (n_inputs, N): Σ_k Δ and Σ_k Δ²
    sum_d_prof = np.zeros((N_INPUTS, n), dtype=np.float64)
    sum_d_tau = np.zeros((N_INPUTS, n), dtype=np.float64)
    sumsq_d_prof = np.zeros((N_INPUTS, n), dtype=np.float64)
    sumsq_d_tau = np.zeros((N_INPUTS, n), dtype=np.float64)
    sum_tau_shift = np.zeros((N_INPUTS, n), dtype=np.float64)  # signed τ̂ change (bias diagnostic)
    # test-set median Δ per repeat → convergence check over K
    median_per_rep_prof = np.full((n_repeats, N_INPUTS), np.nan)
    median_per_rep_tau = np.full((n_repeats, N_INPUTS), np.nan)

    rng = np.random.default_rng(seed)  # recorded in the outputs
    col_mean = X.mean(axis=0)  # used by 'mean'
    Xw = X.copy()
    t0 = time.time()
    for k in range(n_repeats):
        for c, g in enumerate(groups):
            saved = Xw[:, g].copy()
            if perturbation == "permute":
                Xw[:, g] = saved[rng.permutation(n)]
            elif perturbation == "mean":
                Xw[:, g] = col_mean[g]
            elif perturbation == "min":
                Xw[:, g] = 0.0  # normalized 0 = file-minimum reflectance
            re_um, tau = predict(model, Xw, d["tau_transform"], d["tau_bounds"], device)
            e, a = per_profile_errors(re_um, tau, d["re_true_um"], d["tau_true"])
            dp, dt = e - e0, a - a0
            sum_d_prof[c] += dp
            sum_d_tau[c] += dt
            sumsq_d_prof[c] += dp**2
            sumsq_d_tau[c] += dt**2
            sum_tau_shift[c] += tau - tau_base
            median_per_rep_prof[k, c] = np.median(dp)
            median_per_rep_tau[k, c] = np.median(dt)
            Xw[:, g] = saved  # restore before the next input
        print(f"  repeat {k + 1}/{n_repeats} done ({time.time() - t0:.0f} s)")

    K = n_repeats
    d_prof = sum_d_prof / K  # (640, N) per-profile Δ RMSE, μm
    d_tau = sum_d_tau / K  # (640, N) per-profile Δ|τ error|
    # Monte-Carlo standard error of each per-profile mean over K draws
    if K > 1:
        se_prof = np.sqrt(np.maximum(sumsq_d_prof / K - d_prof**2, 0) * K / (K - 1) / K)
        se_tau = np.sqrt(np.maximum(sumsq_d_tau / K - d_tau**2, 0) * K / (K - 1) / K)
    else:
        se_prof = se_tau = np.zeros_like(d_prof)
    return dict(
        d_prof_um=d_prof,
        d_tau=d_tau,
        mc_se_prof_um=se_prof,
        mc_se_tau=se_tau,
        tau_shift_mean=sum_tau_shift / K,
        median_per_rep_prof_um=median_per_rep_prof,
        median_per_rep_tau=median_per_rep_tau,
        base_rmse_um=e0,
        base_tau_abs_err=a0,
        re_base_um=re_base_um,
        tau_base=tau_base,
        n_repeats=K,
        group_sizes=np.array([len(g) for g in groups]),
    )


# ═════════════════════════════════════════════════════════════════════════
# 4. Summary statistics across test profiles
# ═════════════════════════════════════════════════════════════════════════
def median_ci_order_stat(n: int, conf: float = 0.95) -> tuple[int, int]:
    """0-based ranks (lo, hi) of the distribution-free CI for a median.

    The number of observations below the population median is Binomial(n, ½),
    so [x_(lo), x_(hi)] covers the median with probability ≥ conf (Conover 1999).
    """
    alpha = 1 - conf
    # 1-based lower rank l with P(B < l) ≲ α/2; the upper rank is symmetric,
    # u = n − l + 1.  As 0-based indices into the sorted sample: l − 1 and n − l.
    l_rank = max(int(binom.ppf(alpha / 2, n, 0.5)), 1)
    return l_rank - 1, n - l_rank


def summarize(delta: np.ndarray) -> dict:
    """Median, IQR, mean, fraction > 0 and median CI per input; delta is (640, N)."""
    q25, q50, q75 = np.percentile(delta, [25, 50, 75], axis=1)
    lo, hi = median_ci_order_stat(delta.shape[1])
    srt = np.sort(delta, axis=1)
    return dict(
        median=q50,
        q25=q25,
        q75=q75,
        mean=delta.mean(axis=1),
        frac_positive=(delta > 0).mean(axis=1),
        median_ci_lo=srt[:, lo],
        median_ci_hi=srt[:, hi],
    )


# ═════════════════════════════════════════════════════════════════════════
# 5. The paper figure
# ═════════════════════════════════════════════════════════════════════════
def setup_style():
    """Serif paper style shared with the other paper-3 figures (plot_run004_figures)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
        }
    )


def plot_two_panel(res: dict, out_base: Path, title_note: str = "") -> None:
    """(a) Δ per-profile RMSE and (b) Δ|τ_c error| vs wavelength: median line, IQR band."""
    setup_style()
    wl_um = res["wavelengths_nm"] / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
    panels = [
        ("prof", C_PROFILE, r"$\Delta$ per-profile RMSE  ($\mu$m)", "(a) droplet effective-radius profile"),
        ("tau", C_TAU, r"$\Delta\,|\tau_c$ error$|$", r"(b) cloud optical thickness $\tau_c$"),
    ]
    for ax, (key, color, ylabel, label) in zip(axes, panels):
        med = res[f"{key}_median"][:N_SPECTRAL]
        q25 = res[f"{key}_q25"][:N_SPECTRAL]
        q75 = res[f"{key}_q75"][:N_SPECTRAL]
        for lo, hi in WV_BANDS_UM:
            ax.axvspan(lo, hi, color=C_BAND, alpha=0.35, lw=0, zorder=0)
        ax.fill_between(wl_um, q25, q75, color=color, alpha=0.22, lw=0, zorder=2, label="interquartile range")
        ax.plot(wl_um, med, color=color, lw=1.0, zorder=3, label="median")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4, zorder=1)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlim(wl_um.min() - 0.01, wl_um.max() + 0.01)
    # one legend for both panels (same encoding), plus the band shading
    from matplotlib.patches import Patch

    h, lab = axes[0].get_legend_handles_labels()
    h.append(Patch(color=C_BAND, alpha=0.35, lw=0))
    lab.append(r"H$_2$O vapor band")
    # above the top panel so it never hides a peak
    axes[0].legend(h, lab, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                   borderaxespad=0.2)
    axes[1].set_xlabel(r"Wavelength ($\mu$m)")
    if title_note:
        fig.suptitle(title_note, fontsize=8, color="0.35")
    for ext, dpi in (("png", 500), ("pdf", None)):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_base}.png / .pdf")


# ═════════════════════════════════════════════════════════════════════════
# 6. Driver
# ═════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Per-profile spectral permutation importance (profile + τ_c network)")
    p.add_argument("--tag", default=None, help="training round; uses training_runs/<tag>/selected_model.json")
    p.add_argument("--run-dir", default=None, help="run directory with summary.json + best_model.pt")
    p.add_argument("--h5-path", default=None, help="local HDF5 if the path in summary.json is not reachable")
    p.add_argument("--perturbation", choices=["permute", "mean", "min"], default="permute")
    p.add_argument("--n-repeats", type=int, default=10, help="permutations per input (default 10)")
    p.add_argument("--seed", type=int, default=20261115, help="permutation RNG seed (recorded in outputs)")
    p.add_argument("--group-width-nm", type=float, default=0.0, help="grouped PFI window; 0 = single channel")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--out-dir", default=None, help="default <run_dir>/spectral_feature_importance")
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved .npz")
    p.add_argument("--no-title", action="store_true", help="omit the provenance line above the figure")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = selected_run_dir(args.tag, args.run_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "spectral_feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.perturbation + (f"_group{args.group_width_nm:g}nm" if args.group_width_nm > 0 else "")
    npz_path = out_dir / f"spectral_importance_{suffix}.npz"

    if args.plot_only:
        res = dict(np.load(npz_path, allow_pickle=False))
    else:
        import torch

        torch.manual_seed(args.seed)  # no stochastic torch op is used; seeded anyway
        device = pick_device(args.device)
        print(f"device {device} | perturbation {args.perturbation} | K {args.n_repeats} | seed {args.seed}")
        d = load_model_and_test_set(run_dir, args.h5_path, device)
        r = run_importance(d, device, args.perturbation, args.n_repeats, args.seed, args.group_width_nm)

        # ---- consistency check with step 14's cached predictions, if present
        cache = run_dir / "pred_cache.npz"
        if cache.exists():
            pc = np.load(cache)
            if np.array_equal(pc["h5_idx"], d["test_idx"]):
                dmax = float(np.abs(pc["pred"] - r["re_base_um"]).max())
                print(f"  check vs pred_cache.npz: same test rows, max |Δr_e| = {dmax:.2e} μm")
            else:
                print("  WARNING: pred_cache.npz test rows differ from this split — check seeds")

        sp, st = summarize(r["d_prof_um"]), summarize(r["d_tau"])
        res = dict(
            wavelengths_nm=d["wavelengths_nm"],
            test_h5_idx=d["test_idx"],
            # per-profile matrices (inputs × test profiles), float32 to save space
            d_prof_um=r["d_prof_um"].astype(np.float32),
            d_tau=r["d_tau"].astype(np.float32),
            mc_se_prof_um=r["mc_se_prof_um"].astype(np.float32),
            mc_se_tau=r["mc_se_tau"].astype(np.float32),
            tau_shift_mean=r["tau_shift_mean"].astype(np.float32),
            median_per_rep_prof_um=r["median_per_rep_prof_um"],
            median_per_rep_tau=r["median_per_rep_tau"],
            base_rmse_um=r["base_rmse_um"],
            base_tau_abs_err=r["base_tau_abs_err"],
            group_sizes=r["group_sizes"],
            **{f"prof_{k}": v for k, v in sp.items()},
            **{f"tau_{k}": v for k, v in st.items()},
            perturbation=args.perturbation,
            n_repeats=r["n_repeats"],
            seed=args.seed,
            split_seed=d["split_seed"],
            group_width_nm=args.group_width_nm,
            run_dir=str(run_dir),
            h5_path=d["h5_path"],
        )
        np.savez_compressed(npz_path, **res)
        print(f"  wrote {npz_path}")

        # ---- per-input CSV
        labels = [f"{w:.2f} nm" for w in d["wavelengths_nm"]] + GEOM_NAMES
        cols = ["median", "q25", "q75", "mean", "frac_positive", "median_ci_lo", "median_ci_hi"]
        with (out_dir / f"spectral_importance_{suffix}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["input_idx", "label", "group_size"] + [f"prof_{c}_um" if "frac" not in c else f"prof_{c}" for c in cols]
                       + [f"tau_{c}" for c in cols])
            for i in range(N_INPUTS):
                w.writerow([i, labels[i], int(r["group_sizes"][i])]
                           + [f"{sp[c][i]:.6g}" for c in cols] + [f"{st[c][i]:.6g}" for c in cols])

        # ---- diagnostics printed for the record
        e0, a0 = r["base_rmse_um"], r["base_tau_abs_err"]
        print(f"\n  baseline per-profile RMSE: median {np.median(e0):.4f} μm, "
              f"IQR {np.percentile(e0, 25):.4f}–{np.percentile(e0, 75):.4f} μm")
        print(f"  baseline |τ error|       : median {np.median(a0):.4f}, "
              f"IQR {np.percentile(a0, 25):.4f}–{np.percentile(a0, 75):.4f}")
        for name, s_ in (("profile", sp), ("τ_c", st)):
            top = np.argsort(-s_["median"])[:10]
            print(f"\n  top 10 inputs by median Δ ({name}):")
            for c in top:
                print(f"    {labels[c]:>10}  median {s_['median'][c]:+.4f}  IQR [{s_['q25'][c]:+.4f}, {s_['q75'][c]:+.4f}]")
        if r["n_repeats"] > 1:
            mc = np.median(r["mc_se_prof_um"][:N_SPECTRAL])
            iqr = np.median(sp["q75"][:N_SPECTRAL] - sp["q25"][:N_SPECTRAL])
            print(f"\n  Monte-Carlo SE of a per-profile Δ (median over all): {mc:.4f} μm "
                  f"vs median IQR across profiles {iqr:.4f} μm")
            half = r["n_repeats"] // 2
            if half >= 2:
                a_ = np.nanmean(r["median_per_rep_prof_um"][:half], axis=0)[:N_SPECTRAL]
                b_ = np.nanmean(r["median_per_rep_prof_um"][half:], axis=0)[:N_SPECTRAL]
                print(f"  split-half agreement of median Δ curve (profile): r = {np.corrcoef(a_, b_)[0, 1]:.4f}")
        rho = np.corrcoef(sp["median"][:N_SPECTRAL], st["median"][:N_SPECTRAL])[0, 1]
        print(f"  corr(profile, τ) median-Δ spectra: r = {rho:+.3f}")

    note = "" if args.no_title else (
        f"{Path(str(res['run_dir'])).name} | {res['perturbation']}"
        + (f", K = {int(res['n_repeats'])}" if str(res["perturbation"]) == "permute" else "")
        + (f", window {float(res['group_width_nm']):g} nm" if float(res["group_width_nm"]) > 0 else "")
        + f" | N = {res['d_prof_um'].shape[1]} test profiles"
    )
    plot_two_panel(res, out_dir / f"fig_spectral_importance_{suffix}", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
