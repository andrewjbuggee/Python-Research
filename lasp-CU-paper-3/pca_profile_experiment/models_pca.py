"""
models_pca.py
-------------
Step 2 of the PCA-head experiment: a drop-in replacement for
`DropletProfileNetwork` in which the profile head predicts K principal-component
scores instead of N independent per-level radii.

Design
======
The baseline network outputs N independent scalar values for the droplet profile
(one per vertical level).  With only ~300 unique training profiles, the output
manifold is massively overparameterized — the real physical profile lives on a
low-dimensional manifold (approximately adiabatic + small deviations).

This module fixes that by:
  1. Fitting PCA on the training profiles (done in step 1,
     `analyze_profile_pca.py`) to get a fixed `(mean, components)` basis.
  2. Having the encoder predict K PCA *scores* (unbounded real numbers).
  3. Decoding those scores into an N-level profile via the fixed linear basis:
         profile_norm = scores @ components + mean
     This decoding is a PyTorch buffer so gradients flow through the linear
     combination — but the basis itself is NOT trained.

Uncertainty
-----------
We keep a per-level σ head (identical in shape to the baseline's
`profile_std_head`) so that the Gaussian-NLL loss and the rest of the training
pipeline continue to work unchanged.  The *mean* profile is constrained to the
PCA manifold; the *uncertainty* is still allowed per level.  This is the
simplest architectural change that tests the hypothesis, and it keeps all the
existing metrics (per-level RMSE, tau RMSE, aleatoric σ calibration) directly
comparable with the baseline.

Why keep τ_c in the network?
----------------------------
The user asked in the previous message whether a profile-only network would
help.  That is a *separate* ablation that we want to run on the PCA-head
network independently, so this module keeps the τ head by default and exposes
a `use_tau_head` flag that lets us remove it later without another refactor.

Usage
=====
    from models_pca import PCADropletProfileNetwork, PCARetrievalConfig, load_pca_basis

    basis = load_pca_basis("pca_basis.npz")
    config = PCARetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=basis["components"].shape[1],
        n_pca_components=3,                 # choose K
        hidden_dims=(256, 256, 256, 256),
        dropout=0.2,
        use_tau_head=True,
    )
    model = PCADropletProfileNetwork(config)
    model.register_pca(basis["mean"], basis["components"][:3])

The returned `model.forward(x)` dict has the same keys as the baseline:
    profile, profile_std, tau_c, tau_std,
    profile_normalized, profile_std_normalized,
    tau_normalized, tau_std_normalized,
plus two new entries that are unique to the PCA-head:
    pc_scores           (batch, K) — raw predicted PC scores
    pc_scores_std       (batch, K) — aleatoric σ on each score (normalized space)

These extra keys let the compare script and the loss functions inspect the
latent representation if they want to, but the baseline loss function in
`models.py` (`CombinedLoss`) uses only the top-level keys, so it works as-is.

Author: Claude (assistant) for Andrew J. Buggee
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class PCARetrievalConfig:
    """Configuration for the PCA-head retrieval network."""
    n_wavelengths: int = 636
    n_geometry_inputs: int = 4
    n_levels: int = 7                    # N vertical levels in the output profile
    n_pca_components: int = 3            # K — the *retained* PCA components
    hidden_dims: tuple = (256, 256, 256, 256)
    dropout: float = 0.1
    activation: str = "gelu"
    re_min: float = 1.5
    re_max: float = 50.0
    tau_min: float = 3.0
    tau_max: float = 75.0
    use_tau_head: bool = True            # turn off for the single-task ablation

    # Whether to train the PCA basis jointly with the network.  Default False.
    # Keeping it False means the basis is a *fixed* inductive bias — exactly
    # the hypothesis we want to test.  Setting it to True would let the network
    # learn its own low-dim basis (equivalent to a bottleneck layer) — a
    # separate experiment that we can try later.
    learn_pca_basis: bool = False

    @property
    def input_dim(self) -> int:
        return self.n_wavelengths + self.n_geometry_inputs


# ───────────────────────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────────────────────
class PCADropletProfileNetwork(nn.Module):
    """
    Profile-PCA-head retrieval network.

    Inputs:
        x : (batch, input_dim)  — normalized reflectance spectrum + geometry

    Outputs (dict):
        profile               : (batch, n_levels)  — r_e in μm
        profile_normalized    : (batch, n_levels)  — r_e in [0, 1]
        profile_std           : (batch, n_levels)  — aleatoric σ on r_e (μm)
        profile_std_normalized: (batch, n_levels)  — σ in [0, 1] space
        tau_c                 : (batch, 1)         — optical depth (physical)
        tau_normalized        : (batch, 1)         — τ_c in [0, 1]
        tau_std               : (batch, 1)         — σ(τ_c) in physical units
        tau_std_normalized    : (batch, 1)         — σ(τ_c) in [0, 1]
        pc_scores             : (batch, K)         — predicted PC scores
        pc_scores_std         : (batch, K)         — σ on each PC score

    Notes
    -----
    * The PCA basis is registered as buffers (`pca_mean`, `pca_components`)
      via `.register_pca()`.  The buffers move to the model's device via
      `.to(device)` and are saved in `state_dict()` automatically.

    * The decoded `profile_normalized` is NOT clamped to [0, 1].  In theory
      the model could predict out-of-range values for pathological inputs,
      but the training data keeps profiles inside the manifold and the loss
      pulls predictions back toward the data.  We clamp ONLY when converting
      to physical units for reporting, so the gradient path stays linear.

    * If `use_tau_head=False` the tau head is still instantiated but the
      outputs are frozen zeros (so downstream code using `output['tau_c']`
      doesn't crash).  Set the τ_weight in `SupervisedLoss` to 0 in the
      training script to fully remove τ from the loss.
    """

    def __init__(self, config: Optional[PCARetrievalConfig] = None):
        super().__init__()
        self.config = config or PCARetrievalConfig()
        c = self.config

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[c.activation]

        # ── Shared encoder backbone (same structure as baseline) ──────────
        layers = []
        in_dim = c.input_dim
        for h in c.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                act_cls(),
                nn.Dropout(c.dropout),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        # ── Profile head: predict K unbounded PCA scores ──────────────────
        # No activation — PC scores are real-valued.  Linear init.
        self.pc_score_head = nn.Linear(in_dim, c.n_pca_components)

        # Per-level aleatoric σ head — IDENTICAL to baseline.
        # Predicts log(σ) in normalized space; σ is obtained by exp and clamp.
        self.profile_std_head = nn.Linear(in_dim, c.n_levels)

        # Optional: a σ head on the PC scores themselves, for users who want
        # to compare uncertainty in the latent space.  Cheap (K outputs).
        self.pc_score_std_head = nn.Linear(in_dim, c.n_pca_components)

        # ── τ_c head (optional) ───────────────────────────────────────────
        self.tau_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )
        self.tau_std_head = nn.Linear(in_dim, 1)

        # Physical-bound buffers so `.to(device)` moves them with the model.
        self.register_buffer("re_min",  torch.tensor(c.re_min))
        self.register_buffer("re_max",  torch.tensor(c.re_max))
        self.register_buffer("tau_min", torch.tensor(c.tau_min))
        self.register_buffer("tau_max", torch.tensor(c.tau_max))

        # PCA basis buffers — populated by `.register_pca()`.  Must be set
        # before `forward()` is called.  We pre-register them as None so the
        # attributes exist (avoiding `register_buffer` "already exists"
        # collisions when register_pca() runs); they get real tensors later.
        self.register_buffer("pca_mean", None, persistent=True)
        self.register_buffer("pca_components", None, persistent=True)

    # ------------------------------------------------------------------
    # PCA basis management
    # ------------------------------------------------------------------
    def register_pca(self, mean: np.ndarray, components: np.ndarray) -> None:
        """
        Register the PCA basis produced by `analyze_profile_pca.py`.

        Parameters
        ----------
        mean       : (n_levels,) float — the training mean in normalized [0,1] space
        components : (n_pca_components, n_levels) float — the PC basis rows

        Notes
        -----
        If `config.learn_pca_basis=False` (default), the basis is stored as
        buffers — they move with `.to(device)` and are saved in `state_dict()`
        but are NOT updated by the optimizer.  This is the "fixed inductive
        bias" mode.

        If `config.learn_pca_basis=True`, the basis is stored as Parameters
        so gradients update them.  Use only for comparison experiments.
        """
        K_expected = self.config.n_pca_components
        L_expected = self.config.n_levels
        if components.shape != (K_expected, L_expected):
            raise ValueError(
                f"components has shape {components.shape}, expected "
                f"({K_expected}, {L_expected})"
            )
        if mean.shape != (L_expected,):
            raise ValueError(
                f"mean has shape {mean.shape}, expected ({L_expected},)"
            )

        # Move to the same device as the model.
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        m = torch.tensor(mean,       dtype=torch.float32, device=dev)
        C = torch.tensor(components, dtype=torch.float32, device=dev)

        if self.config.learn_pca_basis:
            # Make them trainable.  We must drop the pre-registered None
            # buffers first — otherwise nn.Module.__setattr__ refuses to
            # promote a buffer slot to a Parameter slot.
            del self._buffers["pca_mean"]
            del self._buffers["pca_components"]
            self.pca_mean       = nn.Parameter(m)
            self.pca_components = nn.Parameter(C)
        else:
            # Fixed buffers: assignment by attribute name updates the
            # existing (None-valued) buffer that was registered in __init__.
            # nn.Module.__setattr__ handles this correctly for buffer slots.
            self.pca_mean = m
            self.pca_components = C

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        if self.pca_mean is None or self.pca_components is None:
            raise RuntimeError(
                "PCA basis has not been registered.  Call model.register_pca(...) "
                "before the first forward pass."
            )
        c = self.config
        features = self.encoder(x)

        # Predict PC scores and their log-σ.
        pc_scores        = self.pc_score_head(features)              # (B, K)
        pc_log_std       = self.pc_score_std_head(features)          # (B, K)
        pc_scores_std    = torch.exp(pc_log_std.clamp(-4.0, 2.0))    # keep stable

        # Decode to normalized profile: (B, K) @ (K, L) + (L,) → (B, L)
        profile_norm = pc_scores @ self.pca_components + self.pca_mean

        # Convert normalized → physical μm, clamping ONLY for the physical
        # output (keeps the training gradient path linear).
        profile_norm_clamped = profile_norm.clamp(0.0, 1.0)
        profile = profile_norm_clamped * (self.re_max - self.re_min) + self.re_min

        # Per-level aleatoric σ — same treatment as baseline.
        profile_log_std      = self.profile_std_head(features)
        profile_std_norm     = torch.exp(profile_log_std.clamp(-4.0, 2.0))
        profile_std          = profile_std_norm * (self.re_max - self.re_min)

        # τ_c head.  If use_tau_head=False we still output the tensor (frozen)
        # so downstream code is agnostic to the flag; set τ_weight=0 in the
        # loss to drop it from training.
        if c.use_tau_head:
            tau_norm     = self.tau_head(features)                   # (B, 1)
            tau_log_std  = self.tau_std_head(features)
            tau_std_norm = torch.exp(tau_log_std.clamp(-4.0, 2.0))
        else:
            # Constant outputs — still differentiable (derivatives are 0).
            tau_norm     = torch.full_like(
                features[:, :1], 0.5  # middle of the [0, 1] range
            ).detach()
            tau_std_norm = torch.full_like(features[:, :1], 1.0).detach()

        tau_c    = tau_norm     * (self.tau_max - self.tau_min) + self.tau_min
        tau_std  = tau_std_norm * (self.tau_max - self.tau_min)

        return {
            "profile":                 profile,
            "profile_std":             profile_std,
            "tau_c":                   tau_c,
            "tau_std":                 tau_std,
            "profile_normalized":      profile_norm_clamped,
            "profile_std_normalized":  profile_std_norm,
            "tau_normalized":          tau_norm,
            "tau_std_normalized":      tau_std_norm,
            "pc_scores":               pc_scores,
            "pc_scores_std":           pc_scores_std,
            # Keep the UN-clamped version handy for debug / physics-penalty
            # analysis; not used by the baseline loss.
            "profile_normalized_raw":  profile_norm,
        }


# ───────────────────────────────────────────────────────────────────────────────
# Convenience: load the saved PCA basis into a dict of arrays
# ───────────────────────────────────────────────────────────────────────────────
def load_pca_basis(path: str | Path) -> dict:
    """
    Load the .npz file written by `analyze_profile_pca.py`.

    Returns a dict with keys:  mean, components, explained,
    train_rmse_per_K, val_rmse_per_K, h5_path, seed, n_val_profiles,
    n_test_profiles, n_levels, re_min, re_max, tau_min, tau_max.

    All numeric values are returned as numpy arrays (or Python scalars for
    string-like entries) so the caller can slice `components[:K]` directly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PCA basis file not found: {path}.  Run analyze_profile_pca.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    # Convert 0-d arrays to Python scalars for convenience.
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.shape == ():
            out[k] = v.item()
    return out
