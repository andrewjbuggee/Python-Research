"""
models_profile_only.py — Profile-only retrieval network and loss.

This is a fork of `models.py`'s DropletProfileNetwork + CombinedLoss with the
optical-depth (τ) head and τ-NLL term removed.  Use this when the retrieval
target is the droplet effective-radius vertical profile only.

Why this exists separately
--------------------------
The PCA-head experiment showed that the spectrum carries roughly one cleanly
cross-profile-retrievable PC of profile information (PC1, mean K-fold R²
≈ 0.59).  τ retrieval is well-studied and dominated by independent spectral
features (visible continuum brightness).  Bundling τ into the same loss as
the profile shifts gradient priorities and was empirically harming per-level
profile RMSE in the K=5/K=7 PCA-head runs.  This module is the clean
"single-task profile retrieval" model used by the new sweep.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models import PhysicsLoss, RetrievalConfig  # reuse existing pieces


# =============================================================================
# Profile-only retrieval network
# =============================================================================

class ProfileOnlyNetwork(nn.Module):
    """
    Profile-only retrieval network.

    Forward pass returns a dict with only the profile mean and per-level
    aleatoric uncertainty — no τ, no τ_std.  Architecture mirrors
    DropletProfileNetwork (encoder MLP + sigmoid-bounded mean head + log-std
    head) so hyperparameters and training code carry over unchanged.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__()
        self.config = config or RetrievalConfig()
        c = self.config

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[c.activation]

        layers = []
        in_dim = c.input_dim
        for h_dim in c.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_fn(),
                nn.Dropout(c.dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.profile_head = nn.Sequential(
            nn.Linear(in_dim, c.n_levels),
            nn.Sigmoid(),
        )
        self.profile_std_head = nn.Linear(in_dim, c.n_levels)

        self.register_buffer('re_min', torch.tensor(c.re_min))
        self.register_buffer('re_max', torch.tensor(c.re_max))

    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)

        profile_norm = self.profile_head(features)
        profile = profile_norm * (self.re_max - self.re_min) + self.re_min

        # Same log-std clamp range as DropletProfileNetwork: prevents both
        # σ-collapse (too-small uncertainty on memorized training profiles)
        # and NLL blow-up.
        profile_log_std = self.profile_std_head(features)
        profile_std_norm = torch.exp(profile_log_std.clamp(-4.0, 2.0))
        profile_std = profile_std_norm * (self.re_max - self.re_min)

        return {
            'profile': profile,
            'profile_std': profile_std,
            'profile_normalized': profile_norm,
            'profile_std_normalized': profile_std_norm,
        }


# =============================================================================
# Profile-only loss
# =============================================================================

class ProfileOnlyLoss(nn.Module):
    """
    Supervised Gaussian-NLL loss + physics regularizers on the profile only.

    Same level-weighting scheme as the original SupervisedLoss.  The four
    physics weights (lambda_physics, lambda_monotonicity, lambda_adiabatic,
    lambda_smoothness) are passed straight through; lambda_physics is the
    OUTER weight on the sum of the three physics terms (matches existing
    CombinedLoss convention from models.py).
    """

    def __init__(self,
                 config: RetrievalConfig,
                 lambda_physics: float = 0.1,
                 lambda_monotonicity: float = 0.0,
                 lambda_adiabatic: float = 0.1,
                 lambda_smoothness: float = 0.1,
                 level_weights: Optional[torch.Tensor] = None,
                 sigma_floor: float = 0.01):
        super().__init__()
        self.config = config
        self.sigma_floor = sigma_floor

        if level_weights is not None:
            w = level_weights.float()
            w = w / w.mean()  # normalize to mean 1 — preserves loss scale
            self.register_buffer('level_weights', w)
        else:
            self.level_weights = None

        self.lambda_physics = lambda_physics
        self.physics = PhysicsLoss(
            lambda_monotonicity=lambda_monotonicity,
            lambda_adiabatic=lambda_adiabatic,
            lambda_smoothness=lambda_smoothness,
        )

    def forward(self, output: dict, profile_true: torch.Tensor,
                tau_true_unused: Optional[torch.Tensor] = None) -> dict:
        """
        Parameters
        ----------
        output       : dict from ProfileOnlyNetwork.forward()
        profile_true : (batch, n_levels) — normalized true profile in [0, 1]
        tau_true_unused : ignored.  Accepted for signature compatibility with
                       sweep_train.py's existing call site (which passes tau).

        Returns
        -------
        losses dict with: total, supervised_profile, physics_*.
        """
        mu    = output['profile_normalized']
        sigma = output['profile_std_normalized'].clamp(min=self.sigma_floor)

        nll = torch.log(sigma) + 0.5 * ((profile_true - mu) / sigma).pow(2)
        if self.level_weights is not None:
            sup = (nll * self.level_weights).mean()
        else:
            sup = nll.mean()

        phys = self.physics(output['profile'])
        total = sup + self.lambda_physics * phys['physics_total']

        return {
            'total': total,
            'supervised_profile': sup,
            'physics_monotonicity': phys['monotonicity'],
            'physics_adiabatic':    phys['adiabatic'],
            'physics_smoothness':   phys['smoothness'],
            'physics_total':        phys['physics_total'],
        }
