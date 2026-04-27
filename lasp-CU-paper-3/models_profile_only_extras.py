"""
models_profile_only_extras.py — Profile-only retrieval network with three
extra scalar inputs appended to the existing 640-dim spectrum+geometry vector.

Inputs (643 total):
    640: log10(reflectance, 636 channels) ⊕ [sza, vza, saz, vaz]   (existing)
      1: log10(tau_c)                              z-scored
      1: log10(wv_above_cloud)  [molec/cm²]        z-scored
      1: log10(wv_in_cloud)     [molec/cm²]        z-scored

Output: 50-level droplet effective-radius profile + per-level σ
        (unchanged from ProfileOnlyNetwork).

The loss reuses ProfileOnlyLoss from models_profile_only.py — the loss only
sees the *output* dict, so it doesn't care about input dimensionality.

Why a separate file
-------------------
Adding scalar physical priors as inputs is a meaningfully different model
class:
  * tau_c is normally a *retrieved* quantity, not an input — providing it
    is an "oracle prior" experiment.  Its information is concentrated in
    visible-continuum reflectance, so the network should already be
    extracting it; the question is whether *explicitly* providing it
    redirects capacity to profile-shape recovery.
  * wv_above_cloud and wv_in_cloud are climatological priors (from ERA5).
    They modulate the absorption bands the network uses for cloud-base
    droplet sizing.  These are *not* recoverable from the spectrum alone
    in a way that's separable from the cloud signal.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models import RetrievalConfig


# =============================================================================
# Profile-only retrieval network with N extra scalar inputs
# =============================================================================

class ProfileOnlyNetworkExtras(nn.Module):
    """
    Same body as ProfileOnlyNetwork, but the first encoder Linear layer takes
    `RetrievalConfig.input_dim + n_extras` inputs instead of `input_dim`.

    Use n_extras=3 to ingest [tau_c, wv_above_cloud, wv_in_cloud] alongside
    the existing 640-dim spectrum+geometry vector.

    Forward pass returns the same dict as ProfileOnlyNetwork:
        profile, profile_std, profile_normalized, profile_std_normalized.
    """

    def __init__(self,
                 config: Optional[RetrievalConfig] = None,
                 n_extras: int = 3):
        super().__init__()
        self.config = config or RetrievalConfig()
        self.n_extras = int(n_extras)
        c = self.config

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[c.activation]

        layers = []
        in_dim = c.input_dim + self.n_extras   # 640 + n_extras
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

    @property
    def input_dim(self) -> int:
        return self.config.input_dim + self.n_extras

    def forward(self, x: torch.Tensor) -> dict:
        # x has shape (batch, input_dim + n_extras)
        features = self.encoder(x)

        profile_norm = self.profile_head(features)
        profile = profile_norm * (self.re_max - self.re_min) + self.re_min

        # Same log-std clamp range as ProfileOnlyNetwork: prevents both
        # σ-collapse on memorized training profiles and NLL blow-up.
        profile_log_std = self.profile_std_head(features)
        profile_std_norm = torch.exp(profile_log_std.clamp(-4.0, 2.0))
        profile_std = profile_std_norm * (self.re_max - self.re_min)

        return {
            'profile': profile,
            'profile_std': profile_std,
            'profile_normalized': profile_norm,
            'profile_std_normalized': profile_std_norm,
        }
