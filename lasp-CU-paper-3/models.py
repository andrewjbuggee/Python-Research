"""
models.py — Neural network architectures and physics-based loss functions.

Contains:
  - DropletProfileNetwork: Direct retrieval network (Stage 1)
  - ForwardModelEmulator: libRadtran emulator (Stage 2, optional)
  - Physics-informed loss functions
  - Utility functions for denormalization

The retrieval network outputs are cleanly separated into:
  - profile: r_e at N vertical levels (cloud top to cloud base)
  - tau_c:   cloud optical depth
This separation enables the Stage 2 emulator to be plugged in by
connecting the retrieval network output to the emulator input.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for the retrieval network."""
    # Input dimensions
    n_wavelengths: int = 636         # HySICS bands; use all 636 spectral channels
    n_geometry_inputs: int = 4      # Solar zenith, viewing zenith, solar azimuth, viewing azimuth
    # Output dimensions
    n_levels: int = 10              # Vertical levels in profile
    # Architecture
    hidden_dims: tuple = (256, 256, 256, 256)
    dropout: float = 0.1
    activation: str = "gelu"
    # Physical bounds (used in output layer)
    re_min: float = 1.25             # μm
    re_max: float = 50.0            # μm
    tau_min: float = 1.5
    tau_max: float = 40.0

    @property
    def input_dim(self):
        return self.n_wavelengths + self.n_geometry_inputs


@dataclass
class EmulatorConfig:
    """Configuration for the forward model emulator (Stage 2)."""
    n_levels: int = 10
    n_geometry_inputs: int = 4           # Solar zenith, viewing zenith, solar azimuth, viewing azimuth
    n_wavelengths_out: int = 636         # output reflectance channels;HySICS bands; use all 636 spectral channels
    hidden_dims: tuple = (256, 256, 256, 256, 256)
    dropout: float = 0.05
    activation: str = "gelu"

    @property
    def input_dim(self):
        return self.n_levels + 1 + self.n_geometry_inputs  # profile + tau_c + geometry


# =============================================================================
# Retrieval Network (Stage 1)
# =============================================================================

class DropletProfileNetwork(nn.Module):
    """
    Direct retrieval network: maps spectral reflectances to a droplet
    effective radius profile at N vertical levels plus cloud optical depth.

    Architecture:
        Input: [R(λ₁), ..., R(λ_K), SZA, VZA, φ, wind_speed]  (normalized)
        → Shared encoder (MLP)
        → Profile head: N outputs passed through sigmoid → [re_min, re_max]
        → Tau head: 1 output passed through sigmoid → [tau_min, tau_max]

    The output is a named dictionary, not a raw tensor, so that:
      1. You always know which output is which
      2. The emulator (Stage 2) can accept these outputs directly
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__()
        self.config = config or RetrievalConfig()
        c = self.config

        # Activation function
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[c.activation]

        # Shared encoder backbone
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

        # Profile head: outputs mean r_e at N levels
        # Sigmoid output scaled to [re_min, re_max]
        self.profile_head = nn.Sequential(
            nn.Linear(in_dim, c.n_levels),
            nn.Sigmoid(),
        )

        # Profile uncertainty head: outputs log-std at N levels (in normalized space)
        # Clamped in forward to keep std in a numerically stable range
        self.profile_std_head = nn.Linear(in_dim, c.n_levels)

        # Tau head: outputs cloud optical depth
        # Sigmoid output scaled to [tau_min, tau_max]
        self.tau_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

        # Tau uncertainty head: outputs log-std of tau (in normalized space).
        # Clamped in forward to the same stable range as profile_std_head.
        self.tau_std_head = nn.Linear(in_dim, 1)

        # Store bounds as buffers (move to GPU with model)
        self.register_buffer('re_min', torch.tensor(c.re_min))
        self.register_buffer('re_max', torch.tensor(c.re_max))
        self.register_buffer('tau_min', torch.tensor(c.tau_min))
        self.register_buffer('tau_max', torch.tensor(c.tau_max))

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Normalized input: reflectances + geometry.

        Returns
        -------
        output : dict with keys:
            'profile'      : (batch, n_levels) — mean r_e in [re_min, re_max] (μm)
                              Index 0 = cloud top, index -1 = cloud base
            'profile_std'  : (batch, n_levels) — std of r_e (μm)
            'tau_c'        : (batch, 1) — optical depth in [tau_min, tau_max]
            'tau_std'      : (batch, 1) — std of tau (same physical units as tau_c)
            'profile_normalized'     : (batch, n_levels) — mean r_e in [0, 1]
            'profile_std_normalized' : (batch, n_levels) — std in [0, 1] space
            'tau_normalized'         : (batch, 1) — tau in [0, 1]
            'tau_std_normalized'     : (batch, 1) — std of tau in [0, 1] space
        """
        features = self.encoder(x)

        # Profile mean: sigmoid → [0,1] then scale to physical range
        profile_norm = self.profile_head(features)       # (batch, n_levels), in [0, 1]
        profile = profile_norm * (self.re_max - self.re_min) + self.re_min

        # Profile std: exp(log_std) in normalized [0,1] space.
        # Clamp log_std to [-6, 2] → std in roughly [0.002, 7.4] normalized units,
        # i.e. [~0.1, ~370] μm — well outside physical range on the high end but
        # the NLL loss will drive it toward the data-appropriate scale.
        profile_log_std = self.profile_std_head(features)
        profile_std_norm = torch.exp(profile_log_std.clamp(-6.0, 2.0))  # (batch, n_levels)
        profile_std = profile_std_norm * (self.re_max - self.re_min)     # physical units (μm)

        # Tau output: sigmoid → [0,1] then scale to physical range
        tau_norm = self.tau_head(features)                # (batch, 1), in [0, 1]
        tau_c = tau_norm * (self.tau_max - self.tau_min) + self.tau_min

        # Tau std: same approach as profile_std — exp(clamped log-std) in normalized space
        tau_log_std = self.tau_std_head(features)
        tau_std_norm = torch.exp(tau_log_std.clamp(-6.0, 2.0))           # (batch, 1)
        tau_std = tau_std_norm * (self.tau_max - self.tau_min)            # physical units

        return {
            'profile': profile,
            'profile_std': profile_std,
            'tau_c': tau_c,
            'tau_std': tau_std,
            'profile_normalized': profile_norm,
            'profile_std_normalized': profile_std_norm,
            'tau_normalized': tau_norm,
            'tau_std_normalized': tau_std_norm,
        }


# =============================================================================
# Forward Model Emulator (Stage 2 — build later if time permits)
# =============================================================================

class ForwardModelEmulator(nn.Module):
    """
    Neural network emulator of libRadtran radiative transfer.

    Maps (r_e profile, τ_c, geometry) → predicted reflectances R̂(λ).

    This is trained separately on libRadtran simulations, then frozen
    and used in the PINN training loop to compute L_data:
        L_data = ||R̂(λ) - R_observed(λ)||² / σ²

    The emulator must be differentiable so gradients flow back through it
    to update the retrieval network weights.

    IMPORTANT: Train and validate this emulator to <0.5% relative error
    before using it in the PINN loop. If the emulator is inaccurate,
    it will degrade retrieval quality.
    """

    def __init__(self, config: Optional[EmulatorConfig] = None):
        super().__init__()
        self.config = config or EmulatorConfig()
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

        layers.append(nn.Linear(in_dim, c.n_wavelengths_out))
        # No final activation — reflectances can be any positive value
        # (though they're typically in [0, 0.8] for cloudy scenes)

        self.network = nn.Sequential(*layers)

    def forward(self, profile: torch.Tensor, tau_c: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        """
        Predict reflectances from cloud state and geometry.

        Parameters
        ----------
        profile : (batch, n_levels) — normalized r_e profile
        tau_c : (batch, 1) — normalized optical depth
        geometry : (batch, n_geometry) — normalized [SZA, VZA, φ, wind_speed]

        Returns
        -------
        reflectances : (batch, n_wavelengths) — predicted TOA reflectance
        """
        x = torch.cat([profile, tau_c, geometry], dim=-1)
        return self.network(x)


# =============================================================================
# Loss Functions
# =============================================================================

class SupervisedLoss(nn.Module):
    """
    Supervised loss for Stage 1.

    Both profile and tau use Gaussian negative log-likelihood (NLL):
        L = mean[ log(σ) + (y - μ)² / (2σ²) ]

    This lets the network learn calibrated uncertainty for both outputs.
    All quantities are in normalized [0, 1] space so the two loss terms
    remain on comparable scales.
    """

    def __init__(self, tau_weight: float = 1.0):
        super().__init__()
        self.tau_weight = tau_weight

    def forward(self, output: dict,
                profile_true: torch.Tensor,
                tau_true: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        output : dict from DropletProfileNetwork.forward()
        profile_true : (batch, n_levels) — normalized true profile
        tau_true : (batch,) — normalized true optical depth

        Returns
        -------
        losses : dict with 'profile_loss', 'tau_loss', 'total'
        """
        mu = output['profile_normalized']           # (batch, n_levels)
        sigma = output['profile_std_normalized']    # (batch, n_levels), positive by construction

        # Gaussian NLL (constant 0.5*log(2π) dropped — doesn't affect gradients)
        profile_loss = (torch.log(sigma) + 0.5 * ((profile_true - mu) / sigma).pow(2)).mean()

        tau_mu = output['tau_normalized'].squeeze(-1)       # (batch,)
        tau_sigma = output['tau_std_normalized'].squeeze(-1)  # (batch,), positive by construction
        tau_loss = (torch.log(tau_sigma) + 0.5 * ((tau_true - tau_mu) / tau_sigma).pow(2)).mean()

        total = profile_loss + self.tau_weight * tau_loss

        return {
            'profile_loss': profile_loss,
            'tau_loss': tau_loss,
            'total': total,
        }


class PhysicsLoss(nn.Module):
    """
    Physics-based penalty terms for the predicted droplet profile.

    These act as soft constraints — replacing the hard constraints
    (r_bot < r_top, physical bounds) from the Gauss-Newton method
    in Papers 1 & 2.

    Terms:
        L_monotonicity: Penalize decreasing r_e from top to base.
                        For an adiabatic cloud, r_e should increase
                        monotonically from cloud top to cloud base.

        L_adiabatic:    Penalize deviation from the adiabatic profile
                        shape (Eq. 4). This is a soft regularizer —
                        the network CAN deviate if the data supports it.

        L_smoothness:   Penalize large jumps between adjacent levels.
                        Encourages physically realistic smooth profiles.
    """

    def __init__(self,
                 lambda_monotonicity: float = 0.01,
                 lambda_smoothness: float = 0.1):
        super().__init__()
        self.lambda_mono = lambda_monotonicity
        self.lambda_smooth = lambda_smoothness

    def monotonicity_loss(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Penalize cases where r_e at a lower level (closer to base)
        is smaller than r_e at the level above (closer to top).

        Profile convention: index 0 = cloud top, index -1 = cloud base.
        For adiabatic clouds: profile[i] <= profile[i+1] for all i.

        Uses ReLU to only penalize violations (one-sided penalty).
        """
        # Difference between adjacent levels (should be >= 0 for adiabatic)
        # profile[:, 1:] - profile[:, :-1] should be >= 0 (base > top)
        diff = profile[:, :-1] - profile[:, 1:]  # positive if decreasing toward base = violation
        violations = torch.relu(diff)  # only penalize when top level > lower level
        return violations.pow(2).mean()

    def adiabatic_loss(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Penalize deviation from the adiabatic functional form.

        For a profile with r_top = profile[:,0] and r_bot = profile[:,-1],
        the adiabatic prediction at level i is:
            r_adiab(i) = (r_bot³ + (r_top³ - r_bot³) * (N-1-i)/(N-1))^(1/3)

        This loss encourages but does not force adiabatic behavior.
        """
        n_levels = profile.shape[1]
        r_top = profile[:, 0:1]    # (batch, 1)
        r_bot = profile[:, -1:]    # (batch, 1)

        # Normalized height: 0 at base, 1 at top
        # Since index 0 = top, index -1 = base:
        z_norm = torch.linspace(1, 0, n_levels, device=profile.device).unsqueeze(0)  # (1, n_levels)

        # Adiabatic profile (Eq. 4)
        r_adiab = (r_bot**3 + (r_top**3 - r_bot**3) * z_norm).clamp(min=1e-6) ** (1/3)

        return (profile - r_adiab).pow(2).mean()

    def smoothness_loss(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Penalize large second-order differences (curvature).
        Encourages smooth profiles without forcing any particular shape.
        """
        # Second-order finite difference
        d2 = profile[:, 2:] - 2 * profile[:, 1:-1] + profile[:, :-2]
        return d2.pow(2).mean()

    def forward(self, profile: torch.Tensor) -> dict:
        """
        Compute all physics loss terms.

        Parameters
        ----------
        profile : (batch, n_levels) — predicted r_e profile in physical
                  units (μm), NOT normalized.

        Returns
        -------
        losses : dict with individual terms and weighted total.
        """
        mono = self.monotonicity_loss(profile)
        smooth = self.smoothness_loss(profile)

        total = (self.lambda_mono * mono +
                 self.lambda_smooth * smooth)

        return {
            'monotonicity': mono,
            'smoothness': smooth,
            'physics_total': total,
        }


class EmulatorDataLoss(nn.Module):
    """
    Stage 2 data fidelity loss using the forward model emulator.

    L_data = Σ_λ (R̂(λ) - R_obs(λ))² / σ_λ²

    This loss requires the emulator to be trained and frozen.
    Gradients flow through the emulator to update the retrieval network.
    """

    def __init__(self, emulator: ForwardModelEmulator,
                 measurement_uncertainty: Optional[torch.Tensor] = None):
        super().__init__()
        self.emulator = emulator

        # Freeze emulator weights — we don't want to change the forward model
        for param in self.emulator.parameters():
            param.requires_grad = False

        # Per-channel measurement uncertainty (σ_λ)
        # If not provided, assume uniform uncertainty
        if measurement_uncertainty is not None:
            self.register_buffer('sigma', measurement_uncertainty)
        else:
            self.sigma = None

    def forward(self, retrieval_output: dict,
                observed_reflectances: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        retrieval_output : dict from DropletProfileNetwork
        observed_reflectances : (batch, n_wavelengths) — measured R(λ)
        geometry : (batch, n_geometry) — normalized geometry inputs

        Returns
        -------
        loss : scalar tensor — weighted reflectance misfit
        """
        # Pass retrieved profile through emulator
        predicted_refl = self.emulator(
            retrieval_output['profile_normalized'],
            retrieval_output['tau_normalized'],
            geometry,
        )

        residual = predicted_refl - observed_reflectances

        if self.sigma is not None:
            # Weighted by per-channel uncertainty
            loss = (residual / self.sigma).pow(2).mean()
        else:
            loss = residual.pow(2).mean()

        return loss


# =============================================================================
# Combined Loss (brings everything together)
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss for training the retrieval network.

    Stage 1: L = L_supervised + λ_phys * L_physics
    Stage 2: L = L_supervised + λ_phys * L_physics + λ_data * L_emulator_data

    Set lambda_emulator_data = 0 for Stage 1 (no emulator).
    """

    def __init__(self,
                 config: RetrievalConfig,
                 lambda_physics: float = 0.1,
                 lambda_monotonicity: float = 0.01,
                 lambda_smoothness: float = 0.1,
                 lambda_emulator_data: float = 0.0,
                 emulator: Optional[ForwardModelEmulator] = None,
                 measurement_uncertainty: Optional[torch.Tensor] = None,
                 ):
        super().__init__()

        self.supervised = SupervisedLoss(tau_weight=1.0)
        self.physics = PhysicsLoss(
            lambda_monotonicity=lambda_monotonicity,
            lambda_smoothness=lambda_smoothness,
        )
        self.lambda_physics = lambda_physics
        self.lambda_emulator = lambda_emulator_data

        # Stage 2: emulator-based data loss
        self.emulator_loss = None
        if emulator is not None and lambda_emulator_data > 0:
            self.emulator_loss = EmulatorDataLoss(emulator, measurement_uncertainty)

    def forward(self, output: dict,
                profile_true: torch.Tensor,
                tau_true: torch.Tensor,
                observed_reflectances: Optional[torch.Tensor] = None,
                geometry: Optional[torch.Tensor] = None,
                ) -> dict:
        """
        Compute total loss.

        Returns dict with all individual terms for logging.
        """
        # Supervised loss (always active)
        sup_losses = self.supervised(output, profile_true, tau_true)

        # Physics loss on predicted profile (physical units)
        phys_losses = self.physics(output['profile'])

        total = sup_losses['total'] + self.lambda_physics * phys_losses['physics_total']

        # Stage 2: emulator data loss
        emulator_data_loss = torch.tensor(0.0, device=total.device)
        if self.emulator_loss is not None and observed_reflectances is not None:
            emulator_data_loss = self.emulator_loss(output, observed_reflectances, geometry)
            total = total + self.lambda_emulator * emulator_data_loss

        return {
            'total': total,
            'supervised_profile': sup_losses['profile_loss'],
            'supervised_tau': sup_losses['tau_loss'],
            'physics_monotonicity': phys_losses['monotonicity'],
            'physics_smoothness': phys_losses['smoothness'],
            'physics_total': phys_losses['physics_total'],
            'emulator_data': emulator_data_loss,
        }
