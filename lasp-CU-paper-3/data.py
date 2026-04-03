"""
data.py — Data loading utilities for cloud droplet profile retrieval.

Handles:
  - Loading libRadtran-generated training data from HDF5 files
  - Loading VOCALS-REx in situ validation data
  - Creating PyTorch Datasets and DataLoaders
  - Train/val/test splitting

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict


# =============================================================================
# Constants
# =============================================================================

# MODIS first 7 spectral channels (nm) — used in Papers 1 & 2
MODIS_WAVELENGTHS = np.array([645, 858.5, 469, 555, 1240, 1640, 2130])

# Number of vertical levels for profile retrieval
# Start with N=10; increase later if information content supports it
DEFAULT_N_LEVELS = 10

# Physical bounds on effective radius (μm)
# Update RE_MAX after running convert_matFiles_to_HDF.py — the scan pass
# will print the observed max across all in-situ profiles.
RE_MIN = 1.25
RE_MAX = 50.0   # TODO: update from scan

# Optical depth bounds
# Update TAU_MAX after running convert_matFiles_to_HDF.py scan.
TAU_MIN = 1.5   # in-situ profiles include sub-cloud layers where tau=0
TAU_MAX = 40.0  # TODO: update from scan


# =============================================================================
# Adiabatic profile utilities
# =============================================================================

def adiabatic_profile(r_top: float, r_bot: float, n_levels: int) -> np.ndarray:
    """
    Compute an adiabatic effective radius profile following Eq. 4 from Paper 1:
        r_e(z) = (r_bot^3 + (r_top^3 - r_bot^3) * z/H)^(1/3)

    Parameters
    ----------
    r_top : float
        Effective radius at cloud top (μm).
    r_bot : float
        Effective radius at cloud base (μm).
    n_levels : int
        Number of vertical levels (evenly spaced from base to top).

    Returns
    -------
    profile : np.ndarray, shape (n_levels,)
        Effective radius at each level, ordered from cloud top (index 0)
        to cloud base (index n_levels-1).
    """
    # z/H from 0 (base) to 1 (top)
    z_norm = np.linspace(0, 1, n_levels)
    profile_base_to_top = (r_bot**3 + (r_top**3 - r_bot**3) * z_norm) ** (1/3)

    # Reverse so index 0 = cloud top, index -1 = cloud base
    # This matches the convention in Papers 1 & 2 where optical depth
    # increases from cloud top (τ=0) to cloud base (τ=τ_c)
    return profile_base_to_top[::-1].copy()


def generate_random_profiles(n_samples: int,
                             n_levels: int = DEFAULT_N_LEVELS,
                             r_top_range: Tuple[float, float] = (4.0, 20.0),
                             r_bot_range: Tuple[float, float] = (1.0, None),
                             tau_range: Tuple[float, float] = (TAU_MIN, TAU_MAX),
                             include_subadiabatic: bool = True,
                             rng: Optional[np.random.Generator] = None,
                             ) -> Dict[str, np.ndarray]:
    """
    Generate random cloud state vectors and their adiabatic profiles.

    Parameters
    ----------
    n_samples : int
        Number of profiles to generate.
    n_levels : int
        Number of vertical levels.
    r_top_range : tuple
        (min, max) for cloud top effective radius (μm).
    r_bot_range : tuple
        (min, max) for cloud base effective radius (μm).
        If max is None, it defaults to r_top for each sample (adiabatic constraint).
    tau_range : tuple
        (min, max) for cloud optical depth.
    include_subadiabatic : bool
        If True, add Gaussian noise to profiles to simulate sub-adiabatic
        deviations (entrainment, etc.). This helps the network generalize
        beyond strictly adiabatic clouds.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    data : dict with keys:
        'profiles'  : (n_samples, n_levels) — r_e profiles, top to base
        'tau_c'     : (n_samples,) — cloud optical depth
        'r_top'     : (n_samples,) — cloud top effective radius
        'r_bot'     : (n_samples,) — cloud base effective radius
    """
    if rng is None:
        rng = np.random.default_rng(42)

    r_top = rng.uniform(r_top_range[0], r_top_range[1], size=n_samples)

    # r_bot must be less than r_top (adiabatic constraint)
    r_bot_max = r_top  # enforce r_bot < r_top
    r_bot_min = np.full(n_samples, r_bot_range[0])
    r_bot = rng.uniform(r_bot_min, r_bot_max)

    tau_c = rng.uniform(tau_range[0], tau_range[1], size=n_samples)

    # Generate adiabatic profiles
    profiles = np.zeros((n_samples, n_levels))
    for i in range(n_samples):
        profiles[i] = adiabatic_profile(r_top[i], r_bot[i], n_levels)

    # Optionally add sub-adiabatic perturbations
    if include_subadiabatic:
        # Add small Gaussian noise, scaling with depth into cloud
        # (more variability deeper in cloud, less at top where we have signal)
        depth_weight = np.linspace(0.2, 1.0, n_levels)  # more noise at base
        noise_scale = 0.3  # μm standard deviation at cloud base
        noise = rng.normal(0, noise_scale, size=(n_samples, n_levels)) * depth_weight

        profiles = profiles + noise

        # Enforce physical bounds after adding noise
        profiles = np.clip(profiles, RE_MIN, RE_MAX)

    return {
        'profiles': profiles.astype(np.float32),
        'tau_c': tau_c.astype(np.float32),
        'r_top': r_top.astype(np.float32),
        'r_bot': r_bot.astype(np.float32),
    }


# =============================================================================
# HDF5 Dataset for libRadtran training data
# =============================================================================

class LibRadtranDataset(Dataset):
    """
    PyTorch Dataset for libRadtran-generated training data stored in HDF5.

    Expected HDF5 structure:
        /reflectances   : (n_samples, n_wavelengths) — TOA reflectance spectra
        /profiles       : (n_samples, n_levels) — r_e profiles (top to base),
                          interpolated to n_levels evenly-spaced altitude levels
        /tau_c          : (n_samples,) — cloud optical depth
        /sza            : (n_samples,) — solar zenith angle (degrees)
        /vza            : (n_samples,) — viewing zenith angle (degrees)
        /vaz            : (n_samples,) — viewing azimuth angle (degrees)
        /saz            : (n_samples,) — solar azimuth angle (degrees)
        /wavelengths    : (n_wavelengths,) — wavelength grid (nm)

    The geometry inputs are concatenated in the order [SZA, VZA, SAZ, VAZ],
    matching the n_geometry_inputs=4 convention in RetrievalConfig.

    Create this HDF5 file by running convert_matFiles_to_HDF.py.
    """

    def __init__(self, h5_path: str, normalize: bool = True):
        """
        Parameters
        ----------
        h5_path : str
            Path to HDF5 file containing libRadtran data.
        normalize : bool
            If True, normalize inputs and targets to [0, 1] range.
        """
        self.h5_path = Path(h5_path)
        self.normalize = normalize

        # Load data into memory (for datasets that fit in RAM)
        # For very large datasets, consider lazy loading with __getitem__
        with h5py.File(self.h5_path, 'r') as f:
            self.reflectances = f['reflectances'][:].astype(np.float32)
            self.profiles = f['profiles'][:].astype(np.float32)
            self.tau_c = f['tau_c'][:].astype(np.float32)

            # Geometry inputs (if present)
            self.sza = f['sza'][:].astype(np.float32) if 'sza' in f else None
            self.vza = f['vza'][:].astype(np.float32) if 'vza' in f else None
            self.vaz = f['vaz'][:].astype(np.float32) if 'vaz' in f else None
            self.saz = f['saz'][:].astype(np.float32) if 'saz' in f else None

            # Metadata
            self.wavelengths = f['wavelengths'][:] if 'wavelengths' in f else None

        self.n_samples = self.reflectances.shape[0]
        self.n_wavelengths = self.reflectances.shape[1]
        self.n_levels = self.profiles.shape[1]

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization()

    def _compute_normalization(self):
        """Compute min/max for input and target normalization."""
        # Input normalization: reflectances are already in [0, ~0.8] range
        self.refl_min = self.reflectances.min(axis=0)
        self.refl_max = self.reflectances.max(axis=0)

        # Target normalization
        self.re_min = RE_MIN
        self.re_max = RE_MAX
        self.tau_min = TAU_MIN
        self.tau_max = TAU_MAX

        # Geometry normalization ranges
        # SZA range kept wide for future datasets with non-zero SZA
        self.sza_range = (0.0, 80.0)    # degrees
        self.vza_range = (0.0, 65.0)    # degrees; current dataset: 0–65
        self.vaz_range = (0.0, 180.0)   # degrees; viewing azimuth
        self.saz_range = (0.0, 180.0)   # degrees; solar azimuth

    def _normalize_input(self, refl, sza=None, vza=None, saz=None, vaz=None):
        """
        Normalize inputs to approximately [0, 1].

        Geometry inputs are appended in the order [SZA, VZA, SAZ, VAZ],
        matching the n_geometry_inputs=4 convention in RetrievalConfig.
        """
        refl_norm = (refl - self.refl_min) / (self.refl_max - self.refl_min + 1e-8)

        parts = [refl_norm]
        if sza is not None:
            parts.append(np.array([(sza - self.sza_range[0]) / (self.sza_range[1] - self.sza_range[0])]))
        if vza is not None:
            parts.append(np.array([(vza - self.vza_range[0]) / (self.vza_range[1] - self.vza_range[0])]))
        if saz is not None:
            parts.append(np.array([(saz - self.saz_range[0]) / (self.saz_range[1] - self.saz_range[0])]))
        if vaz is not None:
            parts.append(np.array([(vaz - self.vaz_range[0]) / (self.vaz_range[1] - self.vaz_range[0])]))

        return np.concatenate(parts).astype(np.float32)

    def _normalize_target(self, profile, tau_c):
        """Normalize targets to [0, 1] using physical bounds."""
        profile_norm = (profile - self.re_min) / (self.re_max - self.re_min)
        tau_norm = (tau_c - self.tau_min) / (self.tau_max - self.tau_min)
        return profile_norm.astype(np.float32), np.float32(tau_norm)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        refl = self.reflectances[idx]
        profile = self.profiles[idx]
        tau_c = self.tau_c[idx]

        sza = self.sza[idx] if self.sza is not None else None
        vza = self.vza[idx] if self.vza is not None else None
        saz = self.saz[idx] if self.saz is not None else None
        vaz = self.vaz[idx] if self.vaz is not None else None

        if self.normalize:
            x = self._normalize_input(refl, sza, vza, saz, vaz)
            profile_norm, tau_norm = self._normalize_target(profile, tau_c)
        else:
            parts = [refl]
            for g in [sza, vza, saz, vaz]:
                if g is not None:
                    parts.append(np.array([g]))
            x = np.concatenate(parts).astype(np.float32)
            profile_norm = profile
            tau_norm = tau_c

        return (
            torch.from_numpy(x),
            torch.from_numpy(profile_norm),
            torch.tensor(tau_norm),
        )


def create_dataloaders(h5_path: str,
                       batch_size: int = 256,
                       train_frac: float = 0.8,
                       val_frac: float = 0.1,
                       num_workers: int = 4,
                       seed: int = 42,
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 training data file.
    batch_size : int
        Batch size for training.
    train_frac, val_frac : float
        Fractions for train/val split. Test = 1 - train - val.
    num_workers : int
        Number of data loading workers.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    dataset = LibRadtranDataset(h5_path, normalize=True)

    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# =============================================================================
# VOCALS-REx in situ data loading
# =============================================================================

def load_vocals_rex_profile(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load a VOCALS-REx in situ vertical profile from a .mat file.

    This is a placeholder — adapt to your actual VOCALS-REx data format.
    Your existing MATLAB code reads CDP and 2DC data; this function
    should load the processed profiles you've already created.

    Parameters
    ----------
    mat_file_path : str
        Path to .mat file containing the in situ profile.

    Returns
    -------
    data : dict with keys:
        'altitude'  : (n_points,) — altitude within cloud (m)
        'r_eff'     : (n_points,) — effective radius (μm)
        'lwc'       : (n_points,) — liquid water content (g/m³)
        'n_c'       : (n_points,) — droplet number concentration (cm⁻³)
        'tau_insitu': float — in situ derived optical depth
        'lwp_insitu': float — in situ derived liquid water path (g/m²)
    """
    # TODO: Implement based on your VOCALS-REx data format.
    # Options:
    #   - scipy.io.loadmat() for MATLAB .mat files
    #   - h5py for HDF5-based .mat files (v7.3+)
    #   - Direct reading from NCAR EOL archive NetCDF files
    #
    # For now, raise an informative error:
    raise NotImplementedError(
        "Implement this function to load your VOCALS-REx data. "
        "See your existing MATLAB code in the "
        "'hyperspectral-retrieval-using-EMIT' repository for reference."
    )
