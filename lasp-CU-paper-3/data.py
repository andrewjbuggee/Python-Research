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
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings


# =============================================================================
# Constants
# =============================================================================

# MODIS first 7 spectral channels (nm) — used in Papers 1 & 2
MODIS_WAVELENGTHS = np.array([645, 858.5, 469, 555, 1240, 1640, 2130])

# HySICS spectral range in the VOCALS-REx training dataset
# 636 channels spanning ~352–2297 nm (band center = mean of lower/upper bounds
# stored in columns 4 & 5 of changing_variables_allStateVectors in each .mat file).
# NOTE: The HDF5 file created before 12-Apr-2026 used wrong columns [3,4] and
#       stored 166–1138 nm; patch_hdf5_wavelengths.py corrects that in-place.
HYSICS_WAV_MIN = 352.0    # nm  (band 1 center: ~351.95 nm)
HYSICS_WAV_MAX = 2297.0   # nm  (band 636 center: ~2297.05 nm)

# Number of vertical levels for profile retrieval.
# Only used by the synthetic-profile code path (adiabatic_profile /
# generate_random_profiles).  The real training pipeline reads n_levels
# directly from the HDF5 shape (RetrievalDataset.__init__ in this file,
# line ~256), so this constant does not affect model training.
# Set to 7 to match the current convert_matFiles_to_HDF.py N_LEVELS.
DEFAULT_N_LEVELS = 7

# ERA5 water vapour column normalization bounds (log10 scale, molec/cm²)
# Only used when use_era5_profile=False (legacy 2-scalar mode).
# Run patch_hdf5_era5.py and read the printed summary to validate these bounds
# against your actual dataset.  Conservative defaults are set here; if
# observed log10 values fall outside [MIN, MAX] the normalized input will
# lie outside [0, 1], which will trigger a runtime warning in EmulatorDataset.
WV_ABOVE_LOG10_MIN = 21.0   # log10(molec/cm²) — very dry above-cloud column
WV_ABOVE_LOG10_MAX = 24.0   # log10(molec/cm²) — very moist above-cloud column
WV_IN_LOG10_MIN    = 20.0   # log10(molec/cm²) — very thin / dry in-cloud layer
WV_IN_LOG10_MAX    = 23.0   # log10(molec/cm²) — thick / moist in-cloud layer

# ERA5 full vapor-concentration profile normalization (molecules/cm³, log10 scale).
# Surface values reach ~3–5 × 10^17 molec/cm³; upper troposphere/stratosphere
# values approach zero.  log10(v + 1.0) maps the range to [0, ~17.5]; dividing
# by ERA5_VAP_LOG10_MAX normalises to [0, 1].
ERA5_VAP_LOG10_MAX = 18.0   # log10(molec/cm³) — conservative upper bound

# Physical bounds on effective radius (μm)
# Update RE_MAX after running convert_matFiles_to_HDF.py — the scan pass
# will print the observed max across all in-situ profiles.
RE_MIN = 1.5
RE_MAX = 50.0   # TODO: update from scan

# Optical depth bounds
# Update TAU_MAX after running convert_matFiles_to_HDF.py scan.
TAU_MIN = 3   # in-situ profiles include sub-cloud layers where tau=0
TAU_MAX = 75.0  # TODO: update from scan


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

    def __init__(self, h5_path: str, normalize: bool = True,
                 instrument: str = 'hysics'):
        """
        Parameters
        ----------
        h5_path : str
            Path to HDF5 file containing libRadtran data.
        normalize : bool
            If True, normalize inputs and targets to [0, 1] range.
        instrument : str
            Which instrument's simulated reflectance to use as model input.
            'hysics' (default) — 0.3 % Gaussian noise (HySICS noise level)
            'emit'             — 4 % Gaussian noise (EMIT noise level)
            The underlying RT simulation is identical; only the noise differs.
        """
        if instrument not in ('hysics', 'emit'):
            raise ValueError(f"instrument must be 'hysics' or 'emit', got {instrument!r}")

        self.h5_path   = Path(h5_path)
        self.normalize = normalize
        self.instrument = instrument

        # Determine which HDF5 keys to load based on instrument selection.
        # New files store per-instrument datasets; fall back to the legacy
        # 'reflectances' key for HDF5 files created before this change.
        refl_key   = f'reflectances_{instrument}'
        uncert_key = f'reflectances_uncertainty_{instrument}'

        # Load data into memory (for datasets that fit in RAM)
        # For very large datasets, consider lazy loading with __getitem__
        with h5py.File(self.h5_path, 'r') as f:
            if refl_key in f:
                self.reflectances = f[refl_key][:].astype(np.float32)
            else:
                # Legacy HDF5 file — single reflectance dataset
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
                       instrument: str = 'hysics',
                       profile_holdout: bool = False,
                       n_val_profiles: int = 14,
                       n_test_profiles: int = 14,
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
        Only used when profile_holdout=False.
    num_workers : int
        Number of data loading workers.
    seed : int
        Random seed for reproducible splits.
    instrument : str
        Which instrument's reflectance to use: 'hysics' or 'emit'.
        See LibRadtranDataset for details.
    profile_holdout : bool
        If True, use profile-aware splits so that val/test sets contain
        cloud profiles NEVER seen during training.  This is strongly
        recommended — it uses the same split as
        create_emulator_dataloaders() for consistency.
    n_val_profiles, n_test_profiles : int
        Number of unique profiles reserved for val/test when
        profile_holdout=True.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    dataset = LibRadtranDataset(h5_path, normalize=True, instrument=instrument)

    if profile_holdout:
        train_idx, val_idx, test_idx = create_profile_aware_splits(
            h5_path,
            n_val_profiles=n_val_profiles,
            n_test_profiles=n_test_profiles,
            seed=seed,
        )
        train_set = Subset(dataset, train_idx)
        val_set   = Subset(dataset, val_idx)
        test_set  = Subset(dataset, test_idx)

        split_mode = "profile-held-out"
        print(f"  Split mode: {split_mode}")
        print(f"  Val profiles (never in train):  {n_val_profiles}")
        print(f"  Test profiles (never in train): {n_test_profiles}")
    else:
        n = len(dataset)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val

        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        split_mode = "random sample-level"
        print(f"  Split mode: {split_mode}")

    # pin_memory speeds up CPU→GPU transfers but is not supported on MPS
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

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


# =============================================================================
# Profile-aware splitting utilities
# =============================================================================

def compute_profile_ids(h5_path: str) -> np.ndarray:
    """
    Assign an integer profile ID (0..N_unique-1) to every sample by
    fingerprinting the raw in-situ profiles stored in the HDF5 file.

    The training dataset is generated from a set of unique measured vertical
    profiles (73 from VOCALS-REx, 63 from ORACLES as of April 2026), each
    simulated under many solar/viewing geometries.  This function identifies
    which unique profile each sample came from, enabling profile-held-out
    train/val/test splits.  The total profile count is inferred automatically
    from the data — no hardcoded value needed.

    Returns
    -------
    profile_ids : np.ndarray, shape (n_samples,), dtype int32
        Integer profile ID for each sample (0-indexed, consistent across calls).
    """
    with h5py.File(h5_path, 'r') as f:
        profiles_raw = f['profiles_raw'][:].astype(np.float32)

    # Fingerprint by rounding the first 5 raw-profile values to 4 decimal places.
    # Sufficient to distinguish all unique profiles across VOCALS-REx and ORACLES.
    fingerprints = np.round(profiles_raw[:, :5], 4)
    _, inverse = np.unique(fingerprints, axis=0, return_inverse=True)
    return inverse.astype(np.int32)


def create_profile_aware_splits(
        h5_path: str,
        n_val_profiles: int = 14,
        n_test_profiles: int = 14,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sample indices so that held-out profiles NEVER appear in training.

    Example with 200 unique profiles and n_val=14, n_test=14:
        Train :  172 profiles  ( 86% of samples)
        Val   :   14 profiles  (  7% of samples)
        Test  :   14 profiles  (  7% of samples)
    Profile counts are inferred from the HDF5 data at runtime.

    Why this matters
    ----------------
    Each spectrum is one of many geometries run on the same in-situ droplet
    profile.  A random sample-level split puts every profile into every split —
    the model effectively sees every cloud structure during training.  A
    profile-aware split gives a true measure of generalisation to unseen cloud
    vertical structures.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 training data file.
    n_val_profiles : int
        Number of unique profiles reserved for validation.
    n_test_profiles : int
        Number of unique profiles reserved for test (never exposed to model).
    seed : int
        Random seed for reproducible profile assignment.  Use the same seed
        across retrieval and emulator training for consistency.

    Returns
    -------
    train_indices, val_indices, test_indices : np.ndarray of int
    """
    profile_ids = compute_profile_ids(h5_path)
    n_unique = int(profile_ids.max()) + 1

    rng = np.random.default_rng(seed)
    shuffled = np.arange(n_unique)
    rng.shuffle(shuffled)

    test_set = set(shuffled[:n_test_profiles].tolist())
    val_set  = set(shuffled[n_test_profiles:n_test_profiles + n_val_profiles].tolist())

    train_mask = np.array([pid not in test_set and pid not in val_set
                           for pid in profile_ids])
    val_mask   = np.array([pid in val_set  for pid in profile_ids])
    test_mask  = np.array([pid in test_set for pid in profile_ids])

    return (np.where(train_mask)[0],
            np.where(val_mask)[0],
            np.where(test_mask)[0])


def _fit_pca(log_refl: np.ndarray, n_components: int) -> tuple:
    """
    Fit PCA on log-reflectances using covariance eigendecomposition.

    Operates on the (n_samples, n_wavelengths) log-reflectance matrix.
    Uses the covariance matrix approach (n_wavelengths × n_wavelengths = 636×636)
    rather than full SVD on the n_samples × n_wavelengths matrix — much faster
    for large datasets and gives the same components.

    Parameters
    ----------
    log_refl : (n_samples, n_wavelengths) float32
        Log10-transformed reflectances: log10(R + log_eps).
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    mean       : (n_wavelengths,) float32 — per-channel mean of log-reflectances
    components : (n_components, n_wavelengths) float32 — PCA loading vectors
    explained  : (n_components,) float64 — fraction of variance explained by each PC
    """
    mean = log_refl.mean(axis=0).astype(np.float32)               # (n_wl,)
    X_c  = (log_refl - mean).astype(np.float64)                    # centred
    cov  = (X_c.T @ X_c) / max(len(X_c) - 1, 1)                  # (n_wl, n_wl)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)               # ascending order
    idx  = np.argsort(eigenvalues)[::-1]                           # descending
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]                            # columns = PCs
    components   = eigenvectors[:, :n_components].T.astype(np.float32)  # (K, n_wl)
    total_var    = eigenvalues.sum()
    explained    = eigenvalues[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
    return mean, components, explained


# =============================================================================
# Emulator Dataset
# =============================================================================

class EmulatorDataset(Dataset):
    """
    Dataset for training the ForwardModelEmulator.

    Inputs  (x): normalized (r_e profile | τ_c | geometry | wv_above | wv_in)
                 — 17 features (15 cloud+geometry + 2 ERA5 water vapour scalars)
    Outputs (y): either log-reflectance spectrum (636 values) or PCA scores
                 (n_pca_components values) when n_pca_components > 0

    This uses the same HDF5 file as LibRadtranDataset but flips the role of
    inputs and outputs: the spectral measurements are the *target* to predict,
    and the cloud state + geometry are the *inputs*.

    Normalization
    -------------
    Inputs are normalized with the same fixed physical bounds used by
    LibRadtranDataset, so the normalized values are numerically identical
    to those produced by the retrieval network's output heads.  This is
    critical for Stage 2: the emulator receives the retrieval network's
    normalized outputs directly.

    Log-reflectance mode (log_reflectance=True, default)
    -----------------------------------------------------
    TOA reflectances span ~4 orders of magnitude: bright continuum channels
    reach ~0.8 while deep H₂O / O₂ absorption channels can be < 1e-4.  Plain
    MSE in linear space is numerically dominated by the bright channels, leaving
    absorption-band channels undertrained.

    Setting log_reflectance=True transforms each target channel before the loss:
        y = log10(R + log_eps)
    where log_eps is a small floor (default 1e-6) to stabilise log(0).  The
    model then predicts log10(R̂ + log_eps) and plain MSE is used as the loss.
    In log-space MSE treats bright and dark channels on equal footing.

    MRE and per-channel diagnostics are always reported in *linear* reflectance
    space so results are physically interpretable:
        R̂ = 10^(pred)
        R  = 10^(target)

    For Stage 2 PINN use, call model.reconstruct(pred) to invert PCA and
    exponentiate back to linear reflectance space before computing the
    spectral data fidelity loss.

    PCA output decomposition (n_pca_components > 0, recommended)
    -------------------------------------------------------------
    Instead of predicting all 636 reflectance channels independently, the
    model predicts N_PCA principal-component scores.  PCA is fit on the
    training-set log10-reflectances (log10(R + log_eps)), and targets are
    transformed to scores before the loss:
        y_pca = (log10(R + log_eps) - mean) @ components.T   (N_PCA,)
    The loss (MSE on scores) is well-conditioned because all scores have
    similar variance.  Reconstruction at inference:
        log10(R̂) = scores @ components + mean
        R̂        = 10^(log10(R̂))
    PCA forces the model to predict spectrally coherent, smooth outputs
    (high-frequency channel-to-channel noise cannot be fit) and reduces
    the output dimensionality from 636 → N_PCA, making training easier.

    Profile-held-out splits
    -----------------------
    Pass a pre-computed index array from create_profile_aware_splits() so
    that the test split contains profiles never seen during training.

    LHC augmentation
    ----------------
    Pass lhc_h5_path to append LHC-sampled RT simulations to the training
    data.  The LHC file must follow the same HDF5 schema as the main dataset
    and must already contain simulated reflectances.  Only pass lhc_h5_path
    for the training split — val/test use only real VOCALS-REx profiles.
    """

    # Physical normalization bounds — must exactly match LibRadtranDataset / RetrievalConfig
    _RE_MIN  = RE_MIN    # 1.25 μm
    _RE_MAX  = RE_MAX    # 50.0 μm
    _TAU_MIN = TAU_MIN   # 1.5
    _TAU_MAX = TAU_MAX   # 40.0

    # Geometry ranges — must exactly match LibRadtranDataset._compute_normalization()
    _SZA_RANGE = (0.0, 80.0)
    _VZA_RANGE = (0.0, 65.0)
    _SAZ_RANGE = (0.0, 180.0)
    _VAZ_RANGE = (0.0, 180.0)

    # ERA5 water vapour normalization (log10 space, molec/cm²) — legacy 2-scalar mode
    _WV_ABOVE_LOG10_MIN = WV_ABOVE_LOG10_MIN
    _WV_ABOVE_LOG10_MAX = WV_ABOVE_LOG10_MAX
    _WV_IN_LOG10_MIN    = WV_IN_LOG10_MIN
    _WV_IN_LOG10_MAX    = WV_IN_LOG10_MAX

    def __init__(self,
                 h5_path: str,
                 indices: Optional[np.ndarray] = None,
                 instrument: str = 'hysics',
                 lhc_h5_path: Optional[str] = None,
                 log_reflectance: bool = True,
                 log_eps: float = 1e-6,
                 use_era5_profile: bool = False,
                 n_pca_components: int = 0,
                 pca_mean: Optional[np.ndarray] = None,
                 pca_components: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        h5_path : str
            Path to the main HDF5 training data file.
        indices : np.ndarray, optional
            Sample indices to include (from create_profile_aware_splits).
            If None, all samples are used.
        instrument : str
            'hysics' or 'emit' — which simulated reflectances to use as targets.
        lhc_h5_path : str, optional
            Path to a libRadtran-completed LHC augmentation HDF5 file.
            Must contain the same keys as the main file including reflectances.
            Only use for training split — do not augment val/test.
        log_reflectance : bool
            If True (default), targets are returned as log10(R + log_eps).
            Use plain nn.MSELoss() with this mode; do NOT use DWMSELoss
            (DWMSE weights by magnitude, which is counterproductive in log space).
        log_eps : float
            Small floor added before log10 to avoid log(0).  Default 1e-6
            (well below any physical reflectance, so it barely affects bright
            channels while stabilising deep absorption-band values near zero).
        use_era5_profile : bool
            If True, use the full 37-level ERA5 vapor concentration profile.
            If False (default), use 2-scalar (wv_above_cloud, wv_in_cloud).
        n_pca_components : int
            If > 0, fit PCA on this dataset's log-reflectances and return
            PCA scores as targets.  Must be 0 for val/test if training set
            has n_pca_components > 0 (pass pca_mean and pca_components instead).
        pca_mean : (n_wavelengths,) float32 or None
            Per-channel log-reflectance mean from training set PCA.
            If provided (together with pca_components), uses these stats
            rather than computing new ones — use for val/test datasets.
        pca_components : (n_pca_components, n_wavelengths) float32 or None
            PCA loading matrix from training set. If provided, n_pca_components
            is inferred from the shape.
        """
        if instrument not in ('hysics', 'emit'):
            raise ValueError(f"instrument must be 'hysics' or 'emit', got {instrument!r}")

        refl_key = f'reflectances_{instrument}'

        with h5py.File(h5_path, 'r') as f:
            all_refl = (f[refl_key][:] if refl_key in f
                        else f['reflectances'][:]).astype(np.float32)
            all_profiles  = f['profiles'][:].astype(np.float32)
            all_tau_c     = f['tau_c'][:].astype(np.float32)
            all_sza       = f['sza'][:].astype(np.float32)
            all_vza       = f['vza'][:].astype(np.float32)
            all_saz       = f['saz'][:].astype(np.float32)
            all_vaz       = f['vaz'][:].astype(np.float32)
            self.wavelengths = (f['wavelengths'][:].astype(np.float32)
                                if 'wavelengths' in f else None)

            if use_era5_profile:
                # Full 37-level vapor concentration profile.
                # Directly encodes the vertical WV distribution needed to
                # predict absorption-band (1.38 μm, 1.9 μm) reflectances.
                if 'era5_vapor_concentration' not in f:
                    raise KeyError(
                        "HDF5 file is missing 'era5_vapor_concentration'. "
                        "Re-run convert_matFiles_to_HDF.py to generate it."
                    )
                all_era5_vap = f['era5_vapor_concentration'][:].astype(np.float32)
                all_wv_above = None
                all_wv_in    = None
            else:
                # Legacy 2-scalar mode — integrated WV columns only.
                if 'wv_above_cloud' not in f or 'wv_in_cloud' not in f:
                    raise KeyError(
                        "HDF5 file is missing ERA5 water vapour datasets "
                        "('wv_above_cloud', 'wv_in_cloud'). "
                        "Run patch_hdf5_era5.py to add them."
                    )
                all_wv_above = f['wv_above_cloud'][:].astype(np.float64)
                all_wv_in    = f['wv_in_cloud'][:].astype(np.float64)
                all_era5_vap = None

        if indices is not None:
            all_refl     = all_refl[indices]
            all_profiles = all_profiles[indices]
            all_tau_c    = all_tau_c[indices]
            all_sza      = all_sza[indices]
            all_vza      = all_vza[indices]
            all_saz      = all_saz[indices]
            all_vaz      = all_vaz[indices]
            if all_era5_vap is not None:
                all_era5_vap = all_era5_vap[indices]
            if all_wv_above is not None:
                all_wv_above = all_wv_above[indices]
                all_wv_in    = all_wv_in[indices]

        # Optional LHC augmentation — training split only
        if lhc_h5_path is not None:
            with h5py.File(lhc_h5_path, 'r') as f:
                if refl_key not in f and 'reflectances' not in f:
                    raise ValueError(
                        f"LHC file {lhc_h5_path!r} has no reflectance key '{refl_key}'. "
                        "Run libRadtran on the LHC parameters first, then add reflectances."
                    )
                lhc_refl     = (f[refl_key][:] if refl_key in f
                                else f['reflectances'][:]).astype(np.float32)
                lhc_profiles = f['profiles'][:].astype(np.float32)
                lhc_tau_c    = f['tau_c'][:].astype(np.float32)
                lhc_sza      = f['sza'][:].astype(np.float32)
                lhc_vza      = f['vza'][:].astype(np.float32)
                lhc_saz      = f['saz'][:].astype(np.float32)
                lhc_vaz      = f['vaz'][:].astype(np.float32)

                if use_era5_profile:
                    if 'era5_vapor_concentration' in f:
                        lhc_era5_vap = f['era5_vapor_concentration'][:].astype(np.float32)
                    else:
                        warnings.warn(
                            "LHC file has no era5_vapor_concentration. "
                            "Filling with training-set column means per level. "
                            "Re-run convert_matFiles_to_HDF.py on the LHC data for best results."
                        )
                        lhc_era5_vap = np.tile(all_era5_vap.mean(axis=0),
                                               (len(lhc_refl), 1))
                    lhc_wv_above = None
                    lhc_wv_in    = None
                else:
                    if 'wv_above_cloud' in f:
                        lhc_wv_above = f['wv_above_cloud'][:].astype(np.float64)
                        lhc_wv_in    = f['wv_in_cloud'][:].astype(np.float64)
                    else:
                        warnings.warn(
                            "LHC file has no ERA5 water vapour data. "
                            "Filling wv_above_cloud / wv_in_cloud with training-set means. "
                            "Run patch_hdf5_era5.py on the LHC file for correct values."
                        )
                        lhc_wv_above = np.full(len(lhc_refl), all_wv_above.mean())
                        lhc_wv_in    = np.full(len(lhc_refl), all_wv_in.mean())
                    lhc_era5_vap = None

            all_refl     = np.concatenate([all_refl,     lhc_refl],     axis=0)
            all_profiles = np.concatenate([all_profiles, lhc_profiles], axis=0)
            all_tau_c    = np.concatenate([all_tau_c,    lhc_tau_c],    axis=0)
            all_sza      = np.concatenate([all_sza,      lhc_sza],      axis=0)
            all_vza      = np.concatenate([all_vza,      lhc_vza],      axis=0)
            all_saz      = np.concatenate([all_saz,      lhc_saz],      axis=0)
            all_vaz      = np.concatenate([all_vaz,      lhc_vaz],      axis=0)
            if use_era5_profile:
                all_era5_vap = np.concatenate([all_era5_vap, lhc_era5_vap], axis=0)
            else:
                all_wv_above = np.concatenate([all_wv_above, lhc_wv_above], axis=0)
                all_wv_in    = np.concatenate([all_wv_in,    lhc_wv_in],    axis=0)

        # Store arrays
        self.reflectances    = all_refl       # targets — raw, not normalized
        self.profiles        = all_profiles
        self.tau_c           = all_tau_c
        self.sza             = all_sza
        self.vza             = all_vza
        self.saz             = all_saz
        self.vaz             = all_vaz
        self.use_era5_profile = use_era5_profile
        self.era5_vapor      = all_era5_vap   # (n, 37) float32 or None
        self.wv_above        = all_wv_above   # above-cloud WV column (molec/cm²) or None
        self.wv_in           = all_wv_in      # in-cloud WV column (molec/cm²) or None

        # Warn if legacy 2-scalar WV values fall outside normalization bounds.
        if not use_era5_profile:
            log_above = np.log10(np.maximum(all_wv_above, 1e10))
            log_in    = np.log10(np.maximum(all_wv_in,    1e10))
            if log_above.min() < self._WV_ABOVE_LOG10_MIN or \
               log_above.max() > self._WV_ABOVE_LOG10_MAX:
                warnings.warn(
                    f"wv_above_cloud log10 range [{log_above.min():.2f}, "
                    f"{log_above.max():.2f}] exceeds normalization bounds "
                    f"[{self._WV_ABOVE_LOG10_MIN}, {self._WV_ABOVE_LOG10_MAX}]. "
                    "Update WV_ABOVE_LOG10_MIN/MAX in data.py."
                )
            if log_in.min() < self._WV_IN_LOG10_MIN or \
               log_in.max() > self._WV_IN_LOG10_MAX:
                warnings.warn(
                    f"wv_in_cloud log10 range [{log_in.min():.2f}, "
                    f"{log_in.max():.2f}] exceeds normalization bounds "
                    f"[{self._WV_IN_LOG10_MIN}, {self._WV_IN_LOG10_MAX}]. "
                    "Update WV_IN_LOG10_MIN/MAX in data.py."
                )

        self.log_reflectance = log_reflectance
        self.log_eps         = log_eps

        # --- PCA output decomposition ---
        if pca_mean is not None and pca_components is not None:
            # Val/test: use training-set PCA parameters directly
            self.pca_mean       = pca_mean.astype(np.float32)
            self.pca_components = pca_components.astype(np.float32)
            self.n_pca_components = int(pca_components.shape[0])
        elif n_pca_components > 0 and log_reflectance:
            # Training set: fit PCA on this dataset's log-reflectances
            log_refl = np.log10(np.maximum(all_refl, self.log_eps))
            mean, comps, explained = _fit_pca(log_refl, n_pca_components)
            self.pca_mean         = mean
            self.pca_components   = comps
            self.n_pca_components = n_pca_components
            cum_var = np.cumsum(explained)
            print(f"  PCA: {n_pca_components} components explain "
                  f"{100*cum_var[-1]:.2f}% of log-reflectance variance")
        else:
            self.pca_mean         = None
            self.pca_components   = None
            self.n_pca_components = 0

        self.n_samples     = self.reflectances.shape[0]
        self.n_wavelengths = self.reflectances.shape[1]

    def _normalize_inputs(self, profile, tau_c, sza, vza, saz, vaz,
                          era5_vapor=None,
                          wv_above=None, wv_in=None) -> np.ndarray:
        """
        Normalize all emulator inputs to [0, 1] using fixed physical bounds.

        Cloud + geometry (15 values): same linear bounds as LibRadtranDataset
        so emulator inputs match the retrieval network's output numerically.

        Atmospheric inputs — two modes:

        use_era5_profile=True (default, 37 values):
            Full ERA5 vapor concentration profile.  Each level is normalized as
                log10(v + 1.0) / ERA5_VAP_LOG10_MAX  ∈ [0, 1]
            This encodes the vertical WV distribution needed to predict spectral
            shape within H₂O absorption bands (1.38 μm, 1.9 μm).

        use_era5_profile=False (legacy, 2 values):
            Integrated above-cloud and in-cloud WV column densities,
            log10-transformed with fixed bounds.
        """
        profile_norm = ((profile - self._RE_MIN)
                        / (self._RE_MAX - self._RE_MIN)).astype(np.float32)
        tau_norm = np.float32((tau_c - self._TAU_MIN)
                              / (self._TAU_MAX - self._TAU_MIN))
        sza_norm = np.float32((sza - self._SZA_RANGE[0])
                              / (self._SZA_RANGE[1] - self._SZA_RANGE[0]))
        vza_norm = np.float32((vza - self._VZA_RANGE[0])
                              / (self._VZA_RANGE[1] - self._VZA_RANGE[0]))
        saz_norm = np.float32((saz - self._SAZ_RANGE[0])
                              / (self._SAZ_RANGE[1] - self._SAZ_RANGE[0]))
        vaz_norm = np.float32((vaz - self._VAZ_RANGE[0])
                              / (self._VAZ_RANGE[1] - self._VAZ_RANGE[0]))

        if era5_vapor is not None:
            # Full 37-level profile: log10(v + 1) / LOG10_MAX → [0, 1]
            # Adding 1 before log10 ensures log10(0) = 0 (not -inf) for
            # very dry levels near the top of the atmosphere.
            vap_norm = (np.log10(era5_vapor.astype(np.float64) + 1.0)
                        / self._ERA5_VAP_LOG10_MAX).clip(0.0, 1.1).astype(np.float32)
            atm_part = vap_norm  # (37,)
        else:
            # Legacy 2-scalar mode
            log_above = np.log10(max(float(wv_above), 1e10))
            log_in    = np.log10(max(float(wv_in),    1e10))
            wv_above_norm = np.float32(
                (log_above - self._WV_ABOVE_LOG10_MIN)
                / (self._WV_ABOVE_LOG10_MAX - self._WV_ABOVE_LOG10_MIN)
            )
            wv_in_norm = np.float32(
                (log_in - self._WV_IN_LOG10_MIN)
                / (self._WV_IN_LOG10_MAX - self._WV_IN_LOG10_MIN)
            )
            atm_part = np.array([wv_above_norm, wv_in_norm], dtype=np.float32)

        return np.concatenate([
            profile_norm,
            np.array([tau_norm], dtype=np.float32),
            np.array([sza_norm, vza_norm, saz_norm, vaz_norm], dtype=np.float32),
            atm_part,
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.use_era5_profile:
            x = self._normalize_inputs(
                self.profiles[idx],
                self.tau_c[idx],
                self.sza[idx],
                self.vza[idx],
                self.saz[idx],
                self.vaz[idx],
                era5_vapor=self.era5_vapor[idx],
            )
        else:
            x = self._normalize_inputs(
                self.profiles[idx],
                self.tau_c[idx],
                self.sza[idx],
                self.vza[idx],
                self.saz[idx],
                self.vaz[idx],
                wv_above=self.wv_above[idx],
                wv_in=self.wv_in[idx],
            )
        y = self.reflectances[idx]   # raw reflectances
        if self.log_reflectance:
            y = np.log10(np.maximum(y, self.log_eps))
            if self.pca_components is not None:
                # Project to PCA score space: (n_wavelengths,) → (n_pca,)
                y = (y - self.pca_mean) @ self.pca_components.T
        return torch.from_numpy(x), torch.from_numpy(y)


def create_emulator_dataloaders(
        h5_path: str,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        instrument: str = 'hysics',
        profile_holdout: bool = True,
        n_val_profiles: int = 10,
        n_test_profiles: int = 10,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        lhc_h5_path: Optional[str] = None,
        log_reflectance: bool = True,
        log_eps: float = 1e-6,
        use_era5_profile: bool = False,
        n_pca_components: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for ForwardModelEmulator training.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 training data.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        DataLoader worker processes.
    seed : int
        Random seed.  Use the same value as create_dataloaders() so that
        retrieval and emulator splits are directly comparable.
    instrument : str
        'hysics' or 'emit'.
    profile_holdout : bool
        If True (strongly recommended), use create_profile_aware_splits() so
        test profiles were never seen during training.  If False, falls back
        to a random sample-level split matching create_dataloaders().
    n_val_profiles, n_test_profiles : int
        Profile counts for val/test when profile_holdout=True.
    train_frac, val_frac : float
        Used only when profile_holdout=False.
    lhc_h5_path : str, optional
        Path to a libRadtran-completed LHC augmentation file.  Appended to
        the training split only.  Ignored if profile_holdout=False.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    if profile_holdout:
        train_idx, val_idx, test_idx = create_profile_aware_splits(
            h5_path,
            n_val_profiles=n_val_profiles,
            n_test_profiles=n_test_profiles,
            seed=seed,
        )
        train_ds = EmulatorDataset(h5_path, indices=train_idx,
                                   instrument=instrument, lhc_h5_path=lhc_h5_path,
                                   log_reflectance=log_reflectance, log_eps=log_eps,
                                   use_era5_profile=use_era5_profile,
                                   n_pca_components=n_pca_components)
        # Val/test reuse training-set PCA components — never refit on held-out data
        val_ds   = EmulatorDataset(h5_path, indices=val_idx,  instrument=instrument,
                                   log_reflectance=log_reflectance, log_eps=log_eps,
                                   use_era5_profile=use_era5_profile,
                                   pca_mean=train_ds.pca_mean,
                                   pca_components=train_ds.pca_components)
        test_ds  = EmulatorDataset(h5_path, indices=test_idx, instrument=instrument,
                                   log_reflectance=log_reflectance, log_eps=log_eps,
                                   use_era5_profile=use_era5_profile,
                                   pca_mean=train_ds.pca_mean,
                                   pca_components=train_ds.pca_components)
    else:
        if lhc_h5_path is not None:
            warnings.warn(
                "lhc_h5_path is ignored when profile_holdout=False. "
                "Enable profile_holdout=True to use LHC augmentation."
            )
        full_ds = EmulatorDataset(h5_path, instrument=instrument,
                                  log_reflectance=log_reflectance, log_eps=log_eps,
                                  use_era5_profile=use_era5_profile,
                                  n_pca_components=n_pca_components)
        n       = len(full_ds)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        n_test  = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed),
        )

    # pin_memory speeds up CPU→GPU transfers but is not supported on MPS
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# =============================================================================
# Latin Hypercube augmentation parameter generation
# =============================================================================

def generate_lhc_parameters(
        n_samples: int,
        output_path: str,
        n_levels: int = DEFAULT_N_LEVELS,
        r_top_range: Tuple[float, float] = (4.0, 20.0),
        r_bot_frac_range: Tuple[float, float] = (0.10, 0.95),
        tau_range: Tuple[float, float] = (TAU_MIN, TAU_MAX),
        sza_range: Tuple[float, float] = (0.0, 75.0),
        vza_range: Tuple[float, float] = (0.0, 65.0),
        saz_range: Tuple[float, float] = (0.0, 180.0),
        vaz_range: Tuple[float, float] = (0.0, 180.0),
        seed: int = 42,
) -> None:
    """
    Generate Latin Hypercube-sampled cloud state + geometry parameters and
    save them to an HDF5 file for running through libRadtran.

    Why LHC augmentation?
    ---------------------
    The 73 K training spectra reflect real-world VOCALS-REx cloud conditions,
    so the (r_e, τ_c) parameter space is biased toward common marine
    stratocumulus states.  The emulator may be less accurate at the edges of
    parameter space (very thin/thick clouds, extreme droplet sizes).  LHC
    sampling ensures uniform coverage of the full physical parameter range
    with no clustering, filling in the gaps.

    Workflow
    --------
    1. Run this function to generate parameter combinations:
           generate_lhc_parameters(2000, "lhc_params.h5")

    2. Load the output file and run each (profile, tau_c, geometry) through
       libRadtran to compute TOA reflectances (use your existing MATLAB RT code).

    3. Add the computed reflectances to the same HDF5 file under keys
       'reflectances_hysics' and/or 'reflectances_emit'.

    4. Pass the completed file as lhc_h5_path to create_emulator_dataloaders().

    Output HDF5 schema (same as main training file, minus reflectances)
    -------------------------------------------------------------------
    /profiles        : (n_samples, n_levels)  adiabatic r_e profiles
    /tau_c           : (n_samples,)
    /sza, /vza, /saz, /vaz : (n_samples,)     geometry angles (degrees)
    /attrs['status'] : 'parameters_only — add reflectances after libRadtran'

    Parameters
    ----------
    n_samples : int
        LHC combinations to generate.  Start with ~2 000; increase to ~10 000
        if emulator error is highest at extremes of parameter space.
    output_path : str
        Where to write the output HDF5 file.
    r_top_range : (min, max)
        Cloud-top effective radius range (μm).
    r_bot_frac_range : (min, max)
        r_bot as a fraction of r_top — enforces r_bot < r_top at all times.
    tau_range : (min, max)
        Cloud optical depth range.
    sza_range, vza_range, saz_range, vaz_range : (min, max)
        Geometry angle ranges (degrees).
    seed : int
        RNG seed for reproducibility.
    """
    try:
        from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
    except ImportError:
        raise ImportError(
            "scipy >= 1.7 is required for Latin Hypercube Sampling.\n"
            "Install it with:  pip install 'scipy>=1.7'"
        )

    # 7-dimensional LHC: r_top, r_bot_frac, tau_c, SZA, VZA, SAZ, VAZ
    sampler     = LatinHypercube(d=7, seed=seed)
    raw_samples = sampler.random(n=n_samples)   # (n_samples, 7) in [0, 1]

    lower = np.array([r_top_range[0],     r_bot_frac_range[0], tau_range[0],
                      sza_range[0],       vza_range[0],        saz_range[0], vaz_range[0]])
    upper = np.array([r_top_range[1],     r_bot_frac_range[1], tau_range[1],
                      sza_range[1],       vza_range[1],        saz_range[1], vaz_range[1]])
    scaled = qmc_scale(raw_samples, lower, upper)   # (n_samples, 7)

    r_top      = scaled[:, 0].astype(np.float32)
    r_bot_frac = scaled[:, 1].astype(np.float32)
    r_bot      = np.maximum(r_bot_frac * r_top, RE_MIN).astype(np.float32)
    tau_c      = scaled[:, 2].astype(np.float32)
    sza        = scaled[:, 3].astype(np.float32)
    vza        = scaled[:, 4].astype(np.float32)
    saz        = scaled[:, 5].astype(np.float32)
    vaz        = scaled[:, 6].astype(np.float32)

    # Build adiabatic 10-level profiles from (r_top, r_bot) pairs
    profiles = np.zeros((n_samples, n_levels), dtype=np.float32)
    for i in range(n_samples):
        profiles[i] = adiabatic_profile(float(r_top[i]), float(r_bot[i]), n_levels)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('profiles', data=profiles)
        f.create_dataset('tau_c',    data=tau_c)
        f.create_dataset('sza',      data=sza)
        f.create_dataset('vza',      data=vza)
        f.create_dataset('saz',      data=saz)
        f.create_dataset('vaz',      data=vaz)
        f.attrs['lhc_n_samples'] = n_samples
        f.attrs['lhc_seed']      = seed
        f.attrs['status']        = (
            'parameters_only — add reflectances_hysics / reflectances_emit '
            'after running libRadtran'
        )

    print(f"LHC parameters saved to: {out_path}")
    print(f"  {n_samples:,} samples | {n_levels} profile levels")
    print(f"  r_top range:  {r_top_range[0]:.1f}–{r_top_range[1]:.1f} μm")
    print(f"  tau_c range:  {tau_range[0]:.1f}–{tau_range[1]:.1f}")
    print(f"  SZA range:    {sza_range[0]:.0f}–{sza_range[1]:.0f}°")
    print()
    print("Next steps:")
    print("  1. Run libRadtran on each (profiles[i], tau_c[i], sza[i], vza[i], saz[i], vaz[i])")
    print("  2. Add results to the HDF5 file:")
    print("       f.create_dataset('reflectances_hysics', data=<array shape (n_samples, 636)>)")
    print("  3. Set lhc_h5_path in emulator.yaml to use this file for training augmentation")
