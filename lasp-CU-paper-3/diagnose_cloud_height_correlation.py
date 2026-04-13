"""
diagnose_cloud_height_correlation.py

Quick diagnostic to test whether emulator error correlates with cloud
top height and geometric thickness — variables not currently provided
as inputs to the ForwardModelEmulator.

If a strong correlation exists, these variables should be added as
inputs to improve accuracy (especially in H2O absorption bands).

Usage
-----
    python diagnose_cloud_height_correlation.py --config emulator.yaml \
        --checkpoint checkpoints/emulator/20260412_173305/best_model.pt

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import ForwardModelEmulator, EmulatorConfig
from data import create_emulator_dataloaders, create_profile_aware_splits


def _split_input(x, n_levels, n_atm=0):
    profile  = x[:, :n_levels]
    tau_c    = x[:, n_levels:n_levels + 1]
    geometry = x[:, n_levels + 1:n_levels + 5]
    atm      = x[:, n_levels + 5:n_levels + 5 + n_atm] if n_atm > 0 else None
    return profile, tau_c, geometry, atm


@torch.no_grad()
def compute_per_sample_mre(model, loader, device, log_reflectance=False,
                           wv_band_only=False, wavelengths=None):
    """
    Compute per-sample MRE (scalar per sample) over the data loader.

    Parameters
    ----------
    wv_band_only : bool
        If True, compute MRE only over the 1.8–2.0 μm water vapor band.
    wavelengths : np.ndarray or None
        Wavelength grid (nm), needed if wv_band_only=True.

    Returns
    -------
    mre_per_sample : np.ndarray, shape (n_samples,)
    """
    model.eval()
    eps = 1e-4
    use_pca = (model.pca_components is not None)
    all_mre = []

    # Build channel mask for water vapor band
    if wv_band_only and wavelengths is not None:
        wv_mask = (wavelengths >= 1800) & (wavelengths <= 2000)
    else:
        wv_mask = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        profile, tau_c, geometry, atm = _split_input(
            x, model.config.n_levels, model.config.n_atm_inputs)
        pred = model(profile, tau_c, geometry, atm)

        if use_pca:
            R_pred = model.reconstruct(pred)
            R_true = model.reconstruct(y)
        elif log_reflectance:
            R_pred = torch.pow(10.0, pred)
            R_true = torch.pow(10.0, y)
        else:
            R_pred = pred
            R_true = y

        rel_err = (R_pred - R_true).abs() / (R_true.abs() + eps)

        if wv_mask is not None:
            wv_mask_t = torch.tensor(wv_mask, device=device)
            rel_err = rel_err[:, wv_mask_t]

        # Mean over wavelengths → one MRE per sample
        sample_mre = rel_err.mean(dim=1).cpu().numpy() * 100.0
        all_mre.append(sample_mre)

    return np.concatenate(all_mre)


def extract_cloud_geometry(h5_path, indices):
    """
    Extract cloud top height and geometric thickness from HDF5 raw profiles.

    Returns
    -------
    cloud_top_km : np.ndarray, shape (n_samples,)
    cloud_thickness_km : np.ndarray, shape (n_samples,)
    """
    with h5py.File(h5_path, 'r') as f:
        z_raw = f['profiles_raw_z'][:].astype(np.float32)  # (n_total, max_levels)

    # Subset to the requested indices
    z_raw = z_raw[indices]

    # Cloud top = max altitude, cloud base = min altitude (ignoring NaN fill)
    cloud_top = np.nanmax(z_raw, axis=1)       # km
    cloud_base = np.nanmin(z_raw, axis=1)      # km
    cloud_thickness = cloud_top - cloud_base    # km

    return cloud_top, cloud_thickness


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose emulator error correlation with cloud geometry")
    parser.add_argument("--config", type=str, default="emulator.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="cloud_height_diagnostic.png")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    h5_path = config['data']['h5_path']
    log_reflectance = config['data'].get('log_reflectance', True)
    n_pca_components = config['data'].get('n_pca_components', 0)
    use_era5_profile = config['data'].get('use_era5_profile', False)

    # --- Get the test split indices ---
    _, _, test_indices = create_profile_aware_splits(
        h5_path,
        n_val_profiles=config['data'].get('n_val_profiles', 10),
        n_test_profiles=config['data'].get('n_test_profiles', 10),
        seed=config['data'].get('seed', 42),
    )
    print(f"Test set: {len(test_indices)} samples")

    # --- Build data loaders (test only needed, but function returns all three) ---
    _, _, test_loader = create_emulator_dataloaders(
        h5_path=h5_path,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 0),
        seed=config['data'].get('seed', 42),
        instrument=config['data'].get('instrument', 'hysics'),
        profile_holdout=True,
        n_val_profiles=config['data'].get('n_val_profiles', 10),
        n_test_profiles=config['data'].get('n_test_profiles', 10),
        log_reflectance=log_reflectance,
        log_eps=float(config['data'].get('log_eps', 1e-6)),
        use_era5_profile=use_era5_profile,
        n_pca_components=n_pca_components,
    )

    # --- Load wavelength grid ---
    with h5py.File(h5_path, 'r') as f:
        wavelengths = f['wavelengths'][:].astype(np.float32)

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    econfig = ckpt.get('emulator_config', {})
    model_cfg = EmulatorConfig(
        n_levels=econfig.get('n_levels', config['model']['n_levels']),
        n_geometry_inputs=econfig.get('n_geometry_inputs',
                                      config['model']['n_geometry_inputs']),
        n_wavelengths_out=econfig.get('n_wavelengths_out',
                                      config['model']['n_wavelengths_out']),
        hidden_dims=econfig.get('hidden_dims', config['model']['hidden_dims']),
        dropout=econfig.get('dropout', config['model']['dropout']),
        activation=econfig.get('activation', config['model']['activation']),
        n_atm_inputs=config['model'].get('n_atm_inputs', 0),
        n_pca_components=config['model'].get('n_pca_components', 0),
    )
    model = ForwardModelEmulator(model_cfg).to(device)

    # Register PCA decoder if used
    if n_pca_components > 0:
        pca_data = test_loader.dataset
        if hasattr(pca_data, 'pca_mean') and pca_data.pca_mean is not None:
            model.register_pca(pca_data.pca_mean, pca_data.pca_components)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Compute per-sample MRE (all channels and H2O band only) ---
    print("Computing per-sample MRE (all channels)...")
    mre_all = compute_per_sample_mre(
        model, test_loader, device, log_reflectance=log_reflectance)

    print("Computing per-sample MRE (1.8–2.0 μm H2O band)...")
    mre_wv = compute_per_sample_mre(
        model, test_loader, device, log_reflectance=log_reflectance,
        wv_band_only=True, wavelengths=wavelengths)

    # --- Extract cloud geometry from HDF5 ---
    print("Extracting cloud top height and geometric thickness...")
    cloud_top, cloud_thickness = extract_cloud_geometry(h5_path, test_indices)

    # --- Correlation statistics ---
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: Emulator Error vs Cloud Geometry")
    print("=" * 60)

    for name, mre in [("All channels", mre_all), ("H2O band (1.8–2.0 μm)", mre_wv)]:
        print(f"\n  --- {name} ---")
        r_top, p_top = stats.pearsonr(cloud_top, mre)
        r_thick, p_thick = stats.pearsonr(cloud_thickness, mre)
        rho_top, p_rho_top = stats.spearmanr(cloud_top, mre)
        rho_thick, p_rho_thick = stats.spearmanr(cloud_thickness, mre)

        print(f"  Cloud top height:")
        print(f"    Pearson  r = {r_top:+.4f}  (p = {p_top:.2e})")
        print(f"    Spearman ρ = {rho_top:+.4f}  (p = {p_rho_top:.2e})")
        print(f"  Cloud geometric thickness:")
        print(f"    Pearson  r = {r_thick:+.4f}  (p = {p_thick:.2e})")
        print(f"    Spearman ρ = {rho_thick:+.4f}  (p = {p_rho_thick:.2e})")

    # --- Also check correlation with tau_c and water vapor ---
    with h5py.File(h5_path, 'r') as f:
        tau_c_all = f['tau_c'][:].astype(np.float32)
        wv_above_all = f['wv_above_cloud'][:].astype(np.float64)
        wv_in_all = f['wv_in_cloud'][:].astype(np.float64)
    tau_c = tau_c_all[test_indices]
    wv_above = wv_above_all[test_indices]
    wv_in = wv_in_all[test_indices]

    print(f"\n  --- Control variables (already in model inputs) ---")
    for vname, v in [("τ_c", tau_c),
                     ("log10(wv_above)", np.log10(wv_above)),
                     ("log10(wv_in)", np.log10(wv_in))]:
        r, p = stats.pearsonr(v, mre_all)
        rho, p_rho = stats.spearmanr(v, mre_all)
        print(f"  {vname} vs all-channel MRE:")
        print(f"    Pearson  r = {r:+.4f}  (p = {p:.2e})")
        print(f"    Spearman ρ = {rho:+.4f}  (p = {p_rho:.2e})")

    # --- Summary statistics of cloud geometry across splits ---
    _, val_indices, _ = create_profile_aware_splits(
        h5_path,
        n_val_profiles=config['data'].get('n_val_profiles', 10),
        n_test_profiles=config['data'].get('n_test_profiles', 10),
        seed=config['data'].get('seed', 42),
    )
    train_mask = np.ones(len(tau_c_all), dtype=bool)
    train_mask[test_indices] = False
    train_mask[val_indices] = False
    train_indices = np.where(train_mask)[0]

    with h5py.File(h5_path, 'r') as f:
        z_raw_full = f['profiles_raw_z'][:].astype(np.float32)
    train_top = np.nanmax(z_raw_full[train_indices], axis=1)
    train_thick = np.nanmax(z_raw_full[train_indices], axis=1) - np.nanmin(z_raw_full[train_indices], axis=1)

    print(f"\n  --- Cloud geometry distribution across splits ---")
    print(f"  {'':>25s}  {'Train':>18s}  {'Test':>18s}")
    print(f"  {'Cloud top (km)':>25s}  {train_top.mean():.3f} ± {train_top.std():.3f}  "
          f"{cloud_top.mean():.3f} ± {cloud_top.std():.3f}")
    print(f"  {'Thickness (km)':>25s}  {train_thick.mean():.3f} ± {train_thick.std():.3f}  "
          f"{cloud_thickness.mean():.3f} ± {cloud_thickness.std():.3f}")
    print(f"  {'Cloud top range (km)':>25s}  [{train_top.min():.3f}, {train_top.max():.3f}]  "
          f"[{cloud_top.min():.3f}, {cloud_top.max():.3f}]")
    print(f"  {'Thickness range (km)':>25s}  [{train_thick.min():.3f}, {train_thick.max():.3f}]  "
          f"[{cloud_thickness.min():.3f}, {cloud_thickness.max():.3f}]")

    # --- Figures ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: all-channel MRE
    for ax, (xlabel, xdata) in zip(axes[0], [
        ("Cloud top height (km)", cloud_top),
        ("Cloud geometric thickness (km)", cloud_thickness),
        ("τ_c", tau_c),
    ]):
        ax.scatter(xdata, mre_all, s=3, alpha=0.3, c='steelblue', rasterized=True)
        r, p = stats.pearsonr(xdata, mre_all)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Per-sample MRE (%)", fontsize=12)
        ax.set_title(f"All channels  |  r = {r:+.3f}  (p = {p:.1e})", fontsize=11)
        # Add trend line
        z = np.polyfit(xdata, mre_all, 1)
        xsort = np.sort(xdata)
        ax.plot(xsort, np.polyval(z, xsort), 'r-', linewidth=2, alpha=0.8)

    # Row 2: H2O band MRE
    for ax, (xlabel, xdata) in zip(axes[1], [
        ("Cloud top height (km)", cloud_top),
        ("Cloud geometric thickness (km)", cloud_thickness),
        ("τ_c", tau_c),
    ]):
        ax.scatter(xdata, mre_wv, s=3, alpha=0.3, c='darkorange', rasterized=True)
        r, p = stats.pearsonr(xdata, mre_wv)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Per-sample MRE (%)", fontsize=12)
        ax.set_title(f"H₂O band (1.8–2.0 μm)  |  r = {r:+.3f}  (p = {p:.1e})", fontsize=11)
        z = np.polyfit(xdata, mre_wv, 1)
        xsort = np.sort(xdata)
        ax.plot(xsort, np.polyval(z, xsort), 'r-', linewidth=2, alpha=0.8)

    fig.suptitle("Emulator Error vs Cloud Geometry (Test Set, Profile-Held-Out)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {args.output}")


if __name__ == '__main__':
    main()
