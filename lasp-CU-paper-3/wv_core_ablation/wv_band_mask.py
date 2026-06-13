"""
wv_band_mask.py — Define the spectral-channel masks for the water-vapor-core
ablation experiment on the published paper-3 network (sweep run_004, variant M0).

Experiment
----------
Permutation importance on run_004 (compute_feature_importance_run004.py) flagged
channels inside several water-vapor absorption bands as informative for the
droplet-size *profile* retrieval — including the 1140 nm and 1900 nm bands where
King & Vaughan reported little information on cloud-top / cloud-base r_e. This
module builds the channel sets for an ablation that physically *removes* those
core channels and retrains, plus a count-matched continuum control.

Two channel sets are produced over the 636 HySICS reflectance channels
(351.95-2297.05 nm, idx 0..635):

  'wv_core'            — the cores of the 940, 1140, 1380, and 1900 nm water-vapor
                         bands. Windows are chosen to remove the central
                         (low-reflectance) absorption core while *keeping* the
                         high-information recovering edges (e.g. 1347/1427 nm and
                         1940-1948 nm), per the user's intent to retain band wings.

  'continuum_control'  — a count-matched set of spectrally redundant continuum
                         channels (neighbor-correlation > 0.999) drawn evenly from
                         five gas-absorption-free atmospheric windows. Removing
                         these isolates the effect of reduced input dimensionality
                         from the effect of losing water-vapor-core information.

Both sets are defined from the *reflectance statistics only* (mean spectrum and
adjacent-channel correlation) — deliberately independent of the permutation
importance the experiment is testing, so the control is not circular.

Note on importance vs. removal: single-channel permutation importance measures
corruption-sensitivity, not removal-impact. Channels that are >99.9% redundant
with their neighbors (and therefore nearly free to remove) can still show large
permutation ΔRMSE. The continuum control is defined by redundancy for exactly
this reason; the retrain is the real test.

Geometry (idx 636..639) and the three extras (idx 640..642, zeroed in M0) are
always retained.

Usage
-----
    # write masks + a diagnostic figure next to this script
    /opt/anaconda3/bin/python3 wv_band_mask.py --plot

    # in the trainer
    import wv_band_mask
    masks = wv_band_mask.build_masks(h5_path)
    keep_spectral_idx = masks['wv_core']['keep_spectral_idx']   # into 0..635

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent
# DEFAULT_H5 = "/Users/andrewbuggee/Documents/VS_CODE/Python-Research/lasp-CU-paper-3/training_data/synthetic_training_data_7-levels_8_May_2026.h5"
DEFAULT_H5 = "/scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_7-levels_8_May_2026.h5"

DEFAULT_IMPORTANCE_NPZ = (REPO / "feature_importance_run004"
                          / "feature_importance_run004.npz")
DEFAULT_OUT = REPO / "wv_core_ablation" / "wv_core_ablation_masks.npz"

N_SPECTRAL = 636          # reflectance channels, idx 0..635
N_TAIL = 7                # 4 geometry (636..639) + 3 extras (640..642), always kept

# --- Water-vapor core windows [nm] (inclusive). -----------------------------
# 940 & 1140 nm are unsaturated: the core itself carries information, so these
# windows remove the informative dip. 1380 & 1900 nm are saturated (core R≈0):
# these windows remove the dead minimum but stop short of the recovering edges,
# which are kept (they carry the band's information and the user wants the edges).
WV_CORE_WINDOWS_NM: Dict[str, Tuple[float, float]] = {
    "940": (932.0, 960.0),
    "1140": (1116.0, 1155.0),
    "1380": (1350.0, 1410.0),
    "1900": (1831.0, 1920.0),
}

# --- Continuum-control windows [nm] (inclusive). ----------------------------
# Gas-absorption-free atmospheric windows of high, smooth reflectance. The
# control is drawn from channels here that are also >0.999 correlated with their
# neighbor (genuinely redundant).
CONTINUUM_WINDOWS_NM: List[Tuple[float, float]] = [
    (780.0, 880.0),    # NIR window, between O2-A and the 940 nm H2O band
    (995.0, 1070.0),   # between 940 and 1140 nm H2O
    (1245.0, 1310.0),  # between 1140 and 1380 nm H2O
    (1560.0, 1745.0),  # 1650 nm SWIR window, between 1380 and 1900 nm H2O
    (2150.0, 2280.0),  # SWIR window past the 1900/2000 nm complex
]
CONTINUUM_MIN_NEIGHBOR_CORR = 0.999


# ---------------------------------------------------------------------------
# Reflectance statistics
# ---------------------------------------------------------------------------
def reflectance_statistics(
    h5_path: str | Path = DEFAULT_H5, instrument: str = "hysics"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wavelengths_nm[636], mean_reflectance[636], neighbor_corr[636]).

    neighbor_corr[i] is the Pearson correlation, across all samples, between
    channel i and channel i+1 (the last entry repeats the previous one). High
    values mark spectrally redundant channels.
    """
    refl_key = f"reflectances_{instrument}"
    with h5py.File(h5_path, "r") as f:
        wavelengths_nm = f["wavelengths"][:].astype(np.float64)
        key = refl_key if refl_key in f else "reflectances"
        refl = f[key][:].astype(np.float64)          # (n_samples, 636)

    mean_reflectance = refl.mean(axis=0)             # (636,)

    # Adjacent-channel Pearson correlation, vectorized over samples.
    z = (refl - refl.mean(axis=0)) / (refl.std(axis=0) + 1e-12)
    adj = (z[:, :-1] * z[:, 1:]).mean(axis=0)        # (635,) corr(i, i+1)
    neighbor_corr = np.empty(refl.shape[1], dtype=np.float64)
    neighbor_corr[:-1] = adj
    neighbor_corr[-1] = adj[-1]
    return wavelengths_nm, mean_reflectance, neighbor_corr


# ---------------------------------------------------------------------------
# Channel-set construction
# ---------------------------------------------------------------------------
def _indices_in_windows(
    wavelengths_nm: np.ndarray, windows_nm: List[Tuple[float, float]]
) -> np.ndarray:
    mask = np.zeros(wavelengths_nm.shape[0], dtype=bool)
    for lo_nm, hi_nm in windows_nm:
        mask |= (wavelengths_nm >= lo_nm) & (wavelengths_nm <= hi_nm)
    return np.where(mask)[0]


def wv_core_remove_indices(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Spectral indices (into 0..635) inside the water-vapor core windows."""
    return _indices_in_windows(wavelengths_nm, list(WV_CORE_WINDOWS_NM.values()))


def continuum_control_remove_indices(
    wavelengths_nm: np.ndarray,
    neighbor_corr: np.ndarray,
    n_target: int,
    min_neighbor_corr: float = CONTINUUM_MIN_NEIGHBOR_CORR,
) -> np.ndarray:
    """Count-matched redundant-continuum indices for the control ablation.

    Channels are restricted to the gas-free continuum windows and to
    neighbor_corr > min_neighbor_corr, then `n_target` are taken evenly spaced
    so the control samples every window rather than clustering.
    """
    in_windows = _indices_in_windows(wavelengths_nm, CONTINUUM_WINDOWS_NM)
    pool = in_windows[neighbor_corr[in_windows] > min_neighbor_corr]
    if pool.size < n_target:
        raise ValueError(
            f"continuum pool has {pool.size} channels but {n_target} requested; "
            f"widen CONTINUUM_WINDOWS_NM or lower min_neighbor_corr."
        )
    picks = np.unique(np.linspace(0, pool.size - 1, n_target).round().astype(int))
    # Even spacing can collide after rounding; top up deterministically.
    i = 0
    chosen = set(picks.tolist())
    while len(chosen) < n_target:
        if i not in chosen:
            chosen.add(i)
        i += 1
    return np.sort(pool[np.array(sorted(chosen))])


def keep_from_remove(remove_idx: np.ndarray, n_spectral: int = N_SPECTRAL) -> np.ndarray:
    """Complement of `remove_idx` within the spectral channels (0..n_spectral-1)."""
    keep = np.ones(n_spectral, dtype=bool)
    keep[remove_idx] = False
    return np.where(keep)[0]


def full_keep_indices(keep_spectral_idx: np.ndarray,
                      n_spectral: int = N_SPECTRAL,
                      n_tail: int = N_TAIL) -> np.ndarray:
    """Indices into the full input vector to keep: spectral-kept ⊕ geometry+extras.

    The model input from LibRadtranDatasetExtras is laid out as
    [636 reflectance | 4 geometry | 3 extras]; the geometry+extras tail is
    always retained.
    """
    tail = np.arange(n_spectral, n_spectral + n_tail)
    return np.concatenate([keep_spectral_idx, tail])


def build_masks(
    h5_path: str | Path = DEFAULT_H5, instrument: str = "hysics"
) -> Dict[str, object]:
    """Build all channel masks. Returns a dict consumed by the trainer.

    Keys:
      'wavelengths_nm', 'mean_reflectance', 'neighbor_corr'
      'full'               : {keep_spectral_idx, remove_spectral_idx, n_kept_spectral}
      'wv_core'            : same, plus 'remove_wavelengths_nm', 'windows_nm'
      'continuum_control'  : same, plus 'remove_wavelengths_nm', 'windows_nm'
    """
    wl_nm, mean_r, nbr = reflectance_statistics(h5_path, instrument)

    wv_remove = wv_core_remove_indices(wl_nm)
    ctrl_remove = continuum_control_remove_indices(wl_nm, nbr, n_target=wv_remove.size)

    def pack(remove_idx: np.ndarray, windows=None) -> Dict[str, object]:
        keep_spectral = keep_from_remove(remove_idx)
        out: Dict[str, object] = {
            "remove_spectral_idx": remove_idx,
            "keep_spectral_idx": keep_spectral,
            "n_kept_spectral": int(keep_spectral.size),
            "remove_wavelengths_nm": wl_nm[remove_idx],
        }
        if windows is not None:
            out["windows_nm"] = windows
        return out

    return {
        "wavelengths_nm": wl_nm,
        "mean_reflectance": mean_r,
        "neighbor_corr": nbr,
        "full": pack(np.array([], dtype=int)),
        "wv_core": pack(wv_remove, WV_CORE_WINDOWS_NM),
        "continuum_control": pack(ctrl_remove, CONTINUUM_WINDOWS_NM),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_masks(masks: Dict[str, object], out_path: str | Path = DEFAULT_OUT) -> Path:
    """Flatten the masks dict and save to .npz for reproducible trainer runs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        "wavelengths_nm": masks["wavelengths_nm"],
        "mean_reflectance": masks["mean_reflectance"],
        "neighbor_corr": masks["neighbor_corr"],
    }
    for name in ("full", "wv_core", "continuum_control"):
        m = masks[name]
        flat[f"{name}__keep_spectral_idx"] = m["keep_spectral_idx"]
        flat[f"{name}__remove_spectral_idx"] = m["remove_spectral_idx"]
    np.savez_compressed(out_path, **flat)
    return out_path


def keep_spectral_idx_for(mask_name: str,
                          masks_npz: str | Path | None = None,
                          h5_path: str | Path = DEFAULT_H5,
                          instrument: str = "hysics") -> np.ndarray:
    """Return the spectral keep-indices for `mask_name`.

    Loads a precomputed .npz if given (and present); otherwise builds masks
    fresh from the HDF5. `mask_name` is 'full', 'wv_core', or 'continuum_control'.
    """
    if mask_name not in ("full", "wv_core", "continuum_control"):
        raise ValueError(f"unknown mask_name {mask_name!r}")
    if masks_npz is not None and Path(masks_npz).exists():
        with np.load(masks_npz) as d:
            return d[f"{mask_name}__keep_spectral_idx"]
    return build_masks(h5_path, instrument)[mask_name]["keep_spectral_idx"]


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
def plot_masks(masks: Dict[str, object],
               importance_npz: str | Path = DEFAULT_IMPORTANCE_NPZ,
               out_png: str | Path | None = None) -> Path:
    """Two-panel figure: mean reflectance and (if available) permutation ΔRMSE,
    with the wv_core and continuum_control removed channels shaded."""
    import matplotlib.pyplot as plt

    wl_nm = masks["wavelengths_nm"]
    mean_r = masks["mean_reflectance"]
    wv_remove = masks["wv_core"]["remove_spectral_idx"]
    ctrl_remove = masks["continuum_control"]["remove_spectral_idx"]

    have_imp = Path(importance_npz).exists()
    n_panels = 2 if have_imp else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 7 if have_imp else 4),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    def shade(ax):
        for c in wv_remove:
            ax.axvspan(wl_nm[c] - 1.5, wl_nm[c] + 1.5, color="tab:red",
                       alpha=0.25, lw=0)
        for c in ctrl_remove:
            ax.axvspan(wl_nm[c] - 1.5, wl_nm[c] + 1.5, color="tab:blue",
                       alpha=0.25, lw=0)

    ax0 = axes[0]
    ax0.plot(wl_nm, mean_r, color="k", lw=1.0, label="mean reflectance")
    shade(ax0)
    ax0.set_ylabel("mean TOA reflectance")
    ax0.set_title(
        f"WV-core ablation channel sets — "
        f"red = wv_core ({wv_remove.size} ch removed), "
        f"blue = continuum_control ({ctrl_remove.size} ch removed); "
        f"kept spectral = {masks['wv_core']['n_kept_spectral']}"
    )
    # legend proxies
    ax0.axvspan(np.nan, np.nan, color="tab:red", alpha=0.25, label="wv_core removed")
    ax0.axvspan(np.nan, np.nan, color="tab:blue", alpha=0.25,
                label="continuum_control removed")
    ax0.legend(loc="upper right", fontsize=8)

    if have_imp:
        with np.load(importance_npz, allow_pickle=True) as d:
            perm = d["perm_mean_um"][:N_SPECTRAL]
        ax1 = axes[1]
        ax1.plot(wl_nm, perm, color="k", lw=0.8)
        shade(ax1)
        ax1.set_ylabel("permutation ΔRMSE [μm]")
        ax1.set_xlabel("wavelength [nm]")
        ax1.set_title("per-channel permutation importance (run_004) under the same shading",
                      fontsize=9)
    else:
        ax0.set_xlabel("wavelength [nm]")

    fig.tight_layout()
    out_png = Path(out_png) if out_png else (DEFAULT_OUT.parent / "wv_core_ablation_masks.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _summary(masks: Dict[str, object]) -> None:
    wl_nm = masks["wavelengths_nm"]
    print(f"Spectral channels: {N_SPECTRAL} "
          f"({wl_nm.min():.1f}-{wl_nm.max():.1f} nm)")
    for name in ("wv_core", "continuum_control"):
        m = masks[name]
        rem = m["remove_spectral_idx"]
        print(f"\n[{name}] removes {rem.size} channels; keeps "
              f"{m['n_kept_spectral']} spectral + {N_TAIL} geom/extras")
        if name == "wv_core":
            for band, (lo, hi) in WV_CORE_WINDOWS_NM.items():
                sub = rem[(wl_nm[rem] >= lo) & (wl_nm[rem] <= hi)]
                print(f"    {band:>5} nm core [{lo:.0f}-{hi:.0f}]: {sub.size} ch")
        else:
            for lo, hi in CONTINUUM_WINDOWS_NM:
                sub = rem[(wl_nm[rem] >= lo) & (wl_nm[rem] <= hi)]
                print(f"    window [{lo:.0f}-{hi:.0f}]: {sub.size} ch")


def main() -> None:
    p = argparse.ArgumentParser(description="Build/visualize WV-core ablation masks.")
    p.add_argument("--h5-path", type=str, default=str(DEFAULT_H5))
    p.add_argument("--instrument", type=str, default="hysics",
                   choices=["hysics", "emit"])
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="Output .npz path for the masks.")
    p.add_argument("--plot", action="store_true",
                   help="Also write a diagnostic PNG next to the .npz.")
    args = p.parse_args()

    masks = build_masks(args.h5_path, args.instrument)
    _summary(masks)
    out = save_masks(masks, args.out)
    print(f"\nWrote masks -> {out}")
    if args.plot:
        png = plot_masks(masks, out_png=Path(args.out).with_suffix(".png"))
        print(f"Wrote figure -> {png}")


if __name__ == "__main__":
    main()
