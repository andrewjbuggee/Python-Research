"""
train_wv_core_ablation.py — Retrain the published paper-3 network (sweep
run_004, variant M0) with water-vapor band-core channels physically removed,
to test whether those cores carry droplet-profile information (cf. King &
Vaughan, who reported little information at the 1150 and 1900 nm bands).

What it does
------------
Reproduces the run_004 training recipe bit-for-bit — same hyperparameters
(read from the run_004 sweep config), same seed-42 profile-aware 80/10/10 split,
same ProfileOnlyLoss, same canonical train/validate/predict helpers — and only
changes the set of reflectance channels fed to the network. The masked channels
are dropped (not zeroed): the first Linear layer narrows from 636+4 to
(n_kept)+4 spectral+geometry inputs, plus the 3 zeroed M0 extras.

Three masks (see wv_band_mask.py):
    full                — keep all 636 channels (reproduction / re-baseline)
    wv_core             — remove the 940/1140/1380/1900 nm absorption cores
    continuum_control   — remove a count-matched set of redundant continuum
                          channels (isolates the dimensionality effect)

Because wv_core and continuum_control remove the *same number* of channels, the
two ablated networks are architecturally identical (same width, same parameter
count, same seed-42 init); the only difference is which channels carry signal.
Run `full` first and confirm it reproduces the published 0.9641 μm before
trusting the ablations.

Outputs (per run, under <output-dir>/run_<mask>/)
    best_model.pt   summary.json   history.json

Run (local, MPS / CPU)
    /opt/anaconda3/bin/python3 train_wv_core_ablation.py --mask wv_core
Run (Alpine) — see run_wv_core_ablation_alpine.sh.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# This script lives in repo_root/wv_core_ablation/; REPO is the repo root (two
# levels up) so the shared modules (models, data, ...) import and the
# config/training_data paths resolve. wv_band_mask is imported from this script's
# own directory, which Python adds to sys.path automatically.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models import RetrievalConfig                              # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras  # noqa: E402
from models_profile_only import ProfileOnlyLoss                 # noqa: E402
from data import (LibRadtranDatasetExtras, create_profile_aware_splits,  # noqa: E402
                  resolve_h5_path)
from train_standalone_profile_only_extras import (              # noqa: E402
    train_one_epoch, validate, predict_test)
import wv_band_mask                                             # noqa: E402

N_EXTRAS = 3              # tau_c, wv_above_cloud, wv_in_cloud (all zeroed in M0)
DEFAULT_CONFIG = (REPO / "hyper_parameter_sweep"
                  / "sweep_configs_profile_only_synthetic_M0" / "run_004.json")
DEFAULT_H5 = REPO / "training_data" / "synthetic_training_data_7-levels_8_May_2026.h5"
PUBLISHED_RMSE_UM = 0.9640962140900748   # run_004 mean test RMSE, for reference


class SpectralMasked(Dataset):
    """Wrap a dataset returning (x, profile, tau); keep only selected x columns.

    `keep_full_idx` indexes into the full input vector
    [636 reflectance | 4 geometry | 3 extras]; it is the spectral-kept indices
    followed by the always-kept geometry+extras tail.
    """

    def __init__(self, base: Dataset, keep_full_idx: np.ndarray):
        self.base = base
        self.keep = torch.as_tensor(keep_full_idx, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, prof, tau = self.base[i]            # x: (643,)
        return x.index_select(0, self.keep), prof, tau   # x: (n_kept+7,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WV-core ablation retrain of run_004.")
    p.add_argument("--mask", required=True,
                   choices=["full", "wv_core", "continuum_control"],
                   help="Which spectral channels to remove.")
    p.add_argument("--config-json", type=str, default=str(DEFAULT_CONFIG),
                   help="run_004 sweep config (source of hyperparameters/extras).")
    p.add_argument("--h5-path", type=str, default=None,
                   help="Full override of the HDF5 path.")
    p.add_argument("--training-data-dir", type=str, default=None,
                   help="Override only the directory of cfg['data']['h5_path'].")
    p.add_argument("--masks-npz", type=str, default=None,
                   help="Precomputed masks .npz (else built fresh from the HDF5).")
    p.add_argument("--output-dir", type=str,
                   default=str(REPO / "wv_core_ablation"),
                   help="Results root; run_<mask>/ is created underneath.")
    p.add_argument("--device", type=str, default=None,
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42,
                   help="Match run_004 (42) for an identical split.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override n_epochs (for quick smoke tests only).")
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config_json).read_text())
    hp = cfg["hyperparams"]
    extras = cfg.get("extras", {})
    zero_tau_c = bool(extras.get("zero_tau_c", True))
    zero_wv_above = bool(extras.get("zero_wv_above", True))
    zero_wv_in = bool(extras.get("zero_wv_in", True))

    # ── HDF5 path resolution (mirrors the sweep trainer) ──────────────────
    if args.h5_path:
        h5_path = Path(args.h5_path)
    elif args.training_data_dir:
        h5_path = resolve_h5_path(cfg["data"]["h5_path"], args.training_data_dir)
    else:
        h5_path = DEFAULT_H5
    h5_path = h5_path.resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    device = pick_device(args.device)

    # ── Reproducible seeding (record the seed in run output) ──────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Spectral mask -> kept channel indices ─────────────────────────────
    keep_spectral_idx = wv_band_mask.keep_spectral_idx_for(
        args.mask, masks_npz=args.masks_npz, h5_path=str(h5_path),
        instrument=cfg["data"].get("instrument", "hysics"))
    keep_full_idx = wv_band_mask.full_keep_indices(keep_spectral_idx)
    n_kept_spectral = int(keep_spectral_idx.size)
    n_removed = wv_band_mask.N_SPECTRAL - n_kept_spectral

    print("=" * 70)
    print(f"  WV-CORE ABLATION — mask = {args.mask}")
    print("=" * 70)
    print(f"  HDF5            : {h5_path}")
    print(f"  device / seed   : {device} / {args.seed}")
    print(f"  spectral kept   : {n_kept_spectral} / {wv_band_mask.N_SPECTRAL} "
          f"({n_removed} removed)")
    print(f"  model input dim : {n_kept_spectral + 4} + {N_EXTRAS} extras "
          f"= {n_kept_spectral + 4 + N_EXTRAS}")
    print(f"  extras (M0)     : tau_c={'0' if zero_tau_c else 'on'} "
          f"wv_above={'0' if zero_wv_above else 'on'} "
          f"wv_in={'0' if zero_wv_in else 'on'}")
    print()

    # ── Full dataset + identical seed-42 profile-aware split ──────────────
    full_ds = LibRadtranDatasetExtras(
        str(h5_path), normalize=True,
        instrument=cfg["data"].get("instrument", "hysics"),
        zero_tau_c=zero_tau_c, zero_wv_above=zero_wv_above, zero_wv_in=zero_wv_in)

    train_idx, val_idx, test_idx = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=cfg["n_val_profiles"],
        n_test_profiles=cfg["n_test_profiles"],
        seed=args.seed)

    pin = torch.cuda.is_available()
    num_workers = cfg["data"].get("num_workers", 4)

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        masked = SpectralMasked(Subset(full_ds, idx), keep_full_idx)
        return DataLoader(masked, batch_size=hp["batch_size"], shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    test_loader = loader(test_idx, False)

    n_levels = full_ds.n_levels
    if len(hp["level_weights"]) != n_levels:
        raise ValueError(
            f"level_weights has {len(hp['level_weights'])} entries but HDF5 has "
            f"{n_levels} levels.")

    # ── Model / loss / optimizer (mirror run_004) ─────────────────────────
    model_cfg = RetrievalConfig(
        n_wavelengths=n_kept_spectral,      # <-- shrunk by the mask
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp["hidden_dims"]),
        dropout=hp["dropout"],
        activation=hp.get("activation", "gelu"))

    # Sanity-check dataset / model dim agreement before building the network.
    sample_x = train_loader.dataset[0][0]
    if int(sample_x.shape[-1]) != model_cfg.input_dim + N_EXTRAS:
        raise RuntimeError(
            f"masked x has {int(sample_x.shape[-1])} features but model expects "
            f"{model_cfg.input_dim + N_EXTRAS}.")

    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)

    level_weights = torch.tensor(hp["level_weights"], dtype=torch.float32)
    criterion = ProfileOnlyLoss(
        config=model_cfg,
        lambda_physics=hp.get("lambda_physics", 0.1),
        lambda_monotonicity=hp.get("lambda_monotonicity", 0.0),
        lambda_adiabatic=hp.get("lambda_adiabatic", 0.1),
        lambda_smoothness=hp.get("lambda_smoothness", 0.1),
        level_weights=level_weights,
        sigma_floor=hp.get("sigma_floor", 0.01)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hp["learning_rate"],
                            weight_decay=hp.get("weight_decay", 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=hp.get("scheduler_patience", 30),
                                  factor=0.5)

    # ── Training loop (mirror sweep_train_profile_only_synthetic) ─────────
    n_epochs = int(args.epochs or hp["n_epochs"])
    early_stop = int(hp.get("early_stop_patience", 150))
    warmup = int(hp.get("warmup_steps", 500))
    aug_noise = float(hp.get("augment_noise_std", 0.0))
    target_lr = float(hp["learning_rate"])

    best_val, best_epoch, best_state, no_improve = float("inf"), -1, None, 0
    history = {"train": [], "val": []}
    global_step = 0

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=aug_noise, warmup_steps=warmup,
            target_lr=target_lr, global_step_start=global_step)
        val_loss = validate(model, val_loader, criterion, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if global_step >= warmup:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:4d} | Train {train_loss:7.4f} | "
                  f"Val {val_loss:7.4f} | LR {optimizer.param_groups[0]['lr']:.1e} "
                  f"| NoImp {no_improve}")
        if no_improve >= early_stop:
            print(f"  Early stop at epoch {epoch}")
            break
    train_seconds = time.time() - t0

    # ── Test on best checkpoint ───────────────────────────────────────────
    model.load_state_dict(best_state)
    pred = predict_test(model, test_loader, device, model_cfg)
    err = pred["pred"] - pred["true"]                       # (n_test, n_levels) μm
    per_level_rmse_um = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
    mean_rmse_um = float(np.mean(per_level_rmse_um))
    mean_sigma_um = float(np.mean(pred["pred_std"]))

    # ── Persist ───────────────────────────────────────────────────────────
    run_dir = Path(args.output_dir) / f"run_{args.mask}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mask": args.mask,
        "seed": args.seed,
        "n_kept_spectral": n_kept_spectral,
        "n_removed_spectral": int(n_removed),
        "removed_spectral_idx": keep_spectral_complement(keep_spectral_idx).tolist(),
        "model_input_dim": model_cfg.input_dim + N_EXTRAS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_trained": len(history["train"]),
        "train_seconds": train_seconds,
        "mean_test_rmse_um": mean_rmse_um,
        "mean_test_sigma_um": mean_sigma_um,
        "per_level_rmse_um": per_level_rmse_um,
        "published_run004_rmse_um": PUBLISHED_RMSE_UM,
        "delta_vs_published_um": mean_rmse_um - PUBLISHED_RMSE_UM,
        "hyperparams": hp,
        "extras": extras,
        "data": {"h5_path": str(h5_path),
                 "instrument": cfg["data"].get("instrument", "hysics")},
        "n_val_profiles": cfg["n_val_profiles"],
        "n_test_profiles": cfg["n_test_profiles"],
        "config_json": str(Path(args.config_json).resolve()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history))
    torch.save(best_state, run_dir / "best_model.pt")

    print(f"\n  best epoch  : {best_epoch} (val {best_val:.4f})")
    print(f"  mean RMSE   : {mean_rmse_um:.4f} μm  "
          f"(run_004 published {PUBLISHED_RMSE_UM:.4f} μm; "
          f"Δ {mean_rmse_um - PUBLISHED_RMSE_UM:+.4f})")
    print("  per-level   : "
          + " ".join(f"L{l + 1}={r:.2f}" for l, r in enumerate(per_level_rmse_um)))
    print(f"  wall time   : {train_seconds:.0f} s")
    print(f"  saved -> {run_dir}")


def keep_spectral_complement(keep_spectral_idx: np.ndarray) -> np.ndarray:
    """Removed spectral indices, for provenance in summary.json."""
    keep = np.ones(wv_band_mask.N_SPECTRAL, dtype=bool)
    keep[keep_spectral_idx] = False
    return np.where(keep)[0]


if __name__ == "__main__":
    main()
