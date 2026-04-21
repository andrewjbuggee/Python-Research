"""
plot_sweep_profiles.py — predicted vs true r_e profiles for top-N runs.

For each of the top-N runs (ranked by mean test RMSE):
  - load best_model.pt + config.json
  - build the test dataloader (profile-held-out split, same seed as training)
  - iterate test batches, DEDUPLICATING by unique true profile so each panel
    shows a different cloud (each held-out cloud appears ~128 times in the
    loader via varied solar/viewing geometry; without deduplication all
    panels in the figure would show the same cloud)
  - pull raw in-situ profiles (before altitude interpolation) from the HDF5
    so they can be overlaid on the retrieval
  - 2×3 panel figure: left y-axis is optical depth τ; right y-axis is
    altitude (km).  True profile and NN retrieval (±1σ) are plotted on
    the interpolated grid; raw in-situ measurement is overlaid.

Optical-depth axis
------------------
If the HDF5 contains a `profiles_raw_tau` dataset (added by the updated
`convert_matFiles_to_HDF.py`), we use the real measured τ(z) to build
the left y-axis and to place the 7 interpolated model levels.  If the
dataset is absent (older HDF5 files), we fall back to a linear
approximation τ(level i) = i/(N-1) · τ_c.  A one-line notice is printed
per run indicating which path was used.

Requires:
    - HDF5 training file accessible (use --h5-path to override if config
      points at Alpine but you are running locally).
    - torch, numpy, matplotlib, pandas, h5py.

Usage:
    python plot_sweep_profiles.py --results-dir sweep_results_3
    python plot_sweep_profiles.py --h5-path /local/path/training.h5
    python plot_sweep_profiles.py --top-n 10 --n-examples 6 --device cpu

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import DropletProfileNetwork, RetrievalConfig
from data import create_dataloaders, resolve_h5_path


GREEN     = '#10B981'   # NN retrieval
RAW_BLUE  = '#1F5FC2'   # raw in-situ


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────
def build_model_from_config(cfg_dict, device, n_levels):
    hp = cfg_dict['hyperparams']
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation='gelu',
    )
    model = DropletProfileNetwork(model_config).to(device)
    return model, model_config


# ─────────────────────────────────────────────────────────────────────────────
# Inference + deduplication
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gather_unique_profiles(model, test_loader, device):
    """
    Walk the *entire* test DataLoader and collect one example per UNIQUE
    true profile.

    The held-out test set contains ~14 distinct clouds, each replicated at
    ~128 sun/view geometries.  Without this deduplication, picking random
    loader indices tends to grab several geometries of the same cloud —
    which is what produced the "all six panels are identical" bug in the
    previous version of this script.

    We previously bailed out as soon as we had n_examples uniques.  We now
    walk the full loader so the caller can rank every unique by RMSE (or
    sample randomly from the full pool) — required for --selection best.
    The cost is one full test-set forward pass, which is cheap (~1800 samples).

    Returns a dict with arrays indexed by unique-profile order (insertion order
    in the loader).  Also returns `subset_positions`, the position of each
    chosen sample in the test Subset; callers use this to look up the
    corresponding row in the underlying HDF5 (via subset.indices) for the
    raw in-situ overlay.
    """
    model.eval()

    subset_pos   = 0   # running index into the test Subset
    seen         = {}  # profile-hash -> subset position of first occurrence
    pred_u       = {}
    pred_std_u   = {}
    tau_pred_u   = {}
    tau_pred_std_u = {}
    prof_norm_u  = {}
    tau_true_u   = {}

    for x, prof, tau in test_loader:
        x_d = x.to(device)
        out = model(x_d)
        pred          = out['profile'].cpu().numpy()
        pred_std      = out['profile_std'].cpu().numpy()
        tau_pred      = out['tau_c'].squeeze(-1).cpu().numpy()
        tau_pred_std  = out['tau_std'].squeeze(-1).cpu().numpy()
        prof_np       = prof.numpy()
        tau_np        = tau.numpy()

        for i in range(prof_np.shape[0]):
            key = tuple(np.round(prof_np[i], 6))  # identical profiles hash the same
            if key not in seen:
                seen[key]           = subset_pos + i
                pred_u[key]         = pred[i]
                pred_std_u[key]     = pred_std[i]
                tau_pred_u[key]     = float(tau_pred[i])
                tau_pred_std_u[key] = float(tau_pred_std[i])
                prof_norm_u[key]    = prof_np[i]
                tau_true_u[key]     = float(tau_np[i])

        subset_pos += prof_np.shape[0]

    keys = list(seen.keys())
    return {
        'subset_positions': np.array([seen[k] for k in keys], dtype=int),
        'pred'            : np.stack([pred_u[k]         for k in keys]),
        'pred_std'        : np.stack([pred_std_u[k]     for k in keys]),
        'tau_pred'        : np.array([tau_pred_u[k]     for k in keys]),
        'tau_pred_std'    : np.array([tau_pred_std_u[k] for k in keys]),
        'prof_norm'       : np.stack([prof_norm_u[k]    for k in keys]),
        'tau_true_norm'   : np.array([tau_true_u[k]     for k in keys]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Raw profile lookup from the HDF5
# ─────────────────────────────────────────────────────────────────────────────
def read_raw_profiles(h5_path, subset, subset_positions):
    """
    Fetch the raw in-situ r_e profile, raw altitude grid, and scalar tau_c
    for each chosen sample.  subset_positions are positions within the test
    Subset; we translate via subset.indices to the underlying HDF5 row.
    """
    if hasattr(subset, 'indices'):
        subset_to_hdf5 = np.asarray(subset.indices, dtype=int)
    else:
        # Fallback: identity mapping (no Subset wrapper).  Would only happen
        # if profile_holdout=False; we warn then proceed.
        print("  [warn] test loader's dataset has no .indices; assuming "
              "identity mapping to HDF5 rows.")
        subset_to_hdf5 = np.arange(len(subset))

    hdf5_rows = subset_to_hdf5[subset_positions]

    raw_re, raw_z, raw_tau, tau_c_vals = [], [], [], []
    with h5py.File(h5_path, 'r') as f:
        n_lev_arr   = f['profile_n_levels']
        raw_re_ds   = f['profiles_raw']
        raw_z_ds    = f['profiles_raw_z']
        tau_c_ds    = f['tau_c']
        # profiles_raw_tau was added by the updated converter; tolerate its
        # absence so this script still runs against older HDF5 files
        # (sweep_results_2 etc.).  When missing, we plot altitude on the
        # left y-axis instead of optical depth.
        has_raw_tau = 'profiles_raw_tau' in f
        raw_tau_ds  = f['profiles_raw_tau'] if has_raw_tau else None
        for r in hdf5_rows:
            n = int(n_lev_arr[r])
            raw_re.append(raw_re_ds[r, :n].astype(np.float32).copy())
            raw_z.append(raw_z_ds[r, :n].astype(np.float32).copy())
            tau_c_vals.append(float(tau_c_ds[r]))
            if has_raw_tau:
                raw_tau.append(raw_tau_ds[r, :n].astype(np.float32).copy())
            else:
                raw_tau.append(None)
    return raw_re, raw_z, raw_tau, np.array(tau_c_vals), has_raw_tau


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel plot helper (also used by the grid figure)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_panel(ax, pred, pred_std, true_um, raw_re, raw_z, raw_tau,
                tau_true, tau_pred, tau_pred_std,
                show_legend=True, tick_fontsize=8, show_uncert=True):
    """
    Draw a single retrieval-vs-truth panel.

    Two y-axis modes:
      raw_tau is a numpy array  → left y-axis = measured optical depth τ;
                                   right y-axis (twinx) = altitude (km).
      raw_tau is None           → left y-axis = altitude (km) only.
                                   This is the backward-compatible path for
                                   HDF5 files that predate /profiles_raw_tau
                                   (e.g. sweep_results_2's training data).

    show_uncert : True  → NN retrieval drawn as squares with horizontal ±1σ caps
                          connected by a dashed line (label includes "±1σ").
                  False → NN retrieval drawn as squares connected by a solid
                          line, no error bars — same style family as the true
                          profile, useful for paper figures where the σ would
                          visually clutter the comparison.
    """
    n_levels = len(pred)

    # Altitudes at the N interpolated model levels: the converter uses
    # linspace between z_raw[0] (top) and z_raw[-1] (base).
    z_top  = float(raw_z[0])
    z_base = float(raw_z[-1])
    z_lvl  = np.linspace(z_top, z_base, n_levels)

    if raw_tau is not None:
        # ── τ-axis mode: project the model levels onto the measured τ(z) ─
        z_asc, tau_asc = raw_z[::-1], raw_tau[::-1]
        tau_lvl         = np.interp(z_lvl, z_asc, tau_asc)
        y_lvl           = tau_lvl              # model points
        y_raw           = raw_tau              # raw in-situ points
        ax.invert_yaxis()                       # τ=0 at top, τ_c at bottom
        ax.set_ylabel(r'Optical depth $\tau$', fontsize=14)
        # Right y-axis: altitude (km).  Place z_top at the top of the plot
        # (matches τ=0 at top) and z_base at the bottom — no extra invert.
        ax2 = ax.twinx()
        ax2.set_ylim(z_base, z_top)
        ax2.set_ylabel('Altitude (km)', fontsize=14)
        ax2.tick_params(labelsize=tick_fontsize)
    else:
        # ── altitude-axis fallback (older HDF5 without raw τ) ────────────
        # No twinx — just altitude on the left, with z_top at the top of the
        # plot.  z_top > z_base so a normal axis with no invert puts higher
        # altitude at the top automatically.
        y_lvl = z_lvl
        y_raw = raw_z
        ax.set_ylabel('Altitude (km)', fontsize=14)
        ax2   = None

    # Raw in-situ profile (blue, partially transparent)
    ax.plot(raw_re, y_raw, '-', color=RAW_BLUE, linewidth=1.5, alpha=0.55,
            label='In-situ raw profile')

    # Interpolated true profile on the model grid (black circles)
    ax.plot(true_um, y_lvl, 'ko-', markersize=8, linewidth=1.5,
            label=f'{n_levels}-level training profile')

    # NN retrieval — with or without ±1σ error bars depending on show_uncert.
    if show_uncert:
        ax.errorbar(pred, y_lvl, xerr=pred_std,
                    fmt='s--', color=GREEN, markersize=6, linewidth=1.5,
                    elinewidth=1.0, capsize=3,
                    label='NN retrieval ±1σ')
    else:
        ax.plot(pred, y_lvl, 's-', color=GREEN, markersize=6, linewidth=1.5,
                label='NN retrieval')

    ax.set_xlabel(r'$r_e$ (μm)', fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=tick_fontsize)

    # Drop the ± uncertainty from the title when uncertainty is hidden in
    # the panel — keeps the visual story consistent (no σ on the curve, no
    # σ in the title).
    if show_uncert:
        title_str = (rf'$\tau_\mathrm{{true}}$ = {tau_true:.2f}   '
                     rf'$\tau_\mathrm{{pred}}$ = {tau_pred:.2f} ± {tau_pred_std:.2f}')
    else:
        title_str = (rf'$\tau_\mathrm{{true}}$ = {tau_true:.2f}   '
                     rf'$\tau_\mathrm{{pred}}$ = {tau_pred:.2f}')
    ax.set_title(title_str, fontsize=14)
    if show_legend:
        ax.legend(fontsize=8, loc='lower right')
    return ax2


def plot_profiles(ex, run_info, out_path, n_examples=6, show_uncert=True):
    """6-panel figure for one run: unique test profiles, retrieved vs true."""
    n_examples = min(n_examples, len(ex['pred']))
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), squeeze=False)
    axes = axes.ravel()

    for j in range(n_examples):
        _draw_panel(
            axes[j],
            pred         = ex['pred'][j],
            pred_std     = ex['pred_std'][j],
            true_um      = ex['true_um'][j],
            raw_re       = ex['raw_re'][j],
            raw_z        = ex['raw_z'][j],
            raw_tau      = ex['raw_tau'][j],
            tau_true     = ex['tau_true'][j],
            tau_pred     = ex['tau_pred'][j],
            tau_pred_std = ex['tau_pred_std'][j],
            show_legend  = (j == 0),
            show_uncert  = show_uncert,
        )

    for ax in axes[n_examples:]:
        ax.axis('off')

    fig.suptitle(
        f"run_{int(run_info['run_id']):03d}: {run_info['run_name']}   "
        f"(mean RMSE {run_info['mean_rmse']:.3f} μm)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def plot_grid_single(per_run_examples, out_path, show_uncert=True):
    """One subplot per top-N run, each showing that run's first unique profile."""
    n = len(per_run_examples)
    ncols = 5 if n >= 5 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.8 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, ex in zip(axes, per_run_examples):
        _draw_panel(
            ax,
            pred         = ex['pred'],
            pred_std     = ex['pred_std'],
            true_um      = ex['true'],
            raw_re       = ex['raw_re'],
            raw_z        = ex['raw_z'],
            raw_tau      = ex['raw_tau'],
            tau_true     = ex['tau_true'],
            tau_pred     = ex['tau_pred'],
            tau_pred_std = ex['tau_pred_std'],
            show_legend  = False,
            tick_fontsize= 9,
            show_uncert  = show_uncert,
        )
        # Match the per-panel rule: omit ±σ from τ_p when uncertainty is hidden.
        if show_uncert:
            tau_p_str = rf"$\tau_p$={ex['tau_pred']:.1f}±{ex['tau_pred_std']:.2f}"
        else:
            tau_p_str = rf"$\tau_p$={ex['tau_pred']:.1f}"
        ax.set_title(f"run_{ex['run_id']:03d} — RMSE {ex['mean_rmse']:.3f} μm\n"
                     rf"$\tau_t$={ex['tau_true']:.1f}, " + tau_p_str,
                     fontsize=8)

    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='sweep_results_3')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--n-examples', type=int, default=6,
                   help='Unique profiles per run in the per-run figure')
    p.add_argument('--selection', choices=['random', 'best'], default='random',
                   help='How to pick the n unique profiles plotted: '
                        '"random" samples uniformly from the test set\'s unique '
                        'clouds; "best" picks the n with the lowest mean RMSE '
                        'across levels (good for showcasing the network at its '
                        'sharpest).  Both modes always show distinct clouds.')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed for --selection random.  rank-offset is '
                        'added so each top-N run shows a different sample.')
    p.add_argument('--show-uncert', choices=['on', 'off'], default='on',
                   help='"on" (default) draws horizontal ±1σ error bars on each '
                        'NN-retrieved level; "off" draws a clean marker-and-line '
                        'in the same style as the true-profile overlay (square '
                        'markers connected by a solid line, no error bars).')
    p.add_argument('--device', default=None,
                   help='"cuda", "cpu", or leave unset for auto-detect')
    p.add_argument('--h5-path', default=None,
                   help='Full HDF5 path override (overrides what is stored in '
                        'each run\'s config.json).  Mutually compatible with '
                        '--training-data-dir: both can be combined to override '
                        'just the directory or both directory and filename.')
    p.add_argument('--training-data-dir', default=None,
                   help='Directory hosting the HDF5 file on this machine. '
                        'Replaces only the directory portion of the path '
                        'stored in each config.json (filename is preserved). '
                        'Useful for switching between Alpine '
                        '(/scratch/alpine/anbu8374/neural_network_training_data/) '
                        'and a local copy without editing any sweep run\'s config.')
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / 'Figures'
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / 'comparison.csv')
    top = df.nsmallest(args.top_n, 'mean_rmse').reset_index(drop=True)
    print(f"Top {len(top)} runs by mean RMSE:\n")

    per_run_examples = []

    for rank, row in top.iterrows():
        rid = int(row['run_id'])
        run_dir = results_dir / f'run_{rid:03d}'
        cfg_path = run_dir / 'config.json'
        ckpt_path = run_dir / 'best_model.pt'

        if not cfg_path.exists() or not ckpt_path.exists():
            print(f"  [skip] rank {rank+1}: missing config or checkpoint in {run_dir}")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        hp = cfg['hyperparams']

        # Path resolution order: explicit --h5-path > config's h5_path,
        # then --training-data-dir overrides the directory portion of whichever
        # was chosen.
        base_path = args.h5_path if args.h5_path else cfg['data']['h5_path']
        h5_path = str(resolve_h5_path(base_path, args.training_data_dir))
        if not Path(h5_path).exists():
            print(f"  [skip] rank {rank+1}: HDF5 not found at {h5_path}")
            print(f"         pass --h5-path /your/local/path to override")
            continue

        _, _, test_loader = create_dataloaders(
            h5_path=h5_path,
            instrument=cfg['data'].get('instrument', 'hysics'),
            batch_size=hp.get('batch_size', 256),
            num_workers=0,
            seed=42,
            profile_holdout=True,
            n_val_profiles=14,
            n_test_profiles=14,
        )

        # Unwrap to find n_levels on the base RetrievalDataset.
        base_ds = test_loader.dataset
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
        n_levels = base_ds.n_levels

        model, model_config = build_model_from_config(cfg, device, n_levels)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        # 1. Gather every unique cloud in the test set (fixes the "6 identical
        #    panels" bug, and also gives us the full pool needed for ranking
        #    in --selection best mode).
        found = gather_unique_profiles(model, test_loader, device)
        n_unique = len(found['pred'])

        # 2. Denormalize ground truth into physical units (μm, dimensionless τ)
        re_min, re_max = model_config.re_min, model_config.re_max
        tau_min, tau_max = model_config.tau_min, model_config.tau_max
        true_um  = found['prof_norm'] * (re_max - re_min) + re_min
        tau_true = found['tau_true_norm'] * (tau_max - tau_min) + tau_min

        # 3. Pick which n_examples uniques to plot, per --selection.
        n_show = min(args.n_examples, n_unique)
        if n_show < args.n_examples:
            print(f"  [note] rank {rank+1}: only {n_unique} unique profiles in "
                  f"the test set (asked for {args.n_examples}); showing all.")

        if args.selection == 'best':
            # Per-cloud mean RMSE in physical units (μm), averaged over levels.
            rmse_per_unique = np.sqrt(
                np.mean((found['pred'] - true_um) ** 2, axis=1)
            )
            pick = np.argsort(rmse_per_unique)[:n_show]
            print(f"  selection=best: per-cloud mean RMSE range chosen = "
                  f"{rmse_per_unique[pick].min():.3f} – "
                  f"{rmse_per_unique[pick].max():.3f} μm "
                  f"(of {n_unique} uniques, range "
                  f"{rmse_per_unique.min():.3f} – {rmse_per_unique.max():.3f})")
        elif args.selection == 'random':
            rng = np.random.default_rng(args.seed + rank)
            pick = rng.choice(n_unique, size=n_show, replace=False)
            print(f"  selection=random (seed={args.seed + rank}): "
                  f"picked {n_show} of {n_unique} unique profiles")
        else:
            raise ValueError(f"unknown --selection {args.selection!r}")

        # 4. Subset every per-unique array down to the picked indices.
        #    Order in `ex` matches `pick` so the panels render in selection order
        #    (best first for 'best', random order for 'random').
        true_um_sel  = true_um[pick]
        tau_true_sel = tau_true[pick]
        sel_subset_positions = found['subset_positions'][pick]

        # 5. Pull raw in-situ profiles for ONLY the picked rows.
        #    raw_tau will be a list of None entries when the HDF5 does not
        #    contain /profiles_raw_tau; _draw_panel handles both cases.
        raw_re, raw_z, raw_tau, _, has_raw_tau = read_raw_profiles(
            h5_path, test_loader.dataset, sel_subset_positions)
        if rank == 0:
            print(f"  tau-axis source: "
                  f"{'measured profiles_raw_tau' if has_raw_tau else 'altitude (no raw tau in HDF5)'}")

        ex = {
            'pred'         : found['pred'][pick],
            'pred_std'     : found['pred_std'][pick],
            'tau_pred'     : found['tau_pred'][pick],
            'tau_pred_std' : found['tau_pred_std'][pick],
            'true_um'      : true_um_sel,
            'tau_true'     : tau_true_sel,
            'raw_re'       : raw_re,
            'raw_z'        : raw_z,
            'raw_tau'      : raw_tau,
        }

        out_path = fig_dir / f"profiles_top{rank+1:02d}_run_{rid:03d}.png"
        plot_profiles(ex, row, out_path, n_examples=args.n_examples,
                      show_uncert=(args.show_uncert == 'on'))
        print(f"  rank {rank+1:2d}  run_{rid:03d}  "
              f"RMSE={row['mean_rmse']:.3f} μm   -> {out_path.name}")

        # Cross-run summary panel: take the first profile from the SELECTED
        # subset, so this run's grid panel matches --selection mode (under
        # 'best' it's the run's lowest-RMSE cloud; under 'random' it's a
        # random cloud).
        per_run_examples.append({
            'run_id'      : rid,
            'mean_rmse'   : row['mean_rmse'],
            'true'        : ex['true_um'][0],
            'pred'        : ex['pred'][0],
            'pred_std'    : ex['pred_std'][0],
            'tau_true'    : ex['tau_true'][0],
            'tau_pred'    : ex['tau_pred'][0],
            'tau_pred_std': ex['tau_pred_std'][0],
            'raw_re'      : ex['raw_re'][0],
            'raw_z'       : ex['raw_z'][0],
            'raw_tau'     : ex['raw_tau'][0],
        })

    if per_run_examples:
        grid_out = fig_dir / f"profiles_top{args.top_n:02d}_grid.png"
        plot_grid_single(per_run_examples, grid_out,
                         show_uncert=(args.show_uncert == 'on'))
        print(f"\nGrid figure: {grid_out}")
    print(f"All figures written to: {fig_dir}/")


if __name__ == '__main__':
    main()
