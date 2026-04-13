"""
generate_sweep.py — Generate 50 hyperparameter configurations for Alpine sweep.

Produces:
    sweep_configs/run_000.json  through  run_049.json
    sweep_configs/sweep_summary.json  (human-readable table of all runs)

Strategy:
    - Uses Latin Hypercube Sampling for continuous parameters (LR, dropout)
      to ensure good coverage without pure random clustering.
    - Systematic grid over categorical choices (hidden_dims, level_weights)
      crossed with sampled continuous params.
    - sigma_floor is swept because it directly controls the NLL stability.

Run this on your MacBook before uploading to Alpine:
    python generate_sweep.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import json
import itertools
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings (same for all runs)
# ─────────────────────────────────────────────────────────────────────────────
FIXED = {
    'data': {
        'h5_path': '/scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_13_April_2026.h5',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    'output_dir': 'sweep_results',
}

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ─────────────────────────────────────────────────────────────────────────────

# Architecture options (categorical)
HIDDEN_DIMS_OPTIONS = [
    [256, 256, 256, 256],        # baseline narrow
    [512, 512, 512, 512],        # current default — uniform wide
    [512, 512, 256, 256],        # tapered (previous default)
    [256, 256, 256, 256, 256],   # deeper narrow (5 layers)
    [512, 512, 512, 512, 512],   # deeper wide (5 layers)
]

# Level weight schemes (categorical)
#   - uniform: equal weight to all levels
#   - ends:    emphasize cloud top + base (where retrieval is hardest)
#   - top:     emphasize cloud top (most observable from satellite)
#   - strong_ends: heavily emphasize boundaries
LEVEL_WEIGHTS_OPTIONS = {
    'uniform':     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'ends':        [4.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 3.0],
    'top':         [6.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'strong_ends': [6.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0],
}

# Continuous parameters (sampled via Latin Hypercube)
# [min, max] in log10 space for LR; linear space for others
LR_LOG10_RANGE     = [-4.0, -2.5]   # 1e-4 to ~3e-3
DROPOUT_RANGE      = [0.05, 0.30]
SIGMA_FLOOR_RANGE  = [0.005, 0.05]  # in normalized [0,1] space
                                     # 0.005 → ~0.24 um, 0.05 → ~2.4 um physical

# Training settings (can be overridden per-run but kept constant for fair comparison)
TRAINING_DEFAULTS = {
    'n_epochs': 400,
    'batch_size': 256,
    'weight_decay': 1e-4,
    'scheduler_patience': 30,
    'early_stop_patience': 80,
    'lambda_physics': 0.1,
    'lambda_monotonicity': 0.0,
    'lambda_adiabatic': 0.1,
    'lambda_smoothness': 0.1,
}

N_RUNS = 50
SEED = 42


def latin_hypercube_sample(n, d, rng):
    """
    Generate n samples in d dimensions using Latin Hypercube Sampling.
    Returns array of shape (n, d) with values in [0, 1].
    """
    result = np.zeros((n, d))
    for dim in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            result[perm[i], dim] = (i + rng.random()) / n
    return result


def generate_configs():
    rng = np.random.default_rng(SEED)
    configs = []

    # ── Strategy: systematic × sampled ─────────────────────────────────────
    # We have 5 architectures × 4 weight schemes = 20 categorical combos.
    # For 50 runs, we cycle through these and sample continuous params.

    categorical_combos = list(itertools.product(
        range(len(HIDDEN_DIMS_OPTIONS)),
        list(LEVEL_WEIGHTS_OPTIONS.keys()),
    ))
    # 20 combos, so each appears 2–3 times with different continuous params
    n_categorical = len(categorical_combos)

    # Latin Hypercube for 3 continuous dimensions: LR, dropout, sigma_floor
    lhs = latin_hypercube_sample(N_RUNS, 3, rng)

    for i in range(N_RUNS):
        # Cycle through categorical combos
        arch_idx, weight_name = categorical_combos[i % n_categorical]

        hidden_dims = HIDDEN_DIMS_OPTIONS[arch_idx]
        level_weights = LEVEL_WEIGHTS_OPTIONS[weight_name]

        # Map LHS samples to physical ranges
        lr = 10 ** (LR_LOG10_RANGE[0] + lhs[i, 0] * (LR_LOG10_RANGE[1] - LR_LOG10_RANGE[0]))
        dropout = DROPOUT_RANGE[0] + lhs[i, 1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0])
        sigma_floor = SIGMA_FLOOR_RANGE[0] + lhs[i, 2] * (SIGMA_FLOOR_RANGE[1] - SIGMA_FLOOR_RANGE[0])

        # Round for readability
        lr = float(f"{lr:.6f}")
        dropout = float(f"{dropout:.3f}")
        sigma_floor = float(f"{sigma_floor:.4f}")

        n_layers = len(hidden_dims)
        width = hidden_dims[0]
        arch_str = f"{n_layers}x{width}" if all(h == width for h in hidden_dims) else "tapered"
        run_name = f"{arch_str}_dr{dropout:.2f}_lr{lr:.1e}_{weight_name}"

        cfg = {
            'run_id': i,
            'run_name': run_name,
            **FIXED,
            'hyperparams': {
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'learning_rate': lr,
                'sigma_floor': sigma_floor,
                'level_weights': level_weights,
                'level_weights_name': weight_name,
                **TRAINING_DEFAULTS,
            },
        }
        configs.append(cfg)

    return configs


def main():
    out_dir = Path('sweep_configs')
    out_dir.mkdir(exist_ok=True)

    configs = generate_configs()

    # Save individual config files
    for cfg in configs:
        path = out_dir / f"run_{cfg['run_id']:03d}.json"
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    # Save human-readable summary
    summary = []
    for cfg in configs:
        hp = cfg['hyperparams']
        summary.append({
            'run_id': cfg['run_id'],
            'run_name': cfg['run_name'],
            'hidden_dims': hp['hidden_dims'],
            'dropout': hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'sigma_floor': hp['sigma_floor'],
            'level_weights_name': hp['level_weights_name'],
        })

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"Generated {len(configs)} sweep configurations in {out_dir}/\n")
    print(f"{'ID':>3}  {'Architecture':<16} {'Dropout':>7} {'LR':>10} "
          f"{'sigma_floor':>11} {'Weights':<12}")
    print("-" * 70)
    for s in summary:
        dims = s['hidden_dims']
        n = len(dims)
        w = dims[0]
        arch = f"{n}x{w}" if all(h == w for h in dims) else f"{dims}"
        print(f"{s['run_id']:3d}  {arch:<16} {s['dropout']:7.3f} "
              f"{s['learning_rate']:10.6f} {s['sigma_floor']:11.4f} "
              f"{s['level_weights_name']:<12}")

    print(f"\nTo run on Alpine: sbatch sweep_alpine.sh")


if __name__ == '__main__':
    main()
