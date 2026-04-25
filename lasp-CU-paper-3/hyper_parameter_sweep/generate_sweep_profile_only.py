"""
generate_sweep_profile_only.py — Sweep generator for the profile-only model.

Sweep axes
----------
Continuous (LHS, 7 dims):
    learning_rate          log-uniform 3e-6 .. 1e-3
    dropout                uniform     0.15 .. 0.40
    lambda_physics         uniform     0.00 .. 0.25
    lambda_monotonicity    uniform     0.00 .. 0.25
    lambda_adiabatic       uniform     0.00 .. 0.25
    lambda_smoothness      uniform     0.00 .. 0.25
    augment_noise_std      uniform     0.000 .. 0.030 (fractional, on top of
                                                       the 0.3 % HySICS noise
                                                       already baked into the
                                                       data)

Categorical (cycled, shuffled):
    hidden_dims    — 5 architectures (carried over from sweep #2's "kept" set)
    level_weights  — 4 schemes: uniform, top, bottom, u_shape
    activation     — gelu, relu, silu
    batch_size     — 128, 256, 512

Per-config K-fold:
    Each config trains `n_folds` independent models (default 5) on profile-aware
    splits and reports `test_mean_rmse_mean ± test_mean_rmse_std`.  Pass
    --n-folds to change.

CLI
---
    python generate_sweep_profile_only.py            # defaults: 150 runs, K=5
    python generate_sweep_profile_only.py --n-folds 3
    python generate_sweep_profile_only.py --n-runs 200 --n-folds 5

Fixed (not part of the sweep)
-----------------------------
    weight_decay        1e-4
    sigma_floor         0.01
    n_epochs            1000
    early_stop_patience 150
    scheduler_patience  30
    warmup_steps        500
    n_test_profiles     14

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import itertools
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────────────────────────────────────
N_LEVELS = 50

FIXED = {
    'data': {
        'h5_path': '/scratch/alpine/anbu8374/neural_network_training_data/'
                   'combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    'output_dir': 'sweep_results_profile_only',
    'n_test_profiles': 14,
}

# ─────────────────────────────────────────────────────────────────────────────
# Categorical axes
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIMS_OPTIONS = [
    [128, 128, 128],                      # 3-layer narrow      (~110K params)
    [256, 256, 256],                      # 3-layer medium      (~350K params)
    [256, 256, 256, 256],                 # 4-layer narrow      (~430K params)
    [512, 512, 256, 256],                 # 4-layer tapered     (~720K params)
    [512, 256, 256, 128, 128],            # 5-layer pyramid     (~440K params)
]

ACTIVATION_OPTIONS = ['gelu', 'relu', 'silu']

BATCH_SIZE_OPTIONS = [128, 256, 512]


def build_level_weights(scheme: str, n_levels: int = N_LEVELS) -> list:
    """
    50-level loss-weighting schemes.

    Convention: index 0 = cloud top, index n_levels-1 = cloud base.
    `ProfileOnlyLoss` normalizes weights to mean 1 internally, so absolute
    scale here is cosmetic — only the *shape* matters.

    Decay length 8 levels gives a ~5:1 weight ratio between the emphasized
    end and the middle for the single-end schemes ('top', 'bottom') and a
    ~3.7:1 ratio for the u-shape (because the two exponentials add a small
    amount in the middle).
    """
    levels = np.arange(n_levels, dtype=float)
    end = n_levels - 1
    decay = 8.0  # levels

    if scheme == 'uniform':
        w = np.ones(n_levels)
    elif scheme == 'top':
        w = 1.0 + 4.0 * np.exp(-levels / decay)
    elif scheme == 'bottom':
        w = 1.0 + 4.0 * np.exp(-(end - levels) / decay)
    elif scheme == 'u_shape':
        w = 1.0 + 4.0 * (np.exp(-levels / decay)
                         + np.exp(-(end - levels) / decay))
    else:
        raise ValueError(f"unknown level-weight scheme: {scheme!r}")
    return [float(x) for x in w]


LEVEL_WEIGHT_SCHEMES = ['uniform', 'top', 'bottom', 'u_shape']

# ─────────────────────────────────────────────────────────────────────────────
# Continuous search space (LHS)
# ─────────────────────────────────────────────────────────────────────────────
LR_LOG10_RANGE             = [-5.52, -3.0]   # 3e-6 to 1e-3
DROPOUT_RANGE              = [0.15, 0.40]
LAMBDA_PHYSICS_RANGE       = [0.0, 0.25]
LAMBDA_MONOTONICITY_RANGE  = [0.0, 0.25]
LAMBDA_ADIABATIC_RANGE     = [0.0, 0.25]
LAMBDA_SMOOTHNESS_RANGE    = [0.0, 0.25]
AUGMENT_NOISE_RANGE        = [0.0, 0.03]

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults (constant across runs)
# ─────────────────────────────────────────────────────────────────────────────
BASE_TRAINING_DEFAULTS = {
    'n_epochs': 1000,
    'scheduler_patience': 30,
    'early_stop_patience': 150,
    'warmup_steps': 500,
    'weight_decay': 1e-4,
    'sigma_floor': 0.01,
    # batch_size and augment_noise_std are now per-run (categorical / LHS).
}

SEED = 42


def latin_hypercube_sample(n, d, rng):
    """Latin Hypercube Sampling: n samples in d dims, uniform in [0,1]."""
    result = np.zeros((n, d))
    for dim in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            result[perm[i], dim] = (i + rng.random()) / n
    return result


def generate_configs(n_runs: int, n_folds: int):
    rng = np.random.default_rng(SEED)
    configs = []

    # Categorical combos: 5 arch × 4 weight × 3 act × 3 bs = 180 unique combos.
    cat_combos = list(itertools.product(
        range(len(HIDDEN_DIMS_OPTIONS)),
        LEVEL_WEIGHT_SCHEMES,
        ACTIVATION_OPTIONS,
        BATCH_SIZE_OPTIONS,
    ))
    rng.shuffle(cat_combos)

    # Cycle through combos so coverage is even.
    cat_assignment = []
    while len(cat_assignment) < n_runs:
        block = cat_combos.copy()
        rng.shuffle(block)
        cat_assignment.extend(block)
    cat_assignment = cat_assignment[:n_runs]

    # 7 continuous dims: LR, dropout, l_phys, l_mono, l_adi, l_sm, augment_noise
    lhs = latin_hypercube_sample(n_runs, 7, rng)

    for i in range(n_runs):
        arch_idx, weight_name, act_name, batch_size = cat_assignment[i]
        hidden_dims   = HIDDEN_DIMS_OPTIONS[arch_idx]
        level_weights = build_level_weights(weight_name)

        lr = 10 ** (LR_LOG10_RANGE[0]
                    + lhs[i, 0] * (LR_LOG10_RANGE[1] - LR_LOG10_RANGE[0]))
        dropout = (DROPOUT_RANGE[0]
                   + lhs[i, 1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]))
        lambda_physics = (LAMBDA_PHYSICS_RANGE[0]
                          + lhs[i, 2] * (LAMBDA_PHYSICS_RANGE[1] - LAMBDA_PHYSICS_RANGE[0]))
        lambda_monotonicity = (LAMBDA_MONOTONICITY_RANGE[0]
                               + lhs[i, 3] * (LAMBDA_MONOTONICITY_RANGE[1] - LAMBDA_MONOTONICITY_RANGE[0]))
        lambda_adiabatic = (LAMBDA_ADIABATIC_RANGE[0]
                            + lhs[i, 4] * (LAMBDA_ADIABATIC_RANGE[1] - LAMBDA_ADIABATIC_RANGE[0]))
        lambda_smoothness = (LAMBDA_SMOOTHNESS_RANGE[0]
                             + lhs[i, 5] * (LAMBDA_SMOOTHNESS_RANGE[1] - LAMBDA_SMOOTHNESS_RANGE[0]))
        augment_noise_std = (AUGMENT_NOISE_RANGE[0]
                             + lhs[i, 6] * (AUGMENT_NOISE_RANGE[1] - AUGMENT_NOISE_RANGE[0]))

        # Round for readability
        lr                  = float(f"{lr:.6f}")
        dropout             = float(f"{dropout:.3f}")
        lambda_physics      = float(f"{lambda_physics:.3f}")
        lambda_monotonicity = float(f"{lambda_monotonicity:.3f}")
        lambda_adiabatic    = float(f"{lambda_adiabatic:.3f}")
        lambda_smoothness   = float(f"{lambda_smoothness:.3f}")
        augment_noise_std   = float(f"{augment_noise_std:.4f}")

        # Architecture descriptor
        n_layers = len(hidden_dims)
        width = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_str = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_str = "tapered"
        else:
            arch_str = "custom"

        run_name = (f"{arch_str}_{act_name}_{weight_name}_b{batch_size}_"
                    f"n{augment_noise_std:.3f}_dr{dropout:.2f}_lr{lr:.1e}_"
                    f"lphys{lambda_physics:.2f}_lmono{lambda_monotonicity:.2f}_"
                    f"ladi{lambda_adiabatic:.2f}_lsm{lambda_smoothness:.2f}")

        cfg = {
            'run_id': i,
            'run_name': run_name,
            **FIXED,
            'hyperparams': {
                'hidden_dims': hidden_dims,
                'activation': act_name,
                'dropout': dropout,
                'learning_rate': lr,
                'lambda_physics': lambda_physics,
                'lambda_monotonicity': lambda_monotonicity,
                'lambda_adiabatic': lambda_adiabatic,
                'lambda_smoothness': lambda_smoothness,
                'level_weights': level_weights,
                'level_weights_name': weight_name,
                'batch_size': batch_size,
                'augment_noise_std': augment_noise_std,
                'n_folds': n_folds,
                **BASE_TRAINING_DEFAULTS,
            },
        }
        configs.append(cfg)

    return configs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--n-runs', type=int, default=150,
                   help='number of sweep configurations (default 150)')
    p.add_argument('--n-folds', type=int, default=5,
                   help='K for K-fold CV per config (default 5)')
    p.add_argument('--out-dir', type=str, default='sweep_configs_profile_only',
                   help='output directory for the JSON config files')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for old in out_dir.glob('run_*.json'):
        old.unlink()
    if (out_dir / 'sweep_summary.json').exists():
        (out_dir / 'sweep_summary.json').unlink()

    configs = generate_configs(args.n_runs, args.n_folds)

    for cfg in configs:
        path = out_dir / f"run_{cfg['run_id']:03d}.json"
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    summary = []
    for cfg in configs:
        hp = cfg['hyperparams']
        summary.append({
            'run_id': cfg['run_id'],
            'run_name': cfg['run_name'],
            'hidden_dims': hp['hidden_dims'],
            'activation': hp['activation'],
            'level_weights_name': hp['level_weights_name'],
            'batch_size': hp['batch_size'],
            'augment_noise_std': hp['augment_noise_std'],
            'dropout': hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'lambda_physics': hp['lambda_physics'],
            'lambda_monotonicity': hp['lambda_monotonicity'],
            'lambda_adiabatic': hp['lambda_adiabatic'],
            'lambda_smoothness': hp['lambda_smoothness'],
        })

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(configs)} sweep configurations in {out_dir}/  "
          f"(K-fold = {args.n_folds})\n")
    print(f"{'ID':>3}  {'Architecture':<18} {'Act':<5} {'Weights':<9} "
          f"{'BS':>4} {'Noise':>6} {'Drop':>5} {'LR':>10} "
          f"{'l_phys':>7} {'l_mono':>7} {'l_adi':>7} {'l_sm':>7}")
    print("-" * 130)
    for s in summary:
        dims = s['hidden_dims']
        n = len(dims)
        w = dims[0]
        if all(h == w for h in dims):
            arch = f"{n}x{w}"
        elif dims[0] > dims[-1]:
            arch = f"{dims[0]}->{dims[-1]} ({n}L)"
        else:
            arch = f"custom-{n}L"
        print(f"{s['run_id']:3d}  {arch:<18} {s['activation']:<5} "
              f"{s['level_weights_name']:<9} {s['batch_size']:4d} "
              f"{s['augment_noise_std']:6.3f} {s['dropout']:5.2f} "
              f"{s['learning_rate']:10.6f} "
              f"{s['lambda_physics']:7.3f} {s['lambda_monotonicity']:7.3f} "
              f"{s['lambda_adiabatic']:7.3f} {s['lambda_smoothness']:7.3f}")

    print(f"\n{'=' * 70}")
    print("Coverage check (LHS spans):")
    print(f"  Learning rate:        {min(s['learning_rate'] for s in summary):.2e} "
          f"to {max(s['learning_rate'] for s in summary):.2e}")
    print(f"  Dropout:              {min(s['dropout'] for s in summary):.3f} "
          f"to {max(s['dropout'] for s in summary):.3f}")
    print(f"  lambda_physics:       {min(s['lambda_physics'] for s in summary):.3f} "
          f"to {max(s['lambda_physics'] for s in summary):.3f}")
    print(f"  lambda_monotonicity:  {min(s['lambda_monotonicity'] for s in summary):.3f} "
          f"to {max(s['lambda_monotonicity'] for s in summary):.3f}")
    print(f"  lambda_adiabatic:     {min(s['lambda_adiabatic'] for s in summary):.3f} "
          f"to {max(s['lambda_adiabatic'] for s in summary):.3f}")
    print(f"  lambda_smoothness:    {min(s['lambda_smoothness'] for s in summary):.3f} "
          f"to {max(s['lambda_smoothness'] for s in summary):.3f}")
    print(f"  augment_noise_std:    {min(s['augment_noise_std'] for s in summary):.4f} "
          f"to {max(s['augment_noise_std'] for s in summary):.4f}")
    print(f"\nCategorical counts:")
    for dims in HIDDEN_DIMS_OPTIONS:
        c = sum(1 for s in summary if s['hidden_dims'] == dims)
        print(f"  hidden_dims={dims}: {c} runs")
    for name in LEVEL_WEIGHT_SCHEMES:
        c = sum(1 for s in summary if s['level_weights_name'] == name)
        print(f"  level_weights={name}: {c} runs")
    for act in ACTIVATION_OPTIONS:
        c = sum(1 for s in summary if s['activation'] == act)
        print(f"  activation={act}: {c} runs")
    for bs in BATCH_SIZE_OPTIONS:
        c = sum(1 for s in summary if s['batch_size'] == bs)
        print(f"  batch_size={bs}: {c} runs")

    print(f"\nFixed across all runs:")
    for k, v in BASE_TRAINING_DEFAULTS.items():
        print(f"  {k}: {v}")
    print(f"  n_test_profiles: {FIXED['n_test_profiles']}")
    print(f"  n_folds: {args.n_folds}")
    print(f"  data file: {FIXED['data']['h5_path']}")

    print(f"\nNext steps:")
    print(f"  1. Update sweep_alpine.sh: --array=0-{args.n_runs - 1}, "
          f"trainer=sweep_train_profile_only.py, configs={out_dir.name}/.")
    print(f"  2. Upload {out_dir.name}/ to Alpine.")
    print(f"  3. sbatch sweep_alpine.sh.")
    print(f"\nNote: each config runs {args.n_folds} folds, so per-job wall time "
          f"is ~{args.n_folds}× sweep #2's per-config time.")


if __name__ == '__main__':
    main()
