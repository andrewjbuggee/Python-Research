"""
generate_sweep_profile_only.py — Sweep generator for the profile-only model.

What this sweeps (per the new experimental plan)
-------------------------------------------------
Continuous (LHS, 7 dims):
    learning_rate          log-uniform 3e-6 .. 1e-3
    dropout                uniform     0.15 .. 0.40
    lambda_physics         uniform     0.00 .. 0.25
    lambda_monotonicity    uniform     0.00 .. 0.25
    lambda_adiabatic       uniform     0.00 .. 0.25
    lambda_smoothness      uniform     0.00 .. 0.25

Categorical (cycled, shuffled):
    hidden_dims (architecture) — five options carried over from sweep #2.

Per-config K-fold:
    Each config trains `n_folds=5` independent models on profile-aware splits
    and reports `test_mean_rmse_mean ± test_mean_rmse_std`.  This gives a
    real uncertainty bar on every config rather than a single point estimate.

Fixed (constant across all configs)
-----------------------------------
    weight_decay        1e-4      (sensible default; not part of this sweep)
    sigma_floor         0.01      (matches DropletProfileNetwork log-std clamp)
    augment_noise_std   0.000     (data already has 0.3% HySICS noise baked in
                                   and we are not retraining the augmentation
                                   axis)
    batch_size          256
    level_weights       uniform (50 entries of 1.0)
    n_epochs            1000
    early_stop_patience 150
    scheduler_patience  30
    warmup_steps        500
    n_folds             5
    n_test_profiles     14        (~5 % of unique profiles — fixed across folds)

Why uniform 50-level weights
----------------------------
The point of the 50-level grid is to get *closer to the true in-situ profile*
during training (mean sampling RMSE 0.19 μm vs ~1 μm at 7 levels).  Once the
network is trained, profile information content is set by the spectrum, not
by how we weight the loss — so we start with uniform weighting and let the
PCA/Platnick analyses guide a follow-up sweep on level_weights if it turns
out post-training-RMSE structure is unevenly distributed across levels.

Target file
-----------
The 50-evenZ-levels HDF5 (combined VOCALS + ORACLES, 23 April 2026).

Run locally:
    python generate_sweep_profile_only.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

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
        # 50-evenZ-levels file.  sweep_train_profile_only.py reads this from
        # cfg['data']['h5_path']; --training-data-dir overrides directory only.
        'h5_path': '/scratch/alpine/anbu8374/neural_network_training_data/'
                   'combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    'output_dir': 'sweep_results_profile_only',
    'n_test_profiles': 14,
}

# ─────────────────────────────────────────────────────────────────────────────
# Categorical: architecture (carried over from sweep #2's "kept" set)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIMS_OPTIONS = [
    [128, 128, 128],                      # 3-layer narrow      (~110K params)
    [256, 256, 256],                      # 3-layer medium      (~350K params)
    [256, 256, 256, 256],                 # 4-layer narrow      (~430K params)
    [512, 512, 256, 256],                 # 4-layer tapered     (~720K params)
    [512, 256, 256, 128, 128],            # 5-layer pyramid     (~440K params)
]

# ─────────────────────────────────────────────────────────────────────────────
# Continuous search space (LHS)
# ─────────────────────────────────────────────────────────────────────────────
LR_LOG10_RANGE             = [-5.52, -3.0]   # 3e-6 to 1e-3
DROPOUT_RANGE              = [0.15, 0.40]
LAMBDA_PHYSICS_RANGE       = [0.0, 0.25]
LAMBDA_MONOTONICITY_RANGE  = [0.0, 0.25]
LAMBDA_ADIABATIC_RANGE     = [0.0, 0.25]
LAMBDA_SMOOTHNESS_RANGE    = [0.0, 0.25]

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults (constant across runs)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_DEFAULTS = {
    'n_epochs': 1000,
    'scheduler_patience': 30,
    'early_stop_patience': 150,
    'warmup_steps': 500,
    # Held fixed (not in the user-requested sweep axes):
    'weight_decay': 1e-4,
    'sigma_floor': 0.01,
    'augment_noise_std': 0.0,
    'batch_size': 256,
    'level_weights': [1.0] * N_LEVELS,
    'level_weights_name': 'uniform',
    'n_folds': 5,
}

N_RUNS = 100
SEED = 42


def latin_hypercube_sample(n, d, rng):
    """Latin Hypercube Sampling: n samples in d dims, uniform in [0,1]."""
    result = np.zeros((n, d))
    for dim in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            result[perm[i], dim] = (i + rng.random()) / n
    return result


def generate_configs():
    rng = np.random.default_rng(SEED)
    configs = []

    # Categorical: just architecture (5 options).  Cycle and shuffle.
    arch_indices = list(range(len(HIDDEN_DIMS_OPTIONS)))
    cycled = []
    while len(cycled) < N_RUNS:
        block = arch_indices.copy()
        rng.shuffle(block)
        cycled.extend(block)
    cycled = cycled[:N_RUNS]

    # 6 continuous dims: LR, dropout, l_phys, l_mono, l_adi, l_sm
    lhs = latin_hypercube_sample(N_RUNS, 6, rng)

    for i in range(N_RUNS):
        hidden_dims = HIDDEN_DIMS_OPTIONS[cycled[i]]

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

        # Round for readability
        lr                  = float(f"{lr:.6f}")
        dropout             = float(f"{dropout:.3f}")
        lambda_physics      = float(f"{lambda_physics:.3f}")
        lambda_monotonicity = float(f"{lambda_monotonicity:.3f}")
        lambda_adiabatic    = float(f"{lambda_adiabatic:.3f}")
        lambda_smoothness   = float(f"{lambda_smoothness:.3f}")

        # Architecture descriptor
        n_layers = len(hidden_dims)
        width = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_str = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_str = "tapered"
        else:
            arch_str = "custom"

        run_name = (f"{arch_str}_dr{dropout:.2f}_lr{lr:.1e}_"
                    f"lphys{lambda_physics:.2f}_lmono{lambda_monotonicity:.2f}_"
                    f"ladi{lambda_adiabatic:.2f}_lsm{lambda_smoothness:.2f}")

        cfg = {
            'run_id': i,
            'run_name': run_name,
            **FIXED,
            'hyperparams': {
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'learning_rate': lr,
                'lambda_physics': lambda_physics,
                'lambda_monotonicity': lambda_monotonicity,
                'lambda_adiabatic': lambda_adiabatic,
                'lambda_smoothness': lambda_smoothness,
                **TRAINING_DEFAULTS,
            },
        }
        configs.append(cfg)

    return configs


def main():
    out_dir = Path('sweep_configs_profile_only')
    out_dir.mkdir(exist_ok=True)

    for old in out_dir.glob('run_*.json'):
        old.unlink()
    if (out_dir / 'sweep_summary.json').exists():
        (out_dir / 'sweep_summary.json').unlink()

    configs = generate_configs()

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
            'dropout': hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'lambda_physics': hp['lambda_physics'],
            'lambda_monotonicity': hp['lambda_monotonicity'],
            'lambda_adiabatic': hp['lambda_adiabatic'],
            'lambda_smoothness': hp['lambda_smoothness'],
        })

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"Generated {len(configs)} sweep configurations in {out_dir}/\n")
    print(f"{'ID':>3}  {'Architecture':<22} {'Drop':>5} {'LR':>10} "
          f"{'l_phys':>7} {'l_mono':>7} {'l_adi':>7} {'l_sm':>7}")
    print("-" * 90)
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
        print(f"{s['run_id']:3d}  {arch:<22} {s['dropout']:5.2f} "
              f"{s['learning_rate']:10.6f} "
              f"{s['lambda_physics']:7.3f} {s['lambda_monotonicity']:7.3f} "
              f"{s['lambda_adiabatic']:7.3f} {s['lambda_smoothness']:7.3f}")

    # Coverage check
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
    print(f"\nArchitecture counts:")
    for idx, dims in enumerate(HIDDEN_DIMS_OPTIONS):
        c = sum(1 for s in summary if s['hidden_dims'] == dims)
        print(f"  {dims}: {c} runs")

    print(f"\nFixed across all runs:")
    for k, v in TRAINING_DEFAULTS.items():
        if k == 'level_weights':
            print(f"  {k}: uniform x{N_LEVELS}")
        else:
            print(f"  {k}: {v}")
    print(f"  n_test_profiles: {FIXED['n_test_profiles']}")
    print(f"  data file: {FIXED['data']['h5_path']}")

    print(f"\nNext steps:")
    print(f"  1. Update sweep_alpine.sh: --array=0-{N_RUNS - 1} and point at "
          f"sweep_configs_profile_only/, sweep_train_profile_only.py.")
    print(f"  2. Upload sweep_configs_profile_only/ to Alpine.")
    print(f"  3. sbatch sweep_alpine.sh.")
    print(f"\nNote: each config now runs {TRAINING_DEFAULTS['n_folds']} folds, so "
          f"per-job wall time is ~{TRAINING_DEFAULTS['n_folds']}x what sweep #2 took.")


if __name__ == '__main__':
    main()
