"""
generate_sweep.py — Generate 100 hyperparameter configurations for Alpine sweep.

Produces:
    sweep_configs/run_000.json  through  run_099.json
    sweep_configs/sweep_summary.json  (human-readable table of all runs)

Strategy:
    - Latin Hypercube Sampling (LHS) for continuous parameters: ensures even
      coverage of the search space without random clustering. (See McKay,
      Beckman & Conover 1979; standard practice in surrogate modelling and
      hyperparameter search.)
    - Systematic cycling over categorical choices (architecture, level_weights)
      so each combination appears multiple times with different continuous
      params — lets you separate categorical effects from continuous ones.

Hyperparameter ranges chosen from common scientific ML conventions:
    - Learning rate: 1e-5 to 5e-3 (log scale).  Standard range for AdamW on
      regression problems.  Bengio (2012, "Practical recommendations") and
      Smith (2017, LR range test) commonly cite 1e-4 to 1e-3 as the "safe"
      band; we extend to bracket it on both sides.
    - Dropout: 0.0 to 0.4.  Srivastava et al. (2014); Goodfellow Ch. 7.12.
      Higher than 0.5 typically harms regression tasks.
    - Weight decay: 1e-6 to 1e-2 (log scale).  Loshchilov & Hutter (2019,
      AdamW) recommend 1e-2 for vision; smaller values for regression.
    - Hidden width: 128 to 512.  Width-depth tradeoff per He et al. 2016.
    - Depth: 3 to 6 layers.  PINN papers (Raissi et al. 2019; Karniadakis
      et al. 2021 review) commonly use 4–8 layers for similar problems.
    - Batch size: 128, 256, 512.  Powers of 2; larger batches give smoother
      gradients but worse generalization (Keskar et al. 2017).

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
# Categorical search space
# ─────────────────────────────────────────────────────────────────────────────

# Architecture options.
# Range from compact (3x128, ~200K params) to large (6x512, ~1.5M params).
# This brackets the "right size" for ~100K training samples per the rule of
# thumb that #params should be 1-10x the dataset size for regression tasks
# without strong overfitting.  Includes both uniform and tapered designs to
# test whether bottlenecking layers help with the high-dim spectral input.
HIDDEN_DIMS_OPTIONS = [
    [128, 128, 128],                      # 3-layer narrow      (~110K params)
    [256, 256, 256],                      # 3-layer medium      (~350K params)
    [256, 256, 256, 256],                 # 4-layer narrow      (~430K params)
    [512, 512, 512, 512],                 # 4-layer wide        (~1.1M params)
    [512, 512, 256, 256],                 # 4-layer tapered     (~720K params)
    [256, 256, 256, 256, 256, 256],       # 6-layer narrow      (~600K params)
    [512, 512, 512, 512, 512],            # 5-layer wide        (~1.4M params)
    [512, 256, 256, 128, 128],            # 5-layer pyramid     (~440K params)
]

# Level weight schemes.
# Cloud-base levels are physically harder (less photon penetration in SWIR),
# so ends/strong_ends emphasize them.  We also test a "deep" scheme that
# specifically biases toward the deeper levels (4–10).
LEVEL_WEIGHTS_OPTIONS = {
    'uniform':     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'ends':        [4.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 3.0],
    'top':         [6.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'strong_ends': [6.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0],
    'deep':        [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
}

# Batch size options.  Smaller batches inject more noise (regularizing effect)
# but train more slowly per epoch.  Keskar et al. (2017) showed large batches
# tend to find sharp minima that generalize worse.
BATCH_SIZE_OPTIONS = [128, 256, 512]

# ─────────────────────────────────────────────────────────────────────────────
# Continuous search space (sampled via Latin Hypercube)
# ─────────────────────────────────────────────────────────────────────────────
# All continuous ranges follow scientific ML conventions cited above.
LR_LOG10_RANGE         = [-5.0, -2.3]    # 1e-5 to ~5e-3
DROPOUT_RANGE          = [0.00, 0.40]    # standard regression dropout
WEIGHT_DECAY_LOG10_RANGE = [-6.0, -2.0]  # 1e-6 to 1e-2
SIGMA_FLOOR_RANGE      = [0.005, 0.05]   # in normalized [0,1] space
                                          # 0.005 → ~0.24 um, 0.05 → ~2.4 um

# ─────────────────────────────────────────────────────────────────────────────
# Training settings (constant across runs for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_DEFAULTS = {
    'n_epochs': 400,
    'scheduler_patience': 30,
    'early_stop_patience': 40,
    'lambda_physics': 0.1,
    'lambda_monotonicity': 0.0,
    'lambda_adiabatic': 0.1,
    'lambda_smoothness': 0.1,
}

N_RUNS = 100
SEED = 42


def latin_hypercube_sample(n, d, rng):
    """
    Generate n samples in d dimensions using Latin Hypercube Sampling.

    LHS divides each dimension into n equal-probability strata and ensures
    one sample per stratum per dimension. This guarantees marginal coverage
    that is much more even than naive random sampling, especially for small n.

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

    # Categorical combinations.  8 architectures × 5 weight schemes × 3 batch
    # sizes = 120 total; we cycle through and sample 100, so most combos appear
    # ~once with different continuous params.
    categorical_combos = list(itertools.product(
        range(len(HIDDEN_DIMS_OPTIONS)),
        list(LEVEL_WEIGHTS_OPTIONS.keys()),
        BATCH_SIZE_OPTIONS,
    ))
    # Shuffle so cycling through doesn't correlate index with any category
    rng.shuffle(categorical_combos)
    n_categorical = len(categorical_combos)

    # 4 continuous dims: LR, dropout, weight_decay, sigma_floor
    lhs = latin_hypercube_sample(N_RUNS, 4, rng)

    for i in range(N_RUNS):
        # Cycle through the (shuffled) categorical combos
        arch_idx, weight_name, batch_size = categorical_combos[i % n_categorical]

        hidden_dims   = HIDDEN_DIMS_OPTIONS[arch_idx]
        level_weights = LEVEL_WEIGHTS_OPTIONS[weight_name]

        # Map LHS samples to physical ranges
        lr = 10 ** (LR_LOG10_RANGE[0]
                    + lhs[i, 0] * (LR_LOG10_RANGE[1] - LR_LOG10_RANGE[0]))
        dropout = (DROPOUT_RANGE[0]
                   + lhs[i, 1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]))
        weight_decay = 10 ** (WEIGHT_DECAY_LOG10_RANGE[0]
                              + lhs[i, 2] * (WEIGHT_DECAY_LOG10_RANGE[1] - WEIGHT_DECAY_LOG10_RANGE[0]))
        sigma_floor = (SIGMA_FLOOR_RANGE[0]
                       + lhs[i, 3] * (SIGMA_FLOOR_RANGE[1] - SIGMA_FLOOR_RANGE[0]))

        # Round for readability
        lr           = float(f"{lr:.6f}")
        dropout      = float(f"{dropout:.3f}")
        weight_decay = float(f"{weight_decay:.2e}")
        sigma_floor  = float(f"{sigma_floor:.4f}")

        # Architecture descriptor
        n_layers = len(hidden_dims)
        width    = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_str = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_str = "tapered"
        else:
            arch_str = "custom"

        run_name = (f"{arch_str}_dr{dropout:.2f}_lr{lr:.1e}_"
                    f"wd{weight_decay:.0e}_b{batch_size}_{weight_name}")

        cfg = {
            'run_id': i,
            'run_name': run_name,
            **FIXED,
            'hyperparams': {
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'sigma_floor': sigma_floor,
                'batch_size': batch_size,
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

    # Clean up any leftover configs from a previous (smaller) sweep
    for old in out_dir.glob('run_*.json'):
        old.unlink()
    if (out_dir / 'sweep_summary.json').exists():
        (out_dir / 'sweep_summary.json').unlink()

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
            'weight_decay': hp['weight_decay'],
            'sigma_floor': hp['sigma_floor'],
            'batch_size': hp['batch_size'],
            'level_weights_name': hp['level_weights_name'],
        })

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"Generated {len(configs)} sweep configurations in {out_dir}/\n")
    print(f"{'ID':>3}  {'Architecture':<22} {'Drop':>5} {'LR':>10} "
          f"{'WD':>10} {'sigma_fl':>9} {'BS':>4} {'Weights':<12}")
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
              f"{s['learning_rate']:10.6f} {s['weight_decay']:10.2e} "
              f"{s['sigma_floor']:9.4f} {s['batch_size']:4d} "
              f"{s['level_weights_name']:<12}")

    # Print summary statistics so the user can confirm coverage
    print(f"\n{'='*70}")
    print("Coverage check (Latin Hypercube spans these ranges):")
    print(f"  Learning rate: {min(s['learning_rate'] for s in summary):.2e} "
          f"to {max(s['learning_rate'] for s in summary):.2e}")
    print(f"  Dropout:       {min(s['dropout'] for s in summary):.3f} "
          f"to {max(s['dropout'] for s in summary):.3f}")
    print(f"  Weight decay:  {min(s['weight_decay'] for s in summary):.2e} "
          f"to {max(s['weight_decay'] for s in summary):.2e}")
    print(f"  Sigma floor:   {min(s['sigma_floor'] for s in summary):.4f} "
          f"to {max(s['sigma_floor'] for s in summary):.4f}")
    print(f"\nCategorical counts:")
    for name in LEVEL_WEIGHTS_OPTIONS:
        c = sum(1 for s in summary if s['level_weights_name'] == name)
        print(f"  weights={name}: {c} runs")
    for bs in BATCH_SIZE_OPTIONS:
        c = sum(1 for s in summary if s['batch_size'] == bs)
        print(f"  batch_size={bs}: {c} runs")

    print(f"\nNext steps:")
    print(f"  1. Update sweep_alpine.sh: change `--array=0-49%16` to `--array=0-99%16`")
    print(f"  2. Upload sweep_configs/ to Alpine")
    print(f"  3. sbatch sweep_alpine.sh")


if __name__ == '__main__':
    main()
