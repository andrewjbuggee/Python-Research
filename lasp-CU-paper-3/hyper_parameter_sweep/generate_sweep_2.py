"""
generate_sweep_2.py — Second-round hyperparameter sweep (100 configurations).

Design changes vs. the first sweep (see generate_sweep.py):

  1. Learning rate range narrowed and shifted down.
       was: 1e-5  to 5e-3   (log)
       now: 3e-6  to 1e-3   (log)
     Rationale: 7 of the top-10 runs from sweep #1 had LR < 1e-4 and four were
     below 4e-5.  The upper half of the old range was essentially wasted.

  2. Largest architectures removed.
       Dropped: [512,512,512,512], [512,512,512,512,512], [256]*6
       Kept:    3x128, 3x256, 4x256, 5-layer tapered, 5-layer pyramid
     Rationale: The three largest networks were all outside the top-10 of
     sweep #1.  With ~100K training samples and only ~300 unique profiles,
     extra capacity is wasted — the network is not optimization-limited,
     it is data-limited.

  3. Dropout narrowed: 0.00–0.40  →  0.15–0.40.
     Rationale: runs with dropout < 0.15 clustered in the bottom half of
     sweep #1 rankings.  This is an overfitting problem; give the regularizer
     room.

  4. n_epochs raised 400 → 1000, early-stop patience 80 → 150.
     Rationale: sweep #1 median best-epoch was 3 — the network was memorizing
     immediately and stopping well before LR schedules or regularization had
     time to act.  Noise augmentation (#6) and warmup (#7) are expected to
     delay overfitting; we need the training budget to actually see it.

  5. Level-weight schemes kept (uniform, ends, top, strong_ends, deep).
     Rationale: uniform won in sweep #1, but the margin was small; the user
     requested one more test round before dropping this axis.

  6. NEW: Training-time input-noise augmentation.
     New axis: `augment_noise_std` in {0.0, 0.005, 0.015, 0.03} fractional.
     Applied as x -> x * (1 + sigma * N(0,1)) per batch per epoch, ON TOP OF
     the 0.3% Gaussian noise already baked into the HDF5 data.

     WHY: Sweep #1 median best-epoch was 3.  With ~150K spectra generated
     from only ~300 unique droplet profiles (via varied sun/view geometry),
     the spectrum-to-profile inverse is massively under-determined *in
     profile space*.  Every epoch the network sees the same handful of
     profile shapes and simply memorizes the inverse on those 300 points,
     which is why overfitting triggers so fast.

     What noise augmentation CAN do: it turns each fixed spectrum into a
     continuous distribution of neighbors, so the network cannot latch onto
     specific spectral fingerprints of individual simulations.  This is the
     same trick Di Noia 2019 (1% reflectance noise) and Segal-Rozenhaimer
     2018 (rho=0.01 to 0.99 noise) used, and should push best-epoch deeper.

     What noise augmentation CANNOT do: create new profile shapes.  If a
     droplet profile is not in your 300, no amount of spectral noise teaches
     the network that profile.  Augmentation helps overfitting to spectra;
     it does not help coverage in profile space.  The fundamental fix for
     the 300-profile bottleneck is more profiles, not more noise.

     Honest expectation: I expect this to push mean RMSE down by ~5-15% and
     push median best-epoch from 3 into the 20-80 range.  If it does not,
     that is strong evidence that profile-space coverage, not memorization,
     is the dominant error source.  Either way it is a useful measurement.

  7. NEW: Linear warmup.
     Fixed warmup_steps = 500 iterations.  LR ramps linearly from 0 to the
     target LR over the first 500 mini-batches, then ReduceLROnPlateau takes
     over.  Rationale: at LRs in the 3e-6 to 1e-3 range, the first epoch can
     be unstable (AdamW moment estimates are zero on step 1, so initial
     updates can overshoot).  Warmup is standard in transformer/LLM training
     and costs almost nothing.

  8. NEW: lambda_monotonicity, lambda_adiabatic, lambda_smoothness swept
     as continuous params.  Sweep #1 held lambda_adiabatic and
     lambda_smoothness fixed at 0.1 and lambda_monotonicity at 0.0 — we
     have no direct evidence any of them help.  Range for each: [0, 0.25],
     LHS-sampled alongside LR/dropout/wd/sigma.  lambda_physics is still
     held fixed at 0.1 (forward-physics consistency term).

Sampling strategy:
    - Latin Hypercube Sampling over 7 continuous dims: LR, dropout, wd,
      sigma_floor, lambda_monotonicity, lambda_adiabatic, lambda_smoothness.
    - Shuffled cycling over the 5x5x3x4 = 300 categorical combos of
      (architecture, weights, batch_size, noise_level).  100 runs ->
      each categorical combo appears 0 or 1 times, but because we shuffle
      and the LHS continuous dims are independent, marginal coverage on
      any single categorical axis is even.

Run locally before uploading:
    python generate_sweep_2.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import json
import itertools
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────────────────────────────────────
FIXED = {
    'data': {
        'h5_path': '/scratch/alpine/anbu8374/neural_network_training_data/combined_vocals_oracles_training_data_13_April_2026.h5',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    'output_dir': 'sweep_results_3',
}

# ─────────────────────────────────────────────────────────────────────────────
# Categorical search space
# ─────────────────────────────────────────────────────────────────────────────

# Architectures — largest (4x512, 5x512, 6x256) dropped.
HIDDEN_DIMS_OPTIONS = [
    [128, 128, 128],                      # 3-layer narrow      (~110K params)
    [256, 256, 256],                      # 3-layer medium      (~350K params)
    [256, 256, 256, 256],                 # 4-layer narrow      (~430K params)
    [512, 512, 256, 256],                 # 4-layer tapered     (~720K params)
    [512, 256, 256, 128, 128],            # 5-layer pyramid     (~440K params)
]

# Level-weight schemes — unchanged from sweep #1 per user request.
LEVEL_WEIGHTS_OPTIONS = {
    'uniform':     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'ends':        [4.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 3.0],
    'top':         [6.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'strong_ends': [6.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0],
    'deep':        [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
}

BATCH_SIZE_OPTIONS = [128, 256, 512]

# NEW: noise-augmentation levels.  Fractional std on input reflectance,
# added per-batch on top of the 0.3% noise already baked into the data.
AUGMENT_NOISE_OPTIONS = [0.000, 0.005, 0.015, 0.030]

# ─────────────────────────────────────────────────────────────────────────────
# Continuous search space (LHS)
# ─────────────────────────────────────────────────────────────────────────────
LR_LOG10_RANGE             = [-5.52, -3.0]   # 3e-6 to 1e-3
DROPOUT_RANGE              = [0.15, 0.40]
WEIGHT_DECAY_LOG10_RANGE   = [-6.0, -2.0]    # 1e-6 to 1e-2
SIGMA_FLOOR_RANGE          = [0.005, 0.05]
LAMBDA_MONOTONICITY_RANGE  = [0.0, 0.25]     # NEW (was fixed at 0.0)
LAMBDA_ADIABATIC_RANGE     = [0.0, 0.25]     # NEW (was fixed at 0.1)
LAMBDA_SMOOTHNESS_RANGE    = [0.0, 0.25]     # NEW (was fixed at 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults (constant across runs)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_DEFAULTS = {
    'n_epochs': 1000,
    'scheduler_patience': 30,
    'early_stop_patience': 150,
    'warmup_steps': 500,            # NEW
    'lambda_physics': 0.1,
    # lambda_monotonicity, lambda_adiabatic, lambda_smoothness are per-run (LHS).
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

    # Categorical combos.  5 arch * 5 weights * 3 bs * 4 noise = 300 combos.
    categorical_combos = list(itertools.product(
        range(len(HIDDEN_DIMS_OPTIONS)),
        list(LEVEL_WEIGHTS_OPTIONS.keys()),
        BATCH_SIZE_OPTIONS,
        AUGMENT_NOISE_OPTIONS,
    ))
    rng.shuffle(categorical_combos)
    n_categorical = len(categorical_combos)

    # 7 continuous dims: LR, dropout, wd, sigma_floor,
    #                    lambda_monotonicity, lambda_adiabatic, lambda_smoothness
    lhs = latin_hypercube_sample(N_RUNS, 7, rng)

    for i in range(N_RUNS):
        arch_idx, weight_name, batch_size, noise_std = categorical_combos[i % n_categorical]

        hidden_dims   = HIDDEN_DIMS_OPTIONS[arch_idx]
        level_weights = LEVEL_WEIGHTS_OPTIONS[weight_name]

        lr = 10 ** (LR_LOG10_RANGE[0]
                    + lhs[i, 0] * (LR_LOG10_RANGE[1] - LR_LOG10_RANGE[0]))
        dropout = (DROPOUT_RANGE[0]
                   + lhs[i, 1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]))
        weight_decay = 10 ** (WEIGHT_DECAY_LOG10_RANGE[0]
                              + lhs[i, 2] * (WEIGHT_DECAY_LOG10_RANGE[1] - WEIGHT_DECAY_LOG10_RANGE[0]))
        sigma_floor = (SIGMA_FLOOR_RANGE[0]
                       + lhs[i, 3] * (SIGMA_FLOOR_RANGE[1] - SIGMA_FLOOR_RANGE[0]))
        lambda_monotonicity = (LAMBDA_MONOTONICITY_RANGE[0]
                               + lhs[i, 4] * (LAMBDA_MONOTONICITY_RANGE[1] - LAMBDA_MONOTONICITY_RANGE[0]))
        lambda_adiabatic = (LAMBDA_ADIABATIC_RANGE[0]
                            + lhs[i, 5] * (LAMBDA_ADIABATIC_RANGE[1] - LAMBDA_ADIABATIC_RANGE[0]))
        lambda_smoothness = (LAMBDA_SMOOTHNESS_RANGE[0]
                             + lhs[i, 6] * (LAMBDA_SMOOTHNESS_RANGE[1] - LAMBDA_SMOOTHNESS_RANGE[0]))

        # Round for readability
        lr                  = float(f"{lr:.6f}")
        dropout             = float(f"{dropout:.3f}")
        weight_decay        = float(f"{weight_decay:.2e}")
        sigma_floor         = float(f"{sigma_floor:.4f}")
        lambda_monotonicity = float(f"{lambda_monotonicity:.3f}")
        lambda_adiabatic    = float(f"{lambda_adiabatic:.3f}")
        lambda_smoothness   = float(f"{lambda_smoothness:.3f}")

        # Architecture descriptor string
        n_layers = len(hidden_dims)
        width    = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_str = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_str = "tapered"
        else:
            arch_str = "custom"

        run_name = (f"{arch_str}_dr{dropout:.2f}_lr{lr:.1e}_"
                    f"wd{weight_decay:.0e}_b{batch_size}_n{noise_std:.3f}_"
                    f"{weight_name}")

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
                'augment_noise_std': noise_std,
                'lambda_monotonicity': lambda_monotonicity,
                'lambda_adiabatic': lambda_adiabatic,
                'lambda_smoothness': lambda_smoothness,
                **TRAINING_DEFAULTS,
            },
        }
        configs.append(cfg)

    return configs


def main():
    out_dir = Path('sweep_configs_2')
    out_dir.mkdir(exist_ok=True)

    # Clean any leftover configs
    for old in out_dir.glob('run_*.json'):
        old.unlink()
    if (out_dir / 'sweep_summary.json').exists():
        (out_dir / 'sweep_summary.json').unlink()

    configs = generate_configs()

    # Individual config files
    for cfg in configs:
        path = out_dir / f"run_{cfg['run_id']:03d}.json"
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    # Summary
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
            'augment_noise_std': hp['augment_noise_std'],
            'level_weights_name': hp['level_weights_name'],
            'lambda_monotonicity': hp['lambda_monotonicity'],
            'lambda_adiabatic': hp['lambda_adiabatic'],
            'lambda_smoothness': hp['lambda_smoothness'],
        })

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"Generated {len(configs)} sweep configurations in {out_dir}/\n")
    print(f"{'ID':>3}  {'Architecture':<22} {'Drop':>5} {'LR':>10} "
          f"{'WD':>10} {'sig_fl':>7} {'BS':>4} {'Noise':>6} "
          f"{'l_mono':>6} {'l_adi':>6} {'l_sm':>6} {'Weights':<12}")
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
        print(f"{s['run_id']:3d}  {arch:<22} {s['dropout']:5.2f} "
              f"{s['learning_rate']:10.6f} {s['weight_decay']:10.2e} "
              f"{s['sigma_floor']:7.4f} {s['batch_size']:4d} "
              f"{s['augment_noise_std']:6.3f} "
              f"{s['lambda_monotonicity']:6.3f} "
              f"{s['lambda_adiabatic']:6.3f} {s['lambda_smoothness']:6.3f} "
              f"{s['level_weights_name']:<12}")

    # Coverage check
    print(f"\n{'='*70}")
    print("Coverage check (Latin Hypercube spans these ranges):")
    print(f"  Learning rate:      {min(s['learning_rate'] for s in summary):.2e} "
          f"to {max(s['learning_rate'] for s in summary):.2e}")
    print(f"  Dropout:            {min(s['dropout'] for s in summary):.3f} "
          f"to {max(s['dropout'] for s in summary):.3f}")
    print(f"  Weight decay:       {min(s['weight_decay'] for s in summary):.2e} "
          f"to {max(s['weight_decay'] for s in summary):.2e}")
    print(f"  Sigma floor:          {min(s['sigma_floor'] for s in summary):.4f} "
          f"to {max(s['sigma_floor'] for s in summary):.4f}")
    print(f"  lambda_monotonicity:  {min(s['lambda_monotonicity'] for s in summary):.3f} "
          f"to {max(s['lambda_monotonicity'] for s in summary):.3f}")
    print(f"  lambda_adiabatic:     {min(s['lambda_adiabatic'] for s in summary):.3f} "
          f"to {max(s['lambda_adiabatic'] for s in summary):.3f}")
    print(f"  lambda_smoothness:    {min(s['lambda_smoothness'] for s in summary):.3f} "
          f"to {max(s['lambda_smoothness'] for s in summary):.3f}")
    print(f"\nCategorical counts:")
    for name in LEVEL_WEIGHTS_OPTIONS:
        c = sum(1 for s in summary if s['level_weights_name'] == name)
        print(f"  weights={name}: {c} runs")
    for bs in BATCH_SIZE_OPTIONS:
        c = sum(1 for s in summary if s['batch_size'] == bs)
        print(f"  batch_size={bs}: {c} runs")
    for ns in AUGMENT_NOISE_OPTIONS:
        c = sum(1 for s in summary if s['augment_noise_std'] == ns)
        print(f"  augment_noise_std={ns}: {c} runs")

    print(f"\nNext steps:")
    print(f"  1. Update sweep_alpine.sh: point --array and config path to sweep_configs_2/")
    print(f"  2. Upload sweep_configs_2/ to Alpine")
    print(f"  3. sbatch sweep_alpine.sh")


if __name__ == '__main__':
    main()
