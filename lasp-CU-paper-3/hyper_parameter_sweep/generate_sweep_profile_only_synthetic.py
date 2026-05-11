"""
generate_sweep_profile_only_synthetic.py

Sweep generator for the profile-only model on the synthetic-cloud training
dataset (`synthetic_training_data_7-levels_8_May_2026.h5`, 8,888 samples,
7-level r_e profiles).

Same axes / sampling strategy as `generate_sweep_profile_only.py` but adapted
to the new dataset:

  • N_LEVELS = 7 (was 50)
  • h5_path  = synthetic-cloud HDF5 (was VOCALS+ORACLES 50-level)
  • n_val_profiles = 889, n_test_profiles = 889 (10% / 10% — was 14 / 14)
  • Single profile-aware split per config (no K-fold) — with 8,888 samples
    the val signal is stable enough that K-fold's per-config std isn't
    needed, and dropping it cuts wall time 5×.
  • Each config is generated THREE times — once per model variant, into
    three separate output directories — so the same hyperparameter draw
    can be compared across M0, M1, M2:
        sweep_configs_profile_only_synthetic_M0/run_NNN.json   (zero all extras)
        sweep_configs_profile_only_synthetic_M1/run_NNN.json   (tau_c only)
        sweep_configs_profile_only_synthetic_M2/run_NNN.json   (tau_c + wv_above)

Usage:
    python generate_sweep_profile_only_synthetic.py            # 100 configs
    python generate_sweep_profile_only_synthetic.py --n-runs 150
"""

from __future__ import annotations
import argparse
import json
import itertools
from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings  (mirror generate_sweep_profile_only.py except where noted)
# ─────────────────────────────────────────────────────────────────────────────
N_LEVELS = 7

H5_FILENAME = 'synthetic_training_data_7-levels_8_May_2026.h5'
H5_DIR_DEFAULT = '/scratch/alpine/anbu8374/neural_network_training_data'

FIXED_PER_RUN = {
    'data': {
        'h5_path':    f'{H5_DIR_DEFAULT}/{H5_FILENAME}',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    # Single split, profile-aware. Synthetic dataset has 8,888 samples →
    # 889 val + 889 test ≈ 80 / 10 / 10.
    'n_val_profiles':  889,
    'n_test_profiles': 889,
}

# Three model variants share the same hyperparameter draws but differ in
# which extras are zeroed.  Each entry is (variant_name, zero_flags).
# zero_flags maps each of the three extras to True (zero) / False (keep).
MODEL_VARIANTS = [
    ('M0', {'zero_tau_c': True,  'zero_wv_above': True,  'zero_wv_in': True}),
    ('M1', {'zero_tau_c': False, 'zero_wv_above': True,  'zero_wv_in': True}),
    ('M2', {'zero_tau_c': False, 'zero_wv_above': False, 'zero_wv_in': True}),
]

# ─────────────────────────────────────────────────────────────────────────────
# Sweep axes — same as generate_sweep_profile_only.py
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIMS_OPTIONS = [
    [128, 128, 128],
    [256, 256, 256],
    [256, 256, 256, 256],
    [512, 512, 256, 256],
    [512, 256, 256, 128, 128],
]
ACTIVATION_OPTIONS = ['gelu', 'relu', 'silu']
BATCH_SIZE_OPTIONS = [128, 256, 512]

LEVEL_WEIGHT_SCHEMES = ['uniform', 'top', 'bottom', 'u_shape']

# Continuous (LHS, 7 dims)
LR_RANGE        = (3e-6, 1e-3)   # log-uniform
DROPOUT_RANGE   = (0.15, 0.40)
L_PHYS_RANGE    = (0.00, 0.25)
L_MONO_RANGE    = (0.00, 0.25)
L_ADI_RANGE     = (0.00, 0.25)
L_SM_RANGE      = (0.00, 0.25)
NOISE_STD_RANGE = (0.000, 0.030)

# Fixed across all runs
WEIGHT_DECAY        = 1e-4
SIGMA_FLOOR         = 0.01
N_EPOCHS            = 1500   # smoke test stalled at no-improve=46 when cap=1000;
                              # 1500 gives slow configs ~500 extra epochs of headroom,
                              # but most runs still self-stop near 1100-1200 via the
                              # 150-epoch early-stop patience below.
EARLY_STOP_PATIENCE = 150
SCHEDULER_PATIENCE  = 30
WARMUP_STEPS        = 500


# ─────────────────────────────────────────────────────────────────────────────
# Level-weight schemes (7 levels here, not 50)
# ─────────────────────────────────────────────────────────────────────────────
def build_level_weights(scheme: str, n_levels: int = N_LEVELS) -> list:
    """7-level loss-weighting schemes. Convention: index 0 = top, n-1 = base."""
    levels = np.arange(n_levels, dtype=float)
    end = n_levels - 1
    decay = max(2.0, n_levels / 5.0)        # ~1.4 levels for n=7

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
        raise ValueError(f"Unknown level_weights scheme: {scheme!r}")
    return [float(x) for x in w]


# ─────────────────────────────────────────────────────────────────────────────
# Latin-Hypercube sampler  (verbatim port from generate_sweep_profile_only)
# ─────────────────────────────────────────────────────────────────────────────
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Standard LHS in [0, 1]^d, n samples, d dimensions."""
    out = np.empty((n, d))
    for dim in range(d):
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        out[:, dim] = bins
    return out


def cycled_categorical(options: list, n_runs: int,
                       rng: np.random.Generator) -> list:
    """Repeat the option list to fill n_runs, then shuffle to decorrelate
    from the order of the LHS draws."""
    out = []
    for i in range(n_runs):
        out.append(options[i % len(options)])
    rng.shuffle(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n-runs',  type=int, default=100,
                    help='Number of hyperparameter configs to generate (per variant). '
                         'Default 100 → 300 total trainings across the 3 variants.')
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-root', type=Path,
                    default=Path(__file__).resolve().parent,
                    help='Directory under which the three '
                         'sweep_configs_profile_only_synthetic_{M0,M1,M2} '
                         'subdirectories will be created.')
    args = ap.parse_args()

    n = args.n_runs
    rng = np.random.default_rng(args.seed)

    # ── Continuous draws via LHS ────────────────────────────────────────
    L = latin_hypercube(n, 7, rng)
    lr      = LR_RANGE[0]      * (LR_RANGE[1] / LR_RANGE[0]) ** L[:, 0]   # log-uniform
    dropout = DROPOUT_RANGE[0] + (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]) * L[:, 1]
    l_phys  = L_PHYS_RANGE[0]  + (L_PHYS_RANGE[1]  - L_PHYS_RANGE[0])  * L[:, 2]
    l_mono  = L_MONO_RANGE[0]  + (L_MONO_RANGE[1]  - L_MONO_RANGE[0])  * L[:, 3]
    l_adi   = L_ADI_RANGE[0]   + (L_ADI_RANGE[1]   - L_ADI_RANGE[0])   * L[:, 4]
    l_sm    = L_SM_RANGE[0]    + (L_SM_RANGE[1]    - L_SM_RANGE[0])    * L[:, 5]
    noise   = NOISE_STD_RANGE[0] + (NOISE_STD_RANGE[1] - NOISE_STD_RANGE[0]) * L[:, 6]

    # ── Categorical draws (cycled, shuffled) ──────────────────────────
    arch_idx_seq    = cycled_categorical(list(range(len(HIDDEN_DIMS_OPTIONS))), n, rng)
    activation_seq  = cycled_categorical(ACTIVATION_OPTIONS, n, rng)
    batch_size_seq  = cycled_categorical(BATCH_SIZE_OPTIONS, n, rng)
    lw_scheme_seq   = cycled_categorical(LEVEL_WEIGHT_SCHEMES, n, rng)

    summary = []
    for i in range(n):
        hidden_dims = HIDDEN_DIMS_OPTIONS[arch_idx_seq[i]]
        lw_scheme   = lw_scheme_seq[i]
        lw          = build_level_weights(lw_scheme, N_LEVELS)

        # Quick string tag for the config (matches the previous sweep's pattern)
        n_layers = len(hidden_dims)
        width    = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_tag = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_tag = f"{n_layers}L_taper{width}_{hidden_dims[-1]}"
        else:
            arch_tag = f"{n_layers}L_custom"

        run_tag = (f"{arch_tag}_{activation_seq[i]}_b{batch_size_seq[i]}_"
                   f"{lw_scheme}_n{noise[i]:.3f}_dr{float(dropout[i]):.2f}_"
                   f"lr{float(lr[i]):.1e}_phys{float(l_phys[i]):.2f}_"
                   f"mono{float(l_mono[i]):.2f}_adi{float(l_adi[i]):.2f}_"
                   f"sm{float(l_sm[i]):.2f}")

        # Hyperparameter block — identical across the 3 variants
        hp = {
            'hidden_dims':         hidden_dims,
            'activation':          activation_seq[i],
            'dropout':             round(float(dropout[i]), 4),
            'learning_rate':       float(lr[i]),
            'lambda_physics':      round(float(l_phys[i]), 4),
            'lambda_monotonicity': round(float(l_mono[i]), 4),
            'lambda_adiabatic':    round(float(l_adi[i]), 4),
            'lambda_smoothness':   round(float(l_sm[i]), 4),
            'level_weights':       lw,
            'level_weights_name':  lw_scheme,
            'batch_size':          batch_size_seq[i],
            'augment_noise_std':   round(float(noise[i]), 4),
            'n_epochs':            N_EPOCHS,
            'scheduler_patience':  SCHEDULER_PATIENCE,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'warmup_steps':        WARMUP_STEPS,
            'weight_decay':        WEIGHT_DECAY,
            'sigma_floor':         SIGMA_FLOOR,
        }

        summary.append({
            'run_id':       i,
            'tag':          run_tag,
            'hidden_dims':  hidden_dims,
            'activation':   activation_seq[i],
            'dropout':      hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'level_weights_name': lw_scheme,
            'batch_size':   batch_size_seq[i],
            'augment_noise_std': hp['augment_noise_std'],
        })

        # ── Write one JSON per (run, model variant) ───────────────────
        for variant_name, zero_flags in MODEL_VARIANTS:
            out_dir = args.out_root / f'sweep_configs_profile_only_synthetic_{variant_name}'
            out_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                'run_id':     i,
                'tag':        run_tag,
                'variant':    variant_name,
                'extras':     zero_flags,
                'hyperparams': hp,
                'data':           dict(FIXED_PER_RUN['data']),
                'n_val_profiles': FIXED_PER_RUN['n_val_profiles'],
                'n_test_profiles': FIXED_PER_RUN['n_test_profiles'],
                'output_dir': f'sweep_results_profile_only_synthetic_{variant_name}',
            }

            with open(out_dir / f'run_{i:03d}.json', 'w') as f:
                json.dump(cfg, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\nGenerated {n} configs × 3 variants = {n * 3} JSON files")
    for variant_name, _ in MODEL_VARIANTS:
        print(f"  → {args.out_root / f'sweep_configs_profile_only_synthetic_{variant_name}'}")

    print(f"\n{'idx':>4}  {'arch':>20}  {'act':>5} {'lw':>8} {'bs':>4}  "
          f"{'noise':>6}  {'dr':>5} {'lr':>10}")
    print('-' * 80)
    for s in summary:
        dims = s['hidden_dims']
        if all(h == dims[0] for h in dims):
            arch = f"{len(dims)}x{dims[0]}"
        else:
            arch = '×'.join(str(h) for h in dims)
        print(f"{s['run_id']:>4}  {arch:>20}  {s['activation']:>5} "
              f"{s['level_weights_name']:>8} {s['batch_size']:>4}  "
              f"{s['augment_noise_std']:>6.3f} {s['dropout']:>5.2f} "
              f"{s['learning_rate']:>10.1e}")

    print(f"\nRanges across {n} configs:")
    print(f"  learning_rate: {min(s['learning_rate'] for s in summary):.2e} "
          f"→ {max(s['learning_rate'] for s in summary):.2e}")
    print(f"  dropout:       {min(s['dropout'] for s in summary):.3f} "
          f"→ {max(s['dropout'] for s in summary):.3f}")
    print(f"  noise std:     {min(s['augment_noise_std'] for s in summary):.4f} "
          f"→ {max(s['augment_noise_std'] for s in summary):.4f}")

    arch_counts = {}
    for s in summary:
        key = tuple(s['hidden_dims'])
        arch_counts[key] = arch_counts.get(key, 0) + 1
    print(f"\nArch coverage:")
    for dims in HIDDEN_DIMS_OPTIONS:
        c = arch_counts.get(tuple(dims), 0)
        print(f"  hidden_dims={dims}: {c} runs")


if __name__ == '__main__':
    main()
