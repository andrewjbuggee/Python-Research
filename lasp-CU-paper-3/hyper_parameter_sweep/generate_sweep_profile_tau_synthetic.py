"""
generate_sweep_profile_tau_synthetic.py

Sweep generator for the PROFILE + τ_c model on the synthetic-cloud training
dataset (`synthetic_training_data_7-levels_8_May_2026.h5`, 41,991 samples,
7-level r_e profiles).

Model: `DropletProfileNetwork` (models.py) — the original two-head retrieval
network.  636 HySICS reflectances + 4 geometry angles in; 7-level r_e profile
(+ per-level σ) AND cloud optical thickness τ_c (+ σ) out.  This is the same
architecture family as the profile-only sweep (M0), with the τ head and the
τ-NLL loss term restored.

Relationship to the profile-only sweep
--------------------------------------
The shared hyperparameter draws here are BIT-IDENTICAL to
`generate_sweep_profile_only_synthetic.py` with the same --seed: the same LHS
call, the same categorical cycling, in the same order, consuming the same RNG
stream.  So `run_017` of this sweep and `run_017` of
`sweep_results_profile_only_synthetic_M0_rev2` differ only by the τ head, the
τ-loss term, and the τ-specific axes below.  That makes the two sweeps a
paired comparison: how much does adding the τ task cost (or help) the profile
retrieval, config for config?

Two τ-specific axes are drawn from a SEPARATE RNG stream (seed + 1) so adding
them does not perturb the shared draws:

  • tau_weight    — log-uniform in [0.02, 2.0].  Weight on the τ NLL relative
                    to the profile NLL.  The profile term is a mean over
                    levels, so tau_weight=1.0 makes τ as important as the
                    whole profile (7× any single level); 0.02 makes it ~1/7 of
                    a single level, i.e. τ nearly free-riding on features the
                    profile task already needs.  models_profile_only.py was
                    written because τ in the loss appeared to hurt per-level
                    profile RMSE — this axis measures where that trade-off
                    actually bites.

  • tau_transform — 'linear' or 'log10', 50/50.  How τ is normalized before
                    the Gaussian NLL.  τ in this dataset is strongly
                    right-skewed (min 3.0, median 8.0, 95th pct 21.3,
                    max 95.0), so the stock linear map onto [TAU_MIN, TAU_MAX]
                    = [3, 97] puts ~95% of samples in the bottom 20% of the
                    τ head's output range.  'log10' maps
                    log10(τ) linearly onto [0, 1] instead, which spreads the
                    populated range over the full sigmoid.  Swept rather than
                    assumed, so the sweep answers which conditioning wins.

Split note
----------
Every sample in the synthetic HDF5 is its own unique profile (n_geometries=1),
so profile-aware and sample-level splits coincide.  Defaults are a TRUE
80/10/10 split: n_val = n_test = 4,199 of 41,991 → 33,593 / 4,199 / 4,199 =
80.0 / 10.0 / 10.0 %.  (The M0_rev2 profile-only sweep used 889/889 =
95.8 / 2.1 / 2.1 % on this file.)  Because `create_profile_aware_splits`
shuffles profile IDs once with the same seed and takes test = shuffled[:n_test],
the M0_rev2 889-profile test set is a strict subset of this sweep's 4,199-
profile test set — but train sets differ (33,593 vs 40,213 samples), so
run-for-run comparisons against M0_rev2 carry a split confound in addition to
the τ-head change.  Override with --n-val-profiles / --n-test-profiles
(889/889 restores the exact M0_rev2 pairing).

Usage:
    python generate_sweep_profile_tau_synthetic.py                 # 100 configs
    python generate_sweep_profile_tau_synthetic.py --n-runs 150
    python generate_sweep_profile_tau_synthetic.py --tau-weight 1.0 \
        --tau-transform log10                       # pin both τ axes

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────────────────────────────────────
N_LEVELS = 7
VARIANT = 'PT'          # profile + tau; stamped into every summary.json so
                        # compare_sweep_profile_only_synthetic.py can label it

H5_FILENAME = 'synthetic_training_data_7-levels_8_May_2026.h5'
H5_DIR_DEFAULT = '/scratch/alpine/anbu8374/neural_network_training_data'

FIXED_PER_RUN = {
    'data': {
        'h5_path':    f'{H5_DIR_DEFAULT}/{H5_FILENAME}',
        'instrument': 'hysics',
        'num_workers': 4,
    },
    # True 80/10/10 on the 41,991-sample file (each sample = unique profile).
    # See the Split note in the module docstring for the M0_rev2 relationship.
    'n_val_profiles':  4199,
    'n_test_profiles': 4199,
}

# ─────────────────────────────────────────────────────────────────────────────
# Sweep axes — shared block identical to generate_sweep_profile_only_synthetic
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

# ── τ-specific axes (separate RNG stream — see module docstring) ─────────────
TAU_WEIGHT_RANGE   = (0.02, 2.0)          # log-uniform
TAU_TRANSFORM_OPTIONS = ['linear', 'log10']

# Fixed across all runs
WEIGHT_DECAY        = 1e-4
SIGMA_FLOOR         = 0.01
N_EPOCHS            = 1500
EARLY_STOP_PATIENCE = 150
SCHEDULER_PATIENCE  = 30
WARMUP_STEPS        = 500

# Which validation quantity drives checkpointing / early stopping.
#   'total'   — profile NLL + tau_weight·τ NLL + physics  (previous τ-model
#               behaviour; the selected epoch depends on tau_weight)
#   'profile' — profile NLL only, so checkpoint selection is comparable across
#               configs with different tau_weight
EARLY_STOP_METRIC = 'total'


# ─────────────────────────────────────────────────────────────────────────────
# Level-weight schemes (7 levels)
# ─────────────────────────────────────────────────────────────────────────────
def build_level_weights(scheme: str, n_levels: int = N_LEVELS) -> List[float]:
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
# Samplers  (verbatim from generate_sweep_profile_only_synthetic.py — keeping
# these identical is what makes the shared draws reproduce bit-for-bit)
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
    ap.add_argument('--n-runs', type=int, default=100,
                    help='Number of hyperparameter configs to generate.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Seed for the shared draws. Use 42 to reproduce the '
                         'profile-only M0 hyperparameters exactly.')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='Config output directory. Default: '
                         '<this file>/sweep_configs_profile_tau_synthetic')
    ap.add_argument('--h5-filename', type=str, default=H5_FILENAME,
                    help='HDF5 filename baked into each config.')
    ap.add_argument('--h5-dir', type=str, default=H5_DIR_DEFAULT,
                    help='HDF5 directory baked into each config (the trainer '
                         'can still override it with --training-data-dir).')
    ap.add_argument('--n-val-profiles', type=int,
                    default=FIXED_PER_RUN['n_val_profiles'])
    ap.add_argument('--n-test-profiles', type=int,
                    default=FIXED_PER_RUN['n_test_profiles'])
    ap.add_argument('--tau-weight', type=float, default=None,
                    help='Pin tau_weight to this value for every run instead '
                         'of sweeping it log-uniformly over '
                         f'{TAU_WEIGHT_RANGE}.')
    ap.add_argument('--tau-transform', type=str, default=None,
                    choices=TAU_TRANSFORM_OPTIONS,
                    help='Pin the τ normalization for every run instead of '
                         'sweeping linear/log10 50-50.')
    ap.add_argument('--early-stop-metric', type=str, default=EARLY_STOP_METRIC,
                    choices=['total', 'profile'],
                    help="Validation quantity used for checkpointing.")
    args = ap.parse_args()

    n = args.n_runs
    out_dir = (args.out_dir
               or Path(__file__).resolve().parent / 'sweep_configs_profile_tau_synthetic')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared draws — identical RNG stream to the profile-only generator ──
    rng = np.random.default_rng(args.seed)

    L = latin_hypercube(n, 7, rng)
    lr      = LR_RANGE[0]      * (LR_RANGE[1] / LR_RANGE[0]) ** L[:, 0]   # log-uniform
    dropout = DROPOUT_RANGE[0] + (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]) * L[:, 1]
    l_phys  = L_PHYS_RANGE[0]  + (L_PHYS_RANGE[1]  - L_PHYS_RANGE[0])  * L[:, 2]
    l_mono  = L_MONO_RANGE[0]  + (L_MONO_RANGE[1]  - L_MONO_RANGE[0])  * L[:, 3]
    l_adi   = L_ADI_RANGE[0]   + (L_ADI_RANGE[1]   - L_ADI_RANGE[0])   * L[:, 4]
    l_sm    = L_SM_RANGE[0]    + (L_SM_RANGE[1]    - L_SM_RANGE[0])    * L[:, 5]
    noise   = NOISE_STD_RANGE[0] + (NOISE_STD_RANGE[1] - NOISE_STD_RANGE[0]) * L[:, 6]

    arch_idx_seq    = cycled_categorical(list(range(len(HIDDEN_DIMS_OPTIONS))), n, rng)
    activation_seq  = cycled_categorical(ACTIVATION_OPTIONS, n, rng)
    batch_size_seq  = cycled_categorical(BATCH_SIZE_OPTIONS, n, rng)
    lw_scheme_seq   = cycled_categorical(LEVEL_WEIGHT_SCHEMES, n, rng)

    # ── τ-specific draws — SEPARATE stream, so the block above is unchanged ─
    tau_rng = np.random.default_rng(args.seed + 1)
    if args.tau_weight is not None:
        tau_weight_seq = [float(args.tau_weight)] * n
    else:
        u = tau_rng.random(n)
        tau_weight_seq = list(
            TAU_WEIGHT_RANGE[0]
            * (TAU_WEIGHT_RANGE[1] / TAU_WEIGHT_RANGE[0]) ** u)   # log-uniform
    if args.tau_transform is not None:
        tau_transform_seq = [args.tau_transform] * n
    else:
        tau_transform_seq = cycled_categorical(TAU_TRANSFORM_OPTIONS, n, tau_rng)

    summary = []
    for i in range(n):
        hidden_dims = HIDDEN_DIMS_OPTIONS[arch_idx_seq[i]]
        lw_scheme   = lw_scheme_seq[i]
        lw          = build_level_weights(lw_scheme, N_LEVELS)

        n_layers = len(hidden_dims)
        width    = hidden_dims[0]
        if all(h == width for h in hidden_dims):
            arch_tag = f"{n_layers}x{width}"
        elif hidden_dims[0] > hidden_dims[-1]:
            arch_tag = f"{n_layers}L_taper{width}_{hidden_dims[-1]}"
        else:
            arch_tag = f"{n_layers}L_custom"

        # Same tag stem as the profile-only sweep (so run_NNN pairs are easy to
        # eyeball side by side) plus the two τ axes.
        run_tag = (f"{arch_tag}_{activation_seq[i]}_b{batch_size_seq[i]}_"
                   f"{lw_scheme}_n{noise[i]:.3f}_dr{float(dropout[i]):.2f}_"
                   f"lr{float(lr[i]):.1e}_phys{float(l_phys[i]):.2f}_"
                   f"mono{float(l_mono[i]):.2f}_adi{float(l_adi[i]):.2f}_"
                   f"sm{float(l_sm[i]):.2f}_"
                   f"tw{float(tau_weight_seq[i]):.3f}_{tau_transform_seq[i]}")

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
            # τ-specific
            'tau_weight':          round(float(tau_weight_seq[i]), 5),
            'tau_transform':       tau_transform_seq[i],
            'early_stop_metric':   args.early_stop_metric,
        }

        cfg = {
            'run_id':          i,
            'tag':             run_tag,
            'variant':         VARIANT,
            'hyperparams':     hp,
            'data': {
                'h5_path':     f'{args.h5_dir.rstrip("/")}/{args.h5_filename}',
                'instrument':  FIXED_PER_RUN['data']['instrument'],
                'num_workers': FIXED_PER_RUN['data']['num_workers'],
            },
            'n_val_profiles':  args.n_val_profiles,
            'n_test_profiles': args.n_test_profiles,
            'output_dir':      'sweep_results_profile_tau_synthetic',
        }

        with open(out_dir / f'run_{i:03d}.json', 'w') as f:
            json.dump(cfg, f, indent=2)

        summary.append({
            'run_id':             i,
            'tag':                run_tag,
            'hidden_dims':        hidden_dims,
            'activation':         activation_seq[i],
            'dropout':            hp['dropout'],
            'learning_rate':      hp['learning_rate'],
            'level_weights_name': lw_scheme,
            'batch_size':         batch_size_seq[i],
            'augment_noise_std':  hp['augment_noise_std'],
            'tau_weight':         hp['tau_weight'],
            'tau_transform':      hp['tau_transform'],
        })

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\nGenerated {n} profile+τ configs → {out_dir}")
    print(f"\n{'idx':>4}  {'arch':>20}  {'act':>5} {'lw':>8} {'bs':>4}  "
          f"{'noise':>6}  {'dr':>5} {'lr':>10}  {'tau_w':>7} {'tau_tf':>7}")
    print('-' * 100)
    for s in summary:
        dims = s['hidden_dims']
        if all(h == dims[0] for h in dims):
            arch = f"{len(dims)}x{dims[0]}"
        else:
            arch = '×'.join(str(h) for h in dims)
        print(f"{s['run_id']:>4}  {arch:>20}  {s['activation']:>5} "
              f"{s['level_weights_name']:>8} {s['batch_size']:>4}  "
              f"{s['augment_noise_std']:>6.3f} {s['dropout']:>5.2f} "
              f"{s['learning_rate']:>10.1e}  {s['tau_weight']:>7.3f} "
              f"{s['tau_transform']:>7}")

    print(f"\nRanges across {n} configs:")
    print(f"  learning_rate: {min(s['learning_rate'] for s in summary):.2e} "
          f"→ {max(s['learning_rate'] for s in summary):.2e}")
    print(f"  dropout:       {min(s['dropout'] for s in summary):.3f} "
          f"→ {max(s['dropout'] for s in summary):.3f}")
    print(f"  noise std:     {min(s['augment_noise_std'] for s in summary):.4f} "
          f"→ {max(s['augment_noise_std'] for s in summary):.4f}")
    print(f"  tau_weight:    {min(s['tau_weight'] for s in summary):.4f} "
          f"→ {max(s['tau_weight'] for s in summary):.4f}")

    n_log = sum(1 for s in summary if s['tau_transform'] == 'log10')
    print(f"  tau_transform: {n_log} log10 / {n - n_log} linear")

    arch_counts = {}
    for s in summary:
        key = tuple(s['hidden_dims'])
        arch_counts[key] = arch_counts.get(key, 0) + 1
    print(f"\nArch coverage:")
    for dims in HIDDEN_DIMS_OPTIONS:
        print(f"  hidden_dims={dims}: {arch_counts.get(tuple(dims), 0)} runs")


if __name__ == '__main__':
    main()
