"""
11b_generate_sweep.py — STEP 11b of the paper-3 pipeline
========================================================
Hyperparameter sweep for the profile + τ network (DropletProfileNetwork:
636 HySICS reflectances + 4 angles → 7-level r_e + σ and τ_c + σ) on a new
training HDF5. Writes one config per trial; the trials are trained on Alpine
with 12_train_configs_alpine.sh (the same array script as step 12) and
analysed with 11c_analyze_sweep.py. Step 12 then retrains the best few with a
longer budget.

Why sweep again (rather than retrain the Aug-2026 winners)
  The data changed in ways that can move the optimum: per-level LWC ceiling,
  VOCALS-REx + ORACLES α, fixed water-cloud geometry, Mie τ_c labels, fewer
  PCs (k_re = 3, k_LWC = 5), 2.4× more samples, and τ normalized with 3 / 40
  instead of 3 / 97 (which changes what tau_weight and tau_transform do).

Design (sources in brackets; the parts marked "inference" are Claude's
reading of the Aug-2026 sweep, not results from a source)
  1. Quasi-random search: a scrambled, seeded Sobol sequence [Sob67] over the
     search space below. The Playbook recommends quasi-random search for this
     exploration stage: it spreads points more evenly than random search, is
     non-adaptive (all trials can run in parallel as one Slurm array) and is
     easy to analyse [Playbook, "Quasi-random search"]. Random rather than
     grid sampling matters because only a few hyperparameters usually matter
     [BB12]. (The claim that Sobol covers the space more evenly than a Latin
     hypercube of the same size is Claude's inference from the two methods'
     constructions — a Latin hypercube is only stratified one dimension at a
     time — not a result taken from these sources.)
  2. Ranking on VALIDATION metrics only. The trainer now writes validation
     RMSEs in physical units; test metrics are recorded but never used to
     choose: selecting among many models with the same data used to report
     their error biases that error optimistically [CT10]. The split seed is
     fixed, so all trials see the same train / validation / test clouds.
  3. Seed replicates: the first --n-replicate-configs configs are also trained
     with --n-extra-seeds other initialisation seeds (same split). Their
     spread is the run-to-run noise; [Bou21] show that initialisation and
     data-order randomness move benchmark results enough to change which
     method appears better. (How many replicates, and the 2σ rule 11c applies
     to them, are Claude's choices, not prescriptions from [Bou21].)
  4. A longer epoch budget. Inference: in the Aug-2026 sweep 80 of 100 runs
     (all of the top 10) had their best epoch within the last 10 % of the
     1,500-epoch cap, so the ranking partly measured how fast a config
     converges (Spearman ρ = −0.50 between learning rate and RMSE). Here the
     default is 3,000 epochs on 2.4× the data (~4.8× the optimizer steps of
     a 1,500-epoch run on the old file); 11c reports how many trials still
     peak near the cap. Checking where the best checkpoints fall relative to
     the step budget follows [Playbook, "A scientific approach…" / training-
     duration guidance]. Early stopping (patience 300) ends hopeless trials.
  5. Batch size fixed at 256, not tuned: "the batch size should not be
     treated as a tunable hyperparameter for validation set performance"
     [Playbook, "Choosing the batch size"]. Inference (Claude's): under a
     fixed epoch budget a smaller batch takes more optimizer steps, which is
     probably why 128 won in Aug-2026.
  6. Search-space bounds moved where the Aug-2026 optimum sat on a boundary
     (inference from that sweep): learning rate up to 3e-3 (top configs
     reached the old 1e-3 limit), dropout down to 0 (lower was better,
     ρ = +0.38 with RMSE), weight decay now swept (with AdamW the useful
     decay depends on the learning rate [LH19]). A best point at a search-
     space boundary signals that the bound should move [Playbook,
     "Quasi-random search"]; 11c flags this.

  7. Log-σ clamp lowered (20 Nov 2026). Both σ heads clamp log σ in
     normalized space; the old hard-coded −4 (r_e σ floor 0.888 μm) bound for
     most predictions in the Aug-2026 run_004 and in PT_sweep_2026-11-15
     run_047 (mean predicted σ 0.888 μm at levels 2–5 against validation
     RMSE 0.18–0.35 μm), so σ there was the clamp, not an estimate. Default
     --log-std-min −8: floor exp(−8) × 48.5 μm = 0.016 μm for r_e and
     exp(−8) × 37 = 0.012 for τ_c (3/40 bounds), about 11× below run_047's
     smallest per-level validation RMSE. sigma_floor (the loss-side floor) is
     set to exp(log_std_min) so it does not bind above the model's clamp.
     Why keep any floor: NLL training needs a lower bound on σ to settle
     (Seitzer et al. 2022, ICLR, App. B.1, https://arxiv.org/abs/2203.09168);
     Kendall & Gal (2017, NeurIPS, Eq. 8) regress log σ² unclamped. Choosing
     −8 (rather than −7) from run_047's errors is Claude's judgment, not
     a value from either source. The value is NOT swept: it is a numerical
     guard, not a modelling choice, and 11c reports how many σ still sit on it.

Not done here (by choice): adaptive search (Bayesian optimisation / TPE) and
early-stopping schedulers such as ASHA [Li20]. Both need a trainer that can
pause and resume and a long-running controller; the Slurm-array quasi-random
design keeps every trial independent. The Playbook recommends quasi-random
search for exploration and adaptive methods for later exploitation [Playbook,
"Quasi-random search"].

References (full)
  [Playbook]  Godbole, V., G. E. Dahl, J. Gilmer, C. J. Shallue and Z. Nado
              (2023): Deep Learning Tuning Playbook, Google Research.
              https://github.com/google-research/tuning_playbook ; sections used:
              "Quasi-random search"
              https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/quasi-random-search
              "Choosing the batch size" (new-project page)
              https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/new-project
              "A scientific approach to improving model performance"
              https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific
  [BB12]      Bergstra, J. and Y. Bengio (2012): Random search for
              hyper-parameter optimization. J. Mach. Learn. Res., 13, 281-305.
              https://jmlr.org/papers/v13/bergstra12a.html
  [CT10]      Cawley, G. C. and N. L. C. Talbot (2010): On over-fitting in
              model selection and subsequent selection bias in performance
              evaluation. J. Mach. Learn. Res., 11, 2079-2107.
              https://www.jmlr.org/papers/v11/cawley10a.html
  [Bou21]     Bouthillier, X., P. Delaunay, M. Bronzi, A. Trofimov, B. Nichyporuk,
              J. Szeto, N. Mohammadi Sepahvand, E. Raff, K. Madan, V. Voleti,
              S. Ebrahimi Kahou, V. Michalski, T. Arbel, C. Pal, G. Varoquaux and
              P. Vincent (2021): Accounting for variance in machine learning
              benchmarks. Proc. Machine Learning and Systems (MLSys), 3.
              https://arxiv.org/abs/2103.03098
  [Li20]      Li, L., K. Jamieson, A. Rostamizadeh, E. Gonina, J. Ben-tzur,
              M. Hardt, B. Recht and A. Talwalkar (2020): A system for massively
              parallel hyperparameter tuning. Proc. MLSys, 2 (ASHA).
              https://arxiv.org/abs/1810.05934
  [Sob67]     Sobol', I. M. (1967): On the distribution of points in a cube and
              the approximate evaluation of integrals. USSR Comput. Math. Math.
              Phys., 7(4), 86-112.  Implementation: scipy.stats.qmc.Sobol
              (scrambled), https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
  [LH19]      Loshchilov, I. and F. Hutter (2019): Decoupled weight decay
              regularization. ICLR. https://arxiv.org/abs/1711.05101 (AdamW)

Inputs
  Both HDF5 arguments name the SAME training file, the one written by step 10.
  They differ in what the script does with the path, because this script can be
  run on the laptop (where the configs are then pushed to Alpine) or on Alpine.

  --tag NAME         Sweep name. Output goes to training_runs/<NAME>/ inside
                     this repo, wherever the script is run. The sweep has no
                     other output location.
  --h5-alpine PATH   Where the HDF5 lives ON ALPINE. This path is NOT opened
                     here. It is only written, as text, into every
                     configs/run_NNN.json ('data': {'h5_path': ...}) and into
                     the printed sbatch line. The GPU jobs open it at training
                     time, so it must be the path as seen from an Alpine
                     compute node.
  --h5-local PATH    A copy of the HDF5 that THIS script can open now, on the
                     machine where it is running. It is read for two facts only:
                       1. the number of samples (len of 'tau_c'), which sets
                          the train/val/test split sizes and the cost estimate;
                       2. the τ normalization pair (attrs tau_c_norm_min/max),
                          which is written into each config.
                     Nothing else is read, and its path is not stored.
  --n-samples N      Instead of --h5-local, when no readable copy is at hand.
                     The τ pair then comes from --tau-bounds, else 3 40.
                     Exactly one of --h5-local / --n-samples is required.

  Running ON ALPINE: the file is local there, so pass the same path to both:
      H5=/scratch/alpine/anbu8374/.../<file>.h5
      python 11b_generate_sweep.py --tag <tag> --h5-alpine $H5 --h5-local $H5
  The configs are then already on Alpine (no push needed). Copy or commit
  training_runs/<tag>/ back to the laptop later so 11c can read it there.

  Running ON THE LAPTOP: --h5-alpine is the scratch path, --h5-local is your
  laptop copy (e.g. training_data/<file>.h5); then commit and push the configs.

  The remaining options (--n-configs, --n-replicate-configs, --n-extra-seeds,
  --n-epochs, --early-stop-patience, --sobol-seed, --tau-bounds,
  --concurrency) have defaults matching the design above; see --help.

Outputs (training_runs/<tag>/)
  configs/run_NNN.json   one per trial (replicates have their own ids)
  runs.txt               run ids, one per line, read by the array script
  search_space.json      the space, seeds, budget and data used (provenance)

Usage
    # laptop
    python 11b_generate_sweep.py --tag PT_sweep_2026-11-15 \
        --h5-alpine /scratch/alpine/anbu8374/neural_network_training_data/<file>.h5 \
        --h5-local training_data/<file>.h5
    # or on Alpine (same file for both)
    python 11b_generate_sweep.py --tag PT_sweep_2026-11-15 --h5-alpine $H5 --h5-local $H5
    then on Alpine, from lasp-CU-paper-3/ (the line is printed):
    sbatch --array=0-<n-1>%8 --export=ALL,TAG=<tag>,H5_PATH=<alpine .h5> 12_train_configs_alpine.sh

Nothing here trains. The Sobol scramble is seeded (--sobol-seed, recorded).
Author: Andrew J. Buggee, 15 Nov 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPO = Path(__file__).resolve().parent
N_LEVELS = 7
VAL_FRAC = TEST_FRAC = 0.10            # true 80/10/10 split, as in Aug 2026
SPLIT_SEED = 42                        # trainer --seed default: fixes the split

# ── Search space ─────────────────────────────────────────────────────────────
# kind: 'log' (log-uniform), 'lin' (uniform) or 'cat' (categorical; each option
# gets an equal share of the Sobol coordinate). The order of this dict fixes
# the Sobol dimensions, so do not reorder it between rounds of one study.
SEARCH_SPACE = {
    'learning_rate':       {'kind': 'log', 'range': (1e-5, 3e-3),
                            'why': 'Aug best 2.8e-5 … 9e-4; old upper bound 1e-3 was reached'},
    'weight_decay':        {'kind': 'log', 'range': (1e-6, 1e-3),
                            'why': 'fixed at 1e-4 in Aug; AdamW decay interacts with lr'},
    'dropout':             {'kind': 'lin', 'range': (0.0, 0.30),
                            'why': 'Aug: lower was better (ρ=+0.38), old lower bound 0.15'},
    'augment_noise_std':   {'kind': 'lin', 'range': (0.0, 0.03),
                            'why': 'fractional input noise; HySICS noise is 0.003'},
    'lambda_physics':      {'kind': 'lin', 'range': (0.0, 0.25), 'why': 'unchanged'},
    'lambda_monotonicity': {'kind': 'lin', 'range': (0.0, 0.25), 'why': 'unchanged'},
    'lambda_adiabatic':    {'kind': 'lin', 'range': (0.0, 0.25),
                            'why': 'Aug: higher was worse (ρ=+0.31); kept for the new profiles'},
    'lambda_smoothness':   {'kind': 'lin', 'range': (0.0, 0.25), 'why': 'unchanged'},
    'tau_weight':          {'kind': 'log', 'range': (0.02, 2.0),
                            'why': 'τ NLL weight; its meaning changed with the 3/40 τ bounds'},
    'hidden_dims':         {'kind': 'cat', 'options': [
                                [256, 256, 256],
                                [256, 256, 256, 256],
                                [512, 512, 256, 256],
                                [512, 256, 256, 128, 128],
                                [1024, 512, 256, 128]],
                            'why': '3x128 dropped (worst in Aug); a wider net added for 2.4x data'},
    'activation':          {'kind': 'cat', 'options': ['gelu', 'silu', 'relu'], 'why': 'unchanged'},
    'level_weights_name':  {'kind': 'cat', 'options': ['uniform', 'top', 'bottom', 'u_shape'],
                            'why': 'unchanged'},
    'tau_transform':       {'kind': 'cat', 'options': ['linear', 'log10'],
                            'why': 're-test: τ now normalized with 3/40'},
}

# Fixed for every trial (the "fixed hyperparameters" of this round)
FIXED_HP = {
    'batch_size':          256,      # see design note 5
    'scheduler_patience':  30,       # ReduceLROnPlateau, factor 0.5 (trainer)
    'warmup_steps':        500,
    'early_stop_metric':   'total',  # as in Aug 2026
    # log_std_min / log_std_max / sigma_floor are added in main() from
    # --log-std-min (design note 7); they used to be −4 / +2 / 0.01.
}
LOG_STD_MAX = 2.0                      # upper clamp: overflow guard only, unchanged
RE_RANGE_UM = 50.0 - 1.5               # models.RetrievalConfig re_max − re_min


def build_level_weights(scheme: str, n_levels: int = N_LEVELS) -> list[float]:
    """Per-level loss weights (index 0 = cloud top). Verbatim from
    hyper_parameter_sweep/generate_sweep_profile_tau_synthetic.py."""
    levels = np.arange(n_levels, dtype=float)
    end = n_levels - 1
    decay = max(2.0, n_levels / 5.0)
    if scheme == 'uniform':
        w = np.ones(n_levels)
    elif scheme == 'top':
        w = 1.0 + 4.0 * np.exp(-levels / decay)
    elif scheme == 'bottom':
        w = 1.0 + 4.0 * np.exp(-(end - levels) / decay)
    elif scheme == 'u_shape':
        w = 1.0 + 4.0 * (np.exp(-levels / decay) + np.exp(-(end - levels) / decay))
    else:
        raise ValueError(f'Unknown level_weights scheme: {scheme!r}')
    return [float(x) for x in w]


def map_point(u: np.ndarray) -> dict:
    """One Sobol point u ∈ [0,1)^d → hyperparameter values (same order as SEARCH_SPACE)."""
    hp = {}
    for ui, (name, spec) in zip(u, SEARCH_SPACE.items()):
        if spec['kind'] == 'log':
            lo, hi = spec['range']
            hp[name] = float(lo * (hi / lo) ** ui)
        elif spec['kind'] == 'lin':
            lo, hi = spec['range']
            hp[name] = float(lo + (hi - lo) * ui)
        else:
            opts = spec['options']
            hp[name] = opts[min(int(ui * len(opts)), len(opts) - 1)]
    return hp


def arch_tag(dims: list[int]) -> str:
    return f'{len(dims)}x{dims[0]}' if len(set(dims)) == 1 else f'{len(dims)}L_{dims[0]}-{dims[-1]}'


def main() -> None:
    p = argparse.ArgumentParser(description='Quasi-random PT hyperparameter sweep (configs only)')
    p.add_argument('--tag', required=True, help='sweep name, e.g. PT_sweep_2026-11-15')
    p.add_argument('--h5-alpine', required=True, help='HDF5 path as seen on Alpine; not opened here, only written into each '
                        'config for the GPU jobs (on Alpine: same as --h5-local)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--h5-local', type=Path, help='a copy of the same HDF5 readable on THIS machine; only the sample '
                        'count and τ bounds are read from it')
    g.add_argument('--n-samples', type=int, help='sample count, if no readable copy (instead of --h5-local)')
    p.add_argument('--n-configs', type=int, default=128,
                   help='Sobol points (a power of 2 keeps the sequence balanced)')
    p.add_argument('--n-replicate-configs', type=int, default=4,
                   help='how many configs also get extra init seeds')
    p.add_argument('--n-extra-seeds', type=int, default=2, help='extra init seeds per replicate config')
    p.add_argument('--n-epochs', type=int, default=3000)
    p.add_argument('--early-stop-patience', type=int, default=300)
    p.add_argument('--sobol-seed', type=int, default=20261115, help='scramble seed (recorded)')
    p.add_argument('--tau-bounds', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                   help='τ normalization pair; default: HDF5 attrs, else 3 40')
    p.add_argument('--concurrency', type=int, default=8, help='max simultaneous array tasks (printed)')
    p.add_argument('--log-std-min', type=float, default=-8.0,
                   help='lower log-σ clamp for every trial (design note 7; legacy −4)')
    args = p.parse_args()
    # Clamp settings shared by every trial; sigma_floor = exp(log_std_min) so
    # the loss floor coincides with the model's floor (trainer checks ≤).
    FIXED_HP.update({'log_std_min': args.log_std_min, 'log_std_max': LOG_STD_MAX,
                     'sigma_floor': float(np.exp(args.log_std_min))})

    # ── data facts: sample count and τ normalization pair ────────────────────
    stored = None
    if args.h5_local:
        import h5py
        with h5py.File(args.h5_local, 'r') as f:
            n_samples = int(f['tau_c'].shape[0])
            if 'tau_c_norm_min' in f.attrs:
                stored = (float(f.attrs['tau_c_norm_min']), float(f.attrs['tau_c_norm_max']))
    else:
        n_samples = args.n_samples
    tau_bounds = tuple(args.tau_bounds) if args.tau_bounds else (stored or (3.0, 40.0))
    n_val, n_test = round(VAL_FRAC * n_samples), round(TEST_FRAC * n_samples)
    n_train = n_samples - n_val - n_test

    if args.n_configs & (args.n_configs - 1):
        print(f'NOTE: --n-configs {args.n_configs} is not a power of 2; the Sobol '
              f'sequence is best balanced at 64, 128, 256 …')

    # ── Sobol points → configs ───────────────────────────────────────────────
    sampler = qmc.Sobol(d=len(SEARCH_SPACE), scramble=True, seed=args.sobol_seed)
    U = sampler.random(args.n_configs)                     # (n_configs, d) in [0, 1)
    out = REPO / 'training_runs' / args.tag
    cfg_dir = out / 'configs'
    if cfg_dir.exists() and any(cfg_dir.iterdir()):
        raise SystemExit(f'{cfg_dir} already has configs; choose a new --tag')
    cfg_dir.mkdir(parents=True)

    def write(run_id: int, hp_draw: dict, init_seed: int, replicate_of: int | None, u: np.ndarray):
        hp = dict(hp_draw)
        hp.update(FIXED_HP)
        hp['level_weights'] = build_level_weights(hp['level_weights_name'])
        hp['n_epochs'] = args.n_epochs
        hp['early_stop_patience'] = args.early_stop_patience
        hp['init_seed'] = init_seed
        tag = (f"{arch_tag(hp['hidden_dims'])}_{hp['activation']}_{hp['level_weights_name']}_"
               f"lr{hp['learning_rate']:.1e}_wd{hp['weight_decay']:.0e}_dr{hp['dropout']:.2f}_"
               f"tw{hp['tau_weight']:.3f}_{hp['tau_transform']}"
               + (f'_seed{init_seed}' if replicate_of is not None else ''))
        cfg = {
            'run_id': run_id, 'tag': tag, 'variant': 'PT', 'hyperparams': hp,
            'data': {'h5_path': args.h5_alpine, 'instrument': 'hysics', 'num_workers': 4},
            'n_val_profiles': n_val, 'n_test_profiles': n_test,
            'tau_bounds': [float(tau_bounds[0]), float(tau_bounds[1])],
            'output_dir': f'training_runs/{args.tag}/results',
            'sweep': {'tag': args.tag, 'sobol_seed': args.sobol_seed, 'sobol_point': u.tolist(),
                      'replicate_of': replicate_of, 'split_seed': SPLIT_SEED},
        }
        json.dump(cfg, open(cfg_dir / f'run_{run_id:03d}.json', 'w'), indent=2)

    draws = [map_point(u) for u in U]
    run_ids = []
    for i, (d, u) in enumerate(zip(draws, U)):
        write(i, d, SPLIT_SEED, None, u)
        run_ids.append(i)
    # seed replicates: same hyperparameters, different init seed, same split.
    # The first Sobol points are spread over the whole space, so these
    # replicates sample the noise at several different places in it.
    nxt = args.n_configs
    for i in range(min(args.n_replicate_configs, args.n_configs)):
        for k in range(1, args.n_extra_seeds + 1):
            write(nxt, draws[i], SPLIT_SEED + k, i, U[i])
            run_ids.append(nxt)
            nxt += 1
    (out / 'runs.txt').write_text('\n'.join(str(r) for r in run_ids) + '\n')

    # ── provenance ───────────────────────────────────────────────────────────
    space_json = {k: {kk: vv for kk, vv in v.items()} for k, v in SEARCH_SPACE.items()}
    json.dump({'tag': args.tag, 'created_by': '11b_generate_sweep.py', 'search_space': space_json,
               'fixed_hyperparams': FIXED_HP, 'n_configs': args.n_configs,
               'n_replicate_configs': args.n_replicate_configs, 'n_extra_seeds': args.n_extra_seeds,
               'sobol_seed': args.sobol_seed, 'split_seed': SPLIT_SEED,
               'n_epochs': args.n_epochs, 'early_stop_patience': args.early_stop_patience,
               'h5_alpine': args.h5_alpine, 'n_samples': n_samples,
               'split': [n_train, n_val, n_test], 'tau_bounds': list(tau_bounds)},
              open(out / 'search_space.json', 'w'), indent=2)

    # ── report ───────────────────────────────────────────────────────────────
    n_tasks = len(run_ids)
    # cost estimate (Claude's, from this repo's data, not a source): Aug-2026
    # L40 median 0.84 s/epoch at batch 256 on 33,593 training samples
    # (hyper_parameter_sweep/sweep_results_profile_tau_synthetic/*/summary.json),
    # scaled linearly with the training-set size
    s_per_epoch = 0.84 * n_train / 33593
    h_cap = s_per_epoch * args.n_epochs / 3600
    print(f'{args.n_configs} Sobol configs + {n_tasks - args.n_configs} seed replicates = {n_tasks} trials')
    print(f'split {n_train}/{n_val}/{n_test} (seed {SPLIT_SEED}); τ bounds {tau_bounds[0]:g}–{tau_bounds[1]:g}')
    print(f'log-σ clamp [{args.log_std_min:g}, {LOG_STD_MAX:g}] → σ floor '
          f'{np.exp(args.log_std_min) * RE_RANGE_UM:.4f} μm (r_e), '
          f'{np.exp(args.log_std_min) * (tau_bounds[1] - tau_bounds[0]):.4f} (τ_c); '
          f'sigma_floor {np.exp(args.log_std_min):.3g}')
    print(f'budget {args.n_epochs} epochs, patience {args.early_stop_patience}; '
          f'≈{s_per_epoch:.1f} s/epoch → ≤{h_cap:.1f} h per trial, ≤{h_cap * n_tasks:.0f} GPU-h in all')
    print(f'  (early stopping usually ends trials sooner; ≈{h_cap * n_tasks / args.concurrency:.0f} h '
          f'wall at {args.concurrency} concurrent)')
    for name, spec in SEARCH_SPACE.items():
        vals = [d[name] for d in draws]
        if spec['kind'] == 'cat':
            cnt = {str(o): sum(str(v) == str(o) for v in vals) for o in spec['options']}
            print(f'  {name:<20} {cnt}')
        else:
            print(f'  {name:<20} {min(vals):.3g} … {max(vals):.3g}  ({spec["kind"]})')
    print(f'\nwrote {cfg_dir.relative_to(REPO)}/ ({n_tasks} files), runs.txt, search_space.json')
    print('If generated on the laptop, push the repo to Alpine. Then, on Alpine from lasp-CU-paper-3/:')
    print(f'  sbatch --array=0-{n_tasks - 1}%{args.concurrency} '
          f'--export=ALL,TAG={args.tag},H5_PATH={args.h5_alpine} 12_train_configs_alpine.sh')
    print(f'Afterwards: python 11c_analyze_sweep.py --tag {args.tag}')


if __name__ == '__main__':
    main()
