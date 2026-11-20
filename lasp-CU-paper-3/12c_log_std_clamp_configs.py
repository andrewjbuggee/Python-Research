"""
12c_log_std_clamp_configs.py — STEP 12c of the paper-3 pipeline
===============================================================
Configs for a controlled test of the log-σ clamp: one chosen configuration
trained at the legacy clamp (log_std_min = −4, the control) and at a lower one
(default −8, the same as 11b_generate_sweep.py), with everything else identical. Step 12b
(12_train_configs_alpine.sh) trains them; 13b_compare_log_std_clamp.py
compares them on the VALIDATION set.

Why (17 Nov 2026): in run_004, 92–98 % of mid-cloud (levels 2–5) test
predictions and 97 % of τ predictions had σ exactly on the −4 floor
(0.888 μm for r_e), so σ there was the clamp value, not an estimate.

What changes between the arms
  hyperparams.log_std_min  −4 (control) vs --log-std-min (default −8)
  hyperparams.sigma_floor  control: the source value (0.01 < exp(−4), inert,
                           as before); test arm: exp(log_std_min), so the
                           loss-side floor equals the model's floor rather
                           than binding above it (0.01 would be 0.485 μm).
Everything else (architecture, LR, level weights, τ settings, epochs, split
seed, data) is copied from the source config.

Seed replicates: each arm is trained with --n-seeds init seeds (weight init,
dropout masks, batch order, input noise; the split is fixed). A difference
between arms smaller than the spread between seeds of one arm is not evidence
of an effect (Bouthillier et al. 2021, Proc. MLSys 3,
https://arxiv.org/abs/2103.03098). Default 2 seeds → 4 runs.

Run ids: arm-major, seed-minor: run 1..n = control seeds, n+1..2n = test arm.

Source config (one of)
  --source-config <path>   a config JSON (e.g. training_runs/<tag>/configs/run_NNN.json
                           or a finished run's config.json)
  --source-tag <tag>       the run step 13 selected for that tag
                           (training_runs/<tag>/selected_model.json → its config.json)

Usage
    python 12c_log_std_clamp_configs.py --tag LSC_2026-11-17 --source-tag PT_2026-11-14 \
        --h5-alpine /scratch/alpine/anbu8374/neural_network_training_data/<file>.h5
    # then, from lasp-CU-paper-3/ on Alpine (the script prints this line):
    sbatch --array=0-3 --export=ALL,TAG=LSC_2026-11-17,H5_PATH=<alpine .h5> 12_train_configs_alpine.sh

Nothing here is stochastic; the init seeds written are source_init_seed + k.
Author: Andrew J. Buggee, 17 Nov 2026
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent
LEGACY_LOG_STD_MIN = -4.0          # = models.LEGACY_LOG_STD_BOUNDS[0]


def source_config(args) -> tuple[dict, str]:
    """The config to copy, and a repo-relative description of where it came from."""
    if args.source_config:
        p = Path(args.source_config).resolve()
    else:
        sel = json.load(open(REPO / 'training_runs' / args.source_tag / 'selected_model.json'))
        p = (REPO / sel['run_dir'] / 'config.json').resolve()
    try:
        where = str(p.relative_to(REPO))
    except ValueError:
        where = str(p)
    return json.load(open(p)), where


def main() -> None:
    p = argparse.ArgumentParser(description='Control vs lowered log-σ clamp configs')
    p.add_argument('--tag', required=True, help='name of this test, e.g. LSC_2026-11-17')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source-config', default=None)
    g.add_argument('--source-tag', default=None)
    p.add_argument('--h5-alpine', default=None,
                   help='HDF5 path on Alpine (default: keep the source config\'s)')
    p.add_argument('--log-std-min', type=float, default=-8.0,
                   help='test-arm clamp (default −8 → 0.016 μm r_e floor, as 11b)')
    p.add_argument('--n-seeds', type=int, default=2, help='init seeds per arm')
    args = p.parse_args()

    if args.log_std_min >= LEGACY_LOG_STD_MIN:
        raise SystemExit(f'--log-std-min must be below the control ({LEGACY_LOG_STD_MIN})')
    src, src_where = source_config(args)
    hp0 = src['hyperparams']
    if float(hp0.get('log_std_min', LEGACY_LOG_STD_MIN)) != LEGACY_LOG_STD_MIN:
        raise SystemExit(f'source config already has log_std_min = {hp0["log_std_min"]}; '
                         f'the control arm needs the legacy {LEGACY_LOG_STD_MIN}')
    base_seed = int(hp0.get('init_seed', 42))

    out = REPO / 'training_runs' / args.tag
    (out / 'configs').mkdir(parents=True, exist_ok=True)
    arms = [('control', LEGACY_LOG_STD_MIN, float(hp0.get('sigma_floor', 0.01))),
            ('lowered', args.log_std_min, math.exp(args.log_std_min))]
    re_range_um = 50.0 - 1.5   # models.RetrievalConfig re_max − re_min

    run_ids, rid = [], 0
    print(f'source: {src_where}  (init seed {base_seed})')
    print(f'{"run":>4} {"arm":>8} {"log_std_min":>11} {"r_e σ floor μm":>15} '
          f'{"sigma_floor":>11} {"init_seed":>9}')
    for arm, lsm, sfloor in arms:
        for k in range(args.n_seeds):
            rid += 1
            cfg = copy.deepcopy(src)
            hp = cfg['hyperparams']
            hp['log_std_min'] = lsm
            hp['log_std_max'] = 2.0
            hp['sigma_floor'] = sfloor
            hp['init_seed'] = base_seed + k
            cfg['run_id'] = rid
            cfg['tag'] = f'{arm}_lsm{lsm:g}_seed{base_seed + k}'
            if args.h5_alpine:
                cfg['data']['h5_path'] = args.h5_alpine
            cfg['output_dir'] = f'training_runs/{args.tag}/results'
            cfg['provenance'] = {
                **cfg.get('provenance', {}),
                'log_std_clamp_test': {'tag': args.tag, 'arm': arm,
                                       'source_config': src_where},
            }
            json.dump(cfg, open(out / 'configs' / f'run_{rid:03d}.json', 'w'), indent=2)
            run_ids.append(rid)
            print(f'{rid:>4} {arm:>8} {lsm:>11g} {math.exp(lsm) * re_range_um:>15.4f} '
                  f'{sfloor:>11.3g} {base_seed + k:>9d}')
    (out / 'runs.txt').write_text('\n'.join(str(i) for i in run_ids) + '\n')

    h5 = args.h5_alpine or src['data']['h5_path']
    print(f'\nwrote {len(run_ids)} configs to {(out / "configs").relative_to(REPO)}')
    print('Upload the repo to Alpine, then from lasp-CU-paper-3/:')
    print(f'  sbatch --array=0-{len(run_ids) - 1} --export=ALL,TAG={args.tag},'
          f'H5_PATH={h5} 12_train_configs_alpine.sh')
    print(f'Afterwards, on the laptop: python 13b_compare_log_std_clamp.py --tag {args.tag}')


if __name__ == '__main__':
    main()
