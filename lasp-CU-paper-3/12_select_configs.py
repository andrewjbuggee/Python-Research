"""
12_select_configs.py — STEP 12a of the paper-3 pipeline
=======================================================
Pick the hyperparameter configurations to retrain on the new training HDF5
and write their config files. Step 12b (12_train_configs_alpine.sh) trains
them on Alpine.

Why retrain a few configs rather than sweep again: the profile + τ (PT) sweep
of Aug 2026 (100 configs, hyper_parameter_sweep/sweep_results_profile_tau_
synthetic/) already mapped the hyperparameter space; the 5-Nov-2026 plan is to
retrain the top ~10 on the new data. Set --top 100 to retrain every config,
which is a full sweep.

What it does
  1. Ranks a sweep and keeps the top N, plus the best-τ config if it is not
     already among them.
       --sweep-tag <tag>  (recommended, Nov 2026): an 11b sweep; reads
                          training_runs/<tag>/comparison.csv from
                          11c_analyze_sweep.py and ranks on VALIDATION r_e RMSE
                          (best-τ by validation τ RMSE).
       default            the Aug-2026 PT sweep; its comparison.csv only has
                          TEST metrics, so that ranking selects on the test set
                          (kept for reproducing the earlier retrain). Selecting
                          on test biases the reported test error low: Cawley &
                          Talbot 2010, J. Mach. Learn. Res. 11, 2079-2107,
                          https://www.jmlr.org/papers/v11/cawley10a.html
  2. Copies each run's config JSON and updates it for the new data:
       data.h5_path            → the new HDF5 (path on Alpine)
       n_val/n_test_profiles   → 10 % of the new sample count each. The trainer
                                 uses these as fixed counts; the PT configs
                                 hold 4,199 = 10 % of the old 41,991-sample
                                 file, which would be only ~4 % of a 100k file.
       hyperparams.n_epochs / early_stop_patience → longer budget (the PT top
                                 runs were still improving at the 1500 cap)
       tau_bounds              → the τ normalization pair (τ_min, τ_max) for the
                                 network: from the HDF5 attrs when --h5-local is
                                 given, else --tau-bounds (default 3 40 = the
                                 generator's τ_c limits). The Aug-2026 sweep used
                                 the legacy 3 / 97, so tau_weight / tau_transform
                                 tuned there act on a different τ scale here.
       output_dir, provenance  → where results go and where the config came from
  3. Writes training_runs/<tag>/configs/run_NNN.json and runs.txt (one run id
     per line, read by the array job), and prints the sbatch command.

Usage
    python 12_select_configs.py --tag PT_2026-11-12 \
        --h5-alpine /scratch/alpine/anbu8374/neural_network_training_data/<new>.h5 \
        --n-samples 100000 [--top 10]
    (or --h5-local <copy>.h5 to read the sample count from the file)
    python 12_select_configs.py --tag PT_final_2026-11 --sweep-tag PT_sweep_2026-11-15 \
        --h5-alpine <alpine .h5> --h5-local <local .h5> --top 5    # from an 11b sweep

Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SWEEP_DIR = REPO / 'hyper_parameter_sweep' / 'sweep_results_profile_tau_synthetic'
CONFIG_DIR = REPO / 'hyper_parameter_sweep' / 'sweep_configs_profile_tau_synthetic'
VAL_FRAC = TEST_FRAC = 0.10          # true 80/10/10 split, as in the PT sweep


def main() -> None:
    p = argparse.ArgumentParser(description='Select PT configs to retrain on a new HDF5')
    p.add_argument('--tag', required=True, help='name of this training round, e.g. PT_2026-11-12')
    p.add_argument('--h5-alpine', required=True, help='path of the new HDF5 on Alpine')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--n-samples', type=int, help='number of samples in the new HDF5')
    g.add_argument('--h5-local', type=Path, help='local copy of the HDF5 (sample count read from it)')
    p.add_argument('--top', type=int, default=10, help='how many configs by profile RMSE (100 = all)')
    p.add_argument('--no-best-tau', action='store_true', help='do not add the best-τ run')
    p.add_argument('--n-epochs', type=int, default=4500)
    p.add_argument('--early-stop-patience', type=int, default=300)
    p.add_argument('--tau-bounds', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                   help='τ normalization pair; default: HDF5 attrs of --h5-local, else 3 40')
    p.add_argument('--sweep-tag', default=None,
                   help='11b sweep to select from (training_runs/<tag>/); ranks on validation')
    p.add_argument('--sweep-dir', type=Path, default=SWEEP_DIR)
    p.add_argument('--config-dir', type=Path, default=CONFIG_DIR)
    args = p.parse_args()

    stored_bounds, tau_lim = None, None
    if args.h5_local:
        import h5py
        with h5py.File(args.h5_local, 'r') as f:
            n_samples = int(f['tau_c'].shape[0])
            tau = f['tau_c'][:]
            tau_lim = (float(tau.min()), float(tau.max()))
            if 'tau_c_norm_min' in f.attrs and 'tau_c_norm_max' in f.attrs:
                stored_bounds = (float(f.attrs['tau_c_norm_min']), float(f.attrs['tau_c_norm_max']))
    else:
        n_samples = args.n_samples
    if args.tau_bounds is not None:
        tau_bounds, tb_src = tuple(args.tau_bounds), '--tau-bounds'
    elif stored_bounds is not None:
        tau_bounds, tb_src = stored_bounds, 'HDF5 attrs'
    else:
        tau_bounds, tb_src = (3.0, 40.0), "default (06 generator's τ_c limits)"
    if stored_bounds is not None and not all(abs(a - b) < 1e-9 for a, b in zip(tau_bounds, stored_bounds)):
        print(f'NOTE: τ bounds {tau_bounds} differ from the HDF5 attrs {stored_bounds}')
    if tau_lim and (tau_lim[0] < tau_bounds[0] or tau_lim[1] > tau_bounds[1]):
        raise SystemExit(f'τ_c in the file spans {tau_lim}, outside the bounds {tau_bounds}')
    print(f'τ normalization bounds: {tau_bounds[0]:g}–{tau_bounds[1]:g} ({tb_src})')
    n_val = round(VAL_FRAC * n_samples)
    n_test = round(TEST_FRAC * n_samples)

    if args.sweep_tag:
        sweep_base = REPO / 'training_runs' / args.sweep_tag
        args.sweep_dir, args.config_dir = sweep_base, sweep_base / 'configs'
    rows = list(csv.DictReader(open(args.sweep_dir / 'comparison.csv')))
    # Rank on validation when the sweep recorded it (11c); the Aug-2026 table
    # only has test metrics.
    prof_key, tau_key = (('mean_val_rmse_um', 'tau_val_rmse') if 'mean_val_rmse_um' in rows[0]
                         else ('mean_test_rmse_um', 'tau_test_rmse'))
    if prof_key == 'mean_test_rmse_um':
        print('NOTE: this sweep has no validation metrics; ranking on TEST RMSE (selection on test)')
    ok = [r for r in rows if r[prof_key] not in ('', 'nan')]
    by_profile = sorted(ok, key=lambda r: float(r[prof_key]))
    for k, r in enumerate(by_profile, 1):
        r['rank'] = k
    chosen = by_profile[:args.top]
    if not args.no_best_tau:
        best_tau = min((r for r in ok if r.get(tau_key) not in (None, '', 'nan')),
                       key=lambda r: float(r[tau_key]))
        if best_tau['run_id'] not in {r['run_id'] for r in chosen}:
            chosen.append(best_tau)

    out = REPO / 'training_runs' / args.tag
    cfg_out = out / 'configs'
    cfg_out.mkdir(parents=True, exist_ok=True)
    run_ids = []
    print(f'{n_samples} samples → train/val/test = {n_samples - n_val - n_test}/{n_val}/{n_test}')
    print(f'ranked on {prof_key} / {tau_key}')
    print(f'{"run":>5} {"rank":>5} {"profile RMSE μm":>16} {"τ RMSE":>8}  tag')
    for r in chosen:
        rid = int(r['run_id'])
        cfg = json.load(open(args.config_dir / f'run_{rid:03d}.json'))
        src_tau_bounds = cfg.get('tau_bounds', [3.0, 97.0])   # Aug sweep: legacy 3/97
        cfg['data']['h5_path'] = args.h5_alpine
        cfg['n_val_profiles'] = n_val
        cfg['n_test_profiles'] = n_test
        cfg['hyperparams']['n_epochs'] = args.n_epochs
        cfg['hyperparams']['early_stop_patience'] = args.early_stop_patience
        cfg['tau_bounds'] = [float(tau_bounds[0]), float(tau_bounds[1])]
        cfg['output_dir'] = f'training_runs/{args.tag}/results'
        cfg['provenance'] = {
            'source_config': str((args.config_dir / f'run_{rid:03d}.json').relative_to(REPO)),
            'source_sweep_rank': int(r['rank']),
            'source_rank_metric': prof_key,
            'source_mean_test_rmse_um': float(r['mean_test_rmse_um']),
            'source_tau_test_rmse': float(r['tau_test_rmse']) if r.get('tau_test_rmse') else None,
            'source_mean_val_rmse_um': float(r['mean_val_rmse_um']) if r.get('mean_val_rmse_um') else None,
            'source_tau_val_rmse': float(r['tau_val_rmse']) if r.get('tau_val_rmse') else None,
            'source_tau_bounds': src_tau_bounds,
            'source_sweep_tag': args.sweep_tag,
            'retrain_tag': args.tag,
        }
        json.dump(cfg, open(cfg_out / f'run_{rid:03d}.json', 'w'), indent=2)
        run_ids.append(rid)
        print(f'{rid:>5} {r["rank"]:>5} {float(r[prof_key]):>16.4f} '
              f'{float(r[tau_key]):>8.3f}  {r["tag"][:60]}')
    (out / 'runs.txt').write_text('\n'.join(str(i) for i in run_ids) + '\n')

    print(f'\nwrote {len(run_ids)} configs to {cfg_out.relative_to(REPO)} and {(out / "runs.txt").relative_to(REPO)}')
    print('Upload the repo to Alpine, then from lasp-CU-paper-3/:')
    print(f'  sbatch --array=0-{len(run_ids) - 1}%6 --export=ALL,TAG={args.tag},'
          f'H5_PATH={args.h5_alpine} 12_train_configs_alpine.sh')


if __name__ == '__main__':
    main()
