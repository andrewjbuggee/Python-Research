"""
13_rank_trained_models.py — STEP 13 of the paper-3 pipeline
===========================================================
Rank the models trained in step 12 and record which one later steps use.

Reads training_runs/<tag>/results/run_NNN/summary.json (written by
sweep_train_profile_tau_synthetic.py) and prints, per run:
  ranking on mean VALIDATION profile RMSE (μm) when recorded; printed
  columns are the test profile RMSE (the paper's primary metric), τ_c RMSE and R²,
  RMSE/σ ratio (calibration of the predicted uncertainty; 1 = calibrated),
  best epoch and epochs trained (a run whose best epoch is near the cap was
  still improving), and the same run's profile / τ RMSE in the Aug-2026 PT
  sweep (from the config provenance) for a paired old-vs-new comparison.
  The old and new test sets differ (different data), so that pairing shows
  the change in difficulty + data, not a controlled experiment.

Writes
  training_runs/<tag>/ranking.csv
  training_runs/<tag>/selected_model.json   the best run by profile RMSE
                                            (or --select RUN_ID); steps 14-16
                                            read it when --run-dir is omitted

Usage
    python 13_rank_trained_models.py --tag PT_2026-11-12 [--select 4]

Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description='Rank the step-12 models')
    p.add_argument('--tag', required=True)
    p.add_argument('--select', type=int, default=None, help='run id to select instead of the best')
    args = p.parse_args()

    base = REPO / 'training_runs' / args.tag
    rows = []
    for s_path in sorted((base / 'results').glob('run_*/summary.json')):
        s = json.load(open(s_path))
        rid = int(s['run_id'])
        cfg_path = base / 'configs' / f'run_{rid:03d}.json'
        prov = json.load(open(cfg_path)).get('provenance', {}) if cfg_path.exists() else {}
        n_cap = s.get('hyperparams', {}).get('n_epochs')
        rows.append(dict(
            run_id=rid, run_dir=str(s_path.parent.relative_to(REPO)),
            mean_val_rmse_um=s.get('mean_val_rmse_um'), tau_val_rmse=s.get('tau_val_rmse'),
            mean_test_rmse_um=s['mean_test_rmse_um'], tau_test_rmse=s.get('tau_test_rmse'),
            tau_test_r2=s.get('tau_test_r2'), rmse_sigma_ratio=s.get('rmse_sigma_ratio'),
            best_epoch=s.get('best_epoch'), epochs_trained=s.get('epochs_trained'), n_epochs_cap=n_cap,
            old_sweep_rmse_um=prov.get('source_mean_test_rmse_um'),
            old_sweep_tau_rmse=prov.get('source_tau_test_rmse'),
            tag=s.get('tag', '')))
    if not rows:
        raise SystemExit(f'No summary.json under {base / "results"}')
    # Select on validation (runs trained after 15 Nov 2026 record it); fall back
    # to test only for older runs. The test RMSE of the selected run is then the
    # number to report — choosing on test would bias it low (Cawley & Talbot
    # 2010, J. Mach. Learn. Res. 11, 2079-2107,
    # https://www.jmlr.org/papers/v11/cawley10a.html).
    rank_key = ('mean_val_rmse_um' if all(r['mean_val_rmse_um'] is not None for r in rows)
                else 'mean_test_rmse_um')
    if rank_key == 'mean_test_rmse_um':
        print('NOTE: runs lack validation metrics; ranking on test RMSE')
    rows.sort(key=lambda r: r[rank_key])

    print(f'ranked on {rank_key}; profile / τ columns below are TEST values for reporting')
    print(f'{"rank":>4} {"run":>4} {"profile RMSE":>12} {"(old)":>7} {"τ RMSE":>7} {"(old)":>7} '
          f'{"τ R²":>6} {"RMSE/σ":>7} {"best/trained":>13}')
    for k, r in enumerate(rows, 1):
        r['rank'] = k
        near_cap = (r['n_epochs_cap'] and r['best_epoch'] is not None
                    and r['best_epoch'] > 0.95 * r['n_epochs_cap'])
        fmt = lambda x, f: (f'{x:{f}}' if isinstance(x, (int, float)) else '—')
        print(f'{k:>4} {r["run_id"]:>4} {fmt(r["mean_test_rmse_um"], "12.4f")} '
              f'{fmt(r["old_sweep_rmse_um"], "7.3f")} {fmt(r["tau_test_rmse"], "7.3f")} '
              f'{fmt(r["old_sweep_tau_rmse"], "7.3f")} {fmt(r["tau_test_r2"], "6.3f")} '
              f'{fmt(r["rmse_sigma_ratio"], "7.2f")} {r["best_epoch"]}/{r["epochs_trained"]}'
              + ('  ← best epoch near the cap: still improving' if near_cap else ''))

    with open(base / 'ranking.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pick = rows[0] if args.select is None else next(r for r in rows if r['run_id'] == args.select)
    json.dump(dict(tag=args.tag, run_id=pick['run_id'], run_dir=pick['run_dir'],
                   mean_test_rmse_um=pick['mean_test_rmse_um'], tau_test_rmse=pick['tau_test_rmse'],
                   mean_val_rmse_um=pick['mean_val_rmse_um'], tau_val_rmse=pick['tau_val_rmse'],
                   selected_by='--select' if args.select is not None else f'lowest {rank_key}'),
              open(base / 'selected_model.json', 'w'), indent=2)
    print(f'\nselected run {pick["run_id"]} → {base.relative_to(REPO)}/selected_model.json')
    print(f'ranking     → {base.relative_to(REPO)}/ranking.csv')


if __name__ == '__main__':
    main()
