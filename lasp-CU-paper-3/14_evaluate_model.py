"""
14_evaluate_model.py — STEP 14 of the paper-3 pipeline
======================================================
Test-set evaluation and the paper figures for one trained profile + τ model.

Runs plot_run_figures_profile_tau.py on the run directory, which
  * runs the model once on its held-out test split and caches the predictions
    in <run>/pred_cache.npz (pred, pred_std, true, tau_pred, tau_pred_std,
    tau_true, h5_idx) — steps 15 and 16 read that cache;
  * writes per_cloud_rmse_paper_{4,6}panel.png (error vs cloud predictors,
    including Wood (2005) adiabaticity with the 7-Nov-2026 base rule, via
    plot_per_cloud_rmse_kfold.compute_predictors), per_level_error_percentiles
    {,_no_sigma}.png, profiles_true_vs_pred_percentiles.png and
    tau_retrieval_diagnostics.png.

The run is the one recorded by step 13 (training_runs/<tag>/selected_model.json)
unless --run-dir is given.

Usage
    python 14_evaluate_model.py --tag PT_2026-11-12 [--h5-path local.h5] [--refresh]
    python 14_evaluate_model.py --run-dir training_runs/PT_2026-11-12/results/run_004

Author: Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def selected_run_dir(tag: str | None, run_dir: str | None) -> Path:
    """--run-dir if given, else the run step 13 selected for --tag."""
    if run_dir:
        return Path(run_dir).resolve()
    if not tag:
        raise SystemExit('give --run-dir or --tag')
    sel = json.load(open(REPO / 'training_runs' / tag / 'selected_model.json'))
    return (REPO / sel['run_dir']).resolve()


def main() -> int:
    p = argparse.ArgumentParser(description='Evaluate one trained PT model and make its figures')
    p.add_argument('--tag', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--h5-path', default=None, help='local HDF5 if the path in config.json is not reachable')
    p.add_argument('--refresh', action='store_true', help='recompute pred_cache.npz')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    args = p.parse_args()

    run_dir = selected_run_dir(args.tag, args.run_dir)
    cmd = [sys.executable, str(REPO / 'plot_run_figures_profile_tau.py'), '--run-dir', str(run_dir)]
    if args.h5_path:
        cmd += ['--h5-path', args.h5_path]
    if args.refresh:
        cmd += ['--refresh']
    if args.device:
        cmd += ['--device', args.device]
    print('running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=REPO)


if __name__ == '__main__':
    sys.exit(main())
