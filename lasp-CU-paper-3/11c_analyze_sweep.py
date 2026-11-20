"""
11c_analyze_sweep.py — STEP 11c of the paper-3 pipeline
=======================================================
Rank and diagnose a sweep made by 11b_generate_sweep.py and trained with
12_train_configs_alpine.sh. Ranks on VALIDATION metrics only; test metrics are
copied into the table for the record, never used to choose, because selecting
on the data used to report error biases that error low [CT10].

What it reports
  ranking     configs by mean validation profile RMSE (μm, the paper's primary
              metric), with validation τ_c RMSE; seed replicates are averaged
              into their parent config.
  noise       run-to-run spread from the seed replicates (pooled standard
              deviation of validation RMSE across init seeds); configs within
              2σ of the best are flagged as not distinguishable. Why seeds
              matter: [Bou21]. The pooled-σ estimate and the 2σ rule (with
              σ·√(1/n₁+1/n₂) for means over n seeds) are Claude's choices —
              a standard two-sample comparison, not a procedure from [Bou21].
  budget      fraction of trials whose best epoch is in the last 10 % of the
              cap. Many → the epoch budget still limits the ranking (as in
              Aug 2026, where it was 80 %); consider a longer budget. Looking
              at where the best checkpoints land follows [Playbook]; the 10 %
              threshold is Claude's choice.
  boundaries  for each continuous hyperparameter, whether the top 10 % of
              trials crowd into the outer 10 % of its range — a sign the range
              should be extended [Playbook, "Quasi-random search"]. The
              thresholds (top 10 %, outer 10 %, half of them) are Claude's.
  sensitivity Spearman ρ of each continuous hyperparameter with validation
              RMSE, and median RMSE per categorical option (descriptive only:
              quasi-random trials vary all hyperparameters at once, so these
              are marginal associations, not effects; Claude's choice of
              summary, in the spirit of the Playbook's per-hyperparameter
              "basin" plots [Playbook, "A scientific approach…"]).
  trade-off   the Pareto set of (validation profile RMSE, validation τ RMSE):
              configs no other config beats on both (standard definition).
  σ checks    per config (from the trainer's 'val_uncertainty', trials since
              17 Nov 2026; blank for older runs): share of validation σ on the
              log-σ clamp floor (r_e, mean over levels; τ_c), and calibration
              std(z), z = error/σ, averaged over levels (1 = calibrated, < 1 σ
              too large, > 1 σ too small). Not used for ranking; reported so a
              floor that still binds, or badly calibrated σ, is seen before a
              config is chosen. The "> 10 % on the floor" warning threshold is
              Claude's choice.

Outputs (training_runs/<tag>/)
  comparison.csv      one row per config, ranked; read by 12_select_configs.py
  sweep_report.json   the numbers above
  sweep_diagnostics.png

Usage
    python 11c_analyze_sweep.py --tag PT_sweep_2026-11-15

Nothing here is stochastic.

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
  [CT10]      Cawley, G. C. and N. L. C. Talbot (2010): On over-fitting in
              model selection and subsequent selection bias in performance
              evaluation. J. Mach. Learn. Res., 11, 2079-2107.
              https://www.jmlr.org/papers/v11/cawley10a.html
  [Bou21]     Bouthillier, X. et al. (2021): Accounting for variance in machine
              learning benchmarks. Proc. MLSys, 3. https://arxiv.org/abs/2103.03098

Author: Andrew J. Buggee, 15 Nov 2026
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parent
BLUE, VERM, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#8c8c8c'


def load_trials(base: Path) -> list[dict]:
    """summary.json of every finished trial, joined with its config's sweep block."""
    rows = []
    for sp in sorted((base / 'results').glob('run_*/summary.json')):
        s = json.load(open(sp))
        cfg = json.load(open(base / 'configs' / f"run_{int(s['run_id']):03d}.json"))
        if 'mean_val_rmse_um' not in s:
            raise SystemExit(f'{sp} has no validation metrics: trained before 15 Nov 2026 trainer update')
        s['_sweep'] = cfg.get('sweep', {})
        rows.append(s)
    if not rows:
        raise SystemExit(f'no finished trials under {base / "results"}')
    return rows


def pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of points not dominated in (x, y), both minimised."""
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        keep[i] = not np.any((x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i])))
    return keep


def main() -> None:
    p = argparse.ArgumentParser(description='Analyse an 11b sweep (validation-based ranking)')
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    base = REPO / 'training_runs' / args.tag
    space = json.load(open(base / 'search_space.json'))
    trials = load_trials(base)
    n_expected = len((base / 'runs.txt').read_text().split())
    print(f'{args.tag}: {len(trials)} of {n_expected} trials finished')

    # ── group seed replicates with their parent config ───────────────────────
    groups: dict[int, list[dict]] = {}
    for s in trials:
        parent = s['_sweep'].get('replicate_of')
        groups.setdefault(int(s['run_id']) if parent is None else int(parent), []).append(s)

    # ── run-to-run noise from configs that have ≥ 2 finished seeds ───────────
    rep_sd, rep_detail = [], {}
    for cid, g in groups.items():
        if len(g) >= 2:
            v = np.array([t['mean_val_rmse_um'] for t in g])
            rep_sd.append(np.var(v, ddof=1))
            rep_detail[cid] = dict(n=len(g), val_rmse_um=v.round(4).tolist(), sd=float(np.std(v, ddof=1)))
    noise_sd = float(np.sqrt(np.mean(rep_sd))) if rep_sd else None   # pooled σ of one run

    # ── one row per config (mean over its seeds) ─────────────────────────────
    cont = [k for k, v in space['search_space'].items() if v['kind'] in ('log', 'lin')]
    cats = [k for k, v in space['search_space'].items() if v['kind'] == 'cat']
    rows = []
    for cid, g in groups.items():
        hp = g[0]['hyperparams']
        m = lambda k: float(np.mean([t[k] for t in g]))                      # noqa: E731

        def mu(k):
            """Mean over seeds (and over levels for per-level lists) of a
            val_uncertainty entry; NaN for trials trained before it existed."""
            v = [np.mean(t['val_uncertainty'][k]) for t in g if 'val_uncertainty' in t]
            return float(np.mean(v)) if v else float('nan')
        rows.append(dict(
            run_id=cid, n_seeds=len(g),
            mean_val_rmse_um=m('mean_val_rmse_um'), tau_val_rmse=m('tau_val_rmse'),
            tau_val_r2=m('tau_val_r2'), val_rmse_sigma_ratio=m('val_rmse_sigma_ratio'),
            mean_test_rmse_um=m('mean_test_rmse_um'), tau_test_rmse=m('tau_test_rmse'),
            best_epoch_fraction=m('best_epoch_fraction'), epochs_trained=m('epochs_trained'),
            train_hours=m('train_seconds') / 3600,
            log_std_min=float(hp.get('log_std_min', -4.0)),
            val_frac_sigma_on_floor=mu('frac_sigma_on_floor'),
            tau_val_frac_sigma_on_floor=mu('tau_frac_sigma_on_floor'),
            val_z_std=mu('z_std_per_level'),
            tau_val_z_std=mu('tau_z_std'),
            **{k: hp[k] for k in cont}, **{k: str(hp[k]) for k in cats}, tag=g[0].get('tag', '')))
    rows.sort(key=lambda r: r['mean_val_rmse_um'])
    for k, r in enumerate(rows, 1):
        r['rank'] = k
    best = rows[0]['mean_val_rmse_um']
    # A config averaged over n seeds has noise σ/√n; compare with the best one's.
    for r in rows:
        r['within_2sd_of_best'] = (None if noise_sd is None else
                                   bool(r['mean_val_rmse_um'] - best
                                        <= 2 * noise_sd * np.sqrt(1 / r['n_seeds'] + 1 / rows[0]['n_seeds'])))
    vx = np.array([r['mean_val_rmse_um'] for r in rows])
    vt = np.array([r['tau_val_rmse'] for r in rows])
    front = pareto(vx, vt)
    for r, f in zip(rows, front):
        r['pareto'] = bool(f)

    # ── budget and boundary checks ───────────────────────────────────────────
    bef = np.array([t['best_epoch_fraction'] for t in trials])
    top = rows[:max(1, len(rows) // 10)]
    boundary = {}
    for k in cont:
        spec = space['search_space'][k]
        lo, hi = spec['range']
        tr = (lambda v: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))) if spec['kind'] == 'log' \
            else (lambda v: (v - lo) / (hi - lo))
        pos = np.array([tr(r[k]) for r in top])
        boundary[k] = dict(top_frac_low=float(np.mean(pos < 0.1)), top_frac_high=float(np.mean(pos > 0.9)))
    sens = {k: float(spearmanr([r[k] for r in rows], vx)[0]) for k in cont}
    cat_med = {k: {o: float(np.median([r['mean_val_rmse_um'] for r in rows if r[k] == o]))
                   for o in sorted({r[k] for r in rows})} for k in cats}

    # ── print ────────────────────────────────────────────────────────────────
    print(f'\n{"rank":>4} {"run":>4} {"seeds":>5} {"val r_e RMSE":>12} {"val τ RMSE":>10} '
          f'{"best ep/cap":>11} {"±2σ":>4} {"Pareto":>6}  config')
    for r in rows[:15]:
        print(f'{r["rank"]:>4} {r["run_id"]:>4} {r["n_seeds"]:>5} {r["mean_val_rmse_um"]:>12.4f} '
              f'{r["tau_val_rmse"]:>10.3f} {r["best_epoch_fraction"]:>11.2f} '
              f'{"yes" if r["within_2sd_of_best"] else "":>4} {"*" if r["pareto"] else "":>6}  {r["tag"]}')
    if noise_sd is None:
        print('\nnoise: no config has ≥ 2 finished seeds yet — cannot judge which gaps are real')
    else:
        n_close = sum(bool(r['within_2sd_of_best']) for r in rows)
        print(f'\nnoise: run-to-run σ of validation r_e RMSE = {noise_sd:.4f} μm '
              f'(pooled over {len(rep_sd)} configs); {n_close} configs within 2σ of the best')
    print(f'budget: {100 * np.mean(bef > 0.9):.0f} % of trials (and '
          f'{100 * np.mean([r["best_epoch_fraction"] > 0.9 for r in top]):.0f} % of the top 10 %) '
          f'had their best epoch in the last 10 % of the {space["n_epochs"]}-epoch cap')
    flagged = [k for k, b in boundary.items() if max(b.values()) >= 0.5]
    print('boundaries: ' + (', '.join(f'{k} ({"low" if boundary[k]["top_frac_low"] >= 0.5 else "high"} end)'
                                      for k in flagged) if flagged else 'none flagged')
          + ' — flagged when half of the top 10 % sit in the outer 10 % of a range')
    print('sensitivity (Spearman ρ with val RMSE): '
          + ', '.join(f'{k} {v:+.2f}' for k, v in sorted(sens.items(), key=lambda kv: -abs(kv[1]))))
    for k, d in cat_med.items():
        print(f'  {k}: ' + ', '.join(f'{o} {v:.3f}' for o, v in d.items()))
    print(f'Pareto set (val r_e RMSE vs val τ RMSE): runs {[r["run_id"] for r in rows if r["pareto"]]}')
    # σ checks on the top 15 (see docstring)
    if any(np.isfinite(r['val_frac_sigma_on_floor']) for r in rows):
        print(f'\nσ checks, top 15 (validation): {"run":>4} {"log-σ min":>9} {"r_e on floor":>12} '
              f'{"τ on floor":>10} {"std(z) r_e":>10} {"std(z) τ":>8}')
        for r in rows[:15]:
            print(f'{"":>31} {r["run_id"]:>4} {r["log_std_min"]:>9g} {r["val_frac_sigma_on_floor"]:>12.2f} '
                  f'{r["tau_val_frac_sigma_on_floor"]:>10.2f} {r["val_z_std"]:>10.2f} {r["tau_val_z_std"]:>8.2f}')
        stuck = [r['run_id'] for r in rows[:15]
                 if max(r['val_frac_sigma_on_floor'], r['tau_val_frac_sigma_on_floor']) > 0.10]
        if stuck:
            print(f'WARNING: > 10 % of validation σ on the clamp floor in top-15 runs {stuck}: '
                  f'their σ is partly the clamp, not an estimate')
    else:
        print('\nσ checks: no trial has val_uncertainty (trained before 17 Nov 2026)')
    print('test metrics are in comparison.csv for the record; they were not used for ranking')

    # ── files ────────────────────────────────────────────────────────────────
    with open(base / 'comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank'] + [k for k in rows[0] if k != 'rank'])
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(tag=args.tag, n_trials=len(trials), n_expected=n_expected, noise_sd_um=noise_sd,
                   replicates=rep_detail, budget_frac_best_epoch_gt_0p9=float(np.mean(bef > 0.9)),
                   boundary=boundary, spearman=sens, categorical_median=cat_med,
                   pareto_run_ids=[r['run_id'] for r in rows if r['pareto']],
                   ranked_run_ids=[r['run_id'] for r in rows]),
              open(base / 'sweep_report.json', 'w'), indent=2)

    # ── figure: rank curve with noise band, Pareto, LR and budget ───────────
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax[0].plot(np.arange(1, len(vx) + 1), vx, 'o', ms=3, color=BLUE)
    if noise_sd is not None:
        ax[0].axhspan(best, best + 2 * noise_sd, color=GREY, alpha=0.25, lw=0,
                      label=f'best + 2σ run-to-run ({2 * noise_sd:.3f} μm)')
        ax[0].legend(frameon=False, fontsize=8)
    ax[0].set(xlabel='rank (validation)', ylabel='validation r$_e$ RMSE (μm)', title='(a) Ranked configs')
    ax[1].scatter(vx, vt, s=10, color=GREY, label='configs')
    ax[1].plot(vx[front][np.argsort(vx[front])], vt[front][np.argsort(vx[front])], 'o-', color=VERM,
               ms=4, lw=1.2, label='Pareto set')
    ax[1].set(xlabel='validation r$_e$ RMSE (μm)', ylabel='validation τ$_c$ RMSE',
              title='(b) Profile vs τ trade-off')
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].scatter([r['learning_rate'] for r in rows], vx, s=10,
                  c=[r['best_epoch_fraction'] for r in rows], cmap='viridis', vmin=0, vmax=1)
    ax[2].set(xscale='log', xlabel='learning rate', ylabel='validation r$_e$ RMSE (μm)',
              title='(c) LR (colour: best epoch / cap)')
    fig.colorbar(ax[2].collections[0], ax=ax[2], label='best epoch / cap')
    fig.suptitle(f'{args.tag}: {len(trials)} trials')
    fig.savefig(base / 'sweep_diagnostics.png', dpi=180)
    plt.close(fig)
    print(f'\nwrote {base.relative_to(REPO)}/comparison.csv, sweep_report.json, sweep_diagnostics.png')


if __name__ == '__main__':
    main()
