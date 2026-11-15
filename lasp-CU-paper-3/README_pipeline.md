# Paper 3 pipeline: synthetic profiles → spectra → trained network → results

Numbered scripts run in order. Steps 06–07 make the training inputs, the
HySICS spectra are computed on Alpine in MATLAB, and steps 10–16 train and
evaluate the profile + τ network (`DropletProfileNetwork`: 7-level r_e + σ and
τ_c + σ from 636 HySICS reflectances + 4 geometry angles).

| Step | Script | Where | What it does |
|---|---|---|---|
| 06 | `06_synthetic_profile_generator.py` | laptop | joint-MVN synthetic profiles (PCs, LWC ceiling, Mie τ_c label) |
| 06b | `06b_replot_pca_diagnostic_3x1.py` | laptop | PCA / in-situ comparison figures |
| 07 | `07_build_training_inputs.py` | laptop | training-input NetCDF + solar/viewing geometry (libRadtran φ0 convention) |
| 08 / 08b | `08_verify_training_inputs.py`, `08b_compare_lwc_insitu_vs_synthetic.py` | laptop | checks of the inputs |
| — | `Matlab-Research/.../9_Nov_2026_synthetic/synthetic_12_Nov_2026.sh` → `run_synthetic_spectra_chunk.m` | Alpine (CPU) | HySICS spectra, 4 uvspec runs per cloud, one `.mat` per task |
| **10** | `10_build_training_hdf5.py` | Alpine or laptop | chunk files → training HDF5; seeded noise; stores Wood (2005) f_ad, cloud base, Γ_ad, LWP truth |
| **11** | `11_check_training_hdf5.py` | laptop | hard checks (exit 1 on failure) + one-page figure |
| **11b** | `11b_generate_sweep.py` | laptop | quasi-random (Sobol) hyperparameter sweep configs + seed replicates for the new HDF5 |
| — | `12_train_configs_alpine.sh` | Alpine (L40 GPU) | trains the sweep trials (same array script as 12b) |
| **11c** | `11c_analyze_sweep.py` | laptop | ranks the sweep on VALIDATION metrics; noise, budget, boundary and sensitivity checks; `comparison.csv` |
| **12a** | `12_select_configs.py` | laptop | top-N configs of a sweep (`--sweep-tag`, validation ranking; default = Aug-2026 sweep) rewritten for a long retrain (4500 epochs) |
| **12b** | `12_train_configs_alpine.sh` | Alpine (L40 GPU) | trains each selected config (array job) |
| **13** | `13_rank_trained_models.py` | laptop | ranks the retrained runs on validation, writes `selected_model.json` |
| **14** | `14_evaluate_model.py` | laptop | test-set predictions (`pred_cache.npz`) + paper figures |
| **15** | `15_lwp_from_retrievals.py` | laptop | LWP from retrieved r_e + τ_c with the Mie-consistent formula |
| **16** | `16_error_vs_adiabaticity.py` | laptop | profile / τ_c / LWP error vs Wood (2005) adiabaticity |

Shared physics: `paper3_physics.py` (Wood 2005 adiabaticity — a verbatim
copy of `plot_per_cloud_rmse_kfold.py`, identical results; Mie extinction as
in the τ_c label; LWP formulas).

## Commands (tags `PT_sweep_2026-11-15` and `PT_2026-11-14` as examples)

```bash
# 10 — on Alpine, where the chunk files are (or after copying them)
python 10_build_training_hdf5.py /scratch/alpine/anbu8374/neural_network_training_data/synthetic_dataSet_ver2_14_Nov_2026 \
    --out /scratch/alpine/anbu8374/neural_network_training_data/synthetic_training_data_ver2_14_Nov_2026.h5 \
    --n-expected 100000

# 10b — only for an HDF5 built before 15 Nov 2026: add the τ normalization attrs (3 / 40)
python 10_build_training_hdf5.py --stamp-tau-bounds <file.h5>

# 11 — laptop copy of the HDF5 in training_data/
python 11_check_training_hdf5.py training_data/synthetic_training_data_ver2_14_Nov_2026.h5 --check-tau 500

# 11b — laptop; writes training_runs/PT_sweep_2026-11-15/ and prints the sbatch line
python 11b_generate_sweep.py --tag PT_sweep_2026-11-15 \
    --h5-alpine /scratch/alpine/anbu8374/neural_network_training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5 \
    --h5-local training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5
# Alpine, from lasp-CU-paper-3/ after pushing (136 trials = 128 Sobol configs + 8 seed replicates)
sbatch --array=0-135%8 --export=ALL,TAG=PT_sweep_2026-11-15,H5_PATH=/scratch/alpine/anbu8374/neural_network_training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5 12_train_configs_alpine.sh

# 11c — laptop, after copying training_runs/PT_sweep_2026-11-15/results back
python 11c_analyze_sweep.py --tag PT_sweep_2026-11-15

# 12a — laptop; long retrain of the sweep's best configs; prints the sbatch line for 12b
python 12_select_configs.py --tag PT_2026-11-14 --sweep-tag PT_sweep_2026-11-15 \
    --h5-alpine /scratch/alpine/anbu8374/neural_network_training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5 \
    --h5-local training_data/synthetic_dataSet_7-levels_14_Nov_2026.h5 --top 5

# 12b — Alpine, from lasp-CU-paper-3/ after pushing the repo
sbatch --array=0-10%6 --export=ALL,TAG=PT_2026-11-14,H5_PATH=<alpine .h5> 12_train_configs_alpine.sh

# 13-16 — laptop, after copying training_runs/PT_2026-11-14/results back
python 13_rank_trained_models.py --tag PT_2026-11-14
python 14_evaluate_model.py      --tag PT_2026-11-14 --h5-path training_data/<file>.h5
python 15_lwp_from_retrievals.py --tag PT_2026-11-14 --h5-path training_data/<file>.h5
python 16_error_vs_adiabaticity.py --tag PT_2026-11-14 --h5-path training_data/<file>.h5
```

Outputs of each training round live in `training_runs/<tag>/`
(`configs/`, `runs.txt`, `results/run_NNN/`, `ranking.csv`, `selected_model.json`).

## What changed relative to the June/August 2026 workflow

* **LWP (step 15).** Previously Paper-2 Eq. 10 with Q_ext = 2
  (`plot_lwp_analysis_profile_tau.py`). The old τ_c label used Q = 2.32 (Mie
  ≈ 2.11 × the water-cloud-file geometry error), so Eq. 10 read ≈ 16 % high
  wherever N is constant with height. The new label is the Mie optical depth
  and the RT now sees exactly that τ_c, so LWP uses the same Mie extinction:
  `LWP = 1000 · τ_c · ∫r_e³dz / ∫ext(r_e;α) r_e³dz` (derivation in
  `paper3_physics.py`; it reduces to Eq. 10 when Q̄_ext = 2). On the 24-cloud
  Alpine test set with true inputs: +0.9 % median vs the LWC-integrated truth,
  against +5.9 % for Eq. 10 with Q = 2.
* **Adiabaticity (steps 10, 11, 16).** Wood (2005) f_ad with the 7-Nov-2026
  cloud-base rule (line fit to the bottom third of LWC(z), fallback
  z_lowest − 0.05/Γ_ad), the same rule as the MATLAB in-situ pipeline, is now
  stored per sample in the HDF5. The paper figures (step 14) already use the
  same function through `plot_per_cloud_rmse_kfold.compute_predictors`.
* **Training data.** One HDF5 from per-chunk files (step 10) instead of
  ~100,000 per-cloud `.mat` files (`convert_synthetic_matFiles_to_HDF.py`).
  Noise is seeded (`RANDOM_SEED = 20261114`, stored as an HDF5 attribute).
* **τ normalization bounds.** The τ head's pair (τ_min, τ_max) used to be the
  constant 3 / 97 in `data.py`. It is now 3 / 40 (the generator's τ_c limits),
  stored as HDF5 attrs by step 10, written into each config by 12a, saved by the
  trainer in `config.json` / `summary.json` / `best_model.pt`, and read back by
  the evaluation scripts (`data.tau_bounds_for_run`). Runs without saved bounds
  fall back to 3 / 97. Note: `Paper 3/short reports …/tau_normalization_bounds_15_Nov_2026.docx`.
* **Hyperparameter sweep (steps 11b/11c, 15 Nov 2026).** A new sweep on the new
  data, not a reuse of the Aug-2026 ranking: Sobol quasi-random search, ranking on
  validation (the Aug comparison ranked on test), seed replicates to measure
  run-to-run noise, a 3,000-epoch budget (the Aug 1,500-epoch cap was binding for
  80 % of runs), batch size fixed at 256, and bounds moved where the Aug optimum
  sat on an edge. The trainer now separates the split seed (`--seed`) from the
  initialisation seed (`hyperparams.init_seed`) and writes validation RMSEs in
  physical units. Design notes and sources: docstring of `11b_generate_sweep.py`.
* **Split sizes.** The PT configs fixed val/test at 4,199 (10 % of the old
  41,991 samples); step 12a resets them to 10 % of the new file.

## Older scripts (kept unchanged so earlier results stay reproducible)

| Purpose | Earlier script | Numbered replacement |
|---|---|---|
| per-cloud `.mat` → HDF5 | `convert_synthetic_matFiles_to_HDF.py` | 10 |
| PT sweep configs / run | `hyper_parameter_sweep/generate_sweep_profile_tau_synthetic.py`, `sweep_alpine_profile_tau_synthetic.sh` | 12a / 12b (retrain top N) |
| PT trainer | `hyper_parameter_sweep/sweep_train_profile_tau_synthetic.py` | used by 12b unchanged |
| long retrain | `shell_scripts/standalone_profile_tau_al40.sh` | 12b with `--top 1` |
| sweep ranking | `hyper_parameter_sweep/compare_sweep_profile_only_synthetic.py` | 13 |
| figures | `plot_run_figures_profile_tau.py` | wrapped by 14 |
| LWP | `plot_lwp_analysis_profile_tau.py`, `compare_lwp_methods_retrieved_profiles.py` | 15 |
| profile-only network (June 2026, M0 run_004) | `train_standalone_profile_only_synthetic.py`, `shell_scripts/standalone_synthetic_NN_al40.sh`, `sweep_alpine_profile_only_synthetic_M0/M2.sh` | superseded by the profile + τ network |

Not yet numbered: feature importance (`compute_/plot_feature_importance_profile_tau.py`)
and the in-situ VOCALS-REx / ORACLES comparison (`03_vocals_validation.ipynb`).
Before applying a new model to the in-situ spectra, check whether those spectra
were computed before the Nov-2026 mixed-layer water-cloud fix (not checked here).

## Sources for the sweep design (steps 11b/11c)

Full citations are in the docstrings of `11b_generate_sweep.py` and `11c_analyze_sweep.py`.

* Godbole, V., G. E. Dahl, J. Gilmer, C. J. Shallue and Z. Nado (2023): *Deep Learning Tuning Playbook*, Google Research — quasi-random search, batch size not tuned for validation performance, checking where best checkpoints land. https://github.com/google-research/tuning_playbook
* Bergstra, J. and Y. Bengio (2012): Random search for hyper-parameter optimization. *J. Mach. Learn. Res.*, 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
* Cawley, G. C. and N. L. C. Talbot (2010): On over-fitting in model selection and subsequent selection bias in performance evaluation. *J. Mach. Learn. Res.*, 11, 2079–2107 — rank on validation, report test. https://www.jmlr.org/papers/v11/cawley10a.html
* Bouthillier, X. et al. (2021): Accounting for variance in machine learning benchmarks. *Proc. MLSys*, 3 — seed replicates. https://arxiv.org/abs/2103.03098
* Li, L. et al. (2020): A system for massively parallel hyperparameter tuning. *Proc. MLSys*, 2 — ASHA (considered, not used). https://arxiv.org/abs/1810.05934
* Sobol', I. M. (1967): *USSR Comput. Math. Math. Phys.*, 7(4), 86–112; implementation `scipy.stats.qmc.Sobol`.
* Loshchilov, I. and F. Hutter (2019): Decoupled weight decay regularization. *ICLR* (AdamW). https://arxiv.org/abs/1711.05101

Not from these sources (Claude's choices, labelled in the code): the 2σ indistinguishability rule, the 10 % budget and boundary thresholds, the search-space edits inferred from the Aug-2026 sweep, and the cost estimate.
