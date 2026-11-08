# Choosing the PC truncation k for the synthetic-profile generator

Analysis of 8 Nov 2026 for Paper 3 (Peter's request: truncate log r_e(z) and log LWC(z) further).

## Scripts (run in this order)

| Script | What it does | Runtime |
|---|---|---|
| `compute_hysics_re_sensitivity.py` | libRadtran/DISORT: per-HySICS-band dR/d ln r_e (τ fixed) and dR/d ln τ (r_e fixed), homogeneous cloud, τ = 8, same geometry as the weighting-function run | ~40 s |
| `pc_truncation_analysis.py` | All truncation criteria (Part A statistical, Part B retrievability), table, summary, 4 figures | ~15 s |

Input file and filters are identical to `06_synthetic_profile_generator.py`
(`insitu_era5_synthetic_inputs_combined_08-Nov-2026.mat`, 3 ≤ τ_c ≤ 40, de-duplicated, L = 80, log space):
**N = 292 profiles** (ORACLES 211, VOCALS-REx 81).

Outputs: `outputs/pc_truncation_summary_2026-11-08.txt`, `outputs/pc_truncation_table_2026-11-08.csv`, `figures/fig1–fig4_*.png`.

## Results (from the 2026-11-08 run)

### Part A: variance retained (log r_e, N = 292)

| k | in-sample, log | leave-one-out, log | leave-one-out, μm | LOO RMSE (μm) |
|---|---|---|---|---|
| 1 | 84.8 % | 84.5 % | 85.3 % | 1.55 |
| 2 | 90.1 % | 89.7 % | 90.5 % | 1.24 |
| 3 | 92.0 % | 91.6 % | 91.9 % | 1.15 |
| 4 | 93.4 % | 92.8 % | 92.7 % | 1.09 |
| 6 (draft 23) | 95.2 % | 94.4 % | 94.1 % | 0.98 |

Mean profile only (k = 0): RMSE 4.08 μm.

### Part A: variance retained (log LWC)

| k | in-sample, log | LOO, log | LOO, g m⁻³ | LOO RMSE (g m⁻³) |
|---|---|---|---|---|
| 1 | 62.1 % | 61.5 % | 68.3 % | 0.112 |
| 2 | 74.2 % | 73.3 % | 79.3 % | 0.091 |
| 3 | 81.7 % | 80.7 % | 82.7 % | 0.083 |
| 4 | 84.9 % | 83.7 % | 83.5 % | 0.081 |
| 8 (draft 23) | 92.5 % | 91.1 % | 89.4 % | 0.065 |

### Part A: stopping rules

| Rule | log r_e | log LWC |
|---|---|---|
| Broken stick | 2 | 3 |
| Parallel analysis (95 %) | 2 | 4 |
| North et al. (1982) separation | 12 | 12 |
| Gavish–Donoho | 27 | 26 |
| FVE ≥ 90 / 95 / 99 % | 2 / 6 / 16 | 7 / 12 / 16 |

North and Gavish–Donoho answer a different question: are the modes distinguishable from *sampling or white noise*? They are, out to about 12 modes. They do not say whether the modes are physically relevant. The 80-level grid is interpolated, so its noise is not white, and G–D over-counts on it.

### Part B: retrievability of log r_e modes by HySICS (linear optimal estimation)

Averaging-kernel diagonal A_kk is the fraction of a mode's natural (in-situ) variability that the measurement determines. DFS is the degrees of freedom for signal in the r_e block. The state is 40 PCs plus ln τ, with σ_rad = 0.3 %.

| σ_fm | DFS | A₁ | A₂ | A₃ | A₄ | A₅ | A₆ |
|---|---|---|---|---|---|---|---|
| 0 % | 2.83 | 1.00 | 0.63 | 0.23 | 0.50 | 0.15 | 0.08 |
| 1.5 % | 2.07 | 0.99 | 0.42 | 0.22 | 0.28 | 0.03 | 0.03 |
| 3 % | 1.88 | 0.99 | 0.36 | 0.19 | 0.23 | 0.01 | 0.02 |

This independent physical estimate agrees with your April PC-score regression, which found PC1 retrievable, PC2 marginal, and PC3+ at the noise floor
(`pca_profile_experiment/figures/pc_retrievability_kfold`).

## Caveats (read before quoting numbers)

1. **Part B is a single scene and linear.** It uses τ = 8, sza 49°, vza 32°, and weighting functions for a 12 → 5 μm adiabatic cloud. Water vapour, cloud height, α, aerosol and surface are treated as known, and band noise is independent across 636 bands. Treat it as an upper bound.
2. **Weighting-function file.** `..._near-nadir-view_...` (sza 10°, vza 0°, Cox–Munk ocean) is sun-glint contaminated: R(τ) falls with τ for τ < 1, and w < 0 near cloud top. Check which file produced draft Fig. 5.
3. **The Twomey count in draft §IV.A (7 pieces) and DFS ≈ 2 here do not contradict each other. They measure different things.**
   * Twomey's count is the number of kernel directions above noise. It ignores how much real clouds vary along those directions.
   * Eq. 5 of the draft also converts the reflectance error to an r_e error with the same σ_rel for every band. Non-absorbing bands have |d ln R / d ln r_e| ≈ 0.1 (this run), so their r_e error is about 10× larger than assumed.
4. A row-wise LOO reconstruction error always decreases with k (Bro et al. 2008). It quantifies what a chosen k retains; it cannot choose k.

## References

**Truncation rules and EOF practice**
- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer. doi:10.1007/b98835. §6.1, rules for choosing the number of PCs (cumulative variance, broken stick, cross-validation).
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences*, 4th ed. Elsevier. The PCA/EOF chapter's section on truncating the PCs.
- Hannachi, A., Jolliffe, I. T., & Stephenson, D. B. (2007). Empirical orthogonal functions and related techniques in atmospheric science: A review. *Int. J. Climatol.*, 27, 1119–1152. doi:10.1002/joc.1499
- North, G. R., Bell, T. L., Cahalan, R. F., & Moeng, F. J. (1982). Sampling errors in the estimation of empirical orthogonal functions. *Mon. Wea. Rev.*, 110, 699–706. doi:10.1175/1520-0493(1982)110<0699:SEITEO>2.0.CO;2
- Jackson, D. A. (1993). Stopping rules in principal components analysis: a comparison of heuristical and statistical approaches. *Ecology*, 74, 2204–2214. doi:10.2307/1939574
- Frontier, S. (1976). Étude de la décroissance des valeurs propres dans une analyse en composantes principales: comparaison avec le modèle du bâton brisé. *J. Exp. Mar. Biol. Ecol.*, 25, 67–75.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30, 179–185. doi:10.1007/BF02289447
- Overland, J. E., & Preisendorfer, R. W. (1982). A significance test for principal components applied to a cyclone climatology. *Mon. Wea. Rev.*, 110, 1–4. Also Preisendorfer, R. W. (1988), *Principal Component Analysis in Meteorology and Oceanography*, Elsevier ("Rule N").
- Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for singular values is 4/√3. *IEEE Trans. Inf. Theory*, 60, 5040–5053. doi:10.1109/TIT.2014.2323359
- Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008). Cross-validation of component models: a critical look at current methods. *Anal. Bioanal. Chem.*, 390, 1241–1251. doi:10.1007/s00216-007-1790-1
- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*, 2nd ed. Springer. Functional PCA chapter.

**Information content / retrievability**
- Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and Practice*. World Scientific. doi:10.1142/3171. Ch. 2 (averaging kernel, degrees of freedom for signal) and ch. 3 (smoothing error).
- Platnick, S. (2000). Vertical photon transport in cloud remote sensing problems. *J. Geophys. Res.*, 105(D18), 22919–22935. doi:10.1029/2000JD900333
- Twomey, S. (1977). *Introduction to the Mathematics of Inversion in Remote Sensing and Indirect Measurements*. Elsevier.
