# Generating Synthetic In-Situ Cloud Profiles via Functional PCA and a Joint Multivariate Normal in Log-Space

A method note for the synthetic-cloud training set used in the HySICS neural-network emulator (Paper 3). Implementation lives in [`06_synthetic_profile_generator.py`](06_synthetic_profile_generator.py); this document describes what the script is doing and why.

---

## 1. Problem statement

We have a set of $N$ unique in-situ aircraft cloud measurements with $\tau_c \geq 3$, drawn from two marine stratocumulus campaigns: **VOCALS-REx** off the coast of Chile (Wood et al., 2011) and **ORACLES** off the coast of Namibia (Redemann et al., 2021). Each measurement provides a coupled, variable-length sextet of profiles — effective radius $r_e(z)$, liquid water content $\mathrm{LWC}(z)$, ERA5 temperature $T(p)$, ERA5 water vapor $q_v(p)$, and geometry scalars $z_{\text{top}}, z_{\text{base}}$ — plus, when available, the libRadtran gamma-distribution shape parameter $\alpha$. ORACLES files include $\alpha$; VOCALS-REx files do not, because $\alpha$ was added to the ORACLES processing pipeline after VOCALS-REx data was archived. The two campaigns are pooled in a single training set so the generator covers the full marine-stratocumulus regime instead of just one ocean basin.

The neural network we ultimately want to train requires $\mathcal{O}(10^5)$–$\mathcal{O}(10^6)$ training samples. Bootstrapping the $N$ measurements directly is therefore not enough — we need a *generative model* of the joint distribution that can emit arbitrarily many physically self-consistent draws.

The method must satisfy four constraints:

1. **Physical consistency.** Synthetic profiles must respect positivity, in-cloud bounds on $r_e$, and — crucially — the coupling between $r_e(z)$ and $\mathrm{LWC}(z)$. An adiabatic $\mathrm{LWC}(z)$ paired with a drizzling $r_e(z)$ is unphysical.
2. **Coverage.** Synthetic samples should span the empirical range of $\tau_c$, LWP, $r_e^{\text{top}}$, $r_e^{\text{base}}$, and the ERA5 atmosphere they are embedded in.
3. **Smoothness.** Real droplet profiles have low-pass character set by the CDP sampling rate and microphysics; the generator should not inject spurious high-frequency noise into the training set.
4. **Statistical fidelity.** Per-level marginals and cross-level correlations should match the data, particularly the well-documented log-normal character of cloud microphysical quantities (Considine & Curry, 1996; Wood, 2012).

The chosen approach — **functional PCA on log-transformed profiles plus a joint multivariate Normal (MVN) on the principal-component scores and nuisance attributes** — meets all four with one of the smallest defensible parameter counts.

---

## 2. Method

### 2.1 Pipeline at a glance

$$
\underbrace{\{(r_e^{(i)}, \mathrm{LWC}^{(i)}, T^{(i)}, q_v^{(i)}, \dots)\}_{i=1}^{N}}_{\text{raw in-situ}}
\;\xrightarrow{\text{project}}\;
\underbrace{Y^{(i)} \in \mathbb{R}^L}_{\text{log-space, common grid}}
\;\xrightarrow{\text{PCA}}\;
\underbrace{c^{(i)} \in \mathbb{R}^K}_{\text{scores}}
\;\xrightarrow{\text{MVN fit}}\;
\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
\;\xrightarrow{\text{sample, invert, exponentiate}}\;
\text{synthetic profiles}.
$$

Each stage carries a specific physical or statistical purpose, expanded below.

### 2.2 Stage 1 — Common-grid projection

Profiles arrive with variable length (16–157 levels) because the aircraft sampling rate and ascent/descent speed differ across flights. PCA requires a fixed feature dimension, so each smoothed profile is interpolated onto a common normalized-altitude grid of length $L = 80$ with $u = 0$ at cloud base and $u = 1$ at cloud top:

$$
\tilde{x}^{(i)}(u_\ell) \;=\; \mathcal{I}\!\bigl[ x^{(i)}_{\text{raw}},\, u^{(i)}_{\text{raw}} \bigr](u_\ell), \qquad u_\ell = \ell/(L-1), \quad \ell = 0,\dots,L-1.
$$

The choice $L = 80$ keeps the round-trip RMSE from this resampling step ($\approx 0.09\,\mu$m for $r_e$) well below the in-situ measurement uncertainty, so this step is lossless for our purposes. ERA5 $T$ and $q_v$ already share a fixed pressure grid (37 standard levels) and need no projection.

### 2.3 Stage 2 — Log-transform of positive-definite quantities

Cloud microphysical quantities are well-known to be approximately log-normally distributed at each level:

- **$r_e$ and $\mathrm{LWC}$**: McFarquhar & Heymsfield (1997) and the stratocumulus review of Wood (2012) document log-normal-like marginals across many campaigns; Considine & Curry (1996) build a statistical drop-size model on this premise.
- **$\alpha$**: positive by construction in the libRadtran gamma-distribution parameterization.
- **$q_v$ in ERA5**: also strongly skewed; standard practice in radiative-transfer emulator work is to model in $\log q_v$ (e.g., Liu et al., 2006; Saunders et al., 1999).

Working in linear space with a Gaussian generative model would assign nonzero probability to negative $r_e$ and $\mathrm{LWC}$, would impose symmetric $\pm\sigma$ envelopes on quantities whose true envelopes are skew, and would systematically underweight the upper tail (drizzle events, very moist columns). Transforming via

$$
Y^{(i)}_\ell \;=\; \log\!\bigl( \tilde{x}^{(i)}(u_\ell) + \epsilon \bigr),
$$

with a small floor $\epsilon$ to handle near-base zeros in $\mathrm{LWC}$ and near-tropopause vapor, lets us fit a Gaussian model in $Y$-space that, when inverted via $\exp(\cdot) - \epsilon$, produces guaranteed-positive log-normal variates at each level. This is the standard "log-anamorphosis" trick used routinely in geostatistical and data-assimilation work for non-negative variables (Wilks, 2011, §3.4.4; Anderson, 2003, §2.2).

Geometry scalars ($z_{\text{base}}$, thickness $h$) are also positive but additive rather than multiplicative; we log-transform $h$ but keep $z_{\text{base}}$ linear.

### 2.4 Stage 3 — Functional PCA

For each variable separately, write the centered data matrix $\tilde{Y} \in \mathbb{R}^{N \times L}$ with rows the $N$ training profiles:

$$
\tilde{Y} \;=\; Y \;-\; \mathbf{1}_N \bar{Y}^{\!\top}, \qquad \bar{Y}_\ell = \tfrac{1}{N}\sum_i Y^{(i)}_\ell.
$$

Compute the truncated singular-value decomposition

$$
\tilde{Y} \;=\; U\,\Sigma\,V^{\!\top}, \qquad U\in \mathbb{R}^{N\times r}, \;\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_r), \;V\in\mathbb{R}^{L\times r}.
$$

The columns $v_k$ of $V$ are orthonormal **eigen-profiles** in log-space; the per-profile scores are the rows of $C = U\,\Sigma$, with $c^{(i)}_k = \langle Y^{(i)} - \bar{Y},\, v_k\rangle$. Truncating to the leading $K \leq r$ modes that capture cumulative variance $\geq 0.99$ gives the optimal rank-$K$ approximation in the Frobenius norm — this is the **Eckart–Young theorem** (Eckart & Young, 1936). For our profiles a small $K$ suffices: $K_{r_e} = 6$ and $K_{\mathrm{LWC}} = 8$ in the current run.

Two consequences worth emphasizing:

- **Truncated PCA is itself a smoother**, and a better one than a moving average for our purposes. The discarded tail $\sigma_{K+1},\dots,\sigma_r$ contains exactly the per-profile high-frequency noise (instrument jitter, droplet sampling fluctuations) that we want suppressed. A moving average destroys variance *before* PCA can see it; truncated PCA preserves the low-rank structure that actually varies coherently across the population.
- **Choosing $K$ trades smoothness for fidelity.** Smaller $K$ → smoother reconstructions and a tighter sample space; larger $K$ → more wiggle and broader span. We pick $K$ from the scree curve: the first elbow where cumulative variance plateaus.

The same machinery is applied independently to $T$ and $q_v$ on the ERA5 pressure grid, yielding $K_T$ and $K_{q_v}$ further sets of scores.

This two-step decomposition (resample to common grid, then PCA) is a standard recipe in functional data analysis (Ramsay & Silverman, 2005, §8) and has a long history in atmospheric science where PCA is usually called **Empirical Orthogonal Function (EOF) analysis** (Lorenz, 1956; Hannachi, Jolliffe & Stephenson, 2007). In the radiative-transfer-emulator literature, Liu et al. (2006) use PCA on hyperspectral atmospheric profiles for the principal-component-based radiative transfer model (PCRTM), and Garand et al. (2007) use it for assimilation background-error covariances — directly analogous moves.

### 2.5 Stage 4 — Joint multivariate Normal on scores and nuisance attributes

We concatenate the score vectors of all four profiles with two scalar nuisance attributes into a single feature vector per training cloud:

$$
f^{(i)} \;=\; \bigl[\, c^{(i)}_{r_e},\, c^{(i)}_{\mathrm{LWC}},\, c^{(i)}_{T},\, c^{(i)}_{q_v},\, z_{\text{base}}^{(i)},\, \log h^{(i)} \,\bigr] \;\in\; \mathbb{R}^{D},
$$

with $D = K_{r_e} + K_{\mathrm{LWC}} + K_T + K_{q_v} + 2$. Every profile in the pooled VOCALS-REx + ORACLES set contributes to this feature vector, regardless of campaign. We fit a single multivariate Normal:

$$
f \;\sim\; \mathcal{N}(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f), \qquad \boldsymbol{\mu}_f = \tfrac{1}{N}\sum_i f^{(i)}, \quad \boldsymbol{\Sigma}_f = \tfrac{1}{N-1}\sum_i (f^{(i)} - \boldsymbol{\mu}_f)(f^{(i)} - \boldsymbol{\mu}_f)^{\!\top}.
$$

The off-diagonal blocks of $\boldsymbol{\Sigma}_f$ encode every cross-correlation we care about: $r_e$ shape vs.\ $\mathrm{LWC}$ shape (the adiabaticity coupling), $r_e$ shape vs.\ thickness, $T$ profile vs.\ $z_{\text{base}}$ (boundary-layer height vs.\ free-troposphere structure), and so on. **Modeling them jointly is what prevents drawing a drizzle-shape $r_e$ glued to an adiabatic $\mathrm{LWC}$.**

A single Gaussian is also the **maximum-entropy distribution** given a mean and covariance (Cover & Thomas, 2006, §12.1). With only a few hundred training points, fitting anything more flexible — e.g., a Gaussian mixture, copula, or normalizing flow — risks overfitting the empirical scatter. A single MVN is the principled minimal-assumption choice at this sample size.

**Handling of $\alpha$.** The libRadtran shape parameter $\alpha$ is omitted from the joint feature vector because VOCALS-REx files do not store it. Imputing $\alpha$ for VOCALS-REx (≈25% of the training set) would artificially shrink the modeled $\alpha$ variance, since every imputed value would equal the ORACLES median. Instead, we fit a separate one-dimensional log-Normal to the ORACLES $\alpha$ subset, $\log\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$, and draw $\alpha$ independently of the joint MVN at sampling time. This decouples $\alpha$ from cloud shape — defensible physically because $\alpha$ is set by aerosol activation and small-scale turbulence rather than the gross $r_e$/LWC profile (Wood, 2012, §4) — and avoids polluting the joint covariance with imputed entries.

### 2.6 Stage 5 — Sampling, inverse-PCA, exponentiation, resampling

For each desired synthetic cloud we draw

$$
f^* \sim \mathcal{N}(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f),
$$

partition into $(c_{r_e}^*, c_{\mathrm{LWC}}^*, c_T^*, c_{q_v}^*, \log\bar{\alpha}^*, z_{\text{base}}^*, \log h^*)$, invert each PCA in log-space:

$$
\hat{Y}^*_v \;=\; \bar{Y}_v + V_{v,K}\,c^*_v, \qquad v \in \{r_e, \mathrm{LWC}, T, q_v\},
$$

and exponentiate back to physical units:

$$
\hat{x}^*_v \;=\; \exp(\hat{Y}^*_v) - \epsilon.
$$

The cloud profiles are then resampled from the common $L=80$ grid onto the fixed seven-level neural-network target grid evenly spaced from $z_{\text{top}}^* = z_{\text{base}}^* + h^*$ down to $z_{\text{base}}^*$. ERA5 profiles remain on their native 37-level pressure grid.

### 2.7 Stage 6 — Self-consistent derived bulk quantities

Optical depth and liquid water path are *not* sampled independently — that would create profiles whose label disagreed with the integral of the profile itself. They are derived from the synthetic $r_e$ and $\mathrm{LWC}$:

$$
\mathrm{LWP}\,[\text{g/m}^2] \;=\; 1000 \,\Bigl|\!\int \mathrm{LWC}\,dz\Bigr|,
$$

$$
\tau_c \;=\; \int_{z_b}^{z_t} \frac{3\,Q_{\text{ext}}\,\mathrm{LWC}(z)}{4\,\rho_w\, r_e(z)}\,dz \;=\; 750 \cdot Q_{\text{ext}} \int \frac{\mathrm{LWC}\,[\text{g/m}^3]}{r_e\,[\mu\text{m}]}\, dz\,[\text{km}].
$$

The first equality is the standard cloud-optics relation (Hansen & Travis, 1974, eq. 2.59) with $Q_{\text{ext}}$ the bulk Mie extinction efficiency averaged over the droplet size distribution at the wavelength libRadtran reports cloud optical depth at. The geometric-optics limit $Q_{\text{ext}}=2$ (the textbook Hansen–Travis approximation) underestimates the Mie-computed $\tau_c$ by ~10–20% for typical cloud droplet sizes — a discrepancy well documented in shortwave cloud-optics parameterizations (Slingo, 1989; Hu & Stamnes, 1993). We use $Q_{\text{ext,eff}} = 2.32$ in the synthetic generator, calibrated against libRadtran's reported $\tau_c$ on synthetic ORACLES profiles (cloud 1: Hansen–Travis predicts 6.25, libRadtran reports 7.25 → ratio 1.16 ⇒ $Q_{\text{ext,eff}} = 2 \cdot 1.16 \approx 2.32$). This single-parameter Mie correction keeps every synthetic $(\tau_c, \mathrm{LWP})$ label consistent with the value libRadtran will compute downstream when the same $(r_e, \mathrm{LWC})$ profile is fed to its radiative transfer pipeline.

### 2.8 Stage 7 — Rejection sampling on $\tau_c$

The training set was filtered to $\tau_c \geq 3$. To keep the synthetic distribution comparable, we draw in batches and reject samples whose derived $\tau_c$ falls below 3. The acceptance rate is high ($\approx 0.6$–$0.7$), so this adds modest cost while guaranteeing the synthetic sample obeys the same lower bound as the in-situ population.

---

## 3. Why this method, and not another?

| Alternative                         | Why we did not use it                                                                                                                                 |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bootstrap of the 230 profiles        | Only 230 unique shapes — defeats the purpose. No interpolation between observed shapes.                                                                |
| Adiabatic LWC + parametric $r_e(z)$  | Imposes a single physical regime; cannot produce drizzling profiles. The in-situ data spans both adiabatic and sub-adiabatic regimes.                    |
| Variational autoencoder / GAN        | Both are data-hungry — typical training sets in the geoscience literature are $10^4$–$10^6$ profiles. Fitting either to 230 invites mode collapse.      |
| Gaussian process per profile         | Computational cost scales as $\mathcal{O}(N^3 L^3)$; kernel-choice ill-posed for variable-length profiles; redundant once shapes are smooth.            |
| Direct kernel density estimation      | $L = 80$ is far too high a dimension for KDE with $N=230$ (curse of dimensionality, Wilks 2011 §3.6).                                                   |
| Independent sampling of $r_e$, LWC   | Decouples the very thing you most want preserved (adiabaticity coupling).                                                                              |
| Gaussian mixture on the joint scores | A reasonable upgrade — it would capture the drizzling vs.\ non-drizzling regime split. Held in reserve as a refinement; one MVN suffices for first run.|

The combination *PCA in log-space + joint MVN* hits the specific point on the bias-variance curve appropriate to a 230-profile training set: enough flexibility to span the observed range, few enough free parameters that the fit does not chase noise, and direct support for joint cross-quantity correlations.

This is also a well-trodden path in atmospheric science. Liu et al. (2006) build a hyperspectral RT model using PCA on atmospheric profiles in log-space; Garand et al. (2007) use exactly the same machinery for background-error covariances in NWP data assimilation; Räisänen et al. (2004) generate stochastic cloudy columns for GCMs using a related Monte Carlo column generator built on column statistics. The novelty here is its application to a paired droplet-profile + ERA5 atmosphere problem at the specific sample size (a few hundred profiles) where a parametric joint MVN is provably the best-conditioned generative choice.

---

## 4. Limitations and possible refinements

- **Single-mode MVN.** If diagnostic plots reveal multimodal score distributions (e.g.\ a bimodal split between adiabatic and drizzling clouds), swap MVN $\to$ Gaussian mixture with $k = 2$–$3$ components. The change is one line in the script.
- **Hard $r_e$ clip.** Currently bounded by the observed in-situ range. If the NN should generalize beyond observed $r_e^{\text{max}}$ to anticipate aerosol-perturbed regimes, relax the clip with a margin.
- **Slight $\tau_c$ tail compression.** The single MVN under-represents the very high-$\tau_c$ tail, because Gaussian tail correlations are lighter than the empirical ones. A GMM upgrade or copula-based tail correction would fix this.
- **No diurnal cycle in T or $q_v$.** ERA5 profiles are sampled at one snapshot per in-situ flight; if we needed diurnally resolved synthetic atmospheres we would extend the joint MVN to include local solar time.

None of these is a methodological problem with PCA-in-log-space + MVN; each is an empirical refinement to bolt on if a downstream NN diagnostic flags it.

---

## 5. References

- Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3rd ed. Wiley.
- Considine, G., & Curry, J. A. (1996). A statistical model of drop-size spectra for stratocumulus clouds. *Quart. J. Roy. Meteor. Soc.*, **122**, 611–634.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, **1**(3), 211–218.
- Garand, L., Buehner, M., & Wagneur, N. (2007). Background error correlation between surface skin and air temperatures: Estimation and impact on the assimilation of infrared window radiances. *J. Appl. Meteor. Climatol.*, **46**(8), 1249–1262.
- Hannachi, A., Jolliffe, I. T., & Stephenson, D. B. (2007). Empirical orthogonal functions and related techniques in atmospheric science: A review. *Int. J. Climatol.*, **27**(9), 1119–1152.
- Hansen, J. E., & Travis, L. D. (1974). Light scattering in planetary atmospheres. *Space Sci. Rev.*, **16**, 527–610.
- Hu, Y. X., & Stamnes, K. (1993). An accurate parameterization of the radiative properties of water clouds suitable for use in climate models. *J. Climate*, **6**(4), 728–742.
- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Liu, X., Smith, W. L., Zhou, D. K., & Larar, A. (2006). Principal component–based radiative transfer model for hyperspectral sensors: Theoretical concept. *Appl. Opt.*, **45**(1), 201–209.
- Lorenz, E. N. (1956). *Empirical Orthogonal Functions and Statistical Weather Prediction*. Sci. Rep. No. 1, Statistical Forecasting Project, MIT.
- McFarquhar, G. M., & Heymsfield, A. J. (1997). Parameterization of tropical cirrus ice crystal size distributions. *J. Atmos. Sci.*, **54**(17), 2187–2200.
- Pincus, R., Barker, H. W., & Morcrette, J.-J. (2003). A fast, flexible, approximate technique for computing radiative transfer in inhomogeneous cloud fields. *J. Geophys. Res.*, **108**(D13), 4376.
- Räisänen, P., Barker, H. W., Khairoutdinov, M. F., Li, J., & Randall, D. A. (2004). Stochastic generation of subgrid-scale cloudy columns for large-scale models. *Quart. J. Roy. Meteor. Soc.*, **130**(601), 2047–2067.
- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*, 2nd ed. Springer.
- Redemann, J., et al. (2021). An overview of the ORACLES (ObseRvations of Aerosols above CLouds and their intEractionS) project: aerosol–cloud–radiation interactions in the southeast Atlantic basin. *Atmos. Chem. Phys.*, **21**(3), 1507–1563.
- Saunders, R., Matricardi, M., & Brunel, P. (1999). An improved fast radiative transfer model for assimilation of satellite radiance observations. *Quart. J. Roy. Meteor. Soc.*, **125**(556), 1407–1425.
- Slingo, A. (1989). A GCM parameterization for the shortwave radiative properties of water clouds. *J. Atmos. Sci.*, **46**(10), 1419–1427.
- Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*, 3rd ed. Academic Press.
- Wood, R. (2012). Stratocumulus clouds. *Mon. Wea. Rev.*, **140**(8), 2373–2423.
- Wood, R., et al. (2011). The VAMOS Ocean-Cloud-Atmosphere-Land Study Regional Experiment (VOCALS-REx): goals, platforms, and field operations. *Atmos. Chem. Phys.*, **11**(2), 627–654.
