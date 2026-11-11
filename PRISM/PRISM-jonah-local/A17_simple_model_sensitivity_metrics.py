"""
Climate-sensitivity diagnostics for Will Kamp's three-layer EBM (simple_model_kamp.py).

Computes, for any parameter set:
    lambda_eff  effective climate feedback parameter [W/m2/K] (positive = restoring)
    ECS         equilibrium climate sensitivity [K], following Gettelman et al. (2019, GRL)
    TCR         transient climate response [K] (1%/yr CO2, years 61-80)
    dT(2100)    SSP2-4.5 warming [K], compared against your CESM2-WACCM ensemble

The model physics is NOT copied here: everything runs through simple_model_kamp.run_ebm and
simple_model_kamp.net_toa_imbalance, so changes to the model file show up here automatically.

Usage (from PRISM/PRISM-jonah-local):
    python J17_simple_model_sensitivity_metrics.py                          # default parameters
    python J17_simple_model_sensitivity_metrics.py --albedo-b-per-K 0.002   # one parameter changed
    python J17_simple_model_sensitivity_metrics.py --gamma-ul 1.5 --upper-depth-m 80
    python J17_simple_model_sensitivity_metrics.py --sweep albedo_b_per_K 0.001 0.002 0.003 0.00526

Run it in the VS Code Interactive Window (# %% cells) to read the markdown derivation below rendered.
"""

# %% [markdown]
# # Deriving $\lambda_\mathrm{eff}$ for the three-layer EBM
#
# ## 1. The model
#
# With $T_a$, $T_u$, $T_l$ the atmosphere, upper-ocean and lower-ocean temperatures [K], $t$ in years,
# and heat capacities $C$ in W yr m$^{-2}$ K$^{-1}$, `simple_model_kamp.layer_derivs` integrates
#
# $$C_a \frac{dT_a}{dt} = N(T_a, t) - \gamma_{au}\,(T_a - T_u)$$
#
# $$C_u \frac{dT_u}{dt} = \gamma_{au}\,(T_a - T_u) - \gamma_{ul}\,(T_u - T_l)$$
#
# $$C_l \frac{dT_l}{dt} = \gamma_{ul}\,(T_u - T_l)$$
#
# where $N$ is the net top-of-atmosphere flux (called `Q` in the model code; $N$ is the usual symbol in
# the feedback literature, e.g. Gregory et al. 2004):
#
# $$N(T_a, t) = \underbrace{S\,\bigl(1-\alpha_\mathrm{earth}(T_a)\bigr)}_{\text{ASR}}
#              - \underbrace{\varepsilon\bigl(\mathrm{CO_2}(t)\bigr)\,\sigma T_a^4}_{\text{OLR}},
#   \qquad \alpha_\mathrm{earth}(T) = A - B\,T .$$
#
# ## 2. At equilibrium the ocean drops out
#
# Set every time derivative to zero and work up from the bottom equation:
#
# - $dT_l/dt = 0 \Rightarrow \gamma_{ul}(T_u - T_l) = 0 \Rightarrow T_u = T_l$
# - $dT_u/dt = 0 \Rightarrow \gamma_{au}(T_a - T_u) = 0 \Rightarrow T_a = T_u$
# - $dT_a/dt = 0 \Rightarrow N(T_a) = 0$
#
# So the equilibrium temperature is the root of the single equation $N(T) = 0$. The heat capacities and
# $\gamma_{au}$, $\gamma_{ul}$ set how fast the system gets there, never where it ends up. **ECS depends
# only on the radiation code** ($S$, $A$, $B$, $\varepsilon$), and TCR and $\Delta T(2100)$ depend on
# the ocean parameters as well.
#
# ## 3. How CO$_2$ enters: the forcing $F$
#
# `simple_model_kamp.emissivity` is built so that a CO$_2$ change reduces the OLR at $T_0$ by exactly the
# Myhre et al. (1998, GRL 25, 2715) forcing:
#
# $$\varepsilon(C) = \varepsilon_0 - \frac{F(C)}{\sigma T_0^4}, \qquad F(C) = K\,\ln\frac{C}{C_0},
#   \qquad K = 5.35\ \mathrm{W\,m^{-2}}$$
#
# so that
#
# $$\mathrm{OLR} = \varepsilon_0\,\sigma T^4 - F(C)\left(\frac{T}{T_0}\right)^4 .$$
#
# At $T = T_0$, raising CO$_2$ from $C_0$ to $C$ raises $N$ by exactly $F(C)$. A doubling gives
#
# $$F_{2\times} = K \ln 2 = 3.708\ \mathrm{W\,m^{-2}} .$$
#
# Gettelman et al. (2019, §2.1) use $F \approx 3.8$ W m$^{-2}$ (Collins et al. 2006) for CESM2. The
# difference is a forcing-definition choice (stratosphere-adjusted vs effective forcing), not a model
# error.
#
# ## 4. Linearizing $N$ about pre-industrial: the definition of $\lambda_\mathrm{eff}$
#
# Pre-industrial ($T_0$, $C_0$) is an equilibrium by construction ($\varepsilon_0$ is tuned so that
# $N(T_0, C_0) = 0$). Perturb CO$_2$ to $C$ and temperature to $T_0 + \Delta T$, then expand to first order:
#
# $$N(T_0 + \Delta T,\, C) \approx \underbrace{N(T_0, C_0)}_{=\,0}
#   + \underbrace{F(C)}_{\text{forcing}} + \left.\frac{\partial N}{\partial T}\right|_{T_0} \Delta T .$$
#
# Define the effective feedback parameter as the restoring rate (positive means stable):
#
# $$\boxed{\lambda_\mathrm{eff} \equiv -\left.\frac{\partial N}{\partial T}\right|_{T_0,\,C_0}} .$$
#
# Differentiating the two terms of $N$ (using $d\alpha_\mathrm{earth}/dT = -B$):
#
# $$\lambda_\mathrm{eff} = \underbrace{4\,\varepsilon_0\,\sigma T_0^3}_{\text{Planck (LW) damping}}
#   \;-\; \underbrace{S\,B}_{\text{ice-albedo (SW) feedback}}
#   = 3.286 - 1.790 = 1.496\ \mathrm{W\,m^{-2}\,K^{-1}}\quad\text{(default parameters)} .$$
#
# At equilibrium $N = 0$, so $0 = F_{2\times} - \lambda_\mathrm{eff}\,\Delta T$, and therefore
#
# $$\mathrm{ECS} = \frac{F_{2\times}}{\lambda_\mathrm{eff}} .$$
#
# This is the same algebra as the old linear-OLR model ($\mathrm{OLR} = a + bT$, where $b$ *was*
# $\lambda_\mathrm{eff}$). The difference is that $\lambda_\mathrm{eff}$ is now a derivative evaluated at
# a state rather than a constant in the code.
#
# ## 5. Tangent vs secant: why two values of $\lambda_\mathrm{eff}$ are printed
#
# $N$ is not linear in $T$. Writing it out with the forcing term from §3,
#
# $$N(T, C) = S\bigl(1 - A + B\,T\bigr) - \varepsilon_0\sigma T^4 + F(C)\left(\frac{T}{T_0}\right)^4 ,$$
#
# $$-\frac{\partial N}{\partial T} = 4\varepsilon_0\sigma T^3 - S\,B - \frac{4\,F(C)\,T^3}{T_0^4} .$$
#
# The local (tangent) feedback therefore drifts with $T$ and $F$. Two values are reported:
#
# - **tangent**: $-\partial N/\partial T$ at ($T_0$, $C_0$), the boxed definition in §4;
# - **secant**: $F_{2\times}/\mathrm{ECS}$, the single number that maps the doubling forcing exactly onto
#   the true equilibrium warming. This is what Gettelman et al. (2019) quote.
#
# Over a doubling they differ by well under 1% for the default parameters. If they diverge strongly for
# a parameter set, the linear picture is breaking down (for example, the albedo clip is engaging).
#
# ## 6. Mapping onto Gettelman et al. (2019) notation
#
# Their eq. (1) is $\mathrm{ECS} = \Delta T_s = -\lambda F$, with $\lambda$ the *climate sensitivity
# parameter* [K/(W m$^{-2}$)] and $\gamma = 1/\lambda$ the *climate feedback parameter*
# [W m$^{-2}$ K$^{-1}$], so that $\gamma = -F/\Delta T_s$ (§3.2). Both are negative for a stable climate.
# Comparing with §4:
#
# $$\gamma_\mathrm{Gettelman} = -\lambda_\mathrm{eff}, \qquad
#   \lambda_\mathrm{Gettelman} = -\frac{1}{\lambda_\mathrm{eff}} .$$
#
# CESM2 has $\gamma = -0.72$ W m$^{-2}$ K$^{-1}$ (their §3.2, with $F \approx 3.8$, ECS $= 5.3$ K), i.e.
# $\lambda_\mathrm{eff} = 0.72$. **Watch the symbol clash:** the "$\lambda$" in this script and in most
# of the feedback literature (e.g. IPCC AR6 Ch. 7's $\alpha$, with sign flipped) is the paper's $\gamma$,
# not the paper's $\lambda$.
#
# ## 7. How each metric is measured
#
# **ECS:** the slab-ocean (SOM) method of Gettelman et al. (2019, §2.1, §3.1). Their SOM fixes the
# deep-ocean heat flux to a zero global mean. Here that is $\gamma_{ul} = 0$, which leaves the atmosphere
# plus upper-ocean slab. CO$_2$ is doubled instantaneously, the run is integrated to equilibrium, and ECS
# is the difference of the final 20-year means of the doubled and control runs. By §2, the deep ocean
# cannot change this answer. Removing it only shortens the spin-up from millennia to decades. Two
# independent checks are printed:
#
# - the exact root of $N(T, 2C_0) = 0$ (Brent's method), which the SOM run must reproduce;
# - a Gregory regression (Gregory et al. 2004, GRL 31, L03205) of annual $N$ on $\Delta T$ over the
#   first 150 years. Its intercept estimates $F$, its slope estimates $-\lambda_\mathrm{eff}$, and its
#   $x$-intercept estimates ECS. In CESM2 this slope changes over time (the "pattern effect"). Here it
#   cannot, because $N$ depends only on $T_a$.
#
# **TCR:** CO$_2$ rises 1% yr$^{-1}$ from $C_0$, which doubles it at year $\ln 2/\ln 1.01 = 69.7$.
# TCR is the mean warming over years 61–80 of the full three-layer model, ocean parameters as set.
#
# **$\Delta T(2100)$:** the SSP2-4.5 run, reported two ways so that it matches how you read CESM2 output:
# (a) the 2081–2100 mean minus the 1850–1900 mean (IPCC AR6 SPM convention), and (b) the 2091–2100
# decadal mean minus the 1850–1859 decadal mean (the bold decadal lines in your J04 figures).
#
# ## 8. What $\lambda_\mathrm{eff}$ in this model does and does not contain
#
# The two terms in §4 are the only feedbacks in the model. $\varepsilon$ depends on CO$_2$ only, not on
# $T$, so there is **no water-vapour or lapse-rate feedback** (AR6 combined: $+1.30$, very likely
# $1.1$–$1.5$ W m$^{-2}$ K$^{-1}$, Forster et al. 2021, Table 7.10). With the original slope
# $B = 0.00526$ K$^{-1}$, the ice-albedo term $S\,B = 1.79$ W m$^{-2}$ K$^{-1}$ is about 5× AR6's
# surface-albedo feedback ($+0.35$, very likely $0.10$–$0.60$; §7.4.2.3). The two errors largely cancel
# in $\lambda_\mathrm{eff}$, so getting the net number right does not mean the decomposition is right.
#
# For reference, the AR6 assessed feedbacks (Table 7.10, W m$^{-2}$ K$^{-1}$, positive = amplifying) are:
# Planck $-3.22$, water vapour + lapse rate $+1.30$, surface albedo $+0.35$, clouds $+0.42$,
# net $-1.16$ (i.e. $\lambda_\mathrm{eff} = 1.16$).

# %%
import argparse
import contextlib
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

# simple_model_kamp and prism_paths live next to this script; make that importable
# no matter which directory the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simple_model_kamp as skm  # noqa: E402
from prism_paths import SPATIAL_AVERAGES_DIR  # noqa: E402

# %%
# == CESM2 reference values ===================================================
# Values from Gettelman et al. (2019) were checked against the paper (GRL 46, 8329-8337,
# doi:10.1029/2019GL083978). Values marked UNVERIFIED are from memory: check them against the cited
# source before using them in anything formal.
CESM2_REFERENCE = {
    # Table 1 / section 3.1: SOM ECS of CESM2 (CAM6 atmosphere), +/-0.1 K is the 2-sigma SOM uncertainty
    "ECS_K": (5.3, "Gettelman et al. 2019, Table 1 (CESM2 SOM)"),
    # Section 3.2: gamma = -F/dT_s with F ~ 3.8 W/m2 and dT = 5.3 K -> lambda_eff = +0.72
    "lambda_eff_W_m2_K": (0.72, "Gettelman et al. 2019, sec. 3.2 (gamma = -0.72)"),
    # Forcing definition used by Gettelman et al. for the 284.7 -> 569.4 ppm doubling
    "F_2x_W_m2": (3.8, "Gettelman et al. 2019, sec. 2.1 (Collins et al. 2006)"),
    # WACCM6 is the atmosphere of the CESM2-WACCM runs in your J04 figures
    "ECS_WACCM_K": (4.8, "UNVERIFIED: CESM2-WACCM ECS ~4.75-4.8 K (Gettelman et al. 2019b JGR / Meehl et al. 2020)"),
    "TCR_K": (2.0, "UNVERIFIED: CESM2 TCR ~2.0 K (Meehl et al. 2020, Sci. Adv. 6, eaba1981)"),
}

# CESM2-WACCM members with BOTH a historical (1850-2014) and an SSP2-4.5 (2015-2100) TS file locally.
# The CMIP6 SSP2-4.5 members 001-003 continue the historical members of the same number.
CESM2_MEMBERS = ("001", "002", "003")

# == Experiment settings ======================================================
CTRL_YEAR = skm.SPINUP_START_YEAR  # experiments start here, from the exact pre-industrial equilibrium (T0, T0, T0)
PTS_PER_YEAR = 12  # output points per model year; annual means average these
MEAN_WINDOW_YR = 20  # Gettelman et al. (2019) sec. 3.1: difference of 20-year means
GREGORY_YEARS = 150  # length of the Gregory regression window (CMIP abrupt-4xCO2 convention)
TCR_YEARS = (61, 80)  # 20-year window centred on the 1%/yr doubling year (~70)
TCR_RUN_YEARS = 150  # length of a CMIP DECK 1pctCO2 run (Eyring et al. 2016, GMD 9, 1937)

# %%
# == Parameter overrides ======================================================
# Parameters that can be changed from the command line. Keys are also the names used with --sweep.
PARAM_DOC = {
    "albedo_b_per_K": "slope B of alpha_earth = A - B*T [1/K] (default 0.00526)",
    "albedo_a": "intercept A of alpha_earth [-]; if omitted while B changes, A is reset to keep alpha_earth(T0) fixed",
    "alpha0": "pre-industrial albedo alpha_earth(T0) [-] that the line is rotated about when B changes "
    "(sets A = alpha0 + B*T0; default: the model's current alpha_earth(T0))",
    "k_co2_W_m2": "log-CO2 forcing coefficient K [W/m2]; F_2x = K ln 2 (default 5.35)",
    "upper_depth_m": "upper-ocean layer thickness [m] (default 200)",
    "lower_depth_m": "lower-ocean layer thickness [m] (default 3500)",
    "gamma_au_W_m2_K": "atmosphere <-> upper-ocean exchange coefficient [W/m2/K] (default 5.0)",
    "gamma_ul_W_m2_K": "upper <-> lower-ocean exchange coefficient [W/m2/K] (default 0.7)",
}


@dataclass(frozen=True)
class OceanConfig:
    """Arguments that run_ebm takes directly (heat capacities in W.yr/m2/K, couplings in W/m2/K)."""

    C_a: float
    C_u: float
    C_l: float
    gamma_au: float
    gamma_ul: float

    def run_kwargs(self, **changes) -> dict:
        """Keyword arguments for skm.run_ebm, with optional replacements (e.g. gamma_ul=0 for the SOM)."""
        kw = dict(C_a=self.C_a, C_u=self.C_u, C_l=self.C_l, gamma_au=self.gamma_au, gamma_ul=self.gamma_ul)
        kw.update(changes)
        return kw


@contextlib.contextmanager
def model_parameters(**overrides):
    """
    Temporarily change simple_model_kamp's module-level parameters, then restore them.

    The model reads ALBEDO_A, ALBEDO_B, K_CO2, EPS0, ... as module globals at call time, so changing
    them on the module changes the physics run_ebm sees. Derived constants must be recomputed so that
    pre-industrial (T0, CO2_0) remains an exact equilibrium:
        alpha0 = alpha_earth(T0),  OLR0 = S (1 - alpha0),  EPS0 = OLR0 / (sigma T0^4).
    Without this, changing B would shift the control climate away from T0, and every "warming" would
    include a spurious drift of the control.

    Yields an OceanConfig with the heat capacities and couplings to pass to run_ebm explicitly.
    run_ebm's defaults were bound when the module was imported, so they would not see overrides.
    """
    unknown = set(overrides) - set(PARAM_DOC)
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)}; choose from {sorted(PARAM_DOC)}")

    names = ("ALBEDO_A", "ALBEDO_B", "K_CO2", "alpha0", "OLR0", "EPS0", "C_UPPER", "C_LOWER", "GAMMA_AU", "GAMMA_UL")
    saved = {name: getattr(skm, name) for name in names}
    try:
        # -- shortwave: albedo line ------------------------------------------------------------------
        if "albedo_a" in overrides and "alpha0" in overrides:
            raise ValueError("give albedo_a OR alpha0, not both: alpha0 fixes A through A = alpha0 + B*T0")
        if "albedo_b_per_K" in overrides:
            skm.ALBEDO_B = float(overrides["albedo_b_per_K"])
        if "albedo_a" in overrides:
            skm.ALBEDO_A = float(overrides["albedo_a"])
        elif "albedo_b_per_K" in overrides or "alpha0" in overrides:
            # Rotate the albedo line about T0 so that only the FEEDBACK (the slope) changes, not the
            # base-state energy budget: alpha_earth(T0) = alpha0_target exactly.
            alpha0_target = float(overrides.get("alpha0", saved["alpha0"]))
            skm.ALBEDO_A = alpha0_target + skm.ALBEDO_B * skm.T0

        # -- longwave: CO2 forcing strength ----------------------------------------------------------
        if "k_co2_W_m2" in overrides:
            skm.K_CO2 = float(overrides["k_co2_W_m2"])

        # -- re-establish (T0, CO2_0) as the exact equilibrium (see docstring) -----------------------
        skm.alpha0 = float(skm.alpha_earth(skm.T0))
        skm.OLR0 = skm.S * (1 - skm.alpha0)
        skm.EPS0 = skm.OLR0 / (skm.SIGMA * skm.T0**4)

        # -- ocean layers ----------------------------------------------------------------------------
        if "upper_depth_m" in overrides:
            skm.C_UPPER = skm.ocean_layer_C(float(overrides["upper_depth_m"]))
        if "lower_depth_m" in overrides:
            skm.C_LOWER = skm.ocean_layer_C(float(overrides["lower_depth_m"]))
        skm.GAMMA_AU = float(overrides.get("gamma_au_W_m2_K", skm.GAMMA_AU))
        skm.GAMMA_UL = float(overrides.get("gamma_ul_W_m2_K", skm.GAMMA_UL))

        # initial_state() is lru-cached on its arguments only, not on the module globals changed above
        skm.initial_state.cache_clear()
        yield OceanConfig(skm.C_ATMOS, skm.C_UPPER, skm.C_LOWER, skm.GAMMA_AU, skm.GAMMA_UL)
    finally:
        for name, value in saved.items():
            setattr(skm, name, value)
        skm.initial_state.cache_clear()


@contextlib.contextmanager
def co2_experiment(name: str, years: np.ndarray, ppm: np.ndarray):
    """
    Temporarily register an idealized CO2 pathway (abrupt-2x, 1%/yr) as a model scenario.

    skm.co2_ppm(t, scenario) interpolates _CO2_YEARS[scenario] / _CO2_PPM[scenario], so adding an entry
    there lets the UNMODIFIED run_ebm and net_toa_imbalance run the experiment. np.interp holds the end
    values outside the given years.
    """
    skm._CO2_YEARS[name] = np.asarray(years, dtype=float)
    skm._CO2_PPM[name] = np.asarray(ppm, dtype=float)
    try:
        yield name
    finally:
        del skm._CO2_YEARS[name], skm._CO2_PPM[name]


# %%
# == Small helpers ============================================================
def annual_means(t_yr: np.ndarray, x: np.ndarray, start_year: int, n_years: int) -> np.ndarray:
    """
    Annual means of x sampled on the uniform grid that run_ebm returns for n_pts = PTS_PER_YEAR*n_years + 1.
    The last point (t = start_year + n_years) starts the next year and is dropped.  Shape: (n_years,).
    """
    assert np.isclose(t_yr[0], start_year) and len(t_yr) == PTS_PER_YEAR * n_years + 1
    return x[: PTS_PER_YEAR * n_years].reshape(n_years, PTS_PER_YEAR).mean(axis=1)


def run_annual(scenario: str, n_years: int, cfg: OceanConfig, **run_changes) -> tuple[np.ndarray, np.ndarray]:
    """Run run_ebm from CTRL_YEAR for n_years; return annual-mean T_a [K] and N [W/m2], each shape (n_years,)."""
    t_yr, ta_K, _, _, n_W_m2 = skm.run_ebm(
        scenario=scenario,
        t_span=(CTRL_YEAR, CTRL_YEAR + n_years),
        n_pts=PTS_PER_YEAR * n_years + 1,
        **cfg.run_kwargs(**run_changes),
    )
    return annual_means(t_yr, ta_K, CTRL_YEAR, n_years), annual_means(t_yr, n_W_m2, CTRL_YEAR, n_years)


# %%
# == lambda_eff: tangent decomposition at (T0, CO2_0) =========================
def feedback_decomposition() -> dict:
    """
    Local (tangent) feedback terms at pre-industrial, from the markdown derivation, section 4:
        Planck  =  d(OLR)/dT = 4 eps0 sigma T0^3                       [W/m2/K]
        SW      =  d(ASR)/dT = -S d(alpha_earth)/dT  (= S*B unclipped)  [W/m2/K]
        lambda_eff_tangent = Planck - SW = -dN/dT
    SW is a central finite difference of the model's own alpha_net, so a clip at ALBEDO_MIN/MAX is
    handled automatically. The full lambda is also differenced directly from skm.net_toa_imbalance,
    as a check that the two-term decomposition is complete.
    """
    dT_K = 1e-3  # finite-difference half-width [K]; N is smooth here, so truncation error is ~1e-9
    T0 = skm.T0
    planck = 4 * skm.EPS0 * skm.SIGMA * T0**3
    asr = lambda T: skm.S * (1 - skm.alpha_net(T, 0.0))  # noqa: E731
    sw = (asr(T0 + dT_K) - asr(T0 - dT_K)) / (2 * dT_K)
    # scenario 'none' holds CO2 at CO2_0, i.e. emissivity = EPS0
    n_of_T = lambda T: skm.net_toa_imbalance(CTRL_YEAR, T, scenario="none")  # noqa: E731
    lam_direct = -(n_of_T(T0 + dT_K) - n_of_T(T0 - dT_K)) / (2 * dT_K)
    return {"planck_W_m2_K": planck, "sw_albedo_W_m2_K": sw, "lambda_tangent_W_m2_K": lam_direct}


# %%
# == ECS: slab-ocean abrupt-2xCO2 (Gettelman et al. 2019, sec. 2.1) ===========
def equilibrium_root_K(scenario: str) -> float:
    """
    Exact equilibrium T [K] of the model for a fixed-CO2 scenario: the root of N(T) = 0 (markdown sec. 2).
    The bracket is found by stepping upward from T0, where N > 0 for any positive forcing.
    """
    n_of_T = lambda T: skm.net_toa_imbalance(CTRL_YEAR + 1, T, scenario=scenario)  # noqa: E731
    lo_K = skm.T0
    for hi_K in skm.T0 + np.arange(0.5, 150.0, 0.5):
        if n_of_T(hi_K) < 0:
            return brentq(n_of_T, lo_K, hi_K, xtol=1e-10)
        lo_K = hi_K
    raise RuntimeError("no equilibrium within T0 + 150 K: the model has no stable warm state for this forcing")


def ecs_slab_ocean(cfg: OceanConfig, lam_tangent_W_m2_K: float) -> dict:
    """
    Slab-ocean ECS following Gettelman et al. (2019, secs. 2.1 and 3.1):
      * SOM analog: gamma_ul = 0, so no net heat reaches the deep ocean (their prescribed Q-flux has a
        zero global mean). The slab is atmosphere + upper-ocean layer.
      * instantaneous doubling, CO2_0 -> 2 CO2_0, held fixed; a control run at CO2_0;
      * ECS = (last-20-yr mean T_a, doubled) - (last-20-yr mean T_a, control).

    Run length: the slab relaxes with e-folding time tau = (C_a + C_u) / lambda_eff [yr]. Running for
    25 tau leaves a residual of exp(-25) ~ 1e-11 of the response, with a 300-yr floor and a 10,000-yr cap.
    The final-window N is returned as a convergence check (it should be ~0).
    """
    tau_yr = (cfg.C_a + cfg.C_u) / lam_tangent_W_m2_K
    n_years = int(np.clip(25 * tau_yr, 300, 10_000))

    co2_2x_ppm = 2 * skm.CO2_0
    with co2_experiment("abrupt2x", [CTRL_YEAR, CTRL_YEAR + n_years], [co2_2x_ppm, co2_2x_ppm]) as scen:
        ta_2x_K, n_2x_W_m2 = run_annual(scen, n_years, cfg, gamma_ul=0.0)
        ta_ctrl_K, _ = run_annual("none", n_years, cfg, gamma_ul=0.0)  # 'none' = CO2 held at CO2_0
        T_root_K = equilibrium_root_K(scen)

    window = slice(-MEAN_WINDOW_YR, None)
    ecs_K = ta_2x_K[window].mean() - ta_ctrl_K[window].mean()

    # Gregory et al. (2004) regression over the first GREGORY_YEARS years: N = F + slope * dT,
    # where slope = -lambda_eff. ECS estimate = -F / slope (where the fitted line crosses N = 0).
    n_fit = min(GREGORY_YEARS, n_years)
    dT_K = ta_2x_K[:n_fit] - ta_ctrl_K[:n_fit]
    slope_W_m2_K, intercept_W_m2 = np.polyfit(dT_K, n_2x_W_m2[:n_fit], 1)

    return {
        "ECS_K": ecs_K,
        "ECS_root_K": T_root_K - skm.T0,
        "N_final_W_m2": n_2x_W_m2[window].mean(),
        "som_years": n_years,
        "tau_slab_yr": tau_yr,
        "gregory_F_W_m2": intercept_W_m2,
        "gregory_lambda_W_m2_K": -slope_W_m2_K,
        "gregory_ECS_K": -intercept_W_m2 / slope_W_m2_K,
    }


# %%
# == TCR: 1%/yr CO2, full three-layer model ===================================
def tcr_one_percent(cfg: OceanConfig) -> float:
    """
    TCR [K]: mean T_a warming over years 61-80 of a 1%/yr CO2 increase from CO2_0, which doubles CO2 at
    year ln2/ln1.01 = 69.7. Uses the full model, including the deep ocean, with the parameters as set.
    The control is the exact pre-industrial equilibrium T0, so no separate control run is needed.
    """
    years = CTRL_YEAR + np.arange(TCR_RUN_YEARS + 1)
    ppm = skm.CO2_0 * 1.01 ** (years - CTRL_YEAR)
    with co2_experiment("onepct", years, ppm) as scen:
        ta_K, _ = run_annual(scen, TCR_RUN_YEARS, cfg)
    first, last = TCR_YEARS
    return ta_K[first - 1 : last].mean() - skm.T0  # index 0 is year 1


# %%
# == dT(2100): SSP2-4.5 =======================================================
def warming_metrics(year: np.ndarray, ts_K: np.ndarray) -> dict:
    """
    Two warming measures from an annual-mean temperature series (same function for the model and CESM2):
      dT_AR6     mean(2081-2100) - mean(1850-1900)  (IPCC AR6 SPM baseline and long-term period)
      dT_decadal mean(2091-2100) - mean(1850-1859)  (decadal endpoints, as in the J04 figure's bold lines)
    """
    mean_over = lambda y0, y1: ts_K[(year >= y0) & (year <= y1)].mean()  # noqa: E731
    return {
        "dT_AR6_K": mean_over(2081, 2100) - mean_over(1850, 1900),
        "dT_decadal_K": mean_over(2091, 2100) - mean_over(1850, 1859),
    }


def model_ssp245_warming(cfg: OceanConfig) -> dict:
    """SSP2-4.5 warming of the model, 1850-2100 inclusive (251 annual means). Beyond 2100, CO2 is held at its 2100 value."""
    n_years = 2100 - CTRL_YEAR + 1
    ta_K, _ = run_annual("ssp245", n_years, cfg)
    return warming_metrics(CTRL_YEAR + np.arange(n_years), ta_K)


def cesm2_ssp245_warming() -> dict | None:
    """
    The same warming measures from your CESM2-WACCM output: J02 global means ('G', gaussian-weight
    area-weighted), historical 1850-2014 + SSP2-4.5 2015-2100, members CESM2_MEMBERS.

    CESM h0 time stamps mark the END of each monthly average, so the series is split into calendar years
    by month count, not by the time coordinate. Returns None if the files are not on this machine.
    """
    import xarray as xr  # only needed for the CESM2 comparison

    root = Path(SPATIAL_AVERAGES_DIR)
    per_member = []
    for mm in CESM2_MEMBERS:
        hist = sorted(
            glob.glob(str(root / f"CESM2_WACCM_HIST/**/*historical-WACCM.{mm}.cam.h0.TS.*.nc"), recursive=True)
        )
        ssp = sorted(
            glob.glob(str(root / f"CESM2_WACCM_SSP2-4.5/**/*SSP2-4.5-WACCM.{mm}.cam.h0.TS.*.nc"), recursive=True)
        )
        if not hist or not ssp:
            return None
        ts_K = np.concatenate([xr.open_dataset(f)["TS"].sel(spatial="G").values for f in hist + ssp])
        n_years = 2100 - 1850 + 1
        if ts_K.size != 12 * n_years:
            raise ValueError(f"member {mm}: expected {12 * n_years} months for 1850-2100, found {ts_K.size}")
        per_member.append(ts_K.reshape(n_years, 12).mean(axis=1))  # annual means, shape (251,)

    year = np.arange(1850, 2101)
    members = [warming_metrics(year, ts) for ts in per_member]
    ens = warming_metrics(year, np.mean(per_member, axis=0))
    for key in list(ens):
        values = [m[key] for m in members]
        ens[f"{key}_range"] = (min(values), max(values))
    return ens


# %%
# == All metrics for one parameter set ========================================
def compute_metrics(**overrides) -> dict:
    """lambda_eff (tangent and secant), ECS (+ checks), TCR and SSP2-4.5 warming for one parameter set."""
    with model_parameters(**overrides) as cfg:
        # If the unclipped line A - B*T0 lies outside [ALBEDO_MIN, ALBEDO_MAX], the model sits on a flat,
        # clipped segment at T0: the base albedo is a bound (e.g. snowball 0.65) and the SW feedback is 0.
        alpha_line_T0 = skm.ALBEDO_A - skm.ALBEDO_B * skm.T0
        clipped = not (skm.ALBEDO_MIN <= alpha_line_T0 <= skm.ALBEDO_MAX)
        fb = feedback_decomposition()
        fb["albedo_clipped_at_T0"] = clipped
        fb["alpha_line_T0"] = alpha_line_T0
        if fb["lambda_tangent_W_m2_K"] <= 0:
            # Net positive feedback at T0: the pre-industrial state is unstable (runaway), so no ECS exists
            return {"cfg": cfg, **fb, "unstable": True}
        f_2x_W_m2 = skm.K_CO2 * np.log(2.0)  # markdown sec. 3
        ecs = ecs_slab_ocean(cfg, fb["lambda_tangent_W_m2_K"])
        tcr_K = tcr_one_percent(cfg)
        ssp = model_ssp245_warming(cfg)
        albedo = (skm.ALBEDO_A, skm.ALBEDO_B, skm.alpha0)
    return {
        "cfg": cfg,
        "unstable": False,
        "albedo": albedo,
        "F_2x_W_m2": f_2x_W_m2,
        **fb,
        "lambda_secant_W_m2_K": f_2x_W_m2 / ecs["ECS_K"],  # markdown sec. 5
        **ecs,
        "TCR_K": tcr_K,
        **ssp,
    }


# %%
# == Output ===================================================================
def print_report(m: dict, cesm2: dict | None, overrides: dict) -> None:
    """Full report for one parameter set, with CESM2 references side by side."""
    cfg = m["cfg"]
    print("=" * 96)
    print(
        "Parameters"
        + (
            "  (overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()) + ")"
            if overrides
            else "  (model defaults)"
        )
    )
    print(f"  C_a = {cfg.C_a:.3f}   C_u = {cfg.C_u:.2f}   C_l = {cfg.C_l:.1f}  [W yr/m2/K]")
    print(f"  gamma_au = {cfg.gamma_au:.3f}   gamma_ul = {cfg.gamma_ul:.3f}  [W/m2/K]")
    if m["unstable"]:
        print(f"\n  lambda_eff (tangent) = {m['lambda_tangent_W_m2_K']:.3f} W/m2/K <= 0: runaway; ECS undefined.")
        return
    A, B, a0 = m["albedo"]
    print(f"  alpha_earth = {A:.4f} - {B:.5f} T   (alpha0 = {a0:.4f})")
    if m["albedo_clipped_at_T0"]:
        print(
            f"  WARNING: A - B*T0 = {m['alpha_line_T0']:.3f} is outside [{skm.ALBEDO_MIN}, {skm.ALBEDO_MAX}], so "
            f"alpha_earth(T0) is CLIPPED to {a0:.2f}\n           and the SW feedback is 0. Set A = alpha0 + B*T0 "
            f"(e.g. --alpha0 0.3043) to rotate the line about T0."
        )

    print("\nFeedback decomposition at (T0, CO2_0)   [W/m2/K, positive = restoring]")
    print(f"  Planck (LW)  4 eps0 sigma T0^3 = {m['planck_W_m2_K']:+.3f}")
    print(f"  ice-albedo (SW)  dASR/dT       = {-m['sw_albedo_W_m2_K']:+.3f}   (AR6 surface albedo: -0.35)")
    print(f"  lambda_eff, tangent            = {m['lambda_tangent_W_m2_K']:+.3f}")

    ref = CESM2_REFERENCE
    w = 34
    print(f"\n{'metric':{w}s} {'model':>8s} {'CESM2':>8s}   source of CESM2 value")
    print("-" * 96)
    print(f"{'F_2x [W/m2]':{w}s} {m['F_2x_W_m2']:8.3f} {ref['F_2x_W_m2'][0]:8.2f}   {ref['F_2x_W_m2'][1]}")
    print(
        f"{'lambda_eff = F_2x/ECS [W/m2/K]':{w}s} {m['lambda_secant_W_m2_K']:8.3f} {ref['lambda_eff_W_m2_K'][0]:8.2f}   {ref['lambda_eff_W_m2_K'][1]}"
    )
    print(
        f"{'  = -gamma (Gettelman eq. 1)':{w}s} {-m['lambda_secant_W_m2_K']:8.3f} {-ref['lambda_eff_W_m2_K'][0]:8.2f}"
    )
    print(f"{'ECS (slab ocean) [K]':{w}s} {m['ECS_K']:8.2f} {ref['ECS_K'][0]:8.2f}   {ref['ECS_K'][1]}")
    print(f"{'':{w}s} {'':8s} {ref['ECS_WACCM_K'][0]:8.2f}   {ref['ECS_WACCM_K'][1]}")
    print(f"{'TCR (1%/yr, yrs 61-80) [K]':{w}s} {m['TCR_K']:8.2f} {ref['TCR_K'][0]:8.2f}   {ref['TCR_K'][1]}")
    print(f"{'TCR / ECS [-]':{w}s} {m['TCR_K'] / m['ECS_K']:8.2f} {ref['TCR_K'][0] / ref['ECS_K'][0]:8.2f}")
    for key, label in [
        ("dT_AR6_K", "dT SSP2-4.5, 2081-2100 vs 1850-1900"),
        ("dT_decadal_K", "dT SSP2-4.5, 2091-2100 vs 1850s"),
    ]:
        if cesm2 is None:
            print(
                f"{label + ' [K]':{w}s} {m[key]:8.2f} {'n/a':>8s}   CESM2-WACCM files not found under {SPATIAL_AVERAGES_DIR}"
            )
        else:
            lo, hi = cesm2[f"{key}_range"]
            print(
                f"{label + ' [K]':{w}s} {m[key]:8.2f} {cesm2[key]:8.2f}   CESM2-WACCM ens. mean {'/'.join(CESM2_MEMBERS)} (members {lo:.2f}-{hi:.2f})"
            )

    print("\nChecks")
    print(f"  ECS from SOM run vs exact root of N(T, 2xCO2) = 0 : {m['ECS_K']:.4f} vs {m['ECS_root_K']:.4f} K")
    print(
        f"  SOM run: {m['som_years']} yr (slab tau = {m['tau_slab_yr']:.1f} yr), final-20-yr N = {m['N_final_W_m2']:.1e} W/m2"
    )
    print(
        f"  lambda_eff tangent vs secant                    : {m['lambda_tangent_W_m2_K']:.3f} vs {m['lambda_secant_W_m2_K']:.3f} W/m2/K"
    )
    print(
        f"  Gregory regression (first {GREGORY_YEARS} yr): F = {m['gregory_F_W_m2']:.3f} W/m2, "
        f"lambda = {m['gregory_lambda_W_m2_K']:.3f} W/m2/K, ECS = {m['gregory_ECS_K']:.2f} K"
    )


def print_sweep(name: str, values: list[float], fixed: dict, cesm2: dict | None) -> None:
    """One row per value of the swept parameter; all other parameters as in `fixed`."""
    print("=" * 96)
    print(f"Sweep of {name}" + (f"  (fixed: {fixed})" if fixed else ""))
    print(f"{name:>16s} {'lam_tan':>8s} {'lam_sec':>8s} {'ECS':>6s} {'TCR':>6s} {'dT_AR6':>7s} {'dT_dec':>7s}")
    print(f"{'':>16s} {'W/m2/K':>8s} {'W/m2/K':>8s} {'K':>6s} {'K':>6s} {'K':>7s} {'K':>7s}")
    print("-" * 64)
    for v in values:
        m = compute_metrics(**fixed, **{name: v})
        if m["unstable"]:
            print(f"{v:16.6g} {m['lambda_tangent_W_m2_K']:8.3f}   runaway (lambda <= 0)")
            continue
        print(
            f"{v:16.6g} {m['lambda_tangent_W_m2_K']:8.3f} {m['lambda_secant_W_m2_K']:8.3f} {m['ECS_K']:6.2f} "
            f"{m['TCR_K']:6.2f} {m['dT_AR6_K']:7.2f} {m['dT_decadal_K']:7.2f}"
            + (f"   WARNING: alpha_earth(T0) clipped to {m['albedo'][2]:.2f}" if m["albedo_clipped_at_T0"] else "")
        )
    print("-" * 64)
    ref = CESM2_REFERENCE
    dT_ar6 = f"{cesm2['dT_AR6_K']:7.2f}" if cesm2 else f"{'n/a':>7s}"
    dT_dec = f"{cesm2['dT_decadal_K']:7.2f}" if cesm2 else f"{'n/a':>7s}"
    print(
        f"{'CESM2 ref.':>16s} {'':>8s} {ref['lambda_eff_W_m2_K'][0]:8.2f} {ref['ECS_K'][0]:6.2f} {ref['TCR_K'][0]:6.2f} {dT_ar6} {dT_dec}"
    )
    print("  (CESM2 TCR is UNVERIFIED; dT values are the CESM2-WACCM ensemble mean of your local files)")


# %%
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, doc in PARAM_DOC.items():
        p.add_argument("--" + key.replace("_", "-"), dest=key, type=float, default=None, help=doc)
    p.add_argument(
        "--sweep",
        nargs="+",
        metavar=("NAME", "VALUE"),
        help=f"sweep one parameter: NAME V1 V2 ... ; NAME in {list(PARAM_DOC)}",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: getattr(args, k) for k in PARAM_DOC if getattr(args, k) is not None}
    cesm2_targets = cesm2_ssp245_warming()

    if args.sweep:
        sweep_name, *sweep_values = args.sweep
        if sweep_name not in PARAM_DOC or not sweep_values:
            sys.exit(f"--sweep needs a parameter name from {list(PARAM_DOC)} followed by values")
        print_sweep(sweep_name, [float(v) for v in sweep_values], overrides, cesm2_targets)
    else:
        print_report(compute_metrics(**overrides), cesm2_targets, overrides)
