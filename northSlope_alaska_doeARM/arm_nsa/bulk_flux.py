"""Bulk-aerodynamic sensible and latent heat fluxes over snow from NSA C1 data.

Why this module exists
----------------------
No eddy-correlation system has ever operated at NSA C1 (Barrow); the ECOR is
at E10 Oliktok Point (see seb.py). The turbulent terms of the Sledd et al.
(2025) surface energy budget therefore have to be PARAMETERIZED from what is
measured at Barrow:

    T_skin   GNDIRT brightness temperature (corrected) or LWU inversion
    T_air    MET 2-m temperature, or any tower level (2/10/20/40 m)
    RH       MET / tower (w.r.t. liquid water, Vaisala convention)
    U        MET 10-m wind, or any tower level
    p        MET station pressure

Method (Monin-Obukhov similarity theory, MOST)
----------------------------------------------
With all fluxes POSITIVE UPWARD (the Sledd Eq. (1) convention),

    tau = rho u*^2
    SH  = -rho c_p u* theta*
    LH  = -rho L_s u* q*

where the scaling parameters follow from the measured surface-to-level
differences through the integrated flux-profile relations:

    u*     = kappa U                 / [ ln(z_u/z_0) - psi_m(z_u/L) + psi_m(z_0/L) ]
    theta* = kappa (theta_a - theta_s) / [ ln(z_t/z_T) - psi_h(z_t/L) + psi_h(z_T/L) ]
    q*     = kappa (q_a - q_s)       / [ ln(z_q/z_Q) - psi_h(z_q/L) + psi_h(z_Q/L) ]

    L = u*^2 theta_v / (kappa g theta_v*),   theta_v* = theta*(1 + 0.61 q) + 0.61 theta q*

and the system is iterated to convergence in L (Garratt 1992, Sect. 3.3;
Andreas et al. 2010, Sect. 2). theta_s uses the snow-surface temperature; the
surface specific humidity is saturation OVER ICE at T_skin, which is the
standard closure for snow and sea ice (Andreas et al. 2010; Persson et al.
2002, Sect. 3.2).

Stability functions
    unstable (zeta < 0): Paulson (1970) / Businger-Dyer, x = (1 - 16 zeta)^1/4.
    stable   (zeta > 0): default "grachev2007", the SHEBA flux-profile
        relations (Grachev et al. 2007, Boundary-Layer Meteorol. 124, 315-333),
        derived from a full year over Arctic sea ice and the natural choice
        for the very stable winter boundary layer at Barrow:
            phi_m = 1 + a_m zeta (1 + zeta)^1/3 / (1 + b_m zeta),  a_m = 5, b_m = a_m/6.5
            phi_h = 1 + (a_h zeta + b_h zeta^2) / (1 + c_h zeta + zeta^2),
                                                     a_h = 5, b_h = 5, c_h = 3
        Their integrated forms psi = int_0^zeta (1 - phi)/zeta' dzeta' are
        evaluated here by NUMERICAL quadrature on a cached grid rather than by
        transcribing the closed-form expressions, so a mis-remembered
        coefficient cannot silently enter; the phi forms above are the only
        thing to check against the paper (their Eqs. 9 and 10). The
        alternative "beljaars1991" is the Beljaars & Holtslag (1991, J. Appl.
        Meteor. 30, 327-341) modification of Holtslag & de Bruin (1988):
            psi_m = -[a zeta + b (zeta - c/d) exp(-d zeta) + b c/d]
            psi_h = -[(1 + 2 a zeta/3)^1.5 + b (zeta - c/d) exp(-d zeta) + b c/d - 1]
            a = 1, b = 2/3, c = 5, d = 0.35.

Roughness lengths
    z_0 is a FREE PARAMETER of this method, not a measurement. Over winter
    snow it is small: Andreas et al. (2010, J. Hydrometeorol. 11, 87-104)
    report values of order 10^-4 m for SHEBA sea ice, and snow-covered
    surfaces generally fall in 10^-4 .. 10^-3 m. The default here is
    3 x 10^-4 m; treat it as uncertain by a factor of ~3 and test the
    sensitivity (a factor of 3 in z_0 changes the neutral C_D at 10 m by
    ~25%). The scalar roughness lengths z_T and z_Q are NOT set equal to z_0:
    they follow the Andreas (1987, Boundary-Layer Meteorol. 38, 159-184)
    surface-renewal model as a function of the roughness Reynolds number
    R* = u* z_0 / nu, which gives z_T, z_Q << z_0 over aerodynamically rough
    snow. The coefficients are those of Andreas (1987) Table 1 as commonly
    reproduced; verify against the paper before publication.

Sign convention and comparison with ERA5
    SH and LH here are positive UPWARD, matching seb.compute_seb_terms() and
    the ERA5 seb_terms.py (which negates ERA5's downward-positive msshf /
    mslhf). In Arctic winter over snow, SH is usually NEGATIVE (downward: the
    air is warmer than the radiatively cooled surface) and LH is small and of
    either sign (deposition/sublimation).

Limits of the method to keep in mind
    * At very low wind (< ~1 m/s) and strong stability MOST is at the edge of
      its validity; results are returned but flagged (converged = False or
      zeta at the cap).
    * The 10-m wind footprint (MET) and the ~10-m IRT/pyrgeometer footprint do
      not coincide exactly with a 2-m T/RH sensor; sub-metre surface
      heterogeneity (sastrugi, drifts) is not represented.
    * Blowing snow adds a sublimation sink that the bulk LH does not see.

References
    Andreas, E. L. (1987). A theory for the scalar roughness and the scalar
        transfer coefficients over snow and sea ice. Boundary-Layer
        Meteorol., 38, 159-184.
    Andreas, E. L., et al. (2010). Parameterizing turbulent exchange over sea
        ice in winter. J. Hydrometeorol., 11, 87-104.
    Beljaars, A. C. M., & Holtslag, A. A. M. (1991). Flux parameterization
        over land surfaces for atmospheric models. J. Appl. Meteor., 30,
        327-341.
    Garratt, J. R. (1992). The Atmospheric Boundary Layer. Cambridge UP.
    Grachev, A. A., et al. (2007). SHEBA flux-profile relationships in the
        stable atmospheric boundary layer. Boundary-Layer Meteorol., 124,
        315-333.
    Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and
        supercooled water for atmospheric applications. QJRMS, 131, 1539-1565.
    Paulson, C. A. (1970). The mathematical representation of wind speed and
        temperature profiles in the unstable atmospheric surface layer.
        J. Appl. Meteor., 9, 857-861.
    Persson, P. O. G., et al. (2002). Measurements near the Atmospheric
        Surface Flux Group tower at SHEBA: Near-surface conditions and
        surface energy budget. JGR, 107(C10), 8045.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VON_KARMAN = 0.40
GRAVITY_M_S2 = 9.81
R_DRY_J_KG_K = 287.05  # gas constant, dry air
CP_DRY_J_KG_K = 1004.7  # specific heat of dry air at constant pressure
EPSILON_MW = 0.622  # molecular weight ratio water / dry air
# Latent heat of sublimation [J/kg]; varies < 0.5% over 230-273 K (Rogers &
# Yau 1989, Table 2.1), so a constant is adequate here.
L_SUBLIMATION_J_KG = 2.834e6

DEFAULT_Z0_M = 3.0e-4  # aerodynamic roughness of winter snow; see module doc
MIN_WIND_M_S = 0.3  # floor to keep u* finite in calm conditions
ZETA_MAX = 10.0  # cap on z/L in strongly stable conditions
ZETA_MIN = -10.0  # cap on -z/L in strongly unstable conditions


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------


def sat_vapor_pressure_ice_pa(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over ice [Pa], Murphy & Koop (2005) Eq. 7.

    Valid for T > 110 K; accurate to better than 0.1% in the atmospheric range.
    """
    t_k = np.asarray(t_k, dtype=float)
    return np.exp(9.550426 - 5723.265 / t_k + 3.53068 * np.log(t_k) - 0.00728332 * t_k)


def sat_vapor_pressure_liquid_pa(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over (supercooled) liquid water [Pa].

    Murphy & Koop (2005) Eq. 10, valid 123 K < T < 332 K. Needed because the
    Vaisala humidity sensors on the ARM MET and tower report RH with respect
    to LIQUID water at all temperatures.
    """
    t_k = np.asarray(t_k, dtype=float)
    return np.exp(
        54.842763
        - 6763.22 / t_k
        - 4.210 * np.log(t_k)
        + 0.000367 * t_k
        + np.tanh(0.0415 * (t_k - 218.8))
        * (53.878 - 1331.22 / t_k - 9.44523 * np.log(t_k) + 0.014025 * t_k)
    )


def specific_humidity_kg_kg(e_pa: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Specific humidity from vapour pressure and total pressure."""
    return EPSILON_MW * e_pa / (p_pa - (1.0 - EPSILON_MW) * e_pa)


def air_kinematic_viscosity_m2_s(t_k: np.ndarray) -> np.ndarray:
    """Kinematic viscosity of air [m^2/s] as a function of temperature.

    Polynomial in degC used by Andreas' bulk flux algorithms (e.g. Andreas
    et al. 2010, following Andreas 1989); 1.16e-5 m^2/s at -20 degC.
    """
    t_c = np.asarray(t_k, dtype=float) - 273.15
    return 1.326e-5 * (1.0 + 6.542e-3 * t_c + 8.301e-6 * t_c**2 - 4.84e-9 * t_c**3)


# ---------------------------------------------------------------------------
# Stability functions
# ---------------------------------------------------------------------------


def _psi_unstable(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Paulson (1970) / Businger-Dyer psi_m, psi_h for zeta < 0."""
    x = (1.0 - 16.0 * zeta) ** 0.25
    psi_m = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )
    psi_h = 2.0 * np.log((1.0 + x**2) / 2.0)
    return psi_m, psi_h


def _phi_grachev2007(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SHEBA phi_m, phi_h for zeta >= 0 (Grachev et al. 2007, Eqs. 9-10)."""
    a_m, b_m = 5.0, 5.0 / 6.5
    a_h, b_h, c_h = 5.0, 5.0, 3.0
    phi_m = 1.0 + a_m * zeta * (1.0 + zeta) ** (1.0 / 3.0) / (1.0 + b_m * zeta)
    phi_h = 1.0 + (a_h * zeta + b_h * zeta**2) / (1.0 + c_h * zeta + zeta**2)
    return phi_m, phi_h


# psi(zeta) = int_0^zeta [1 - phi(z')] / z' dz' evaluated once on a fine grid
# (log-spaced so the near-neutral region is resolved) and interpolated. The
# integrand is finite at 0: (1 - phi)/zeta -> -a_m and -a_h respectively.
_GRACHEV_GRID = np.concatenate(([0.0], np.logspace(-6, np.log10(ZETA_MAX * 10), 4000)))


def _cumulative_psi(phi_fn) -> Tuple[np.ndarray, np.ndarray]:
    z = _GRACHEV_GRID
    phi_m, phi_h = phi_fn(z)
    integrand_m = np.empty_like(z)
    integrand_h = np.empty_like(z)
    integrand_m[1:] = (1.0 - phi_m[1:]) / z[1:]
    integrand_h[1:] = (1.0 - phi_h[1:]) / z[1:]
    # Analytic limits at zeta = 0 (d phi / d zeta at 0): -a_m and -a_h.
    integrand_m[0] = -5.0
    integrand_h[0] = -5.0
    dz = np.diff(z)
    psi_m = np.concatenate(
        ([0.0], np.cumsum(0.5 * (integrand_m[1:] + integrand_m[:-1]) * dz))
    )
    psi_h = np.concatenate(
        ([0.0], np.cumsum(0.5 * (integrand_h[1:] + integrand_h[:-1]) * dz))
    )
    return psi_m, psi_h


_PSI_M_GRACHEV, _PSI_H_GRACHEV = _cumulative_psi(_phi_grachev2007)


def _psi_stable_grachev2007(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    psi_m = np.interp(zeta, _GRACHEV_GRID, _PSI_M_GRACHEV)
    psi_h = np.interp(zeta, _GRACHEV_GRID, _PSI_H_GRACHEV)
    return psi_m, psi_h


def _psi_stable_beljaars1991(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Beljaars & Holtslag (1991) psi_m, psi_h for zeta >= 0."""
    a, b, c, d = 1.0, 2.0 / 3.0, 5.0, 0.35
    common = b * (zeta - c / d) * np.exp(-d * zeta) + b * c / d
    psi_m = -(a * zeta + common)
    psi_h = -((1.0 + 2.0 * a * zeta / 3.0) ** 1.5 + common - 1.0)
    return psi_m, psi_h


_STABLE_SCHEMES = {
    "grachev2007": _psi_stable_grachev2007,
    "beljaars1991": _psi_stable_beljaars1991,
}


def psi_functions(
    zeta: np.ndarray, stable_scheme: str = "grachev2007"
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrated stability functions (psi_m, psi_h) for any sign of zeta."""
    zeta = np.clip(np.asarray(zeta, dtype=float), ZETA_MIN, ZETA_MAX)
    psi_m = np.zeros_like(zeta)
    psi_h = np.zeros_like(zeta)
    unstable = zeta < 0
    stable = zeta > 0
    if unstable.any():
        pm, ph = _psi_unstable(zeta[unstable])
        psi_m[unstable], psi_h[unstable] = pm, ph
    if stable.any():
        pm, ph = _STABLE_SCHEMES[stable_scheme](zeta[stable])
        psi_m[stable], psi_h[stable] = pm, ph
    return psi_m, psi_h


# ---------------------------------------------------------------------------
# Scalar roughness lengths (Andreas 1987)
# ---------------------------------------------------------------------------

# ln(z_s / z_0) = b0 + b1 ln(R*) + b2 [ln(R*)]^2 in three roughness-Reynolds-
# number regimes. Coefficients as tabulated in Andreas (1987) Table 1 and
# reproduced in Andreas et al. (2010); the smooth-regime constants differ for
# T and Q because Pr = 0.71 and Sc = 0.63. VERIFY before publication.
_ANDREAS_COEFFS = {
    # regime: ((b0_T, b1_T, b2_T), (b0_Q, b1_Q, b2_Q))
    "smooth": ((1.250, 0.0, 0.0), (1.610, 0.0, 0.0)),  # R* <= 0.135
    "transition": ((0.149, -0.550, 0.0), (0.351, -0.628, 0.0)),  # 0.135 < R* < 2.5
    "rough": ((0.317, -0.565, -0.183), (0.396, -0.512, -0.180)),  # R* >= 2.5
}


def scalar_roughness_andreas1987(
    z0_m: np.ndarray, ustar_m_s: np.ndarray, t_air_k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (z_T, z_Q) [m] from the Andreas (1987) surface-renewal model."""
    nu = air_kinematic_viscosity_m2_s(t_air_k)
    re_star = np.maximum(ustar_m_s, 1e-6) * z0_m / nu
    ln_re = np.log(re_star)
    z_t = np.empty_like(re_star)
    z_q = np.empty_like(re_star)
    regime = np.where(re_star <= 0.135, 0, np.where(re_star < 2.5, 1, 2))
    for idx, name in enumerate(("smooth", "transition", "rough")):
        sel = regime == idx
        if not sel.any():
            continue
        (b0t, b1t, b2t), (b0q, b1q, b2q) = _ANDREAS_COEFFS[name]
        lr = ln_re[sel]
        z_t[sel] = z0_m[sel] * np.exp(b0t + b1t * lr + b2t * lr**2)
        z_q[sel] = z0_m[sel] * np.exp(b0q + b1q * lr + b2q * lr**2)
    return z_t, z_q


# ---------------------------------------------------------------------------
# The bulk flux calculation
# ---------------------------------------------------------------------------


def bulk_fluxes(
    t_skin_k: xr.DataArray,
    t_air_k: xr.DataArray,
    rh_air_pct: xr.DataArray,
    wspd_m_s: xr.DataArray,
    p_pa: xr.DataArray,
    z_u_m: float = 10.0,
    z_t_m: float = 2.0,
    z_q_m: Optional[float] = None,
    z0_m: float = DEFAULT_Z0_M,
    stable_scheme: str = "grachev2007",
    scalar_roughness: str = "andreas1987",
    n_iter: int = 30,
    tol: float = 1e-3,
) -> xr.Dataset:
    """Bulk-aerodynamic SH and LH over snow, positive UPWARD.

    Parameters
    ----------
    t_skin_k:
        Surface (skin) temperature [K], e.g. seb.skin_temperature_from_irt()
        or seb.skin_temperature_from_lwup().
    t_air_k, rh_air_pct:
        Air temperature [K] and relative humidity w.r.t. LIQUID water [%] at
        height z_t_m / z_q_m (MET 2 m, or a tower level).
    wspd_m_s:
        Wind speed [m/s] at height z_u_m (MET/tower 10 m by default).
    p_pa:
        Station pressure [Pa] (MET atmos_pressure is in kPa: multiply by 1e3).
    z_u_m, z_t_m, z_q_m:
        Measurement heights [m AGL]; z_q_m defaults to z_t_m.
    z0_m:
        Aerodynamic roughness length [m]; a free parameter, see module doc.
    stable_scheme:
        "grachev2007" (default, SHEBA) or "beljaars1991".
    scalar_roughness:
        "andreas1987" (default) or "equal" (z_T = z_Q = z_0, the crude choice).
    n_iter, tol:
        Iteration cap and relative convergence tolerance on u*.

    Returns
    -------
    Dataset on the input time axis with
        sh_up_W_m2, lh_up_W_m2   turbulent fluxes, positive upward [W m-2]
        tau_N_m2                 momentum flux [N m-2]
        ustar_m_s, tstar_K, qstar_kg_kg   MOST scaling parameters
        obukhov_length_m, zeta   L and z_u/L (zeta clipped to +-10)
        c_d, c_h, c_e            bulk transfer coefficients referred to z_u/z_t
        z_t_m_eff, z_q_m_eff     scalar roughness lengths used [m]
        rho_kg_m3, q_air_kg_kg, q_sfc_kg_kg   diagnostics
        converged                bool, iteration met `tol`
    All inputs are aligned to t_skin_k's time axis by nearest neighbour.
    """
    if z_q_m is None:
        z_q_m = z_t_m
    time = t_skin_k["time"]

    def _al(da: xr.DataArray) -> np.ndarray:
        return np.asarray(
            da.reindex(
                time=time, method="nearest", tolerance=np.timedelta64(90, "s")
            ).values,
            dtype=float,
        )

    ts = np.asarray(t_skin_k.values, dtype=float)
    ta = _al(t_air_k)
    rh = _al(rh_air_pct)
    u = np.maximum(_al(wspd_m_s), MIN_WIND_M_S)
    p = _al(p_pa)

    # --- humidities -------------------------------------------------------
    e_air = rh / 100.0 * sat_vapor_pressure_liquid_pa(ta)
    q_air = specific_humidity_kg_kg(e_air, p)
    q_sfc = specific_humidity_kg_kg(
        sat_vapor_pressure_ice_pa(ts), p
    )  # saturated over ice

    # --- potential temperatures at the measurement height and the surface ---
    # theta - T = (g / c_p) z to first order (dry adiabatic); ~0.02 K at 2 m,
    # ~0.1 K at 10 m -- small but systematic, so it is kept.
    theta_a = ta + GRAVITY_M_S2 / CP_DRY_J_KG_K * z_t_m
    theta_s = ts
    theta_v = theta_a * (1.0 + 0.61 * q_air)
    rho = p / (R_DRY_J_KG_K * ta * (1.0 + 0.61 * q_air))

    z0 = np.full_like(u, z0_m)
    ln_zu_z0 = np.log(z_u_m / z0)

    # --- neutral first guess -----------------------------------------------
    ustar = VON_KARMAN * u / ln_zu_z0
    zeta = np.zeros_like(u)
    converged = np.zeros_like(u, dtype=bool)
    valid = (
        np.isfinite(ts)
        & np.isfinite(ta)
        & np.isfinite(rh)
        & np.isfinite(u)
        & np.isfinite(p)
    )

    for iteration in range(n_iter):
        if scalar_roughness == "andreas1987":
            z_t, z_q = scalar_roughness_andreas1987(z0, ustar, ta)
        else:
            z_t, z_q = z0.copy(), z0.copy()
        psi_m_zu, _ = psi_functions(zeta, stable_scheme)
        psi_m_z0, _ = psi_functions(zeta * z0 / z_u_m, stable_scheme)
        _, psi_h_zt = psi_functions(zeta * z_t_m / z_u_m, stable_scheme)
        _, psi_h_zT = psi_functions(zeta * z_t / z_u_m, stable_scheme)
        _, psi_h_zq = psi_functions(zeta * z_q_m / z_u_m, stable_scheme)
        _, psi_h_zQ = psi_functions(zeta * z_q / z_u_m, stable_scheme)

        denom_m = ln_zu_z0 - psi_m_zu + psi_m_z0
        denom_t = np.log(z_t_m / z_t) - psi_h_zt + psi_h_zT
        denom_q = np.log(z_q_m / z_q) - psi_h_zq + psi_h_zQ
        # Guard against the profile functions collapsing the denominators
        # (extreme stability); keeps u* positive and finite.
        denom_m = np.maximum(denom_m, 0.1 * ln_zu_z0)
        denom_t = np.maximum(denom_t, 0.1 * np.log(z_t_m / z_t))
        denom_q = np.maximum(denom_q, 0.1 * np.log(z_q_m / z_q))

        ustar_new = VON_KARMAN * u / denom_m
        tstar = VON_KARMAN * (theta_a - theta_s) / denom_t
        qstar = VON_KARMAN * (q_air - q_sfc) / denom_q
        tv_star = tstar * (1.0 + 0.61 * q_air) + 0.61 * theta_a * qstar
        with np.errstate(divide="ignore", invalid="ignore"):
            obukhov = ustar_new**2 * theta_v / (VON_KARMAN * GRAVITY_M_S2 * tv_star)
            zeta_new = np.where(
                np.isfinite(obukhov) & (obukhov != 0), z_u_m / obukhov, 0.0
            )
        zeta_new = np.clip(zeta_new, ZETA_MIN, ZETA_MAX)
        # Convergence is judged on BOTH u* and zeta, and never on the first
        # pass: the first iterate is evaluated at zeta = 0 and therefore
        # reproduces the neutral initial guess exactly, so a u*-only test
        # would declare convergence before any stability correction had been
        # applied and silently return neutral fluxes.
        rel_u = np.abs(ustar_new - ustar) / np.maximum(ustar, 1e-6)
        rel_z = np.abs(zeta_new - zeta) / np.maximum(np.abs(zeta_new), 1e-3)
        converged = valid & (rel_u < tol) & (rel_z < 10.0 * tol)
        ustar, zeta = ustar_new, zeta_new
        if iteration > 0 and converged[valid].all():
            break

    with np.errstate(divide="ignore", invalid="ignore"):
        obukhov = np.where(zeta != 0, z_u_m / zeta, np.inf)
    sh = -rho * CP_DRY_J_KG_K * ustar * tstar
    lh = -rho * L_SUBLIMATION_J_KG * ustar * qstar
    tau = rho * ustar**2
    c_d = (ustar / u) ** 2
    # Bulk coefficients defined by SH = rho c_p C_H U (theta_s - theta_a) and
    # LH = rho L_s C_E U (q_s - q_a); with SH = -rho c_p u* theta* this gives
    # C_H = u* theta* / [U (theta_a - theta_s)], positive for any stability.
    with np.errstate(divide="ignore", invalid="ignore"):
        c_h = np.where(
            np.abs(theta_a - theta_s) > 1e-3,
            ustar * tstar / (u * (theta_a - theta_s)),
            np.nan,
        )
        c_e = np.where(
            np.abs(q_air - q_sfc) > 1e-9, ustar * qstar / (u * (q_air - q_sfc)), np.nan
        )

    for arr in (sh, lh, tau, ustar, tstar, qstar, obukhov, zeta, c_d, c_h, c_e):
        arr[~valid] = np.nan

    def _da(values, units, long_name):
        return xr.DataArray(
            values,
            coords={"time": time},
            dims="time",
            attrs={"units": units, "long_name": long_name},
        )

    out = xr.Dataset(
        {
            "sh_up_W_m2": _da(sh, "W m-2", "bulk sensible heat flux, positive upward"),
            "lh_up_W_m2": _da(
                lh, "W m-2", "bulk latent heat flux (sublimation), positive upward"
            ),
            "tau_N_m2": _da(tau, "N m-2", "momentum flux"),
            "ustar_m_s": _da(ustar, "m s-1", "friction velocity"),
            "tstar_K": _da(tstar, "K", "temperature scale theta*"),
            "qstar_kg_kg": _da(qstar, "kg kg-1", "humidity scale q*"),
            "obukhov_length_m": _da(obukhov, "m", "Obukhov length L"),
            "zeta": _da(
                zeta, "1", f"stability parameter z_u/L (clipped to +-{ZETA_MAX:g})"
            ),
            "c_d": _da(c_d, "1", f"drag coefficient at {z_u_m:g} m"),
            "c_h": _da(
                c_h, "1", f"heat transfer coefficient ({z_u_m:g} m wind, {z_t_m:g} m T)"
            ),
            "c_e": _da(
                c_e,
                "1",
                f"moisture transfer coefficient ({z_u_m:g} m wind, {z_q_m:g} m q)",
            ),
            "z_t_m_eff": _da(z_t, "m", "scalar roughness length for temperature"),
            "z_q_m_eff": _da(z_q, "m", "scalar roughness length for humidity"),
            "rho_kg_m3": _da(rho, "kg m-3", "air density"),
            "q_air_kg_kg": _da(q_air, "kg kg-1", "air specific humidity"),
            "q_sfc_kg_kg": _da(
                q_sfc,
                "kg kg-1",
                "surface specific humidity (saturation over ice at T_skin)",
            ),
            "converged": _da(converged, "1", "MOST iteration converged"),
        }
    )
    out.attrs.update(
        {
            "method": "Monin-Obukhov bulk aerodynamic, iterated in L",
            "stable_scheme": stable_scheme,
            "unstable_scheme": "Paulson1970/Businger-Dyer",
            "scalar_roughness": scalar_roughness,
            "z0_m": z0_m,
            "z_u_m": z_u_m,
            "z_t_m": z_t_m,
            "z_q_m": z_q_m,
            "sign_convention": "SH, LH positive upward (Sledd et al. 2025 Eq. 1)",
        }
    )
    return out


def neutral_transfer_coefficients(
    z_u_m: float, z_t_m: float, z0_m: float, z_t_rough_m: Optional[float] = None
) -> Dict[str, float]:
    """Neutral C_D and C_H for quick sensitivity checks on z_0.

    C_DN = kappa^2 / ln(z_u/z_0)^2,  C_HN = kappa^2 / [ln(z_u/z_0) ln(z_t/z_T)].
    """
    if z_t_rough_m is None:
        z_t_rough_m = z0_m
    ln_u = np.log(z_u_m / z0_m)
    ln_t = np.log(z_t_m / z_t_rough_m)
    return {"c_dn": VON_KARMAN**2 / ln_u**2, "c_hn": VON_KARMAN**2 / (ln_u * ln_t)}
