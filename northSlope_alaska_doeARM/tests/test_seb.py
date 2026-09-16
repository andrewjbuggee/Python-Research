"""Tests for the surface-energy-budget modules (seb, bulk_flux, ground_flux, cloud_water).

Everything here runs on synthetic arrays; no ARM files or credentials needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import bulk_flux as bf  # noqa: E402
from arm_nsa import cloud_water as cw  # noqa: E402
from arm_nsa import config, ground_flux as gf, seb  # noqa: E402


def _times(n: int, step: str = "1h") -> xr.DataArray:
    t = np.datetime64("2026-01-01T00:00:00") + np.arange(n) * np.timedelta64(
        1, step[-1]
    ) * int(step[:-1])
    return xr.DataArray(t, dims="time")


def _da(values, t) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float), coords={"time": t}, dims="time"
    )


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


def test_seb_tiers_reference_registered_keys_and_nest():
    for name, keys in seb.SEB_VARIABLE_SETS.items():
        for key in keys:
            assert key in config.DATASTREAMS, f"{name}: {key} not in config.DATASTREAMS"
            assert (
                key in seb.SEB_STREAMS
            ), f"{name}: {key} has no SebStream sizing entry"
    core, rec, ext = (
        set(seb.SEB_VARIABLE_SETS[k]) for k in ("core", "recommended", "extended")
    )
    assert core < rec < ext
    # The very large products must never be in a tier.
    assert "microbase" not in ext and "kazr" not in ext


def test_e10_streams_are_labelled_as_oliktok():
    for key in ("ecor_e10", "sebs_e10"):
        assert seb.SEB_STREAMS[key].site == "E10"
        assert all("E10" in ds for ds in config.get_spec(key).datastreams)
    for key, info in seb.SEB_STREAMS.items():
        if info.site == "C1":
            assert all("C1" in ds for ds in config.get_spec(key).datastreams), key


def test_estimate_size_scales_with_days():
    one = seb.estimate_size_gb(["qcrad", "thermocldphase"], 1)
    ten = seb.estimate_size_gb(["qcrad", "thermocldphase"], 10)
    assert ten["qcrad"] == pytest.approx(10 * one["qcrad"])
    assert np.isnan(seb.estimate_size_gb(["kazr"], 1)["kazr"])


# ---------------------------------------------------------------------------
# Thermodynamics and stability functions
# ---------------------------------------------------------------------------


def test_saturation_vapour_pressure_at_triple_point_and_ordering():
    # Both formulations meet at 611.7 Pa at 273.16 K (Murphy & Koop 2005).
    assert bf.sat_vapor_pressure_ice_pa(273.16) == pytest.approx(611.7, rel=2e-3)
    assert bf.sat_vapor_pressure_liquid_pa(273.16) == pytest.approx(611.7, rel=2e-3)
    t = np.array([233.15, 253.15, 263.15])
    assert np.all(bf.sat_vapor_pressure_ice_pa(t) < bf.sat_vapor_pressure_liquid_pa(t))


def test_psi_functions_neutral_zero_and_signs():
    for scheme in ("grachev2007", "beljaars1991"):
        pm, ph = bf.psi_functions(np.array([0.0]), scheme)
        assert pm[0] == 0.0 and ph[0] == 0.0
        pm, ph = bf.psi_functions(np.array([-1.0, 1.0]), scheme)
        assert pm[0] > 0 and ph[0] > 0  # unstable: positive
        assert pm[1] < 0 and ph[1] < 0  # stable: negative
    # Stable branch must be monotonically decreasing.
    z = np.linspace(0, 10, 200)
    pm, ph = bf.psi_functions(z, "grachev2007")
    assert np.all(np.diff(pm) <= 0) and np.all(np.diff(ph) <= 0)


def test_grachev_psi_matches_direct_quadrature():
    """The cached psi grid must reproduce a fresh, fine quadrature of (1-phi)/zeta."""
    zeta = 2.0
    z = np.linspace(1e-9, zeta, 200_001)
    phi_m, phi_h = bf._phi_grachev2007(z)
    psi_m_ref = np.trapezoid((1 - phi_m) / z, z)
    psi_h_ref = np.trapezoid((1 - phi_h) / z, z)
    pm, ph = bf.psi_functions(np.array([zeta]), "grachev2007")
    assert pm[0] == pytest.approx(psi_m_ref, rel=1e-3)
    assert ph[0] == pytest.approx(psi_h_ref, rel=1e-3)


# ---------------------------------------------------------------------------
# Bulk fluxes
# ---------------------------------------------------------------------------


def test_bulk_flux_signs_symmetry_and_neutral_drag():
    t = _times(3)
    # stable (air 2 K warmer), unstable (surface 2 K warmer), neutral
    out = bf.bulk_fluxes(
        t_skin_k=_da([250, 252, 250], t),
        t_air_k=_da([252, 250, 250], t),
        rh_air_pct=_da([80, 80, 80], t),
        wspd_m_s=_da([8, 8, 8], t),
        p_pa=_da([101000] * 3, t),
        z_u_m=10.0,
        z_t_m=2.0,
    )
    sh = out["sh_up_W_m2"].values
    assert sh[0] < 0 < sh[1]  # downward when air warmer, upward when surface warmer
    # Stability matters: the same |dT| gives a WEAKER flux in the stable case
    # (damped mixing) than in the unstable one (enhanced), by ~10% at
    # |zeta| ~ 0.1. Equality here would mean the stability correction was
    # never applied (the neutral-first-iterate bug this test now guards).
    assert abs(sh[0]) < abs(sh[1])
    assert abs(sh[0]) == pytest.approx(abs(sh[1]), rel=0.5)
    assert out["zeta"].values[0] > 0 > out["zeta"].values[1]
    # Neutral case: C_D = kappa^2 / ln(z/z0)^2 within the potential-T correction.
    c_dn = bf.neutral_transfer_coefficients(10.0, 2.0, bf.DEFAULT_Z0_M)["c_dn"]
    assert out["c_d"].values[2] == pytest.approx(c_dn, rel=0.02)
    assert np.all(out["c_h"].values > 0) and np.all(out["c_e"].values > 0)
    assert bool(out["converged"].all())


def test_bulk_latent_flux_sign_follows_humidity_gradient():
    """Saturated-over-ice surface, dry air -> sublimation (LH > 0); moist air -> deposition."""
    t = _times(2)
    out = bf.bulk_fluxes(
        t_skin_k=_da([255, 255], t),
        t_air_k=_da([255, 255], t),
        rh_air_pct=_da(
            [40, 100], t
        ),  # w.r.t. liquid; 100% over liquid is supersaturated over ice
        wspd_m_s=_da([6, 6], t),
        p_pa=_da([101000, 101000], t),
    )
    lh = out["lh_up_W_m2"].values
    assert lh[0] > 0 > lh[1]


def test_bulk_flux_stability_scheme_changes_stable_fluxes():
    """The two stable schemes must give different (not identical) fluxes."""
    t = _times(1)
    kw = dict(
        t_skin_k=_da([245.0], t),
        t_air_k=_da([252.0], t),
        rh_air_pct=_da([80.0], t),
        wspd_m_s=_da([2.5], t),
        p_pa=_da([101000.0], t),
    )
    a = bf.bulk_fluxes(**kw, stable_scheme="grachev2007")
    b = bf.bulk_fluxes(**kw, stable_scheme="beljaars1991")
    assert a["zeta"].values[0] > 0.3
    assert a["sh_up_W_m2"].values[0] != pytest.approx(
        b["sh_up_W_m2"].values[0], rel=1e-4
    )
    # both are weaker than the neutral estimate for the same inputs
    neutral = bf.neutral_transfer_coefficients(10.0, 2.0, bf.DEFAULT_Z0_M)["c_hn"]
    assert a["c_h"].values[0] < neutral and b["c_h"].values[0] < neutral


def test_bulk_flux_handles_nan_and_calm():
    t = _times(2)
    out = bf.bulk_fluxes(
        t_skin_k=_da([np.nan, 250], t),
        t_air_k=_da([252, 256], t),
        rh_air_pct=_da([80, 80], t),
        wspd_m_s=_da([5, 0.0], t),
        p_pa=_da([101000, 101000], t),
    )
    assert np.isnan(out["sh_up_W_m2"].values[0])
    assert np.isfinite(out["sh_up_W_m2"].values[1])
    assert out["zeta"].values[1] <= bf.ZETA_MAX


# ---------------------------------------------------------------------------
# Skin temperature and Eq. (1) terms
# ---------------------------------------------------------------------------


def test_skin_temperature_round_trip():
    t = _times(3)
    t_true = _da([245.0, 255.0, 265.0], t)
    lwd = _da([150.0, 200.0, 280.0], t)
    lwu = (
        seb.SNOW_EMISSIVITY * seb.SIGMA_SB_W_M2_K4 * t_true**4
        + (1 - seb.SNOW_EMISSIVITY) * lwd
    )
    back = seb.skin_temperature_from_lwup(lwu, lwd)
    np.testing.assert_allclose(back.values, t_true.values, rtol=1e-6)
    # A blackbody-equivalent brightness temperature inverts to the same T.
    t_b = ((lwu) / seb.SIGMA_SB_W_M2_K4) ** 0.25
    np.testing.assert_allclose(
        seb.skin_temperature_from_irt(t_b, lwd).values, t_true.values, rtol=1e-6
    )


def _synthetic_qcrad(n=5):
    t = _times(n)
    return xr.Dataset(
        {
            "swdn_w_m2": _da(np.zeros(n), t),
            "swup_w_m2": _da(np.zeros(n), t),
            "lwdn_w_m2": _da(np.linspace(160, 260, n), t),
            "lwup_w_m2": _da(np.linspace(220, 250, n), t),
        }
    )


def test_compute_seb_terms_names_match_era5_and_identities():
    q = _synthetic_qcrad()
    t = q["time"]
    sh = _da(np.full(t.size, -10.0), t)
    lh = _da(np.full(t.size, -2.0), t)
    g = _da(np.full(t.size, 5.0), t)
    out = seb.compute_seb_terms(q, sh_up_w_m2=sh, lh_up_w_m2=lh, g_up_w_m2=g)
    era5_names = {
        "lwd_W_m2",
        "lwu_W_m2",
        "lw_net_W_m2",
        "swd_W_m2",
        "swu_W_m2",
        "swn_W_m2",
        "sh_up_W_m2",
        "lh_up_W_m2",
        "forcing_W_m2",
        "na_flux_W_m2",
        "t_skin_from_lwu_K",
    }
    assert era5_names <= set(out.data_vars)
    # Eq. (5): NA = LWD - LWU + SWD - SWU - SH - LH
    expected_na = q["lwdn_w_m2"] - q["lwup_w_m2"] - sh - lh
    np.testing.assert_allclose(out["na_flux_W_m2"].values, expected_na.values)
    # Eq. (1) with SWT = 0: residual = NA + G (= M)
    np.testing.assert_allclose(out["residual_W_m2"].values, (expected_na + g).values)
    assert out["forcing_W_m2"].values == pytest.approx(q["lwdn_w_m2"].values)
    assert "positive UPWARD" in out.attrs["convention"]


# ---------------------------------------------------------------------------
# Ground heat flux
# ---------------------------------------------------------------------------


def test_thermal_inertia_reproduces_half_space_sinusoid():
    """For T_s = A sin(wt) the half-space flux is A sqrt(k rho c) sqrt(w) sin(wt + pi/4)."""
    step = 3600.0
    t_s = np.arange(0, 20 * 86400, step)
    amp, w, ti = 5.0, 2 * np.pi / 86400, 350.0
    da = xr.DataArray(
        250 + amp * np.sin(w * t_s),
        coords={"time": np.datetime64("2026-01-01") + t_s.astype("timedelta64[s]")},
        dims="time",
    )
    g = gf.ground_flux_thermal_inertia(da, thermal_inertia_j_m2_k_s05=ti)
    q_down = -g.values  # positive into the medium
    expected = amp * ti * np.sqrt(w) * np.sin(w * t_s + np.pi / 4)
    tail = slice(-48, None)  # after the initial transient
    assert np.nanmax(np.abs(q_down[tail])) == pytest.approx(
        amp * ti * np.sqrt(w), rel=0.03
    )
    rms = np.sqrt(np.nanmean((q_down[tail] - expected[tail]) ** 2))
    assert rms < 0.05 * amp * ti * np.sqrt(w)


def test_thermal_inertia_rejects_irregular_axis():
    t = np.array(
        ["2026-01-01T00", "2026-01-01T01", "2026-01-01T03"], dtype="datetime64[ns]"
    )
    da = xr.DataArray([250.0, 251.0, 252.0], coords={"time": t}, dims="time")
    with pytest.raises(ValueError):
        gf.ground_flux_thermal_inertia(da)


def test_residual_and_conduction_sign_conventions():
    t = _times(2)
    na = _da([-10.0, 5.0], t)
    assert list(gf.ground_flux_residual(na).values) == [10.0, -5.0]
    g = gf.ground_flux_snow_conduction(
        t_skin_k=_da([250.0, 250.0], t),
        t_snow_base_k=_da([260.0, 240.0], t),  # base warmer -> upward (positive)
        snow_depth_m=_da([0.3, 0.3], t),
        k_snow_w_m_k=xr.DataArray(0.15),
    )
    assert g.values[0] > 0 > g.values[1]
    assert g.values[0] == pytest.approx(0.15 * 10 / 0.3)


def test_sturm_conductivity_is_continuous_and_increasing():
    rho = np.linspace(50, 500, 100)
    k = gf.snow_thermal_conductivity_sturm1997(rho)
    assert np.all(np.diff(k) > -1e-3)
    assert gf.snow_thermal_conductivity_sturm1997(155.9) == pytest.approx(
        gf.snow_thermal_conductivity_sturm1997(156.1), rel=0.05
    )


# ---------------------------------------------------------------------------
# Cloud water path
# ---------------------------------------------------------------------------


def test_iwp_from_uniform_layer_and_phase_mask():
    t = _times(2)
    h = np.arange(160.0, 160.0 + 30.0 * 20, 30.0)  # 20 gates, 30 m
    z = np.full((2, h.size), np.nan)
    z[:, 5:15] = -10.0  # a 300-m layer at -10 dBZ
    refl = xr.DataArray(z, coords={"time": t, "height": h}, dims=("time", "height"))
    iwp = cw.iwp_from_reflectivity(refl)
    expected = config.IWC_PREFACTOR_A * (10 ** (-1.0)) ** config.IWC_EXPONENT_B * 300.0
    np.testing.assert_allclose(iwp.values, expected, rtol=1e-6)
    # Phase mask: first profile all liquid (code 1) -> 0; second all ice (2) -> unchanged.
    pm = xr.DataArray(
        np.vstack([np.full(h.size, 1), np.full(h.size, 2)]),
        coords={"time": t, "height": h},
        dims=("time", "height"),
    )
    masked = cw.iwp_from_reflectivity(refl, phase_mask=pm)
    assert masked.values[0] == 0.0
    assert masked.values[1] == pytest.approx(expected)


def test_cloud_temperature_at_boundaries_interpolates_profile():
    t = _times(2)
    h = np.arange(0.0, 3001.0, 100.0)
    lapse = -6.5e-3  # K/m
    prof = xr.DataArray(
        np.tile(-10.0 + lapse * h, (2, 1)),
        coords={"time": t, "height": h},
        dims=("time", "height"),
        attrs={"units": "degC"},
    )
    base = xr.DataArray(
        [[500.0, np.nan], [1250.0, 2000.0]],
        coords={"time": t, "layer": [0, 1]},
        dims=("time", "layer"),
    )
    top = base + 300.0
    out = cw.cloud_temperature_at_boundaries(base, top, prof)
    assert out["t_lowest_base_K"].values[0] == pytest.approx(273.15 - 10 + lapse * 500)
    assert out["t_top_K"].values[1, 1] == pytest.approx(273.15 - 10 + lapse * 2300)
    assert np.isnan(out["t_base_K"].values[0, 1])


def test_merge_lwp_prefers_primary_and_records_source():
    t = _times(3)
    p = _da([10.0, np.nan, np.nan], t)
    s = _da([12.0, 15.0, np.nan], t)
    out = cw.merge_lwp(p, s)
    assert list(out["lwp_source"].values) == [1, 2, 0]
    assert out["lwp_g_m2"].values[0] == 10.0 and out["lwp_g_m2"].values[1] == 15.0


def test_arscl_sentinel_handling():
    t = _times(3)
    cb = _da([-1.0, -2.0, 500.0], t)
    out = seb._flag_to_nan(cb, (-1.0, -2.0))
    assert (
        np.isnan(out.values[0]) and np.isnan(out.values[1]) and out.values[2] == 500.0
    )


# ---------------------------------------------------------------------------
# Two-pyrgeometer best-estimate LWD
# ---------------------------------------------------------------------------


def test_best_estimate_lwdn_rules():
    t = _times(5)
    ta = _da([250.0] * 5, t)
    bb = seb.SIGMA_SB_W_M2_K4 * 250.0**4  # ~221.6 W/m2
    # 0: agree -> mean; 1: pyrg1 impossible (> bb+25) -> pyrg2; 2: pyrg2 NaN -> pyrg1;
    # 3: both plausible, disagree -> preferred (2), flagged; 4: both impossible -> NaN
    lw1 = _da([200.0, 400.0, 205.0, 215.0, 500.0], t)
    lw2 = _da([204.0, 198.0, np.nan, 190.0, 20.0], t)
    sirs = xr.Dataset({"lwdn_w_m2": lw1, "lwdn2_w_m2": lw2})
    out = seb.best_estimate_lwdn(sirs, t_air_k=ta)
    assert list(out["lwdn_source"].values) == [1, 3, 2, 4, 0]
    np.testing.assert_allclose(
        out["lwdn_be_w_m2"].values[:4], [202.0, 198.0, 205.0, 190.0]
    )
    assert np.isnan(out["lwdn_be_w_m2"].values[4])
    assert out["lwdn_spread_w_m2"].values[0] == pytest.approx(-4.0)
    assert bb + seb.LWDN_TA_UPPER_MARGIN_W_M2 < 400.0  # sanity on the bound used above
    # prefer=1 flips the disagreement case
    out1 = seb.best_estimate_lwdn(sirs, t_air_k=ta, prefer=1)
    assert out1["lwdn_be_w_m2"].values[3] == 215.0


def test_compute_seb_terms_uses_best_estimate_when_sirs_given():
    q = _synthetic_qcrad(4)
    t = q["time"]
    sirs = xr.Dataset(
        {
            "lwdn_w_m2": _da([400.0, 400.0, 400.0, 400.0], t),  # pyrg1 broken
            "lwdn2_w_m2": _da([180.0, 190.0, 200.0, 210.0], t),
        }
    )
    met = xr.Dataset({"temp_2m_c": _da([-25.0] * 4, t), "wspd_m_s": _da([5.0] * 4, t)})
    out = seb.compute_seb_terms(q, met=met, sirs=sirs)
    np.testing.assert_allclose(out["lwd_W_m2"].values, [180.0, 190.0, 200.0, 210.0])
    assert list(out["lwd_source"].values) == [3, 3, 3, 3]
    assert "lwd_spread_W_m2" in out
