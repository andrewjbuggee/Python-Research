"""Tests for the cloud-property analysis layer (offline, synthetic data only).

Covers: QC bit decoding, unit conversion, window statistics, paired
statistics, case-insensitive/optional variable resolution, and an
end-to-end run (synthetic files -> combine -> filter -> collocate -> save)
that checks a bias built into the synthetic data is recovered.
"""

import datetime as dt

import netCDF4
import numpy as np
import pytest
import xarray as xr

from EPCAPE.analysis_tools import filters, qc, units
from EPCAPE.analysis_tools.stats import paired_stats
from EPCAPE.download_data.sync import resolve_variables

from synthetic_cloud_vaps import write_campaign


# -- QC ---------------------------------------------------------------------------
def _qc_ds(values, **attrs):
    t = np.arange(len(values)).astype("datetime64[s]")
    return xr.Dataset(
        {"x": ("time", np.ones(len(values))), "qc_x": ("time", np.asarray(values, float), attrs)},
        coords={"time": t},
    )


def test_bad_mask_uses_assessments_and_ignores_indeterminate():
    attrs = {
        "bit_1_description": "missing",
        "bit_1_assessment": "Bad",
        "bit_2_description": "Value is less than the valid_min",
        "bit_2_assessment": "Bad",
        "bit_3_description": "delta",
        "bit_3_assessment": "Indeterminate",
    }
    ds = _qc_ds([0, 1, 2, 4, 6, np.nan], **attrs)
    assert qc.bad_mask(ds, "x").values.tolist() == [False, True, True, False, True, True]
    assert qc.bad_mask(ds, "x", indeterminate_is_bad=True).values[3]
    # ignoring the minimum test keeps bit-2-only samples (pattern as used by mwrlos)
    from EPCAPE.instruments.mwr.mwrlos import MIN_TEST_PATTERNS

    assert qc.bad_mask(ds, "x", ignore_tests_matching=MIN_TEST_PATTERNS).values.tolist() == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_bad_mask_flag_masks_style_and_undescribed():
    ds = _qc_ds(
        [0, 1, 2], flag_masks=[1, 2], flag_meanings="missing suspicious", flag_assessments="Bad Indeterminate"
    )
    assert qc.bad_mask(ds, "x").values.tolist() == [False, True, False]
    ds = _qc_ds([0, 3])
    assert qc.bad_mask(ds, "x").values.tolist() == [False, True]


def test_bad_mask_without_qc_field_flags_nothing():
    ds = xr.Dataset({"x": ("time", [1.0, 2.0])})
    assert not qc.bad_mask(ds, "x").values.any()


# -- units ------------------------------------------------------------------------
@pytest.mark.parametrize("unit,factor", [("cm", 1e4), ("mm", 1e3), ("g/m2", 1.0), ("kg m-2", 1e3)])
def test_water_path_to_gm2(unit, factor):
    da = xr.DataArray([0.01], attrs={"units": unit}, name="lwp")
    assert units.water_path_to_gm2(da).values[0] == pytest.approx(0.01 * factor)


def test_water_path_unknown_unit_raises():
    with pytest.raises(ValueError):
        units.water_path_to_gm2(xr.DataArray([1.0], attrs={"units": "furlongs"}, name="lwp"))


@pytest.mark.parametrize("unit", ["microns", "micron", "um", "μm", "micrometers"])
def test_radius_units(unit):
    assert units.radius_to_um(xr.DataArray([8.0], attrs={"units": unit})).values[0] == 8.0


# -- window statistics --------------------------------------------------------------
def test_window_stats_matches_brute_force():
    from EPCAPE.comparisons.cloud_optical_properties.collocate import window_stats

    rng = np.random.default_rng(0)
    t = np.datetime64("2023-07-01") + (np.arange(500) * 20).astype("timedelta64[s]")
    v = rng.normal(10, 2, t.size)
    v[rng.random(t.size) < 0.2] = np.nan
    da = xr.DataArray(v, coords={"time": t}, dims="time")
    centers = t[::37] + np.timedelta64(7, "s")
    hw = np.timedelta64(150, "s")
    st = window_stats(da, centers, hw)
    for i, c in enumerate(centers):
        sel = (t >= c - hw) & (t <= c + hw)
        vals = v[sel][np.isfinite(v[sel])]
        assert st["n"][i] == vals.size
        assert st["n_total"][i] == sel.sum()
        if vals.size:
            assert st["mean"][i] == pytest.approx(vals.mean())
        if vals.size > 1:
            assert st["std"][i] == pytest.approx(vals.std(ddof=1))


def test_lwp_from_tau_re():
    from EPCAPE.comparisons.cloud_optical_properties.collocate import lwp_from_tau_re_gm2

    # tau = 10, r_e = 10 um -> (2/3) * 1e6 g m-3 * 10 * 1e-5 m = 66.7 g m-2
    assert lwp_from_tau_re_gm2(10.0, 10.0) == pytest.approx(200.0 / 3.0)


# -- statistics -------------------------------------------------------------------
def test_paired_stats_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = 2 * x + 1
    s = paired_stats(x, y)
    assert s["n"] == 4
    assert s["bias"] == pytest.approx(np.mean(x[:4] + 1))
    assert s["r"] == pytest.approx(1.0)
    assert s["rma_slope"] == pytest.approx(2.0)
    assert s["rma_intercept"] == pytest.approx(1.0)


# -- variable resolution ------------------------------------------------------------
def test_resolve_variables_case_insensitive_and_optional():
    available = {"time": ("time",), "lwp": ("time",), "qc_lwp": ("time",), "Cloudfraction": ("time",)}
    out = resolve_variables(["Lwp"], available, optional=["cloudfraction", "not_there"])
    assert out == ["time", "lwp", "qc_lwp", "Cloudfraction"]
    with pytest.raises(ValueError):
        resolve_variables(["missing_var"], available)


# -- end to end -------------------------------------------------------------------
def test_end_to_end_recovers_built_in_tau_bias(tmp_path, monkeypatch):
    monkeypatch.setenv("EPCAPE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("EPCAPE_MACHINE", raising=False)
    monkeypatch.delenv("EPCAPE_CONFIG", raising=False)
    write_campaign(tmp_path / "data", dt.date(2023, 7, 1), days=3, seed=3)

    from EPCAPE.comparisons.cloud_optical_properties import collocate
    from EPCAPE.analysis_tools.derived import save_derived
    from EPCAPE.instruments.mfrsr import mfrsrcldod
    from EPCAPE.instruments.mwr import mwrlos
    from EPCAPE.instruments.sunphotometer import sphotcod

    log = lambda *_: None  # noqa: E731
    m = mfrsrcldod.standardize(mfrsrcldod.load(log=log))
    s = sphotcod.standardize(sphotcod.load(log=log))
    w = mwrlos.standardize(mwrlos.load(log=log))
    ok_m = filters.combine(mfrsrcldod.tau_criteria(m))
    ok_s = filters.combine(sphotcod.tau_criteria(s))
    ok_w = filters.combine(mwrlos.lwp_criteria(w))

    # the synthetic wet-window event (day 2) must be rejected
    assert not ok_w.where(w["wet_window"] == 1, drop=True).any()

    matched = collocate.match_to_times(
        {"sphot_tau": filters.apply(s["tau"], ok_s)},
        {"mfrsr_tau": filters.apply(m["tau"], ok_m), "mwr_lwp_gm2": filters.apply(w["lwp_gm2"], ok_w)},
        half_width_min=2.5,
    )
    pair = np.isfinite(matched["sphot_tau"]) & (matched["mfrsr_tau_coverage"] >= 0.6)
    st = paired_stats(matched["mfrsr_tau"].where(pair), matched["sphot_tau"].where(pair), log=True)
    assert st["n"] > 20
    assert 8 < st["rel_bias_pct"] < 22  # built in: +15%

    matched["pair_tau"] = pair
    out = save_derived(matched, "test_matched", settings={"window": 2.5, "flag": None})
    with netCDF4.Dataset(out) as nc:
        assert nc.variables["time"].units.startswith("seconds since 1970-01-01")
        assert nc.variables["pair_tau"].dtype == np.int8
        assert nc.getncattr("setting_window") == 2.5


def test_mwr_keeps_values_below_valid_min(tmp_path, monkeypatch):
    """TR-016: negative LWP flagged only by the minimum test is usable (physically zero)."""
    monkeypatch.setenv("EPCAPE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("EPCAPE_MACHINE", raising=False)
    write_campaign(tmp_path / "data", dt.date(2023, 7, 1), days=1, seed=3)
    from EPCAPE.instruments.mwr import mwrlos

    raw = mwrlos.load(log=lambda *_: None)
    flagged_min = (raw["qc_liq"].fillna(0).astype(int) & 2) != 0
    assert int(flagged_min.sum()) == 10
    std = mwrlos.standardize(raw)
    assert not std["qc_bad_lwp"].where(flagged_min, drop=True).any()


# -- added 2026-10-06 after checking the real EPCAPE files ---------------------------
def _synthetic(tmp_path, monkeypatch, days=2):
    monkeypatch.setenv("EPCAPE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("EPCAPE_MACHINE", raising=False)
    monkeypatch.delenv("EPCAPE_CONFIG", raising=False)
    write_campaign(tmp_path / "data", dt.date(2023, 7, 1), days=days, seed=5)


def test_sphot_gain_selection_and_flag_bits(tmp_path, monkeypatch):
    _synthetic(tmp_path, monkeypatch)
    from EPCAPE.instruments.sunphotometer import sphotcod

    raw = sphotcod.load(log=lambda *_: None)
    assert raw["cloud_optical_depth"].dims == ("time", "gain")
    std = sphotcod.standardize(raw)  # default: mean of A and K
    assert std["tau"].dims == ("time",) and std.attrs["gain"] == "mean of A and K"
    np.testing.assert_allclose(std["tau"].values, raw["cloud_optical_depth"].sel(gain=2).values)
    assert sphotcod.standardize(raw, gain=None)["tau"].dims == ("time", "gain")

    flag = std["retrieval_flag"].values
    strict = filters.combine(sphotcod.tau_criteria(std, sphotcod.Criteria())).values
    relaxed = filters.combine(
        sphotcod.tau_criteria(std, sphotcod.Criteria(ignore_flag_bits=sphotcod.SPECTRAL_SIGNATURE_BITS))
    ).values
    assert np.array_equal(strict, flag == 0)  # every test is "Bad"
    assert np.array_equal(relaxed, (flag & ~16) == 0)  # bit 5 (value 16) ignored
    table = qc.bit_breakdown(std["retrieval_flag"])
    assert set(table["bit"].iloc[:-1]) == {1, 2, 3, 4, 5, 6, 7}


def test_combine_keeps_per_file_static_variables(tmp_path, monkeypatch):
    """SPHOT albedo has no time dimension and changes daily: each day's values must survive."""
    _synthetic(tmp_path, monkeypatch, days=3)
    from EPCAPE.instruments.sunphotometer import sphotcod

    std = sphotcod.standardize(sphotcod.load(log=lambda *_: None))
    a858 = sphotcod.albedo_at(std, 858.0)
    per_day = a858.to_series().resample("1D").first().dropna()
    assert std["modis_albedo"].dims == ("time", "modis_channel")
    assert per_day.nunique() == per_day.size > 1


def test_all_nan_ancillary_is_skipped_not_failed(tmp_path, monkeypatch):
    _synthetic(tmp_path, monkeypatch)
    from EPCAPE.instruments.mfrsr import mfrsrcldod

    std = mfrsrcldod.standardize(mfrsrcldod.load(log=lambda *_: None))
    std["ir_temp_K"] = std["ir_temp_K"] * np.nan  # as at EPCAPE: never filled
    crit = mfrsrcldod.tau_criteria(std)
    assert crit["IR sky T > 268 K"] is None
    assert int(filters.combine(crit).sum()) > 0
    assert "skipped" in filters.funnel(crit)["criterion"].str.cat()


def test_mwr_default_keeps_wet_window_but_drops_rain(tmp_path, monkeypatch):
    _synthetic(tmp_path, monkeypatch)
    from EPCAPE.instruments.mwr import mwrlos

    std = mwrlos.standardize(mwrlos.load(log=lambda *_: None))
    ok = filters.combine(mwrlos.lwp_criteria(std)).values
    wet = std["wet_window"].values == 1
    lwp = std["lwp_gm2"].values
    assert ok[wet & (lwp < 1000) & (std["tb31_K"].values < 100)].all()  # heater on, plausible LWP: kept
    assert not ok[lwp >= 1000].any()  # rain / water on window: dropped
    strict = filters.combine(mwrlos.lwp_criteria(std, mwrlos.Criteria(reject_wet_window=True))).values
    assert not strict[wet].any()
