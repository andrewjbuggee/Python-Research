"""Offline tests for comparisons/seasonal_averages (no ARM or library access)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from EPCAPE.analysis_tools import qc
from EPCAPE.comparisons.seasonal_averages import quantities as q
from EPCAPE.comparisons.seasonal_averages import seasonal as s


# --- reading the spreadsheet cells ------------------------------------------
@pytest.mark.parametrize(
    "cell, mean, std",
    [
        ("16 ± 2.8", 16.0, 2.8),
        ("1192  ± 397", 1192.0, 397.0),
        ("1.0 * 10^2 ± 66", 100.0, 66.0),
        ("5.0*10^2", 500.0, np.nan),
        (".39 ± 0.19", 0.39, 0.19),
        ("0.11. ± 0.10", 0.11, 0.10),
        ("626 + 297", 626.0, 297.0),
        ("2.8 ± n/a", 2.8, np.nan),
        (0.85, 0.85, np.nan),
        ("n/a", np.nan, np.nan),
    ],
)
def test_parse_cell(cell, mean, std):
    m, sd, _ = s.parse_cell(cell)
    np.testing.assert_allclose([m, sd], [mean, std], equal_nan=True)


# --- seasons ----------------------------------------------------------------
def test_winter_spans_both_years_and_campaign_excludes_outside_days():
    t = pd.to_datetime(["2023-02-14 23:00", "2023-02-15 00:00", "2023-03-31 23:59", "2023-04-01 00:00",
                        "2024-01-15 00:00", "2024-02-14 23:59", "2024-02-15 00:00"])
    np.testing.assert_array_equal(s.season_mask(t, "Winter"), [0, 1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(s.season_mask(t, "EPCAPE"), [0, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(s.season_mask(t, "Spring"), [0, 0, 0, 1, 0, 0, 0])


def test_seasonal_stats_mean_std_and_counts():
    t = pd.date_range("2023-07-01", periods=4, freq="1D")
    stats = s.seasonal_stats(pd.Series([1.0, 2.0, 3.0, np.nan], index=t))
    assert stats.loc["Summer", "n"] == 3 and stats.loc["Summer", "mean"] == 2.0
    assert stats.loc["Summer", "std"] == pytest.approx(1.0)  # sample std (ddof=1)
    assert stats.loc["Winter", "n"] == 0 and np.isnan(stats.loc["Winter", "mean"])


# --- QC = 0 -----------------------------------------------------------------
def test_qc_is_zero_drops_indeterminate_and_missing_qc():
    ds = xr.Dataset(
        {"x": ("time", [1.0, 2.0, 3.0, 4.0]), "qc_x": ("time", [0.0, 4.0, np.nan, 0.0])},
        coords={"time": pd.date_range("2023-07-01", periods=4, freq="1min")},
    )
    ds["qc_x"].attrs.update(bit_3_description="something odd", bit_3_assessment="Indeterminate")
    np.testing.assert_array_equal(qc.qc_is_zero(ds, "x").values, [True, False, False, True])
    # bad_mask's default keeps Indeterminate samples; qc_is_zero does not
    assert not bool(qc.bad_mask(ds, "x").values[1])
    no_qc = ds.drop_vars("qc_x")
    assert qc.qc_is_zero(no_qc, "x").values.all() and not qc.has_qc(no_qc, "x")


# --- LCL --------------------------------------------------------------------
def test_lambertw_lower_branch_inverts():
    x = np.array([-1 / np.e + 1e-9, -0.3, -0.1, -1e-4, -1e-10])
    w = q._lambertw_m1(x)
    np.testing.assert_allclose(w * np.exp(w), x, rtol=1e-9, atol=1e-15)
    assert np.all(w <= -1)


def test_lcl_romps_against_lawrence_rule_of_thumb():
    # Romps (2017) and the 125 m/K rule (Lawrence 2005) agree within a few percent near the surface.
    for p, t_k, rh in [(100000, 300.0, 0.5), (101325, 288.15, 0.7), (100860, 293.05, 0.88)]:
        z = float(q.lcl_romps2017_m(p, t_k, rh))
        td = q.dewpoint_magnus_c(t_k - 273.15, 100 * rh)
        assert z == pytest.approx(float(q.lcl_lawrence2005_m(t_k - 273.15, td)), rel=0.05)
    assert float(q.lcl_romps2017_m(100000, 300.0, 0.5)) == pytest.approx(1433.8, abs=1.0)
    assert float(q.lcl_romps2017_m(100000, 290.0, 1.0)) == 0.0


# --- rain -------------------------------------------------------------------
def test_rain_events_merge_short_gaps_and_compute_intensity():
    t = pd.date_range("2023-07-01", periods=300, freq="1min")
    r = pd.Series(0.0, index=t)
    r.iloc[10:40] = 2.0  # 30 min at 2 mm/h -> 1 mm
    r.iloc[70:100] = 4.0  # 30-min gap: same event; 30 min at 4 mm/h -> 2 mm
    r.iloc[250:260] = 6.0  # 150-min gap: new event; 10 min at 6 mm/h -> 1 mm
    ev = q.rain_events(r, max_gap_min=60)
    assert len(ev) == 2
    first = ev.iloc[0]
    assert first["accumulation_mm"] == pytest.approx(3.0)
    assert first["duration_h"] == pytest.approx(1.5)  # minute 10 to minute 100
    assert first["mean_rate_mm_h"] == pytest.approx(2.0)
    assert ev.iloc[1]["mean_rate_mm_h"] == pytest.approx(6.0)


# --- GCVI / AMS -------------------------------------------------------------
def test_gcvi_residuals_divide_by_ef_and_require_long_segments():
    t = pd.date_range("2023-07-01 00:00", "2023-07-01 01:00", freq="1s")
    ef = pd.Series(np.nan, index=t)
    ef["2023-07-01 00:00:00":"2023-07-01 00:19:59"] = 5.0  # 20-min segment
    ef["2023-07-01 00:40:00":"2023-07-01 00:44:59"] = 5.0  # 5-min segment: too short
    gcvi = pd.DataFrame({"EF": ef, "cut_um": 9.2})
    segs = q.gcvi_segments(gcvi)
    assert len(segs) == 2 and segs["duration_min"].round().tolist() == [20.0, 5.0]
    ams = pd.DataFrame(
        {"organics_ugm3": [5.0, 5.0, 5.0, 5.0]},
        index=pd.to_datetime(["2023-07-01 00:05", "2023-07-01 00:19", "2023-07-01 00:41", "2023-07-01 00:55"]),
    )
    samples, used = q.gcvi_ams_residuals(ams, gcvi, segs, min_duration_min=15)
    # 00:05 counts; 00:19 + 2 min runs past the segment end; 00:41 is in the short segment
    assert samples.index.tolist() == [pd.Timestamp("2023-07-01 00:05")]
    assert samples["organics_resid_ugm3"].iloc[0] == pytest.approx(1.0)
    assert len(used) == 1


# --- ARSCL layers -----------------------------------------------------------
def test_arscl_empty_layers_are_minus_one_not_nan():
    time = pd.date_range("2023-07-01", periods=3, freq="5min")
    base = np.array([[300.0, -1.0], [300.0, 2000.0], [-1.0, -1.0]])
    top = np.array([[600.0, -1.0], [650.0, 2500.0], [-1.0, -1.0]])
    ds = xr.Dataset({"cloud_layer_base_height": (("time", "layer"), base),
                     "cloud_layer_top_height": (("time", "layer"), top)}, coords={"time": time})
    any_layer, _ = q.lowest_layer_top_m(ds, window_min=5)
    single, _ = q.lowest_layer_top_m(ds, single_layer=True, window_min=5)
    assert any_layer.tolist()[:2] == [600.0, 650.0] and np.isnan(any_layer.iloc[2])  # -1 = clear sky
    assert single.iloc[0] == 600.0 and np.isnan(single.iloc[1])  # second layer present at t1


# --- SW / LW averaging window (minutes; 0 = no averaging) ---------------------
def _radflux(n_min=20):
    """Synthetic 1-min RADFLUX-like dataset: SW = 0.5 x clear-sky, LW = 300..319 W m-2."""
    t = pd.date_range("2023-07-01 18:00", periods=n_min, freq="1min")
    cs = np.linspace(800.0, 900.0, n_min)
    ds = xr.Dataset(
        {
            "downwelling_shortwave": ("time", 0.5 * cs),
            "qc_downwelling_shortwave": ("time", np.zeros(n_min)),
            "clearsky_downwelling_shortwave": ("time", cs),
            "downwelling_longwave": ("time", 300.0 + np.arange(n_min)),
            "qc_downwelling_longwave": ("time", np.zeros(n_min)),
            "cosine_zenith": ("time", np.full(n_min, 0.8)),
        },
        coords={"time": t},
    )
    ds["qc_downwelling_shortwave"].values[3] = 4  # one sample fails QC (Indeterminate bit)
    return ds


def test_window_mean_zero_is_every_sample_and_lengths_are_arbitrary():
    t = pd.date_range("2023-07-01", periods=15, freq="1min")
    x = pd.Series(np.arange(15.0), index=t)
    x.iloc[2] = np.nan
    np.testing.assert_array_equal(q.window_mean(x, 0).values, x.dropna().values)
    m = q.window_mean(x, 7.5)  # 7.5-min windows: [0, 7.5) and [7.5, 15) minutes
    assert len(m) == 2
    assert m.iloc[0] == pytest.approx(np.mean([0, 1, 3, 4, 5, 6, 7]))  # samples at 0..7 min, minus the NaN
    assert m.iloc[1] == pytest.approx(np.mean(np.arange(8.0, 15.0)))
    assert q.window_mean(x, 15).iloc[0] == pytest.approx(np.nanmean(x.values))
    with pytest.raises(ValueError):
        q.window_mean(x, -1)


def test_sw_and_lw_window_minutes():
    ds = _radflux()
    t0, info0 = q.sw_transmittance(ds, window_min=0)
    assert len(t0) == 19 and np.allclose(t0.values, 0.5)  # every QC = 0 sample, no averaging
    t5, _ = q.sw_transmittance(ds, window_min=5)
    assert len(t5) == 4 and np.allclose(t5.values, 0.5)
    lw0, _ = q.lw_down(ds, window_min=0)
    lw10, _ = q.lw_down(ds, window_min=10)
    assert len(lw0) == 20 and lw10.tolist() == [304.5, 314.5]
    assert info0["window_min"] == 0


# --- cloudy-scene LWP ----------------------------------------------------------
def test_cloudy_masks_and_cloudy_lwp():
    t = pd.date_range("2023-07-01", periods=6, freq="1min")
    # MWRRET v1: qc bit 2 = clear sky (value missing); bit 3 = no input file (value 1 km, Indeterminate)
    r1 = xr.Dataset(
        {
            "cloud_base_height": ("time", [0.3, np.nan, 0.4, 1.0, 0.5, 0.2]),
            "qc_cloud_base_height": ("time", [0, 2, 0, 4, 0, 0]),
            "phys_lwp": ("time", [50.0, 1.0, 70.0, 80.0, 90.0, 100.0]),
            "qc_phys_lwp": ("time", [0, 2, 0, 0, 0, 1]),
        },
        coords={"time": t},
    )
    r1["phys_lwp"].attrs["units"] = "g/m^2"
    np.testing.assert_array_equal(q.cloudy_mwrret1(r1).values, [True, False, True, False, True, True])
    lwp, info = q.lwp_cloudy_gm2(r1, "phys_lwp", q.cloudy_mwrret1(r1), window_min=2)
    # kept: t0 (50), t2 (70), t4 (90); t5 fails qc_phys_lwp; t1 clear; t3 filled cloud base
    assert lwp.tolist() == [50.0, 70.0, 90.0]
    assert info["cloudy_fraction"] == pytest.approx(3 / 4)  # 4 samples pass LWP QC, 3 are cloudy

    # MWRRET v2: cbh_detected = -0.001 when clear; detection_status 4 = obscured, no base
    r2 = xr.Dataset(
        {
            "cbh_detected": ("time", [0.3, -0.001, 0.4, -0.001, 0.5, 0.2]),
            "detection_status": ("time", [1, 0, 2, 4, 1, 1]),
            "phys_lwp": ("time", [50.0, -3.0, 70.0, 80.0, 90.0, 100.0]),
            "qc_phys_lwp": ("time", [0, 0, 0, 0, 0, 0]),
            "phys_qc_flag": ("time", [0, 0, 0, 0, 1, 0]),
        },
        coords={"time": t},
    )
    r2["phys_lwp"].attrs["units"] = "g/m^2"
    np.testing.assert_array_equal(q.cloudy_mwrret2(r2).values, [True, False, True, False, True, True])
    lwp2, _ = q.lwp_cloudy_gm2(r2, "phys_lwp", q.cloudy_mwrret2(r2), require_zero=("phys_qc_flag",), window_min=1)
    assert lwp2.dropna().tolist() == [50.0, 70.0, 100.0]  # t4 fails phys_qc_flag; clear and obscured dropped

    # window_min = 0: no averaging, every sample that passed QC + cloud filter, at its own time
    raw2, _ = q.lwp_cloudy_gm2(r2, "phys_lwp", q.cloudy_mwrret2(r2), require_zero=("phys_qc_flag",), window_min=0)
    assert raw2.tolist() == [50.0, 70.0, 100.0] and list(raw2.index) == [t[0], t[2], t[5]]

    # min_gm2: keep only LWP > min_gm2 per sample (zero and negative retrievals under a cloud base dropped)
    r3 = r2.copy()
    r3["phys_lwp"] = ("time", [50.0, -3.0, 0.0, 80.0, 90.0, -2.0])
    r3["phys_lwp"].attrs["units"] = "g/m^2"
    all3, _ = q.lwp_cloudy_gm2(r3, "phys_lwp", q.cloudy_mwrret2(r3), require_zero=("phys_qc_flag",), window_min=0)
    assert all3.tolist() == [50.0, 0.0, -2.0]  # default: no lower limit
    pos3, info3 = q.lwp_cloudy_gm2(r3, "phys_lwp", q.cloudy_mwrret2(r3), require_zero=("phys_qc_flag",),
                                   window_min=0, min_gm2=0.0)
    assert pos3.tolist() == [50.0] and list(pos3.index) == [t[0]]
    assert info3["qc"][-1]["n_finite"] == 3 and info3["qc"][-1]["n_kept"] == 1


def test_solar_zenith_at_the_pier_on_the_june_solstice():
    # Local solar noon at the Scripps Pier (32.867 N, 117.257 W) is ~19:50 UTC; at the June solstice the
    # noon zenith angle is |lat - 23.44| ~ 9.43 deg, and at 08:00 UTC (local midnight) the sun is down.
    t = pd.date_range("2023-06-21 00:00", "2023-06-21 23:59", freq="1min")
    sza = q.solar_zenith_deg(t, 32.867, -117.257)
    assert sza.min() == pytest.approx(9.43, abs=0.3)
    assert pd.Timestamp("2023-06-21 19:44") <= t[np.argmin(sza)] <= pd.Timestamp("2023-06-21 19:56")
    assert sza[t.get_loc(pd.Timestamp("2023-06-21 08:00"))] > 90


def test_ceilometer_cloud_frequency_and_goes_like_lwp_windows():
    # two hours of 16-s ceilometer samples: hour 1 = low cloud for 3/4 of the hour, then clear;
    # hour 2 = a 5-km base (not low) for half, then status 0 for the rest, but only 40 % of it reported
    t = pd.date_range("2023-07-01 18:00", periods=450, freq="16s")
    status = np.r_[np.ones(169), np.zeros(56), np.ones(45), np.zeros(45), np.full(135, -1)]
    cbh = np.r_[np.full(169, 400.0), np.full(56, np.nan), np.full(45, 5000.0), np.full(180, np.nan)]
    ceil = xr.Dataset({"first_cbh": ("time", cbh), "qc_first_cbh": ("time", np.where(np.isfinite(cbh), 0, 1)),
                       "detection_status": ("time", status)}, coords={"time": t})
    cf = q.ceilometer_low_cloud_fraction(ceil, max_cbh_m=3500, window_min=60, min_coverage=0.5)
    assert cf["low_cloud_fraction"].iloc[0] == pytest.approx(169 / 225)
    assert np.isnan(cf["low_cloud_fraction"].iloc[1])  # 90 of 225 samples known: under-covered
    # LWP samples: two in hour 1 (mean 60), one in hour 2; hour 1 is daytime at the Pier (18:30 UTC midpoint)
    samples = pd.Series([40.0, 80.0, 30.0], index=pd.to_datetime(
        ["2023-07-01 18:10", "2023-07-01 18:50", "2023-07-01 19:20"]))
    w = q.lwp_goes_like_windows(samples, cf, window_min=60, lat_deg=32.867, lon_deg=-117.257, sza_max_deg=80)
    assert w["lwp_gm2"].tolist() == [60.0, 30.0] and w["n_samples"].tolist() == [2, 1]
    assert bool(w["daytime"].iloc[0]) and np.isnan(w["low_cloud_fraction"].iloc[1])
    # sky classes: hour 1 is cloudy (fraction 0.75) but not overcast; hour 2 has no fraction (under-covered)
    assert q.goes_like_sky(w, "cloudy").tolist() == [60.0]
    assert q.goes_like_sky(w, "overcast").empty
    # RADFLUX zenith option: a cosine of 1 at the midpoint -> 0 deg; no RADFLUX value nearby -> NaN, not daytime
    cz = pd.Series([1.0], index=pd.to_datetime(["2023-07-01 18:30"]))
    w2 = q.lwp_goes_like_windows(samples, cf, window_min=60, lat_deg=0, lon_deg=0, cosine_zenith=cz)
    assert w2["sza_deg"].iloc[0] == pytest.approx(0.0) and np.isnan(w2["sza_deg"].iloc[1])
    assert w2["daytime"].tolist() == [True, False]


def test_arscl_state_and_per_sample_low_cloud_mask():
    # ARSCL flag values: -1 clear, -2 possible clear (unknown); bases above max_cbh_m are "no low cloud"
    t = pd.date_range("2023-07-01 18:00", periods=5, freq="4s")
    arscl = xr.Dataset({"cloud_base_best_estimate": ("time", [500.0, -1.0, -2.0, 6000.0, np.nan])}, coords={"time": t})
    np.testing.assert_array_equal(q.arscl_low_cloud_state(arscl, max_cbh_m=3500).to_numpy(),
                                  [1.0, 0.0, np.nan, 0.0, np.nan])
    # MWR samples at the same times: only the one under low cloud is kept; then the 10-250 g m-2 range
    mwr = xr.Dataset({"phys_lwp": ("time", [80.0, 90.0, 70.0, 60.0, 50.0]), "qc_phys_lwp": ("time", [0] * 5),
                      "phys_qc_flag": ("time", [0] * 5)}, coords={"time": t})
    mwr["phys_lwp"].attrs["units"] = "g/m^2"
    no_rain = pd.Series([0.0], index=pd.to_datetime(["2023-07-01 00:00"]))
    kept, counts = q.lwp_goes_like_samples_gm2(mwr, no_rain, q.arscl_low_cloud_state(arscl, max_cbh_m=3500))
    assert kept.tolist() == [80.0] and counts["low cloud base overhead"] == 1


def test_nonlow_cloud_states_and_highest_arscl_top():
    # ceilometer: low base, high base (non-low), clear, a base failing QC (unknown)
    t = pd.date_range("2023-07-01 18:00", periods=4, freq="16s")
    ceil = xr.Dataset({"first_cbh": ("time", [400.0, 5000.0, np.nan, 800.0]),
                       "qc_first_cbh": ("time", [0, 0, 0, 1]),
                       "detection_status": ("time", [1, 1, 0, 1])}, coords={"time": t})
    nonlow = q.ceilometer_nonlow_cloud_state(ceil, max_cbh_m=3500).to_numpy()
    low = q.ceilometer_low_cloud_state(ceil, max_cbh_m=3500).to_numpy()
    np.testing.assert_array_equal(nonlow, [0.0, 1.0, 0.0, np.nan])
    np.testing.assert_array_equal(np.isnan(nonlow), np.isnan(low))  # same known samples, so TCF - LCF is exact
    # ARSCL: low deck under cirrus (highest top 9 km), single low layer, clear (-1 everywhere), missing
    ta = pd.date_range("2023-07-01 18:00", periods=4, freq="4s")
    top = np.array([[600.0, 9000.0], [700.0, -1.0], [-1.0, -1.0], [np.nan, np.nan]])
    arscl = xr.Dataset({"cloud_layer_top_height": (("time", "layer"), top)}, coords={"time": ta})
    highest = q.arscl_highest_top_m(arscl)
    np.testing.assert_array_equal(highest.to_numpy(), [9000.0, 700.0, -1.0, np.nan])
    np.testing.assert_array_equal(q.arscl_nonlow_cloud_state(highest, max_top_m=3500).to_numpy(),
                                  [1.0, 0.0, 0.0, np.nan])


@pytest.mark.parametrize("sampling, expected", [
    ("hour", 40.0),          # every sample, 10-250 per sample: 20 (clear but >= 10) and 60 -> 40
    ("cloud_sample", 60.0),  # only the samples under low cloud, then 10-250: 60
    ("hour_mean", 95.5),     # every sample averaged (2, 20, 60, 300 -> 95.5), then 10-250 on the mean
])
def test_goes_like_sampling_modes(sampling, expected):
    t = pd.to_datetime(["2023-07-01 18:05", "2023-07-01 18:20", "2023-07-01 18:35", "2023-07-01 18:50"])
    lwp = pd.Series([2.0, 20.0, 60.0, 300.0], index=t)
    state = pd.Series([0.0, 0.0, 1.0, 1.0], index=t)  # clear, clear, low cloud, low cloud
    cf = q.low_cloud_fraction(state, window_min=60, min_coverage=0.0)
    w, _ = q.lwp_goes_like(lwp, cf, sampling=sampling, window_min=60, lat_deg=32.867, lon_deg=-117.257,
                           cloud_state=state, lwp_range_gm2=(10, 250))
    assert q.goes_like_sky(w, "cloudy").tolist() == [pytest.approx(expected)]


def test_near_rain_works_on_any_time_stamps():
    rain_t = pd.date_range("2023-07-01 12:00", periods=3, freq="1min")  # rain 12:00-12:02
    rate = pd.Series([0.0, 2.0, 0.0], index=rain_t)  # only 12:01 is wet
    # native, irregular sample times (seconds off any grid)
    samples = pd.to_datetime(["2023-07-01 11:30:59", "2023-07-01 11:31:01", "2023-07-01 12:30:59",
                              "2023-07-01 12:31:01", "2023-07-01 14:00:17"])
    np.testing.assert_array_equal(q.near_rain(rate, samples, window_min=30),
                                  [False, True, True, False, False])
    # a 5-min window starting 11:26 covers 11:26-11:31, so 11:31 + 30 min reaches the 12:01 rain
    starts = pd.to_datetime(["2023-07-01 11:20", "2023-07-01 11:26"])
    np.testing.assert_array_equal(q.near_rain(rate, starts, window_min=30, value_length_min=5), [False, True])
    assert not q.near_rain(rate.iloc[[0, 2]], samples).any()  # no rain at all -> all dry


def test_cloud_top_source_selection():
    # 3 samples, 4 gates (centres 160-250 m); lowest-layer tops at 220, 190 and none
    t = pd.date_range("2023-07-01", periods=3, freq="4s")
    ds = xr.Dataset(
        {
            "cloud_layer_top_height": (("time", "layer"), [[220.0, -1.0], [190.0, -1.0], [-1.0, -1.0]]),
            "cloud_layer_base_height": (("time", "layer"), [[160.0, -1.0], [160.0, -1.0], [-1.0, -1.0]]),
            "cloud_source_flag": (("time", "height"), [[2, 2, 2, 1], [4, 4, 1, 1], [1, 1, 1, 1]]),
        },
        coords={"time": t, "height": [160.0, 190.0, 220.0, 250.0]},
    )
    assert q.arscl_cloud_top_source(ds).values.tolist() == [2, 4, -1]  # code at the top gate; -1 = no layer
    every, _ = q.lowest_layer_top_m(ds, window_min=0)
    radar, info = q.lowest_layer_top_m(ds, window_min=0, cloud_sources=(2, 3, 5))
    assert every.tolist() == [220.0, 190.0] and radar.tolist() == [220.0]  # lidar-only top dropped
    assert info["top_source_counts"] == {2: 1, 4: 1}
    with pytest.raises(KeyError):
        q.lowest_layer_top_m(ds.drop_vars("cloud_source_flag"), cloud_sources=(2,))


# --- FM-120 cloud / haze hours ------------------------------------------------------
def test_fm120_cloud_haze_classification_and_hours():
    t = pd.date_range("2023-07-01", periods=5, freq="5min")
    fm = pd.DataFrame({"nd_cm3": [100.0, 100.0, 3.0, 0.5, np.nan],
                       "lwc_gm3": [0.2, 0.005, 0.004, 0.0005, 0.3]}, index=t)
    cls = q.fm120_classify(fm, cloud_nd_min_cm3=50, cloud_lwc_min_gm3=0.01,
                           haze_nd_min_cm3=1, haze_lwc_min_gm3=0.001)
    assert cls["cloud"].tolist() == [True, False, False, False, False]
    assert cls["haze"].tolist() == [False, True, True, False, False]  # droplets, but not cloud
    assert cls["operating"].tolist() == [True, True, True, True, False]
    hrs = q.interval_hours(cls["haze"])
    assert hrs.sum() == pytest.approx(2 * 5 / 60)


def test_visibility_intervals_and_classification():
    # 1-s visibility over three 5-min intervals: fog, haze, and a gap
    t = pd.date_range("2023-07-01 00:00:00", "2023-07-01 00:09:59", freq="1s")
    vis = pd.Series(np.where(t < pd.Timestamp("2023-07-01 00:05"), 500.0, 2000.0), index=t)
    vis.iloc[10] = 1e8  # spike above the sensor ceiling -> dropped
    vis.iloc[11] = 0.0  # zero -> dropped
    starts = pd.date_range("2023-07-01", periods=3, freq="5min")
    st = q.visibility_on_intervals(vis, starts, min_samples=125)
    assert st["vis_median_m"].tolist()[:2] == [500.0, 2000.0] and np.isnan(st["vis_median_m"].iloc[2])
    assert st["n_vis"].tolist() == [298, 300, 0]
    # defaults = the BAMS-paper definitions: cloud = vis < 1 km & LWC > 0.01; haze = vis < 5 km & LWC < 0.01
    fm = pd.DataFrame({"nd_cm3": [100.0, 10.0, 5.0], "lwc_gm3": [0.2, 0.005, 0.3]}, index=starts)
    cls = q.fm120_classify_visibility(fm, st["vis_median_m"])
    assert cls["cloud"].tolist() == [True, False, False]  # 500 m, LWC 0.2
    assert cls["haze"].tolist() == [False, True, False]  # 2 km, LWC 0.005; the third has no visibility
    assert cls["has_vis"].tolist() == [True, True, False]
    # fog with almost no liquid water is haze; 2 km with LWC >= 0.01 is neither
    fm2 = fm.assign(lwc_gm3=[0.005, 0.05, 0.3])
    cls2 = q.fm120_classify_visibility(fm2, st["vis_median_m"])
    assert cls2["haze"].tolist() == [True, False, False] and not cls2["cloud"].any()
    assert cls2["low_vis_neither"].tolist() == [False, True, False]
    # the sheet's reading: haze = 1 km < vis < 5 km with no LWC condition
    cls3 = q.fm120_classify_visibility(fm2, st["vis_median_m"], haze_vis_min_m=1000.0, haze_lwc_max_gm3=np.inf)
    assert cls3["haze"].tolist() == [False, True, False]
    assert cls3["low_vis_neither"].tolist() == [True, False, False]
    # cloud and haze never overlap, even with no LWC condition on haze
    cls4 = q.fm120_classify_visibility(fm, st["vis_median_m"], haze_lwc_max_gm3=np.inf)
    assert not (cls4["cloud"] & cls4["haze"]).any()
