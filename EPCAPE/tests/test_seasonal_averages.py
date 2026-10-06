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
    any_layer, _ = q.lowest_layer_top_m(ds, period="5min")
    single, _ = q.lowest_layer_top_m(ds, single_layer=True, period="5min")
    assert any_layer.tolist()[:2] == [600.0, 650.0] and np.isnan(any_layer.iloc[2])  # -1 = clear sky
    assert single.iloc[0] == 600.0 and np.isnan(single.iloc[1])  # second layer present at t1
