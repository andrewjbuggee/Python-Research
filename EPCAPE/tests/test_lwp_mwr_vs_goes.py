"""Offline tests for comparisons/lwp_mwr_vs_goes (no ARM access)."""

import numpy as np
import pandas as pd
import pytest

from EPCAPE.comparisons.lwp_mwr_vs_goes import goes18 as g
from EPCAPE.comparisons.lwp_mwr_vs_goes import matching as m


def _series(values, start="2023-07-01 00:00", step_s=10):
    t = pd.date_range(start, periods=len(values), freq=pd.Timedelta(seconds=step_s))
    return pd.Series(np.asarray(values, dtype=float), index=t)


def test_window_means_counts_and_nan():
    # 10-s samples for 2 min: window [0, 60 s) holds 6 samples, one NaN; [60, 120 s) holds 6
    s = _series([1, 2, 3, np.nan, 5, 6, 10, 10, 10, 10, 10, 10])
    starts = pd.DatetimeIndex(["2023-07-01 00:00", "2023-07-01 00:01", "2023-07-01 00:05"])
    w = m.window_means(s, starts, pd.Timedelta(minutes=1))
    assert w["n"].tolist() == [5, 6, 0]
    assert w["mean"].iloc[0] == pytest.approx((1 + 2 + 3 + 5 + 6) / 5)
    assert w["mean"].iloc[1] == pytest.approx(10.0)
    assert np.isnan(w["mean"].iloc[2])  # empty window -> NaN, not 0


def test_window_means_overlapping_windows():
    s = _series(np.arange(12))
    starts = pd.DatetimeIndex(["2023-07-01 00:00:00", "2023-07-01 00:00:30"])
    w = m.window_means(s, starts, pd.Timedelta(minutes=1))
    assert w["mean"].tolist() == pytest.approx([2.5, 5.5])  # samples 0-5 and 3-8


def test_state_fraction_coverage():
    # ceilometer-like 16-s states: 1 low cloud, 0 clear, NaN unknown
    st = _series([1, 1, 0, np.nan, 0, 1, 1, 1], step_s=15)  # 2 min of samples
    starts = pd.DatetimeIndex(["2023-07-01 00:00", "2023-07-01 00:01"])
    f = m.state_fraction(st, starts, pd.Timedelta(minutes=1), min_coverage=0.5)
    assert f.iloc[0] == pytest.approx(2 / 3)  # unknown sample left out of numerator and denominator
    assert f.iloc[1] == pytest.approx(3 / 4)
    f_strict = m.state_fraction(st, starts, pd.Timedelta(minutes=1), min_coverage=1.0)
    assert np.isnan(f_strict.iloc[0]) and f_strict.iloc[1] == pytest.approx(0.75)


def test_kazr_window_cth_literal_and_mean():
    # tops (m): -1 no layer, 500/800 low, 3000 high; the second 800 m sample precipitates
    top = _series([-1, 500, 800, 3000, 800, np.nan])
    precip = np.array([False, False, False, False, True, False])
    starts = pd.DatetimeIndex(["2023-07-01 00:00"])
    k = m.kazr_window_cth_m(top, precip, starts, pd.Timedelta(minutes=1), cth_max_m=2000.0)
    assert k["cth_low_mean_m"].iloc[0] == pytest.approx(650.0)  # mean of 500, 800 (precip and > 2 km removed)
    assert k["cth_all_mean_m"].iloc[0] == pytest.approx((500 + 800 + 3000) / 3)
    assert k["n_low"].iloc[0] == 2 and k["n_layer"].iloc[0] == 3


def test_image_windows_bin_and_centered():
    idx = pd.DatetimeIndex(["2023-07-01 15:00:21", "2023-07-01 15:30:21"])
    images = pd.DataFrame({"scan_time": idx + pd.Timedelta(seconds=87)}, index=idx)
    assert m.image_windows(images, alignment="bin").tolist() == [pd.Timestamp("2023-07-01 15:00"),
                                                                pd.Timestamp("2023-07-01 15:30")]
    c = m.image_windows(images, alignment="centered")
    assert c[0] == pd.Timestamp("2023-07-01 15:01:48") - pd.Timedelta(minutes=15)


def test_constraint_masks_inequalities():
    pairs = pd.DataFrame({
        "mwr_lwp_gm2": [10.0, 250.0, 9.9, 100.0], "goes_lwp_gm2": [50.0, 50.0, 50.0, 50.0],
        "goes_lcf": [0.11, 0.10, 0.5, 0.5], "ceil_cff": [0.5, 0.5, 0.5, np.nan],
        "goes_tcf": [0.21, 0.5, 0.5, 0.5], "goes_sza_deg": [69.9, 70.0, 10.0, 10.0],
        "goes_cth_km": [2.0, 1.0, 1.0, 2.01], "goes_cod": [2.01, 2.0, 5.0, 5.0],
        "kazr_cth_low_mean_m": [900.0, np.nan, 900.0, 900.0], "kazr_cth_all_mean_m": [900.0, 2500.0, 900.0, 900.0],
    })
    mk = m.constraint_masks(pairs)
    assert mk["MWR LWP 10-250 g m-2"].tolist() == [True, True, False, True]  # inclusive range
    assert mk["GOES-18 LCF > 10 %"].tolist() == [True, False, True, True]  # strict
    assert mk["ceilometer CFF > 10 %"].tolist() == [True, True, True, False]  # NaN fails
    assert mk["GOES-18 TCF - LCF <= 10 %"].tolist() == [True, False, True, True]
    assert mk["GOES-18 SZA < 70 deg"].tolist() == [True, False, True, True]
    assert mk["GOES-18 CTH <= 2 km"].tolist() == [True, True, True, False]
    assert mk["KAZR CTH <= 2 km"].tolist() == [True, False, True, True]
    assert mk["GOES-18 COD > 2"].tolist() == [True, False, True, True]
    assert m.all_constraints(mk).tolist() == [True, False, False, False]


def test_pier_box_index():
    lat = np.arange(42.75, 28.0, -0.5)  # the ARM grid: 30 box centres, descending
    lon = np.arange(-129.75, -112.0, 0.5)  # 36 box centres
    i, j = g.pier_box_index(lat, lon, 32.8654, -117.2536)
    assert lat[i] == pytest.approx(32.75) and lon[j] == pytest.approx(-117.25)
    with pytest.raises(ValueError):
        g.pier_box_index(lat, lon, 20.0, -117.25)
