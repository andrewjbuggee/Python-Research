"""Offline tests for comparisons/seasonal_averages/kavin.py: Kavin's MATLAB calculations re-run in Python.

Each test builds a tiny input whose MATLAB result can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from EPCAPE.comparisons.seasonal_averages import kavin as k


# --- row 10: the zero-padded LWP array -----------------------------------------------
def test_kavin_lwp_box_zero_padding_and_pacific_months():
    # Hourly values (UTC). With initial_rows = 3, Mar 2023 (5 values) grows the array to 5 rows,
    # and MATLAB fills rows 4-5 of every other column with 0.
    feb = pd.date_range("2023-02-20 20:00", periods=2, freq="h")  # 12:00 PST on Feb 20
    mar = pd.date_range("2023-03-10 20:00", periods=5, freq="h")
    apr = pd.DatetimeIndex(["2023-05-01 05:00"])  # 22:00 PDT on Apr 30 -> April in Pacific time
    lwp = pd.Series(np.arange(1.0, 9.0), index=feb.append(mar).append(apr))
    box = k.kavin_lwp_box(lwp, initial_rows=3)
    assert box.shape == (5, 13)
    np.testing.assert_array_equal(box[:, 0], [1, 2, np.nan, 0, 0])  # Feb 2023
    np.testing.assert_array_equal(box[:, 1], [3, 4, 5, 6, 7])  # Mar 2023
    np.testing.assert_array_equal(box[:, 2], [8, np.nan, np.nan, 0, 0])  # Apr 2023 (Pacific time)
    np.testing.assert_array_equal(box[:, 12], [np.nan, np.nan, np.nan, 0, 0])  # Feb 2024, empty
    # without the padding the grown rows are NaN
    assert np.isnan(k.kavin_lwp_box(lwp, initial_rows=3, zero_pad=False)[3:, 0]).all()
    # statistics over column 1 only (Feb 2023): mean of 1, 2, 0, 0
    cols = {season: (1,) for season in ("EPCAPE", "Spring", "Summer", "Fall", "Winter")}
    st = k.kavin_lwp_stats(box, cols)
    assert st.loc["EPCAPE", "mean"] == pytest.approx(0.75)
    assert st.loc["EPCAPE", "n_zero_pad"] == 2 and st.loc["EPCAPE", "n"] == 4


# --- row 7: Lubin's SW tables -----------------------------------------------------------
def _sw_table(n_days: int) -> pd.DataFrame:
    """Day x hour table with value = day + hour / 100, and a -0.05 night placeholder on day 1."""
    vals = np.array([[d + h / 100 for h in range(24)] for d in range(1, n_days + 1)])
    vals[0, 0] = -0.05
    return pd.DataFrame(vals, index=pd.RangeIndex(1, n_days + 1), columns=pd.RangeIndex(0, 24))


def test_kavin_sw_cells_drop_day31_and_hour0_keep_placeholders():
    cells = k.kavin_sw_cells(_sw_table(31), month=10)
    assert cells.shape == (30, 24)  # days 1-30; hours 1-23 + a NaN column
    assert cells[0, 0] == pytest.approx(1.01) and np.isnan(cells[:, 23]).all()
    assert cells[-1, 0] == pytest.approx(30.01)  # day 31 dropped
    assert k.kavin_sw_cells(_sw_table(28), month=2).shape == (28, 24)
    t = _sw_table(31)
    t.iloc[0, 5] = -0.05  # a placeholder inside hours 1-23 is kept by Kavin's cells ...
    assert (k.kavin_sw_cells(t, 10) < 0).sum() == 1
    assert np.isnan(k.corrected_sw_cells(t)[0, 5])  # ... and removed by the corrected cells


def test_kavin_sw_stats_missing_month_is_nan_with_partial():
    sw = {10: _sw_table(31), 11: _sw_table(30), 12: _sw_table(31), 4: _sw_table(30)}
    st = k.kavin_sw_stats(sw)
    fall = np.vstack([k.kavin_sw_cells(sw[m], m) for m in (10, 11, 12)])
    assert st.loc["Fall", "mean"] == pytest.approx(np.nanmean(fall))
    assert st.loc["Fall", "std"] == pytest.approx(np.nanstd(fall, ddof=1))
    assert np.isnan(st.loc["Spring", "mean"]) and st.loc["Spring", "n"] == 0
    assert st.loc["Spring", "months_missing"] == "MAY, JUN" and np.isfinite(st.loc["Spring", "partial_mean"])


def test_lubin_sw_series_pst_to_utc_and_february_years():
    t = pd.DataFrame(np.nan, index=pd.RangeIndex(1, 29), columns=pd.RangeIndex(0, 24))
    t.loc[1, 5] = 0.5  # Feb 1, 05:00 PST -> 2024 (campaign ends 2024-02-14)
    t.loc[20, 10] = 0.7  # Feb 20, 10:00 PST -> 2023
    s = k.lubin_sw_series({2: t})
    assert s.loc[pd.Timestamp("2024-02-01 13:00")] == 0.5
    assert s.loc[pd.Timestamp("2023-02-20 18:00")] == 0.7


def test_read_lubin_sw_header_and_placeholders(tmp_path):
    header = ",".join(str(h) for h in range(24)) + ",Means"
    day1 = ",".join(["-0.05"] * 5 + ["0.5"] * 13 + ["-0.05"] * 6) + ",-0.05"
    day2 = ",".join([""] * 5 + ["0.6"] * 13 + [""] * 6) + ","
    (tmp_path / "JUL_SW_Boxplot.csv").write_text("﻿" + "\n".join([header, day1, day2]), encoding="utf-8")
    sw = k.read_lubin_sw(tmp_path)
    assert list(sw) == [7] and sw[7].shape == (2, 24)
    assert sw[7].loc[1, 0] == -0.05 and np.isnan(sw[7].loc[2, 0]) and sw[7].loc[2, 5] == 0.6


# --- row 8: Lubin's LW medians -----------------------------------------------------------
def test_kavin_lw_stats_mean_of_diurnal_cycle():
    hours = np.arange(24)
    cycle = 5 * np.sin(2 * np.pi * hours / 24)
    med = pd.DataFrame({10: 300 + cycle, 11: 310 + cycle, 12: 320 + cycle}, index=hours)
    st = k.kavin_lw_stats(med)
    assert st.loc["Fall", "mean"] == pytest.approx(310.0)
    assert st.loc["Fall", "std"] == pytest.approx(np.std(cycle, ddof=1))  # spread of the 24 hours
    assert st.loc["Fall", "n"] == 24 and np.isnan(st.loc["Summer", "mean"])


def test_read_lubin_lw_medians_prefers_season_file_and_checks_agreement(tmp_path):
    hours = range(24)
    son = pd.DataFrame({"Hour": hours, "SEP": 380.0, "OCT": 345.0, "NOV": 308.0})
    son.to_csv(tmp_path / "SON_LW_Stepplot.csv", index=False)
    month = pd.DataFrame({"Hour": hours, "Min": 1.0, "25%ile": 2.0, "Median": 345.0, "75%ile": 3.0, "Max": 4.0})
    month.to_csv(tmp_path / "OCT_LW_Stepplot.csv", index=False)
    month.assign(Median=316.0).to_csv(tmp_path / "DEC_LW_Stepplot.csv", index=False)
    med, src = k.read_lubin_lw_medians(tmp_path)
    assert list(med.columns) == [9, 10, 11, 12] and src[10] == "SON_LW_Stepplot.csv" and src[12] == "DEC_LW_Stepplot.csv"
    month.assign(Median=999.0).to_csv(tmp_path / "OCT_LW_Stepplot.csv", index=False)
    with pytest.raises(ValueError):
        k.read_lubin_lw_medians(tmp_path)


# --- rows 31-32: Kavin's visibility windows and hour counts --------------------------------
def test_kavin_interval_visibility_windows_split_and_counts():
    starts = pd.DatetimeIndex(["2023-07-19 00:00", "2023-07-19 00:05", "2023-07-19 00:10",
                               "2023-07-19 00:20", "2023-07-19 00:25", "2023-07-19 00:30"])
    t1 = pd.date_range("2023-07-19 00:00:00", "2023-07-19 00:12:00", freq="1s")
    file1 = pd.Series(500.0, index=t1)
    file1.loc["2023-07-19 00:01:00"] = 1e8  # spike: not screened by Kavin's code
    t2 = pd.date_range("2023-07-19 00:20:01", "2023-07-19 00:35:00", freq="1s")
    file2 = pd.Series(2000.0, index=t2)
    vis = k.kavin_interval_visibility(starts, [file1, file2])
    assert vis.iloc[0] > 1e5  # the spike dominates the mean of row 0
    assert vis.iloc[1] == 500.0  # [00:05, 00:10], both ends included
    assert vis.iloc[2] == 500.0  # [00:10, 00:20] spans the gap; file 1 ends at 00:12
    assert vis.iloc[3] == 0.0  # the row at file 2's start (00:20:01 floored to 00:20) is set to 0
    assert vis.iloc[4] == 2000.0  # [00:25, 00:30] from file 2
    assert vis.iloc[5] == 0.0  # the last row keeps its initial 0
    lwc = pd.Series([0.2, 0.2, 0.005, 0.2, 0.005, 0.005], index=starts)
    hours = k.kavin_fm120_hours(lwc, vis.where(vis.index != starts[4], 3000.0))
    # cloud: rows 1, 2 (500 m, LWC 0.2 / 0.005 -> only row 1) and row 3 (0 m, LWC 0.2); haze: row 4 (3 km)
    assert hours.loc["Summer", "cloud_n"] == 2 and hours.loc["Summer", "haze_n"] == 1
    assert hours.loc["EPCAPE", "cloud_h"] == pytest.approx(2 / 12)


# --- rows 27 and 30: activated fraction -------------------------------------------------------
def test_activated_flags_counts_class_rows_with_both_diameters():
    d = pd.DataFrame({
        "Dta": [300.0, 200.0, 400.0, np.nan, 500.0],
        "Dta_critical": [250.0, 250.0, 400.0, 100.0, 100.0],
        "ClrCldHaz_class": [1.0, 1.0, 1.0, 1.0, 2.0],
    }, index=pd.date_range("2023-07-01", periods=5, freq="min"))
    cloud = k.activated_flags(d, 1)
    assert cloud.tolist() == [1.0, 0.0, 0.0]  # 300 > 250; 200 <= 250; 400 <= 400; NaN row dropped
    assert k.activated_flags(d, 2).tolist() == [1.0]


# --- rows 25, 26, 28, 29: Abbey's native-time classes on the FM-120 rows -----------------------------
def test_abbey_class_on_intervals_majority_and_per_sample():
    # native samples: interval 00:00 mostly cloud (2 of 3 classified; one unclassified ignored),
    # interval 00:05 one haze sample, interval 00:10 nothing classified
    times = pd.DatetimeIndex(["2023-07-01 00:00:10", "2023-07-01 00:01:00", "2023-07-01 00:02:00",
                              "2023-07-01 00:03:00", "2023-07-01 00:06:00", "2023-07-01 00:11:00"])
    d = pd.DataFrame({"ClrCldHaz_class": [1.0, 1.0, 2.0, np.nan, 2.0, np.nan]}, index=times)
    starts = pd.date_range("2023-07-01 00:00", periods=3, freq="5min")
    assert k.abbey_class_on_intervals(d, starts, 1).tolist() == [True, False, False]
    assert k.abbey_class_on_intervals(d, starts, 2).tolist() == [False, True, False]
    nd = pd.Series([200.0, 30.0, 5.0], index=starts, name="nd_cm3")
    per = k.abbey_class_per_sample(d, nd, 1)  # each cloud sample takes its interval's value
    assert per.tolist() == [200.0, 200.0] and list(per.index) == list(times[:2])


# --- rows 33-42: classing the GCVI-AMS samples --------------------------------------------------------
def test_segment_of_class_at_and_segment_majority():
    segs = pd.DataFrame({"start": pd.to_datetime(["2023-07-01 00:00", "2023-07-01 01:00"]),
                         "end": pd.to_datetime(["2023-07-01 00:30", "2023-07-01 01:30"])})
    t = pd.DatetimeIndex(["2023-07-01 00:10", "2023-07-01 00:45", "2023-07-01 01:30"])
    assert k.segment_of(t, segs).tolist() == [0, -1, 1]  # end time included
    # FM-120 rows: 00:05 cloud; an AMS sample at 00:04 (V-mode middle 00:05) takes that row's class
    rows = pd.Series([False, True, False], index=pd.date_range("2023-07-01 00:00", periods=3, freq="5min"))
    ams_t = pd.DatetimeIndex(["2023-07-01 00:02", "2023-07-01 00:04", "2023-07-01 02:00"])
    assert k.class_at(ams_t, rows).tolist() == [False, True, False]  # 02:00 has no FM-120 row
    # Abbey's samples: segment 0 mostly cloud, segment 1 mostly haze
    d = pd.DataFrame({"ClrCldHaz_class": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0]},
                     index=pd.to_datetime(["2023-07-01 00:01", "2023-07-01 00:02", "2023-07-01 00:03",
                                           "2023-07-01 01:01", "2023-07-01 01:02", "2023-07-01 01:03"]))
    assert k.segment_majority_class(d, segs, 1).tolist() == [True, False]
    assert k.segment_majority_class(d, segs, 2).tolist() == [False, True]


# --- Abbey's 15-min file (Kavin's AMSforAndrew.mlx) -----------------------------------------------------------------
def test_kavin_15min_hours_effective_diameter_and_class_of_times():
    t = pd.date_range("2023-04-30 23:30", periods=4, freq="15min")  # two intervals in April, two in May
    cls = pd.Series([1.0, 2.0, 1.0, np.nan], index=t)
    h = k.kavin_15min_hours(cls)
    assert h.loc["Spring", "cloud_h"] == 0.5 and h.loc["Spring", "haze_h"] == 0.25 and h.loc["EPCAPE", "cloud_n"] == 2
    # effective diameter: one bin -> its diameter; two equal-N bins at 2 and 4 um -> (8 + 64) / (4 + 16) = 3.6
    dsd = pd.DataFrame([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], columns=[2.0, 4.0])
    d = k.effective_diameter_um(dsd, np.array([2.0, 4.0]))
    assert d.iloc[0] == pytest.approx(2.0) and d.iloc[1] == pytest.approx(3.6) and np.isnan(d.iloc[2])
    # class of arbitrary times = class of the 15-min interval holding them
    times = pd.DatetimeIndex(["2023-04-30 23:44:59", "2023-04-30 23:45:00", "2023-05-01 02:00"])
    assert k.class_of_times(times, cls)[:2].tolist() == [1.0, 2.0] and np.isnan(k.class_of_times(times, cls)[2])
