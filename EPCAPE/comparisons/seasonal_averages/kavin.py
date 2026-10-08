"""Kavin Pugazhenthi's calculations for the EPCAPE seasonal-averages table, re-run in Python.

Kavin shared his MATLAB live script (``ForAndrewVariablesmlx.mlx``, 2026-10-07) and the
input files it reads (``EPCAPE/Kavin_data_for_BAMS_paper/``, git-ignored). This module
re-implements each section of that script line by line, so every number in the sheet
can be traced to its input file. Where a quirk of the code changes a number, a
corrected variant runs on the same data, which separates "different method" from
"different data" when the result is set next to this check's own values.

  Live-script section                                  Sheet rows  Functions
  "Dan LW SW Code"                                     7, 8        read_lubin_sw, kavin_sw_stats,
                                                                   lubin_sw_series, read_lubin_lw_medians,
                                                                   kavin_lw_stats, monthly_hourly_medians
  "LWP Code"                                           10          read_kavin_lwp, kavin_lwp_box,
                                                                   kavin_lwp_stats
  "Hours Calculations (Big File)"                      31, 32      kavin_interval_visibility,
                                                                   kavin_fm120_hours
  "Critical Diameters & % Activated Code from Abbey"   27, 30      read_dcrit_native, activated_flags
  (same section, lines 11-12: cloud/haze classes)      25, 26,     abbey_class_on_intervals,
                                                       28, 29      abbey_class_per_sample
  (Abbey's classes applied to the GCVI-AMS samples)    33-42       class_at, segment_of,
                                                                   segment_majority_class
  AMSforAndrew.mlx (Abbey's 15-min file and classes)  25, 26, 28, read_kavin_15min, kavin_15min_hours,
                                                       29, 33-42,  effective_diameter_um, class_of_times
                                                       54, 55

The MATLAB lines being reproduced are quoted in comments, prefixed with ``%``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from EPCAPE.comparisons.seasonal_averages.seasonal import SEASON_ORDER

MONTH_ABBR = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# Kavin's seasons in the SW/LW and FM-120 sections: calendar quarters of any year.
# Winter is Jan-Mar, not the sheet's header "FM23+JF24" (the campaign has data in
# Feb 2023 from the 15th, Mar 2023, Jan 2024 and Feb 2024 to the 14th).
KAVIN_QUARTERS: Dict[str, Tuple[int, ...]] = {
    "Spring": (4, 5, 6),
    "Summer": (7, 8, 9),
    "Fall": (10, 11, 12),
    "Winter": (1, 2, 3),
}

# Dan Lubin's tables label hours in Pacific Standard Time (UTC-8). Shown in the notebook
# (Section 1b): his hourly SW transmittance correlates best with RADFLUX when shifted by
# 8 h, and his LW medians match RADFLUX's diurnal cycle best with the same shift.
LUBIN_UTC_OFFSET_H = 8


def _finite_stats(values: np.ndarray, ddof: int = 1) -> Dict[str, float]:
    """mean, std (sample, ddof = 1 like MATLAB's std), median, n of the finite values."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    return {
        "mean": v.mean() if v.size else np.nan,
        "std": v.std(ddof=ddof) if v.size > ddof else np.nan,
        "median": np.median(v) if v.size else np.nan,
        "n": int(v.size),
    }


def quarter_stats(values: pd.Series, *, utc_offset_h: float = LUBIN_UTC_OFFSET_H) -> pd.DataFrame:
    """Mean, std, median, n per KAVIN_QUARTERS season, plus EPCAPE = every value.

    values: UTC-indexed Series. The calendar month is taken in local time
    (UTC - utc_offset_h), as Lubin's monthly tables are. Rows = SEASON_ORDER."""
    v = pd.Series(values).astype(float)
    month = (v.index - pd.Timedelta(hours=utc_offset_h)).month
    rows = {"EPCAPE": _finite_stats(v.to_numpy())}
    for season, months in KAVIN_QUARTERS.items():
        rows[season] = _finite_stats(v.to_numpy()[np.isin(month, months)])
    return pd.DataFrame.from_dict(rows, orient="index").loc[SEASON_ORDER]


# =============================================================================
# Rows 7 and 8: Dan Lubin's SW transmittance and LW tables ("Dan LW SW Code")
# =============================================================================
def read_lubin_sw(folder: Path) -> Dict[int, pd.DataFrame]:
    """Dan Lubin's hourly SW transmittance, one table per calendar month.

    Files <MON>_SW_Boxplot.csv: a header line "0,1,...,23,Means", then one line per day
    of the month (day 1 first). Columns 0-23 are the hour of day in PST (UTC-8); each
    value is that hour's transmittance (measured SW down over a clear-sky model; sheet
    row 7's note names Atwater & Ball 1981). Night hours are empty, except on day 1,
    where every night hour holds -0.05: a plotting placeholder, not a measurement.
    The "Means" column is ignored, as in Kavin's code.

    Returns {month number: DataFrame(index = day of month 1.., columns = hour 0..23)}
    for the files present; values as stored, placeholders included."""
    tables = {}
    for month, abbr in enumerate(MONTH_ABBR, start=1):
        path = Path(folder) / f"{abbr}_SW_Boxplot.csv"
        if not path.is_file():
            continue
        # skiprows=1: the header line; utf-8-sig: the files start with a byte-order mark
        df = pd.read_csv(path, header=None, skiprows=1, encoding="utf-8-sig")
        hours = df.iloc[:, :24].apply(pd.to_numeric, errors="coerce")
        hours.index = pd.RangeIndex(1, len(hours) + 1, name="day")
        hours.columns = pd.RangeIndex(0, 24, name="hour_pst")
        tables[month] = hours
    return tables


def kavin_sw_cells(table: pd.DataFrame, month: int) -> np.ndarray:
    """The cells of one month that Kavin's code keeps, exactly as MATLAB reads them.

      % sepSW = SEP_SW_Boxplot{2:31,["Var2","Var3",...,"Var24"]};
      % sepSW(:,24) = nan(30,1);
      % febSW = FEB_SW_Boxplot{2:29, ...};   febSW(:,24) = nan(28,1);

    The Var1..Var25 names show that readtable did not use the header line for names,
    so it read "0,1,...,23,Means" as table row 1. Table rows 2:31 are therefore days
    1-30, and day 31 of a 31-day month is dropped. Var2..Var24 are hours 1-23, so
    hour 0 (night in PST) is dropped, and column 24 is all NaN. The day-1 -0.05
    placeholders are kept and enter the means. (The other reading, header used for
    names, would make rows 2:31 = days 2-31; it gives fall 0.68 ± 0.21 instead of the
    sheet's 0.66 ± 0.24, which this reading reproduces.)

    Returns an array (days, 24): 30 days (28 in February) x hours 1-23 + a NaN column."""
    n_days = 28 if month == 2 else 30
    cells = table.reindex(index=range(1, n_days + 1), columns=range(1, 24)).to_numpy(float)
    return np.hstack([cells, np.full((n_days, 1), np.nan)])


def corrected_sw_cells(table: pd.DataFrame) -> np.ndarray:
    """Every day and every hour of the month, with the -0.05 placeholders removed.

    No transmittance in the files is genuinely negative; the only negative value
    is the -0.05 placeholder."""
    cells = table.to_numpy(float)
    return np.where(cells < 0, np.nan, cells)


def kavin_sw_stats(sw: Mapping[int, pd.DataFrame], *, exact: bool = True) -> pd.DataFrame:
    """Seasonal statistics of Lubin's hourly transmittance, as Kavin's code computes them.

      % sprSWavg = mean([aprSW; maySW; junSW],3,'omitmissing');
      % sprSWavgmean = mean(sprSWavg,'all','omitmissing');
      % sprSWavgstd = std(sprSWavg,[],'all','omitmissing');
      % AnnualSWmean = mean([janSW;febSW;...;decSW],'all','omitmissing')

    A mean over dimension 3 of a 2-D array returns the array unchanged, so each season
    is the mean and sample std over every (day, hour) cell of its three stacked months.
    Each daylight hour counts once, at any sun elevation. Seasons are calendar quarters
    (KAVIN_QUARTERS); EPCAPE uses all twelve months.

    exact=True uses Kavin's cells (kavin_sw_cells); exact=False uses every day and hour
    with the placeholders removed (corrected_sw_cells).

    A season with a missing month file can't be reproduced: its mean, std and median are
    NaN and n is 0. partial_mean, partial_std and months_missing report what the
    available months give.
    Returns a DataFrame with rows SEASON_ORDER."""
    seasons = {"EPCAPE": tuple(range(1, 13)), **KAVIN_QUARTERS}
    rows = {}
    for season, months in seasons.items():
        have = [m for m in months if m in sw]
        missing = [MONTH_ABBR[m - 1] for m in months if m not in sw]
        cells = [kavin_sw_cells(sw[m], m) if exact else corrected_sw_cells(sw[m]) for m in have]
        part = _finite_stats(np.vstack(cells)) if cells else _finite_stats(np.array([]))
        full = part if not missing else {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
        rows[season] = {**full, "partial_mean": part["mean"], "partial_std": part["std"],
                        "partial_n": part["n"], "months_missing": ", ".join(missing)}
    return pd.DataFrame.from_dict(rows, orient="index").loc[SEASON_ORDER]


def lubin_sw_series(
    sw: Mapping[int, pd.DataFrame], *, utc_offset_h: float = LUBIN_UTC_OFFSET_H
) -> pd.Series:
    """Lubin's hourly transmittance as one Series indexed by the hour's START in UTC.

    Placeholders are removed. Hours are PST, so UTC = local + utc_offset_h. Years:
    Mar-Dec 2023 and Jan 2024. February holds days from two years in one table; days
    1-14 are taken as 2024 and 15-28 as 2023, matching the campaign (2023-02-15 to
    2024-02-14). That is an assumption; the file doesn't say."""
    parts = []
    for month, table in sw.items():
        cells = corrected_sw_cells(table)
        days = np.asarray(table.index)
        for i, day in enumerate(days):
            year = 2024 if (month == 1 or (month == 2 and day < 15)) else 2023
            try:
                start = pd.Timestamp(year, month, int(day))
            except ValueError:  # e.g. a 31st row in a 30-day month: no such date
                continue
            times = start + pd.to_timedelta(np.arange(24) + utc_offset_h, unit="h")
            parts.append(pd.Series(cells[i], index=times))
    out = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return out.dropna().rename("lubin_sw_transmittance")


def hourly_ratio(num: pd.Series, den: pd.Series, *, min_count: int = 30) -> pd.Series:
    """Hourly <num> / <den> over the samples where both are finite and den > 0.

    Hour bins are [h, h + 1) on the series' own time stamps (pass START-of-sample
    times). Hours with fewer than min_count usable samples are NaN. With 1-min RADFLUX
    data this gives one transmittance per hour, at the resolution of Lubin's table."""
    ok = num.notna() & den.notna() & (den > 0)
    n = num.where(ok).resample("1h").count()
    ratio = num.where(ok).resample("1h").mean() / den.where(ok).resample("1h").mean()
    return ratio.where(n >= min_count)


def read_lubin_lw_medians(folder: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Lubin's monthly median of each hour's LW down (W m-2), from whichever files exist.

    Kavin's code reads the medians from the 3-month files (columns Hour, then one per month):
      % decLW = DJF_LW_Stepplot{:,"DEC"};  marLW = table2array(MAM_LW_Stepplot(:,"MAR")); ...
    The per-month files <MON>_LW_Stepplot.csv (Hour, Min, 25%ile, Median, 75%ile, Max)
    hold the same medians. They are identical for SEP, OCT and NOV in
    SON_LW_Stepplot.csv, so they stand in for 3-month files that are missing. Where
    both exist they must agree, or a ValueError is raised.

    Returns (DataFrame(index = hour 0..23 PST, columns = month numbers present),
             {month number: name of the file it came from})."""
    folder = Path(folder)
    medians: Dict[int, np.ndarray] = {}
    source: Dict[int, str] = {}
    for name in ("DJF", "MAM", "JJA", "SON"):  # what Kavin's code reads
        path = folder / f"{name}_LW_Stepplot.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig").set_index("Hour").sort_index()
        for col in df.columns:
            month = MONTH_ABBR.index(col.strip().upper()) + 1
            medians[month] = df[col].to_numpy(float)
            source[month] = path.name
    for month, abbr in enumerate(MONTH_ABBR, start=1):
        path = folder / f"{abbr}_LW_Stepplot.csv"
        if not path.is_file():
            continue
        med = pd.read_csv(path, encoding="utf-8-sig").set_index("Hour").sort_index()["Median"].to_numpy(float)
        if month in medians:
            if not np.allclose(medians[month], med, rtol=0, atol=1e-6, equal_nan=True):
                raise ValueError(f"{path.name} 'Median' differs from {abbr} in {source[month]}")
            continue
        medians[month] = med
        source[month] = path.name
    table = pd.DataFrame(medians, index=pd.RangeIndex(0, 24, name="hour_pst"))
    return table[sorted(table.columns)], source


def kavin_lw_stats(medians: pd.DataFrame) -> pd.DataFrame:
    """Seasonal LW statistics as Kavin's code computes them from monthly hourly medians.

      % falLWavg = mean([octLW, novLW, decLW], 2,'omitmissing');      (24 hourly values)
      % falLWavgmean = mean(falLWavg,'all','omitmissing');
      % falLWavgstd = std(falLWavg,[],'all','omitmissing');
      % AnnualLWmean = mean([janLW,...,decLW],"all",'omitmissing');  AnnualLWstd = std(..., "all")

    A season's value is the mean of its months' median diurnal cycles. Its std is the
    hour-to-hour spread of that 24-hour cycle, NOT the variability of LW from day to
    day. The campaign mean and std are over all 12 x 24 monthly hourly medians.

    medians: DataFrame(index = hour 0..23, columns = month numbers), as from
    read_lubin_lw_medians or monthly_hourly_medians. Seasons with a missing month get
    NaN, with partial_* columns as in kavin_sw_stats.
    Returns a DataFrame with rows SEASON_ORDER."""
    seasons = {"EPCAPE": tuple(range(1, 13)), **KAVIN_QUARTERS}
    rows = {}
    for season, months in seasons.items():
        have = [m for m in months if m in medians.columns]
        missing = [MONTH_ABBR[m - 1] for m in months if m not in medians.columns]
        if not have:
            part = _finite_stats(np.array([]))
        elif season == "EPCAPE":
            part = _finite_stats(medians[have].to_numpy())  # all hours x months
        else:
            cycle = medians[have].mean(axis=1, skipna=True)  # 'omitmissing' mean across months
            part = _finite_stats(cycle.to_numpy())
        full = part if not missing else {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
        rows[season] = {**full, "partial_mean": part["mean"], "partial_std": part["std"],
                        "partial_n": part["n"], "months_missing": ", ".join(missing)}
    return pd.DataFrame.from_dict(rows, orient="index").loc[SEASON_ORDER]


def monthly_hourly_medians(
    values: pd.Series, *, utc_offset_h: float = LUBIN_UTC_OFFSET_H
) -> pd.DataFrame:
    """Lubin's LW statistic applied to another record.

    First the hourly means, with the hour of day in PST (UTC - utc_offset_h). Then,
    for each calendar month, the median of each hour across the month's days.
    values: UTC-indexed Series whose time stamps mark the START of each sample.
    Returns DataFrame(index = hour 0..23 PST, columns = month number); February
    pools both campaign years, as one table per calendar month does."""
    local = values.copy()
    local.index = local.index - pd.Timedelta(hours=utc_offset_h)
    hourly = local.resample("1h").mean().dropna()
    table = hourly.groupby([hourly.index.hour, hourly.index.month]).median().unstack()
    table.index.name = "hour_pst"
    return table


# =============================================================================
# Row 10: liquid water path ("LWP Code")
# =============================================================================
KAVIN_LWP_FILE = "epcmwrret_turnM1_hourly_averages.nc"

# Column j (1-based, as in MATLAB) of Kavin's yearlylwpbox holds this (year, month):
#   % for i = 1:11: yearlylwpbox(1:heldsize,i) = lwp7913(month(d7913) == i+1 & year(d7913) == 2023)
#   % for i = 1:2:  yearlylwpbox(1:heldsize,i+11) = lwp7913(month(d7913) == i & year(d7913) == 2024)
KAVIN_LWP_MONTHS: List[Tuple[int, int]] = [(2023, m) for m in range(2, 13)] + [(2024, 1), (2024, 2)]

# Columns that reproduce the sheet's row 10 (found by trying every set of up to four
# columns): 2:4 = Mar-May, 5:7 = Jun-Aug, 8:10 = Sep-Nov, 11:13 = Dec-Feb, the
# meteorological seasons, not the sheet's AMJ/JAS/OND/FM+JF headers. Campaign, spring,
# summer and fall match to the sheet's precision; the winter mean matches (33.2 vs 33)
# but its std does not (121 vs 92). The only column sets that give 33 ± 92 make no sense
# as a season (e.g. Jul + Oct + Nov + Jan 2024: 32.2 ± 92.1).
KAVIN_LWP_COLUMNS_SHEET: Dict[str, Tuple[int, ...]] = {
    "EPCAPE": tuple(range(1, 14)),
    "Spring": (2, 3, 4),
    "Summer": (5, 6, 7),
    "Fall": (8, 9, 10),
    "Winter": (11, 12, 13),
}
# The columns in the live script as shared:
#   % mean(yearlylwpbox(:,4:6)) ... (:,7:9) ... (:,10:12) ... (:,1:3)
# Column j holds month j+1, so these are May-Jul, Aug-Oct, Nov-Jan and Feb-Apr.
# They look like an attempt at Apr-Jun, Jul-Sep, Oct-Dec and Jan-Mar, written as if
# column j held month j (my inference). They do not reproduce the sheet.
KAVIN_LWP_COLUMNS_SHARED: Dict[str, Tuple[int, ...]] = {
    "EPCAPE": tuple(range(1, 14)),
    "Spring": (4, 5, 6),
    "Summer": (7, 8, 9),
    "Fall": (10, 11, 12),
    "Winter": (1, 2, 3),
}


def read_kavin_lwp(folder: Path) -> pd.DataFrame:
    """Kavin's hourly LWP file: hourly means of MWRRET v2 phys_lwp (epcmwrret2turnM1.c1).

    File attributes: "Hourly averages of epcmwrret2turnM1.c1"; time = hour START
    (UTC); lwp_count = "number of good LWP samples per hour". The notebook (Section 3b)
    rebuilds it from the ARM files: phys_lwp is the mean of the QC = 0 samples with
    LWP > 0, and lwp_count is the number of ALL QC = 0 samples. There is no cloud or
    rain screening, so clear-sky hours enter with small positive values (negative
    clear-sky noise is dropped, zero-mean noise is not) and wet-radome hours with
    large ones.
    Returns DataFrame(index = hour start UTC; phys_lwp_gm2, lwp_count)."""
    path = Path(folder) / KAVIN_LWP_FILE
    with xr.open_dataset(path, decode_times=False) as ds:
        units = ds["phys_lwp"].attrs.get("units", "")
        if units not in ("g/m^2", "g m-2", "g/m2"):
            raise ValueError(f"{path.name}: unexpected phys_lwp units {units!r}")
        time = pd.to_datetime(ds["time"].values, unit="s")  # seconds since 1970-01-01 UTC
        return pd.DataFrame(
            {"phys_lwp_gm2": ds["phys_lwp"].values.astype(float), "lwp_count": ds["lwp_count"].values},
            index=pd.DatetimeIndex(time, name="time"),
        )


def kavin_lwp_box(
    lwp_gm2: pd.Series,
    *,
    tz: str = "America/Los_Angeles",
    initial_rows: int = 500,
    zero_pad: bool = True,
) -> np.ndarray:
    """MATLAB's ``yearlylwpbox``, built exactly as Kavin's code builds it.

      % d7913 = datetime(d7913,'TimeZone','America/Los_Angeles');   months in Pacific time
      % yearlylwpbox = nan(500,13);
      % for i = 1:11
      %     heldsize = size(lwp7913(month(d7913) == i+1 & year(d7913) == 2023),1);
      %     yearlylwpbox(1:heldsize,i) = lwp7913(month(d7913) == i+1 & year(d7913) == 2023);
      % end
      % (then Jan and Feb 2024 into columns 12 and 13)

    Column j (1-based) holds month KAVIN_LWP_MONTHS[j-1]. Most months hold more than
    500 hourly values. Assigning past the last row makes MATLAB grow the array, and it
    fills the new elements of every OTHER column with 0 (MathWorks documentation,
    "Creating, Concatenating, and Expanding Matrices": MATLAB pads with zeros to keep
    the matrix rectangular). Those zeros are not missing values, so 'omitmissing'
    keeps them in the means. So in the final array, column j has its n_j values, then
    NaN up to row 500, then 0 down to the longest month's length. zero_pad=False puts
    NaN there instead, which is what the code intends. NaN hours in the file stay NaN
    either way.

    Returns an array (rows, 13)."""
    utc = lwp_gm2.index if lwp_gm2.index.tz is not None else lwp_gm2.index.tz_localize("UTC")
    local = utc.tz_convert(tz)
    columns = [lwp_gm2.to_numpy(float)[(local.year == y) & (local.month == m)] for y, m in KAVIN_LWP_MONTHS]
    n_rows = max(initial_rows, max(len(c) for c in columns))
    box = np.full((n_rows, len(columns)), np.nan)
    for j, col in enumerate(columns):
        box[: len(col), j] = col
        if zero_pad:
            box[max(len(col), initial_rows):, j] = 0.0
    return box


def kavin_lwp_stats(box: np.ndarray, columns: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """mean(yearlylwpbox(:,cols),'all','omitmissing') and std(...,[],'all','omitmissing').

    columns: season -> 1-based MATLAB column numbers (KAVIN_LWP_COLUMNS_SHEET or
    KAVIN_LWP_COLUMNS_SHARED). n_zero_pad counts the zeros MATLAB added. The file has
    no LWP <= 0, so every zero in the array is padding.
    Returns a DataFrame with rows SEASON_ORDER."""
    rows = {}
    for season in SEASON_ORDER:
        sub = box[:, [c - 1 for c in columns[season]]]
        rows[season] = {**_finite_stats(sub), "n_zero_pad": int(np.sum(sub == 0)),
                        "months": ", ".join(f"{MONTH_ABBR[KAVIN_LWP_MONTHS[c - 1][1] - 1].title()} "
                                            f"{KAVIN_LWP_MONTHS[c - 1][0]}" for c in columns[season])}
    return pd.DataFrame.from_dict(rows, orient="index").loc[SEASON_ORDER]


# =============================================================================
# Rows 31-32: FM-120 cloud and haze hours ("Hours Calculations (Big File)")
# =============================================================================
def kavin_interval_visibility(starts: pd.DatetimeIndex, vis_files: Sequence[pd.Series]) -> pd.Series:
    """Visibility (m) of each FM-120 row as Kavin's code computes it (``avgvis``).

      % avgvis = zeros(90011,1);
      % for i = 1:36951
      %     avgvis(i) = mean(tbl1min.visibility_m_(tbl1min.datetime_GMT_ <= t5min(i+1) & ...
      %                      tbl1min.datetime_GMT_ >= t5min(i)),'omitmissing');
      % end
      % avgvis(36952) = 0;
      % for i = 36953:90010
      %     avgvis(i) = mean(tbl2min.visibility_m_( ... same window ... ),'omitmissing');
      % end

    For each row this is the MEAN of the raw ~1-s samples from the row's time to the NEXT
    row's time, both ends included. Across an FM-120 gap the window therefore spans the
    whole gap. Zeros and spikes are not screened out (the record holds values up to ~1e8 m).
    Rows before the second file starts read the first file. The row at the second file's
    first sample, floored to 5 min (2023-07-19 04:00 UTC, MATLAB row 36952), is set to
    0 m. Later rows read the second file. The last row keeps its initial 0 m. A window
    with no samples gives NaN (MATLAB's mean of an empty set).

    starts: FM-120 row times (UTC), sorted. vis_files: the two visibility files as read
    by sources.load_visibility_msd(separate=True).
    Returns a Series on `starts`."""
    if len(vis_files) != 2:
        raise ValueError(f"Kavin's code reads exactly two visibility files, got {len(vis_files)}")
    t = starts.to_numpy()
    out = np.full(t.size, np.nan)

    def window_means(vis: pd.Series, lo_t: np.ndarray, hi_t: np.ndarray) -> np.ndarray:
        """mean of vis samples with lo_t <= time <= hi_t, NaN samples ignored, via cumulative sums."""
        ts = vis.index.to_numpy()
        x = vis.to_numpy(float)
        ok = np.isfinite(x)
        csum = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
        ccount = np.concatenate([[0], np.cumsum(ok)])
        lo = np.searchsorted(ts, lo_t, side="left")  # first sample >= start
        hi = np.searchsorted(ts, hi_t, side="right")  # one past the last sample <= end
        n = ccount[hi] - ccount[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (csum[hi] - csum[lo]) / n, np.nan)

    split_time = np.datetime64(vis_files[1].index.min().floor("5min"))
    split = int(np.searchsorted(t, split_time, side="left"))  # first row on or after it
    rows_1 = np.arange(0, min(split, t.size - 1))
    rows_2 = np.arange(split + 1, t.size - 1)
    out[rows_1] = window_means(vis_files[0], t[rows_1], t[rows_1 + 1])
    out[rows_2] = window_means(vis_files[1], t[rows_2], t[rows_2 + 1])
    if split < t.size:
        out[split] = 0.0  # % avgvis(36952) = 0;
    out[-1] = 0.0  # never assigned: stays at its zeros(...) initial value
    return pd.Series(out, index=starts, name="kavin_avgvis_m")


def kavin_fm120_hours(lwc_gm3: pd.Series, avgvis_m: pd.Series) -> pd.DataFrame:
    """Cloud and haze hours per KAVIN_QUARTERS season, counted as Kavin's code counts them.

      % totalcloudhours = sum(avgvis < 1000 & LWC > 0.01)/12
      % totalhazehours  = sum(avgvis < 5000 & avgvis > 1000)/12
      % summercloudhours = sum(avgvis < 1000 & LWC > 0.01 & month(t5min) >= 7 & month(t5min) <= 9)/12
      % (and the same for months 4-6, 10-12 and 1-3)

    Each FM-120 row is 5 min (hence /12). Haze has no LWC condition. Every row counts,
    including those after 2024-02-14. The months are those of the UTC time stamps.
    (The script also defines suindex, faindex and wiindex with impossible conditions,
    e.g. month <= 7 & month >= 9, but never uses them; spindex is correct.)

    The sheet holds these values truncated to whole hours, and its campaign cell is
    the sum of the four truncated seasons. The notebook shows both.
    Returns DataFrame(rows = SEASON_ORDER; cloud_h, haze_h, cloud_n, haze_n)."""
    lwc = lwc_gm3.reindex(avgvis_m.index)
    cloud = (avgvis_m < 1000) & (lwc > 0.01)
    haze = (avgvis_m < 5000) & (avgvis_m > 1000)
    month = avgvis_m.index.month
    rows = {"EPCAPE": np.ones(month.size, bool)}
    rows.update({s: np.isin(month, m) for s, m in KAVIN_QUARTERS.items()})
    out = {s: {"cloud_h": cloud[sel].sum() / 12, "haze_h": haze[sel].sum() / 12,
               "cloud_n": int(cloud[sel].sum()), "haze_n": int(haze[sel].sum())} for s, sel in rows.items()}
    return pd.DataFrame.from_dict(out, orient="index").loc[SEASON_ORDER]


# =============================================================================
# Rows 27 and 30: activated fraction ("Critical Diameters & % Activated Code from Abbey")
# =============================================================================
DCRIT_NATIVE_FILE = "ET_Dcrit_nativetime.csv"
DCRIT_15MIN_FILE = "ET_Dcrit_15min.csv"
CLASS_CODES = {1: "cloud", 2: "haze"}  # ClrCldHaz_class, per Kavin's code (indexcld == 1, indexhaz == 2)


def read_dcrit_native(folder: Path) -> pd.DataFrame:
    """ET_Dcrit_nativetime.csv (from Abbey Williams' code, via Kavin), one row per sample.

    Columns: Dta, Dta_critical, Dta_critical_wet_fKDd, ClrCldHaz_class (1 = cloud,
    2 = haze, empty = neither). The diameters look like nm: Kavin divides the 15-min
    wet critical diameters by 10^3, and the wet values are 0.8-20 um after that. The
    files do not define Dta and Dta_critical; I don't have a source for their meaning.
    Times are as written ("10-Mar-2023 22:38:16"), presumably UTC. Repeated time
    stamps and repeated rows are kept, as in Kavin's code.

    Limited data: the rows run from 2023-03-10 to 2024-01-31, about 1-4 s apart, mostly
    while the GCVI sampled. Dta_critical is present in only 12,282 of the 114,528 rows,
    all in Jun-Sep and Dec 2023 (95 % in Jun-Aug). Anything built on Dta_critical (rows
    27 and 30) therefore says almost nothing about spring, fall or winter. The class
    column covers the whole record (Mar 2023 - Jan 2024), so the droplet rows (25, 26,
    28, 29) have wider coverage.
    Returns a DataFrame indexed by time."""
    path = Path(folder) / DCRIT_NATIVE_FILE
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), format="%d-%b-%Y %H:%M:%S"), name="time")
    return df


def activated_flags(dcrit: pd.DataFrame, class_code: int) -> pd.Series:
    """1.0 where Dta > Dta_critical (activated), 0.0 where Dta <= Dta_critical.

    Only rows of class `class_code` with both diameters finite are kept:
      % indexcld = find(table2array(ET_Dcrit_nativetime(:,"ClrCldHaz_class")) == 1);
      % keep_idx = ~isnan(Dtacld) & ~isnan(Dtacrit_cld);
      % NotActivatedNumber = length(find(Dtacld_clean <= Dtacrit_cld_clean));
      % ActivatedNumber = length(find(Dtacld_clean > Dtacrit_cld_clean));
      % PercentActivatedCloud = (ActivatedNumber / (ActivatedNumber + NotActivatedNumber))*100;

    The mean of the flags over any set of rows is that set's activated fraction. It
    counts samples, not time: every native-time row has equal weight, whatever its
    duration. Kavin's code computes only the campaign value; the notebook applies the
    same count within each season.

    Limited data: Dta_critical exists only in Jun-Sep and Dec 2023, so the flags exist
    only there. "Spring" is June alone (941 cloud and 19 haze samples), "fall" is
    December alone (138 cloud samples, no haze), winter has none, and the campaign
    value is in effect a Jun-Aug value (95 % of the samples).
    Returns a Series indexed by time."""
    sel = dcrit[(dcrit["ClrCldHaz_class"] == class_code)
                & dcrit["Dta"].notna() & dcrit["Dta_critical"].notna()]
    return (sel["Dta"] > sel["Dta_critical"]).astype(float).rename(f"activated_{CLASS_CODES[class_code]}")


def abbey_class_on_intervals(
    dcrit: pd.DataFrame, starts: pd.DatetimeIndex, class_code: int, *, interval_min: float = 5.0
) -> pd.Series:
    """True for each FM-120 row whose interval was mostly class `class_code` in Abbey's record.

    Abbey Williams' code classifies each native-time sample of ET_Dcrit_nativetime.csv
    (ClrCldHaz_class). Kavin's live script selects the classes on its lines 11-12:
      % indexcld = find(table2array(ET_Dcrit_nativetime(:,"ClrCldHaz_class")) == 1);
      % indexhaz = find(table2array(ET_Dcrit_nativetime(:,"ClrCldHaz_class")) == 2);
    The native samples are ~1-4 s apart, while the FM-120 files hold 5-min means labelled
    by interval START. Each FM-120 row [t, t + interval_min) therefore takes the class
    held by MORE THAN HALF of the classified native samples inside it. Unclassified rows
    (empty ClrCldHaz_class) are ignored, and a row with no classified sample is neither
    class. In this file no interval mixes cloud and haze samples, so "majority" and "any"
    select the same rows (the notebook checks this).

    Coverage: classes exist only from 2023-03-10 to 2024-01-31, mostly while the GCVI
    sampled, so most FM-120 rows are neither class.

    dcrit: read_dcrit_native() output. starts: FM-120 row times (UTC).
    Returns a boolean Series on `starts`."""
    classified = dcrit["ClrCldHaz_class"].dropna()
    interval = classified.index.floor(pd.Timedelta(minutes=interval_min))
    share = (classified == class_code).groupby(interval).mean()  # fraction of classified samples in this class
    majority = share.index[share > 0.5]
    return pd.Series(starts.isin(majority), index=starts, name=f"abbey_{CLASS_CODES[class_code]}")


def abbey_class_per_sample(
    dcrit: pd.DataFrame, values: pd.Series, class_code: int, *, interval_min: float = 5.0
) -> pd.Series:
    """The FM-120 value of the interval containing each class-`class_code` native sample.

    Indexed by native time. Every native sample counts once, so an FM-120 row counts as
    many times as it holds class samples. That weighting follows the native sampling
    rate, not time. It is a sensitivity test for abbey_class_on_intervals, where each
    FM-120 row counts once.
    values: an FM-120 column (UTC index = interval start)."""
    times = dcrit.index[dcrit["ClrCldHaz_class"] == class_code]
    out = values.reindex(times.floor(pd.Timedelta(minutes=interval_min)))
    out.index = times
    return out.rename(f"{values.name}_per_{CLASS_CODES[class_code]}_sample")


# =============================================================================
# Rows 33-42: GCVI-AMS cloud and haze hours and residuals, split with Abbey's classes
# =============================================================================
def segment_of(times: pd.DatetimeIndex, segments: pd.DataFrame) -> np.ndarray:
    """Index (0-based row of `segments`) of the segment containing each time, or -1.

    segments: DataFrame with start and end columns (sorted, non-overlapping), e.g.
    quantities.gcvi_segments output. A time inside [start, end] belongs to it."""
    seg = segments.reset_index(drop=True)
    start, end = seg["start"].to_numpy(), seg["end"].to_numpy()
    i = np.searchsorted(start, times.to_numpy(), side="right") - 1  # last segment starting at or before t
    inside = (i >= 0) & (times.to_numpy() <= end[np.clip(i, 0, None)])
    return np.where(inside, i, -1)


def class_at(
    times: pd.DatetimeIndex, row_class: pd.Series, *, offset_min: float = 1.0, interval_min: float = 5.0
) -> np.ndarray:
    """Class of each sample, taken from the FM-120 row that holds time + offset_min.

    row_class: boolean Series on the FM-120 row START times (abbey_class_on_intervals,
    or a visibility mask). The default offset of 1 min puts an AMS sample, whose time
    stamp is the START of its 2-min V-mode period, at the period's mid-point. A sample
    whose row is missing (FM-120 off) is not in the class.
    Returns a boolean array, one value per time."""
    rows = (times + pd.Timedelta(minutes=offset_min)).floor(pd.Timedelta(minutes=interval_min))
    return row_class.reindex(rows, fill_value=False).to_numpy(dtype=bool)


def segment_majority_class(dcrit: pd.DataFrame, segments: pd.DataFrame, class_code: int) -> np.ndarray:
    """True for each segment in which more than half of Abbey's classified samples are `class_code`.

    A sensitivity test that classifies whole GCVI sampling segments instead of 5-min
    FM-120 rows. Abbey's samples lie almost all inside GCVI segments (~96 % of cloud,
    ~92 % of haze samples). Segments with no classified sample are in neither class.
    Returns a boolean array, one value per row of `segments`."""
    classified = dcrit["ClrCldHaz_class"].dropna()
    which = segment_of(classified.index, segments)
    frame = pd.DataFrame({"segment": which, "is_class": (classified == class_code).to_numpy()})
    share = frame[frame["segment"] >= 0].groupby("segment")["is_class"].mean()
    out = np.zeros(len(segments), dtype=bool)
    out[share.index[share > 0.5].to_numpy()] = True
    return out


# =============================================================================
# Abbey Williams' 15-min merged file (Kavin's AMSforAndrew.mlx): rows 25, 26, 28, 29, 33-42, 54, 55
# =============================================================================
KAVIN_15MIN_FILE = "EPCAPE_15mindata.nc"
# Order of the AMS species along the file's AMS_species dimension, i.e. the rows of Kavin's
# AMS_mass_15min_EF (Abbey, via Kavin: "Row 1 is nitrate, row 2 is sulfate, row 3 is organics,
# row 4 is ammonium, and row 5 is chloride"); read_kavin_15min checks it against the file's labels.
AMS_15MIN_SPECIES = ("nitrate", "sulfate", "organics", "ammonium", "chloride")


def read_kavin_15min(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Abbey Williams' 15-min merged EPCAPE file (EPCAPE_15mindata.nc): only the variables used here.

    Kavin's AMSforAndrew.mlx reads ClrCldHaz_class from this file, and AMS_mass_15min_EF /
    AMS_massfrac_15min_EF from EPCAPE_15min_data.mat (not provided). The file's AMS_mass equals the
    CE-corrected AMS (AMS_CE_corrected_msptof.nc) averaged over each 15-min interval while the
    CVI is off, and the CE-corrected AMS divided by EF while it is on (checked in the notebook,
    Section 10e). It reproduces every one of the sheet's residual values, so it stands in for
    AMS_mass_15min_EF.

    time: "nanoseconds since 2023-02-15" (UTC), 35,041 steps of 15 min (2023-02-15 00:00 to
    2024-02-15 00:00), each labelled by its interval START; float jitter of a few ns is rounded off.

    Returns (frame, dsd):
      frame columns  clr_cld_haz_class  1 = cloud, 2 = haze (defined only while the CVI is on),
                                        0 = CVI off, NaN = CVI in transition
                     cvi_on_class       1 = CVI on, 0 = off, 0.5 = transition
                     <species>_ugm3     AMS_mass for AMS_15MIN_SPECIES
                     <species>_frac     AMS_massfrac (0, not NaN, when the AMS has no data)
                     fm120_nd_cm3, fm120_ed_um, fm120_lwc_gm3   FM-120 number, ED, LWC
      dsd            FM120_DSD, dN per bin (cm-3); columns = FM120_diam (um), the bins' UPPER edges
                     (3, 4, ..., 50 um for the 2-3, ..., 48-50 um bins of readme_FM120.md)."""
    path = Path(folder) / KAVIN_15MIN_FILE
    with xr.open_dataset(path) as ds:
        labels = [str(s).strip().lower() for s in ds["AMS_species"].values]
        if tuple(labels) != AMS_15MIN_SPECIES:
            raise ValueError(f"{path.name}: AMS_species is {labels}, expected {AMS_15MIN_SPECIES}")
        time = pd.DatetimeIndex(ds["time"].values).round("1s")
        frame = pd.DataFrame({"clr_cld_haz_class": ds["ClrCldHaz_class"].values,
                              "cvi_on_class": ds["CVIon_class"].values,
                              "fm120_nd_cm3": ds["FM120_Nd"].values, "fm120_ed_um": ds["FM120_ED"].values,
                              "fm120_lwc_gm3": ds["FM120_LWC"].values}, index=pd.DatetimeIndex(time, name="time"))
        mass, frac = ds["AMS_mass"].values, ds["AMS_massfrac"].values  # (time, species)
        for j, species in enumerate(AMS_15MIN_SPECIES):
            frame[f"{species}_ugm3"] = mass[:, j]
            frame[f"{species}_frac"] = frac[:, j]
        dsd = pd.DataFrame(ds["FM120_DSD"].values, index=frame.index,
                           columns=pd.Index(ds["FM120_diam"].values.astype(float), name="upper_edge_um"))
    return frame, dsd


def kavin_15min_hours(clr_cld_haz_class: pd.Series) -> pd.DataFrame:
    """Cloud and haze hours as Kavin's AMSforAndrew.mlx counts them.

      % totalcldhours = sum(index == 1)/4
      % totalcldsprhours = sum(index == 1 & month(d) >= 4 & month(d) <= 6)/4   (likewise 7-9, 10-12, 1-3; haze = 2)

    Each 15-min interval classed cloud (haze) counts 0.25 h, whether or not the AMS reported in it.
    Seasons are calendar quarters of the UTC time; EPCAPE = every interval. The sheet holds these
    values ROUNDED (e.g. 13.75 -> 14, 100.5 -> 101).
    Returns DataFrame(rows = SEASON_ORDER; cloud_h, haze_h, cloud_n, haze_n)."""
    month = clr_cld_haz_class.index.month
    rows = {"EPCAPE": np.ones(month.size, bool), **{s: np.isin(month, m) for s, m in KAVIN_QUARTERS.items()}}
    cloud = (clr_cld_haz_class == 1).to_numpy()
    haze = (clr_cld_haz_class == 2).to_numpy()
    out = {s: {"cloud_h": (cloud & sel).sum() / 4, "haze_h": (haze & sel).sum() / 4,
               "cloud_n": int((cloud & sel).sum()), "haze_n": int((haze & sel).sum())} for s, sel in rows.items()}
    return pd.DataFrame.from_dict(out, orient="index").loc[SEASON_ORDER]


def effective_diameter_um(dsd: pd.DataFrame, diameters_um: np.ndarray) -> pd.Series:
    """Effective diameter of each size distribution: sum(N_i D_i^3) / sum(N_i D_i^2), in um.

    dsd: dN per bin (rows = times, columns = bins); diameters_um: the diameter given to each bin.
    Kavin's values are reproduced with the bins' UPPER edges (the file's FM120_diam); the bin
    mid-points give the physically consistent value (smaller by roughly half a bin width).
    Rows with no droplets give NaN."""
    n = dsd.to_numpy(float)  # (times, bins)
    d = np.asarray(diameters_um, float)  # (bins,)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_eff = (n * d ** 3).sum(axis=1) / (n * d ** 2).sum(axis=1)
    return pd.Series(d_eff, index=dsd.index, name="d_eff_um").where(lambda x: np.isfinite(x))


def class_of_times(times: pd.DatetimeIndex, interval_class: pd.Series, *, interval_min: float = 15.0) -> np.ndarray:
    """The 15-min class (clr_cld_haz_class) of the interval [t, t + 15 min) holding each time; NaN if none."""
    start = times.floor(pd.Timedelta(minutes=interval_min))
    return interval_class.reindex(start).to_numpy(dtype=float)
