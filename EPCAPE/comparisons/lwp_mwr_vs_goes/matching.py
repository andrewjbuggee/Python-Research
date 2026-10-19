"""Ground-based Pier records averaged over the GOES-18 image windows, and Seethala's matched constraints.

Seethala Chellappan's description of her MWR vs GOES-18 statistic (email of 19 Oct 2026, 11:21 PDT):
  d) "I averaged ground-based measurements to 30-minutes (to match GOES-18 temporal resolution). Before
     averaging, I removed the LWP data if there is a reported surface precipitation. For the KAZR, removed
     precipitating times as well as times when CTH>2km before doing the averaging."
  i) "Retrieved data for Scripps Pier grid from GOES-18 0.2-degree by 0.2-degree datafile."
  j) "Applied these conditions uniformly to both the ground based and remote sensors before computing the
     statistics: MWR3&GOES18 LWP: 10-250 g/m2, GOES18 LCF & Ceil CFF > 10%, GOES18 TCF-LCF <= 10%,
     GOES18 SZA < 70-deg, GOES18 & KAZR CTH <= 2 km, GOES18 cloud optical depth > 2."
So one matched pair = one GOES-18 image and the 30-min ground-based averages that go with it, and the
MWR and GOES-18 statistics are taken over the SAME pairs.

Every ground-based record here is averaged with ``window_means``: non-overlapping windows given by their
start times and one width, so the MWR, ceilometer and KAZR values of a pair all cover the same interval.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

LOW_CLOUD = 1.0  # value of a cloud-state series (EPCAPE.comparisons.seasonal_averages.quantities.LOW_CLOUD)


def window_means(values: pd.Series, starts: pd.DatetimeIndex, width: pd.Timedelta) -> pd.DataFrame:
    """Mean and count of the finite samples of `values` in each window [start, start + width).

    values: samples indexed by UTC time (any order; sorted here). starts: window start times (any order;
    windows may overlap, each is computed on its own). Uses cumulative sums, so it is fast for the ~22
    million MWRRET v2 samples. Returns DataFrame indexed by `starts` with columns mean (NaN for an empty
    window) and n (finite samples)."""
    starts = pd.DatetimeIndex(starts)
    values = values.sort_index()
    t_ns = values.index.asi8
    v = values.to_numpy(dtype=float)
    finite = np.isfinite(v)
    # cumulative sums with a leading 0, so the sum over samples [a, b) is c[b] - c[a]
    c_sum = np.concatenate([[0.0], np.cumsum(np.where(finite, v, 0.0))])
    c_n = np.concatenate([[0], np.cumsum(finite)])
    a = np.searchsorted(t_ns, starts.asi8, side="left")  # first sample at or after the start
    b = np.searchsorted(t_ns, (starts + width).asi8, side="left")  # first sample at or after the end
    n = c_n[b] - c_n[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c_sum[b] - c_sum[a]) / n
    return pd.DataFrame({"mean": np.where(n > 0, mean, np.nan), "n": n}, index=starts)


def median_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing of a record's time stamps (e.g. ~1.3 s MWRRET v2, 16 s ceilometer, 4 s ARSCL)."""
    return pd.Timedelta(int(np.median(np.diff(index.asi8))), unit="ns")


def image_windows(images: pd.DataFrame, *, alignment: str = "bin", width_min: float = 30.0) -> pd.DatetimeIndex:
    """Start time of the ground-based averaging window of each GOES-18 image.

    alignment
      "bin":      the 30-min bin of the clock that holds the image's nominal time ([hh:00, hh:30) or
                  [hh:30, hh+1:00)); "averaged to 30-minutes" read literally. The Pier box is scanned
                  ~1.5 min after the nominal time (scan_time), so the image lies at the START of its bin.
      "centered": [scan_time - 15 min, scan_time + 15 min): the window centred on when the satellite
                  actually viewed the box.
    """
    width = pd.Timedelta(minutes=width_min)
    if alignment == "bin":
        return pd.DatetimeIndex(images.index.floor(width))
    if alignment == "centered":
        # a box with no scan time falls back to the nominal image time
        scan = pd.Series(pd.DatetimeIndex(images["scan_time"]), index=images.index).fillna(
            pd.Series(images.index, index=images.index))
        return pd.DatetimeIndex(scan - width / 2)
    raise ValueError(f"alignment must be 'bin' or 'centered', not {alignment!r}")


def state_fraction(state: pd.Series, starts: pd.DatetimeIndex, width: pd.Timedelta, *,
                   min_coverage: float = 0.5, value: float = LOW_CLOUD) -> pd.Series:
    """Fraction of a window's known samples whose state equals `value` (e.g. ceilometer low cloud = CFF).

    Unknown (NaN) samples are left out of numerator and denominator. NaN when fewer than `min_coverage` of
    the samples the window should hold (width / median sampling interval) have a known state."""
    hit = (state == value).astype(float).where(state.notna())  # 1 / 0 / NaN
    w = window_means(hit, starts, width)
    expected = width / median_interval(state.index)
    return w["mean"].where(w["n"] >= min_coverage * expected)


def kazr_window_cth_m(highest_top_m: pd.Series, precip: np.ndarray, starts: pd.DatetimeIndex,
                      width: pd.Timedelta, *, cth_max_m: float = 2000.0) -> pd.DataFrame:
    """The 30-min KAZR-ARSCL cloud-top statistics behind "KAZR CTH <= 2 km" (her step d, then j).

    highest_top_m: top of the highest ARSCL layer per 4-s sample (m; -1 = no layer; NaN = missing), from
    EPCAPE.comparisons.seasonal_averages.quantities.arscl_highest_top_m. precip: True where the sample is
    precipitating (removed first). Returns, per window:
      cth_low_mean_m   mean top of the non-precipitating samples with a layer topped at <= cth_max_m: her
                       step d ("removed precipitating times as well as times when CTH > 2 km before doing
                       the averaging"). Finite exactly when at least one such sample exists, and then <=
                       cth_max_m by construction, so "KAZR CTH <= 2 km" read literally = "finite".
      cth_all_mean_m   mean top of every non-precipitating sample with a layer (tops above 2 km included):
                       the alternative reading, where the 30-min mean itself must be <= 2 km.
      n_low, n_layer   sample counts behind the two means.
    """
    top = highest_top_m.where(~np.asarray(precip, dtype=bool))
    layer = top.where(top >= 0)  # samples with an ARSCL layer
    low = layer.where(layer <= cth_max_m)
    w_low = window_means(low, starts, width)
    w_all = window_means(layer, starts, width)
    return pd.DataFrame({"cth_low_mean_m": w_low["mean"], "n_low": w_low["n"],
                         "cth_all_mean_m": w_all["mean"], "n_layer": w_all["n"]}, index=starts)


def constraint_masks(pairs: pd.DataFrame, *, lwp_range_gm2=(10.0, 250.0), cf_min: float = 0.10,
                     nonlow_max: float = 0.10, sza_max_deg: float = 70.0, cth_max_km: float = 2.0,
                     cod_min: float = 2.0, kazr_rule: str = "literal") -> Dict[str, np.ndarray]:
    """Seethala's constraints (email of 19 Oct 2026, item j), one boolean mask per constraint, in her order.

    pairs: one row per GOES-18 image with the columns of the notebook's pair table (goes_*, mwr_lwp_gm2,
    ceil_cff, kazr_*). Her inequalities are kept as written: LWP inclusive range, "> 10%" strict, "<= 10%",
    "< 70-deg" strict, "<= 2 km", "> 2" strict. NaN fails every test.
    kazr_rule: "literal" (a window mean of tops <= 2 km exists; see kazr_window_cth_m), "mean_all" (the mean of
    all tops <= 2 km) or "none" (KAZR not used)."""
    lo, hi = lwp_range_gm2
    m = {
        "MWR LWP 10-250 g m-2": pairs["mwr_lwp_gm2"].between(lo, hi).to_numpy(),
        "GOES-18 LWP 10-250 g m-2": pairs["goes_lwp_gm2"].between(lo, hi).to_numpy(),
        "GOES-18 LCF > 10 %": (pairs["goes_lcf"] > cf_min).to_numpy(),
        "ceilometer CFF > 10 %": (pairs["ceil_cff"] > cf_min).to_numpy(),
        "GOES-18 TCF - LCF <= 10 %": ((pairs["goes_tcf"] - pairs["goes_lcf"]) <= nonlow_max).to_numpy(),
        "GOES-18 SZA < 70 deg": (pairs["goes_sza_deg"] < sza_max_deg).to_numpy(),
        "GOES-18 CTH <= 2 km": (pairs["goes_cth_km"] <= cth_max_km).to_numpy(),
    }
    if kazr_rule == "literal":
        m["KAZR CTH <= 2 km"] = np.isfinite(pairs["kazr_cth_low_mean_m"]).to_numpy()
    elif kazr_rule == "mean_all":
        m["KAZR CTH <= 2 km"] = (pairs["kazr_cth_all_mean_m"] <= 1000.0 * cth_max_km).to_numpy()
    elif kazr_rule != "none":
        raise ValueError(f"kazr_rule must be 'literal', 'mean_all' or 'none', not {kazr_rule!r}")
    m["GOES-18 COD > 2"] = (pairs["goes_cod"] > cod_min).to_numpy()
    return m


def all_constraints(masks: Dict[str, np.ndarray], *, skip=()) -> np.ndarray:
    """True where every mask (except those named in `skip`) is True."""
    keep: Optional[np.ndarray] = None
    for name, m in masks.items():
        if name in skip:
            continue
        keep = m.copy() if keep is None else keep & m
    return keep
