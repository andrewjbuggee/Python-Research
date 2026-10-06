"""Put the three products on common sample times for pairwise comparison.

The products sample very differently:

    MFRSR   every 20 s, daytime, hemispheric diffuse field (whole sky dome)
    SPHOT   one retrieval per 5-15 min, daytime, only when cloud blocks the sun;
            1.2 deg zenith field of view (~10 m across at 500 m cloud base)
    MWR     every few seconds, day and night; ~5 deg field of view (~50 m)

(Footprint diameter ~ 2 * h * tan(FOV / 2): for h = 500 m, 1.2 deg -> 10.5 m
and 5.9 deg -> 51.5 m. The MFRSR diffuse field integrates over the whole
cloud deck within ~ a few cloud-base heights, i.e. ~ km scales.)

So the sparse sunphotometer sets the comparison times. Around each SPHOT
sample time, the 20 s MFRSR and the MWR are averaged over a window of
+/- ``half_width`` (default 2.5 min, comparable to the < 5 min cloud-mode
cycle and the MFRSRCLDOD 5-min running mean). The within-window standard
deviation and the fraction of valid samples come along. Both say how steady
and how overcast the scene was, and both explain disagreement.

All functions take *masked* inputs: values that failed an instrument's
criteria must already be NaN (see ``EPCAPE.analysis_tools.filters.apply``). Window
statistics then count only valid samples, and ``coverage`` = valid / all
samples in the window measures how much of the window passed.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import xarray as xr

LWP_FACTOR_HOMOGENEOUS = 2.0 / 3.0  # LWP = (2/3) rho_w tau r_e, vertically uniform LWC (Stephens 1978)
LWP_FACTOR_ADIABATIC = 5.0 / 9.0  # LWP = (5/9) rho_w tau r_e,top, adiabatic LWC (Wood & Hartmann 2006,
# J. Climate 19, 1748; Szczodrak et al. 2001, JAS 58, 2912)
RHO_W_G_PER_M3 = 1.0e6  # liquid water density


def lwp_from_tau_re_gm2(tau, r_e_um, factor: float = LWP_FACTOR_HOMOGENEOUS):
    """LWP (g m-2) = factor * rho_w * tau * r_e.

    With rho_w = 1e6 g m-3 and r_e in um (1e-6 m) the constants cancel:
    LWP[g m-2] = factor * tau * r_e[um]."""
    return factor * RHO_W_G_PER_M3 * tau * (r_e_um * 1e-6)


def window_stats(da: xr.DataArray, centers: np.ndarray, half_width: np.timedelta64) -> Dict[str, np.ndarray]:
    """Mean, sample std, valid count, total count and coverage of `da` within
    [c - half_width, c + half_width] for each center time c.

    Cumulative sums make this O(N + M) for N samples and M centers, so a year
    of 20 s data is handled in well under a second.

    Returns arrays of length len(centers):
        mean, std (ddof=1; NaN if < 2 valid), n (valid), n_total, coverage = n / n_total
    """
    t = np.asarray(da["time"].values)
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = np.asarray(da.values, dtype=float)[order]
    valid = np.isfinite(v)
    v0 = np.where(valid, v, 0.0)
    # Prepend 0 so that sums over [lo, hi) are csum[hi] - csum[lo]
    c_n = np.concatenate([[0], np.cumsum(valid)])
    c_s = np.concatenate([[0.0], np.cumsum(v0)])
    c_s2 = np.concatenate([[0.0], np.cumsum(v0**2)])

    centers = np.asarray(centers, dtype="datetime64[ns]")
    lo = np.searchsorted(t, centers - half_width, side="left")
    hi = np.searchsorted(t, centers + half_width, side="right")
    n = (c_n[hi] - c_n[lo]).astype(float)
    n_total = (hi - lo).astype(float)
    s = c_s[hi] - c_s[lo]
    s2 = c_s2[hi] - c_s2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 1, (s2 - n * mean**2) / (n - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))  # clip: tiny negative values from round-off
        coverage = np.where(n_total > 0, n / n_total, np.nan)
    return {"mean": mean, "std": std, "n": n, "n_total": n_total, "coverage": coverage}


def window_average(
    fields: Mapping[str, xr.DataArray], centers: np.ndarray, half_width_min: float
) -> xr.Dataset:
    """Window statistics of several masked fields at `centers`.

    Output variables per field F: F (mean), F_sd, F_n, F_coverage."""
    hw = np.timedelta64(int(round(half_width_min * 60e3)), "ms")
    out = xr.Dataset(coords={"time": np.asarray(centers, dtype="datetime64[ns]")})
    for name, da in fields.items():
        st = window_stats(da, centers, hw)
        units = da.attrs.get("units", "")
        out[name] = ("time", st["mean"], {"long_name": f"window mean of {name}", "units": units})
        out[f"{name}_sd"] = (
            "time",
            st["std"],
            {"long_name": f"window standard deviation of {name}", "units": units},
        )
        out[f"{name}_n"] = ("time", st["n"], {"long_name": f"valid samples of {name} in window"})
        out[f"{name}_coverage"] = (
            "time",
            st["coverage"],
            {"long_name": f"fraction of {name} samples in window that are valid"},
        )
    out.attrs["window_half_width_min"] = half_width_min
    return out


def match_to_times(
    target: Mapping[str, xr.DataArray], sources: Mapping[str, xr.DataArray], half_width_min: float
) -> xr.Dataset:
    """Target samples (kept as-is) plus window statistics of `sources` around them.

    target  : masked fields on the comparison time axis (e.g. sunphotometer)
    sources : masked fields to be averaged in windows (e.g. MFRSR, MWR)
    """
    first = next(iter(target.values()))
    centers = first["time"].values
    out = window_average(sources, centers, half_width_min)
    for name, da in target.items():
        out[name] = ("time", np.asarray(da.values, dtype=float), dict(da.attrs))
    return out
