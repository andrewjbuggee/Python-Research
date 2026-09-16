"""Surface-flux responses to radiative forcing at NSA C1, in the Sledd et al. (2025) framework.

The observational counterpart of ``ERA5/surface_energy_budget/turbulent_flux_response.py``
and the engine behind ``seb_flux_response_to_DLR.ipynb`` and
``seb_flux_response_to_DLR_and_SW.ipynb`` in this directory. Everything the
notebooks draw is computed here so the two cannot drift apart.

The framework (Sledd et al. 2025, Sect. 3.2, after Miller et al. 2017)
=======================================================================
Variability in the atmospheric radiative forcing

    F = LWD + SWN,        SWN = SWD - SWU

drives changes in the remaining surface fluxes, the *responders*. The response
of each responder is the slope of an ordinary least-squares regression of that
term on F, computed within one calendar month, and it is the fraction of an
additional W m^-2 of forcing that the term disposes of. With every responder
counted as energy LEAVING the surface (Sledd Eq. 1: SH, LH positive upward;
G positive toward the surface, so -G is energy going down),

    d(LWD + SWN)/dF = 1 = d(LWU + SH + LH + M - SWT - G)/dF          (Eq. 9)

is the closure test: if every term were measured without error the responses
would sum to one. The "net atmospheric flux"

    NA = LWD - LWU + SWD - SWU - SH - LH = M - SWT - G                (Eq. 5)

is the remainder on the atmospheric side, and dNA/dF = 1 - f_LWU - f_SH - f_LH
by construction -- the same quantity the ERA5 notebooks call f_res.

Two forcers are supported, exactly as in the ERA5 notebooks:

    forcer="lwd"   x = LWD.  Five responders: LWU, SH, LH, SWN (as a
                   responder, with the sign flipped), and the remainder.
    forcer="fnet"  x = LWD + SWN, Sledd's forcer. Four responders: SWN is now
                   part of what is being perturbed.

What is different at Barrow, and it matters for reading every number
====================================================================
At MOSAiC every responder was measured independently: LWU by pyrgeometer, SH
and LH by eddy covariance, G by ice-mass-balance buoys. At NSA C1 only the
radiative terms are measured. SH and LH are BULK PARAMETERIZATIONS
(arm_nsa/bulk_flux.py) driven by T_skin - T_2m and the wind; G is ESTIMATED
(arm_nsa/ground_flux.py), here from the skin-temperature history. Consequences:

* f_LWU is a measurement-based response (pyrgeometers), directly comparable
  to Sledd's.
* f_SH and f_LH are the responses of the parameterization: they say how much
  of the forcing the bulk formula assigns to turbulence, given the measured
  T_skin, T_2m and wind. They inherit the roughness length (a free parameter;
  the sensitivity is computed below) and the stability functions.
* f_G from the thermal-inertia estimate is NOT independent of f_LWU: both
  derive from the same skin temperature. So Sledd's Eq. (9) closure --
  "measured responses sum to one" -- cannot be an independent test at Barrow
  the way it was at MOSAiC. The honest closure statement here is the
  comparison of the two routes to the subsurface response, dNA/dF (measured
  radiation minus parameterized turbulence) against d(-G_TI)/dF (thermal
  inertia), which is the analogue of Sledd's NA-vs-(-G) comparison and
  carries the same "different footprints and unaccounted terms" caveat.
* SWT (shortwave transmitted below the surface) is taken as zero: Sledd's
  Beer's-law slab is 6 cm of snow/ice over a transparent ocean; at Barrow the
  snow is deeper than that all season and the tundra beneath it is opaque, so
  whatever shortwave penetrates the top of the pack is absorbed within it and
  belongs to G, not to a separate transmitted term. M = 0 (no melt Oct-Mar).

Estimator notes (identical to the ERA5 notebooks)
=================================================
* The response goes on y. slope(y|x) = cov/var_x IS the derivative; inverting
  slope(x|y) overstates it by 1/r^2.
* Plain OLS is the headline (Sledd); a multiple regression that holds T_2m
  (and optionally wind) fixed is the other end of the confounder/mediator
  bracket and is reported beside it.
* Confidence intervals come from a moving-block bootstrap (7-day blocks), not
  from the textbook standard error, which assumes independent samples and is
  wrong by the square root of the effective-sample ratio. One cold season at
  10-min holds ~26 seven-day blocks, so intervals are wide and honest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import bulk_flux, config

# ---------------------------------------------------------------------------
# Constants and labels
# ---------------------------------------------------------------------------

WINTER_MONTHS: Tuple[int, ...] = (10, 11, 12, 1, 2, 3)
WINTER_LABELS: Tuple[str, ...] = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar")
# Sledd et al. (2025) restrict the winter model comparison to 15 Oct - 14 Mar.
SLEDD_WINTER_START = (10, 15)
SLEDD_WINTER_END = (3, 14)

# Population gates. Cloud fraction is the hourly fraction of 4-s ARSCL samples
# with a detected cloud base (0 .. 1). The LWP gates follow the emissivity
# argument of the ERA5 notebooks (Stephens 1978, eps = 1 - exp(-0.15 LWP)):
# 10 g m-2 is where a liquid cloud is already 78% emissive.
CLOUDY_MIN_CLOUD_FRACTION = 0.95
CLEAR_MAX_CLOUD_FRACTION = 0.05
LIQUID_MIN_LWP_G_M2 = 10.0

DEFAULT_BLOCK_DAYS = 7
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260915
MIN_BOOTSTRAP_BLOCKS = 8

# Default roughness lengths for the sensitivity of the bulk-flux responses.
Z0_SENSITIVITY_M: Tuple[float, ...] = (1.0e-4, 3.0e-4, 1.0e-3)

# The digitised MOSAiC responses of Sledd et al. (2025) Fig. 2a, Oct-Mar, on
# the DLR + SWN forcer. Imported from the ERA5 module when it is importable so
# the two notebooks share one copy; the fallback is a verbatim duplicate of
# that module's table (digitised against the 0.2 gridlines to about +-0.02).
_ERA5_DIR = Path(__file__).resolve().parents[2] / "ERA5" / "surface_energy_budget"


def _try_import_era5_module(name: str):
    if str(_ERA5_DIR) not in sys.path and _ERA5_DIR.is_dir():
        sys.path.insert(0, str(_ERA5_DIR))
    try:
        return __import__(name)
    except Exception:  # noqa: BLE001 -- optional cross-project dependency
        return None


_tfr = _try_import_era5_module("turbulent_flux_response")
SLEDD_FIG2A_WINTER: Dict[str, Dict[int, float]] = (
    _tfr.SLEDD_FIG2A_WINTER
    if _tfr is not None and hasattr(_tfr, "SLEDD_FIG2A_WINTER")
    else {
        "lwu": {10: 0.535, 11: 0.530, 12: 0.500, 1: 0.440, 2: 0.565, 3: 0.535},
        "sh": {10: 0.150, 11: 0.125, 12: 0.250, 1: 0.215, 2: 0.130, 3: 0.320},
        "lh": {10: 0.020, 11: 0.005, 12: 0.025, 1: 0.010, 2: -0.010, 3: 0.005},
        "g": {10: 0.240, 11: 0.280, 12: 0.210, 1: 0.320, 2: 0.255, 3: 0.235},
        "na": {10: 0.285, 11: 0.335, 12: 0.220, 1: 0.335, 2: 0.310, 3: 0.120},
        "swt": {10: 0.000, 11: 0.000, 12: 0.000, 1: 0.000, 2: 0.000, 3: -0.010},
        "total": {10: 0.940, 11: 0.940, 12: 0.960, 1: 0.995, 2: 0.945, 3: 1.075},
    }
)
# Sledd et al. (2025) Fig. 1, December 2019, all sites, 10-min data.
SLEDD_FIG1_DECEMBER = {"lwu": 0.50, "sh": 0.25, "lh": 0.02, "g": 0.20}

# One style per responder, shared by every figure (colours follow the ERA5
# module so a term has the same colour in both projects).
TERM_STYLE: Dict[str, Tuple[str, str, str]] = {
    "f_lwu": ("Upwelling LW", "#B2182B", "o"),
    "f_sh": ("Sensible heat (bulk)", "#2A9D8F", "s"),
    "f_lh": ("Latent heat (bulk)", "#3A6FD8", "X"),
    "f_sw": ("Net shortwave", "#DD8452", "D"),
    "f_res": ("NA / residual (subsurface)", "#7B2D8E", "^"),
    "f_g_ti": ("-G, thermal inertia", "#E0A000", ">"),
}

POP_LABELS: Dict[str, str] = {
    "all": "all sky",
    "cloudy": f"cloudy (cloud fraction >= {CLOUDY_MIN_CLOUD_FRACTION:g})",
    "liquid": f"liquid-bearing overcast (LWP >= {LIQUID_MIN_LWP_G_M2:g} g m$^{{-2}}$)",
    "clear": f"clear (cloud fraction <= {CLEAR_MAX_CLOUD_FRACTION:g})",
}
POP_COLORS: Dict[str, str] = {
    "all": "#333333",
    "cloudy": "#1F77B4",
    "liquid": "#2CA02C",
    "clear": "#9467BD",
}


# ---------------------------------------------------------------------------
# Loading the observational product
# ---------------------------------------------------------------------------


def load_product(
    path: "str | Path",
    exclude_lwd_disagreements: bool = False,
    require_irt_skin: bool = False,
) -> xr.Dataset:
    """Open a file written by scripts/build_nsa_seb_hourly.py and derive the forcers.

    Parameters
    ----------
    path:
        The netCDF product (hourly or 10-min).
    exclude_lwd_disagreements:
        Drop samples whose LWD came from a flagged disagreement between the two
        pyrgeometers (lwd_source == 4). Default keeps them (pyrgeometer 2 was
        used); the sensitivity to this choice is one line in the report.
    require_irt_skin:
        Drop samples whose skin temperature fell back to the LWU inversion
        (t_skin_source == 2).

    Returns
    -------
    Dataset with the analysis columns:
        lwd, lwu, swd, swu, swn, fnet, sh_up, lh_up, na, g_ti, neg_g_ti,
        t_skin, t_skin_lwu, t_2m, dskt (T_skin - T_2m), wspd, lwp, iwp,
        cloud_fraction, month
    """
    ds = xr.open_dataset(path)
    out = xr.Dataset(coords={"time": ds["time"]})
    out["lwd"] = ds["lwd_W_m2"]
    out["lwu"] = ds["lwu_W_m2"]
    out["swd"] = ds["swd_W_m2"]
    out["swu"] = ds["swu_W_m2"]
    out["swn"] = ds["swn_W_m2"]
    out["fnet"] = ds["lwd_W_m2"] + ds["swn_W_m2"]
    out["sh_up"] = ds["sh_up_W_m2"]
    out["lh_up"] = ds["lh_up_W_m2"]
    # NA is rebuilt from the product columns rather than read from the file:
    # the file's na_flux_W_m2 is the mean of the 1-min NA, which differs at
    # the 1e-4 level from the combination of the averaged components whenever
    # the valid-minute sets differ between terms. The combination is what
    # makes d(NA)/dF = 1 - f_lwu - f_sh - f_lh hold exactly (self_check).
    out["na"] = (
        ds["lwd_W_m2"]
        - ds["lwu_W_m2"]
        + ds["swn_W_m2"]
        - ds["sh_up_W_m2"]
        - ds["lh_up_W_m2"]
    )
    out["na_product"] = ds["na_flux_W_m2"]
    out["g_ti"] = ds["g_thermal_inertia_W_m2"]  # positive toward the surface
    out["neg_g_ti"] = -ds["g_thermal_inertia_W_m2"]  # energy leaving downward
    out["t_skin"] = ds["t_skin_K"]
    out["t_skin_lwu"] = ds["t_skin_from_lwu_K"]
    out["t_2m"] = ds["t_2m_K"]
    out["dskt"] = ds["t_skin_K"] - ds["t_2m_K"]
    out["wspd"] = ds["wind_speed_10m_m_s"]
    out["rh_2m"] = ds["rh_2m_pct"]
    out["p_sfc"] = ds["p_sfc_Pa"]
    for name in (
        "lwp_g_m2",
        "iwp_g_m2",
        "cloud_fraction",
        "lwd_source",
        "t_skin_source",
    ):
        if name in ds:
            out[name.replace("_g_m2", "")] = ds[name]
    out["month"] = ds["time"].dt.month
    keep = np.isfinite(out["lwd"]) & np.isfinite(out["lwu"])
    if exclude_lwd_disagreements and "lwd_source" in out:
        keep = keep & (out["lwd_source"] != 4)
    if require_irt_skin and "t_skin_source" in out:
        keep = keep & (out["t_skin_source"] == 1)
    out = out.isel(time=np.where(keep.values)[0])
    out.attrs = dict(ds.attrs)
    out.attrs["source_file"] = str(path)
    out.attrs["exclude_lwd_disagreements"] = int(exclude_lwd_disagreements)
    out.attrs["require_irt_skin"] = int(require_irt_skin)
    ds.close()
    return out


def population_masks(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """Boolean masks over time for the standard populations."""
    n = ds.sizes["time"]
    cf = ds["cloud_fraction"].values if "cloud_fraction" in ds else np.full(n, np.nan)
    lwp = ds["lwp"].values if "lwp" in ds else np.full(n, np.nan)
    masks = {
        "all": np.ones(n, dtype=bool),
        "cloudy": cf >= CLOUDY_MIN_CLOUD_FRACTION,
        "liquid": (cf >= CLOUDY_MIN_CLOUD_FRACTION) & (lwp >= LIQUID_MIN_LWP_G_M2),
        "clear": cf <= CLEAR_MAX_CLOUD_FRACTION,
    }
    return masks


def month_mask(ds: xr.Dataset, months: Iterable[int]) -> np.ndarray:
    m = ds["month"].values
    return np.isin(m, list(months))


def sledd_winter_mask(ds: xr.Dataset) -> np.ndarray:
    """15 Oct .. 14 Mar, the window of Sledd et al. (2025) Sect. 4.3."""
    t = ds["time"].values.astype("datetime64[D]")
    mo = ds["time"].dt.month.values
    dy = ds["time"].dt.day.values
    after_start = (mo > SLEDD_WINTER_START[0]) | (
        (mo == SLEDD_WINTER_START[0]) & (dy >= SLEDD_WINTER_START[1])
    )
    before_end = (mo < SLEDD_WINTER_END[0]) | (
        (mo == SLEDD_WINTER_END[0]) & (dy <= SLEDD_WINTER_END[1])
    )
    return after_start | before_end


# ---------------------------------------------------------------------------
# Regression primitives
# ---------------------------------------------------------------------------


def ols(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Plain least squares of y on x, returning slope, intercept, r, r2 and moments.

    slope = cov(x, y) / var(x): the derivative dy/dx averaged over the range
    of x the data cover. NaNs in either variable drop the pair.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    n = int(good.sum())
    if n < 3:
        return {
            k: float("nan")
            for k in (
                "slope",
                "intercept",
                "r",
                "r2",
                "x_mean",
                "y_mean",
                "x_sd",
                "y_sd",
            )
        } | {"n": n}
    xg, yg = x[good], y[good]
    xm, ym = xg.mean(), yg.mean()
    dx, dy = xg - xm, yg - ym
    var_x = float(np.dot(dx, dx))
    var_y = float(np.dot(dy, dy))
    cov = float(np.dot(dx, dy))
    if var_x <= 0:
        slope = float("nan")
    else:
        slope = cov / var_x
    r = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else float("nan")
    return {
        "slope": slope,
        "intercept": ym - slope * xm if np.isfinite(slope) else float("nan"),
        "r": r,
        "r2": r * r if np.isfinite(r) else float("nan"),
        "n": n,
        "x_mean": float(xm),
        "y_mean": float(ym),
        "x_sd": float(np.sqrt(var_x / max(n - 1, 1))),
        "y_sd": float(np.sqrt(var_y / max(n - 1, 1))),
    }


def partial_slope(
    y: np.ndarray, x: np.ndarray, controls: Sequence[np.ndarray] = ()
) -> float:
    """Coefficient of x in y = a + b x + sum_k c_k z_k (least squares).

    With no controls this is ols()["slope"]. With controls it is the response
    to x holding the controls fixed -- a genuine partial derivative, but see
    the mediator caveat in the module docstring.
    """
    cols = [np.asarray(x, dtype=float)] + [np.asarray(z, dtype=float) for z in controls]
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y)
    for c in cols:
        good &= np.isfinite(c)
    if good.sum() < len(cols) + 2:
        return float("nan")
    A = np.column_stack([np.ones(good.sum())] + [c[good] for c in cols])
    coef, *_ = np.linalg.lstsq(A, y[good], rcond=None)
    return float(coef[1])


CONTROL_SETS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "t2m": ("t_2m",),
    "t2m_wind": ("t_2m", "wspd"),
}


def _col(ds: xr.Dataset, name: str, mask: np.ndarray) -> np.ndarray:
    return ds[name].values[mask]


PARTITION_COLUMNS: Tuple[str, ...] = (
    "lwd",
    "fnet",
    "swn",
    "lwu",
    "sh_up",
    "lh_up",
    "na",
    "neg_g_ti",
    "t_skin",
    "t_2m",
    "dskt",
    "wspd",
)


def frame_of(ds: xr.Dataset, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """The analysis columns of one population as plain numpy arrays.

    Every estimator below works on this dict rather than on the Dataset, so
    the bootstrap can resample rows with a single fancy-index per draw
    instead of paying xarray's per-variable overhead 2,000 times.
    """
    return {
        k: np.asarray(ds[k].values[mask], dtype=float)
        for k in PARTITION_COLUMNS
        if k in ds
    }


def _partial(frame: Dict[str, np.ndarray], y: str, x: str, control: str) -> float:
    ctrl = [frame[c] for c in CONTROL_SETS[control]]
    return partial_slope(frame[y], frame[x], ctrl)


def slope_of(
    ds: xr.Dataset, mask: np.ndarray, y: str, x: str, control: str = "none"
) -> float:
    ctrl = [_col(ds, c, mask) for c in CONTROL_SETS[control]]
    return partial_slope(_col(ds, y, mask), _col(ds, x, mask), ctrl)


def stats_of(ds: xr.Dataset, mask: np.ndarray, y: str, x: str) -> Dict[str, float]:
    return ols(_col(ds, x, mask), _col(ds, y, mask))


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------

FORCER_KEY = {"lwd": "lwd", "fnet": "fnet"}
FORCER_LABEL = {"lwd": "DLR", "fnet": "DLR + SW$_{net}$"}


def partition_frame(
    frame: Dict[str, np.ndarray], forcer: str = "fnet", control: str = "none"
) -> Dict[str, float]:
    """partition() on a frame_of() dict; see partition() for the definitions."""
    x = FORCER_KEY[forcer]
    f_lwu = _partial(frame, "lwu", x, control)
    f_sh = _partial(frame, "sh_up", x, control)
    f_lh = _partial(frame, "lh_up", x, control)
    out = {
        "forcer": forcer,
        "control": control,
        "f_lwu": f_lwu,
        "f_sh": f_sh,
        "f_lh": f_lh,
    }
    if forcer == "lwd":
        f_sw = -_partial(frame, "swn", x, control)
        out["f_sw"] = f_sw
        out["f_res"] = 1.0 - f_lwu - f_sh - f_lh - f_sw
    else:
        out["f_res"] = 1.0 - f_lwu - f_sh - f_lh
    out["f_res_direct"] = _partial(frame, "na", x, control)
    out["f_g_ti"] = _partial(frame, "neg_g_ti", x, control)
    out["total_measured"] = f_lwu + f_sh + f_lh + out["f_g_ti"]
    out["dskt_dF"] = _partial(frame, "t_skin", x, control)
    out["dt2m_dF"] = _partial(frame, "t_2m", x, control)
    out["ddskt_dF"] = _partial(frame, "dskt", x, control)
    out["n"] = int(np.isfinite(frame[x]).sum())
    for key in (
        "lwd",
        "swn",
        "fnet",
        "lwu",
        "sh_up",
        "lh_up",
        "t_skin",
        "t_2m",
        "wspd",
    ):
        out[f"{key}_mean"] = float(np.nanmean(frame[key])) if out["n"] else float("nan")
    return out


def partition(
    ds: xr.Dataset, mask: np.ndarray, forcer: str = "fnet", control: str = "none"
) -> Dict[str, float]:
    """Split each additional W m-2 of the forcer into where it goes.

    Every fraction is "energy leaving the surface per W m-2 of forcing":

        f_lwu  =  d(LWU)/dF
        f_sh   =  d(SH_up)/dF          (bulk parameterization)
        f_lh   =  d(LH_up)/dF          (bulk parameterization)
        f_sw   = -d(SWN)/dF            (forcer="lwd" only; SWN as a responder)
        f_res  =  1 - (sum of the above)  = d(NA)/dF by linearity (Eq. 5)

    plus, from the independent (but not independent-of-T_skin) estimate,

        f_g_ti = d(-G_TI)/dF           thermal-inertia subsurface response
        total_measured = f_lwu + f_sh + f_lh + f_g_ti   (Sledd Eq. 9 analogue)

    and the temperature responses dT_skin/dF, dT_2m/dF, d(T_skin - T_2m)/dF.
    """
    return partition_frame(frame_of(ds, mask), forcer, control)


def partition_r2(
    ds: xr.Dataset, mask: np.ndarray, forcer: str = "fnet"
) -> Dict[str, float]:
    """r^2 of each responder's plain regression on the forcer (marker sizing)."""
    x = FORCER_KEY[forcer]
    keys = {
        "f_lwu": "lwu",
        "f_sh": "sh_up",
        "f_lh": "lh_up",
        "f_sw": "swn",
        "f_res": "na",
        "f_g_ti": "neg_g_ti",
    }
    return {k: stats_of(ds, mask, v, x)["r2"] for k, v in keys.items()}


def ledger(part: Dict[str, float]) -> Dict[str, float]:
    """The sign-honest normalisation of the ERA5 notebooks.

    Negative fractions are sources (they arrive with the forcing anomaly),
    positive ones are sinks; both sides normalised by the gross energy
    G = 1 + sum(max(-f, 0)) = sum(max(f, 0)) so each sums to one.
    """
    keys = [k for k in ("f_lwu", "f_sh", "f_lh", "f_sw", "f_res") if k in part]
    vals = {k: part[k] for k in keys}
    gross = 1.0 + sum(max(-v, 0.0) for v in vals.values())
    return {
        "gross": gross,
        "supply": {
            "forcing": 1.0 / gross,
            **{k: max(-v, 0.0) / gross for k, v in vals.items() if v < 0},
        },
        "disposal": {k: max(v, 0.0) / gross for k, v in vals.items() if v > 0},
    }


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------


def block_ids(ds: xr.Dataset, block_days: int = DEFAULT_BLOCK_DAYS) -> np.ndarray:
    """Index of the `block_days`-long block each sample falls in."""
    t = ds["time"].values
    days = (t - t.min()).astype("timedelta64[s]").astype(float) / 86400.0
    return (days // block_days).astype(int)


def _block_rows(
    ds: xr.Dataset, mask: np.ndarray, block_days: int
) -> Dict[int, np.ndarray]:
    ids = block_ids(ds, block_days)
    present = np.unique(ids[mask])
    return {int(b): np.where(mask & (ids == b))[0] for b in present}


def block_bootstrap(
    ds: xr.Dataset,
    mask: np.ndarray,
    stat: Callable[[Dict[str, np.ndarray]], "float | Dict[str, float]"],
    block_days: int = DEFAULT_BLOCK_DAYS,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = 95.0,
) -> Dict[str, Dict[str, float]]:
    """Percentile intervals for `stat(frame)` by resampling whole blocks.

    `stat` takes a frame_of() dict and returns either a float or a dict of
    floats; the result is {name: {"value", "lo", "hi", "se", ...}} (name is
    "stat" for a scalar stat).

    Blocks -- not samples -- are the units the record supplies independently:
    LWD at Barrow has a lag-1 hourly autocorrelation near 0.99, so a season of
    hourly samples holds a few dozen independent weather situations, not a
    few thousand points. A block keeps a weather system intact and carries
    its internal correlation along. The RNG seed is recorded in the output.
    """
    rows_by_block = _block_rows(ds, mask, block_days)
    blocks = np.array(sorted(rows_by_block))
    n_block = blocks.size
    if n_block < MIN_BOOTSTRAP_BLOCKS:
        raise ValueError(
            f"only {n_block} blocks of {block_days} days in this population; a "
            f"percentile interval from that few is not worth quoting"
        )
    full_rows = np.concatenate([rows_by_block[b] for b in blocks])
    full_mask = np.zeros(ds.sizes["time"], dtype=bool)
    full_mask[full_rows] = True
    frame = frame_of(ds, full_mask)
    # positions of each block's rows inside the frame
    pos = {}
    start = 0
    for b in blocks:
        n = rows_by_block[b].size
        pos[b] = np.arange(start, start + n)
        start += n
    value = stat(frame)
    scalar = not isinstance(value, dict)
    names = ["stat"] if scalar else list(value)
    draws = {k: np.empty(n_boot) for k in names}
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        chosen = rng.choice(blocks, size=n_block, replace=True)
        idx = np.concatenate([pos[b] for b in chosen])
        sub = {k: v[idx] for k, v in frame.items()}
        res = stat(sub)
        if scalar:
            draws["stat"][i] = res
        else:
            for k in names:
                draws[k][i] = res[k]
    lo_q, hi_q = 50.0 - ci / 2.0, 50.0 + ci / 2.0
    out = {}
    for k in names:
        d = draws[k][np.isfinite(draws[k])]
        out[k] = {
            "value": float(value if scalar else value[k]),
            "lo": float(np.percentile(d, lo_q)) if d.size else float("nan"),
            "hi": float(np.percentile(d, hi_q)) if d.size else float("nan"),
            "se": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
            "n_block": int(n_block),
            "n_boot": int(n_boot),
            "block_days": block_days,
            "seed": seed,
        }
    return out


def naive_se(ds: xr.Dataset, mask: np.ndarray, y: str, x: str) -> float:
    """The textbook OLS standard error -- WRONG HERE, kept for the comparison."""
    st = stats_of(ds, mask, y, x)
    n, r2 = st["n"], st["r2"]
    if not np.isfinite(r2) or n < 3 or st["x_sd"] <= 0:
        return float("nan")
    return float(st["y_sd"] / st["x_sd"] * np.sqrt(max(1.0 - r2, 0.0) / (n - 2)))


def bootstrap_partition(
    ds: xr.Dataset,
    mask: np.ndarray,
    forcer: str = "fnet",
    control: str = "none",
    terms: Sequence[str] = (
        "f_lwu",
        "f_sh",
        "f_lh",
        "f_sw",
        "f_res",
        "f_g_ti",
        "total_measured",
        "dskt_dF",
    ),
    **kw,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap intervals for every partition term at once (one partition per draw)."""
    keep = [t for t in terms if not (forcer == "fnet" and t == "f_sw")]
    res = block_bootstrap(
        ds,
        mask,
        lambda fr: {t: partition_frame(fr, forcer, control)[t] for t in keep},
        **kw,
    )
    return res


# ---------------------------------------------------------------------------
# Month by month
# ---------------------------------------------------------------------------


def monthly_partition(
    ds: xr.Dataset,
    mask: np.ndarray,
    forcer: str = "fnet",
    control: str = "none",
    months: Tuple[int, ...] = WINTER_MONTHS,
    min_n: int = 100,
) -> Dict[str, np.ndarray]:
    """The partition one calendar month at a time (Sledd et al. Fig. 2a)."""
    keys = ["f_lwu", "f_sh", "f_lh", "f_res", "f_g_ti", "total_measured"]
    if forcer == "lwd":
        keys.insert(3, "f_sw")
    out: Dict[str, np.ndarray] = {k: np.full(len(months), np.nan) for k in keys}
    out.update({f"r2_{k}": np.full(len(months), np.nan) for k in keys})
    for k in ("n", "forcer_mean", "swn_mean", "t_skin_mean", "dskt_dF"):
        out[k] = np.full(len(months), np.nan)
    mo = ds["month"].values
    for i, m in enumerate(months):
        mm = mask & (mo == m)
        if mm.sum() < min_n:
            continue
        p = partition(ds, mm, forcer, control)
        r2 = partition_r2(ds, mm, forcer)
        for k in keys:
            out[k][i] = p[k]
            out[f"r2_{k}"][i] = r2.get(k, np.nan)
        out["n"][i] = p["n"]
        out["forcer_mean"][i] = p[f"{FORCER_KEY[forcer]}_mean"]
        out["swn_mean"][i] = p["swn_mean"]
        out["t_skin_mean"][i] = p["t_skin_mean"]
        out["dskt_dF"][i] = p["dskt_dF"]
    out["months"] = np.array(months)
    return out


# ---------------------------------------------------------------------------
# Bulk-flux sensitivity: the responses as a function of the free parameter
# ---------------------------------------------------------------------------


def recompute_bulk_fluxes(
    ds: xr.Dataset, z0_m: float, stable_scheme: str = "grachev2007"
) -> xr.Dataset:
    """Bulk SH/LH from the product-resolution state for another roughness length.

    Uses the averaged T_skin, T_2m, RH, U and p of the product rather than the
    1-min inputs, so it is a sensitivity of the RESPONSE to z_0, not a
    replacement for the product's own fluxes (which average the 1-min fluxes).
    """
    fl = bulk_flux.bulk_fluxes(
        t_skin_k=ds["t_skin"],
        t_air_k=ds["t_2m"],
        rh_air_pct=ds["rh_2m"],
        wspd_m_s=ds["wspd"],
        p_pa=ds["p_sfc"],
        z0_m=z0_m,
        stable_scheme=stable_scheme,
    )
    out = ds.copy()
    out["sh_up"] = fl["sh_up_W_m2"]
    out["lh_up"] = fl["lh_up_W_m2"]
    out["na"] = ds["lwd"] - ds["lwu"] + ds["swn"] - fl["sh_up_W_m2"] - fl["lh_up_W_m2"]
    out.attrs["z0_m"] = z0_m
    out.attrs["stable_scheme"] = stable_scheme
    return out


def z0_sensitivity(
    ds: xr.Dataset,
    mask: np.ndarray,
    forcer: str = "fnet",
    z0_values: Sequence[float] = Z0_SENSITIVITY_M,
    schemes: Sequence[str] = ("grachev2007", "beljaars1991"),
) -> List[Dict[str, float]]:
    rows = []
    for scheme in schemes:
        for z0 in z0_values:
            d2 = recompute_bulk_fluxes(ds, z0, scheme)
            p = partition(d2, mask, forcer)
            rows.append(
                {
                    "scheme": scheme,
                    "z0_m": z0,
                    **{
                        k: p[k]
                        for k in ("f_lwu", "f_sh", "f_lh", "f_res", "sh_up_mean")
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# ERA5 at the Barrow grid cell
# ---------------------------------------------------------------------------

ERA5_BARROW_DIR = _ERA5_DIR / "data" / "barrow"


def load_era5_cell(
    start: str,
    end: str,
    lat: float = config.SITE_LAT_DEG,
    lon: float = config.SITE_LON_DEG,
    era5_dir: "str | Path" = ERA5_BARROW_DIR,
) -> Optional[xr.Dataset]:
    """The ERA5 grid cell nearest NSA C1 as an analysis dataset with the same columns.

    Uses ERA5/surface_energy_budget/seb_terms.compute_seb_terms for the sign
    conventions, so f_sh here is the response of ERA5's own (downward-
    positive, negated) turbulent flux. ERA5 has no G; only the residual.
    Returns None when the ERA5 files or module are not available.
    """
    import glob

    seb_terms = _try_import_era5_module("seb_terms")
    files = sorted(glob.glob(str(Path(era5_dir) / "era5_seb_barrow_*.nc")))
    if seb_terms is None or not files:
        return None
    pieces = []
    for f in files:
        with xr.open_dataset(f) as d:
            tcoord = "valid_time" if "valid_time" in d.coords else "time"
            sub = d.sel(latitude=lat, longitude=lon, method="nearest")
            sub = sub.sel({tcoord: slice(start, f"{end}T23:59:59")})
            if sub.sizes.get(tcoord, 0) == 0:
                continue
            pieces.append(sub.load())
    if not pieces:
        return None
    cell = xr.concat(pieces, dim=tcoord).sortby(tcoord)
    _, idx = np.unique(cell[tcoord].values, return_index=True)
    cell = cell.isel({tcoord: idx})
    if tcoord != "time":
        cell = cell.rename({tcoord: "time"})
    terms = seb_terms.compute_seb_terms(cell)
    out = xr.Dataset(coords={"time": terms["time"]})
    out["lwd"] = terms["lwd_W_m2"]
    out["lwu"] = terms["lwu_W_m2"]
    out["swn"] = terms["swn_W_m2"]
    out["swd"] = terms["swd_W_m2"]
    out["swu"] = terms["swu_W_m2"]
    out["fnet"] = terms["lwd_W_m2"] + terms["swn_W_m2"]
    out["sh_up"] = terms["sh_up_W_m2"]
    out["lh_up"] = terms["lh_up_W_m2"]
    out["na"] = terms["na_flux_W_m2"]
    out["neg_g_ti"] = xr.full_like(out["na"], np.nan)
    out["g_ti"] = xr.full_like(out["na"], np.nan)
    out["t_skin"] = terms["t_skin_K"]
    out["t_skin_lwu"] = terms["t_skin_from_lwu_K"]
    out["t_2m"] = cell["t2m"]
    out["dskt"] = terms["t_skin_K"] - cell["t2m"]
    out["wspd"] = terms["wind_speed_10m_m_s"]
    out["lwp"] = (
        cell["tclw"] * 1000.0 if "tclw" in cell else xr.full_like(out["na"], np.nan)
    )
    out["iwp"] = (
        cell["tciw"] * 1000.0 if "tciw" in cell else xr.full_like(out["na"], np.nan)
    )
    out["cloud_fraction"] = (
        cell["tcc"] if "tcc" in cell else xr.full_like(out["na"], np.nan)
    )
    out["month"] = out["time"].dt.month
    out.attrs = {
        "source": "ERA5 hourly, nearest grid cell to NSA C1",
        "cell_lat": float(cell["latitude"]),
        "cell_lon": float(cell["longitude"]),
        "note": "coastal cell; ERA5 land-sea mask 0.62 land",
    }
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def self_check(
    ds: xr.Dataset, mask: Optional[np.ndarray] = None, tol: float = 1e-9
) -> bool:
    """Assert the identities every estimate depends on."""
    if mask is None:
        mask = np.ones(ds.sizes["time"], dtype=bool)
    # 1. a variable regressed on itself has slope 1
    assert abs(stats_of(ds, mask, "lwd", "lwd")["slope"] - 1.0) < tol
    # 2. the remainder equals the direct regression of NA (linearity), on rows
    #    where every term is finite
    full = mask & np.isfinite(ds["na"].values) & np.isfinite(ds["swn"].values)
    for forcer in ("lwd", "fnet"):
        p = partition(ds, full, forcer)
        assert abs(p["f_res"] - p["f_res_direct"]) < 1e-6, (
            forcer,
            p["f_res"],
            p["f_res_direct"],
        )
    # 3. controlling on a variable drives its own slope to zero
    ctrl = [_col(ds, "t_2m", full)]
    assert (
        abs(partial_slope(_col(ds, "t_2m", full), _col(ds, "lwd", full), ctrl)) < 1e-6
    )
    # 4. slope(y|x) * slope(x|y) == r^2
    a = stats_of(ds, full, "lwu", "lwd")
    b = stats_of(ds, full, "lwd", "lwu")
    assert abs(a["slope"] * b["slope"] - a["r2"]) < 1e-9
    return True


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def variable_check_table(ds: xr.Dataset) -> str:
    """Which Sledd et al. (2025) terms are measured, parameterized, or absent here."""
    rows = [
        (
            "LWD",
            "measured (SKYRAD pyrgeometers, QCRAD QC; two-pyrgeometer best estimate)",
            "lwd",
        ),
        ("LWU", "measured (GNDRAD pyrgeometer at 10 m)", "lwu"),
        ("SWD, SWU", "measured (QCRAD); SWN = SWD - SWU", "swn"),
        (
            "T_skin",
            "IRT brightness T corrected for emissivity 0.985 + reflected LWD; LWU inversion as fallback (Sledd: LWU inversion)",
            "t_skin",
        ),
        ("SH", "BULK PARAMETERIZATION (MOST; Sledd: eddy covariance)", "sh_up"),
        ("LH", "BULK PARAMETERIZATION (Sledd: eddy covariance)", "lh_up"),
        (
            "G",
            "ESTIMATED from T_skin history (thermal inertia); also as Eq. (5) residual (Sledd: IMB conduction + storage in a 6-cm slab)",
            "g_ti",
        ),
        (
            "SWT",
            "zero: snow deeper than the 6-cm slab over opaque tundra (Sledd: Beer's law, Eq. 6)",
            None,
        ),
        ("M", "zero, frozen surface Oct-Mar (Sledd: residual, ~0 in winter)", None),
        ("NA", "LWD - LWU + SWN - SH - LH (Eq. 5)", "na"),
        ("T_2m, U_10m", "measured (MET), for the controlled regressions", "t_2m"),
        ("LWP", "MWRRET + MWR3C (cloud population gate)", "lwp"),
        (
            "cloud fraction",
            "ARSCL cloud-base detections (cloud population gate)",
            "cloud_fraction",
        ),
    ]
    lines = [f"{'term':<14}{'valid':>8}   how it is obtained at NSA C1"]
    for name, how, key in rows:
        if key is not None and key in ds:
            v = ds[key].values
            valid = f"{100 * np.isfinite(v).mean():5.1f}%"
        else:
            valid = "   --  "
        lines.append(f"{name:<14}{valid:>8}   {how}")
    return "\n".join(lines)


def format_partition(p: Dict[str, float]) -> str:
    keys = (
        ["f_lwu", "f_sh", "f_lh"]
        + (["f_sw"] if "f_sw" in p else [])
        + ["f_res", "f_g_ti", "total_measured"]
    )
    parts = "  ".join(f"{k}={p[k]:+.3f}" for k in keys)
    return f"{parts}  | dTs/dF={p['dskt_dF']:.4f} K/(W m-2)  d(Ts-T2m)/dF={p['ddskt_dF']:+.4f}  n={p['n']}"


def print_report(
    ds: xr.Dataset,
    masks: Optional[Dict[str, np.ndarray]] = None,
    forcer: str = "fnet",
    populations: Sequence[str] = ("all", "cloudy", "liquid", "clear"),
    controls: Sequence[str] = ("none", "t2m"),
) -> None:
    masks = masks or population_masks(ds)
    print(
        f"Forcer: {FORCER_LABEL[forcer]}    (fractions = energy leaving the surface per W m-2 of forcing)"
    )
    for pop in populations:
        for control in controls:
            p = partition(ds, masks[pop], forcer, control)
            print(f"  {pop:<8}{control:<9} {format_partition(p)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# All figures return the matplotlib Figure and optionally save a PNG under
# out_dir. Styling deliberately follows the ERA5 module: density-coloured
# scatter with the unweighted least-squares line in black, the equation and
# r^2 in a box, the sample count above the panel.

DENSITY_CMAP = "viridis"
NOTE_BOX = dict(
    boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, edgecolor="#999999"
)
FIG_DPI = 300

RESPONDER_AXES: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "lwu": ("Upwelling LW, LWU [W m$^{-2}$]", (150.0, 330.0)),
    "sh_up": ("Sensible heat, SH$_{up}$ (bulk) [W m$^{-2}$]", (-80.0, 40.0)),
    "lh_up": ("Latent heat, LH$_{up}$ (bulk) [W m$^{-2}$]", (-25.0, 25.0)),
    "na": ("Net atmospheric flux, NA [W m$^{-2}$]", (-90.0, 50.0)),
    "neg_g_ti": ("$-G$, thermal inertia [W m$^{-2}$]", (-60.0, 60.0)),
    "swn": ("Net shortwave, SW$_{net}$ [W m$^{-2}$]", (0.0, 60.0)),
    "t_skin": ("Skin temperature, $T_{skin}$ [K]", (230.0, 275.0)),
    "dskt": ("$T_{skin} - T_{2m}$ [K]", (-10.0, 6.0)),
    "lwd": ("Downwelling LW, LWD [W m$^{-2}$]", (110.0, 330.0)),
    "fnet": ("LWD + SW$_{net}$ [W m$^{-2}$]", (110.0, 360.0)),
    "lwp": ("Liquid water path [g m$^{-2}$]", (0.0, 350.0)),
}


def _save(fig, out_dir, stem: str, dpi: int = FIG_DPI):
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    return fig


def density_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    xr_: Optional[Tuple[float, float]] = None,
    yr_: Optional[Tuple[float, float]] = None,
    bins: int = 120,
    title: str = "",
    title_color: str = "#222222",
    reference_line: Optional[Tuple[float, float, str]] = None,
) -> Dict[str, float]:
    """One density scatter with its y-on-x least-squares line and annotation."""
    from matplotlib.colors import LogNorm

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    st = ols(x[good], y[good])
    xr_ = xr_ or (np.nanpercentile(x[good], 0.5), np.nanpercentile(x[good], 99.5))
    yr_ = yr_ or (np.nanpercentile(y[good], 0.5), np.nanpercentile(y[good], 99.5))
    h, xe, ye = np.histogram2d(x[good], y[good], bins=bins, range=[xr_, yr_])
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    xi, yi = np.nonzero(h)
    if xi.size:
        frac = h[xi, yi] / h.max()
        order = np.argsort(frac)
        ax.scatter(
            xc[xi][order],
            yc[yi][order],
            c=frac[order],
            s=7,
            cmap=DENSITY_CMAP,
            norm=LogNorm(vmin=1e-2, vmax=1.0),
            linewidths=0,
        )
        lo, hi = xc[xi.min()], xc[xi.max()]
        xx = np.array([lo, hi])
        ax.plot(
            xx, st["intercept"] + st["slope"] * xx, color="#000000", lw=1.3, zorder=5
        )
    if reference_line is not None:
        a, b, lab = reference_line
        xx = np.array(xr_)
        ax.plot(xx, a + b * xx, color="#B2182B", lw=1.2, ls="--", zorder=4, label=lab)
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_xlim(*xr_)
    ax.set_ylim(*yr_)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_title(title, fontsize=10, color=title_color, fontweight="bold", loc="left")
    outside = 1.0 - h.sum() / max(good.sum(), 1)
    ax.annotate(
        f"n = {st['n']:,}"
        + (f"  |  {100 * outside:.1f}% outside axes" if outside > 0.005 else ""),
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax.text(
        0.03,
        0.965,
        f"slope = {st['slope']:+.3f}\nintercept = {st['intercept']:+.1f}\n$r^2$ = {st['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox=NOTE_BOX,
        zorder=7,
    )
    return st


def fig_responder_vs_forcer(
    ds: xr.Dataset,
    masks: Dict[str, np.ndarray],
    y: str,
    forcer: str = "fnet",
    populations: Sequence[str] = ("all", "cloudy", "clear"),
    out_dir=None,
    stem: Optional[str] = None,
    title: Optional[str] = None,
):
    """One responder against the forcer, one panel per population."""
    import matplotlib.pyplot as plt

    x = FORCER_KEY[forcer]
    xlabel, xr_ = RESPONDER_AXES[x]
    ylabel, yr_ = RESPONDER_AXES[y]
    fig, axes = plt.subplots(
        1, len(populations), figsize=(4.6 * len(populations), 4.2), squeeze=False
    )
    for ax, pop in zip(axes[0], populations):
        m = masks[pop]
        density_panel(
            ax,
            ds[x].values[m],
            ds[y].values[m],
            xlabel,
            ylabel,
            xr_,
            yr_,
            title=POP_LABELS[pop],
            title_color=POP_COLORS[pop],
        )
    fig.suptitle(
        title
        or f"{ylabel.split(' [')[0]} against {FORCER_LABEL[forcer]} -- NSA C1, {ds.attrs.get('period', '')}",
        fontsize=11.5,
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"{y}_vs_{x}")


def fig_all_responders(
    ds: xr.Dataset,
    mask: np.ndarray,
    forcer: str = "fnet",
    population: str = "all",
    out_dir=None,
    stem: Optional[str] = None,
):
    """The Sledd et al. Fig. 1 layout: every responder against the forcer, one population."""
    import matplotlib.pyplot as plt

    x = FORCER_KEY[forcer]
    xlabel, xr_ = RESPONDER_AXES[x]
    responders = ["lwu", "sh_up", "lh_up", "na", "neg_g_ti"] + (
        ["swn"] if forcer == "lwd" else ["t_skin"]
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ax, y in zip(axes.ravel(), responders):
        ylabel, yr_ = RESPONDER_AXES[y]
        density_panel(
            ax, ds[x].values[mask], ds[y].values[mask], xlabel, ylabel, xr_, yr_
        )
    fig.suptitle(
        f"Responders against {FORCER_LABEL[forcer]} -- NSA C1, {POP_LABELS[population]}, "
        f"{ds.attrs.get('period', '')} (after Sledd et al. 2025 Fig. 1)",
        fontsize=11.5,
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"all_responders_vs_{x}_{population}")


def fig_partition_bars(
    parts: Dict[str, Dict[str, float]],
    forcer: str = "fnet",
    intervals: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    out_dir=None,
    stem: Optional[str] = None,
    title: Optional[str] = None,
):
    """Grouped bars: the partition fractions for each population, with optional CIs."""
    import matplotlib.pyplot as plt

    terms = (
        ["f_lwu", "f_sh", "f_lh"]
        + (["f_sw"] if forcer == "lwd" else [])
        + ["f_res", "f_g_ti"]
    )
    pops = list(parts)
    fig, ax = plt.subplots(figsize=(1.6 * len(terms) + 4.0, 4.6))
    width = 0.8 / len(pops)
    for j, pop in enumerate(pops):
        vals = [parts[pop][t] for t in terms]
        xpos = np.arange(len(terms)) + (j - (len(pops) - 1) / 2) * width
        bars = ax.bar(
            xpos,
            vals,
            width,
            color=POP_COLORS.get(pop, "#888888"),
            label=f"{POP_LABELS.get(pop, pop)} (n={parts[pop]['n']:,})",
            alpha=0.9,
        )
        if intervals and pop in intervals:
            for xp, t in zip(xpos, terms):
                iv = intervals[pop].get(t)
                if iv:
                    ax.errorbar(
                        xp,
                        iv["value"],
                        yerr=[[iv["value"] - iv["lo"]], [iv["hi"] - iv["value"]]],
                        fmt="none",
                        ecolor="#000000",
                        elinewidth=1.0,
                        capsize=3,
                    )
        for xp, v in zip(xpos, vals):
            ax.text(
                xp,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:+.2f}",
                ha="center",
                fontsize=6.8,
            )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels([TERM_STYLE[t][0] for t in terms], fontsize=8.5)
    ax.set_ylabel(
        f"response to {FORCER_LABEL[forcer]}\n(fraction of 1 W m$^{{-2}}$ leaving the surface)"
    )
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(
        title
        or f"Partition of a {FORCER_LABEL[forcer]} anomaly -- NSA C1 observations",
        fontsize=11,
        loc="left",
    )
    note = (
        "f_LWU, f_SH, f_LH and f_res (= dNA/dF) sum to one by construction; f_G(TI) is the "
        "independent thermal-inertia estimate\n(SH, LH are bulk parameterizations; "
        "error bars: 95% moving-block bootstrap, 7-day blocks)"
    )
    ax.annotate(
        note,
        (0.0, -0.22),
        xycoords="axes fraction",
        fontsize=7,
        color="#555555",
        va="top",
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"partition_bars_{forcer}")


def fig_ledger(
    parts: Dict[str, Dict[str, float]],
    forcer: str = "fnet",
    out_dir=None,
    stem: Optional[str] = None,
):
    """Supply/disposal ledger normalised by the gross energy (ERA5 notebook 4b)."""
    import matplotlib.pyplot as plt

    pops = list(parts)
    fig, axes = plt.subplots(
        1, len(pops), figsize=(3.4 * len(pops), 4.4), sharey=True, squeeze=False
    )
    for ax, pop in zip(axes[0], pops):
        led = ledger(parts[pop])
        bottom = 0.0
        ax.bar(
            0, led["supply"]["forcing"], 0.6, color="#C8A02C", label="forcing anomaly"
        )
        bottom = led["supply"]["forcing"]
        for k, v in led["supply"].items():
            if k == "forcing":
                continue
            ax.bar(
                0,
                v,
                0.6,
                bottom=bottom,
                color=TERM_STYLE[k][1],
                hatch="//",
                label=f"{TERM_STYLE[k][0]} (source)",
            )
            bottom += v
        bottom = 0.0
        for k, v in led["disposal"].items():
            ax.bar(
                1, v, 0.6, bottom=bottom, color=TERM_STYLE[k][1], label=TERM_STYLE[k][0]
            )
            bottom += v
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["supply", "disposal"])
        ax.set_title(
            f"{POP_LABELS[pop]}\ngross = {led['gross']:.2f} W m$^{{-2}}$ per W m$^{{-2}}$",
            fontsize=8.5,
        )
        ax.legend(
            fontsize=6.2,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=1,
        )
    axes[0][0].set_ylabel("fraction of the gross energy in motion")
    fig.suptitle(
        f"Sign-honest ledger of the {FORCER_LABEL[forcer]} partition -- NSA C1",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"ledger_{forcer}")


def fig_monthly_response(
    ds: xr.Dataset,
    masks: Dict[str, np.ndarray],
    forcer: str = "fnet",
    populations: Sequence[str] = ("all", "cloudy"),
    era5: Optional[xr.Dataset] = None,
    show_sledd: bool = True,
    out_dir=None,
    stem: Optional[str] = None,
    min_n: int = 100,
):
    """Sledd et al. Fig. 2a for NSA C1: monthly responses, marker size ~ r^2.

    Overlays the digitised MOSAiC values (fnet forcer only) as hollow black
    markers and, if given, the ERA5 Barrow-cell responses as dashed lines.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(populations),
        figsize=(5.6 * len(populations), 4.8),
        sharey=True,
        squeeze=False,
    )
    xpos = np.arange(len(WINTER_MONTHS))
    keys = (
        ["f_lwu", "f_sh", "f_lh"]
        + (["f_sw"] if forcer == "lwd" else [])
        + ["f_res", "f_g_ti"]
    )
    sledd_key = {
        "f_lwu": "lwu",
        "f_sh": "sh",
        "f_lh": "lh",
        "f_res": "na",
        "f_g_ti": "g",
    }
    era_monthly = None
    if era5 is not None:
        era_monthly = monthly_partition(
            era5, population_masks(era5)["all"], forcer, min_n=min_n
        )
    for ax, pop in zip(axes[0], populations):
        r = monthly_partition(ds, masks[pop], forcer, min_n=min_n)
        for k in keys:
            lab, col, mk = TERM_STYLE[k]
            ax.plot(xpos, r[k], color=col, lw=1.5, zorder=3)
            ax.scatter(
                xpos,
                r[k],
                s=18 + 110 * np.nan_to_num(r[f"r2_{k}"]),
                color=col,
                marker=mk,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=4,
                label=lab if ax is axes[0][0] else None,
            )
            if era_monthly is not None and k in era_monthly and k != "f_g_ti":
                ax.plot(
                    xpos,
                    era_monthly[k],
                    color=col,
                    lw=1.0,
                    ls="--",
                    alpha=0.8,
                    zorder=2,
                    label=(
                        f"ERA5 cell ({lab.split(' (')[0]})"
                        if ax is axes[0][0]
                        else None
                    ),
                )
            if show_sledd and forcer == "fnet" and k in sledd_key:
                sv = [SLEDD_FIG2A_WINTER[sledd_key[k]][m] for m in WINTER_MONTHS]
                ax.scatter(
                    xpos + 0.12,
                    sv,
                    s=70,
                    facecolor="none",
                    edgecolor=col,
                    marker=mk,
                    linewidth=1.3,
                    zorder=5,
                    label=(
                        f"MOSAiC ({lab.split(' (')[0]})" if ax is axes[0][0] else None
                    ),
                )
        ax.plot(
            xpos,
            r["total_measured"],
            color="#000000",
            lw=1.2,
            ls=":",
            marker="D",
            ms=4,
            zorder=3,
            label="total: LWU+SH+LH+(-G TI)" if ax is axes[0][0] else None,
        )
        if show_sledd and forcer == "fnet":
            ax.scatter(
                xpos + 0.12,
                [SLEDD_FIG2A_WINTER["total"][m] for m in WINTER_MONTHS],
                s=50,
                facecolor="none",
                edgecolor="#000000",
                marker="D",
                linewidth=1.0,
                zorder=5,
            )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.axhline(1, color="#333333", lw=0.8, ls=":")
        ax.set_xticks(xpos)
        ax.set_xticklabels(WINTER_LABELS)
        ax.set_title(
            f"{POP_LABELS[pop]}  (n per month: "
            + ", ".join(f"{int(v)}" if np.isfinite(v) else "-" for v in r["n"])
            + ")",
            fontsize=8.6,
            loc="left",
        )
        ax.grid(alpha=0.2, lw=0.5)
        ax.annotate(
            "mean SW$_{net}$: "
            + "  ".join(f"{v:.0f}" if np.isfinite(v) else "-" for v in r["swn_mean"])
            + " W m$^{-2}$",
            (0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#555555",
        )
        ax.set_ylim(-0.3, 1.3)
    axes[0][0].set_ylabel(
        f"response to {FORCER_LABEL[forcer]}\n(fraction, energy leaving the surface)"
    )
    # Legend below the panels: inside the axes it covers the October-November
    # "total" markers, which sit exactly where an upper-left legend would go.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6.8,
        frameon=False,
        loc="lower center",
        ncol=min(6, max(3, len(labels) // 3)),
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Month-by-month responses to {FORCER_LABEL[forcer]} -- NSA C1 {ds.attrs.get('period', '')} "
        "(filled: this work, marker size ~ r$^2$; hollow: MOSAiC, Sledd et al. 2025 Fig. 2a)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return _save(fig, out_dir, stem or f"monthly_response_{forcer}")


def fig_control_ladder(
    ds: xr.Dataset,
    masks: Dict[str, np.ndarray],
    forcer: str = "fnet",
    populations: Sequence[str] = ("all", "cloudy", "clear"),
    out_dir=None,
    stem: Optional[str] = None,
):
    """How each response moves as T_2m, then wind, are held fixed."""
    import matplotlib.pyplot as plt

    terms = ["f_lwu", "f_sh", "f_lh", "f_res"]
    controls = list(CONTROL_SETS)
    fig, axes = plt.subplots(
        1,
        len(populations),
        figsize=(4.4 * len(populations), 4.2),
        sharey=True,
        squeeze=False,
    )
    for ax, pop in zip(axes[0], populations):
        for t in terms:
            vals = [partition(ds, masks[pop], forcer, c)[t] for c in controls]
            lab, col, mk = TERM_STYLE[t]
            ax.plot(
                controls,
                vals,
                color=col,
                marker=mk,
                lw=1.4,
                label=lab if ax is axes[0][0] else None,
            )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(POP_LABELS[pop], fontsize=9.5, color=POP_COLORS[pop], loc="left")
        ax.set_xlabel("held fixed in the regression")
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel(f"response to {FORCER_LABEL[forcer]}")
    axes[0][0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(
        "The control ladder: plain regression (Sledd) -> holding T$_{2m}$ -> T$_{2m}$ and wind fixed",
        fontsize=10.5,
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"control_ladder_{forcer}")


def fig_z0_sensitivity(
    rows: List[Dict[str, float]],
    forcer: str = "fnet",
    out_dir=None,
    stem: Optional[str] = None,
):
    """The bulk-flux responses as a function of the roughness length and stable scheme."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scheme, ls in (("grachev2007", "-"), ("beljaars1991", "--")):
        sub = [r for r in rows if r["scheme"] == scheme]
        if not sub:
            continue
        z = [r["z0_m"] for r in sub]
        for t in ("f_sh", "f_lh", "f_res", "f_lwu"):
            lab, col, mk = TERM_STYLE[t]
            ax.plot(
                z,
                [r[t] for r in sub],
                color=col,
                ls=ls,
                marker=mk,
                lw=1.4,
                label=f"{lab} [{scheme}]",
            )
    ax.set_xscale("log")
    ax.set_xlabel("aerodynamic roughness length $z_0$ [m]")
    ax.set_ylabel(f"response to {FORCER_LABEL[forcer]}")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=6.6, frameon=False, ncol=2)
    ax.set_title(
        "Sensitivity of the bulk-flux responses to the free parameters",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"z0_sensitivity_{forcer}")


def fig_thermal_freedom(
    ds: xr.Dataset,
    masks: Dict[str, np.ndarray],
    forcer: str = "fnet",
    populations: Sequence[str] = ("all", "cloudy", "liquid", "clear"),
    out_dir=None,
    stem: Optional[str] = None,
):
    """dT_skin/dF and d(T_skin - T_2m)/dF: what decides the LWU and SH responses."""
    import matplotlib.pyplot as plt

    x = FORCER_KEY[forcer]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    vals = [partition(ds, masks[p], forcer) for p in populations]
    xpos = np.arange(len(populations))
    axes[0].bar(
        xpos - 0.2,
        [v["dskt_dF"] for v in vals],
        0.38,
        color="#B2182B",
        label="d$T_{skin}$/dF",
    )
    axes[0].bar(
        xpos + 0.2,
        [v["dt2m_dF"] for v in vals],
        0.38,
        color="#888888",
        label="d$T_{2m}$/dF",
    )
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([p for p in populations])
    axes[0].set_ylabel("K per W m$^{-2}$")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(
        "(a) thermal freedom of the surface and of the air", fontsize=9.5, loc="left"
    )
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(xpos, [v["ddskt_dF"] for v in vals], 0.5, color="#2A9D8F")
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([p for p in populations])
    axes[1].set_ylabel("K per W m$^{-2}$")
    axes[1].set_title(
        "(b) d($T_{skin} - T_{2m}$)/dF: the sign of the SH response",
        fontsize=9.5,
        loc="left",
    )
    axes[1].grid(axis="y", alpha=0.2)
    m = masks["all"]
    xlabel, xr_ = RESPONDER_AXES[x]
    density_panel(
        axes[2],
        ds[x].values[m],
        ds["t_skin"].values[m],
        xlabel,
        RESPONDER_AXES["t_skin"][0],
        xr_,
        RESPONDER_AXES["t_skin"][1],
        title="(c) skin temperature against the forcer, all sky",
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem or f"thermal_freedom_{forcer}")


def fig_resolution_comparison(
    parts: Dict[str, Dict[str, float]],
    forcer: str = "fnet",
    out_dir=None,
    stem: Optional[str] = None,
):
    """The same partition at several averaging windows (10-min vs hourly)."""
    return fig_partition_bars(
        parts,
        forcer,
        out_dir=out_dir,
        stem=stem or f"resolution_{forcer}",
        title=f"Partition of a {FORCER_LABEL[forcer]} anomaly at 10-min and hourly resolution",
    )


def fig_lwp_vs_dlr(
    ds: xr.Dataset,
    mask: np.ndarray,
    out_dir=None,
    stem: str = "lwd_vs_lwp",
    reference: Optional[Tuple[float, float, str]] = (
        228.26,
        0.27,
        "published Barrow fit y = 0.27x + 228.26",
    ),
):
    """DLR against LWP (the ARM observational relation the ERA5 notebook compares to)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    density_panel(
        ax,
        ds["lwp"].values[mask],
        ds["lwd"].values[mask],
        RESPONDER_AXES["lwp"][0],
        RESPONDER_AXES["lwd"][0],
        RESPONDER_AXES["lwp"][1],
        (100.0, 350.0),
        title="DLR against LWP, NSA C1 (cloudy hours)",
        reference_line=reference,
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem)
