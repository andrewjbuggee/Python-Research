"""Paired-comparison statistics for two estimates of the same quantity.

These describe agreement between two retrievals. Neither is ground truth, so
these are comparisons, not validation. Conventions:

    bias            mean(y - x)                 same units as the data
    relative bias   100 * mean(y - x) / mean(x) percent of the reference x
    RMSD            sqrt(mean((y - x)^2))       root-mean-square difference
    Pearson r       linear correlation (optionally of log10 values)
    RMA slope/int.  reduced-major-axis (geometric-mean) regression line

Why RMA rather than ordinary least squares: OLS assumes x has no error, so
when both x and y are noisy retrievals the OLS slope is biased toward zero
(regression dilution). RMA treats x and y symmetrically:
slope = sign(r) * std(y) / std(x), intercept = mean(y) - slope * mean(x).
Reference: Ricker, W. E. (1973), Linear regressions in fishery research,
J. Fish. Res. Board Can., 30, 409-434.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def paired_stats(x, y, *, log: bool = False) -> Dict[str, float]:
    """Agreement statistics of y against reference x over pairs where both are finite.

    With log=True, r and the RMA fit use log10 values (only positive pairs);
    bias and RMSD stay in linear units. Optical depth spans 1-100+, so a few
    thick clouds otherwise dominate a linear correlation."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if log:
        ok &= (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = {"n": n}
    if n < 3:
        return {
            **out,
            **{
                k: np.nan
                for k in (
                    "mean_x",
                    "mean_y",
                    "bias",
                    "median_bias",
                    "rel_bias_pct",
                    "rmsd",
                    "r",
                    "rma_slope",
                    "rma_intercept",
                )
            },
        }
    d = y - x
    out.update(
        mean_x=float(x.mean()),
        mean_y=float(y.mean()),
        bias=float(d.mean()),
        median_bias=float(np.median(d)),
        rel_bias_pct=float(100 * d.mean() / x.mean()) if x.mean() != 0 else np.nan,
        rmsd=float(np.sqrt(np.mean(d**2))),
    )
    fx, fy = (np.log10(x), np.log10(y)) if log else (x, y)
    sx, sy = fx.std(ddof=1), fy.std(ddof=1)
    r = float(np.corrcoef(fx, fy)[0, 1]) if sx > 0 and sy > 0 else np.nan
    slope = float(np.sign(r) * sy / sx) if np.isfinite(r) and sx > 0 else np.nan
    out.update(r=r, rma_slope=slope, rma_intercept=float(fy.mean() - slope * fx.mean()))
    out["log"] = bool(log)
    return out


def format_stats(s: Dict[str, float], units: str = "") -> str:
    """Multi-line text for a figure annotation."""
    if s["n"] < 3:
        return f"N = {s['n']} (too few pairs)"
    u = f" {units}" if units else ""
    r_label = "r (log10)" if s.get("log") else "r"
    # A percentage of a reference mean smaller than the scatter is meaningless
    # (e.g. LWP near zero), so it is shown only when |mean_x| > RMSD.
    rel = f" ({s['rel_bias_pct']:+.0f}%)" if abs(s["mean_x"]) > s["rmsd"] else ""
    return (
        f"N = {s['n']:,}\n"
        f"bias = {s['bias']:+.2f}{u}{rel}\n"
        f"RMSD = {s['rmsd']:.2f}{u}\n"
        f"{r_label} = {s['r']:.2f}"
    )
