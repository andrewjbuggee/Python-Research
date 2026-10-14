#!/usr/bin/env python
"""Digitize Dié Wang's monthly-accumulation bars and compare them with ours.

Dié's newest rain figure (sent 2026-10-12, saved as
data/processed/Monthly Precip - 2DVD - data from Wang Die/
monthly_rain_rate_and_accumulation.png) has two panels: (a) 1-min rain rate
box plots, (b) monthly rainfall accumulation (mm) as bars for LD_M1, VDIS_M1
and LD_S1. Only the PNG was shared, not the numbers, so the bar heights are
read back from the pixels here. (Figure A3 in EPCAPEoverviewFigs261005.pptx is
her OLDER version, whose panel (b) is daily box plots, so it cannot be
compared with monthly totals.)

How the digitizing works (all automatic, nothing clicked by hand)
  1. Axes frames = long dark pixel lines. Panel (b) is the lower frame.
  2. y calibration: the y tick marks are short dark lines just left of the
     left spine. Their pixel rows are fitted (least squares, linear) to the
     tick labels 0, 20, ..., 140 mm read off the figure (Y_TICKS_MM). The fit
     residual is reported as a check.
  3. x: the x tick marks below the bottom spine give the 13 month centres.
  4. Bars: each instrument's fill colour (matplotlib colour at alpha 0.8
     over white, e.g. royalblue (65,105,225) -> (103,135,231)) is masked.
     A bar is a run of adjacent pixel columns whose fill reaches down to the
     bottom spine (this excludes the legend patches, which float). The bar's
     data value is the centre of its black top edge line (matplotlib draws
     the edge centred on the data boundary): top of fill minus half the edge
     thickness measured in that column.
  5. Each bar goes to the month whose x tick is nearest its centre.

Resolution: about 13 px per mm, so a bar is read to roughly ±0.1 mm. Bars
shorter than about twice the edge width are drawn as solid black and have no
fill to find; they are reported as NaN with flag "below_resolution".

compare_monthly() then gives a table: Dié (digitized), ours, difference
(mm), percent difference 100 (ours - Dié) / Dié, and whether the difference
is within the digitizing resolution.

Run from the EPCAPE folder:
  python variables/precipitation/digitize_die_figure.py
Needs data/processed/derived/monthly_rain_accumulation/ (run
monthly_rain_accumulation.py first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# EPCAPE/ is the package: put its parent folder on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from EPCAPE.download_data.config import active_machine, load_config  # noqa: E402

# Tick labels on panel (b)'s y axis, bottom to top (read off the figure).
Y_TICKS_MM: Tuple[float, ...] = (0, 20, 40, 60, 80, 100, 120, 140)
# Month tick labels, left to right.
MONTHS: Tuple[str, ...] = tuple(
    f"{t:%Y-%m}" for t in pd.date_range("2023-02-01", "2024-02-01", freq="MS")
)
# Bar fill RGB as rendered (matplotlib named colour at alpha 0.8 over white).
FILL_RGB: Dict[str, Tuple[int, int, int]] = {
    "LD_M1": (140, 153, 166),  # slategray (112,128,144)
    "VDIS_M1": (255, 183, 51),  # orange (255,165,0)
    "LD_S1": (103, 135, 231),  # royalblue (65,105,225)
}
COLOR_TOL = 12  # max per-channel difference to count as the fill colour
DARK_SUM = 200  # R+G+B below this = black ink (edges, spines, ticks)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def runs(idx: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs [first, last] in a sorted integer array."""
    if idx.size == 0:
        return []
    cut = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def find_frames(dark: np.ndarray) -> List[Dict[str, int]]:
    """Axes frames as dicts top/bottom/left/right (inner edge of each spine).

    Spines are the only dark lines spanning most of the image width
    (horizontal) or a quarter of its height (vertical)."""
    h, w = dark.shape
    hlines = runs(np.where(dark.sum(1) > 0.6 * w)[0])
    vlines = runs(np.where(dark.sum(0) > 0.25 * h)[0])
    if len(hlines) % 2 or len(vlines) != 2:
        raise RuntimeError(f"unexpected frame lines: rows {hlines}, cols {vlines}")
    left, right = vlines[0][1] + 1, vlines[1][0] - 1
    return [
        {"top": hlines[i][1] + 1, "bottom": hlines[i + 1][0] - 1, "left": left, "right": right,
         "bottom_spine": hlines[i + 1]}
        for i in range(0, len(hlines), 2)
    ]


def y_calibration(dark: np.ndarray, frame: Dict[str, int], tick_values: Sequence[float]) -> Tuple[np.poly1d, float]:
    """Linear map pixel row -> mm from the y tick marks left of the spine.

    Returns (row_to_mm, rms residual in mm)."""
    spine_left = frame["left"] - 1
    # tick marks: a short dark run ending at the spine; sample a column ~10 px left
    col = dark[frame["top"] - 10 : frame["bottom"] + 10, spine_left - 15]
    centres = np.array([(a + b) / 2 for a, b in runs(np.where(col)[0])]) + frame["top"] - 10
    if centres.size != len(tick_values):
        raise RuntimeError(f"found {centres.size} y ticks, expected {len(tick_values)}")
    vals = np.asarray(sorted(tick_values, reverse=True), float)  # rows grow downward
    coef = np.polyfit(centres, vals, 1)
    resid = vals - np.polyval(coef, centres)
    return np.poly1d(coef), float(np.sqrt(np.mean(resid**2)))


def x_tick_centres(dark: np.ndarray, frame: Dict[str, int]) -> np.ndarray:
    """Columns of the x tick marks just below the bottom spine."""
    row = dark[frame["bottom_spine"][1] + 8, frame["left"] : frame["right"]]
    return np.array([(a + b) / 2 for a, b in runs(np.where(row)[0])]) + frame["left"]


# ---------------------------------------------------------------------------
# digitizing
# ---------------------------------------------------------------------------
def digitize_bars(png: Path, panel: int = 1, y_ticks: Sequence[float] = Y_TICKS_MM,
                  months: Sequence[str] = MONTHS) -> Tuple[pd.DataFrame, dict]:
    """Bar heights (mm) in panel `panel` (0 = top) of Dié's figure.

    Returns (long table: month, instrument, die_mm, flag, x_px, top_px; info)."""
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)  # (rows, cols, 3)
    dark = im.sum(axis=2) < DARK_SUM
    frame = find_frames(dark)[panel]
    row_to_mm, y_rms = y_calibration(dark, frame, y_ticks)
    xticks = x_tick_centres(dark, frame)
    if xticks.size != len(months):
        raise RuntimeError(f"found {xticks.size} x ticks, expected {len(months)}")
    mm_per_px = abs(row_to_mm.coeffs[0])

    rows = []
    bottom = frame["bottom"]
    for name, rgb in FILL_RGB.items():
        fill = (np.abs(im - np.array(rgb)).max(axis=2) <= COLOR_TOL)
        fill[:, : frame["left"]] = False
        fill[:, frame["right"] :] = False
        fill[: frame["top"]] = False
        fill[bottom + 1 :] = False
        # columns whose fill is present in the band just above the bottom edge
        # line: bars stand on the axis, legend patches do not
        band = fill[bottom - 40 : bottom + 1].any(axis=0)
        for c0, c1 in runs(np.where(band)[0]):
            cols = np.arange(c0, c1 + 1)
            core = cols[len(cols) // 4 : len(cols) - len(cols) // 4]  # avoid side edges
            tops = []
            for c in core:
                f = np.where(fill[: bottom + 1, c])[0]
                # contiguous fill run that touches the bottom band
                r = [run for run in runs(f) if run[1] >= bottom - 40][-1]
                top_fill = r[0]
                # black edge directly above the fill; its centre is the data value
                k = top_fill - 1
                while k > frame["top"] and dark[k, c]:
                    k -= 1
                tops.append((k + 1 + top_fill - 1) / 2.0)  # centre of edge run
            top_px = float(np.median(tops))
            x = (c0 + c1) / 2
            month = months[int(np.argmin(np.abs(xticks - x)))]
            rows.append({"month": month, "instrument": name, "die_mm": float(row_to_mm(top_px)),
                         "flag": "", "x_px": x, "top_px": top_px})

    out = pd.DataFrame(rows)
    # months/instruments with no bar found -> NaN; a bar may be absent (0 mm) or
    # too short to have a visible fill
    full = pd.MultiIndex.from_product([months, list(FILL_RGB)], names=["month", "instrument"])
    out = out.set_index(["month", "instrument"])
    if out.index.duplicated().any():
        raise RuntimeError(f"two bars assigned to one month: {out.index[out.index.duplicated()].tolist()}")
    out = out.reindex(full)
    out.loc[out["die_mm"].isna(), "flag"] = "no_bar_or_below_resolution"
    info = {"frame": frame, "y_fit_rms_mm": y_rms, "mm_per_px": mm_per_px, "png": str(png)}
    return out.reset_index(), info


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare_monthly(die: pd.DataFrame, ours: pd.DataFrame, label: str, tol_mm: float) -> pd.DataFrame:
    """Join Dié's digitized bars with our monthly totals.

    die  : long table month, instrument, die_mm, flag (from digitize_bars)
    ours : wide table, index month, one column per instrument (mm)
    tol_mm : digitizing uncertainty; |diff| <= tol_mm counts as agreement

    Percent difference = 100 (ours - Dié) / Dié. When Dié's bar is below
    tol_mm, percent differences are dominated by reading error and are left
    out (pct_diff NaN); the mm difference is still given."""
    o = ours.rename_axis("month").reset_index().melt(id_vars="month", var_name="instrument", value_name="ours_mm")
    t = die.merge(o, on=["month", "instrument"], how="left")
    t.insert(0, "ours_source", label)
    t["diff_mm"] = t["ours_mm"] - t["die_mm"]
    big = t["die_mm"] > tol_mm
    t["pct_diff"] = np.where(big, 100.0 * t["diff_mm"] / t["die_mm"], np.nan)
    t["within_reading_error"] = t["diff_mm"].abs() <= tol_mm
    # Dié shows no bar: agreement if ours is also ~0 or missing
    nobar = t["die_mm"].isna()
    t.loc[nobar, "within_reading_error"] = t.loc[nobar, "ours_mm"].fillna(0).abs() <= tol_mm
    return t[["ours_source", "month", "instrument", "die_mm", "ours_mm", "diff_mm", "pct_diff",
              "within_reading_error", "flag"]]


def main(argv=None) -> int:
    cfg = load_config()
    processed = active_machine(cfg).processed_dir()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--png", type=Path,
                   default=processed / "Monthly Precip - 2DVD - data from Wang Die" / "monthly_rain_rate_and_accumulation.png")
    p.add_argument("--ours-dir", type=Path, default=processed / "derived" / "monthly_rain_accumulation")
    p.add_argument("--tol-mm", type=float, default=0.3,
                   help="digitizing uncertainty (mm); default ~2 px plus the y-fit residual")
    args = p.parse_args(argv)

    die, info = digitize_bars(args.png)
    print(f"Digitized {args.png.name}: {info['mm_per_px']:.4f} mm per pixel, "
          f"y-tick fit rms {info['y_fit_rms_mm']:.3f} mm")

    # Our two sets of monthly totals: Dié's own algorithm on our download, and
    # the default product of monthly_rain_accumulation.py (coverage-screened,
    # VDIS per Zeen Zhu). The first should match her bars to reading error;
    # the second differs where the methods differ (VDIS Oct-Feb).
    die_method = pd.read_csv(args.ours_dir / "monthly_rain_accumulation_die_method.csv", index_col=0)
    wide = pd.read_csv(args.ours_dir / "monthly_rain_accumulation_wide.csv", index_col=0)
    default = wide[[f"{n}_accum_mm" for n in FILL_RGB]]
    default.columns = list(FILL_RGB)

    table = pd.concat([
        compare_monthly(die, die_method[list(FILL_RGB)], "Dié's method, our data", args.tol_mm),
        compare_monthly(die, default, "monthly_rain_accumulation.py default", args.tol_mm),
    ], ignore_index=True)
    out = args.ours_dir / "die_figure_vs_ours.csv"
    table.to_csv(out, index=False, float_format="%.2f")

    pd.set_option("display.width", 200)
    for src, t in table.groupby("ours_source", sort=False):
        print(f"\n=== {src} ===")
        print(t.drop(columns="ours_source").round(2).to_string(index=False))
        ok = t["within_reading_error"].sum()
        print(f"{ok}/{len(t)} within ±{args.tol_mm} mm; "
              f"median |pct diff| {t['pct_diff'].abs().median():.2f}% (bars > {args.tol_mm} mm)")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
