#!/usr/bin/env python
"""Do the FM-120 and AMS cloud/haze sampling hours cover the same times, and when did each instrument have no data?

Why: the seasonal-averages table (EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx, sheet Cloud&MetQuantities) gives
two sets of on-site cloud and haze sampling hours at Mt. Soledad:
  rows 31-32  "On-Site Cloud/Haze Sampling by Fog Drop Monitor"  (FM-120 + visibility, 5-min intervals)
  rows 34-35  "On-Site Cloud/Haze Sampling for AMS Residuals"    (Abbey Williams' 15-min classes + AMS + GCVI EF)
This script puts both on one 5-min time grid and splits them into the hours both sets hold (the intersection) and
the hours only one holds (the two complements). For every complement hour it says why the other set does not hold
it, and in particular whether the other instrument had no data. It also lists when each instrument had no data.

Data (paths from config.yaml; the campaign window is 15 Feb 2023 00:00 to 15 Feb 2024 00:00 UTC):
  FM-120        daily 5-min files from Dalhousie (sources.load_fm120): one row per 5-min interval the instrument
                ran, labelled by the interval START; lwc_gm3 (liquid water content, g m-3), nd_cm3 (2-50 um).
  visibility    msd_visibility_*.csv (sources.load_visibility_msd): ~1-s Mt. Soledad visibility (m), from
                24 Feb 2023.
  15-min file   Abbey Williams' EPCAPE_15mindata.nc (kavin.read_kavin_15min), 15-min steps labelled by START:
                ClrCldHaz_class (1 cloud, 2 haze, only while the CVI is on; 0 CVI off; NaN when CVIon_class = 0.5),
                CVIon_class (1 on, 0 off, 0.5 = CVI switching within the interval, or state unknown: the first
                0.5 block runs from 15 Feb to 10 Mar 2023), AMS_mass (5 species, ug m-3).
  AMS           AMS_CE_corrected_msptof.nc (sources.load_ams_soledad): one sample per ~5.6-min AMS cycle, time =
                START of its 2-min V-mode period. Flagged data were removed before distribution (file README).
  GCVI          bb2743661p_36_1_CVI_EF.csv (sources.load_gcvi): 1-s enhancement factor EF, logged only while the
                GCVI runs.

Definitions (the same as the table rows; the script checks that it reproduces them):
  FM-120 class of a 5-min slot (rows 31-32; Lynn Russell's EPCAPE BAMS-paper definitions, notebook Section 9):
      visibility = median of the slot's valid 1-s samples (0 < vis <= 48,270 m; >= 125 samples, else none)
      cloud    visibility < 1 km and LWC > 0.01 g m-3
      haze     visibility < 5 km and LWC < 0.01 g m-3
      neither  visibility < 5 km but neither cloud nor haze (1-5 km with LWC >= 0.01 g m-3)
      clear    visibility >= 5 km
  AMS class of a 15-min interval (rows 34-35; notebook Sections 10e and 16):
      cloud (haze)   ClrCldHaz_class = 1 (2), AMS_mass present, and >= 1 s with EF > 1 in the 1-s GCVI record
  Each 15-min interval is three 5-min slots, so both sets are counted on the same 5-min grid and every hour total
  below is (number of slots) x 5 min. The FM-120 rows sit exactly on this grid (checked at run time).
  "No data":
      FM-120         no 5-min row with both nd_cm3 and lwc_gm3 in the daily files. Rows exist only while the
                     instrument ran, and the monthly hours of operation reproduce Table 1 of the PI's report
                     (DOE/SC-ARM-24-023; notebook Section 9), so here "no data" means "not running".
      visibility     fewer than 125 valid 1-s samples in the 5-min slot
      AMS            no AMS sample in the slot's 15-min interval. The 15-min interval is used because one AMS cycle
                     (~5.6 min) is longer than a slot. The distributed file has flagged data removed, so "no data"
                     means off, calibrating, OR recording but removed in QC; the file cannot tell these apart. The
                     table's DataAvailability sheet (HR-ToF-AMS row) has much higher availability in the months it
                     flags heavily (May 2023: 71 % available, 47 % flagged; this file 41 %), so some AMS "no data"
                     time was recorded but removed (my inference from those two numbers).
  Why a complement slot is missing from the other set: the first reason in the lists below that applies.
      FM-120 class X, not AMS class X:  AMS no data (CVI on) | AMS no data (CVI off or 0.5) |
          CVI off | CVIon_class = 0.5 | CVI on, Abbey's class is the other class | CVI on, class X, no AMS_mass in
          Abbey's file | CVI on, class X, no EF logged
      AMS class X, not FM-120 class X:  FM-120 no data | no usable visibility | FM-120 says the other class |
          FM-120 says neither | FM-120 says clear
  Seasons are the sheet's (seasonal.SEASONS, UTC): Spring AMJ 2023, Summer JAS 2023, Fall OND 2023, Winter
  15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024.

Outputs (in sampling_overlap/, next to this script; --out chooses another folder):
  overlap_hours_by_season.csv        rows 31/34 (cloud) and 32/35 (haze): totals, both, FM-120 only, AMS only
  fm120_only_reasons_by_season.csv   why FM-120 cloud/haze slots are not AMS cloud/haze, hours per season
  ams_only_reasons_by_season.csv     why AMS cloud/haze slots are not FM-120 cloud/haze, hours per season
  state_crosstab_hours.csv           every FM-120 state against every AMS state, hours over the campaign
  no_data_hours_by_season.csv        hours with no data: FM-120, visibility, AMS, FM-120 and AMS both
  no_data_periods.csv                every period with no data (start, end, hours), per instrument and both
  sampling_overlap_timeline.png/.pdf the campaign timeline of all of the above
  slots.csv                          (with --save-slots) the state of every 5-min slot, ~105,000 rows

Examples (from the Python-Research folder)
  python EPCAPE/comparisons/seasonal_averages/fm120_vs_ams_sampling_overlap.py
  python EPCAPE/comparisons/seasonal_averages/fm120_vs_ams_sampling_overlap.py --min-period-h 12 --save-slots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
REPO = Path(__file__).resolve().parents[2]  # the EPCAPE folder
sys.path.insert(0, str(REPO.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from EPCAPE.comparisons.seasonal_averages import kavin as K  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import sources as SRC  # noqa: E402
from EPCAPE.plotting import GRID, INK, INK_2, INK_3, concise_dates, use_style  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: the same values as check_seasonal_averages.ipynb, so the totals equal the table rows
# ---------------------------------------------------------------------------
KAVIN_DIR = Path(os.environ.get("EPCAPE_KAVIN_DIR", REPO / "Kavin_data_for_BAMS_paper")).expanduser()
SLOT_MIN = 5.0  # grid step (minutes) = the FM-120 averaging interval
AMS_INTERVAL_MIN = 15.0  # Abbey's interval length (minutes) = 3 slots
SLOT_H = SLOT_MIN / 60.0  # hours represented by one slot
FM120_CLOUD_VIS_MAX_M = 1000.0  # cloud: median visibility < 1 km ...
FM120_CLOUD_LWC_MIN_GM3 = 0.01  # ... and LWC > 0.01 g m-3
FM120_HAZE_VIS_MAX_M = 5000.0  # haze: median visibility < 5 km ...
FM120_HAZE_LWC_MAX_GM3 = 0.01  # ... and LWC < 0.01 g m-3
FM120_VIS_MIN_SAMPLES = 125  # a slot needs >= 125 valid 1-s visibility samples (a full slot has ~250)

# The table's values (EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx, columns E-I = EPCAPE, Spring, Summer, Fall,
# Winter), used only to check that this script counts the same hours.
TABLE_HOURS = {
    "row 31, FM-120 cloud": (480, 137, 217, 66.8, 58.8),
    "row 32, FM-120 haze": (395, 77.6, 161, 56, 100),
    "row 34, AMS cloud": (248, 68.2, 136, 41.2, 2.75),
    "row 35, AMS haze": (135, 24, 65.8, 30, 15.5),
}

# Reason labels (the order of each tuple is the order in which reasons are tested: the first that applies wins)
AMS_OFF_CVI_ON = "AMS no data, CVI on"
AMS_OFF_CVI_OTHER = "AMS no data, CVI off or 0.5"
CVI_OFF = "CVI off (AMS sampling ambient air)"
CVI_HALF = "CVIon_class = 0.5 (switching or unknown)"
OTHER_CLASS = "CVI on, Abbey's class = {other}"
NO_MASS = "CVI on, {kind}, no AMS_mass in Abbey's file"
NO_EF = "CVI on, {kind}, no EF logged"
FM_OFF = "FM-120 no data (not running)"
NO_VIS = "FM-120 recording, no usable visibility"
FM_OTHER = "FM-120 says {other}"
FM_NEITHER = "FM-120 says neither (vis < 5 km, LWC >= 0.01)"
FM_CLEAR = "FM-120 says clear (vis >= 5 km)"


# ---------------------------------------------------------------------------
# The common 5-min grid
# ---------------------------------------------------------------------------
def campaign_grid() -> pd.DatetimeIndex:
    """5-min slot START times over the sheet's campaign window (15 Feb 2023 00:00 to 14 Feb 2024 23:55 UTC)."""
    first_day, last_day = S.SEASONS["EPCAPE"][0]
    end = pd.Timestamp(last_day) + pd.Timedelta(days=1)  # the whole last day is included
    return pd.date_range(first_day, end, freq=f"{SLOT_MIN:g}min", inclusive="left", name="time")


def hours_by_season(mask: pd.Series) -> pd.Series:
    """Hours per sheet season of the True slots in `mask` (indexed by the grid)."""
    return S.seasonal_sums(mask.astype(float) * SLOT_H)


def runs(mask: pd.Series) -> pd.DataFrame:
    """Contiguous runs of True slots: start (first slot's start), end (last slot's end), hours."""
    m = mask.to_numpy().astype(np.int8)
    step = np.diff(np.r_[0, m, 0])  # +1 where a run starts, -1 one past where it ends
    first, past_last = np.flatnonzero(step == 1), np.flatnonzero(step == -1)
    return pd.DataFrame({"start": mask.index[first],
                         "end": mask.index[past_last - 1] + pd.Timedelta(minutes=SLOT_MIN),
                         "hours": (past_last - first) * SLOT_H})


# ---------------------------------------------------------------------------
# FM-120 side (rows 31-32)
# ---------------------------------------------------------------------------
def fm120_states(grid: pd.DatetimeIndex) -> pd.DataFrame:
    """FM-120 state of every 5-min slot, with the table's BAMS-paper cloud/haze definitions.

    Returns a DataFrame on `grid`:
      fm_recording  the daily files have a row with both nd_cm3 and lwc_gm3 for the slot
      vis_ok        the slot has a usable median visibility (>= FM120_VIS_MIN_SAMPLES valid 1-s samples)
      vis_median_m  that visibility (m)
      fm_state      'cloud', 'haze', 'neither', 'clear', 'no visibility' or 'no data'"""
    fm = SRC.load_fm120()
    # the FM-120 averages are labelled by interval start on a 5-min clock (00:00, 00:05, ...); if a row ever sat off
    # that clock, putting it on the grid by its label would shift it, so stop instead
    off_grid = int((fm.index != fm.index.floor(f"{SLOT_MIN:g}min")).sum())
    if off_grid:
        raise ValueError(f"{off_grid} FM-120 rows are not on the {SLOT_MIN:g}-min clock")

    # visibility of each FM-120 interval: median of the valid ~1-s samples in [t, t + 5 min)
    vis_1s_m = SRC.load_visibility_msd()
    vis = Q.visibility_on_intervals(vis_1s_m, grid, interval_min=SLOT_MIN, min_samples=FM120_VIS_MIN_SAMPLES)
    del vis_1s_m  # ~25 million samples; only the 5-min medians are needed

    # classes on the FM-120 rows (quantities.fm120_classify_visibility = notebook Section 9, step 5)
    cls = Q.fm120_classify_visibility(
        fm, vis["vis_median_m"].reindex(fm.index), cloud_vis_max_m=FM120_CLOUD_VIS_MAX_M,
        cloud_lwc_min_gm3=FM120_CLOUD_LWC_MIN_GM3, haze_vis_max_m=FM120_HAZE_VIS_MAX_M,
        haze_lwc_max_gm3=FM120_HAZE_LWC_MAX_GM3)
    cls = cls.reindex(grid, fill_value=False)  # slots with no FM-120 row: no data, no class

    out = pd.DataFrame({"fm_recording": cls["operating"], "vis_ok": vis["vis_median_m"].notna(),
                        "vis_median_m": vis["vis_median_m"]}, index=grid)
    # one state per slot; later assignments overwrite earlier ones, so the most specific comes last
    state = pd.Series("clear", index=grid, dtype=object)  # recording, visibility >= 5 km
    state[cls["operating"] & (out["vis_median_m"] < FM120_HAZE_VIS_MAX_M)] = "neither"
    state[cls["haze"]] = "haze"
    state[cls["cloud"]] = "cloud"
    state[cls["operating"] & ~out["vis_ok"]] = "no visibility"
    state[~cls["operating"]] = "no data"
    out["fm_state"] = state
    return out


# ---------------------------------------------------------------------------
# AMS side (rows 34-35)
# ---------------------------------------------------------------------------
def ams_states(grid: pd.DatetimeIndex) -> pd.DataFrame:
    """AMS-residual state of every 5-min slot, taken from the 15-min interval that holds it.

    Returns a DataFrame on `grid`:
      ams_recording  >= 1 AMS sample (CE-corrected file) in the 15-min interval
      cvi_on_class   Abbey's CVIon_class (1 on, 0 off, 0.5 switching/unknown)
      abbey_class    Abbey's ClrCldHaz_class (1 cloud, 2 haze, 0 CVI off, NaN when CVIon_class = 0.5)
      has_ams_mass   AMS_mass present in Abbey's file for the interval
      ef_logged      >= 1 s with EF > 1 in the 1-s GCVI record during the interval
      ams_state      'cloud' or 'haze' (rows 34-35) or '' (neither)"""
    f15, _ = K.read_kavin_15min(KAVIN_DIR)  # Abbey's 15-min file (the FM-120 size distributions are not needed)
    interval = f"{AMS_INTERVAL_MIN:g}min"

    # AMS recording: any sample of the CE-corrected file whose V-mode START falls in the interval
    ams = SRC.load_ams_soledad()
    n_ams = pd.Series(1, index=ams.index.floor(interval)).groupby(level=0).size()
    # seconds with EF > 1 in each interval (the "no EF logged" filter of notebook Section 16)
    gcvi = SRC.load_gcvi()
    ef_s = gcvi["EF"].where(gcvi["EF"] > 1).resample(interval).count()
    del ams, gcvi  # the 1-s GCVI record has ~2.5 million rows

    has_mass = f15[[f"{s}_ugm3" for s in K.AMS_15MIN_SPECIES]].notna().any(axis=1)
    per_interval = pd.DataFrame({
        "ams_recording": n_ams.reindex(f15.index).fillna(0) > 0,
        "cvi_on_class": f15["cvi_on_class"],
        "abbey_class": f15["clr_cld_haz_class"],
        "has_ams_mass": has_mass,
        "ef_logged": ef_s.reindex(f15.index).fillna(0) > 0,
    })
    # every slot takes the values of the 15-min interval holding it
    out = per_interval.reindex(grid.floor(interval))
    out.index = grid
    for col in ("ams_recording", "has_ams_mass", "ef_logged"):
        out[col] = out[col].fillna(False).astype(bool)

    in_rows = out["has_ams_mass"] & out["ef_logged"]  # the AMS-residual sample of rows 34-35
    state = pd.Series("", index=grid, dtype=object)
    state[(out["abbey_class"] == 1) & in_rows] = "cloud"
    state[(out["abbey_class"] == 2) & in_rows] = "haze"
    out["ams_state"] = state
    return out


def ams_detail(a: pd.DataFrame) -> pd.Series:
    """One label per slot describing the AMS side in full (for the cross-tabulation)."""
    label = pd.Series("CVI off", index=a.index, dtype=object)
    label[a["cvi_on_class"] == 0.5] = "CVIon_class = 0.5"
    for code, kind in ((1, "cloud"), (2, "haze")):
        is_kind = (a["abbey_class"] == code)
        label[is_kind & ~a["has_ams_mass"]] = f"CVI on, {kind}, no AMS_mass"
        label[is_kind & a["has_ams_mass"] & ~a["ef_logged"]] = f"CVI on, {kind}, no EF"
        label[a["ams_state"] == kind] = f"AMS {kind} (row {34 if kind == 'cloud' else 35})"
    label[~a["ams_recording"] & (a["cvi_on_class"] == 1)] = AMS_OFF_CVI_ON
    label[~a["ams_recording"] & (a["cvi_on_class"] != 1)] = AMS_OFF_CVI_OTHER
    return label


# ---------------------------------------------------------------------------
# Why a complement slot is missing from the other set
# ---------------------------------------------------------------------------
def reasons_fm120_only(a: pd.DataFrame, kind: str) -> List[Tuple[str, pd.Series]]:
    """(reason, mask) pairs, in test order, for slots missing from the AMS `kind` set."""
    code, other = (1, "haze") if kind == "cloud" else (2, "cloud")
    return [
        (AMS_OFF_CVI_ON, ~a["ams_recording"] & (a["cvi_on_class"] == 1)),
        (AMS_OFF_CVI_OTHER, ~a["ams_recording"]),
        (CVI_OFF, a["cvi_on_class"] == 0),
        (CVI_HALF, a["cvi_on_class"] == 0.5),
        (OTHER_CLASS.format(other=other), a["abbey_class"] == 3 - code),
        (NO_MASS.format(kind=kind), (a["abbey_class"] == code) & ~a["has_ams_mass"]),
        (NO_EF.format(kind=kind), (a["abbey_class"] == code) & ~a["ef_logged"]),
    ]


def reasons_ams_only(f: pd.DataFrame, kind: str) -> List[Tuple[str, pd.Series]]:
    """(reason, mask) pairs, in test order, for slots missing from the FM-120 `kind` set."""
    other = "haze" if kind == "cloud" else "cloud"
    return [
        (FM_OFF, f["fm_state"] == "no data"),
        (NO_VIS, f["fm_state"] == "no visibility"),
        (FM_OTHER.format(other=other), f["fm_state"] == other),
        (FM_NEITHER, f["fm_state"] == "neither"),
        (FM_CLEAR, f["fm_state"] == "clear"),
    ]


def attribute(complement: pd.Series, reasons: List[Tuple[str, pd.Series]]) -> pd.DataFrame:
    """Hours per season of the `complement` slots, split by the first reason that applies to each.

    Every complement slot must get a reason; an 'unexplained' row would show a gap in the logic."""
    left = complement.copy()
    rows = {}
    for label, mask in reasons:
        hit = left & mask
        rows[label] = hours_by_season(hit)
        left &= ~hit
    rows["unexplained"] = hours_by_season(left)
    table = pd.DataFrame(rows).T[S.SEASON_ORDER]
    table.loc["total"] = table.sum()
    return table


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Lane colours: neutral greys for "no data", one hue per class. The three EPCAPE instrument hues
# (plotting.COLORS: blue, orange, aqua) are reserved for MFRSR, sunphotometer and MWR, so they are not used here.
CLOUD_COLOR = "#5b4bb7"  # violet
HAZE_COLOR = "#c99700"  # dark gold (lighter than the violet, so the pair also differs in lightness)


def plot_timeline(lanes: List[Tuple[str, pd.Series, str]], out_dir: Path) -> None:
    """One horizontal lane per state; every contiguous run of True slots is drawn as a bar."""
    use_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    # lanes top to bottom; a blank lane after the "no data" group and after the cloud group
    y, y_ticks, y_labels = 0.0, [], []
    for label, mask, color in lanes:
        if label is None:  # group separator
            y -= 0.6
            continue
        r = runs(mask)
        start_num = mdates.date2num(r["start"].to_numpy())
        width_days = r["hours"].to_numpy() / 24.0
        # a thin edge in the same colour keeps single 5-min slots visible at this time scale
        ax.broken_barh(list(zip(start_num, width_days)), (y - 0.35, 0.7), facecolors=color, edgecolors=color,
                       linewidth=0.3)
        y_ticks.append(y)
        y_labels.append(f"{label}: {r['hours'].sum():,.0f} h")
        y -= 1.0
    ax.set_yticks(y_ticks, y_labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(False)  # the season lines are the only vertical guides (month gridlines would mix with them)
    ax.set_ylim(y + 0.4, 0.6)

    # season boundaries (sheet seasons), named along the top
    for first, _ in (r for s in ("Spring", "Summer", "Fall") for r in S.SEASONS[s]):
        ax.axvline(pd.Timestamp(first), color=INK_3, lw=0.8, zorder=0)
    ax.axvline(pd.Timestamp(S.SEASONS["Winter"][1][0]), color=INK_3, lw=0.8, zorder=0)
    for name, (first, last) in (("Winter", ("2023-02-15", "2023-03-31")), ("Spring", S.SEASONS["Spring"][0]),
                                ("Summer", S.SEASONS["Summer"][0]), ("Fall", S.SEASONS["Fall"][0]),
                                ("Winter", ("2024-01-01", "2024-02-14"))):
        mid = pd.Timestamp(first) + (pd.Timestamp(last) - pd.Timestamp(first)) / 2
        ax.text(mid, 0.75, name, ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_xlim(pd.Timestamp("2023-02-15"), pd.Timestamp("2024-02-15"))
    concise_dates(ax)
    ax.set_title("Mt. Soledad: FM-120 and AMS cloud/haze sampling, and data gaps (5-min slots, UTC)",
                 loc="left", color=INK, pad=18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sampling_overlap_timeline.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "sampling_overlap",
                        help="output folder (default: sampling_overlap/ next to this script)")
    parser.add_argument("--min-period-h", type=float, default=24.0,
                        help="print no-data periods at least this long (hours); the CSV lists all of them")
    parser.add_argument("--save-slots", action="store_true", help="also write the state of every 5-min slot")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)

    # --- 1. States of every 5-min slot ----------------------------------------------------------------------
    grid = campaign_grid()
    f = fm120_states(grid)
    a = ams_states(grid)
    print(f"{len(grid):,} slots of {SLOT_MIN:g} min, {grid[0]} to {grid[-1]} UTC")

    sets: Dict[str, Dict[str, pd.Series]] = {
        kind: {"FM-120": f["fm_state"] == kind, "AMS": a["ams_state"] == kind} for kind in ("cloud", "haze")}

    # --- 2. Check: the four sets give the table's hours ----------------------------------------------------------
    # The table holds 3 significant figures (format "%.3g", which rounds a half to the even digit: 100.5 -> 100),
    # so the script's hours are compared after the same rounding.
    check, all_match = {}, True
    for (name, table_values), (kind, inst) in zip(TABLE_HOURS.items(), [("cloud", "FM-120"), ("haze", "FM-120"),
                                                                          ("cloud", "AMS"), ("haze", "AMS")]):
        ours = hours_by_season(sets[kind][inst])
        rounded = np.array([float(f"{v:.3g}") for v in ours])
        match = np.allclose(rounded, table_values)
        all_match &= match
        check[(name, "this script")] = list(ours.round(2).to_numpy()) + [""]
        check[(name, "table")] = list(table_values) + ["yes" if match else "NO"]
    check = pd.DataFrame(check, index=S.SEASON_ORDER + ["matches"]).T
    print("\nCheck against EPCAPE_SeasonalAveragesTable_20_Oct_2026.xlsx (hours; 'matches' after rounding to 3 "
          "significant figures):")
    print(check.to_string())
    if not all_match:
        print("WARNING: at least one row differs from the table; the definitions or the data have changed")

    # --- 3. Intersection and complements, per class ------------------------------------------------------------
    overlap_rows, fm_only_tables, ams_only_tables = {}, {}, {}
    for kind, row_fm, row_ams in (("cloud", 31, 34), ("haze", 32, 35)):
        fm_set, ams_set = sets[kind]["FM-120"], sets[kind]["AMS"]
        overlap_rows[(kind, f"FM-120 total (row {row_fm})")] = hours_by_season(fm_set)
        overlap_rows[(kind, f"AMS total (row {row_ams})")] = hours_by_season(ams_set)
        overlap_rows[(kind, "both (intersection)")] = hours_by_season(fm_set & ams_set)
        overlap_rows[(kind, "FM-120 only")] = hours_by_season(fm_set & ~ams_set)
        overlap_rows[(kind, "AMS only")] = hours_by_season(ams_set & ~fm_set)
        # reasons, first match wins
        fm_only_tables[kind] = attribute(fm_set & ~ams_set, reasons_fm120_only(a, kind))
        ams_only_tables[kind] = attribute(ams_set & ~fm_set, reasons_ams_only(f, kind))
    overlap = pd.DataFrame(overlap_rows).T[S.SEASON_ORDER]
    fm_only = pd.concat(fm_only_tables, names=["class", "reason"])
    ams_only = pd.concat(ams_only_tables, names=["class", "reason"])
    print("\nCloud and haze sampling hours on the 5-min grid:")
    print(overlap.round(1).to_string())
    print("\nWhy FM-120 cloud/haze slots are not AMS cloud/haze (hours; first reason that applies):")
    print(fm_only.round(1).to_string())
    print("\nWhy AMS cloud/haze slots are not FM-120 cloud/haze (hours; first reason that applies):")
    print(ams_only.round(1).to_string())
    unexplained = fm_only.xs("unexplained", level="reason").to_numpy().sum() + \
        ams_only.xs("unexplained", level="reason").to_numpy().sum()
    if unexplained > 0:
        print(f"WARNING: {unexplained:g} complement hours have no reason (a gap in the reason lists)")

    # --- 4. Full cross-tabulation of the two sides, campaign hours ---------------------------------------------------
    fm_label = f["fm_state"].map({"cloud": "cloud (row 31)", "haze": "haze (row 32)", "neither": "neither",
                                  "clear": "clear", "no visibility": "no visibility", "no data": "no data"})
    # rows = AMS states, columns = FM-120 states (the AMS side has more states, so this prints narrower)
    crosstab = pd.crosstab(ams_detail(a).rename("AMS state"), fm_label.rename("FM-120 state"),
                           margins=True, margins_name="total") * SLOT_H
    print("\nEvery AMS state (rows) against every FM-120 state (columns), campaign hours:")
    print(crosstab.round(1).to_string())

    # --- 5. When the instruments had no data ---------------------------------------------------------------
    down = {
        "FM-120": ~f["fm_recording"],
        "visibility": ~f["vis_ok"],
        "AMS": ~a["ams_recording"],
        "FM-120 and AMS both": ~f["fm_recording"] & ~a["ams_recording"],
    }
    season_h = hours_by_season(pd.Series(True, index=grid))  # length of each season in hours
    down_h = pd.DataFrame({name: hours_by_season(m) for name, m in down.items()}).T[S.SEASON_ORDER]
    down_pct = (100 * down_h / season_h).round(1)
    down_table = pd.concat({"hours": down_h.round(1), "% of season": down_pct}, axis=1)
    print("\nNo data, hours and % of each season:")
    print(down_table.to_string())
    periods = pd.concat({name: runs(m) for name, m in down.items()}, names=["instrument", "period"]).reset_index(level=0)
    periods["hours"] = periods["hours"].round(3)
    long = periods[periods["hours"] >= args.min_period_h]
    for name in down:
        p = periods[periods["instrument"] == name]
        lp = long[long["instrument"] == name]
        print(f"\n{name}: {len(p)} periods with no data, {p['hours'].sum():,.1f} h; "
              f"{len(lp)} of them >= {args.min_period_h:g} h ({lp['hours'].sum():,.1f} h):")
        if len(lp):
            print(lp[["start", "end", "hours"]].to_string(index=False))

    # --- 6. Write the tables -----------------------------------------------------------------------------------------
    overlap.to_csv(args.out / "overlap_hours_by_season.csv", float_format="%.2f")
    fm_only.to_csv(args.out / "fm120_only_reasons_by_season.csv", float_format="%.2f")
    ams_only.to_csv(args.out / "ams_only_reasons_by_season.csv", float_format="%.2f")
    crosstab.to_csv(args.out / "state_crosstab_hours.csv", float_format="%.2f")
    down_table.to_csv(args.out / "no_data_hours_by_season.csv")
    periods.to_csv(args.out / "no_data_periods.csv", index=False)
    if args.save_slots:
        slots = pd.concat([f[["fm_recording", "vis_ok", "fm_state"]], a], axis=1)
        slots.to_csv(args.out / "slots.csv")

    # --- 7. Timeline figure ------------------------------------------------------------------------------------------
    lanes = [
        ("FM-120 no data (not running)", down["FM-120"], INK_3),
        ("Visibility no data", down["visibility"], INK_3),
        ("AMS no data (off or removed in QC)", down["AMS"], INK_3),
        ("FM-120 and AMS both no data", down["FM-120 and AMS both"], INK),
        (None, None, None),
        ("FM-120 cloud (row 31)", sets["cloud"]["FM-120"], CLOUD_COLOR),
        ("AMS cloud (row 34)", sets["cloud"]["AMS"], CLOUD_COLOR),
        ("Cloud in both", sets["cloud"]["FM-120"] & sets["cloud"]["AMS"], CLOUD_COLOR),
        (None, None, None),
        ("FM-120 haze (row 32)", sets["haze"]["FM-120"], HAZE_COLOR),
        ("AMS haze (row 35)", sets["haze"]["AMS"], HAZE_COLOR),
        ("Haze in both", sets["haze"]["FM-120"] & sets["haze"]["AMS"], HAZE_COLOR),
    ]
    plot_timeline(lanes, args.out)
    print(f"\nWrote the tables and sampling_overlap_timeline.png/.pdf to {args.out}")


if __name__ == "__main__":
    main()
