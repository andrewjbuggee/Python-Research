#!/usr/bin/env python3
"""Downwelling longwave and LW cloud radiative forcing under pure-liquid vs
pure-ice cloud columns, by season.

The question: at NSA, how much surface downwelling longwave (LW-down) arrives
under a column that is entirely liquid, versus one that is entirely ice, how
much of that is cloud radiative forcing, and how does it split by season?

Method
------
1. Read the THERMOCLDPHASE VAP (Zhang et al. 2025, DOE/SC-ARM-TR-325), built
   from MPL/HSRL backscatter + depolarization, ARSCL radar moments, MWRRET
   LWP, and INTERPSONDE temperature. Phase codes come from config.THERMO_*
   (read off the files' flag_values -- they are NOT the Shupe-Turner codes).
2. Collapse each 30 s profile to one column category, by either of two rules
   (--classify):

   pixel  Uses cloud_phase_<lidar>(time, height), the per-volume phase on the
          30 m grid. A column is pure_liquid only if EVERY hydrometeor pixel
          in it is liquid, pure_ice only if every one is ice. All-or-nothing.

   layer  Uses cloud_phase_layer_<lidar>(time, layer), the VAP's own per-layer
          phase for up to 10 ARSCL cloud layers, on its own 4-value scale
          (0 clear, 1 liquid, 2 ice, 3 mixed). The VAP assigns each layer by
          the fraction of ice-containing pixels within it -- liquid < 0.1,
          mixed 0.1-0.9, ice > 0.9 (report Sect. 2) -- so this is a GRADED
          criterion that tolerates a minority of off-phase pixels, where
          "pixel" mode would reject the column outright. A column is
          pure_liquid if every cloud layer is liquid, pure_ice if every one
          is ice.

   Comparing the two is the cheapest available test of how sensitive the
   result is to the all-or-nothing rule.
3. Pair each column with the nearest QCRAD flux sample within --tolerance.
4. Compute LW cloud radiative forcing and split by season.

Cloud radiative forcing
-----------------------
Following Shupe and Intrieri (2004, J. Climate 17, 616-628, doi:10.1175/1520-
0442(2004)017<0616:CRFOTA>2.0.CO;2), Eq. (1a): CF_LW = F(A_c) - F(0), the
all-sky minus clear-sky net surface LW flux. As they note in their Sect. 3a,
the upwelling surface-emission terms very nearly cancel between the two, so
only the downwelling fluxes are considered:

    CF_LW ~= LW_down(cloudy) - LW_down(clear)

!! IMPORTANT METHODOLOGICAL DIFFERENCE. Shupe and Intrieri computed the
clear-sky term with a radiative transfer model (SBDART) run hourly on the
observed sounding, explicitly BECAUSE Arctic clear-sky periods are too sparse
and unrepresentative to composite ("Due to the ubiquitous presence of clouds
in the Arctic ... a radiative transfer model was employed to calculate the
clear sky fluxes instead of relying upon sporadic measurements of clear-sky
conditions"). This script has no radiative transfer model, so it uses an
OBSERVED clear-sky composite instead. That is a weaker baseline: clear and
cloudy columns are not sampled under the same temperature and humidity, so
the difference carries an atmospheric-state bias on top of the cloud effect.
--crf-baseline month matches the composite by calendar month, which removes
the seasonal cycle in that bias but not synoptic covariance. Treat the
absolute CF_LW values here as an observational proxy, not a reproduction of
their method; the liquid-minus-ice contrast is far more robust than either
absolute value, because both share the same baseline.

Their SHEBA annual means, drawn as reference lines on the CF_LW panel:
liquid-containing cloud scenes 52 W/m^2, ice-only cloud scenes 16 W/m^2. Note
those are annual means over a drifting ice camp in 1997-98, with a modeled
clear sky and a "liquid-containing" class that includes mixed-phase columns --
so they are a sanity-check magnitude, not a like-for-like target.

Seasons (NOT config.WINTER_MONTHS, the Hartig26 Nov-Mar season used elsewhere):
    Winter = October through March    (10, 11, 12, 1, 2, 3)
    Summer = April through September  (4, 5, 6, 7, 8, 9)

Interpretation notes
--------------------
* Expect liquid columns well above ice columns. Liquid reaches emissivity ~1
  in tens of metres and is typically low and warm; ice columns are frequently
  optically thin, higher, and colder. Shupe and Intrieri found 95% of the
  radiatively important scenes had bases below 4.3 km and base temperatures
  warmer than -31 C, so the cloud_base_km column of the stats table is the
  first thing to check when a category behaves oddly.
* Error bars are +/- 1 standard deviation, i.e. the SPREAD of the
  distribution, not a standard error. Consecutive 30 s samples are strongly
  autocorrelated (one cloud event supplies hundreds), so the effective sample
  size is far below n and significance tests on these counts would be
  overconfident. Treat n as "how much data", not "how many independent
  observations".
* This is a PURITY filter, not a frequency-of-occurrence measure: most Arctic
  columns are mixed or multi-layer and land in "other" by design.
* THERMOCLDPHASE carries no LWC/IWC. A true "mostly liquid / mostly ice"
  threshold on ice water fraction needs ACRED or MICROBASE; --classify layer
  is the closest graded proxy available from this product alone.

Prerequisites
-------------
    python scripts/download_nsa_data.py --datastreams thermocldphase qcrad \
        --start 2014-02-01 --end 2014-11-30

Examples
--------
# Both seasons, all-or-nothing pixel rule (the default):
python scripts/analyze_phase_lw_seasonal.py --start 2014-02-09 --end 2014-11-13

# Winter only, using the VAP's graded per-layer phase:
python scripts/analyze_phase_lw_seasonal.py --start 2014-02-09 --end 2014-11-13 \
    --seasons winter --classify layer

# Count drizzle/rain as liquid and snow as ice; HSRL instead of MPL:
python scripts/analyze_phase_lw_seasonal.py --start 2014-02-09 --end 2014-11-13 \
    --strictness inclusive --lidar hsrl

Outputs
-------
figures/phase_lw_seasonal_<...>.png        LW-down bars, CF_LW bars, distributions
data/processed/phase_lw_seasonal_<...>.nc   paired column category + fluxes + CF_LW
data/processed/phase_lw_seasonal_<...>.csv  per-season, per-category statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _plot_common import (  # noqa: E402
    add_data_root_argument,
    apply_data_root,
    finish_figure,
    use_headless_backend_if_needed,
)

# Column categories produced by the classifiers.
COLUMN_CODES = {
    0: "missing",
    1: "clear",
    2: "pure_ice",
    3: "pure_liquid",
    4: "other",
    5: "mixed_more_liquid",
    6: "mixed_more_ice",
}

# Ice fraction separating the two halves of the VAP's mixed range. The 0.1 and
# 0.9 ends are the VAP's own liquid/mixed/ice boundaries (config.THERMO_FRC_*);
# only this midpoint is a choice made here.
MIXED_SPLIT_FRC_ICE = 0.5

# Plot order runs ice -> liquid so the phase gradient reads left to right.
BASE_PLOT_ORDER = ["clear", "pure_ice", "pure_liquid"]
MIXED_PLOT_ORDER = [
    "clear",
    "pure_ice",
    "mixed_more_ice",
    "mixed_more_liquid",
    "pure_liquid",
]
DISPLAY_NAMES = {
    "clear": "Clear Sky",
    "pure_ice": "Pure Ice",
    "mixed_more_ice": "Mixed (More Ice)",
    "mixed_more_liquid": "Mixed (More Liquid)",
    "pure_liquid": "Pure Liquid",
    "other": "Other",
}
CATEGORY_COLORS = {
    "clear": "lightgray",
    "pure_ice": "steelblue",
    "mixed_more_ice": "lightsteelblue",
    "mixed_more_liquid": "sandybrown",
    "pure_liquid": "orangered",
    "other": "plum",
}

SEASONS = {
    "Winter (Oct-Mar)": (10, 11, 12, 1, 2, 3),
    "Summer (Apr-Sep)": (4, 5, 6, 7, 8, 9),
}
SEASON_KEYS = {"winter": "Winter (Oct-Mar)", "summer": "Summer (Apr-Sep)"}

# Shupe and Intrieri (2004) Table/abstract values: SHEBA annual-mean surface
# CF_LW by cloud scene, drifting ice camp Oct 1997 - Oct 1998, clear sky from
# the SBDART radiative transfer model. Their "liquid-containing" class includes
# mixed-phase columns, which this script's "pure_liquid" deliberately excludes.
SHUPE_INTRIERI_2004_CF_LW = {"pure_liquid": 52.0, "pure_ice": 16.0}


def phase_code_sets(strictness: str):
    """Return (liquid_codes, ice_codes) for the requested strictness.

    strict     -- liquid = {liquid}, ice = {ice}. Any precipitation pixel in
                  the column disqualifies it from both, so the populations are
                  non-precipitating cloud only.
    inclusive  -- liquid also admits drizzle / liquid+drizzle / rain, ice also
                  admits snow. Broader samples, but the liquid population then
                  mixes cloud droplets with precipitation-sized drops, which
                  have different radiative behaviour.

    Applies to --classify pixel only; the per-layer phase scale has no
    precipitation classes to include or exclude.
    """
    from arm_nsa import config

    if strictness == "strict":
        return tuple(config.THERMO_LIQUID_CODES), tuple(config.THERMO_ICE_CODES)
    if strictness == "inclusive":
        return (
            tuple(config.THERMO_LIQUID_CODES)
            + tuple(config.THERMO_LIQUID_PRECIP_CODES),
            tuple(config.THERMO_ICE_CODES) + tuple(config.THERMO_FROZEN_PRECIP_CODES),
        )
    raise ValueError(f"unknown strictness {strictness!r}")


def column_ice_fraction(ds, pixel_phase_var: str, apply_qc_flags: bool):
    """Fraction of a column's hydrometeor pixels that are ice-containing.

    Uses the VAP's own frc_ice definition -- ice, mixed-phase, or snow count as
    ice-containing (report Sect. 2) -- but applied over the whole column rather
    than per cloud layer, because the files publish only each layer's resulting
    LABEL, never the fraction that produced it. Splitting the mixed range any
    finer therefore requires recomputing the fraction from the pixel mask, and
    the column is the level at which the rest of this script already works.

    Returns NaN for columns with no hydrometeor pixels, so every downstream
    comparison against a threshold is False there.
    """
    import numpy as np

    from arm_nsa import config
    from arm_nsa.qc import apply_qc

    phase = apply_qc(ds, pixel_phase_var) if apply_qc_flags else ds[pixel_phase_var]
    values = phase.values
    valid = np.isfinite(values)
    is_cloud = valid & (values >= 1) & (values <= 7)
    is_ice_containing = valid & np.isin(
        values, config.THERMO_ICE_CONTAINING_CODES
    )
    n_cloud = is_cloud.sum(axis=1)
    n_ice = is_ice_containing.sum(axis=1)
    return np.where(n_cloud > 0, n_ice / np.maximum(n_cloud, 1), np.nan)


def _apply_mixed_split(code, frc_ice):
    """Carve the two mixed categories out of whatever is still 'other'.

    Only code 4 is reassigned, so pure_liquid and pure_ice keep exactly the
    definitions they had before this option existed and the two runs stay
    comparable. Columns whose ice fraction falls outside the VAP's mixed range
    (<= 0.1 or >= 0.9) but which failed a purity test for some other reason --
    precipitation pixels under --strictness strict, an unknown pixel, too few
    cloud pixels -- stay in 'other' rather than being forced into a bin.
    """
    import numpy as np

    from arm_nsa import config

    unresolved = code == 4
    with np.errstate(invalid="ignore"):
        more_liquid = (
            unresolved
            & (frc_ice > config.THERMO_FRC_ICE_LIQUID_MAX)
            & (frc_ice < MIXED_SPLIT_FRC_ICE)
        )
        more_ice = (
            unresolved
            & (frc_ice >= MIXED_SPLIT_FRC_ICE)
            & (frc_ice < config.THERMO_FRC_ICE_ICE_MIN)
        )
    code[more_liquid] = 5
    code[more_ice] = 6
    return code


def _empty_result(n_time: int):
    import numpy as np

    return {
        "code": np.zeros(n_time, dtype=np.int8),
        "n_cloud_pixels": np.zeros(n_time, dtype=np.int32),
        "cloud_base_km": np.full(n_time, np.nan),
        "cloud_top_km": np.full(n_time, np.nan),
    }


def classify_columns_pixel(
    ds,
    phase_var: str,
    liquid_codes,
    ice_codes,
    min_cloud_pixels: int,
    apply_qc_flags: bool,
    include_mixed: bool = False,
):
    """All-or-nothing column rule on the per-pixel phase mask.

    Reduces (time, height) to (time,) inside this function so a multi-month
    analysis never holds every profile in memory: at 2880 x 596 float32 per
    file that would be ~7 MB/day, or several GB across a full record.
    """
    import numpy as np

    from arm_nsa import config
    from arm_nsa.qc import apply_qc

    phase = apply_qc(ds, phase_var) if apply_qc_flags else ds[phase_var]
    values = phase.values  # (time, height)
    heights_km = ds["height"].values  # km, see config.THERMO_HEIGHT_UNITS

    valid = np.isfinite(values)
    # Hydrometeor pixels are every non-clear, non-unknown class: codes 1-7.
    is_cloud = valid & (values >= 1) & (values <= 7)
    is_unknown = valid & (values == config.THERMO_UNKNOWN_CODE)
    is_liquid = valid & np.isin(values, liquid_codes)
    is_ice = valid & np.isin(values, ice_codes)

    n_valid = valid.sum(axis=1)
    n_cloud = is_cloud.sum(axis=1)
    n_unknown = is_unknown.sum(axis=1)
    n_liquid = is_liquid.sum(axis=1)
    n_ice = is_ice.sum(axis=1)

    has_valid = n_valid > 0
    # An unknown pixel could be concealing the other phase, so it disqualifies
    # a column from any "pure" label rather than being quietly ignored.
    no_unknown = n_unknown == 0
    thick_enough = n_cloud >= min_cloud_pixels

    out = _empty_result(values.shape[0])
    code = out["code"]
    code[has_valid] = 4  # other, until proven otherwise
    code[has_valid & no_unknown & (n_cloud == 0)] = 1  # clear
    code[thick_enough & no_unknown & (n_ice == n_cloud)] = 2
    code[thick_enough & no_unknown & (n_liquid == n_cloud)] = 3
    out["n_cloud_pixels"] = n_cloud.astype(np.int32)
    if include_mixed:
        _apply_mixed_split(code, column_ice_fraction(ds, phase_var, apply_qc_flags))

    # Cloud-base / -top height, useful for explaining the LW-down difference
    # (a low warm base radiates far more than a high cold one).
    any_cloud = n_cloud > 0
    if any_cloud.any():
        idx = np.where(any_cloud)[0]
        cloudy = is_cloud[idx]
        first = cloudy.argmax(axis=1)
        last = cloudy.shape[1] - 1 - cloudy[:, ::-1].argmax(axis=1)
        out["cloud_base_km"][idx] = heights_km[first]
        out["cloud_top_km"][idx] = heights_km[last]
    return out


def classify_columns_layer(
    ds,
    layer_phase_var: str,
    apply_qc_flags: bool,
    include_mixed: bool = False,
    pixel_phase_var: str = "",
):
    """Graded column rule on the VAP's own per-layer phase.

    cloud_phase_layer_<lidar>(time, layer) is already a per-layer liquid /
    mixed / ice call, assigned by the fraction of ice-containing pixels in the
    layer (thresholds 0.1 and 0.9, report Sect. 2). A column counts as pure
    when every one of its cloud layers carries the same phase, so a layer with
    a minority of off-phase pixels no longer disqualifies the whole column the
    way it does under the pixel rule.

    These files ship no qc_cloud_phase_layer_* companion; apply_qc is a no-op
    when a variable has no qc_ partner, so it is still safe to call.
    """
    import numpy as np

    from arm_nsa.qc import apply_qc

    phase = apply_qc(ds, layer_phase_var) if apply_qc_flags else ds[layer_phase_var]
    values = phase.values  # (time, layer)
    # (time, layer, bound) in km: bound 0 = base, bound 1 = top.
    heights_km = ds["cloud_layer_heights"].values

    valid = np.isfinite(values)
    is_layer = valid & (values >= 1) & (values <= 3)  # liquid, ice, or mixed
    is_liquid = valid & (values == 1)
    is_ice = valid & (values == 2)

    n_valid = valid.sum(axis=1)
    n_layer = is_layer.sum(axis=1)
    n_liquid = is_liquid.sum(axis=1)
    n_ice = is_ice.sum(axis=1)

    has_valid = n_valid > 0
    has_layer = n_layer >= 1

    out = _empty_result(values.shape[0])
    code = out["code"]
    code[has_valid] = 4
    code[has_valid & (n_layer == 0)] = 1  # clear
    code[has_layer & (n_ice == n_layer)] = 2
    code[has_layer & (n_liquid == n_layer)] = 3
    # "n_cloud_pixels" carries the layer count in this mode.
    out["n_cloud_pixels"] = n_layer.astype(np.int32)
    if include_mixed:
        # The per-layer field cannot supply this: it reports each layer's
        # label, not the frc_ice behind it, so the split is computed from the
        # pixel mask even in layer mode.
        _apply_mixed_split(
            code, column_ice_fraction(ds, pixel_phase_var, apply_qc_flags)
        )

    if has_layer.any():
        base = np.where(is_layer, heights_km[:, :, 0], np.nan)
        top = np.where(is_layer, heights_km[:, :, 1], np.nan)
        idx = np.where(has_layer)[0]
        # Rows are guaranteed to hold at least one finite value, so nanmin /
        # nanmax cannot hit the all-NaN warning path.
        out["cloud_base_km"][idx] = np.nanmin(base[idx], axis=1)
        out["cloud_top_km"][idx] = np.nanmax(top[idx], axis=1)
    return out


def read_column_categories(
    key: str,
    start_date: str,
    end_date: str,
    classify_mode: str,
    lidar: str,
    liquid_codes,
    ice_codes,
    min_cloud_pixels: int,
    apply_qc_flags: bool,
    verbose: bool,
    include_mixed: bool = False,
):
    """Build the per-time column-category series over a date range."""
    import numpy as np
    import xarray as xr

    from arm_nsa import config
    from arm_nsa.readers import files_in_range, resolve_variable

    pixel_var = f"cloud_phase_{lidar}"
    phase_var = pixel_var if classify_mode == "pixel" else f"cloud_phase_layer_{lidar}"
    # The mixed split always needs the pixel mask, whichever mode is running.
    required = (phase_var,) if not include_mixed else (phase_var, pixel_var)

    paths = files_in_range(key, start_date, end_date)
    if not paths:
        raise SystemExit(
            f"error: no local THERMOCLDPHASE files for '{key}' in "
            f"{start_date}..{end_date} under {config.RAW_DATA_DIR}.\n"
            f"  Download them first:\n"
            f"    python scripts/download_nsa_data.py --datastreams thermocldphase "
            f"--start {start_date} --end {end_date}"
        )

    # Mixing c0 and c1 over their 2014-2020 overlap would silently blend two
    # data levels, so say plainly which levels are contributing.
    levels: dict = {}
    for p in paths:
        levels[p.parent.name] = levels.get(p.parent.name, 0) + 1
    print(f"THERMOCLDPHASE files: {len(paths)}")
    for name, count in sorted(levels.items()):
        print(f"  {name}: {count} file(s)")
    if len(levels) > 1:
        print(
            "  WARNING: more than one data level is contributing. c0 and c1 "
            "overlap 2014-2020;\n  pass --datastream <name> to pin one for a "
            "reproducible result.",
            file=sys.stderr,
        )

    pieces, times, skipped = [], [], 0
    for i, path in enumerate(paths, start=1):
        if verbose or i % 25 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {path.name}")
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            try:
                for name in required:
                    resolve_variable(ds, (name,))
            except KeyError:
                # HSRL fields are absent wherever no HSRL was deployed; skip
                # rather than abort a multi-month run.
                skipped += 1
                continue
            if classify_mode == "pixel":
                result = classify_columns_pixel(
                    ds, phase_var, liquid_codes, ice_codes, min_cloud_pixels,
                    apply_qc_flags, include_mixed,
                )
            else:
                result = classify_columns_layer(
                    ds, phase_var, apply_qc_flags, include_mixed, pixel_var
                )
            pieces.append(result)
            times.append(ds["time"].values)
    if skipped:
        print(
            f"  note: {skipped} file(s) had no '{phase_var}' variable and were "
            f"skipped (expected when requesting --lidar hsrl at a site or "
            f"period without an HSRL).",
            file=sys.stderr,
        )
    if not pieces:
        raise SystemExit(
            f"error: none of the {len(paths)} file(s) contained '{phase_var}'."
        )

    out = xr.Dataset(
        {
            name: ("time", np.concatenate([p[name] for p in pieces]))
            for name in ("code", "n_cloud_pixels", "cloud_base_km", "cloud_top_km")
        },
        coords={"time": np.concatenate(times)},
    ).rename({"code": "column_code"})
    out = out.sortby("time")
    _, unique_idx = np.unique(out["time"].values, return_index=True)
    if unique_idx.size != out.sizes["time"]:
        out = out.isel(time=unique_idx)
    out["column_code"].attrs.update(
        long_name="cloud-phase column category",
        flag_values=list(COLUMN_CODES.keys()),
        flag_meanings=" ".join(COLUMN_CODES.values()),
        classify_mode=classify_mode,
        phase_variable=phase_var,
        liquid_codes=str(liquid_codes),
        ice_codes=str(ice_codes),
        min_cloud_pixels=min_cloud_pixels,
    )
    return out.sel(time=slice(start_date, f"{end_date}T23:59:59")), phase_var


def season_of(months_array, season_months):
    """Boolean mask selecting samples whose month is in `season_months`."""
    import numpy as np

    return np.isin(months_array, list(season_months))


def annual_mean_crf(codes, crf_lw, months, categories):
    """Annual-mean CF_LW per category, for comparison with Shupe & Intrieri.

    Their quoted 52 / 16 W/m^2 are ANNUAL means over a complete SHEBA year, so
    a season-split table cannot be compared with them directly. Two averages
    are reported here because they diverge whenever the record is not a
    complete, evenly sampled year:

      month_weighted   mean of the per-calendar-month means, so each month
                       counts once no matter how many samples it supplied.
                       The more honest annual mean under uneven sampling.
      sample_weighted  plain mean over every sample. This is what a complete,
                       evenly sampled year would give, but it over-weights
                       whichever months happen to hold the most data -- which
                       for a partial record is a real bias, since the missing
                       months are usually the extreme ones.

    A gap between the two columns is itself the diagnostic: it measures how
    unevenly the year is sampled.
    """
    import numpy as np
    import pandas as pd

    name_of = {v: k for k, v in COLUMN_CODES.items()}
    months_present = sorted(int(m) for m in np.unique(months))
    rows = []
    for category in categories:
        is_cat = (codes == name_of[category]) & np.isfinite(crf_lw)
        monthly = []
        for month in months_present:
            sel = is_cat & (months == month)
            if sel.sum() > 0:
                monthly.append(crf_lw[sel].mean())
        rows.append(
            {
                "category": DISPLAY_NAMES[category],
                "month_weighted": np.mean(monthly) if monthly else np.nan,
                "sample_weighted": crf_lw[is_cat].mean() if is_cat.any() else np.nan,
                # Only the pure categories have a Shupe & Intrieri counterpart;
                # the mixed split is finer than any class they published.
                "shupe_intrieri_2004": SHUPE_INTRIERI_2004_CF_LW.get(
                    category, np.nan
                ),
                "n_months": len(monthly),
                "n_samples": int(is_cat.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("category"), months_present


def clear_sky_reference(codes, lwdn, months, baseline: str, min_clear: int):
    """Per-sample clear-sky LW-down baseline for the CF_LW calculation.

    Returns (reference_array, notes). The reference is NaN wherever the
    matching group had fewer than `min_clear` valid clear-sky samples, which
    propagates to NaN CF_LW rather than a silently unreliable number.
    """
    import numpy as np

    clear_code = {v: k for k, v in COLUMN_CODES.items()}["clear"]
    reference = np.full(lwdn.shape, np.nan)
    notes = []

    if baseline == "month":
        groups = {m: months == m for m in np.unique(months)}
        label = "month"
    else:
        groups = {
            name: season_of(months, mons) for name, mons in SEASONS.items()
        }
        label = "season"

    for name, in_group in groups.items():
        is_clear = in_group & (codes == clear_code) & np.isfinite(lwdn)
        n = int(is_clear.sum())
        if n < min_clear:
            notes.append(f"{label} {name}: only {n} clear samples -> CF_LW undefined")
            continue
        reference[in_group] = lwdn[is_clear].mean()
    return reference, notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--datastream",
        default="thermocldphase",
        help=(
            "pipeline key or a raw ARM datastream name to pin one data level "
            "(e.g. nsathermocldphaseC1.c0). Default: the 'thermocldphase' key, "
            "which reads every level present locally."
        ),
    )
    parser.add_argument(
        "--classify",
        choices=["pixel", "layer"],
        default="pixel",
        help=(
            "pixel: all-or-nothing rule on the 30 m phase mask -- every "
            "hydrometeor pixel must share the phase. layer: graded rule on the "
            "VAP's own per-layer phase, which already tolerates a minority of "
            "off-phase pixels (liquid < 0.1 ice fraction, ice > 0.9). "
            "Default: pixel."
        ),
    )
    parser.add_argument(
        "--lidar",
        choices=["mplgr", "hsrl"],
        default="mplgr",
        help=(
            "which phase retrieval to use: mplgr (MPL intensity gradient, "
            "available for the whole record) or hsrl (calibrated HSRL "
            "extinction, only where an HSRL was deployed). Default: mplgr."
        ),
    )
    parser.add_argument(
        "--strictness",
        choices=["strict", "inclusive"],
        default="strict",
        help=(
            "strict: pure liquid = liquid pixels only, pure ice = ice pixels "
            "only (any precipitation disqualifies the column). inclusive: also "
            "admit drizzle/liquid+drizzle/rain as liquid and snow as ice. "
            "Applies to --classify pixel only. Default: strict."
        ),
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help=(
            "split the mixed-phase columns now lumped into 'other' into two "
            "categories by column ice fraction: 'Mixed (More Liquid)' for "
            f"{0.1} < frc_ice < {MIXED_SPLIT_FRC_ICE} and 'Mixed (More Ice)' "
            f"for {MIXED_SPLIT_FRC_ICE} <= frc_ice < {0.9}. The outer bounds "
            "are the VAP's own liquid/mixed/ice boundaries; only the midpoint "
            "is a choice. Pure Ice and Pure Liquid keep their existing "
            "definitions, so results stay comparable with runs without it."
        ),
    )
    parser.add_argument(
        "--seasons",
        choices=["both", "winter", "summer"],
        default="both",
        help=(
            "which seasons to plot. Statistics for both are always written to "
            "the CSV; this selects the figure only. Default: both."
        ),
    )
    parser.add_argument(
        "--min-cloud-pixels",
        type=int,
        default=3,
        help=(
            "minimum hydrometeor pixels in a column for a 'pure' label under "
            "--classify pixel; at 30 m resolution the default 3 requires a "
            "~90 m deep cloud and rejects single-pixel speckle. Ignored in "
            "layer mode. Default: 3."
        ),
    )
    parser.add_argument(
        "--crf-baseline",
        choices=["month", "season"],
        default="month",
        help=(
            "how to composite the observed clear-sky LW-down used as the "
            "CF_LW baseline. month matches by calendar month, removing the "
            "seasonal cycle from the clear-vs-cloudy state bias; season uses "
            "one value per season. Default: month."
        ),
    )
    parser.add_argument(
        "--min-clear-samples",
        type=int,
        default=100,
        help=(
            "minimum clear-sky samples in a baseline group before CF_LW is "
            "computed for it; below this the group's CF_LW is left undefined "
            "rather than reported unreliably. Default: 100."
        ),
    )
    parser.add_argument(
        "--tolerance",
        default="3min",
        help="max column-to-flux time offset when pairing (default 3min)",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="skip ARM embedded QC masking of the phase field",
    )
    parser.add_argument("--no-show", action="store_true", help="save only")
    parser.add_argument("--verbose", action="store_true", help="list every file read")
    add_data_root_argument(parser)
    args = parser.parse_args()
    use_headless_backend_if_needed(show=not args.no_show)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from arm_nsa import config
    from arm_nsa.shupe_turner import pair_scene_with_flux
    from arm_nsa.surface import read_qcrad

    apply_data_root(args.data_root)

    liquid_codes, ice_codes = phase_code_sets(args.strictness)
    meanings = config.THERMO_PHASE_MEANINGS
    plot_categories = MIXED_PLOT_ORDER if args.include_mixed else BASE_PLOT_ORDER
    if args.include_mixed:
        print(
            f"mixed split ON: ice fraction "
            f"{config.THERMO_FRC_ICE_LIQUID_MAX} < frc_ice < "
            f"{MIXED_SPLIT_FRC_ICE} -> Mixed (More Liquid); "
            f"{MIXED_SPLIT_FRC_ICE} <= frc_ice < "
            f"{config.THERMO_FRC_ICE_ICE_MIN} -> Mixed (More Ice)\n"
            f"  frc_ice counts "
            f"{[meanings[c] for c in config.THERMO_ICE_CONTAINING_CODES]} "
            f"as ice-containing, over all hydrometeor pixels in the column"
        )
    if args.classify == "pixel":
        print(
            f"classify: pixel (all-or-nothing)  |  lidar: {args.lidar}  |  "
            f"strictness: {args.strictness}\n"
            f"  liquid = {[meanings[c] for c in liquid_codes]}\n"
            f"  ice    = {[meanings[c] for c in ice_codes]}"
        )
    else:
        print(
            f"classify: layer (graded, VAP per-layer phase)  |  "
            f"lidar: {args.lidar}\n"
            f"  layer scale: {config.THERMO_LAYER_PHASE_MEANINGS}\n"
            f"  (--strictness and --min-cloud-pixels do not apply)"
        )

    # ------------------------------------------------------------------ load
    print("\nreading THERMOCLDPHASE and classifying columns ...")
    columns, phase_var = read_column_categories(
        args.datastream,
        args.start,
        args.end,
        args.classify,
        args.lidar,
        liquid_codes,
        ice_codes,
        args.min_cloud_pixels,
        apply_qc_flags=not args.no_qc,
        verbose=args.verbose,
        include_mixed=args.include_mixed,
    )

    # The requested window routinely runs past what has been downloaded. Clip
    # to the phase record actually present so the QCRAD read, the table header
    # and the figure title all describe the same period -- and so a partial
    # archive is never silently reported under the requested dates.
    data_start = pd.Timestamp(columns["time"].values.min())
    data_end = pd.Timestamp(columns["time"].values.max())
    if (
        data_start.date() > pd.Timestamp(args.start).date()
        or data_end.date() < pd.Timestamp(args.end).date()
    ):
        print(
            f"  NOTE: requested {args.start}..{args.end}, but the downloaded "
            f"THERMOCLDPHASE record\n"
            f"  covers only {data_start.date()}..{data_end.date()}. Everything "
            f"below describes the data\n"
            f"  actually present, not the requested window.",
            file=sys.stderr,
        )
    qcrad_start, qcrad_end = str(data_start.date()), str(data_end.date())
    print(f"\nreading QCRAD broadband fluxes ({qcrad_start}..{qcrad_end}) ...")
    qcrad = read_qcrad(qcrad_start, qcrad_end)
    paired = pair_scene_with_flux(
        columns["column_code"], qcrad, tolerance=args.tolerance
    )
    paired = paired.rename({"scene_code": "column_code"})
    for name in ("n_cloud_pixels", "cloud_base_km", "cloud_top_km"):
        paired[name] = columns[name]

    # ------------------------------------------------------- cloud forcing
    codes = paired["column_code"].values
    lwdn = paired["lwdn_w_m2"].values.astype(float)
    months = paired["time"].dt.month.values

    reference, notes = clear_sky_reference(
        codes, lwdn, months, args.crf_baseline, args.min_clear_samples
    )
    for note in notes:
        print(f"  CF_LW note: {note}", file=sys.stderr)
    crf_lw = lwdn - reference
    paired["lwdn_clear_ref_w_m2"] = ("time", reference)
    paired["crf_lw_w_m2"] = ("time", crf_lw)
    paired["crf_lw_w_m2"].attrs.update(
        units="W/m^2",
        long_name="surface longwave cloud radiative forcing (downwelling only)",
        definition=(
            "LW_down minus an observed clear-sky composite matched by "
            f"{args.crf_baseline}; Shupe & Intrieri (2004) Eq. 1a with the "
            "upwelling terms cancelled per their Sect. 3a. NOT their SBDART "
            "modelled clear sky -- see script docstring."
        ),
    )

    # -------------------------------------------------------------- statistics
    name_of = {v: k for k, v in COLUMN_CODES.items()}
    rows = []
    for season, season_months in SEASONS.items():
        in_season = season_of(months, season_months)
        n_classified = int((in_season & (codes != 0)).sum())
        for category in plot_categories + ["other"]:
            sel = in_season & (codes == name_of[category])
            vals = lwdn[sel & np.isfinite(lwdn)]
            cvals = crf_lw[sel & np.isfinite(crf_lw)]
            base = paired["cloud_base_km"].values[sel]
            rows.append(
                {
                    "season": season,
                    "category": category,
                    "n": int(vals.size),
                    "occurrence_pct": (
                        100.0 * sel.sum() / n_classified if n_classified else np.nan
                    ),
                    "lwdn_mean": vals.mean() if vals.size else np.nan,
                    "lwdn_std": vals.std() if vals.size else np.nan,
                    "lwdn_median": np.median(vals) if vals.size else np.nan,
                    "crf_lw_mean": cvals.mean() if cvals.size else np.nan,
                    "crf_lw_std": cvals.std() if cvals.size else np.nan,
                    "cloud_base_km_median": (
                        np.nanmedian(base) if np.isfinite(base).any() else np.nan
                    ),
                }
            )
    stats = pd.DataFrame(rows).set_index(["season", "category"])

    # Period actually represented in the figure: classified samples falling in
    # a plotted season. With --seasons winter this is narrower than the loaded
    # record, so it is what the title should quote.
    season_names = (
        list(SEASONS) if args.seasons == "both" else [SEASON_KEYS[args.seasons]]
    )
    plotted_months = [m for s in season_names for m in SEASONS[s]]
    in_plot = season_of(months, plotted_months) & (codes != 0)
    times = paired["time"].values
    plot_start = pd.Timestamp(times[in_plot].min())
    plot_end = pd.Timestamp(times[in_plot].max())

    print(f"\n{'=' * 92}")
    print(
        f"LW-down and LW cloud forcing by column phase\n"
        f"data used:  {data_start.date()} .. {data_end.date()}"
        f"   (requested {args.start} .. {args.end})\n"
        f"in figure:  {plot_start.date()} .. {plot_end.date()}  [{args.seasons}]\n"
        f"classify={args.classify}, lidar={args.lidar}, "
        f"CF_LW baseline=observed clear sky matched by {args.crf_baseline}"
    )
    print("=" * 92)
    print(stats.round(2).to_string())

    # ---------------------------------------------- annual means vs S&I 2004
    annual, months_present = annual_mean_crf(
        codes, crf_lw, months, [c for c in plot_categories if c != "clear"]
    )
    missing_months = [m for m in range(1, 13) if m not in months_present]
    print(f"\n{'-' * 92}")
    print("ANNUAL-mean CF_LW vs Shupe & Intrieri (2004) [W/m^2]")
    print(f"{'-' * 92}")
    print(annual.round(2).to_string())
    if missing_months:
        print(
            f"\n  !! Not a complete year: calendar month(s) "
            f"{missing_months} have no data at all.\n"
            f"     These 'annual' means therefore omit part of the seasonal "
            f"cycle and are NOT\n"
            f"     directly comparable with the SHEBA full-year values. Their "
            f"liquid-containing\n"
            f"     class also includes mixed-phase columns, which 'Pure "
            f"Liquid' here excludes."
        )
    liq_a = annual.loc[DISPLAY_NAMES["pure_liquid"], "month_weighted"]
    ice_a = annual.loc[DISPLAY_NAMES["pure_ice"], "month_weighted"]
    if np.isfinite(liq_a) and np.isfinite(ice_a):
        si_gap = (
            SHUPE_INTRIERI_2004_CF_LW["pure_liquid"]
            - SHUPE_INTRIERI_2004_CF_LW["pure_ice"]
        )
        print(
            f"\n  annual liquid - ice = {liq_a - ice_a:.1f} "
            f"(month-weighted)   S&I 2004: {si_gap:.0f}"
        )

    print(f"\n{'-' * 92}")
    print("Seasonal CF_LW (S&I values repeated for scale; theirs are annual)")
    print(f"{'-' * 92}")
    for season in SEASONS:
        parts = []
        for category in ("pure_liquid", "pure_ice"):
            value = stats.loc[(season, category), "crf_lw_mean"]
            ref = SHUPE_INTRIERI_2004_CF_LW[category]
            parts.append(
                f"{DISPLAY_NAMES[category]}: {value:6.1f} (S&I {ref:.0f})"
                if np.isfinite(value)
                else f"{DISPLAY_NAMES[category]}:    n/a"
            )
        print(f"  {season:<20s} " + "   ".join(parts) + "  [W/m^2]")
        liq = stats.loc[(season, "pure_liquid"), "crf_lw_mean"]
        ice = stats.loc[(season, "pure_ice"), "crf_lw_mean"]
        if np.isfinite(liq) and np.isfinite(ice):
            si_gap = (
                SHUPE_INTRIERI_2004_CF_LW["pure_liquid"]
                - SHUPE_INTRIERI_2004_CF_LW["pure_ice"]
            )
            print(
                f"  {'':<20s} liquid - ice = {liq - ice:5.1f} "
                f"(S&I {si_gap:.0f})"
            )

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    x = np.arange(len(season_names))

    def grouped_bars(
        ax, categories, value_col, err_col, ylabel, title, show_n=True
    ):
        width = 0.78 / len(categories)
        for j, category in enumerate(categories):
            means = [stats.loc[(s, category), value_col] for s in season_names]
            errs = [stats.loc[(s, category), err_col] for s in season_names]
            counts = [int(stats.loc[(s, category), "n"]) for s in season_names]
            offset = (j - (len(categories) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=errs,
                capsize=3,
                color=CATEGORY_COLORS[category],
                edgecolor="k",
                lw=0.5,
                label=DISPLAY_NAMES[category],
                error_kw={"lw": 0.8, "ecolor": "0.35"},
            )
            for rect, mean, err, n in zip(bars, means, errs, counts):
                if not np.isfinite(mean):
                    continue
                err = err if np.isfinite(err) else 0.0
                if mean >= 0:
                    y, va = mean + err + 2, "bottom"
                else:
                    y, va = mean - err - 2, "top"
                # Mathtext bolds the value alone, so the count stays subordinate
                # in a single text call. n is dropped on the CF_LW panel, where
                # it would just repeat the LW-down panel.
                label = f"$\\mathbf{{{mean:.1f}}}$"
                if show_n:
                    label += f"\nn={n}"
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(season_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        # Autoscale ignores the annotation text, so the tallest bar's label
        # would otherwise be clipped by the axes frame.
        ax.margins(y=0.16)

    # (a) absolute LW-down. Clear Sky belongs here: it is the reference level
    # the two cloud categories are read against.
    grouped_bars(
        axes[0],
        plot_categories,
        "lwdn_mean",
        "lwdn_std",
        "downwelling LW [W m$^{-2}$]",
        f"LW-down by column phase ({args.classify} rule)\n"
        f"bars = mean, whiskers = $\\pm$1 s.d. of the distribution",
    )
    axes[0].legend(fontsize=8, loc="upper left")

    # (b) cloud radiative forcing, with the SHEBA reference lines. Clear Sky is
    # omitted: CF_LW is defined as the departure FROM clear sky, so a clear-sky
    # bar is identically zero by construction and carries no information.
    grouped_bars(
        axes[1],
        [c for c in plot_categories if c != "clear"],
        "crf_lw_mean",
        "crf_lw_std",
        "CF$_{LW}$ [W m$^{-2}$]",
        f"LW cloud radiative forcing\n"
        f"(LW-down minus observed clear sky, matched by {args.crf_baseline})",
        show_n=False,
    )
    axes[1].axhline(0, color="k", lw=0.8)
    for category, ref in SHUPE_INTRIERI_2004_CF_LW.items():
        axes[1].axhline(
            ref,
            color=CATEGORY_COLORS[category],
            ls=":",
            lw=1.6,
            label=f"S&I 2004 {DISPLAY_NAMES[category].lower()} ({ref:.0f})",
        )
    axes[1].legend(fontsize=7, loc="best")

    # (c) distributions -- a bar chart alone hides whether the populations
    # merely shift or barely overlap.
    ax = axes[2]
    bins = np.linspace(120, 340, 56)
    linestyles = ["-", "--"]
    for season, ls in zip(season_names, linestyles):
        in_season = season_of(months, SEASONS[season])
        for category in ("pure_ice", "pure_liquid"):
            sel = in_season & (codes == name_of[category]) & np.isfinite(lwdn)
            if sel.sum() > 10:
                ax.hist(
                    lwdn[sel],
                    bins=bins,
                    histtype="step",
                    lw=1.6,
                    ls=ls,
                    density=True,
                    color=CATEGORY_COLORS[category],
                    label=f"{DISPLAY_NAMES[category]}, {season.split(' ')[0]}",
                )
    ax.set_xlabel("downwelling LW [W m$^{-2}$]")
    ax.set_ylabel("probability density")
    ax.set_title("LW-down distributions", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    season_label = "both seasons" if args.seasons == "both" else args.seasons
    # Quote the range the plotted data actually spans, never the requested one.
    fig.suptitle(
        f"NSA C1 surface downwelling longwave and LW cloud forcing under "
        f"pure-liquid vs pure-ice columns\n"
        f"{plot_start.date()} to {plot_end.date()} ({season_label}; "
        f"{args.classify} rule, {args.lidar})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    # ------------------------------------------------------------------ output
    stem = (
        f"{plot_start.date()}_{plot_end.date()}_{args.classify}_{args.lidar}_"
        f"{args.strictness}_{args.seasons}"
    )
    finish_figure(fig, f"phase_lw_seasonal_{stem}.png", show=not args.no_show)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nc_path = config.PROCESSED_DATA_DIR / f"phase_lw_seasonal_{stem}.nc"
    out = paired.copy()
    out.attrs.update(
        title="THERMOCLDPHASE column phase categories paired with QCRAD fluxes",
        classify_mode=args.classify,
        phase_variable=phase_var,
        strictness=args.strictness,
        min_cloud_pixels=args.min_cloud_pixels,
        crf_baseline=args.crf_baseline,
        winter_months=str(SEASONS["Winter (Oct-Mar)"]),
        summer_months=str(SEASONS["Summer (Apr-Sep)"]),
        note=(
            "column_code: 0 missing, 1 clear, 2 pure_ice, 3 pure_liquid, "
            "4 other. Phase codes per config.THERMO_* (NOT Shupe-Turner "
            "codes)."
        ),
    )
    out.to_netcdf(nc_path)
    csv_path = config.PROCESSED_DATA_DIR / f"phase_lw_seasonal_{stem}.csv"
    stats.to_csv(csv_path)
    print(f"\npaired series written: {nc_path}")
    print(f"statistics written:    {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
