"""Shupe-Turner cloud microphysics: reading and cloud-phase scene analysis.

The MICROBASE2SHUPETURN product (`nsamicrobase2shupeturnC1.c1`) packages the
Arctic-specific multi-sensor retrieval suite of Shupe et al. (2015, JAMC,
doi:10.1175/JAMC-D-15-0054.1), built on the Shupe (2007, GRL, doi:10.1029/
2007GL031008) hydrometeor phase classifier. It combines KAZR cloud radar, MPL
depolarization lidar, ceilometer, microwave radiometer, and radiosondes into:

* `CloudPhaseMask(time, height)` -- integer hydrometeor phase code per volume,
* liquid/ice water content profiles and their column integrals (LWP/IWP),
* at ~1-minute resolution, approximately 2004-2019 at NSA C1.

This is the observationally strongest phase discrimination available at NSA
(radar Doppler moments + lidar depolarization + MWR LWP + sonde temperature),
and it is what Bertrand et al. (2025, Nat. Commun., doi:10.1038/
s41467-025-64441-8) aggregate into the four radiatively distinct scene types:
clear sky, ice-only cloud, mixed-phase cloud, liquid-only cloud.

Scene classification rules (Bertrand25)
---------------------------------------
Following the Bertrand25 analysis code (Zenodo doi:10.5281/zenodo.15786066),
with volume codes grouped per config.py (liquid = {3, 5}, ice = {1, 2},
mixed = {7}):

    mixed-phase  : any mixed volume in the column, OR liquid and ice volumes
                   both present anywhere in the column (even at different
                   heights -- "include ice and liquid at different levels as
                   mixed-phase")
    liquid-only  : liquid volume(s) present, and not mixed-phase
    ice-only     : ice volume(s) present, and neither liquid nor mixed-phase
    clear        : a valid profile with no hydrometeor volumes at all
    other        : hydrometeors present but only of codes outside the three
                   groups (drizzle- or rain-only profiles; vanishingly rare
                   in Arctic winter)
    missing      : no valid phase data in the profile

Note the paper's Methods text describes a subtler liquid-only rule (ice at the
boundaries of the liquid-containing layer); the published code implements the
simpler column rule above for scene frequencies, and that is what we reproduce.

The scene series pairs naturally with QCRAD broadband fluxes (surface.read_
qcrad) to answer: what does downwelling longwave look like under each phase?
See scripts/analyze_phase_radiation.py.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .qc import apply_qc
from .readers import files_in_range, resolve_variable

SCENE_CODES: Dict[int, str] = {
    0: "missing",
    1: "clear",
    2: "ice_only",
    3: "mixed_phase",
    4: "liquid_only",
    5: "other_hydrometeor",
}

# Variables besides the phase mask that are carried along when present in the
# files (canonical name -> candidates). All are optional: product versions
# differ, and the scene classification needs only the phase mask.
_OPTIONAL_VARS: Dict[str, Tuple[str, ...]] = {
    "cloud_fraction": ("Avg_CloudFraction", "cloud_fraction"),
    "lwc_g_m3": ("Avg_Retrieved_LWC", "Avg_LWC", "lwc"),
    "iwc_g_m3": ("Avg_Retrieved_IWC", "Avg_IWC", "iwc"),
    "lwp_g_m2": ("Avg_LWP", "lwp"),
    "iwp_g_m2": ("Avg_IWP", "iwp"),
}
_PHASE_CANDIDATES = ("CloudPhaseMask", "cloud_phase_mask", "CloudPhase")


def read_shupe_turner(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read the Shupe-Turner product for a date range.

    Parameters
    ----------
    start_date, end_date:
        Inclusive "YYYY-MM-DD" bounds. The product exists approximately
        2004-2019 at NSA (config.SHUPETURN_APPROX_*); outside that, expect
        FileNotFoundError even after downloading.

    Returns
    -------
    Dataset with at least phase_code(time, height) [float; NaN = missing
    volume] on the product's native grid, plus whichever of cloud_fraction,
    lwc_g_m3 / iwc_g_m3, lwp_g_m2 / iwp_g_m2 the files carry.

    Notes
    -----
    Files are read one at a time and concatenated (they are daily and modest
    in size, ~tens of MB). The phase mask is kept on the native height grid --
    scene classification is a column reduction, so no regridding is needed.
    """
    paths = files_in_range("shupeturn", start_date, end_date)
    if not paths:
        raise FileNotFoundError(
            f"No local Shupe-Turner files in {start_date}..{end_date} under "
            f"{config.RAW_DATA_DIR}. Download first:\n"
            f"  python scripts/download_nsa_data.py --datastreams shupeturn "
            f"--start {start_date} --end {end_date}\n"
            f"(product coverage is approximately "
            f"{config.SHUPETURN_APPROX_START}..{config.SHUPETURN_APPROX_END})"
        )

    pieces: List[xr.Dataset] = []
    for path in paths:
        if verbose:
            print(f"reading {path.name}")
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            phase_native = resolve_variable(ds, _PHASE_CANDIDATES)
            out_vars: Dict[str, xr.DataArray] = {}
            if apply_qc_flags:
                out_vars["phase_code"] = apply_qc(ds, phase_native)
            else:
                out_vars["phase_code"] = ds[phase_native].astype(float)
            out_vars["phase_code"].attrs.setdefault("source_variable", phase_native)
            for canonical, candidates in _OPTIONAL_VARS.items():
                try:
                    native = resolve_variable(ds, candidates)
                except KeyError:
                    continue  # optional variable absent in this product version
                arr = apply_qc(ds, native) if apply_qc_flags else ds[native]
                arr.attrs.setdefault("source_variable", native)
                out_vars[canonical] = arr
            piece = xr.Dataset(out_vars)
            piece.attrs["source_file"] = path.name
            pieces.append(piece.load())

    combined = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    combined = combined.sortby("time")
    _, unique_idx = np.unique(combined["time"].values, return_index=True)
    if unique_idx.size != combined.sizes["time"]:
        combined = combined.isel(time=unique_idx)
    combined = combined.sel(time=slice(start_date, f"{end_date}T23:59:59"))
    combined["phase_code"].attrs.update(
        long_name="Shupe-Turner hydrometeor phase code per volume",
        flag_notes=(
            "0 clear, 1 ice, 2 snow, 3 liquid, 5 liquid+drizzle, 7 mixed "
            "(groupings in arm_nsa.config; other codes = liquid precip)"
        ),
    )
    return combined


def _height_dim(phase_code: xr.DataArray) -> str:
    """Name of the vertical dimension of the phase mask."""
    non_time = [d for d in phase_code.dims if d != "time"]
    if len(non_time) != 1:
        raise ValueError(
            f"phase_code should have dims (time, height); found {phase_code.dims}"
        )
    return non_time[0]


def classify_scene(st: xr.Dataset) -> xr.DataArray:
    """Collapse each phase-mask profile into one scene category.

    Applies the Bertrand25 column rules (module docstring) to
    st["phase_code"](time, height).

    Returns
    -------
    Integer (int8) DataArray over time with SCENE_CODES semantics in attrs:
    0 missing, 1 clear, 2 ice_only, 3 mixed_phase, 4 liquid_only,
    5 other_hydrometeor.
    """
    phase = st["phase_code"]
    zdim = _height_dim(phase)

    valid = phase.notnull()
    liquid_any = phase.isin(list(config.ST_LIQUID_CODES)).any(dim=zdim)
    ice_any = phase.isin(list(config.ST_ICE_CODES)).any(dim=zdim)
    mixed_any = phase.isin(list(config.ST_MIXED_CODES)).any(dim=zdim)
    known_codes = (
        [config.ST_CLEAR_CODE]
        + list(config.ST_LIQUID_CODES)
        + list(config.ST_ICE_CODES)
        + list(config.ST_MIXED_CODES)
    )
    # Hydrometeors of unlisted codes (drizzle 4, rain 6, ...): present but in
    # none of the three groups.
    other_any = (valid & ~phase.isin(known_codes)).any(dim=zdim)
    any_valid = valid.any(dim=zdim)

    # Mutually exclusive scenes, Bertrand25 precedence:
    is_mixed = mixed_any | (liquid_any & ice_any)
    is_liquid = liquid_any & ~is_mixed
    is_ice = ice_any & ~is_mixed & ~liquid_any
    is_other = other_any & ~is_mixed & ~is_liquid & ~is_ice
    is_clear = any_valid & ~is_mixed & ~is_liquid & ~is_ice & ~other_any

    codes = np.zeros(phase.sizes["time"], dtype=np.int8)  # 0 = missing
    codes[is_clear.values] = 1
    codes[is_ice.values] = 2
    codes[is_mixed.values] = 3
    codes[is_liquid.values] = 4
    codes[is_other.values] = 5

    out = xr.DataArray(
        codes,
        dims=("time",),
        coords={"time": st["time"]},
        name="scene_code",
    )
    out.attrs.update(
        long_name="cloud phase scene classification (Bertrand25 rules)",
        flag_values=list(SCENE_CODES.keys()),
        flag_meanings=" ".join(SCENE_CODES.values()),
        rule=(
            "mixed = mixed volume present OR (liquid AND ice volumes in "
            "column); liquid_only = liquid, not mixed; ice_only = ice, no "
            "liquid, not mixed"
        ),
    )
    return out


def scene_occurrence(scene_code: xr.DataArray) -> Dict[str, float]:
    """Occurrence fraction of each scene type among classified samples.

    "Classified" excludes code 0 (missing). Also returns:
        n_classified            sample count used as the denominator
        liquid_containing_any   mixed_phase + liquid_only
        cloudy_any              1 - clear (all hydrometeor scenes)
    """
    codes = scene_code.values
    classified = codes != 0
    n = int(classified.sum())
    out: Dict[str, float] = {"n_classified": float(n)}
    if n == 0:
        return out
    for code, name in SCENE_CODES.items():
        if code == 0:
            continue
        out[name] = float((codes[classified] == code).mean())
    out["liquid_containing_any"] = out.get("mixed_phase", 0.0) + out.get(
        "liquid_only", 0.0
    )
    out["cloudy_any"] = 1.0 - out.get("clear", 0.0)
    return out


def pair_scene_with_flux(
    scene_code: xr.DataArray,
    flux: xr.Dataset,
    flux_vars: Optional[Sequence[str]] = None,
    tolerance: str = "3min",
) -> xr.Dataset:
    """Attach radiation samples to each classified scene time.

    For every Shupe-Turner time stamp, takes the nearest flux sample within
    `tolerance` (both records are ~1-min, so this is a clock alignment, not an
    average). Scene times with no flux sample inside the tolerance get NaN.

    Parameters
    ----------
    scene_code:
        Output of classify_scene().
    flux:
        Time-indexed Dataset, e.g. surface.read_qcrad() output.
    flux_vars:
        Which flux variables to carry (default: all data variables).

    Returns
    -------
    Dataset on the scene time axis: scene_code + selected flux variables,
    plus net longwave `lwnet_w_m2` when both lwdn_w_m2 and lwup_w_m2 are
    present (downward positive, the Bertrand25 sign convention).
    """
    if flux_vars is None:
        flux_vars = list(flux.data_vars)
    aligned = flux[list(flux_vars)].reindex(
        time=scene_code["time"], method="nearest", tolerance=pd.Timedelta(tolerance)
    )
    paired = aligned.assign(scene_code=scene_code)
    if "lwdn_w_m2" in paired and "lwup_w_m2" in paired:
        paired["lwnet_w_m2"] = paired["lwdn_w_m2"] - paired["lwup_w_m2"]
        paired["lwnet_w_m2"].attrs.update(
            units="W/m^2",
            long_name="net surface longwave flux (down minus up, downward positive)",
        )
    return paired


def scene_flux_stats(paired: xr.Dataset, var: str = "lwdn_w_m2") -> pd.DataFrame:
    """Distribution statistics of one flux variable per scene type.

    Returns a DataFrame indexed by scene name with columns:
        n, occurrence_pct, mean, std, p10, p25, median, p75, p90
    Occurrence is over classified scenes (missing excluded); statistics are
    over the samples of that scene with valid flux data.
    """
    codes = paired["scene_code"].values
    values = paired[var].values.astype(float)
    n_classified = int((codes != 0).sum())

    rows = []
    for code, name in SCENE_CODES.items():
        if code == 0:
            continue
        in_scene = codes == code
        flux_vals = values[in_scene & np.isfinite(values)]
        rows.append(
            {
                "scene": name,
                "n": int(flux_vals.size),
                "occurrence_pct": 100.0 * in_scene.sum() / max(n_classified, 1),
                "mean": np.mean(flux_vals) if flux_vals.size else np.nan,
                "std": np.std(flux_vals) if flux_vals.size else np.nan,
                "p10": np.percentile(flux_vals, 10) if flux_vals.size else np.nan,
                "p25": np.percentile(flux_vals, 25) if flux_vals.size else np.nan,
                "median": np.median(flux_vals) if flux_vals.size else np.nan,
                "p75": np.percentile(flux_vals, 75) if flux_vals.size else np.nan,
                "p90": np.percentile(flux_vals, 90) if flux_vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("scene")


def restrict_to_months(
    ds: xr.Dataset | xr.DataArray, months: Iterable[int]
) -> xr.Dataset | xr.DataArray:
    """Keep only samples whose month is in `months` (e.g. (11,12,1,2,3))."""
    month_values = ds["time"].dt.month
    keep = np.isin(month_values.values, list(months))
    return ds.isel(time=np.where(keep)[0])
