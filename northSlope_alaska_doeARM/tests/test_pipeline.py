"""Unit tests for the science-critical pieces of arm_nsa, on synthetic data.

No network access and no real ARM files are needed: each test builds tiny
xarray/numpy structures with known answers. Run with either

    pytest tests/ -v
or
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402
from arm_nsa.mwr import _LWP_TO_G_M2, _convert_units  # noqa: E402
from arm_nsa.phase import classify_phase  # noqa: E402
from arm_nsa.qc import apply_qc, qc_mask  # noqa: E402
from arm_nsa.radar import (  # noqa: E402
    clear_sky_flag,
    cloud_fraction_profile,
    iwp_g_m2,
    radar_height_grid_m,
    window_coverage_fraction,
)
from arm_nsa.sonde import _interp_profile, find_saturated_layers  # noqa: E402

# ---------------------------------------------------------------------------
# Saturated layers (Hartig26 rules: 95% threshold, 30-m gap fill, 30-m min)
# ---------------------------------------------------------------------------

# 5-m grid from 8 m to 3000 m, like the real sonde grid but shorter.
_Z = np.arange(8.0, 3000.0, 5.0)


def _rh_profile(segments) -> np.ndarray:
    """RH profile of 50% everywhere except the given (z0, z1, rh) segments."""
    rh = np.full(_Z.shape, 50.0)
    for z0, z1, value in segments:
        rh[(_Z >= z0) & (_Z <= z1)] = value
    return rh


def test_single_layer_detected():
    rh = _rh_profile([(500, 800, 97.0)])
    layers = find_saturated_layers(rh, _Z)
    assert len(layers) == 1
    assert abs(layers[0].base_m - 503.0) <= 5.0  # first grid cell >= 500 m
    assert abs(layers[0].top_m - 798.0) <= 5.0
    assert layers[0].depth_m >= 295.0


def test_small_gap_is_filled():
    # 20-m dry gap inside a saturated region must NOT split the layer.
    rh = _rh_profile([(500, 800, 97.0), (600, 615, 90.0)])
    layers = find_saturated_layers(rh, _Z)
    assert len(layers) == 1


def test_large_gap_splits_layer():
    # A 100-m dry gap exceeds the 30-m fill limit -> two layers.
    rh = _rh_profile([(500, 900, 97.0), (650, 745, 90.0)])
    layers = find_saturated_layers(rh, _Z)
    assert len(layers) == 2
    assert layers[0].top_m < layers[1].base_m


def test_thin_layer_discarded():
    # A 20-m saturated sliver is below the 30-m minimum depth.
    rh = _rh_profile([(1500, 1515, 96.0)])
    layers = find_saturated_layers(rh, _Z)
    assert layers == []


def test_nan_rh_is_unsaturated():
    rh = _rh_profile([(500, 800, 97.0)])
    rh[_Z > 2000] = np.nan  # e.g. sensor dropout aloft
    layers = find_saturated_layers(rh, _Z)
    assert len(layers) == 1


def test_no_layers_when_dry():
    assert find_saturated_layers(_rh_profile([]), _Z) == []


# ---------------------------------------------------------------------------
# Sonde interpolation
# ---------------------------------------------------------------------------


def test_interp_handles_duplicates_and_span():
    grid = np.array([0.0, 10.0, 20.0, 30.0])
    # Altitudes include a duplicate (balloon bobbing); values linear in z.
    alt = np.array([5.0, 15.0, 15.0, 25.0])
    val = np.array([5.0, 15.0, 99.0, 25.0])  # duplicate gets dropped
    out = _interp_profile(alt, val, grid)
    assert np.isnan(out[0])  # below sounding span: no extrapolation
    assert abs(out[1] - 10.0) < 1e-9
    assert abs(out[2] - 20.0) < 1e-9
    assert np.isnan(out[3])  # above span


# ---------------------------------------------------------------------------
# ARM QC bitmask handling
# ---------------------------------------------------------------------------


def _qc_dataset(qc_attrs) -> xr.Dataset:
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time")
    qc = xr.DataArray([0, 1, 2, 4], dims="time", attrs=qc_attrs)
    return xr.Dataset({"x": data, "qc_x": qc})


def test_qc_new_style_bad_only():
    ds = _qc_dataset(
        {"flag_masks": [1, 2, 4], "flag_assessments": ["Bad", "Indeterminate", "Bad"]}
    )
    mask = qc_mask(ds, "x")
    assert mask.values.tolist() == [False, True, False, True]
    cleaned = apply_qc(ds, "x")
    assert np.isnan(cleaned.values[[1, 3]]).all()
    assert np.isfinite(cleaned.values[[0, 2]]).all()


def test_qc_include_indeterminate():
    ds = _qc_dataset(
        {"flag_masks": [1, 2, 4], "flag_assessments": ["Bad", "Indeterminate", "Bad"]}
    )
    mask = qc_mask(ds, "x", include_indeterminate=True)
    assert mask.values.tolist() == [False, True, True, True]


def test_qc_old_style_bits():
    ds = _qc_dataset(
        {
            "bit_1_description": "missing", "bit_1_assessment": "Bad",
            "bit_2_description": "low", "bit_2_assessment": "Indeterminate",
            "bit_3_description": "high", "bit_3_assessment": "Bad",
        }
    )
    mask = qc_mask(ds, "x")
    assert mask.values.tolist() == [False, True, False, True]


def test_qc_missing_sentinel_masked():
    ds = xr.Dataset({"x": xr.DataArray([1.0, -9999.0, 3.0], dims="time")})
    assert qc_mask(ds, "x").values.tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Radar-derived quantities
# ---------------------------------------------------------------------------


def _kazr_window(refl_dbz: np.ndarray, has_data=None) -> xr.Dataset:
    """Build a synthetic KAZR window Dataset (time, height)."""
    n_time, n_height = refl_dbz.shape
    grid = radar_height_grid_m()[:n_height]
    ds = xr.Dataset(
        {
            "reflectivity_dbz": (("time", "height"), refl_dbz),
            "detected": (("time", "height"), np.isfinite(refl_dbz)),
        },
        coords={
            "time": np.arange(n_time).astype("datetime64[s]").astype("datetime64[ns]"),
            "height": grid,
        },
    )
    if has_data is None:
        has_data = np.ones(n_time, dtype=bool)
    ds["has_data"] = ("time", np.asarray(has_data, dtype=bool))
    return ds


def test_iwp_constant_reflectivity_analytic():
    # 0 dBZ everywhere -> Ze = 1 mm^6/m^3 -> IWC = a * 1**b = 0.1 g/m^3.
    # IWP = IWC * dz * n_bins, identical for both averaging orders.
    n_time, n_height = 10, 50
    refl = np.zeros((n_time, n_height))
    window = _kazr_window(refl)
    expected = config.IWC_PREFACTOR_A * config.RADAR_GRID_STEP_M * n_height
    assert abs(iwp_g_m2(window, average_first=True) - expected) < 1e-6
    assert abs(iwp_g_m2(window, average_first=False) - expected) < 1e-6


def test_iwp_zero_for_clear_but_reporting_radar():
    refl = np.full((10, 50), np.nan)  # nothing detected
    window = _kazr_window(refl, has_data=np.ones(10, dtype=bool))
    assert iwp_g_m2(window) == 0.0


def test_iwp_nan_when_radar_off():
    refl = np.full((10, 50), np.nan)
    window = _kazr_window(refl, has_data=np.zeros(10, dtype=bool))
    assert np.isnan(iwp_g_m2(window))


def test_averaging_order_matters_for_variable_ze():
    # IWC ~ Ze**0.63 is concave, so mean-then-power >= power-then-mean
    # (Jensen); with Ze alternating 10 and 1000 the two must differ.
    refl = np.empty((2, 30))
    refl[0, :], refl[1, :] = 10.0, 30.0  # dBZ
    window = _kazr_window(refl)
    first = iwp_g_m2(window, average_first=True)
    second = iwp_g_m2(window, average_first=False)
    assert first > second > 0


def test_coverage_fraction():
    refl = np.full((10, 50), np.nan)
    has_data = np.array([True] * 5 + [False] * 5)
    window = _kazr_window(refl, has_data=has_data)
    assert abs(window_coverage_fraction(window) - 0.5) < 1e-9


def test_cloud_fraction_profile_ignores_missing():
    # 4 reporting samples, 2 detecting at bin 0 -> fraction 0.5 at bin 0.
    refl = np.full((6, 20), np.nan)
    refl[0, 0] = refl[1, 0] = -20.0
    has_data = np.array([True, True, True, True, False, False])
    window = _kazr_window(refl, has_data=has_data)
    frac = cloud_fraction_profile(window)
    assert abs(float(frac[0]) - 0.5) < 1e-9
    assert float(frac[1]) == 0.0


def test_clear_sky_flag_cases():
    n_time = 20
    n_height = radar_height_grid_m().size

    # (1) reporting radar, zero detections -> clear.
    refl = np.full((n_time, n_height), np.nan)
    assert clear_sky_flag(_kazr_window(refl)) is True

    # (2) a shallow 60-m echo (2 bins) in a couple of samples -> still clear.
    refl2 = refl.copy()
    refl2[3, 10:12] = -30.0
    assert clear_sky_flag(_kazr_window(refl2)) is True

    # (3) a 150-m deep echo (5 bins) exceeds the 100-m layer limit -> cloudy.
    refl3 = refl.copy()
    refl3[3, 10:15] = -30.0
    assert clear_sky_flag(_kazr_window(refl3)) is False

    # (4) persistent deep cloud -> cloudy.
    refl4 = refl.copy()
    refl4[:, 20:60] = -10.0
    assert clear_sky_flag(_kazr_window(refl4)) is False

    # (5) radar off -> cannot claim clear.
    off = _kazr_window(refl, has_data=np.zeros(n_time, dtype=bool))
    assert clear_sky_flag(off) is False


# ---------------------------------------------------------------------------
# MWR unit conversion
# ---------------------------------------------------------------------------


def test_lwp_unit_conversion_cm_to_g_m2():
    # 0.01 cm of liquid = 100 g/m^2 (1 cm = 10^4 g/m^2).
    arr = xr.DataArray([0.01], dims="time", attrs={"units": "cm"})
    out = _convert_units(arr, _LWP_TO_G_M2, "g/m^2")
    assert abs(float(out[0]) - 100.0) < 1e-9
    assert out.attrs["units"] == "g/m^2"


def test_lwp_unknown_units_passthrough_with_note():
    arr = xr.DataArray([42.0], dims="time", attrs={"units": "furlongs"})
    out = _convert_units(arr, _LWP_TO_G_M2, "g/m^2")
    assert float(out[0]) == 42.0
    assert "units_note" in out.attrs


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------


def _library_row(
    lwp_g_m2, clear, coverage, n_layers, ceil_frac
) -> xr.Dataset:
    def col(values, dtype=float):
        return ("launch_time", np.asarray(values, dtype=dtype))

    n = len(lwp_g_m2)
    return xr.Dataset(
        {
            "lwp_g_m2": col(lwp_g_m2),
            "clear_sky": col(clear, dtype=bool),
            "radar_coverage_fraction": col(coverage),
            "n_saturated_layers": col(n_layers, dtype=int),
            "ceil_cloud_fraction": col(ceil_frac),
        },
        coords={"launch_time": np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")},
    )


def test_phase_classification_matrix():
    library = _library_row(
        lwp_g_m2=[50.0, 5.0, 5.0, 2.0, np.nan, 30.0],
        clear=[False, False, False, True, False, False],
        coverage=[1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        n_layers=[1, 1, 0, 0, 0, 0],
        ceil_frac=[1.0, 1.0, 1.0, 0.0, np.nan, np.nan],
    )
    codes = classify_phase(library).values.tolist()
    assert codes[0] == 4  # LWP 50: liquid confident
    assert codes[1] == 3  # cloudy, LWP 5, saturated layer: liquid probable
    assert codes[2] == 2  # cloudy, LWP 5, no saturated layer: ice probable
    assert codes[3] == 1  # radar clear, LWP small: clear
    assert codes[4] == 0  # nothing usable: missing
    assert codes[5] == 4  # MWR alone is sufficient for liquid confident


def _run_all() -> int:
    """Plain-python runner (no pytest needed)."""
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as err:
                failures += 1
                print(f"FAIL {name}: {err}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
