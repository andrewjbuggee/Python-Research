"""Unit tests for Shupe-Turner scene classification and flux pairing.

All synthetic, no files needed. Run with pytest or plain python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa.shupe_turner import (  # noqa: E402
    SCENE_CODES,
    classify_scene,
    pair_scene_with_flux,
    restrict_to_months,
    scene_flux_stats,
    scene_occurrence,
)

_NAME_TO_CODE = {name: code for code, name in SCENE_CODES.items()}


def _st_dataset(profiles) -> xr.Dataset:
    """Build a phase-mask Dataset from a list of per-time profiles."""
    arr = np.array(profiles, dtype=float)
    n_time, n_height = arr.shape
    times = (
        np.datetime64("2015-01-10T00:00:00", "ns")
        + np.arange(n_time) * np.timedelta64(60, "s")
    )
    return xr.Dataset(
        {"phase_code": (("time", "height"), arr)},
        coords={"time": times, "height": 100.0 + 90.0 * np.arange(n_height)},
    )


def test_scene_classification_matrix():
    nan = np.nan
    profiles = [
        [0, 0, 0, 0, 0],      # all clear volumes            -> clear
        [0, 1, 2, 0, 0],      # ice + snow volumes           -> ice_only
        [0, 3, 3, 0, 0],      # liquid volumes               -> liquid_only
        [0, 5, 0, 0, 0],      # liquid+drizzle volume        -> liquid_only
        [0, 3, 0, 1, 0],      # liquid AND ice, diff levels  -> mixed_phase
        [0, 7, 0, 0, 0],      # explicit mixed volume        -> mixed_phase
        [0, 4, 0, 0, 0],      # drizzle only (unlisted code) -> other
        [nan, nan, nan, nan, nan],  # no data                -> missing
    ]
    expected = [
        "clear", "ice_only", "liquid_only", "liquid_only",
        "mixed_phase", "mixed_phase", "other_hydrometeor", "missing",
    ]
    scene = classify_scene(_st_dataset(profiles))
    got = [SCENE_CODES[int(c)] for c in scene.values]
    assert got == expected


def test_drizzle_with_ice_is_ice_only_per_bertrand():
    # Bertrand25 groups liquid as {3, 5} only, so drizzle (4) + ice (1) does
    # NOT count as mixed -- the profile classifies as ice_only. Intentional
    # reproduction of their rules; negligible in Arctic winter.
    scene = classify_scene(_st_dataset([[0, 4, 1, 0, 0]]))
    assert SCENE_CODES[int(scene.values[0])] == "ice_only"


def test_scene_occurrence_fractions():
    profiles = (
        [[0, 0, 0, 0, 0]] * 2       # clear x2
        + [[0, 1, 0, 0, 0]] * 1     # ice x1
        + [[0, 7, 0, 0, 0]] * 1     # mixed x1
        + [[np.nan] * 5] * 4        # missing x4 (excluded from denominator)
    )
    occ = scene_occurrence(classify_scene(_st_dataset(profiles)))
    assert occ["n_classified"] == 4.0
    assert abs(occ["clear"] - 0.5) < 1e-9
    assert abs(occ["ice_only"] - 0.25) < 1e-9
    assert abs(occ["mixed_phase"] - 0.25) < 1e-9
    assert abs(occ["liquid_containing_any"] - 0.25) < 1e-9
    assert abs(occ["cloudy_any"] - 0.5) < 1e-9


def test_pairing_alignment_and_tolerance():
    scene = classify_scene(_st_dataset([[0, 0, 0, 0, 0]] * 8))
    # Flux offset by +20 s from the first four scene stamps. Scene minute 4
    # is 40 s from the last flux sample (within the 1-min tolerance, pairs);
    # minutes 5-7 are > 1 min away and must come back NaN.
    flux_times = scene["time"].values[:4] + np.timedelta64(20, "s")
    flux = xr.Dataset(
        {"lwdn_w_m2": ("time", np.array([200.0, 210.0, 220.0, 230.0]))},
        coords={"time": flux_times},
    )
    paired = pair_scene_with_flux(scene, flux, tolerance="1min")
    values = paired["lwdn_w_m2"].values
    assert np.allclose(values[:4], [200.0, 210.0, 220.0, 230.0])
    assert values[4] == 230.0  # 40 s away: inside tolerance
    assert np.isnan(values[5:]).all()  # > 1 min away: rejected


def test_net_lw_computed_when_up_present():
    scene = classify_scene(_st_dataset([[0, 0, 0, 0, 0]] * 3))
    flux = xr.Dataset(
        {
            "lwdn_w_m2": ("time", np.array([200.0, 250.0, 300.0])),
            "lwup_w_m2": ("time", np.array([240.0, 250.0, 260.0])),
        },
        coords={"time": scene["time"].values},
    )
    paired = pair_scene_with_flux(scene, flux)
    assert np.allclose(paired["lwnet_w_m2"].values, [-40.0, 0.0, 40.0])


def test_scene_flux_stats_table():
    # 3 clear minutes at 160 W/m^2, 2 mixed minutes at 300 W/m^2.
    profiles = [[0] * 5] * 3 + [[0, 7, 0, 0, 0]] * 2
    scene = classify_scene(_st_dataset(profiles))
    flux = xr.Dataset(
        {"lwdn_w_m2": ("time", np.array([160.0] * 3 + [300.0] * 2))},
        coords={"time": scene["time"].values},
    )
    stats = scene_flux_stats(pair_scene_with_flux(scene, flux))
    assert stats.loc["clear", "n"] == 3
    assert abs(stats.loc["clear", "median"] - 160.0) < 1e-9
    assert abs(stats.loc["clear", "occurrence_pct"] - 60.0) < 1e-9
    assert stats.loc["mixed_phase", "n"] == 2
    assert abs(stats.loc["mixed_phase", "mean"] - 300.0) < 1e-9
    assert stats.loc["liquid_only", "n"] == 0


def test_restrict_to_months():
    n = 90 * 24  # hourly samples over ~3 months starting Dec 1
    times = (
        np.datetime64("2014-12-01T00:00:00", "ns")
        + np.arange(n) * np.timedelta64(3600, "s")
    )
    ds = xr.Dataset({"x": ("time", np.arange(n, dtype=float))},
                    coords={"time": times})
    kept = restrict_to_months(ds, (12, 1))
    months = pd.DatetimeIndex(kept["time"].values).month
    assert set(months) == {12, 1}
    assert kept.sizes["time"] < n


def _run_all() -> int:
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
