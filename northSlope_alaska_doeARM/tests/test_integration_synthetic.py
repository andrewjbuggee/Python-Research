"""End-to-end pipeline test on synthetic ARM-format netCDF files.

Fabricates one day (2022-01-05) of NSA-like files for all six datastreams in
a temporary data root -- a sounding at 11:29 UTC through a liquid cloud at
400-700 m, an MWR reporting LWP = 60 g/m^2, a ceilometer base at ~450 m, a
KAZR echo co-located with the cloud, plus MET and QCRAD series -- then runs
the real readers and coordinate.build_library() against them and checks the
numbers that come out.

This exercises everything the unit tests cannot: filename parsing, globbing
across product-era directories, netCDF round-trips, QC application, candidate
variable-name resolution, window averaging, and phase classification.

Run:  python tests/test_integration_synthetic.py   (or pytest)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import config  # noqa: E402

DAY = "2022-01-05"
LAUNCH = np.datetime64(f"{DAY}T11:29:00", "ns")


def _times(start: str, end: str, step_s: float) -> np.ndarray:
    t0 = np.datetime64(start, "ns")
    t1 = np.datetime64(end, "ns")
    step = np.timedelta64(int(step_s * 1e9), "ns")
    return np.arange(t0, t1, step)


def _write(ds: xr.Dataset, datastream: str, stamp: str) -> None:
    out_dir = config.RAW_DATA_DIR / datastream
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_dir / f"{datastream}.{stamp}.nc")


def _make_sonde() -> None:
    """One launch at 11:29 UTC, saturated (97% RH) between 400 and 700 m."""
    n = 1200
    alt_m = np.linspace(10.0, 12_000.0, n)
    time = LAUNCH + (np.arange(n) * np.timedelta64(1, "s"))
    rh = np.full(n, 60.0)
    rh[(alt_m >= 400.0) & (alt_m <= 700.0)] = 97.0
    tdry_c = 15.0 - 0.0065 * alt_m - 40.0  # cold Arctic profile, ~-40 C sfc
    ds = xr.Dataset(
        {
            "pres": ("time", 1013.0 * np.exp(-alt_m / 8000.0)),
            "tdry": ("time", tdry_c),
            "dp": ("time", tdry_c - 2.0),
            "rh": ("time", rh),
            "wspd": ("time", np.full(n, 5.0)),
            "deg": ("time", np.full(n, 90.0)),
            "u_wind": ("time", np.full(n, -5.0)),
            "v_wind": ("time", np.zeros(n)),
            "alt": ("time", alt_m),
        },
        coords={"time": time},
    )
    _write(ds, "nsasondewnpnC1.b1", f"{DAY.replace('-', '')}.112900")


def _make_mwr() -> None:
    """LWP 60 g/m^2 and PWV 0.5 cm all day, with a benign QC variable."""
    time = _times(f"{DAY}T00:00:00", f"{DAY}T23:59:59", 30.0)
    n = time.size
    lwp = xr.DataArray(
        np.full(n, 60.0), dims="time", attrs={"units": "g/m^2"}
    )
    qc = xr.DataArray(
        np.zeros(n, dtype=np.int32),
        dims="time",
        attrs={
            "flag_masks": np.array([1, 2], dtype=np.int32),
            "flag_assessments": ["Bad", "Indeterminate"],
        },
    )
    ds = xr.Dataset(
        {
            "be_lwp": lwp,
            "qc_be_lwp": qc,
            "be_pwv": xr.DataArray(
                np.full(n, 0.5), dims="time", attrs={"units": "cm"}
            ),
        },
        coords={"time": time},
    )
    _write(ds, "nsamwrret1liljclouC1.c2", f"{DAY.replace('-', '')}.000000")


def _make_ceil() -> None:
    """Cloud base 450 m with detection status 1 from 11:00 to 13:00 UTC."""
    time = _times(f"{DAY}T00:00:00", f"{DAY}T23:59:59", 15.0)
    n = time.size
    cloudy = (time >= np.datetime64(f"{DAY}T11:00:00")) & (
        time < np.datetime64(f"{DAY}T13:00:00")
    )
    cbh = np.where(cloudy, 450.0, np.nan)
    status = np.where(cloudy, 1, 0).astype(np.int32)
    ds = xr.Dataset(
        {
            "first_cbh": ("time", cbh),
            "detection_status": ("time", status),
        },
        coords={"time": time},
    )
    _write(ds, "nsaceilC1.b1", f"{DAY.replace('-', '')}.000000")


def _make_kazr() -> None:
    """Echo (-15 dBZ, SNR 5 dB) at 400-700 m, 11:00-13:00; noise elsewhere."""
    time = _times(f"{DAY}T11:00:00", f"{DAY}T13:00:00", 4.0)
    rng = np.arange(100.0, 12_000.0, 30.0)
    n_t, n_r = time.size, rng.size
    refl = np.full((n_t, n_r), -60.0, dtype=np.float32)  # noise floor
    snr = np.full((n_t, n_r), -20.0, dtype=np.float32)  # below -13 dB screen
    in_cloud = (rng >= 400.0) & (rng <= 700.0)
    refl[:, in_cloud] = -15.0
    snr[:, in_cloud] = 5.0
    ds = xr.Dataset(
        {
            "reflectivity": (("time", "range"), refl),
            "signal_to_noise_ratio_copol": (("time", "range"), snr),
        },
        coords={"time": time, "range": rng},
    )
    # Use the newest product name to prove era-globbing works.
    _write(ds, "nsakazrcfrcorgeC1.c0", f"{DAY.replace('-', '')}.110000")


def _make_met() -> None:
    time = _times(f"{DAY}T00:00:00", f"{DAY}T23:59:59", 60.0)
    n = time.size
    ds = xr.Dataset(
        {
            "temp_mean": xr.DataArray(
                np.full(n, -25.0), dims="time", attrs={"units": "degC"}
            ),
            "rh_mean": ("time", np.full(n, 80.0)),
            "wspd_vec_mean": ("time", np.full(n, 4.0)),
            "wdir_vec_mean": ("time", np.full(n, 45.0)),
            "atmos_pressure": ("time", np.full(n, 101.3)),
        },
        coords={"time": time},
    )
    _write(ds, "nsametC1.b1", f"{DAY.replace('-', '')}.000000")


def _make_qcrad() -> None:
    time = _times(f"{DAY}T00:00:00", f"{DAY}T23:59:59", 60.0)
    n = time.size
    ds = xr.Dataset(
        {
            "BestEstimate_down_short_hemisp": ("time", np.zeros(n)),  # polar night
            "down_long_hemisp_shaded": ("time", np.full(n, 230.0)),  # cloudy state
            "up_short_hemisp": ("time", np.zeros(n)),
            "up_long_hemisp": ("time", np.full(n, 250.0)),
        },
        coords={"time": time},
    )
    _write(ds, "nsaqcrad1longC1.c2", f"{DAY.replace('-', '')}.000000")


def _make_shupeturn() -> None:
    """1-min phase mask: 6 h each of clear, liquid, mixed, ice scenes."""
    time = _times(f"{DAY}T00:00:00", f"{DAY}T23:59:59", 60.0)
    n_t = time.size
    height = 100.0 + 90.0 * np.arange(20)
    phase = np.zeros((n_t, height.size), dtype=np.float32)  # 0 = clear
    hours = (time - np.datetime64(f"{DAY}T00:00:00")) / np.timedelta64(1, "h")
    liquid_block = (hours >= 6) & (hours < 12)
    mixed_block = (hours >= 12) & (hours < 18)
    ice_block = hours >= 18
    phase[np.ix_(liquid_block, [2, 3, 4])] = 3.0  # liquid volumes
    phase[np.ix_(mixed_block, [2, 3])] = 3.0  # liquid ...
    phase[np.ix_(mixed_block, [6])] = 1.0  # ... plus ice aloft -> mixed
    phase[np.ix_(ice_block, [5, 6, 7])] = 2.0  # snow volumes -> ice_only
    ds = xr.Dataset(
        {
            "CloudPhaseMask": (("time", "height"), phase),
            "Avg_CloudFraction": (("time", "height"), (phase > 0).astype(np.float32)),
        },
        coords={"time": time, "height": height},
    )
    _write(ds, "nsamicrobase2shupeturnC1.c1", f"{DAY.replace('-', '')}.000000")


def test_scene_classification_and_flux_pairing_on_synthetic_day() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="arm_nsa_synth_st_"))
    orig = (config.DATA_ROOT, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
    config.DATA_ROOT = tmp_root
    config.RAW_DATA_DIR = tmp_root / "raw"
    config.PROCESSED_DATA_DIR = tmp_root / "processed"
    try:
        _make_shupeturn()
        _make_qcrad()

        from arm_nsa.shupe_turner import (
            SCENE_CODES,
            classify_scene,
            pair_scene_with_flux,
            read_shupe_turner,
            scene_flux_stats,
            scene_occurrence,
        )
        from arm_nsa.surface import read_qcrad

        st = read_shupe_turner(DAY, DAY)
        assert "cloud_fraction" in st  # optional variable picked up

        scene = classify_scene(st)
        occ = scene_occurrence(scene)
        # Four 6-h blocks -> 25% each of clear/liquid/mixed/ice.
        assert abs(occ["clear"] - 0.25) < 0.01
        assert abs(occ["liquid_only"] - 0.25) < 0.01
        assert abs(occ["mixed_phase"] - 0.25) < 0.01
        assert abs(occ["ice_only"] - 0.25) < 0.01
        assert abs(occ["liquid_containing_any"] - 0.5) < 0.01

        paired = pair_scene_with_flux(scene, read_qcrad(DAY, DAY))
        stats = scene_flux_stats(paired, "lwdn_w_m2")
        # Synthetic QCRAD LW-down is constant 230 W/m^2.
        for name in ("clear", "ice_only", "mixed_phase", "liquid_only"):
            assert stats.loc[name, "n"] > 300
            assert abs(stats.loc[name, "median"] - 230.0) < 1e-6
        # Net LW = 230 - 250 = -20 W/m^2 everywhere.
        stats_net = scene_flux_stats(paired, "lwnet_w_m2")
        assert abs(stats_net.loc["clear", "mean"] - (-20.0)) < 1e-6
        assert SCENE_CODES[0] == "missing"  # sanity on the coding contract
        print("shupe-turner integration test PASSED")
    finally:
        config.DATA_ROOT, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR = orig
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_full_pipeline_on_synthetic_day() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="arm_nsa_synth_"))
    # Point the pipeline's data root at the temp dir. config functions read
    # these module globals at call time, so patching them is sufficient.
    orig = (config.DATA_ROOT, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
    config.DATA_ROOT = tmp_root
    config.RAW_DATA_DIR = tmp_root / "raw"
    config.PROCESSED_DATA_DIR = tmp_root / "processed"
    try:
        for maker in (_make_sonde, _make_mwr, _make_ceil, _make_kazr,
                      _make_met, _make_qcrad):
            maker()

        from arm_nsa.coordinate import build_library, save_library
        from arm_nsa.phase import classify_phase

        library = build_library(DAY, DAY, include_extensions=True, verbose=False)

        assert library.sizes["launch_time"] == 1
        row = library.isel(launch_time=0)

        # Sounding: exactly one saturated layer, spanning ~400-700 m.
        assert int(row["n_saturated_layers"]) == 1
        assert 350.0 <= float(row["lowest_base_m"]) <= 450.0
        assert 650.0 <= float(row["lowest_top_m"]) <= 750.0

        # MWR window mean.
        assert abs(float(row["lwp_g_m2"]) - 60.0) < 1e-6
        assert abs(float(row["pwv_cm"]) - 0.5) < 1e-6

        # Ceilometer: base at 450 m, cloud in every sample of the window.
        assert abs(float(row["first_cbh_m"]) - 450.0) < 1e-6
        assert abs(float(row["ceil_cloud_fraction"]) - 1.0) < 1e-6

        # Radar: full coverage, cloudy (not clear), positive IWP, and the
        # cloud fraction profile peaking to 1 inside 400-700 m.
        assert abs(float(row["radar_coverage_fraction"]) - 1.0) < 1e-6
        assert not bool(row["clear_sky"])
        # Analytic IWP: Ze = 10**(-1.5) mm^6/m^3 over an 11-bin, 30-m layer.
        iwc_g_m3 = config.IWC_PREFACTOR_A * (10.0 ** (-15.0 / 10.0)) ** config.IWC_EXPONENT_B
        expected_iwp = iwc_g_m3 * 30.0 * 11
        assert abs(float(row["iwp_g_m2"]) - expected_iwp) / expected_iwp < 0.15
        prof = row["cloud_frac_profile"].values
        heights = library["radar_height"].values
        assert prof[(heights > 450) & (heights < 650)].min() > 0.99
        assert prof[heights > 1000].max() == 0.0

        # Extensions.
        assert abs(float(row["temp_2m_c"]) - (-25.0)) < 1e-6
        assert abs(float(row["lwdn_w_m2"]) - 230.0) < 1e-6

        # Phase: LWP 60 >= 10 g/m^2 -> liquid_confident (4).
        code = int(classify_phase(library).values[0])
        assert code == 4

        # Round-trip through save_library.
        out_path = save_library(library)
        reread = xr.open_dataset(out_path)
        assert reread.sizes["launch_time"] == 1
        reread.close()
        print("integration test PASSED")
    finally:
        config.DATA_ROOT, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR = orig
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_full_pipeline_on_synthetic_day()
    test_scene_classification_and_flux_pairing_on_synthetic_day()
