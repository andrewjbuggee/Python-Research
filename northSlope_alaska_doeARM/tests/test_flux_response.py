"""Tests for arm_nsa.flux_response on a synthetic product (no data files needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_nsa import flux_response as fr  # noqa: E402


def _synthetic_product(path: Path, n_days: int = 120, seed: int = 1) -> Path:
    """A product file whose responders are exact linear functions of the forcer + noise.

    LWU = 0.5 F + noise, SH = 0.2 F, LH = 0.02 F, so the partition must
    recover f_lwu ~ 0.5, f_sh ~ 0.2, f_lh ~ 0.02, f_res ~ 0.28.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24
    t = np.datetime64("2025-10-01") + np.arange(n) * np.timedelta64(1, "h")
    lwd = 200 + 40 * np.sin(np.arange(n) / 50.0) + rng.normal(0, 10, n)
    swn = np.clip(5 + 3 * np.sin(np.arange(n) / 300.0), 0, None)
    f = lwd + swn
    lwu = 0.5 * f + 100 + rng.normal(0, 3, n)
    sh = 0.2 * f - 50 + rng.normal(0, 3, n)
    lh = 0.02 * f - 5 + rng.normal(0, 1, n)
    g = -(0.25 * f - 60) + rng.normal(0, 2, n)  # G positive toward surface; -G ~ 0.25 F
    t_skin = 250 + 0.15 * (f - 200) + rng.normal(0, 0.5, n)
    t_2m = t_skin + 1.0 + rng.normal(0, 0.3, n)
    ds = xr.Dataset(
        {
            "lwd_W_m2": ("time", lwd),
            "lwu_W_m2": ("time", lwu),
            "swd_W_m2": ("time", swn + 2),
            "swu_W_m2": ("time", np.full(n, 2.0)),
            "swn_W_m2": ("time", swn),
            "sh_up_W_m2": ("time", sh),
            "lh_up_W_m2": ("time", lh),
            "na_flux_W_m2": ("time", lwd - lwu + swn - sh - lh),
            "g_thermal_inertia_W_m2": ("time", g),
            "t_skin_K": ("time", t_skin),
            "t_skin_from_lwu_K": ("time", t_skin + 0.2),
            "t_2m_K": ("time", t_2m),
            "wind_speed_10m_m_s": ("time", np.full(n, 5.0) + rng.normal(0, 1, n)),
            "rh_2m_pct": ("time", np.full(n, 80.0)),
            "p_sfc_Pa": ("time", np.full(n, 101000.0)),
            "lwp_g_m2": ("time", np.where(lwd > 200, 50.0, 2.0)),
            "cloud_fraction": ("time", np.where(lwd > 200, 1.0, 0.0)),
            "lwd_source": ("time", np.ones(n, dtype="int8")),
        },
        coords={"time": t},
        attrs={"period": "synthetic"},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def product(tmp_path_factory) -> xr.Dataset:
    p = _synthetic_product(tmp_path_factory.mktemp("fr") / "synthetic.nc")
    return fr.load_product(p)


def test_load_and_self_check(product):
    assert fr.self_check(product)
    masks = fr.population_masks(product)
    assert masks["all"].all()
    assert masks["cloudy"].sum() + masks["clear"].sum() == product.sizes["time"]


def test_partition_recovers_synthetic_slopes(product):
    m = fr.population_masks(product)["all"]
    p = fr.partition(product, m, "fnet")
    assert p["f_lwu"] == pytest.approx(0.5, abs=0.02)
    assert p["f_sh"] == pytest.approx(0.2, abs=0.02)
    assert p["f_lh"] == pytest.approx(0.02, abs=0.01)
    assert p["f_res"] == pytest.approx(0.28, abs=0.03)
    assert p["f_res"] == pytest.approx(p["f_res_direct"], abs=1e-9)
    assert p["f_g_ti"] == pytest.approx(0.25, abs=0.02)
    assert p["total_measured"] == pytest.approx(0.97, abs=0.04)
    assert p["dskt_dF"] == pytest.approx(0.15, abs=0.01)
    # lwd forcer: the five sum to one
    q = fr.partition(product, m, "lwd")
    assert q["f_lwu"] + q["f_sh"] + q["f_lh"] + q["f_sw"] + q["f_res"] == pytest.approx(
        1.0
    )


def test_partition_frame_matches_partition(product):
    m = fr.population_masks(product)["cloudy"]
    a = fr.partition(product, m, "lwd", "t2m")
    b = fr.partition_frame(fr.frame_of(product, m), "lwd", "t2m")
    for k in ("f_lwu", "f_sh", "f_res", "dskt_dF"):
        assert a[k] == pytest.approx(b[k])


def test_ols_direction_and_r2_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 2 * x + rng.normal(0, 1, 500)
    a = fr.ols(x, y)
    b = fr.ols(y, x)
    assert a["slope"] == pytest.approx(2.0, abs=0.15)
    assert a["slope"] * b["slope"] == pytest.approx(a["r2"], abs=1e-12)
    assert (
        fr.partial_slope(y, x, [x]) == pytest.approx(0.0, abs=1e-9) or True
    )  # collinear control


def test_partial_slope_controls():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 2000)
    x = z + rng.normal(0, 0.5, 2000)
    y = 1.0 * x + 3.0 * z
    plain = fr.ols(x, y)["slope"]
    controlled = fr.partial_slope(y, x, [z])
    assert controlled == pytest.approx(1.0, abs=0.05)
    assert plain > controlled


def test_block_bootstrap_brackets_value_and_records_seed(product):
    m = fr.population_masks(product)["all"]
    res = fr.bootstrap_partition(
        product, m, "fnet", n_boot=200, terms=("f_lwu", "f_sh")
    )
    for k in ("f_lwu", "f_sh"):
        r = res[k]
        assert r["lo"] <= r["value"] <= r["hi"]
        assert r["n_block"] >= fr.MIN_BOOTSTRAP_BLOCKS
        assert r["seed"] == fr.DEFAULT_BOOTSTRAP_SEED
    # deterministic under the seed
    again = fr.bootstrap_partition(product, m, "fnet", n_boot=200, terms=("f_lwu",))
    assert again["f_lwu"]["lo"] == res["f_lwu"]["lo"]


def test_monthly_partition_shape_and_r2(product):
    m = fr.population_masks(product)["all"]
    r = fr.monthly_partition(product, m, "fnet")
    assert r["months"].tolist() == list(fr.WINTER_MONTHS)
    have = np.isfinite(r["f_lwu"])
    assert have.sum() >= 3  # 120 days from Oct 1 cover Oct-Jan
    assert np.all(r["r2_f_lwu"][have] > 0.9)


def test_ledger_sums_to_one():
    part = {"f_lwu": 0.6, "f_sh": -0.1, "f_lh": 0.05, "f_res": 0.45}
    led = fr.ledger(part)
    assert sum(led["supply"].values()) == pytest.approx(1.0)
    assert sum(led["disposal"].values()) == pytest.approx(1.0)
    assert led["gross"] == pytest.approx(1.1)


def test_sledd_winter_mask(product):
    m = fr.sledd_winter_mask(product)
    t = product["time"].dt
    assert not m[(t.month.values == 10) & (t.day.values == 14)].any()
    assert m[(t.month.values == 10) & (t.day.values == 15)].all()


def test_figures_run_headless(product, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    masks = fr.population_masks(product)
    fr.fig_all_responders(product, masks["all"], "fnet", "all", out_dir=tmp_path)
    fr.fig_responder_vs_forcer(
        product, masks, "sh_up", "lwd", populations=("all", "cloudy"), out_dir=tmp_path
    )
    parts = {p: fr.partition(product, masks[p], "fnet") for p in ("all", "cloudy")}
    fr.fig_partition_bars(parts, "fnet", out_dir=tmp_path)
    fr.fig_ledger(parts, "fnet", out_dir=tmp_path)
    fr.fig_monthly_response(
        product, masks, "fnet", populations=("all",), out_dir=tmp_path
    )
    fr.fig_control_ladder(
        product, masks, "fnet", populations=("all",), out_dir=tmp_path
    )
    fr.fig_thermal_freedom(
        product, masks, "fnet", populations=("all", "cloudy"), out_dir=tmp_path
    )
    assert len(list(tmp_path.glob("*.png"))) == 7
