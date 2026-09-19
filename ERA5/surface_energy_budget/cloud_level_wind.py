#!/usr/bin/env python3
"""The wind AT CLOUD LEVEL, from the pressure-level archive, for Taylor's hypothesis.

``cloud_spatial_extent.py`` converts a cloud duration into a horizontal length
with L = U dt and takes U from the 10 m wind, because that is the only wind in
the single-level archive. Under a stable Arctic boundary layer the 10 m wind is
a LOWER bound on the wind that advects a cloud a few hundred metres up, so the
lengths it gives are conservative by an unknown factor. This module supplies
the missing piece: the wind inside the liquid cloud itself, from the
pressure-level archive.

WHAT IS COMPUTED, PER CELL AND HOUR
===================================
The archive holds specific cloud liquid water content ``clwc`` (kg per kg of
moist air, a grid-box mean) on pressure levels, and -- once downloaded with
``download_era5_pressure.py --var-set winds`` -- the wind components ``u``,
``v`` on the same levels. Hydrostatic balance puts dp/g kilograms of air above
unit area between two pressure surfaces, so the liquid mass in the layer that
level k owns is

    w_k = clwc_k * dp_k / g                                   [kg m-2]     (1)

with dp_k the layer's pressure thickness, clipped at the surface pressure so a
level under the terrain contributes nothing (``layer_thickness_pa`` in
``convert_specific_to_absolute.py``; the layer edges sit at the midpoints
between levels). The in-cloud wind is the liquid-mass-weighted mean over levels,

    u_cl = sum_k w_k u_k / sum_k w_k,    v_cl likewise,                    (2)
    U_cl = sum_k w_k |u_k, v_k| / sum_k w_k                                (3)

(2) is the vector mean, the displacement per unit time of the liquid-weighted
column; (3) is the mean speed, matching the scalar convention of the 10 m
estimate. Both are returned. Weighting by liquid mass rather than by cloud
fraction answers the question asked -- where is the LIQUID, and how fast is it
moving -- and it is the same weighting ERA5's own tclw uses to form the column,
so sum_k w_k is the pressure-level LWP and can be checked against tclw.

The liquid-weighted mean pressure

    p_cl = sum_k w_k p_k / sum_k w_k                                       (4)

is returned too: it is the pressure the wind is being taken at, and its
distribution is the "cloud level" the module's name refers to. It needs only
``clwc``, so it is available before the winds are.

WHERE IT IS COMPUTED
====================
Only at cell-hours the single-level filter calls liquid-containing (``E.liq``
from ``prepare_extent``), so the population is exactly the one every other
figure uses. Two things can still leave a cell-hour without a value:

* the pressure-level integral finds no liquid at all where ``tclw`` says there
  is some -- a thin layer the 23 levels straddle, about 15% of liquid
  cell-hours. There are no weights to average with, so the wind at a single
  FALLBACK level is used instead: 950 hPa by default, the measured median of
  p_cl, i.e. where the liquid usually is. The count is reported and the
  cell-hours are flagged (``used_fallback``); ``fallback_hpa=None`` leaves
  them NaN instead;
* the hour is outside the pressure-level archive. It covers two complete cold
  seasons (2024/25 and 2025/26) against eleven in the single-level archive, so
  the comparison is made on those two, with the 10 m estimate recomputed on
  the same hours rather than borrowed from the eleven-season run.

WINDS ON FEWER LEVELS THAN THE CLOUD WATER
==========================================
The cloud-water archive runs to 200 hPa; the winds are fetched on the lower
troposphere (1000-500 hPa) because that is where the liquid is -- the notebook
measures the share of liquid mass above 500 hPa from the cloud-water archive
itself and reports it. The weights are formed on the cloud-water levels and
then restricted to the levels the winds exist on; the liquid above them is
dropped from the mean and its share returned as ``liquid_share_unwinded``.

WHAT IT DOES NOT FIX
====================
Taylor's hypothesis is still the hypothesis: a better U changes the scale of L,
not the finding in section 4 of the notebook that short ERA5 events at a cell
end when the liquid-containing STATE switches off rather than when a cloud edge
passes. Expect the cloud-level wind to move the Taylor lengths up by the
in-cloud-to-10 m speed ratio, and the pairing against the snapshot extents to
improve only where advection really was the cause.
"""

from __future__ import annotations

import glob
import time
from datetime import date
from pathlib import Path

import numpy as np

import cloud_spatial_extent as cse
from cloud_spatial_extent import (
    DOMAIN_COLOR,
    SITE_COLOR,
    _cell_reference_lines,
    _q,
    _save,
    _season_span,
    _threshold_box,
    _tidy,
    precip_banner,
    precip_label,
)
from plot_lwp_histogram_by_surface_class import (
    DEFAULT_DURATION_MODE,
    DEFAULT_WIND_SOURCE,
    DURATION_MODES,
    WIND_SOURCES,
    threshold_box_lines,
)
from convert_specific_to_absolute import G_M_S2, layer_thickness_pa
from download_era5_seb import days_covered_by_file
from seb_analysis_common import resolve_data_root

# Colour for the cloud-level series, distinct from the 10 m red and the ARM blue.
CLOUD_WIND_COLOR = "#2a9d8f"      # teal
DEFAULT_PRESSURE_SUFFIX = "_pressure"
DEFAULT_WIND_SUFFIX = "_pressure_wind"


# ----------------------------------------------------------------------------
# Locating and opening the archives
# ----------------------------------------------------------------------------
def archive_dir(E, suffix: str, data_root=None) -> Path:
    """``<root>/<region><suffix>`` for the run's storage setting."""
    root = (Path(data_root) if data_root is not None
            else resolve_data_root(E.args.storage, E.args.data_root))
    return root / f"{E.args.region}{suffix}"


def files_for_times(directory: Path, times: np.ndarray) -> list[str]:
    """The archive files whose named day range overlaps ``times``.

    Uses the day range encoded in the file name (the same convention the
    downloader's resume relies on) rather than opening every file.
    """
    days = set(np.unique(times.astype("datetime64[D]")).astype("datetime64[D]")
               .astype(object))
    days = {date(d.year, d.month, d.day) for d in days}
    out = []
    for f in sorted(glob.glob(str(Path(directory) / "*.nc"))):
        if days_covered_by_file(Path(f)) & days:
            out.append(f)
    return out


def open_archive(files: list[str]):
    """Open a list of chunk files as one time-ordered lazy dataset."""
    import xarray as xr
    if not files:
        raise FileNotFoundError("no files to open")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="valid_time",
                           data_vars="minimal", coords="minimal",
                           compat="override", join="override",
                           engine="netcdf4")
    t = ds["valid_time"].values
    if np.any(np.diff(t) <= np.timedelta64(0, "ns")):
        ds = ds.sortby("valid_time")
    return ds


def wind_archive_available(E, data_root=None,
                           wind_suffix: str = DEFAULT_WIND_SUFFIX) -> bool:
    if getattr(E.args, "storage", None) == "aws":
        return True          # the bucket holds u and v for every month
    d = archive_dir(E, wind_suffix, data_root)
    return d.is_dir() and bool(files_for_times(d, E.times))


def download_command(E, start: str, end: str, levels: str = "lower") -> str:
    """The downloader invocation that fetches the winds this module needs."""
    return (f"python download_era5_pressure.py --region {E.args.region} "
            f"--storage {E.args.storage} --var-set winds --levels {levels} "
            f"--dir-suffix _wind --start {start} --end {end}")


# ----------------------------------------------------------------------------
# The computation
# ----------------------------------------------------------------------------
DEFAULT_FALLBACK_HPA = 950.0   # the median liquid-weighted pressure, measured


def cloud_level_wind(E, data_root=None, pressure_suffix: str = DEFAULT_PRESSURE_SUFFIX,
                     wind_suffix: str = DEFAULT_WIND_SUFFIX,
                     block_hours: int = 48, require_wind: bool = False,
                     fallback_hpa: float | None = DEFAULT_FALLBACK_HPA) -> dict:
    """Liquid-mass-weighted wind, speed and pressure at every liquid cell-hour.

    Returns a dict of ``(n_t, n_y, n_x)`` float32 arrays aligned with
    ``E.times``, NaN wherever there is no value:

    ``u_cl_m_s, v_cl_m_s``   eq. (2);  ``U_cl_m_s`` eq. (3)   (winds needed)
    ``p_cl_hpa``             eq. (4), from clwc alone
    ``lwp_pl_g``             pressure-level LWP, sum_k w_k, g m-2
    ``liquid_share_unwinded`` share of the column's liquid on levels without
                             wind (above the wind archive's top level)

    plus ``has_pl`` (n_t,) bool -- the hour is in the cloud-water archive --,
    ``has_wind`` (n_t,) bool, the level lists, and counts of the cell-hours
    that were liquid-containing but found no liquid on the levels.

    ``fallback_hpa``: a liquid-containing cell-hour whose column holds no
    liquid on the pressure levels (a thin layer the 23 levels straddle; about
    15% of them) has no weights to average with. Rather than leave the event
    without a wind, the wind at this single level is used -- 950 hPa by
    default, the measured median of p_cl, i.e. where the liquid usually is.
    ``used_fallback`` marks those cell-hours and ``n_fallback`` counts them;
    pass ``None`` to leave them NaN instead.

    Without a wind archive (``require_wind=False``) the wind fields are
    returned all-NaN and the rest is still computed, so the "where is the
    liquid" figures can be drawn before the download exists.
    """
    n_t, n_y, n_x = E.liq.shape
    if getattr(E.args, "storage", None) == "aws":
        # --storage aws: lazy pressure-level datasets from the NCAR bucket over
        # the run's own hours, same variables/levels as the local archives.
        from aws_pipeline import s3_storage

        ds_c, ds_w = s3_storage.pressure_level_datasets(
            E, pressure_suffix, wind_suffix, require_wind)
        # No files behind these: record the S3 sources where the result dict
        # would otherwise list the local chunk files.
        cloud_files = [f"s3://{ds_c.attrs.get('source', 'nsf-ncar-era5')}"]
        wind_files = [f"s3://{ds_w.attrs.get('source', 'nsf-ncar-era5')}"]
    else:
        cloud_dir = archive_dir(E, pressure_suffix, data_root)
        cloud_files = files_for_times(cloud_dir, E.times)
        if not cloud_files:
            raise FileNotFoundError(f"no pressure-level files under {cloud_dir} "
                                    f"overlap the run's hours")
        wind_dir = archive_dir(E, wind_suffix, data_root)
        wind_files = files_for_times(wind_dir, E.times) if wind_dir.is_dir() else []
        if require_wind and not wind_files:
            raise FileNotFoundError(
                f"no wind files under {wind_dir}. Fetch them with:\n  "
                + download_command(E, str(E.times[0].astype('datetime64[D]')),
                                   str(E.times[-1].astype('datetime64[D]'))))

        ds_c = open_archive(cloud_files)[["clwc"]]
        ds_w = open_archive(wind_files)[["u", "v"]] if wind_files else None

    # Hours common to the run and the archives.
    t_run = E.times.astype("datetime64[ns]")
    t_c = ds_c["valid_time"].values.astype("datetime64[ns]")
    common_c = np.intersect1d(t_run, t_c)
    if ds_w is not None:
        t_w = ds_w["valid_time"].values.astype("datetime64[ns]")
        common_w = np.intersect1d(common_c, t_w)
    else:
        common_w = np.empty(0, dtype="datetime64[ns]")
    idx_run = np.searchsorted(t_run, common_c)          # into E arrays
    has_pl = np.zeros(n_t, dtype=bool)
    has_pl[idx_run] = True
    has_wind = np.zeros(n_t, dtype=bool)
    has_wind[np.searchsorted(t_run, common_w)] = True

    # Levels: cloud-water levels sorted ascending in pressure; wind levels are
    # a subset (the lower troposphere).
    p_c = ds_c["pressure_level"].values.astype(float)
    order_c = np.argsort(p_c)
    p_c_s = p_c[order_c]                                 # hPa ascending
    if ds_w is not None:
        p_w = ds_w["pressure_level"].values.astype(float)
        order_w = np.argsort(p_w)
        p_w_s = p_w[order_w]
        # position of each wind level inside the cloud-level list
        pos_w = np.searchsorted(p_c_s, p_w_s)
        if not np.allclose(p_c_s[pos_w], p_w_s):
            raise ValueError("wind levels are not a subset of the cloud-water levels")
    else:
        p_w_s, pos_w = np.empty(0), np.empty(0, dtype=int)

    sp_all = E.ds["sp"]                                  # lazy single-level sp, Pa
    out = {k: np.full((n_t, n_y, n_x), np.nan, dtype=np.float32)
           for k in ("u_cl_m_s", "v_cl_m_s", "U_cl_m_s", "p_cl_hpa",
                     "lwp_pl_g", "liquid_share_unwinded")}
    used_fallback = np.zeros((n_t, n_y, n_x), dtype=bool)
    k_fb = None
    if ds_w is not None and fallback_hpa is not None:
        k_fb = int(np.argmin(np.abs(p_w_s - fallback_hpa)))
        if not np.isclose(p_w_s[k_fb], fallback_hpa):
            raise ValueError(f"fallback level {fallback_hpa} hPa is not a wind "
                             f"level; have {p_w_s}")
    n_liq_hours = n_no_liquid_on_levels = n_fallback = 0
    # Liquid mass on the levels, all and above the wind archive's top level
    # (500 hPa when no winds are present), to state what the wind levels miss.
    top_wind_hpa = float(p_w_s.min()) if p_w_s.size else 500.0
    mass_total = mass_above = 0.0
    t0 = time.time()
    for b0 in range(0, common_c.size, block_hours):
        tb = common_c[b0:b0 + block_hours]
        ib = idx_run[b0:b0 + block_hours]
        liq = E.liq[ib]                                   # (t, y, x)
        if not liq.any():
            continue
        clwc = (ds_c["clwc"].sel(valid_time=tb).values[:, order_c]
                .astype(np.float64))                      # (t, lev, y, x)
        sp = sp_all.sel(valid_time=tb).values.astype(np.float64)     # (t, y, x)
        # layer_thickness_pa wants the level axis LAST and pressure in Pa.
        dp = np.moveaxis(layer_thickness_pa(p_c_s * 100.0, sp), -1, 1)  # (t, lev, y, x)
        w = clwc * dp / G_M_S2                            # eq. (1), kg m-2
        w[:, :, ~liq.any(axis=0)] = 0.0                   # only where needed
        w = np.where(liq[:, None], w, 0.0)
        w_tot = w.sum(axis=1)                             # (t, y, x)
        mass_total += float(w_tot.sum())
        mass_above += float(w[:, p_c_s < top_wind_hpa].sum())
        n_liq_hours += int(liq.sum())
        n_no_liquid_on_levels += int((liq & (w_tot <= 0)).sum())
        ok = w_tot > 0
        den = np.where(ok, w_tot, 1.0)
        out["lwp_pl_g"][ib] = np.where(ok, w_tot * 1000.0, np.nan)
        out["p_cl_hpa"][ib] = np.where(
            ok, (w * p_c_s[None, :, None, None]).sum(axis=1) / den, np.nan)

        if ds_w is not None:
            in_w = np.isin(tb, common_w)
            if not in_w.any():
                continue
            tbw, ibw = tb[in_w], ib[in_w]
            uw = ds_w["u"].sel(valid_time=tbw).values[:, order_w].astype(np.float64)
            vw = ds_w["v"].sel(valid_time=tbw).values[:, order_w].astype(np.float64)
            ww = w[in_w][:, pos_w]                        # weights on wind levels
            ww_tot = ww.sum(axis=1)
            okw = ww_tot > 0
            denw = np.where(okw, ww_tot, 1.0)
            u_cl = (ww * uw).sum(axis=1) / denw
            v_cl = (ww * vw).sum(axis=1) / denw
            U_cl = (ww * np.hypot(uw, vw)).sum(axis=1) / denw
            if k_fb is not None:
                # No liquid on the levels but liquid-containing by the
                # single-level filter: take the wind at the fallback level.
                fb = liq[in_w] & ~okw
                u_cl = np.where(fb, uw[:, k_fb], u_cl)
                v_cl = np.where(fb, vw[:, k_fb], v_cl)
                U_cl = np.where(fb, np.hypot(uw[:, k_fb], vw[:, k_fb]), U_cl)
                okw = okw | fb
                used_fallback[ibw] = fb
                n_fallback += int(fb.sum())
            out["u_cl_m_s"][ibw] = np.where(okw, u_cl, np.nan)
            out["v_cl_m_s"][ibw] = np.where(okw, v_cl, np.nan)
            out["U_cl_m_s"][ibw] = np.where(okw, U_cl, np.nan)
            wt_all = w_tot[in_w]
            out["liquid_share_unwinded"][ibw] = np.where(
                wt_all > 0, 1.0 - ww_tot / np.where(wt_all > 0, wt_all, 1.0), np.nan)
    print(f"  cloud-level wind: {int(has_pl.sum()):,} hours with cloud water on "
          f"levels, {int(has_wind.sum()):,} with winds   |   "
          f"{n_no_liquid_on_levels:,} of {n_liq_hours:,} liquid cell-hours "
          f"({100 * n_no_liquid_on_levels / max(n_liq_hours, 1):.1f}%) hold no "
          f"liquid on the levels"
          + (f", {n_fallback:,} of those given the {fallback_hpa:g} hPa wind"
             if k_fb is not None else "")
          + f"   |   {time.time() - t0:.0f} s")
    out.update({
        "used_fallback": used_fallback, "n_fallback": n_fallback,
        "top_wind_hpa": top_wind_hpa,
        "liquid_mass_share_above_wind_levels": (mass_above / mass_total
                                                if mass_total > 0 else np.nan),
        "fallback_hpa": fallback_hpa if k_fb is not None else None,
        "has_pl": has_pl, "has_wind": has_wind,
        "levels_cloud_hpa": p_c_s, "levels_wind_hpa": p_w_s,
        "n_liq_cell_hours": n_liq_hours,
        "n_no_liquid_on_levels": n_no_liquid_on_levels,
        "cloud_files": cloud_files, "wind_files": wind_files,
    })
    return out


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def print_cloud_wind_report(E, CL: dict) -> None:
    liq = E.liq
    p = CL["p_cl_hpa"][liq]
    p = p[np.isfinite(p)]
    lwp_pl = CL["lwp_pl_g"][liq]
    print(f"Cloud level, liquid-mass weighted   |   {precip_label(E.args)}")
    print(f"  hours in the cloud-water archive: {int(CL['has_pl'].sum()):,} of "
          f"{E.times.size:,};  with winds: {int(CL['has_wind'].sum()):,}")
    print(f"  liquid-weighted pressure at liquid cell-hours: median "
          f"{np.median(p):.0f} hPa, IQR {_q(p, 25):.0f}-{_q(p, 75):.0f}, "
          f"p10 {_q(p, 10):.0f}, p90 {_q(p, 90):.0f}")
    print(f"  share of liquid cell-hours with p_cl > 850 hPa: "
          f"{100 * np.mean(p > 850):.0f}%   > 700 hPa: {100 * np.mean(p > 700):.0f}%")
    print(f"  liquid mass above {CL['top_wind_hpa']:.0f} hPa (the top wind level): "
          f"{100 * CL['liquid_mass_share_above_wind_levels']:.3f}% of the total on "
          f"the levels")
    print(f"  liquid cell-hours with no liquid on the levels: "
          f"{CL['n_no_liquid_on_levels']:,} of {CL['n_liq_cell_hours']:,} "
          f"({100 * CL['n_no_liquid_on_levels'] / max(CL['n_liq_cell_hours'], 1):.1f}%)"
          + (f"; {CL['n_fallback']:,} given the {CL['fallback_hpa']:g} hPa wind"
             if CL["fallback_hpa"] is not None else ""))
    if CL["has_wind"].any():
        sel = liq & np.isfinite(CL["U_cl_m_s"])
        U10 = np.hypot(E.u10, E.v10)[sel]
        Ucl = CL["U_cl_m_s"][sel]
        r = Ucl / np.where(U10 > 0, U10, np.nan)
        r = r[np.isfinite(r)]
        print(f"  {int(sel.sum()):,} liquid cell-hours with a cloud-level wind")
        print(f"  U_cl: median {np.median(Ucl):.1f} m/s, IQR {_q(Ucl, 25):.1f}-"
              f"{_q(Ucl, 75):.1f}   |   U_10 same hours: median {np.median(U10):.1f}")
        print(f"  U_cl / U_10 per cell-hour: median {np.median(r):.2f}, IQR "
              f"{_q(r, 25):.2f}-{_q(r, 75):.2f}")
        su = CL["liquid_share_unwinded"][sel]
        print(f"  liquid above the wind levels: mean share "
              f"{100 * np.nanmean(su):.2f}% of the column's liquid")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_liquid_level(E, CL: dict, out_dir=None, dpi=None):
    """Where the liquid is: the liquid-weighted pressure at liquid cell-hours."""
    import matplotlib.pyplot as plt

    liq = E.liq
    iy, ix = E.site_iy, E.site_ix
    pops = [("domain", CL["p_cl_hpa"][liq], DOMAIN_COLOR, 2.2),
            ("ARM cell", CL["p_cl_hpa"][:, iy, ix][liq[:, iy, ix]], SITE_COLOR, 1.6)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bins = np.arange(500, 1012.5, 12.5)
    for lab, a, c, lw in pops:
        a = a[np.isfinite(a)]
        ax.hist(a, bins=bins, density=True, histtype="step", color=c, lw=lw,
                orientation="horizontal",
                label=f"{lab}: {a.size:,} cell-hours, median {np.median(a):.0f} hPa, "
                      f"IQR {_q(a, 25):.0f}–{_q(a, 75):.0f}")
    ax.invert_yaxis()
    ax.set_ylabel("liquid-mass-weighted pressure of the column $p_{cl}$ [hPa]")
    ax.set_xlabel("probability density [hPa$^{-1}$]")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False)
    _threshold_box(ax, E, loc="upper right")
    _tidy(ax)
    ax.set_title(f"{precip_banner(E.args)}\nWhere the liquid sits at liquid-containing "
                 f"cell-hours, {E.args.region} region, {_season_span(E)}",
                 fontsize=10.5)
    return _save(fig, E, out_dir, "cloud_level_pressure", dpi)


def fig_cloud_wind_vs_10m(E, CL: dict, out_dir=None, dpi=None):
    """(a) speed PDFs at liquid cell-hours, (b) the per-cell-hour ratio."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    sel = E.liq & np.isfinite(CL["U_cl_m_s"])
    U10 = np.hypot(E.u10, E.v10)[sel].astype(float)
    Ucl = CL["U_cl_m_s"][sel].astype(float)
    Uvec = np.hypot(CL["u_cl_m_s"], CL["v_cl_m_s"])[sel].astype(float)
    iy, ix = E.site_iy, E.site_ix
    s_site = sel[:, iy, ix]
    U10_s = np.hypot(E.u10, E.v10)[:, iy, ix][s_site]
    Ucl_s = CL["U_cl_m_s"][:, iy, ix][s_site]

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    bins = np.arange(0, 30.5, 0.5)
    for lab, a, c, ls, lw in (
            ("10 m, domain", U10, DOMAIN_COLOR, "--", 1.4),
            ("cloud level, domain (mean speed)", Ucl, CLOUD_WIND_COLOR, "-", 2.2),
            ("cloud level, domain (|vector mean|)", Uvec, CLOUD_WIND_COLOR, ":", 1.4),
            ("10 m, ARM cell", U10_s, SITE_COLOR, "--", 1.2),
            ("cloud level, ARM cell", Ucl_s, SITE_COLOR, "-", 1.8)):
        if a.size == 0:
            continue
        ax_p.hist(a, bins=bins, density=True, histtype="step", color=c, ls=ls, lw=lw,
                  label=f"{lab}: median {np.median(a):.1f} m s$^{{-1}}$")
    ax_p.set_xlim(0, 25)
    ax_p.set_xlabel("wind speed at liquid-containing cell-hours [m s$^{-1}$]")
    ax_p.set_ylabel("probability density [(m s$^{-1}$)$^{-1}$]")
    ax_p.xaxis.set_major_locator(MultipleLocator(5))
    ax_p.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=2, frameon=False)
    ax_p.set_title("(a)  10 m wind against the liquid-weighted cloud-level wind",
                   loc="left", fontsize=10.5)
    _tidy(ax_p)

    r = Ucl / np.where(U10 > 0.1, U10, np.nan)
    r = r[np.isfinite(r)]
    ax_r.hist(r, bins=np.geomspace(0.2, 10, 61), density=True, histtype="step",
              color=CLOUD_WIND_COLOR, lw=2.2,
              label=f"domain: median {np.median(r):.2f}, IQR {_q(r, 25):.2f}–{_q(r, 75):.2f}")
    rs = Ucl_s / np.where(U10_s > 0.1, U10_s, np.nan)
    rs = rs[np.isfinite(rs)]
    if rs.size:
        ax_r.hist(rs, bins=np.geomspace(0.2, 10, 61), density=True, histtype="step",
                  color=SITE_COLOR, lw=1.6,
                  label=f"ARM cell: median {np.median(rs):.2f}, IQR {_q(rs, 25):.2f}–{_q(rs, 75):.2f}")
    ax_r.axvline(1.0, color="0.4", lw=0.9, ls=":")
    ax_r.set_xscale("log")
    ax_r.set_xlabel("$U_{cl}\\,/\\,U_{10}$ per liquid-containing cell-hour")
    ax_r.set_ylabel("probability density")
    ax_r.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                frameon=False)
    ax_r.set_title("(b)  the ratio Taylor's L scales by", loc="left", fontsize=10.5)
    _tidy(ax_r)
    fig.suptitle(f"{precip_banner(E.args)}\nCloud-level wind from the pressure-level "
                 f"archive, {E.args.region} region, {_season_span(E)}   |   "
                 f"{int(sel.sum()):,} liquid cell-hours", fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, E, out_dir, "cloud_wind_vs_10m", dpi)


def fig_ratio_map(E, CL: dict, out_dir=None, dpi=350, height: float = 6.0,
                  legend_fontsize=None, tick_fontsize=None):
    """Per cell: median U_cl / U_10 over its liquid cell-hours, and median p_cl."""
    import matplotlib.pyplot as plt
    from cloud_spatial_extent import (_draw_one, _fonts, _projection,
                                      _site_legend_below, _subtitle,
                                      projected_aspect)

    legend_fontsize, tick_fontsize = _fonts(E, legend_fontsize, tick_fontsize)
    U10 = np.hypot(E.u10, E.v10)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(E.liq & np.isfinite(CL["U_cl_m_s"]) & (U10 > 0.1),
                         CL["U_cl_m_s"] / U10, np.nan)
        p_cl = np.where(E.liq, CL["p_cl_hpa"], np.nan)
    panels = [
        (np.nanmedian(ratio, axis=0), "median $U_{cl}/U_{10}$ at liquid cell-hours",
         "Greens"),
        (np.nanmedian(p_cl, axis=0), "median liquid-weighted pressure $p_{cl}$ [hPa]",
         "Purples_r"),
    ]
    aspect = projected_aspect(E)
    fig, axes = plt.subplots(1, 2, figsize=(2 * height * aspect * 1.5, height),
                             subplot_kw={"projection": _projection(E)})
    for ax, (field, lab, cmap) in zip(axes, panels):
        vmin, vmax = np.nanpercentile(field, [1, 99])
        mesh = _draw_one(ax, E, field, vmin, vmax, cmap, tick_fontsize=tick_fontsize)
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05,
                          pad=0.06, aspect=28)
        cb.set_label(lab, fontsize=10)
        cb.ax.tick_params(labelsize=tick_fontsize)
        ax.set_title(f"domain median {np.nanmedian(field):.2f}   |   ARM cell "
                     f"{field[E.site_iy, E.site_ix]:.2f}", fontsize=10)
    _site_legend_below(fig, legend_fontsize)
    fig.suptitle(f"{precip_banner(E.args)}\nCloud-level wind against the 10 m wind, "
                 f"per cell   |   {E.args.region} region, {_season_span(E)}\n"
                 f"{_subtitle(E)}", fontsize=11, y=1.02)
    return _save(fig, E, out_dir, "cloud_wind_ratio_maps", dpi)


def fig_taylor_cloud_wind(E, ev10: dict, evcl: dict, out_dir=None, dpi=None):
    """Taylor lengths with the 10 m wind and with the cloud-level wind.

    (a) the two L distributions over the same events; (b) one point per
    event, L_cl against L_10 -- the ratio is the event-mean U_cl / U_10.
    """
    import matplotlib.pyplot as plt

    ok = np.isfinite(evcl["L_km"])
    L10, Lcl = ev10["L_km"][ok], evcl["L_km"][ok]
    site = ev10["is_site"][ok]
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    bins = np.geomspace(1.0, 1e4, 81)
    for lab, a, c, lw in (("10 m wind, all cells", L10, DOMAIN_COLOR, 2.0),
                          ("cloud-level wind, all cells", Lcl, CLOUD_WIND_COLOR, 2.2),
                          ("10 m wind, ARM cell", L10[site], SITE_COLOR, 1.2),
                          ("cloud-level wind, ARM cell", Lcl[site], SITE_COLOR, 1.8)):
        ls = "--" if lab.startswith("10 m") else "-"
        if a.size == 0:
            continue
        ax_h.hist(a, bins=bins, density=True, histtype="step", color=c, lw=lw, ls=ls,
                  label=f"{lab}: {a.size:,} events, median {np.median(a):,.0f} km, "
                        f"IQR {_q(a, 25):,.0f}–{_q(a, 75):,.0f} km")
    _cell_reference_lines(ax_h, E)
    ax_h.set_xscale("log")
    ax_h.set_xlim(bins[0], bins[-1])
    ax_h.set_xlabel("Taylor horizontal extent $L = U\\,\\Delta t$ [km]")
    ax_h.set_ylabel("probability density per event")
    ax_h.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=1, frameon=False)
    _threshold_box(ax_h, E, loc="upper right")
    ax_h.set_title("(a)  the same events, two winds", loc="left", fontsize=10.5)
    _tidy(ax_h)

    r = Lcl / L10
    ax_s.hexbin(L10, Lcl, gridsize=50, xscale="log", yscale="log", bins="log",
                cmap="Greys", mincnt=1, linewidths=0.2)
    lim = (2.0, max(L10.max(), Lcl.max()) * 1.3)
    ax_s.plot(lim, lim, color="0.35", lw=1.0, ls="--", label="1:1")
    ax_s.plot(lim, [lim[0] * np.median(r), lim[1] * np.median(r)], color=CLOUD_WIND_COLOR,
              lw=1.4, label=f"median ratio {np.median(r):.2f}")
    ax_s.set_xlim(lim)
    ax_s.set_ylim(lim)
    ax_s.set_xlabel("$L$ with the 10 m wind [km]")
    ax_s.set_ylabel("$L$ with the cloud-level wind [km]")
    ax_s.text(0.03, 0.97, f"{L10.size:,} events\nmedian $L_{{cl}}/L_{{10}}$ = "
              f"{np.median(r):.2f}, IQR {_q(r, 25):.2f}–{_q(r, 75):.2f}",
              transform=ax_s.transAxes, ha="left", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.8))
    ax_s.legend(fontsize=8.5, loc="lower right")
    ax_s.set_title("(b)  event by event", loc="left", fontsize=10.5)
    _tidy(ax_s)
    fig.suptitle(f"{precip_banner(E.args)}\nTaylor extent with the cloud-level wind, "
                 f"ERA5, {E.args.region} region, {_season_span(E)}", fontsize=11.5,
                 y=1.0)
    fig.tight_layout()
    return _save(fig, E, out_dir, "taylor_cloud_wind", dpi)


def fig_taylor_monthly_box_two_winds(E, ev10: dict, evcl: dict, out_dir=None, dpi=None):
    """Monthly boxes of L, 10 m wind beside cloud-level wind, all cells."""
    import calendar
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ok = np.isfinite(evcl["L_km"])
    months = E.months
    a = [ev10["L_km"][ok & (ev10["month"] == m)] for m in months]
    b = [evcl["L_km"][ok & (evcl["month"] == m)] for m in months]
    x = np.arange(len(months))
    w = 0.34
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(months), 6.2))
    for cols, off, color, ls in ((a, -w / 2 - 0.03, DOMAIN_COLOR, ":"),
                                 (b, w / 2 + 0.03, CLOUD_WIND_COLOR, "-")):
        keep = [j for j, c in enumerate(cols) if c.size]
        bp = ax.boxplot([cols[j] for j in keep], positions=x[keep] + off, widths=w,
                        whis=(0, 100), showfliers=False, patch_artist=True,
                        manage_ticks=False)
        for key in ("boxes", "whiskers", "caps", "medians"):
            for art in bp[key]:
                art.set_color("black" if key == "medians" else color)
                art.set_linestyle(ls)
                art.set_linewidth(1.7 if key == "medians" else 1.5)
                if key == "boxes":
                    art.set_facecolor("white")
                    art.set_edgecolor(color)
        for j in keep:
            ax.text(x[j] + off, np.median(cols[j]), f" {np.median(cols[j]):,.0f}",
                    ha="left", va="bottom", fontsize=7.5, color="0.25")
    ax.set_yscale("log")
    ax.set_ylim(1.0, max(c.max() for c in b if c.size) * 8.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{calendar.month_abbr[m]}\n{a[j].size:,} events"
                        for j, m in enumerate(months)], fontsize=9)
    ax.set_ylabel("Taylor extent of a liquid-containing event, $L = U\\,\\Delta t$ [km]",
                  fontsize=10.5)
    handles = [cse.plt_rect(DOMAIN_COLOR, ":"), cse.plt_rect(CLOUD_WIND_COLOR, "-"),
               Line2D([0], [0], color="black", lw=1.7)]
    ax.legend(handles, ["10 m wind", "liquid-weighted cloud-level wind",
                        "median [km], value at right"],
              title="whiskers: min to max, no outliers drawn   |   the same events, "
                    "all cells, assigned to their starting month",
              title_fontsize=8, fontsize=9, ncol=3, framealpha=0.9, loc="upper right")
    _threshold_box(ax, E, loc="upper left")
    _tidy(ax)
    fig.suptitle(f"{precip_banner(E.args)}\nHorizontal extent of liquid-containing "
                 f"cloud events by month, two advection winds, ERA5, "
                 f"{E.args.region} region, {_season_span(E)}", fontsize=11.5, y=0.97)
    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.08, right=0.985)
    return _save(fig, E, out_dir, "taylor_monthly_box_two_winds", dpi)


# ----------------------------------------------------------------------------
# Slide figure: cloud duration and in-cloud wind speed at the ARM cell
# ----------------------------------------------------------------------------
DURATION_COLOR_OV = SITE_COLOR       # the duration histogram and its axis
WIND_COLOR_OV = CLOUD_WIND_COLOR     # the wind histogram and its axis


def resolve_duration_mode(duration_mode, args=None) -> str:
    """Pick what the duration histogram counts, falling back to ``--duration-mode``."""
    if duration_mode is None:
        duration_mode = getattr(args, "duration_mode", DEFAULT_DURATION_MODE)
    if duration_mode not in DURATION_MODES:
        raise ValueError(f"duration_mode must be one of {DURATION_MODES}, got "
                         f"{duration_mode!r}")
    return duration_mode


def resolve_wind_source(wind_source, args=None) -> str:
    """Pick the wind for the slide figure, falling back to ``--wind-source``."""
    if wind_source is None:
        wind_source = getattr(args, "wind_source", DEFAULT_WIND_SOURCE)
    if wind_source not in WIND_SOURCES:
        raise ValueError(f"wind_source must be one of {WIND_SOURCES}, got "
                         f"{wind_source!r}")
    return wind_source


def site_wind_at_liquid_hours(E, CL: dict | None, wind_source: str) -> dict:
    """The wind speed at every liquid-containing hour of the ARM cell.

    ``'10m'``: the single-level 10 m wind held in ``E`` -- every season of
    the run. ``'cloud'``: the liquid-mass-weighted cloud-level wind from
    :func:`cloud_level_wind`, which exists only for the seasons the
    pressure-level archive covers; ``E`` must then be the run ``CL`` was
    computed on. Returns ``{"speed_m_s": (n,), "month": (n,) calendar month
    of each sample, "n_hours": int, "seasons": str, "label": str}``.
    """
    iy, ix = E.site_iy, E.site_ix
    liq = E.liq[:, iy, ix]
    month_of_hour = np.asarray(E.months)[E.mi]
    if wind_source == "10m":
        speed = np.hypot(E.u10, E.v10)[:, iy, ix][liq].astype(float)
        month = month_of_hour[liq]
        label = "10 m wind speed at liquid-containing hours"
    else:
        if CL is None:
            raise ValueError(
                "wind_source='cloud' needs CL from cloud_level_wind.cloud_level_wind(E) "
                "on a run the pressure-level archive covers; pass CL= (and E_pl=), "
                "or use wind_source='10m'")
        U = CL["U_cl_m_s"][:, iy, ix]
        if U.shape[0] != liq.shape[0]:
            raise ValueError("CL is not aligned with E: compute both on the same run")
        ok = liq & np.isfinite(U)
        if not ok.any():
            raise ValueError("no cloud-level wind at the ARM cell: is the wind "
                             "archive present? See cloud_level_wind.download_command")
        speed = U[ok].astype(float)
        month = month_of_hour[ok]
        label = "cloud-level wind speed at liquid-containing hours"
    return {"speed_m_s": speed, "month": month, "n_hours": int(speed.size),
            "seasons": _season_span(E), "label": label,
            "source": wind_source}


def fig_cloudDuration_andWind_forOV(E, ev: dict, E_pl=None, CL: dict | None = None,
                                    wind_source: str | None = None,
                                    duration_mode: str | None = None,
                                    out_dir=None, dpi=None,
                                    dur_xmax_h: float = 48.0, dur_bin_h: float = 1.0,
                                    wind_xmax_m_s: float = 25.0, wind_bin_m_s: float = 1.0,
                                    threshold_box: bool = True,
                                    fill_alpha: float = 0.45):
    """Cloud duration and in-cloud wind speed at the ARM cell, one panel.

    Two histograms on one set of axes, each with ITS OWN x axis along the
    bottom, coloured to match: the liquid-containing EVENT duration in ERA5
    (dark blue, upper axis of the two) and the wind speed at the cell's
    liquid-containing HOURS (teal, the lower, offset axis). The y axis is
    the share of samples in each bin, in percent, so the two distributions
    are on a common footing even though one counts events and the other
    hours. Dashed lines mark the two medians.

    ``duration_mode`` chooses what the dark-blue histogram counts (``None``
    follows the run's ``--duration-mode``):

    ``'event'``  one sample per liquid-containing EVENT at the cell -- a
                 maximal run of consecutive liquid-containing hours -- with
                 its duration in hours, from ``ev``
                 (:func:`cloud_spatial_extent.taylor_events` on ``E``; only
                 its ARM-cell events are used, censored ones included, since
                 a run that touches a break is a lower bound on a real event
                 and dropping it would remove the long clouds
                 preferentially). Axis 0 to ``dur_xmax_h``.
    ``'daily'``  one sample per calendar DAY with its liquid-containing
                 hours, 0-24, days with none included at 0
                 (:func:`cloud_spatial_extent.site_daily_liquid_hours`).
                 This is the distribution behind the hours-per-day table;
                 its mean is the table's pooled ``h / day``. Axis 0-24 h
                 regardless of ``dur_xmax_h``.

    ``wind_source`` chooses the wind (``None`` follows the run's
    ``--wind-source``):

    ``'cloud'``  the liquid-mass-weighted wind at cloud level from the
                 pressure-level archive (:func:`cloud_level_wind`), which
                 covers 2024/25-2025/26 only. Pass ``E_pl`` (the run on
                 those seasons) and ``CL`` (computed on it). The duration
                 histogram still comes from ``E``, so the two histograms then
                 span DIFFERENT seasons; the legend says which.
    ``'10m'``    the single-level 10 m wind in ``E`` itself -- every season,
                 the same population as the durations, and a lower bound on
                 the wind at cloud level under a stable Arctic boundary layer.

    Precipitation filter, overcast gate and phase thresholds are those of
    ``E``; the threshold box states them. Saved under
    ``cloudDuration_andWind_OV_<mode>_<source>``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    duration_mode = resolve_duration_mode(duration_mode, E.args)
    wind_source = resolve_wind_source(wind_source, E.args)
    if wind_source == "cloud":
        E_w = E_pl if E_pl is not None else E
        wind = site_wind_at_liquid_hours(E_w, CL, "cloud")
    else:
        wind = site_wind_at_liquid_hours(E, None, "10m")

    if duration_mode == "event":
        sel = ev["is_site"]
        dur_h = ev["length_h"][sel].astype(float)
        if dur_h.size == 0:
            raise ValueError("no liquid-containing events at the ARM cell")
        dur_unit, dur_what = "events", "cloud duration"
        dur_axis_label = "Liquid-containing cloud duration  [hours]"
    else:
        daily = cse.site_daily_liquid_hours(E)
        dur_h = daily["hours"]
        dur_xmax_h = 24.0                      # a day is the whole axis
        dur_unit, dur_what = "days", "hours per day"
        dur_axis_label = "Liquid-containing hours per day  [hours]"
    speed = wind["speed_m_s"]

    dur_med, spd_med = float(np.median(dur_h)), float(np.median(speed))
    dur_beyond = 100.0 * np.mean(dur_h > dur_xmax_h)
    spd_beyond = 100.0 * np.mean(speed > wind_xmax_m_s)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax_w = ax.twiny()

    # --- duration: integer hours, so bins are centred on each value. Event
    # durations start at 1 h; daily hours start at 0, so that mode gets a
    # bin centred on zero for the liquid-free days. -------------------------
    dur_lo = 0.5 * dur_bin_h if duration_mode == "event" else -0.5 * dur_bin_h
    dur_bins = np.arange(dur_lo, dur_xmax_h + dur_bin_h, dur_bin_h)
    dur_note = (f", {dur_beyond:.0f}% beyond {dur_xmax_h:g} h" if dur_beyond >= 0.5
                else "")
    if duration_mode == "daily":
        dur_note = (f", mean {dur_h.mean():.1f} h, "
                    f"{100 * np.mean(dur_h == 0):.0f}% of days with none")
    ax.hist(dur_h, bins=dur_bins, weights=np.full(dur_h.size, 100.0 / dur_h.size),
            histtype="stepfilled", color=DURATION_COLOR_OV, alpha=fill_alpha,
            edgecolor=DURATION_COLOR_OV, lw=1.4, zorder=3,
            label=f"ERA5 liquid-containing {dur_what}, {_season_span(E)}: "
                  f"{dur_h.size:,} {dur_unit}, median {dur_med:.0f} h{dur_note}")
    ax.axvline(dur_med, color=DURATION_COLOR_OV, ls=(0, (5, 3)), lw=1.6, zorder=4)
    # Each median line is labelled with its value AND unit, in its own colour:
    # both x axes span the full width, so a bare line has two readings and
    # only the colour says which axis it belongs to.
    _med_txt = dict(xycoords=("data", "axes fraction"), textcoords="offset points",
                    va="top", fontsize=10.5, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    # Side of its own median line depends on wind_source. With the cloud-level
    # wind, the duration label reads fine pushed right, so it stays to the
    # left of its own line to clear the wind median (see below). With the
    # 10 m wind, whose axis (0-25 m/s) spans nearly the same pixel width as
    # the duration axis, the two labels are swapped instead so they still
    # clear each other.
    dur_ha, dur_dx = ("left", 4) if wind_source == "10m" else ("right", -4)
    ax.annotate(f"median {dur_med:.0f} h", xy=(dur_med, 0.985), xytext=(dur_dx, 0),
                color=DURATION_COLOR_OV, ha=dur_ha, **_med_txt)

    # --- wind: continuous, so plain bins from zero -------------------------
    wind_bins = np.arange(0.0, wind_xmax_m_s + wind_bin_m_s, wind_bin_m_s)
    ax_w.hist(speed, bins=wind_bins, weights=np.full(speed.size, 100.0 / speed.size),
              histtype="stepfilled", color=WIND_COLOR_OV, alpha=fill_alpha,
              edgecolor=WIND_COLOR_OV, lw=1.4, zorder=3,
              label=f"ERA5 {wind['label']}, {wind['seasons']}: "
                    f"{speed.size:,} hours, median {spd_med:.1f} m s$^{{-1}}$"
                    + (f", {spd_beyond:.0f}% beyond {wind_xmax_m_s:g} m s$^{{-1}}$"
                       if spd_beyond >= 0.5 else ""))
    ax_w.axvline(spd_med, color=WIND_COLOR_OV, ls=(0, (5, 3)), lw=1.6, zorder=4)
    wind_ha, wind_dx = ("right", -4) if wind_source == "10m" else ("left", 4)
    ax_w.annotate(f"median {spd_med:.1f} m s$^{{-1}}$", xy=(spd_med, 0.925), xytext=(wind_dx, 0),
                  color=WIND_COLOR_OV, ha=wind_ha, **_med_txt)

    # --- two x axes along the bottom, each in its histogram's colour --------
    # The daily axis starts half a bin left of zero so the zero-day bar is
    # not cut in half by the spine.
    ax.set_xlim(dur_bins[0] if duration_mode == "daily" else 0.0, dur_xmax_h)
    ax.xaxis.set_major_locator(MultipleLocator(6 if dur_xmax_h >= 36 else 3))
    ax.set_xlabel(dur_axis_label, color=DURATION_COLOR_OV, fontsize=11)
    ax.tick_params(axis="x", colors=DURATION_COLOR_OV, labelsize=10)
    ax.spines["bottom"].set_color(DURATION_COLOR_OV)
    ax.spines["bottom"].set_linewidth(1.6)

    ax_w.set_xlim(0, wind_xmax_m_s)
    ax_w.xaxis.set_ticks_position("bottom")
    ax_w.xaxis.set_label_position("bottom")
    ax_w.spines["bottom"].set_position(("outward", 52))
    ax_w.spines["bottom"].set_color(WIND_COLOR_OV)
    ax_w.spines["bottom"].set_linewidth(1.6)
    ax_w.spines["top"].set_visible(False)
    for sp in ("left", "right"):
        ax_w.spines[sp].set_visible(False)
    ax_w.xaxis.set_major_locator(MultipleLocator(5))
    ax_w.set_xlabel(("Cloud-level" if wind_source == "cloud" else "10 m")
                    + " wind speed at liquid-containing hours  [m s$^{-1}$]",
                    color=WIND_COLOR_OV, fontsize=11)
    ax_w.tick_params(axis="x", colors=WIND_COLOR_OV, labelsize=10)

    ax.set_ylabel("Share of samples in bin  [%]", fontsize=11)
    # Headroom above the tallest bar for the threshold box; the y axis is
    # shared by the twin, so one limit governs both histograms.
    top = 100.0 * max(np.histogram(dur_h, bins=dur_bins)[0].max() / dur_h.size,
                      np.histogram(speed, bins=wind_bins)[0].max() / speed.size)
    ax.set_ylim(0, 1.15 * top)
    _tidy(ax)
    ax.grid(False, axis="x")

    # Legend mid-height on the right, where neither histogram reaches; the
    # threshold box below it, top edge pinned to the 30% gridline.
    # h1, l1 = ax.get_legend_handles_labels()
    # h2, l2 = ax_w.get_legend_handles_labels()
    # ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.92)
    if threshold_box:
        # x in axes fraction (right edge), y in data units (% share of
        # samples) so the box's top edge sits exactly on the 30% line
        # regardless of how the histogram's own headroom (`top`, above)
        # comes out for a given run -- unless 30% is above the axes, as it
        # is for the daily histogram (tallest bar ~18%), in which case the
        # box drops to just under the top of the axes rather than floating
        # off the panel.
        y_box = min(30.0, 0.97 * ax.get_ylim()[1])
        ax.text(0.995, y_box, "\n".join(threshold_box_lines(E)),
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=8.5, linespacing=1.4, zorder=6,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="0.55", linewidth=0.8, alpha=0.94))

    ax.set_title(f"{precip_banner(E.args)}\nUtqia\u0121vik grid cell "
                 f"({E.site_lat:.2f} N, {E.site_lon:.2f} E), {E.args.region} region",
                 fontsize=10.5)
    fig.subplots_adjust(bottom=0.26)
    return _save(fig, E, out_dir,
                 f"cloudDuration_andWind_OV_{duration_mode}_{wind_source}", dpi)
