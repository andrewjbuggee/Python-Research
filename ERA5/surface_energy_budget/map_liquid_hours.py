#!/usr/bin/env python3
"""Spatial maps of liquid-containing cloud hours over the Barrow region.

WHAT THIS COMPUTES
==================
The single-cell analysis in ``comparison_with_Genie_Obs.ipynb`` collapses the
domain to the one grid cell holding the ARM facility. This module keeps the
(latitude, longitude) axes instead, so every cell gets its own answer, using the
IDENTICAL definitions: the same ``tcc`` gate, the same fractional LWP/IWP
boundaries, the same minimum water paths, the same season window.

Two quantities per cell, per season:

    hours     n_liq / n_valid * season_hours[s]        [h]
    fraction  100 * n_liq / n_valid                    [% of the season]

They carry the same information -- ``hours = fraction/100 * season_hours`` --
but the season length differs between leap and common years, so the two maps are
not simply a rescaling of one another across seasons.

WHY THE NORMALISATION IS A RATIO AND NOT A RAW COUNT
====================================================
Dividing by the cell's own valid hours before scaling to the nominal window
means a season with missing files is scaled UP rather than counted short, which
is what the rest of this project does (see ``to_hours_per_season``). It also
makes the invariant below hold exactly per cell rather than approximately:

    liquid-containing fraction  <=  cloudy fraction

because liquid-containing is a subset of cloudy by construction and both are
divided by the same denominator. :func:`check_invariants` asserts it.

AREA WEIGHTING DOES NOT APPEAR HERE, DELIBERATELY
=================================================
Every number is per cell, so no cos(latitude) weighting is involved: that
weighting exists to average correctly OVER cells, and nothing here averages over
cells. It would be a bug to apply it. The one place it would matter -- a
domain-mean figure -- is not drawn.
"""

from __future__ import annotations

import numpy as np

import plot_lwp_histogram_by_surface_class as lwph
from plot_lwp_histogram_by_surface_class import (
    LIQUID_VAR_LABEL,
    iter_time_blocks,
    parse_utc_hours,
    phase_masks,
    precip_mask,
    resolve_phase_thresholds,
    season_layout,
    season_window_hours,
    select_seasons,
    site_cell_mask,
    utc_hour_mask,
    window_label,
)
from seb_analysis_common import load_seb_data, resolve_region_dir
from types import SimpleNamespace

# The phases whose union is "liquid containing", matching season_phase_binary.
LIQUID_CONTAINING: tuple[str, ...] = ("liquid", "mixed")


class MapAnalysis(SimpleNamespace):
    """Per-cell, per-season cloud statistics. Built by :func:`prepare_maps`."""


def prepare_maps(argv=None, args=None, **overrides) -> MapAnalysis:
    """One streaming pass, accumulating per-cell counts on the native grid.

    Takes the same options as ``plot_lwp_histogram_by_surface_class`` -- the
    parser is shared, so "the same inputs" is literal rather than a claim about
    two parallel implementations.
    """
    if args is None:
        args = lwph.parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    phase_kw = resolve_phase_thresholds(args)
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)

    need = {"tcc", "tciw", args.liquid_var}
    missing = sorted(need - set(ds.data_vars))
    if missing:
        raise KeyError(f"dataset is missing {missing}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    season_hours = season_window_hours(layout, keep_idx)          # (season,)

    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    use_step = use_step & utc_hour_mask(ds, args)
    if not use_step.any():
        raise ValueError("no time steps left after the season and hour filters")

    # Remap the archive's season index onto the SELECTED seasons, the same trap
    # that bit the threshold fit: layout["seasons"] lists every season on disk.
    remap = np.full(len(layout["seasons"]), -1, dtype=np.intp)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.intp)

    n_y, n_x = ds.sizes["latitude"], ds.sizes["longitude"]
    n_s = len(keep_idx)
    counts = {k: np.zeros((n_s, n_y, n_x)) for k in
              ("valid", "cloudy", "liquid", "mixed", "ice", "none")}

    read_vars = ["tcc", "tciw", args.liquid_var]
    if args.no_precip:
        read_vars += [v for v in lwph.PRECIP_SOURCE_VARS[args.precip_var]
                      if v not in read_vars]

    print(f"\n  Reading {n_s} season(s) on the native "
          f"{n_y} x {n_x} grid: {used}")
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        keep = use_step[slice(i0, i0 + n_t)]
        if not keep.any():
            continue
        si = remap[s_idx[slice(i0, i0 + n_t)][keep]]

        tcc = block["tcc"].values[keep]
        lwp_g = block[args.liquid_var].values[keep] * 1000.0   # kg m-2 -> g m-2
        iwp_g = block["tciw"].values[keep] * 1000.0

        valid = np.isfinite(tcc) & np.isfinite(lwp_g) & np.isfinite(iwp_g)
        raining = precip_mask(block, keep, args)
        cloudy = valid & (tcc >= args.min_cloud_fraction) & ~raining
        phases = phase_masks(lwp_g, iwp_g, phase_kw)

        # np.add.at on the season axis: a block can straddle two seasons, and
        # bincount would need a flat index over the whole grid for no gain.
        np.add.at(counts["valid"], si, valid)
        np.add.at(counts["cloudy"], si, cloudy)
        for name in ("liquid", "mixed", "ice", "none"):
            np.add.at(counts[name], si, cloudy & phases[name])

    n_valid = counts["valid"]
    den = np.where(n_valid > 0, n_valid, np.nan)
    n_liq = sum(counts[p] for p in LIQUID_CONTAINING)
    # Ice-only absorbs the "no phase" residual, exactly as the single-cell
    # figures do, so the two categories still sum to the cloudy total.
    n_ice = counts["ice"] + counts["none"]

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = n_liq / den                                  # (s, y, x)
        out = MapAnalysis(
            args=args, ds=ds, layout=layout, keep_idx=keep_idx, used=list(used),
            mode_label=mode_label, phase_kw=phase_kw,
            lat=np.asarray(ds["latitude"].values, dtype=float),
            lon=np.asarray(ds["longitude"].values, dtype=float),
            season_hours=season_hours,
            counts=counts,
            hours=frac * season_hours[:, None, None],       # (s, y, x) hours
            fraction_pct=100.0 * frac,                      # (s, y, x) %
            cloudy_pct=100.0 * counts["cloudy"] / den,
            ice_hours=n_ice / den * season_hours[:, None, None],
            n_valid=n_valid,
            tag=(f"season{used[0]}" if len(used) == 1
                 else f"mean{used[0]}-{used[-1]}"),
        )
    out.hours_mean = np.nanmean(out.hours, axis=0)           # (y, x)
    out.fraction_pct_mean = np.nanmean(out.fraction_pct, axis=0)
    out.cloudy_pct_mean = np.nanmean(out.cloudy_pct, axis=0)
    out.site_mask, out.site_lat, out.site_lon = site_cell_mask(ds)
    check_invariants(out)
    return out


def check_invariants(M: MapAnalysis, tol: float = 1e-9) -> None:
    """Assert the two properties the maps are only meaningful if they hold.

    1. The liquid-containing fraction never exceeds the cloudy fraction. Both
       are subsets of the same denominator, so a violation would mean the phase
       masks are not nested inside ``cloudy`` -- a real bug, not a rounding
       matter.
    2. Every fraction lies in [0, 100].
    """
    ok = np.isfinite(M.fraction_pct) & np.isfinite(M.cloudy_pct)
    over = M.fraction_pct[ok] - M.cloudy_pct[ok]
    if over.size and over.max() > tol:
        n = int(np.count_nonzero(over > tol))
        raise AssertionError(
            f"liquid-containing fraction exceeds the cloudy fraction in {n} "
            f"cell-seasons, by up to {over.max():.6g} points; the phase masks "
            f"are not nested inside 'cloudy'")
    f = M.fraction_pct[np.isfinite(M.fraction_pct)]
    if f.size and (f.min() < -tol or f.max() > 100.0 + tol):
        raise AssertionError(f"fraction outside [0, 100]: "
                             f"{f.min():.6g} to {f.max():.6g}")


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------
# Sequential, perceptually uniform, and legible in both light and dark: the
# quantity is a magnitude with a meaningful zero, so a diverging map would imply
# a midpoint that does not exist.
HOURS_CMAP = "viridis"
FRACTION_CMAP = "viridis"

QUANTITIES = {
    "hours": ("hours", "Liquid-containing cloud hours per season",
              "hours per season [h]", HOURS_CMAP),
    "fraction": ("fraction_pct",
                 "Fraction of the season with liquid-containing cloud",
                 "fraction of the season [%]", FRACTION_CMAP),
    "cloudy": ("cloudy_pct", "Fraction of the season overcast",
               "fraction of the season [%]", FRACTION_CMAP),
}


def _projection(M):
    """North polar stereographic centred on the domain.

    PlateCarree would stretch a 70-80 N box badly -- 15 degrees of longitude is
    about 570 km at 70 N and 290 km at 80 N, so a rectangular plot misstates the
    shape of everything in it.
    """
    import cartopy.crs as ccrs
    return ccrs.NorthPolarStereo(central_longitude=float(np.mean(M.lon)))


def projected_aspect(M) -> float:
    """Domain width / height in PROJECTED coordinates.

    Needed because the domain is a trapezoid, not a rectangle: 15 degrees of
    longitude is about 570 km at 70 N and 290 km at 80 N, against 1,110 km of
    latitude span. Sizing a panel as a square therefore leaves two thirds of it
    empty. Sampling the boundary rather than assuming a formula keeps this
    correct if the region or the projection changes.
    """
    import cartopy.crs as ccrs

    proj = _projection(M)
    lo = np.linspace(M.lon.min(), M.lon.max(), 25)
    la = np.linspace(M.lat.min(), M.lat.max(), 25)
    LO, LA = np.meshgrid(lo, la)
    xy = proj.transform_points(ccrs.PlateCarree(), LO.ravel(), LA.ravel())
    x, y = xy[:, 0], xy[:, 1]
    return float((x.max() - x.min()) / (y.max() - y.min()))


def _cell_edges(centres):
    """Cell edges from centres, for pcolormesh's non-interpolating form."""
    c = np.asarray(centres, dtype=float)
    step = np.diff(c)
    first = c[0] - step[0] / 2.0
    last = c[-1] + step[-1] / 2.0
    return np.concatenate([[first], c[:-1] + step / 2.0, [last]])


def _draw_one(ax, M, field, vmin, vmax, cmap, mark_site=True):
    """One map panel: the field, the coast, and the ARM cell."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pc = ccrs.PlateCarree()
    lon_e, lat_e = _cell_edges(M.lon), _cell_edges(M.lat)
    mesh = ax.pcolormesh(lon_e, lat_e, field, transform=pc, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="flat")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8,
                   edgecolor="white")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none",
                   edgecolor="none")
    ax.set_extent([M.lon.min(), M.lon.max(), M.lat.min(), M.lat.max()], crs=pc)
    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="white",
                      alpha=0.35)
    gl.top_labels = gl.right_labels = False
    if mark_site:
        ax.plot(M.site_lon, M.site_lat, marker="*", ms=13, mfc="red",
                mec="white", mew=1.0, transform=pc, zorder=6)
    return mesh


def _wrap(text: str, width: int) -> str:
    """Wrap a title so a narrow, tall figure is not stretched by its text."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width)) or text


def _subtitle(M) -> str:
    a, pk = M.args, M.phase_kw
    bits = [f"tcc $\\geq$ {a.min_cloud_fraction:g}"]
    if pk["mode"] == "fraction":
        bits.append(f"liquid containing: LWP/(LWP+IWP) > "
                    f"{100 * (1 - pk['ice_fraction_min']):g}%")
        bits.append(f"min LWP/IWP {pk['min_lwp_g']:g}/{pk['min_iwp_g']:g} "
                    f"g m$^{{-2}}$")
    bits.append(LIQUID_VAR_LABEL[a.liquid_var])
    uh = parse_utc_hours(getattr(a, "utc_hours", None))
    if uh:
        bits.append(f"{len(uh)} of 24 UTC hours")
    return "   |   ".join(bits)


def precip_banner(args, mathtext: bool = True) -> str:
    """Unmissable statement of the precipitation setting, for a figure title.

    The two versions of every figure are otherwise near-identical, so the label
    has to be the first thing read rather than one clause among several in a
    subtitle -- a reader comparing two printouts must never have to hunt for it.
    """
    if not getattr(args, "no_precip", False):
        return "ALL SKY \u2014 no precipitation filter"
    if args.precip_var == "rate":
        unit = "mm hr$^{-1}$" if mathtext else "mm/hr"
        lt = "$<$" if mathtext else "<"
        return (f"PRECIPITATION FILTERED \u2014 "
                f"tp {lt} {args.precip_rate_max:g} {unit}")
    unit = "g m$^{-2}$" if mathtext else "g m-2"
    lt = "$<$" if mathtext else "<"
    return (f"PRECIPITATION FILTERED \u2014 rain+snow path {lt} "
            f"{args.precip_path_max:g} {unit}")


def precip_suffix(args) -> str:
    """File-name tag for the precipitation setting.

    Without it an all-sky run and a filtered run write to the same path and the
    second silently overwrites the first -- the two differ only in an argument,
    not in region, tag or quantity.
    """
    if not getattr(args, "no_precip", False):
        return "allsky"
    if args.precip_var == "rate":
        return f"noprecip{args.precip_rate_max:g}mmhr".replace(".", "p")
    return f"noprecip{args.precip_path_max:g}gm2".replace(".", "p")


def _save(fig, M, out_dir, stem, dpi):
    from pathlib import Path as _P
    if out_dir is None:
        return fig
    path = _P(out_dir) / (f"{M.args.region}_{stem}_"
                          f"{precip_suffix(M.args)}_{M.tag}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or M.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def fig_season_panels(M: MapAnalysis, quantity: str = "hours", n_cols: int = 6,
                      out_dir=None, dpi=350, vmin=None, vmax=None,
                      panel_h: float = 4.4, panel_w: float | None = None):
    """One map panel per season, on a SHARED colour scale.

    The shared scale is the point: per-panel scales would make every season look
    the same and hide the interannual range, which is the only thing a
    multi-season figure is for. Pass vmin/vmax to override.
    """
    import matplotlib.pyplot as plt

    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; "
                         f"choose from {list(QUANTITIES)}")
    attr, title, cbar_label, cmap = QUANTITIES[quantity]
    data = getattr(M, attr)                                  # (s, y, x)
    n_s = data.shape[0]
    n_cols = min(n_cols, n_s)
    n_rows = int(np.ceil(n_s / n_cols))
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    proj = _projection(M)
    if panel_w is None:
        # Width from the projected aspect, plus a little room for the title.
        panel_w = max(panel_h * projected_aspect(M) * 1.30, 1.5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols, panel_h * n_rows),
                             subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()
    mesh = None
    for k in range(n_s):
        ax = axes[k]
        mesh = _draw_one(ax, M, data[k], vmin, vmax, cmap)
        y = M.used[k]
        med = float(np.nanmedian(data[k]))
        unit = "h" if quantity == "hours" else "%"
        ax.set_title(f"{y}/{(y + 1) % 100:02d}\nmedian {med:,.0f} {unit}",
                     fontsize=10.5)
    for ax in axes[n_s:]:
        ax.set_visible(False)

    cb = fig.colorbar(mesh, ax=axes[:n_s].tolist(), orientation="horizontal",
                      fraction=0.035, pad=0.04, aspect=45)
    cb.set_label(cbar_label, fontsize=11.5)
    fig.suptitle(f"{precip_banner(M.args)}\n{title} — {M.args.region} region, "
                 f"{M.args.season_start[0]:02d}-{M.args.season_start[1]:02d} to "
                 f"{M.args.season_end[0]:02d}-{M.args.season_end[1]:02d}"
                 f"\n{_subtitle(M)}\nred star: the Utqiagvik ARM cell   |   "
                 f"shared colour scale across panels",
                 fontsize=12.5, y=0.995)
    return _save(fig, M, out_dir, f"map_{quantity}_by_season", dpi)


def fig_season_mean(M: MapAnalysis, quantity: str = "hours", out_dir=None,
                    dpi=350, vmin=None, vmax=None, height: float = 8.2,
                    figsize=None):
    """The across-season mean at each cell, as one map.

    An unweighted mean over seasons, so every season counts once regardless of
    how many hours of it the archive holds -- consistent with the rest of the
    project, and the reason --min-season-coverage exists to keep a badly
    incomplete season out of it.
    """
    import matplotlib.pyplot as plt

    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; "
                         f"choose from {list(QUANTITIES)}")
    attr, title, cbar_label, cmap = QUANTITIES[quantity]
    data = getattr(M, f"{attr}_mean")
    unit = "h" if quantity == "hours" else "%"
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    if figsize is None:
        # A wide canvas around a tall trapezoid is mostly whitespace; size the
        # canvas to the data instead, leaving margin for the title block.
        figsize = (max(height * projected_aspect(M) * 1.55, 4.2), height)
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": _projection(M)})
    mesh = _draw_one(ax, M, data, vmin, vmax, cmap)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.05, aspect=32)
    cb.set_label(f"mean {cbar_label}", fontsize=11.5)

    site = float(data[np.argwhere(M.site_mask)[0][0],
                      np.argwhere(M.site_mask)[0][1]])
    ax.set_title(
        f"{precip_banner(M.args)}\n{_wrap(title, 46)}\nmean over "
        f"{len(M.used)} seasons "
        f"({M.used[0]}/{(M.used[0]+1) % 100:02d}–"
        f"{M.used[-1]}/{(M.used[-1]+1) % 100:02d})   |   "
        f"{M.args.region} region"
        f"\n{_subtitle(M)}"
        f"\ndomain min {np.nanmin(data):,.0f} {unit}   median "
        f"{np.nanmedian(data):,.0f} {unit}   max {np.nanmax(data):,.0f} {unit}"
        f"   |   ARM cell {site:,.0f} {unit}",
        fontsize=11.0)
    return _save(fig, M, out_dir, f"map_{quantity}_season_mean", dpi)


def print_map_report(M: MapAnalysis) -> None:
    """Per-season domain statistics, and the ARM cell for cross-checking.

    The ARM-cell column is the tie to ``comparison_with_Genie_Obs.ipynb``: it
    must reproduce that notebook's liquid-containing hours for the same
    settings, which is what makes this map trustworthy rather than merely
    plausible.
    """
    iy, ix = np.argwhere(M.site_mask)[0]
    print(f"*** {precip_banner(M.args, mathtext=False)} ***")
    print(f"{M.args.region} region   |   {M.lat.min():.2f}-{M.lat.max():.2f} N, "
          f"{M.lon.min():.2f}-{M.lon.max():.2f} E   |   "
          f"{M.lat.size} x {M.lon.size} cells")
    print(f"season window {window_label(M.season_hours)}   |   "
          f"ARM cell ({M.site_lat:.2f} N, {M.site_lon:.2f} E)\n")
    print(f"{'season':<10}{'window':>8}" +
          "".join(f"{c:>10}" for c in
                  ("min h", "med h", "max h", "ARM h", "med %", "ARM %",
                   "ARM cld%")))
    for k, y in enumerate(M.used):
        h, f, c = M.hours[k], M.fraction_pct[k], M.cloudy_pct[k]
        print(f"{y}/{(y+1) % 100:02d}{'':<3}{M.season_hours[k]:>8,.0f}"
              f"{np.nanmin(h):>10,.0f}{np.nanmedian(h):>10,.0f}"
              f"{np.nanmax(h):>10,.0f}{h[iy, ix]:>10,.0f}"
              f"{np.nanmedian(f):>9.1f}%{f[iy, ix]:>9.1f}%{c[iy, ix]:>9.1f}%")
    print(f"\n{'mean':<10}{'':>8}{np.nanmin(M.hours_mean):>10,.0f}"
          f"{np.nanmedian(M.hours_mean):>10,.0f}{np.nanmax(M.hours_mean):>10,.0f}"
          f"{M.hours_mean[iy, ix]:>10,.0f}"
          f"{np.nanmedian(M.fraction_pct_mean):>9.1f}%"
          f"{M.fraction_pct_mean[iy, ix]:>9.1f}%"
          f"{M.cloudy_pct_mean[iy, ix]:>9.1f}%")
