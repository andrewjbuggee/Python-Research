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

import calendar

import numpy as np

import plot_lwp_histogram_by_surface_class as lwph
from plot_lwp_histogram_by_surface_class import (
    LIQUID_VAR_LABEL,
    iter_time_blocks,
    parse_utc_hours,
    phase_masks,
    season_month_axis,
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
    # LWP summed over liquid-containing hours, so a per-cell mean LWP of that
    # population is sum / count. Kept separate from the season-total counts
    # because the mean must be over liquid-containing hours ONLY.
    sum_lwp = np.zeros((n_s, n_y, n_x))

    # Month-resolved copies of the same three quantities, for one season drawn
    # month by month, plus the monthly-mean sea ice concentration that puts
    # the ice edge on those panels. (season, month, y, x): eleven seasons, six
    # months, 2,501 cells -- about 1.3 MB each, so nothing to economise on.
    months, mi_of_slot = season_month_axis(layout["slots"])
    n_m = len(months)
    dos = layout["dos"]
    m_valid = np.zeros((n_s, n_m, n_y, n_x))
    m_liq = np.zeros((n_s, n_m, n_y, n_x))
    m_lwp = np.zeros((n_s, n_m, n_y, n_x))
    m_sic = np.zeros((n_s, n_m, n_y, n_x))
    m_sic_n = np.zeros((n_s, n_m, n_y, n_x))

    read_vars = ["tcc", "tciw", args.liquid_var, "siconc"]
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
        mi = mi_of_slot[dos[slice(i0, i0 + n_t)][keep]]

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
        liq = cloudy & (phases["liquid"] | phases["mixed"])
        np.add.at(sum_lwp, si, np.where(liq, lwp_g, 0.0))

        sic = block["siconc"].values[keep]
        sic_ok = np.isfinite(sic)
        np.add.at(m_valid, (si, mi), valid)
        np.add.at(m_liq, (si, mi), liq)
        np.add.at(m_lwp, (si, mi), np.where(liq, lwp_g, 0.0))
        np.add.at(m_sic, (si, mi), np.where(sic_ok, sic, 0.0))
        np.add.at(m_sic_n, (si, mi), sic_ok)

    n_valid = counts["valid"]
    den = np.where(n_valid > 0, n_valid, np.nan)
    n_liq = sum(counts[p] for p in LIQUID_CONTAINING)
    # Ice-only absorbs the "no phase" residual, exactly as the single-cell
    # figures do, so the two categories still sum to the cloudy total.
    n_ice = counts["ice"] + counts["none"]

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = n_liq / den                                  # (s, y, x)
        lwp_mean = np.where(n_liq > 0, sum_lwp / np.where(n_liq > 0, n_liq, 1),
                            np.nan)                         # g m-2
        m_den = np.where(m_valid > 0, m_valid, np.nan)
        # Fraction of the MONTH with liquid-containing cloud, so a partly
        # sampled month is a rate rather than a shortfall. Hours per day is the
        # same number times 24; both are kept.
        m_fraction_pct = 100.0 * m_liq / m_den
        m_hours_per_day = 24.0 * m_liq / m_den
        m_lwp_mean = np.where(m_liq > 0, m_lwp / np.where(m_liq > 0, m_liq, 1),
                              np.nan)
        m_sic_mean = np.where(m_sic_n > 0, m_sic / np.where(m_sic_n > 0,
                                                            m_sic_n, 1), np.nan)
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
            lwp_mean=lwp_mean,                              # (s, y, x) g m-2
            months=months,
            month_fraction_pct=m_fraction_pct,              # (s, m, y, x) %
            month_hours_per_day=m_hours_per_day,            # (s, m, y, x)
            month_lwp_mean=m_lwp_mean,                      # (s, m, y, x)
            month_siconc_mean=m_sic_mean,                   # (s, m, y, x)
            month_n_valid=m_valid,
            tag=(f"season{used[0]}" if len(used) == 1
                 else f"mean{used[0]}-{used[-1]}"),
        )
    out.hours_mean = np.nanmean(out.hours, axis=0)           # (y, x)
    out.fraction_pct_mean = np.nanmean(out.fraction_pct, axis=0)
    out.cloudy_pct_mean = np.nanmean(out.cloudy_pct, axis=0)
    # Mean LWP over seasons, weighted by each season's liquid-containing hours
    # so it is the mean over ALL such hours rather than a mean of season means.
    with np.errstate(invalid="ignore", divide="ignore"):
        tot_liq = n_liq.sum(axis=0)
        out.lwp_mean_all = np.where(tot_liq > 0,
                                    sum_lwp.sum(axis=0)
                                    / np.where(tot_liq > 0, tot_liq, 1), np.nan)
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
# Colour maps chosen to read as the physical quantity rather than as a scale:
#
#   gray   cloud occurrence -- black where liquid cloud is rare, WHITE where it
#          is common, because clouds are white. Used for hours, fraction and
#          the overcast fraction alike, since all three are "how often".
#   Blues  liquid water path -- white for thin, DEEP BLUE for thick, more
#          liquid reading as more blue.
#
# Both are sequential with a meaningful zero, which is what the quantities
# are. The cost is that line art in a single colour vanishes at one end of a
# greyscale map, so every coastline and ice contour is drawn black with a thin
# white halo (see _halo) and reads on both black and white.
HOURS_CMAP = "gray"
FRACTION_CMAP = "gray"
LWP_CMAP = "Blues"


# Line art that must read on black, every grey, white, and deep blue. Each is
# a saturated hue with a thin white halo: the hue carries it on light and mid
# tones, the halo on dark ones. Defined once so the two-panel and monthly
# figures cannot drift apart.
GRID_COLOR = "#1f5fa8"        # blue -- distinct from the greys, and the halo
                              # carries it over the blue map's dark end
ICE_EDGE_COLOR = "#e6550d"    # orange -- complementary to the blue map, and
                              # unlike anything on the grey one
COAST_COLOR = "black"

# Defaults for --map-legend-fontsize / --map-tick-fontsize, used when a figure
# is called with legend_fontsize=None / tick_fontsize=None and the run's args
# carry no value (an Analysis built before the flags existed).
DEFAULT_MAP_LEGEND_FONTSIZE = 12.0
DEFAULT_MAP_TICK_FONTSIZE = 10.0


def _fonts(M, legend_fontsize, tick_fontsize):
    """(legend, tick) sizes: the argument, else the run's flag, else default."""
    a = M.args
    if legend_fontsize is None:
        legend_fontsize = getattr(a, "map_legend_fontsize",
                                  DEFAULT_MAP_LEGEND_FONTSIZE)
    if tick_fontsize is None:
        tick_fontsize = getattr(a, "map_tick_fontsize",
                                DEFAULT_MAP_TICK_FONTSIZE)
    return float(legend_fontsize), float(tick_fontsize)


def _halo(lw: float = 1.6):
    """A white stroke behind a line, so it reads on any background."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground="white")]

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


def _draw_one(ax, M, field, vmin, vmax, cmap, mark_site=True, labels=True,
              tick_fontsize=DEFAULT_MAP_TICK_FONTSIZE):
    """One map panel: the field, the coast, and the ARM cell."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pc = ccrs.PlateCarree()
    lon_e, lat_e = _cell_edges(M.lon), _cell_edges(M.lat)
    mesh = ax.pcolormesh(lon_e, lat_e, field, transform=pc, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="flat")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.9,
                   edgecolor=COAST_COLOR, path_effects=_halo(2.4), zorder=4)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none",
                   edgecolor="none")
    ax.set_extent([M.lon.min(), M.lon.max(), M.lat.min(), M.lat.max()], crs=pc)
    # Labelled gridlines along the left and bottom only; on a polar
    # stereographic axes the top and right edges are curved and their labels
    # land inside neighbouring panels.
    sides = (labels if isinstance(labels, (list, tuple))
             else (["left", "bottom"] if labels else False))
    gl = ax.gridlines(draw_labels=sides,
                      linewidth=0.45, color=GRID_COLOR, alpha=0.9,
                      path_effects=_halo(1.4),
                      xlocs=list(range(-170, -140, 5)),
                      ylocs=list(range(70, 81, 2)), rotate_labels=False,
                      x_inline=False, y_inline=False)
    gl.xlabel_style = gl.ylabel_style = {"size": tick_fontsize, "color": "0.3"}
    if mark_site:
        ax.plot(M.site_lon, M.site_lat, marker="*", ms=13, mfc="red",
                mec="white", mew=1.0, transform=pc, zorder=6)
    return mesh


def _site_legend(fig, loc=(0.985, 0.012), fontsize=DEFAULT_MAP_LEGEND_FONTSIZE):
    """The red star, named, in the figure margin rather than over the map."""
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], marker="*", ms=12, mfc="red", mec="white",
                       mew=1.0, ls="none")],
               ["Utqia\u0121vik (DOE ARM site)"], loc="lower right",
               bbox_to_anchor=loc, fontsize=fontsize, framealpha=0.95,
               handletextpad=0.4)


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
                      panel_h: float = 4.4, panel_w: float | None = None,
                      legend_fontsize: float | None = None,
                      tick_fontsize: float | None = None):
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
    legend_fontsize, tick_fontsize = _fonts(M, legend_fontsize, tick_fontsize)
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
        # Tick labels on the first column and last row only; every panel
        # labelled would be unreadable at this density.
        r, c = divmod(k, n_cols)
        sides = ([sd for sd, on in (("left", c == 0), ("bottom", r == n_rows - 1))
                  if on] or False)
        mesh = _draw_one(ax, M, data[k], vmin, vmax, cmap, labels=sides,
                         tick_fontsize=tick_fontsize)
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
    cb.ax.tick_params(labelsize=tick_fontsize)
    _site_legend(fig, fontsize=legend_fontsize)
    fig.suptitle(f"{precip_banner(M.args)}\n{title} — {M.args.region} region, "
                 f"{M.args.season_start[0]:02d}-{M.args.season_start[1]:02d} to "
                 f"{M.args.season_end[0]:02d}-{M.args.season_end[1]:02d}"
                 f"\n{_subtitle(M)}\nred star: the Utqiagvik ARM cell   |   "
                 f"shared colour scale across panels",
                 fontsize=12.5, y=0.995)
    return _save(fig, M, out_dir, f"map_{quantity}_by_season", dpi)


def fig_season_mean(M: MapAnalysis, quantity: str = "hours", out_dir=None,
                    dpi=350, vmin=None, vmax=None, height: float = 8.2,
                    figsize=None, legend_fontsize: float | None = None,
                    tick_fontsize: float | None = None):
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
    legend_fontsize, tick_fontsize = _fonts(M, legend_fontsize, tick_fontsize)
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
    mesh = _draw_one(ax, M, data, vmin, vmax, cmap, tick_fontsize=tick_fontsize)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.05, aspect=32)
    cb.set_label(f"mean {cbar_label}", fontsize=11.5)
    cb.ax.tick_params(labelsize=tick_fontsize)
    _site_legend(fig, fontsize=legend_fontsize)

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


# ----------------------------------------------------------------------------
# Two-panel: mean fraction beside mean LWP of liquid-containing cloud
# ----------------------------------------------------------------------------
def fig_fraction_and_lwp(M: MapAnalysis, out_dir=None, dpi=None,
                         height: float = 7.6,
                         legend_fontsize: float | None = None,
                         tick_fontsize: float | None = None):
    """Season-mean liquid-containing fraction beside its mean LWP, per cell.

    Left: fraction of the season with liquid-containing cloud, as in
    :func:`fig_season_mean`. Right: mean LWP over those liquid-containing hours
    -- weighted by hours across seasons, so it is the mean over every such hour
    rather than a mean of season means. The pair separates HOW OFTEN a cell has
    liquid cloud from HOW MUCH liquid it holds when it does, which need not
    vary together across the domain.
    """
    import matplotlib.pyplot as plt

    legend_fontsize, tick_fontsize = _fonts(M, legend_fontsize, tick_fontsize)
    proj = _projection(M)
    asp = projected_aspect(M)
    panel_w = max(height * asp * 1.45, 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 0.6, height),
                             subplot_kw={"projection": proj})
    specs = (
        (M.fraction_pct_mean, "fraction of the season with liquid-containing "
                              "cloud [%]", FRACTION_CMAP),
        (M.lwp_mean_all, "mean LWP of liquid-containing cloud [g m$^{-2}$]",
         LWP_CMAP),
    )
    iy, ix = np.argwhere(M.site_mask)[0]
    for ax, (field, cbl, cmap) in zip(axes, specs):
        mesh = _draw_one(ax, M, field, float(np.nanmin(field)),
                         float(np.nanmax(field)), cmap, tick_fontsize=tick_fontsize)
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05,
                          pad=0.06, aspect=30)
        cb.set_label(cbl, fontsize=10.5)
        cb.ax.tick_params(labelsize=tick_fontsize)
        unit = "%" if cmap is FRACTION_CMAP else "g m$^{-2}$"
        ax.set_title(f"domain median {np.nanmedian(field):,.1f} {unit}   |   "
                     f"ARM cell {field[iy, ix]:,.1f} {unit}", fontsize=10)
    _site_legend(fig, fontsize=legend_fontsize)
    fig.suptitle(f"{precip_banner(M.args)}\n"
                 f"Liquid-containing cloud: how often, and how much liquid — "
                 f"{M.args.region}, mean over {len(M.used)} seasons "
                 f"({M.used[0]}/{(M.used[0]+1) % 100:02d}–"
                 f"{M.used[-1]}/{(M.used[-1]+1) % 100:02d})\n{_subtitle(M)}",
                 fontsize=11.5, y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.98,
                        wspace=0.12)
    return _save(fig, M, out_dir, "map_fraction_and_lwp", dpi)


# ----------------------------------------------------------------------------
# One season, month by month: hours per day and mean LWP, with the ice edge
# ----------------------------------------------------------------------------
# Sea ice concentration contoured on every panel: the open-water edge and the
# start of full pack ice. The same construction plot_monthly_flux_maps.py uses
# for the turbulent-flux maps, with the levels this analysis classifies on.
ICE_EDGE_LEVELS = (0.05, 0.95)
ICE_EDGE_STYLES = ("--", "-")


def fig_season_monthly_maps(M: MapAnalysis, season: int, out_dir=None,
                            dpi=None, panel_h: float = 3.9,
                            ice_levels=ICE_EDGE_LEVELS,
                            legend_fontsize: float | None = None,
                            tick_fontsize: float | None = None):
    """Two rows by month for ONE season: liquid hours per day, and mean LWP.

    Top row: fraction of the month with liquid-containing cloud overhead, per
    cell, on the same black-to-white scale as the season maps -- a partly
    sampled month reads as a rate rather than a shortfall. Bottom row: mean
    LWP over those hours, on the white-to-blue scale. Each row has one colour
    scale across its months.

    On every panel, the monthly-mean sea ice concentration contoured at
    ``ice_levels`` -- dashed where open water gives way to ice, solid where
    full pack ice begins -- in ``ICE_EDGE_COLOR`` with a white halo, chosen to
    read on both the grey and the blue map. The classes in this project are cut
    at the same concentrations, so the lines are the boundaries between open
    ocean, marginal ice zone and sea ice as the season freezes up.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if season not in M.used:
        raise ValueError(f"season {season} not in this run: {M.used}")
    legend_fontsize, tick_fontsize = _fonts(M, legend_fontsize, tick_fontsize)
    si = M.used.index(season)
    months = list(M.months)
    n_m = len(months)
    frac = M.month_fraction_pct[si]           # (m, y, x)
    lwp = M.month_lwp_mean[si]
    sic = M.month_siconc_mean[si]
    nv = M.month_n_valid[si]

    proj = _projection(M)
    panel_w = max(panel_h * projected_aspect(M) * 1.25, 1.6)
    fig, axes = plt.subplots(2, n_m, figsize=(panel_w * n_m + 0.8,
                                              2 * panel_h + 1.6),
                             subplot_kw={"projection": proj})
    rows = (
        (frac, "fraction of the month with liquid-containing cloud [%]",
         FRACTION_CMAP),
        (lwp, "mean LWP of liquid-containing cloud [g m$^{-2}$]", LWP_CMAP),
    )
    pc = ccrs.PlateCarree()
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.03, right=0.93,
                        hspace=0.16, wspace=0.05)
    for r, (data, cbl, cmap) in enumerate(rows):
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        mesh = None
        for j, m in enumerate(months):
            ax = axes[r, j]
            sides = ([sd for sd, on in (("left", j == 0), ("bottom", r == 1))
                      if on] or False)
            mesh = _draw_one(ax, M, data[j], vmin, vmax, cmap, labels=sides,
                             tick_fontsize=tick_fontsize)
            ice = sic[j]
            if np.isfinite(ice).any():
                cs = ax.contour(M.lon, M.lat, ice, levels=list(ice_levels),
                                colors=ICE_EDGE_COLOR, linewidths=1.2,
                                linestyles=list(ICE_EDGE_STYLES),
                                transform=pc, zorder=5)
                cs.set_path_effects(_halo(2.8))
            if r == 0:
                # Days of the month with data, so a short month is visible.
                days = float(np.nanmax(nv[j])) / 24.0
                ax.set_title(f"{calendar.month_abbr[m]}\n({days:.0f} days)",
                             fontsize=10)
        # A dedicated colourbar axes beside the row: letting colorbar() steal
        # space from six cartopy axes put it on top of the last panel.
        fig.canvas.draw()
        bb = axes[r, -1].get_position()
        cax = fig.add_axes([bb.x1 + 0.012, bb.y0, 0.012, bb.height])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(cbl, fontsize=9.5)
        cb.ax.tick_params(labelsize=tick_fontsize)

    _site_legend(fig, loc=(0.985, 0.005), fontsize=legend_fontsize)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=ICE_EDGE_COLOR, lw=1.2,
                       ls=ICE_EDGE_STYLES[0], path_effects=_halo(2.8)),
                Line2D([0], [0], color=ICE_EDGE_COLOR, lw=1.2,
                       ls=ICE_EDGE_STYLES[1], path_effects=_halo(2.8))],
               [f"sea ice concentration {ice_levels[0]:g} (ice edge)",
                f"sea ice concentration {ice_levels[1]:g} (full pack ice)"],
               loc="lower left", bbox_to_anchor=(0.01, 0.005),
               fontsize=legend_fontsize, framealpha=0.95, ncol=2)
    fig.suptitle(f"{precip_banner(M.args)}\n"
                 f"Liquid-containing cloud through the {season}/"
                 f"{(season + 1) % 100:02d} season — {M.args.region}"
                 f"\n{_subtitle(M)}", fontsize=11.5, y=0.995)
    return _save(fig, M, out_dir, f"map_monthly_{season}", dpi)
