#!/usr/bin/env python3
"""Horizontal extent of liquid-containing cloud over the Barrow region, ERA5.

Two independent estimates of "how wide is a liquid-containing cloud", built on
the IDENTICAL cloud definition used everywhere else in this project -- the
``tcc`` gate, the fractional LWP/IWP phase boundaries, the minimum water paths,
the precipitation filter and the Oct-Mar season window from
``plot_lwp_histogram_by_surface_class`` (the parser is shared, so "identical" is
literal). Every grid cell of the region is treated on the same footing; nothing
is collapsed to the ARM cell except where the notebook asks for that cell as a
cross-check.

METHOD 1 -- TAYLOR'S FROZEN-TURBULENCE HYPOTHESIS, CELL BY CELL
================================================================
Taylor (1938, Proc. R. Soc. Lond. A 164, 476-490) assumes the field is advected
past a fixed point faster than it evolves, so a duration measured at the point
maps onto a distance along the wind:

    L = U * dt                                                          (1)

At each grid cell an EVENT is a maximal run of consecutive hours in which every
hour meets the liquid-containing criterion, exactly as ``liquid_cloud_runs``
defines it for the ARM cell (a run cannot cross a gap in the archive or a season
boundary; a run touching such a break is CENSORED, kept and flagged). ``dt`` is
the run length in hours and ``U`` is the mean 10 m wind speed over those hours,

    U_scalar = mean( sqrt(u10^2 + v10^2) )                              (2)

which is what was asked for. The magnitude of the mean wind VECTOR,

    U_vector = | mean(u10), mean(v10) |                                 (3)

is carried alongside because it is the quantity that actually turns a duration
into a net displacement (as ``taylor_hypothesis_cloud_spatial_scale.ipynb``
argued for the tower data): a wind that veers during a long event advects the
cloud less far than the scalar mean implies, and (3) <= (2) always. Both are
returned; the figures use (2) unless told otherwise.

Why the 10 m wind and not a cloud-level wind: the archive carries no pressure-
level winds for the region, and most cold-season liquid clouds at Utqiagvik are
boundary-layer clouds with bases below 500 m. A 10 m wind under a stable Arctic
boundary layer is a LOWER bound on the wind at cloud level, so the lengths here
are biased low by whatever shear sits between the surface and cloud base -- the
tower notebook found 6.7 m/s at 40 m against 5.3 m/s at 10 m, a 25% difference
over 30 m. Read the Taylor lengths as conservative.

METHOD 2 -- CONNECTED COMPONENTS, SNAPSHOT BY SNAPSHOT
======================================================
No hypothesis about advection. At every hour the (lat, lon) field of the
liquid-containing flag is a binary image; its connected components -- groups of
adjacent flagged cells -- are the clouds, and their size is measured directly on
the sphere. Isolated single cells are components too, and are kept: a histogram
that dropped them would be a histogram of large clouds only.

Component size is the awkward part of this method: the components are irregular
blobs, and "width" has no single direction. Several measures are therefore
computed, each answering a different question:

    area_km2       sum of the exact spherical cell areas, R^2 dlam (sin phi_n -
                   sin phi_s)   -- the one number with no direction in it
    d_eq_km        2 sqrt(area / pi): the diameter of the circle with the same
                   area. Rotation-invariant, the standard single-number size of
                   an irregular region (equivalent diameter, as in scikit-image's
                   regionprops). PRIMARY size measure.
    feret_max_km   the longest extent over 18 directions 10 deg apart (the
    feret_min_km   maximum Feret / caliper diameter) and the shortest; their
                   ratio is the elongation
    span_ew_km     extents along the parallel and along the meridian -- the two
    span_ns_km     Feret directions at 0 and 90 deg
    extent_wind_km the extent along the component's own MEAN 10 m WIND. This is
                   the chord Taylor's hypothesis estimates, and so the quantity
                   that can be compared with method 1 like for like.

Extents are measured on a local tangent-plane grid centred on the domain,
x = R cos(phi) (lam - lam0), y = R (phi - phi0), so a zonal separation is scaled
by the cosine of ITS OWN row's latitude. Every extent adds the footprint of one
cell along the measured direction, so a single cell has a non-zero width equal
to its own size in that direction rather than zero.

A component that touches the edge of the domain may continue beyond it; it is
flagged ``touches_edge`` -- the spatial analogue of a censored run -- and kept.

CONNECTIVITY
------------
Two flagged cells sharing only a corner are joined under 8-connectivity and
separated under 4-connectivity. The default here is 8: at 0.25 deg the fields
are smooth and a corner contact almost always sits inside one cloud system. The
choice is a parameter and the notebook reports both.

WHAT IS AND IS NOT COMPARABLE BETWEEN THE TWO METHODS
=====================================================
The sampling units differ. Method 1 yields ONE sample per cell-event; method 2
yields one sample per component per HOUR, so a cloud that persists for twelve
hours is counted twelve times. A per-event Taylor length weighted by its
duration is the like-for-like companion to the snapshot histogram, and
:func:`fig_method_comparison` draws both. The cleanest pairing is at a single
cell: for each Taylor event at the ARM cell, the mean over its hours of the
along-wind extent of the component that CONTAINED the ARM cell in each hour.
That is a one-to-one comparison and is drawn as a scatter.

The two also fail in different ways. Method 1 cannot see anything shorter than
one hour of advection (U x 1 h, about 18 km at 5 m/s -- the same order as the
cell size) and stretches the frozen-field assumption hard for multi-day runs;
method 2 cannot resolve anything smaller than one cell, whose footprint is
strongly anisotropic here -- 27.8 km north-south against 7-9 km east-west -- and
is limited to the 1,100 km by 300-570 km domain.

THE GRID CELL IS NOT SQUARE
---------------------------
0.25 deg of latitude is 27.8 km everywhere; 0.25 deg of longitude is 8.9 km at
71.3 N and 4.8 km at 80 N. A single-cell component therefore has d_eq of about
17.7 km at the ARM latitude, an east-west span of 8.9 km and a north-south span
of 27.8 km. Every histogram below carries these three lengths as reference lines
so the resolution floor is visible rather than mistaken for physics.
"""

from __future__ import annotations

import calendar
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

import plot_lwp_histogram_by_surface_class as lwph
from plot_lwp_histogram_by_surface_class import (
    GENIE_LIQUID_COLOR,
    LIQUID_VAR_LABEL,
    PRECIP_SOURCE_VARS,
    excluded_month_mask,
    iter_time_blocks,
    parse_utc_hours,
    phase_masks,
    precip_label,
    precip_mask,
    resolve_phase_thresholds,
    season_layout,
    season_month_axis,
    select_seasons,
    site_cell_mask,
    threshold_box_lines,
)
from map_liquid_hours import (
    _draw_one,
    _fonts,
    _projection,
    _site_legend,
    _subtitle,
    _wrap,
    precip_banner,
    precip_suffix,
    projected_aspect,
)
from seb_analysis_common import load_seb_data, resolve_region_dir

# Mean Earth radius [km], IUGG value. Used for every length on the sphere.
R_EARTH_KM = 6371.0
SECONDS_PER_HOUR = 3600.0
M_PER_KM = 1000.0

# Hours in one ERA5 time step; run lengths are index differences times this.
STEP_H = lwph.HOURS_PER_STEP

# The phases whose union is "liquid containing", matching season_phase_binary
# and map_liquid_hours.
LIQUID_CONTAINING: tuple[str, ...] = ("liquid", "mixed")

# Colours. The domain-wide ERA5 series keeps the red that every ERA5
# liquid-containing figure in the project uses; the ARM cell is the dark blue
# of the tower notebook's 40 m series so the two notebooks read together.
DOMAIN_COLOR = GENIE_LIQUID_COLOR
SITE_COLOR = "#12395E"
# Snapshot-method series: one fixed hue per size measure, never cycled.
SNAPSHOT_COLORS = {
    "d_eq_km": "#6a3d9a",  # purple  -- equivalent diameter
    "extent_wind_km": "#e6550d",  # orange  -- along the wind
    "feret_max_km": "#1f5fa8",  # blue    -- longest caliper
    "feret_min_km": "#7fa8ce",  # light blue -- shortest caliper
}
SNAPSHOT_LABELS = {
    "d_eq_km": "equivalent diameter $2\\sqrt{A/\\pi}$",
    "extent_wind_km": "extent along the mean 10 m wind",
    "feret_max_km": "longest extent (max Feret)",
    "feret_min_km": "shortest extent (min Feret)",
}
# Single-hue sequential maps for the magnitude maps: one hue, light to dark.
DURATION_CMAP = "Greys"
WIND_CMAP = "Greens"
LENGTH_CMAP = "Purples"


class ExtentAnalysis(SimpleNamespace):
    """The hourly liquid-containing mask and 10 m wind on the native grid.

    Built by :func:`prepare_extent`. Holds, for every kept hour of the
    selected seasons: ``liq`` (bool, time x lat x lon), ``u10`` and ``v10``
    (float32, same shape, m s-1), the timestamps, season and month index of
    every hour, the contiguity flag between consecutive hours, and the grid
    geometry from :func:`grid_geometry`.
    """


# ----------------------------------------------------------------------------
# Grid geometry
# ----------------------------------------------------------------------------
def grid_geometry(lat_deg: np.ndarray, lon_deg: np.ndarray) -> dict:
    """Cell sizes, areas and local tangent-plane coordinates of the grid.

    For a regular lat/lon grid with spacing ``dphi`` x ``dlam`` (radians), the
    exact area of the cell centred on latitude ``phi`` is

        A = R^2 * dlam * ( sin(phi + dphi/2) - sin(phi - dphi/2) )

    (the zonal band between two parallels, cut by two meridians). Its east-west
    width at the centre is ``R cos(phi) dlam`` and its north-south height
    ``R dphi``; the height is the same in every row, the width is not.

    The local coordinates are

        x = R cos(phi) (lam - lam0),   y = R (phi - phi0)

    with (phi0, lam0) the domain centre. This is the equirectangular
    approximation applied row by row: separations along a parallel and along a
    meridian are exact; an oblique separation between two cells in different
    rows is approximated with each cell's own cosine, which over a 10 deg by
    15 deg domain at 70-80 N differs from the great-circle distance by well
    under a percent for the spans that matter here (a few cells) and by a few
    percent for a component spanning the whole domain. Neither is worth a
    projection library inside a loop over 50,000 hours.

    Returns lengths in km and areas in km^2.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    lon = ((lon + 180.0) % 360.0) - 180.0  # signed, -180..180
    dphi = np.deg2rad(abs(float(np.median(np.diff(lat)))))
    dlam = np.deg2rad(abs(float(np.median(np.diff(lon)))))
    phi = np.deg2rad(lat)

    dy_km = R_EARTH_KM * dphi  # scalar
    dx_km = R_EARTH_KM * np.cos(phi) * dlam  # (n_y,)
    area_row_km2 = (
        R_EARTH_KM**2 * dlam * np.abs(np.sin(phi + dphi / 2) - np.sin(phi - dphi / 2))
    )
    n_x = lon.size
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    y_km = R_EARTH_KM * np.deg2rad(lat - lat0)  # (n_y,)
    x_km = (
        R_EARTH_KM * np.cos(phi)[:, None] * np.deg2rad(lon - lon0)[None, :]
    )  # (n_y, n_x)
    return {
        "lat_deg": lat,
        "lon_deg": lon,
        "lat0_deg": lat0,
        "lon0_deg": lon0,
        "dphi_deg": float(np.rad2deg(dphi)),
        "dlam_deg": float(np.rad2deg(dlam)),
        "dy_km": float(dy_km),
        "dx_km": dx_km,
        "area_km2": np.repeat(area_row_km2[:, None], n_x, axis=1),
        "x_km": x_km,
        "y_km": y_km,
    }


# ----------------------------------------------------------------------------
# One pass over the archive
# ----------------------------------------------------------------------------
def prepare_extent(argv=None, args=None, **overrides) -> ExtentAnalysis:
    """Stream the archive once and keep the hourly mask and 10 m wind.

    Same options as ``plot_lwp_histogram_by_surface_class.prepare`` (the parser
    is shared), so ``prepare_extent(region="barrow", years=(2019,),
    no_precip=True, ...)`` selects exactly the population the other notebooks
    analyse.

    Memory: the mask is one byte per cell-hour and each wind component four,
    so eleven Oct-Mar seasons of the 41 x 61 Barrow grid (about 48,000 hours)
    cost roughly 0.12 + 2 x 0.48 = 1.1 GB. That is deliberate: both methods
    need the full time axis at every cell -- a run length cannot be recovered
    from any per-season accumulator, and a component needs the whole field at
    one hour -- and holding it once beats re-reading the archive per method.
    """
    if args is None:
        args = lwph.parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)
    if parse_utc_hours(getattr(args, "utc_hours", None)):
        raise ValueError(
            "--utc-hours leaves gaps in the hourly record, so consecutive "
            "steps are no longer consecutive HOURS and a run length is not a "
            "duration. Re-run without it."
        )

    print("=" * 72)
    print("Horizontal extent of liquid-containing cloud, ERA5")
    print("=" * 72)
    phase_kw = resolve_phase_thresholds(args)
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)

    read_vars = ["tcc", "tciw", args.liquid_var, "u10", "v10"]
    if args.no_precip:
        read_vars += [
            v for v in PRECIP_SOURCE_VARS[args.precip_var] if v not in read_vars
        ]
    missing = sorted(set(read_vars) - set(ds.data_vars))
    if missing:
        raise KeyError(f"dataset is missing {missing}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    s_idx, in_window, dos = layout["s_idx"], layout["in_window"], layout["dos"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    if not use_step.any():
        raise ValueError("no time steps left after the season filter")
    # layout["seasons"] lists every season in the archive; remap onto the
    # SELECTED ones so season index 0 is used[0], as every sibling module does.
    remap = np.full(len(layout["seasons"]), -1, dtype=np.intp)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.intp)
    months, mi_of_slot = season_month_axis(layout["slots"])

    n_t = int(use_step.sum())
    n_y, n_x = ds.sizes["latitude"], ds.sizes["longitude"]
    print(f"  Source     : {region_dir}")
    print(f"  Grid       : {n_y} x {n_x} cells")
    print(
        f"  Cloudy     : tcc >= {args.min_cloud_fraction:g}   |   "
        f"{LIQUID_VAR_LABEL[args.liquid_var]}   |   {precip_label(args)}"
    )
    print(f"  Seasons    : {used}")
    print(
        f"  Hours kept : {n_t:,}   ->   mask {n_t * n_y * n_x / 1e6:.0f} MB, "
        f"wind 2 x {4 * n_t * n_y * n_x / 1e6:.0f} MB"
    )

    liq = np.zeros((n_t, n_y, n_x), dtype=bool)
    u10 = np.empty((n_t, n_y, n_x), dtype=np.float32)
    v10 = np.empty((n_t, n_y, n_x), dtype=np.float32)
    times = np.empty(n_t, dtype="datetime64[ns]")
    s_of = np.empty(n_t, dtype=np.intp)
    mi = np.empty(n_t, dtype=np.intp)
    n_valid = np.zeros((n_y, n_x), dtype=np.int64)

    times_all = np.asarray(ds["valid_time"].values)
    t0 = time.time()
    pos = 0
    for i0, block in iter_time_blocks(
        ds, read_vars, args.block_hours, keep_mask=use_step
    ):
        n_b = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_b)
        keep = use_step[sl]
        if not keep.any():
            continue
        k = int(keep.sum())
        tcc = block["tcc"].values[keep]
        lwp_g = block[args.liquid_var].values[keep] * 1000.0  # kg m-2 -> g m-2
        iwp_g = block["tciw"].values[keep] * 1000.0
        valid = np.isfinite(tcc) & np.isfinite(lwp_g) & np.isfinite(iwp_g)
        # The same masking, in the same order, as map_liquid_hours: a raining
        # scene leaves the CLOUDY population, and the phase masks partition
        # what remains.
        raining = precip_mask(block, keep, args)
        cloudy = valid & (tcc >= args.min_cloud_fraction) & ~raining
        ph = phase_masks(lwp_g, iwp_g, phase_kw)
        liq[pos : pos + k] = cloudy & (ph["liquid"] | ph["mixed"])
        u10[pos : pos + k] = block["u10"].values[keep]
        v10[pos : pos + k] = block["v10"].values[keep]
        times[pos : pos + k] = times_all[sl][keep]
        s_of[pos : pos + k] = remap[s_idx[sl][keep]]
        mi[pos : pos + k] = mi_of_slot[dos[sl][keep]]
        n_valid += valid.sum(axis=0)
        pos += k
    if pos != n_t:
        raise AssertionError(f"filled {pos} of {n_t} hours")
    if (s_of < 0).any():
        raise AssertionError("an hour outside the selected seasons was kept")
    print(f"  Read in {time.time() - t0:.0f} s")

    # Time order is what the run detection relies on. load_seb_data already
    # concatenates the files chronologically; this guards the assumption
    # rather than trusting it.
    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        order = np.argsort(times, kind="stable")
        liq, u10, v10 = liq[order], u10[order], v10[order]
        times, s_of, mi = times[order], s_of[order], mi[order]
        print("  (time axis re-sorted)")

    n_wind_nan = int(np.count_nonzero(~np.isfinite(u10) | ~np.isfinite(v10)))
    if n_wind_nan:
        print(
            f"  !! {n_wind_nan:,} cell-hours with a non-finite 10 m wind",
            file=sys.stderr,
        )

    # contig[i]: hour i follows hour i-1 with no gap and in the same season.
    step_h = np.diff(times).astype("timedelta64[s]").astype(np.float64) / 3600.0
    contig = np.zeros(n_t, dtype=bool)
    contig[1:] = np.isclose(step_h, STEP_H) & (s_of[1:] == s_of[:-1])
    n_holes = int(
        np.count_nonzero((s_of[1:] == s_of[:-1]) & ~np.isclose(step_h, STEP_H))
    )

    # Liquid-containing hours per cell and season, the tie to map_liquid_hours:
    # this must equal counts["liquid"] + counts["mixed"] there exactly.
    n_liq_season = np.zeros((len(keep_idx), n_y, n_x), dtype=np.int64)
    np.add.at(n_liq_season, s_of, liq)

    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    site_mask, site_lat, site_lon = site_cell_mask(ds)
    geom = grid_geometry(lat, lon)
    iy, ix = (int(v) for v in np.argwhere(site_mask)[0])
    print(
        f"  ARM cell   : ({site_lat:.2f} N, {site_lon:.2f} E), "
        f"{geom['dx_km'][iy]:.1f} km E-W x {geom['dy_km']:.1f} km N-S, "
        f"d_eq of one cell {2 * np.sqrt(geom['area_km2'][iy, ix] / np.pi):.1f} km"
    )
    print(
        f"  Liquid-containing cell-hours: {int(liq.sum()):,} of "
        f"{n_t * n_y * n_x:,} ({100 * liq.mean():.1f}%)   |   holes in the "
        f"hourly record: {n_holes}"
    )

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    return ExtentAnalysis(
        args=args,
        ds=ds,
        layout=layout,
        keep_idx=keep_idx,
        used=list(used),
        mode_label=mode_label,
        phase_kw=phase_kw,
        tag=tag,
        liq=liq,
        u10=u10,
        v10=v10,
        times=times,
        s_of=s_of,
        mi=mi,
        contig=contig,
        n_holes=n_holes,
        months=list(months),
        n_valid=n_valid,
        n_liq_season=n_liq_season,
        lat=lat,
        lon=geom["lon_deg"],
        geom=geom,
        site_mask=site_mask,
        site_lat=site_lat,
        site_lon=site_lon,
        site_iy=iy,
        site_ix=ix,
    )


# ----------------------------------------------------------------------------
# Method 1: Taylor, cell by cell
# ----------------------------------------------------------------------------
def taylor_events(E: ExtentAnalysis, exclude_months=(), cell_chunk: int = 256,
                  wind: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Every liquid-containing event at every cell, with its Taylor length.

    ``wind`` substitutes another advection wind for the 10 m one: a pair of
    ``(u, v)`` arrays shaped like ``E.liq`` in m s-1, NaN where undefined --
    ``cloud_level_wind.cloud_level_wind`` supplies the liquid-weighted wind at
    cloud level this way. The events themselves are unchanged (they come from
    the mask alone); only the means are taken over the event's hours that
    HAVE a wind, and an event with none gets NaN. ``n_wind_h`` records how
    many of each event's hours contributed.

    Run detection is ``liquid_cloud_runs`` applied to all cells at once: a run
    starts at a liquid hour whose predecessor is not liquid or not contiguous,
    and ends at a liquid hour whose successor is not liquid or not contiguous.
    Cells are processed in chunks, laid out cell-major, so that one
    ``np.add.reduceat`` per quantity sums each run's hours: the summand is
    zero outside liquid hours, so the sum from one run start to the next is
    exactly the sum over that run, and the last run of a cell runs into the
    zeros before the first run of the next cell.

    Returns a dict of parallel arrays over events:

    ``iy, ix``          cell indices;  ``lat, lon``  cell centre [deg]
    ``t_start, t_end``  indices into ``E.times`` (inclusive)
    ``length_h``        duration [h]
    ``U_mean_m_s``      mean of the 10 m wind SPEED over the event, eq. (2)
    ``U_vec_m_s``       magnitude of the mean 10 m wind VECTOR, eq. (3)
    ``u_mean_m_s, v_mean_m_s``  the vector-mean components
    ``L_km``            U_mean_m_s x length_h, eq. (1) with the scalar mean
    ``L_vec_km``        the same with the vector mean
    ``censored``        the run touches a break, so length_h is a lower bound
    ``month, season``   the calendar month and season (start year) it BEGAN in
    ``is_site``         the event is at the ARM cell

    ``exclude_months`` takes calendar (year, month) pairs as
    ``GENIE_EXCLUDED_MONTHS`` does; events starting in one are dropped and
    counted in ``n_dropped``. Default: nothing excluded, since the ARM
    instrument outages that list records have no bearing on ERA5 over the
    whole domain.
    """
    n_t, n_y, n_x = E.liq.shape
    n_cell = n_y * n_x
    liq2 = E.liq.reshape(n_t, n_cell)
    if wind is None:
        u_src, v_src = E.u10, E.v10
    else:
        u_src, v_src = wind
        if u_src.shape != E.liq.shape or v_src.shape != E.liq.shape:
            raise ValueError("wind arrays must be shaped like E.liq")
    u2 = u_src.reshape(n_t, n_cell)
    v2 = v_src.reshape(n_t, n_cell)
    contig = E.contig
    next_contig = np.append(contig[1:], False)

    parts = defaultdict(list)
    for c0 in range(0, n_cell, cell_chunk):
        c1 = min(c0 + cell_chunk, n_cell)
        L = np.ascontiguousarray(liq2[:, c0:c1].T)  # (k, n_t)
        prev = np.zeros_like(L)
        prev[:, 1:] = L[:, :-1]
        nxt = np.zeros_like(L)
        nxt[:, :-1] = L[:, 1:]
        start = L & ~(prev & contig[None, :])
        end = L & ~(nxt & next_contig[None, :])
        s_c, s_t = np.nonzero(start)  # row-major: by cell, then time
        e_c, e_t = np.nonzero(end)
        if s_c.size == 0:
            continue
        if s_c.size != e_c.size or not np.array_equal(s_c, e_c) or np.any(e_t < s_t):
            raise AssertionError("unbalanced run starts and ends")
        length = e_t - s_t + 1

        Lf = L.astype(np.float64)
        uT = np.ascontiguousarray(u2[:, c0:c1].T).astype(np.float64)
        vT = np.ascontiguousarray(v2[:, c0:c1].T).astype(np.float64)
        # Hours that count towards the wind mean: liquid AND carrying a wind.
        # With the 10 m wind that is every liquid hour; a substitute wind may
        # have gaps, which are excluded from the mean rather than zeroed into it.
        Wf = Lf * (np.isfinite(uT) & np.isfinite(vT))
        uT = np.nan_to_num(uT) * Wf
        vT = np.nan_to_num(vT) * Wf
        speedT = np.hypot(uT, vT)  # zero outside counted hours
        flat_starts = s_c * n_t + s_t
        hours = np.add.reduceat(Lf.ravel(), flat_starts)
        if not np.allclose(hours, length):
            raise AssertionError("reduceat did not reproduce the run lengths")
        parts["n_wind_h"].append(np.add.reduceat(Wf.ravel(), flat_starts))
        parts["sum_speed"].append(np.add.reduceat(speedT.ravel(), flat_starts))
        parts["sum_u"].append(np.add.reduceat(uT.ravel(), flat_starts))
        parts["sum_v"].append(np.add.reduceat(vT.ravel(), flat_starts))
        parts["length_h"].append(length.astype(float) * STEP_H)
        parts["t_start"].append(s_t)
        parts["t_end"].append(e_t)
        parts["censored"].append(~contig[s_t] | ~next_contig[e_t])
        cell = c0 + s_c
        parts["iy"].append(cell // n_x)
        parts["ix"].append(cell % n_x)

    ev = {k: np.concatenate(v) for k, v in parts.items()}
    n_h = ev["length_h"] / STEP_H
    total_liq = int(E.liq.sum())
    if int(n_h.sum()) != total_liq:
        raise AssertionError(
            f"hours not conserved: events hold "
            f"{int(n_h.sum()):,} h, mask holds {total_liq:,}"
        )
    # Means over the hours that carried a wind; NaN where none did.
    n_w = ev["n_wind_h"]
    den = np.where(n_w > 0, n_w, np.nan)
    ev["U_mean_m_s"] = ev.pop("sum_speed") / den
    ev["u_mean_m_s"] = ev.pop("sum_u") / den
    ev["v_mean_m_s"] = ev.pop("sum_v") / den
    ev["U_vec_m_s"] = np.hypot(ev["u_mean_m_s"], ev["v_mean_m_s"])
    # Eq. (1): m s-1 x h x 3600 s h-1 / 1000 m km-1.
    ev["L_km"] = ev["U_mean_m_s"] * ev["length_h"] * SECONDS_PER_HOUR / M_PER_KM
    ev["L_vec_km"] = ev["U_vec_m_s"] * ev["length_h"] * SECONDS_PER_HOUR / M_PER_KM
    ev["lat"] = E.lat[ev["iy"]]
    ev["lon"] = E.lon[ev["ix"]]
    ev["month"] = np.asarray(E.months)[E.mi[ev["t_start"]]]
    ev["season"] = np.asarray(E.used)[E.s_of[ev["t_start"]]]
    ev["is_site"] = E.site_mask[ev["iy"], ev["ix"]]
    ev["start"] = E.times[ev["t_start"]]

    keep = np.ones(ev["L_km"].size, dtype=bool)
    if exclude_months:
        drop, _hit = excluded_month_mask(E.used, E.months, E.args, exclude_months)
        keep &= ~drop[E.s_of[ev["t_start"]], E.mi[ev["t_start"]]]
    n_dropped = int((~keep).sum())
    ev = {k: v[keep] for k, v in ev.items()}
    ev["n_dropped"] = n_dropped
    ev["exclude_months"] = tuple(exclude_months)
    ev["n_hours_total"] = total_liq
    return ev


def per_cell_stat(
    E: ExtentAnalysis, ev: dict, key: str, stat: str = "median"
) -> np.ndarray:
    """A (lat, lon) map of one event quantity, summarised per cell.

    NaN where a cell had no events. ``stat`` is "median", "mean" or "count".
    """
    n_y, n_x = E.liq.shape[1:]
    cell = ev["iy"] * n_x + ev["ix"]
    out = np.full(n_y * n_x, np.nan)
    if stat == "count":
        out[:] = np.bincount(cell, minlength=n_y * n_x)
        return out.reshape(n_y, n_x)
    order = np.argsort(cell, kind="stable")
    cell_s, val_s = cell[order], ev[key][order]
    bounds = np.flatnonzero(np.diff(cell_s)) + 1
    for c, chunk in zip(cell_s[np.r_[0, bounds]], np.split(val_s, bounds)):
        out[c] = np.median(chunk) if stat == "median" else np.mean(chunk)
    return out.reshape(n_y, n_x)


# ----------------------------------------------------------------------------
# Method 2: connected components, hour by hour
# ----------------------------------------------------------------------------
DEFAULT_SIZE_BINS_KM = np.geomspace(1.0, 3000.0, 121)  # for the per-cell maps


def snapshot_components(
    E: ExtentAnalysis,
    connectivity: int = 8,
    n_directions: int = 18,
    chunk_hours: int = 1024,
    size_bins_km: np.ndarray = DEFAULT_SIZE_BINS_KM,
) -> dict:
    """Label the liquid-containing field at every hour and size each component.

    Labelling is done with ``scipy.ndimage.label`` on a (time, lat, lon) block
    under a 3-D structuring element whose time planes are EMPTY, so
    components never join across hours -- a block of a thousand snapshots is
    labelled in one call and every statistic below is a ``reduceat`` over the
    entries sorted by label. Nothing loops over components in Python.

    Per component (see the module docstring for the definitions):
    ``t`` (index into ``E.times``), ``season``, ``month``, ``n_cells``,
    ``area_km2``, ``d_eq_km``, ``span_ew_km``, ``span_ns_km``,
    ``feret_max_km``, ``feret_min_km``, ``feret_max_dir_deg``,
    ``extent_wind_km``, ``U_mean_m_s``, ``U_vec_m_s``, ``u_mean_m_s``,
    ``v_mean_m_s``, ``lat_c``, ``lon_c`` (centroid), ``touches_edge``,
    ``has_site``.

    Also returned:
    ``site_extent_wind_km``, ``site_d_eq_km``  (n_t,) the size of the
        component holding the ARM cell at each hour, NaN when it holds none
    ``cell_hist_d_eq``, ``cell_hist_wind``  (n_cell, n_bins) histograms over
        ``size_bins_km`` of the size of the component each cell belonged to,
        one count per cell-hour, for the per-cell median maps
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    n_t, n_y, n_x = E.liq.shape
    n_cell = n_y * n_x
    g = E.geom
    # 3-D structure with empty time planes: 2-D connectivity only.
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1] = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)

    # Per-cell geometry, flattened in the same (lat, lon) raster order as the
    # mask, so an entry's cell index addresses them directly.
    x_flat = g["x_km"].ravel()
    y_flat = np.repeat(g["y_km"], n_x)
    dx_flat = np.repeat(g["dx_km"], n_x)
    dy_km = g["dy_km"]
    area_flat = g["area_km2"].ravel()
    lat_flat = np.repeat(E.lat, n_x)
    lon_flat = np.tile(E.lon, n_y)
    yy, xx = np.divmod(np.arange(n_cell), n_x)
    edge_flat = ((yy == 0) | (yy == n_y - 1) | (xx == 0) | (xx == n_x - 1)).astype(
        np.int8
    )
    site_flat = E.site_mask.ravel().astype(np.int8)

    theta = np.deg2rad(np.arange(n_directions) * 180.0 / n_directions)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    k_ew = int(np.argmin(np.abs(theta)))  # 0 deg
    k_ns = int(np.argmin(np.abs(theta - np.pi / 2)))  # 90 deg
    if not (np.isclose(theta[k_ew], 0) and np.isclose(theta[k_ns], np.pi / 2)):
        raise ValueError(
            "n_directions must put a direction at 0 and 90 deg "
            "(any even number does)"
        )

    bins = np.asarray(size_bins_km, dtype=float)
    n_bins = bins.size - 1
    cell_hist_d_eq = np.zeros((n_cell, n_bins), dtype=np.int64)
    cell_hist_wind = np.zeros((n_cell, n_bins), dtype=np.int64)
    site_ext_wind = np.full(n_t, np.nan)
    site_d_eq = np.full(n_t, np.nan)

    parts = defaultdict(list)
    t0 = time.time()
    for c0 in range(0, n_t, chunk_hours):
        c1 = min(c0 + chunk_hours, n_t)
        lab, n = ndimage.label(E.liq[c0:c1], structure=st)
        if n == 0:
            continue
        flat_lab = lab.ravel()
        nz = np.flatnonzero(flat_lab)
        labs = flat_lab[nz]
        order = np.argsort(labs, kind="stable")
        nz, labs = nz[order], labs[order]
        starts = np.r_[0, np.flatnonzero(np.diff(labs)) + 1]
        if starts.size != n or not np.array_equal(labs[starts], np.arange(1, n + 1)):
            raise AssertionError("label bookkeeping broke")
        n_cells = np.diff(np.append(starts, labs.size))
        t_loc, cell = np.divmod(nz, n_cell)

        def ssum(a):
            return np.add.reduceat(a, starts)

        def smax(a):
            return np.maximum.reduceat(a, starts)

        def smin(a):
            return np.minimum.reduceat(a, starts)

        t_comp = smin(t_loc)
        if not np.array_equal(t_comp, smax(t_loc)):
            raise AssertionError("a component spans more than one hour")
        t_comp = c0 + t_comp

        area = ssum(area_flat[cell])
        u_e = E.u10[c0:c1].ravel()[nz].astype(np.float64)
        v_e = E.v10[c0:c1].ravel()[nz].astype(np.float64)
        u_sum, v_sum = ssum(u_e), ssum(v_e)
        speed_sum = ssum(np.hypot(u_e, v_e))
        x_e, y_e = x_flat[cell], y_flat[cell]
        dx_mean = ssum(dx_flat[cell]) / n_cells

        # Feret extents: the projected span of the cell centres plus the
        # footprint of one cell along that direction.
        spans = np.empty((n, n_directions))
        for k in range(n_directions):
            p = x_e * cos_t[k] + y_e * sin_t[k]
            spans[:, k] = (
                smax(p) - smin(p) + dx_mean * abs(cos_t[k]) + dy_km * abs(sin_t[k])
            )
        # Extent along each component's own mean wind: a per-entry projection
        # onto its component's wind axis, then the same span.
        theta_w = np.arctan2(v_sum, u_sum)
        comp_of_entry = np.repeat(np.arange(n), n_cells)
        cw, sw = np.cos(theta_w), np.sin(theta_w)
        p_w = x_e * cw[comp_of_entry] + y_e * sw[comp_of_entry]
        extent_wind = smax(p_w) - smin(p_w) + dx_mean * np.abs(cw) + dy_km * np.abs(sw)

        d_eq = 2.0 * np.sqrt(area / np.pi)
        parts["t"].append(t_comp)
        parts["n_cells"].append(n_cells)
        parts["area_km2"].append(area)
        parts["d_eq_km"].append(d_eq)
        parts["span_ew_km"].append(spans[:, k_ew])
        parts["span_ns_km"].append(spans[:, k_ns])
        parts["feret_max_km"].append(spans.max(axis=1))
        parts["feret_min_km"].append(spans.min(axis=1))
        parts["feret_max_dir_deg"].append(np.rad2deg(theta[spans.argmax(axis=1)]))
        parts["extent_wind_km"].append(extent_wind)
        parts["U_mean_m_s"].append(speed_sum / n_cells)
        parts["u_mean_m_s"].append(u_sum / n_cells)
        parts["v_mean_m_s"].append(v_sum / n_cells)
        parts["lat_c"].append(ssum(lat_flat[cell]) / n_cells)
        parts["lon_c"].append(ssum(lon_flat[cell]) / n_cells)
        parts["touches_edge"].append(smax(edge_flat[cell]) > 0)
        has_site = smax(site_flat[cell]) > 0
        parts["has_site"].append(has_site)
        site_ext_wind[t_comp[has_site]] = extent_wind[has_site]
        site_d_eq[t_comp[has_site]] = d_eq[has_site]

        # Per-cell histograms of the size of the component the cell is in.
        for hist, val in ((cell_hist_d_eq, d_eq), (cell_hist_wind, extent_wind)):
            b = np.clip(
                np.searchsorted(bins, val[comp_of_entry], side="right") - 1,
                0,
                n_bins - 1,
            )
            hist += np.bincount(cell * n_bins + b, minlength=n_cell * n_bins).reshape(
                n_cell, n_bins
            )

    comp = {k: np.concatenate(v) for k, v in parts.items()}
    comp["U_vec_m_s"] = np.hypot(comp["u_mean_m_s"], comp["v_mean_m_s"])
    comp["season"] = np.asarray(E.used)[E.s_of[comp["t"]]]
    comp["month"] = np.asarray(E.months)[E.mi[comp["t"]]]
    comp["time"] = E.times[comp["t"]]
    # Cell-hours are conserved: every liquid cell-hour is in exactly one
    # component.
    if int(comp["n_cells"].sum()) != int(E.liq.sum()):
        raise AssertionError("cell-hours not conserved by the labelling")
    comp["connectivity"] = connectivity
    comp["directions_deg"] = np.rad2deg(theta)
    comp["site_extent_wind_km"] = site_ext_wind
    comp["site_d_eq_km"] = site_d_eq
    comp["size_bins_km"] = bins
    comp["cell_hist_d_eq"] = cell_hist_d_eq
    comp["cell_hist_wind"] = cell_hist_wind
    print(
        f"  {comp['t'].size:,} component-hours over {n_t:,} hours, "
        f"{connectivity}-connectivity, {time.time() - t0:.0f} s"
    )
    return comp


def cell_median_from_hist(
    hist: np.ndarray, bins: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Per-cell median from a per-cell histogram, to bin resolution.

    The bins are log-spaced at 120 per three decades, so the median is known
    to within about 6%. NaN where a cell has no counts.
    """
    total = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    half = np.where(total > 0, total, 1)[:, None] / 2.0
    k = np.argmax(cum >= half, axis=1)
    centre = np.sqrt(bins[:-1] * bins[1:])
    med = np.where(total > 0, centre[k], np.nan)
    return med.reshape(shape)


def site_event_pairs(E: ExtentAnalysis, ev: dict, comp: dict) -> dict:
    """Pair each Taylor event at the ARM cell with the snapshot extents.

    For each event the snapshot extent is the MEAN over the event's hours of
    the along-wind extent (and the equivalent diameter) of the component that
    contained the ARM cell in that hour. Every one of those hours has such a
    component by construction -- the cell is liquid-containing throughout the
    event -- so no pairing is ever missing.
    """
    sel = ev["is_site"]
    s, e = ev["t_start"][sel], ev["t_end"][sel]
    cs_w = np.r_[0.0, np.nancumsum(comp["site_extent_wind_km"])]
    cs_d = np.r_[0.0, np.nancumsum(comp["site_d_eq_km"])]
    n = (e - s + 1).astype(float)
    if np.any(~np.isfinite(comp["site_extent_wind_km"][s])):
        raise AssertionError("an ARM-cell event hour has no component")
    return {
        "L_km": ev["L_km"][sel],
        "L_vec_km": ev["L_vec_km"][sel],
        "length_h": ev["length_h"][sel],
        "censored": ev["censored"][sel],
        "month": ev["month"][sel],
        "snapshot_extent_wind_km": (cs_w[e + 1] - cs_w[s]) / n,
        "snapshot_d_eq_km": (cs_d[e + 1] - cs_d[s]) / n,
    }


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def _q(a, p):
    return np.percentile(a, p) if a.size else np.nan


def _weighted_median(a: np.ndarray, w: np.ndarray) -> float:
    """Median of ``a`` with weights ``w``: the value at half the total weight."""
    order = np.argsort(a)
    return float(a[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)])


def print_taylor_report(E: ExtentAnalysis, ev: dict) -> None:
    """Domain-wide and ARM-cell summaries of the Taylor lengths, by month."""
    args = E.args
    print(f"Method 1, Taylor: L = U(10 m) x dt   |   {precip_label(args)}")
    print(
        f"  {ev['L_km'].size:,} cell-events over {E.liq.shape[1] * E.liq.shape[2]:,} "
        f"cells, {ev['n_hours_total']:,} liquid-containing cell-hours, "
        f"{int(ev['censored'].sum()):,} censored"
        + (
            f", {ev['n_dropped']:,} dropped by month exclusion"
            if ev["n_dropped"]
            else ""
        )
    )
    for label, sel in (
        ("all cells", np.ones(ev["L_km"].size, bool)),
        ("ARM cell", ev["is_site"]),
    ):
        d, U, L = ev["length_h"][sel], ev["U_mean_m_s"][sel], ev["L_km"][sel]
        print(f"\n  {label}: {L.size:,} events")
        print(f"    {'':<8}{'median':>9}{'q25':>9}{'q75':>9}{'mean':>9}{'max':>9}")
        for name, a, unit in (
            ("dt", d, "h"),
            ("U", U, "m/s"),
            ("L", L, "km"),
            ("L_vec", ev["L_vec_km"][sel], "km"),
        ):
            print(
                f"    {name:<8}{np.median(a):>9.1f}{_q(a, 25):>9.1f}"
                f"{_q(a, 75):>9.1f}{a.mean():>9.1f}{a.max():>9.1f}   [{unit}]"
            )
        print(
            f"    1-h events {100 * np.mean(d == 1):.0f}%   |   events >= 24 h "
            f"{100 * np.mean(d >= 24):.1f}%   |   share of hours in events "
            f">= 24 h {100 * d[d >= 24].sum() / d.sum():.0f}%"
        )
    print(f"\n  by starting month, all cells   [km unless noted]")
    print(
        f"    {'month':<7}{'events':>9}{'dt med':>8}{'U med':>8}{'L med':>8}"
        f"{'L q25':>8}{'L q75':>8}{'L p90':>8}"
    )
    for m in E.months:
        s = ev["month"] == m
        if not s.any():
            continue
        print(
            f"    {calendar.month_abbr[m]:<7}{int(s.sum()):>9,}"
            f"{np.median(ev['length_h'][s]):>8.0f}"
            f"{np.median(ev['U_mean_m_s'][s]):>8.1f}"
            f"{np.median(ev['L_km'][s]):>8.0f}{_q(ev['L_km'][s], 25):>8.0f}"
            f"{_q(ev['L_km'][s], 75):>8.0f}{_q(ev['L_km'][s], 90):>8.0f}"
        )


def print_snapshot_report(E: ExtentAnalysis, comp: dict) -> None:
    """Component statistics, all and with edge-touching components set aside."""
    print(
        f"Method 2, connected components ({comp['connectivity']}-connectivity)"
        f"   |   {precip_label(E.args)}"
    )
    n = comp["t"].size
    single = comp["n_cells"] == 1
    edge = comp["touches_edge"]
    print(
        f"  {n:,} component-hours   |   single-cell {100 * single.mean():.1f}%   |"
        f"   touching the domain edge {100 * edge.mean():.1f}% of components, "
        f"{100 * comp['n_cells'][edge].sum() / comp['n_cells'].sum():.0f}% of cell-hours"
    )
    print(
        f"  hours with at least one component: "
        f"{np.unique(comp['t']).size:,} of {E.liq.shape[0]:,}   |   "
        f"components per such hour: median {np.median(np.bincount(comp['t'])[np.bincount(comp['t']) > 0]):.0f}"
    )
    for label, sel in (
        ("all components", np.ones(n, bool)),
        ("interior only (not touching the edge)", ~edge),
    ):
        print(f"\n  {label}: {int(sel.sum()):,}")
        print(f"    {'':<16}{'median':>9}{'q25':>9}{'q75':>9}{'p90':>9}{'max':>9}")
        for key in (
            "d_eq_km",
            "extent_wind_km",
            "feret_max_km",
            "feret_min_km",
            "span_ew_km",
            "span_ns_km",
            "n_cells",
            "U_mean_m_s",
        ):
            a = comp[key][sel]
            print(
                f"    {key:<16}{np.median(a):>9.1f}{_q(a, 25):>9.1f}"
                f"{_q(a, 75):>9.1f}{_q(a, 90):>9.1f}{a.max():>9.1f}"
            )
        r = comp["feret_max_km"][sel] / comp["feret_min_km"][sel]
        print(f"    elongation (max/min Feret): median {np.median(r):.2f}")
    sw = comp["site_extent_wind_km"]
    print(
        f"\n  ARM cell: in a component {int(np.isfinite(sw).sum()):,} h; that "
        f"component's along-wind extent median {np.nanmedian(sw):.0f} km, "
        f"d_eq median {np.nanmedian(comp['site_d_eq_km']):.0f} km"
    )


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _save(fig, E, out_dir, stem, dpi):
    if out_dir is None:
        return fig
    path = Path(out_dir) / (
        f"{E.args.region}_{stem}_{precip_suffix(E.args)}_" f"{E.tag}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or E.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def _season_span(E) -> str:
    return (
        f"{E.used[0]}/{(E.used[0] + 1) % 100:02d}–"
        f"{E.used[-1]}/{(E.used[-1] + 1) % 100:02d}"
    )


def _threshold_box(ax, E, loc="upper left", fontsize=8.5):
    place = {
        "upper right": (0.995, 0.995, "right", "top"),
        "upper left": (0.005, 0.995, "left", "top"),
        "lower right": (0.995, 0.005, "right", "bottom"),
    }
    x, y, ha, va = place[loc]
    ax.text(
        x,
        y,
        "\n".join(threshold_box_lines(E)),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        linespacing=1.4,
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.8,
            alpha=0.94,
        ),
    )


def _cell_reference_lines(ax, E, y_frac=0.97, fontsize=8):
    """Vertical lines at the ARM cell's E-W width, d_eq and N-S height."""
    g = E.geom
    dx = float(g["dx_km"][E.site_iy])
    deq = 2 * np.sqrt(float(g["area_km2"][E.site_iy, E.site_ix]) / np.pi)
    for x, lab in (
        (dx, f"cell E–W {dx:.0f} km"),
        (deq, f"cell $d_{{eq}}$ {deq:.0f} km"),
        (g["dy_km"], f"cell N–S {g['dy_km']:.0f} km"),
    ):
        ax.axvline(x, color="0.45", lw=0.8, ls=":", zorder=0)
        ax.text(
            x,
            y_frac,
            lab,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=fontsize,
            color="0.35",
        )


def _site_legend_below(fig, fontsize):
    """The red star, named, centred under the colourbars."""
    from matplotlib.lines import Line2D

    fig.legend(
        [
            Line2D(
                [0], [0], marker="*", ms=12, mfc="red", mec="white", mew=1.0, ls="none"
            )
        ],
        ["Utqia\u0121vik (DOE ARM site)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=fontsize,
        framealpha=0.95,
        handletextpad=0.4,
    )


def _tidy(ax):
    ax.grid(True, alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_wind_speed_pdf(
    E: ExtentAnalysis,
    out_dir=None,
    dpi=None,
    tower_median_m_s: float | None = None,
    tower_label: str = "40 m tower median, 2025/26",
):
    """The 10 m wind speed the Taylor conversion uses, four populations.

    Domain-wide all hours, domain-wide liquid-containing hours, and the same
    two at the ARM cell. The domain histograms are per cell-hour, unweighted
    by area: cells are 7-9 km wide against 28 km tall, and a cos(lat) weight
    changes the shape by less than the line width. ``tower_median_m_s`` draws
    the value the tower notebook used, for scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    speed = np.hypot(E.u10, E.v10)
    iy, ix = E.site_iy, E.site_ix
    pops = [
        ("domain, all hours", speed.ravel(), DOMAIN_COLOR, "--", 1.3),
        ("domain, liquid-containing hours", speed[E.liq], DOMAIN_COLOR, "-", 2.2),
        ("ARM cell, all hours", speed[:, iy, ix], SITE_COLOR, "--", 1.3),
        (
            "ARM cell, liquid-containing hours",
            speed[:, iy, ix][E.liq[:, iy, ix]],
            SITE_COLOR,
            "-",
            2.2,
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    bins = np.arange(0, 26.5, 0.5)
    for lab, a, c, ls, lw in pops:
        a = a[np.isfinite(a)]
        ax.hist(
            a,
            bins=bins,
            density=True,
            histtype="step",
            color=c,
            ls=ls,
            lw=lw,
            label=f"{lab}  (median {np.median(a):.1f}, "
            f"IQR {_q(a, 25):.1f}–{_q(a, 75):.1f} m s$^{{-1}}$)",
        )
    if tower_median_m_s is not None:
        ax.axvline(tower_median_m_s, color="0.3", lw=1.0, ls=":")
        ax.text(
            tower_median_m_s + 0.2,
            0.98,
            tower_label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="top",
            fontsize=8.5,
            color="0.3",
        )
    ax.set_xlim(0, 22)
    ax.set_xlabel("ERA5 10 m wind speed $\\sqrt{u_{10}^2 + v_{10}^2}$ [m s$^{-1}$]")
    ax.set_ylabel("probability density [(m s$^{-1}$)$^{-1}$]")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
    )
    _tidy(ax)
    ax.set_title(
        f"{precip_banner(E.args)}\n10 m wind speed, {E.args.region} region, "
        f"{_season_span(E)}, Oct–Mar hours",
        fontsize=10.5,
    )
    return _save(fig, E, out_dir, "wind10m_pdf", dpi)


def fig_taylor_hist(
    E: ExtentAnalysis,
    ev: dict,
    out_dir=None,
    dpi=None,
    key: str = "L_km",
    weighted: bool = False,
):
    """Histogram of the Taylor length, domain-wide and at the ARM cell.

    Log-spaced bins on a log axis: one-hour events at a few m s-1 give tens of
    km, multi-day runs give thousands. ``weighted=True`` weights every event by
    its duration, i.e. one count per cell-HOUR, which is the sampling unit of
    the snapshot method.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    bins = np.geomspace(1.0, 1e4, 81)
    w_all = ev["length_h"] if weighted else None
    for lab, sel, c, lw in (
        ("all cells", np.ones(ev[key].size, bool), DOMAIN_COLOR, 2.2),
        ("ARM cell", ev["is_site"], SITE_COLOR, 1.6),
    ):
        a = ev[key][sel]
        w = w_all[sel] if weighted else None
        ax.hist(
            a,
            bins=bins,
            weights=w,
            density=True,
            histtype="step",
            color=c,
            lw=lw,
            label=f"{lab}: {a.size:,} events, median {np.median(a):,.0f} km, "
            f"IQR {_q(a, 25):,.0f}–{_q(a, 75):,.0f} km",
        )
    # _cell_reference_lines(ax, E)
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(
        "Taylor horizontal extent $L = U_{10}\\,\\Delta t$ [km]"
        + ("   (vector-mean wind)" if key == "L_vec_km" else "")
    )
    ax.set_ylabel(
        "probability density per event"
        if not weighted
        else "probability density per cell-hour (duration-weighted)"
    )
    ax.legend(
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
    )
    _threshold_box(ax, E, loc="upper right")
    _tidy(ax)
    ax.set_title(
        f"{precip_banner(E.args)}\nMethod 1 — one sample per "
        f"liquid-containing event at each cell, {E.args.region} region, "
        f"{_season_span(E)}",
        fontsize=10.5,
    )
    stem = f"taylor_hist_{key}" + ("_weighted" if weighted else "")
    return _save(fig, E, out_dir, stem, dpi)


def _month_boxes(ax, months, groups_a, groups_b, label_a, label_b, unit):
    """Two boxes per month: dotted red (domain) and solid dark blue (ARM cell).

    Same conventions as the project's 7b figures: IQR box, median line,
    whiskers at the full range, nothing drawn as an outlier.
    """
    from matplotlib.lines import Line2D

    x = np.arange(len(months))
    w = 0.34
    for cols, off, color, ls in (
        (groups_a, -w / 2 - 0.03, DOMAIN_COLOR, ":"),
        (groups_b, +w / 2 + 0.03, SITE_COLOR, "-"),
    ):
        keep = [j for j, c in enumerate(cols) if c.size]
        if not keep:
            continue
        bp = ax.boxplot(
            [cols[j] for j in keep],
            positions=x[keep] + off,
            widths=w,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for key in ("boxes", "whiskers", "caps", "medians"):
            for art in bp[key]:
                art.set_color("black" if key == "medians" else color)
                art.set_linestyle(ls)
                art.set_linewidth(1.7 if key == "medians" else 1.5)
                if key == "boxes":
                    art.set_facecolor("white")
                    art.set_edgecolor(color)
        for j in keep:
            med = np.median(cols[j])
            ax.text(
                x[j] + off,
                med,
                f" {med:,.0f}",
                ha="left",
                va="bottom",
                fontsize=7.5,
                color="0.25",
            )
    handles = [
        plt_rect(DOMAIN_COLOR, ":"),
        plt_rect(SITE_COLOR, "-"),
        Line2D([0], [0], color="black", lw=1.7),
    ]
    ax.legend(
        handles,
        [label_a, label_b, f"median [{unit}], value at right"],
        title="whiskers: min to max, no outliers drawn   |   one sample "
        "per event, assigned to its starting month",
        title_fontsize=8,
        fontsize=9,
        ncol=3,
        framealpha=0.9,
        loc="upper right",
    )
    ax.set_xticks(x)


def plt_rect(color, ls):
    import matplotlib.pyplot as plt

    return plt.Rectangle((0, 0), 1, 1, fc="white", ec=color, lw=1.5, ls=ls)


def fig_taylor_monthly_box(
    E: ExtentAnalysis, ev: dict, out_dir=None, dpi=None, key: str = "L_km"
):
    """Taylor length by starting month, domain-wide beside the ARM cell.

    The companion to ``fig_monthly_duration_box``: the same construction with
    the duration replaced by U x dt. Because the 10 m wind barely varies from
    month to month in the cold season, the monthly pattern is the duration's.
    """
    import matplotlib.pyplot as plt

    months = E.months
    dom = [ev[key][ev["month"] == m] for m in months]
    site = [ev[key][(ev["month"] == m) & ev["is_site"]] for m in months]
    n_dom = [c.size for c in dom]
    n_site = [c.size for c in site]
    n_cens = [int(ev["censored"][ev["month"] == m].sum()) for m in months]

    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(months), 6.4))
    _month_boxes(
        ax,
        months,
        dom,
        site,
        "ERA5, all cells of the region",
        "ERA5, ARM cell only",
        "km",
    )
    ax.set_yscale("log")
    hi = max(float(c.max()) for c in dom if c.size)
    ax.set_ylim(1.0, hi * 8.0)
    ax.set_ylabel(
        "Taylor extent of a liquid-containing event, " "$L = U_{10}\\,\\Delta t$ [km]",
        fontsize=10.5,
    )
    ax.set_xticklabels(
        [
            f"{calendar.month_abbr[m]}\n{n_dom[j]:,} events (all cells)"
            f"\n{n_site[j]} at ARM cell\n{n_cens[j]:,} censored"
            for j, m in enumerate(months)
        ],
        fontsize=9,
    )
    _tidy(ax)
    _threshold_box(ax, E, loc="upper left")
    fig.text(
        0.09,
        0.01,
        "censored: the event touches a gap in the archive or a "
        "season boundary, so its length is a lower bound; included rather "
        "than dropped, since dropping would remove the long clouds first",
        fontsize=7.5,
        color="0.35",
    )
    fig.suptitle(
        f"{precip_banner(E.args)}\nHorizontal extent of liquid-containing "
        f"cloud events by month, ERA5, {E.args.region} region, "
        f"{_season_span(E)}   |   {LIQUID_VAR_LABEL[E.args.liquid_var]}"
        f"   |   {ev['L_km'].size:,} events",
        fontsize=11.5,
        y=0.97,
    )
    fig.subplots_adjust(top=0.87, bottom=0.2, left=0.08, right=0.985)
    return _save(fig, E, out_dir, f"taylor_monthly_box_{key}", dpi)


def fig_taylor_maps(
    E: ExtentAnalysis,
    ev: dict,
    out_dir=None,
    dpi=350,
    height: float = 6.0,
    legend_fontsize=None,
    tick_fontsize=None,
    stat: str = "mean",
):
    """Three maps: event duration, event-mean wind, and L, summarised per cell.

    ``stat`` is "mean" (default) or "median" over that cell's events. The mean
    is the default because a median of whole hours is quantised to 2, 3 or 4 h
    and draws as a posterised map; the mean is dominated by the long events,
    which is also where the hours are. Together the three panels show which of
    the two factors in L = U dt sets the spatial pattern.
    """
    import matplotlib.pyplot as plt

    legend_fontsize, tick_fontsize = _fonts(E, legend_fontsize, tick_fontsize)
    panels = [
        ("length_h", f"{stat} event duration [h]", DURATION_CMAP),
        ("U_mean_m_s", f"{stat} event-mean 10 m wind [m s$^{{-1}}$]", WIND_CMAP),
        ("L_km", f"{stat} Taylor extent $L$ [km]", LENGTH_CMAP),
    ]
    aspect = projected_aspect(E)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3 * height * aspect * 1.5, height),
        subplot_kw={"projection": _projection(E)},
    )
    for ax, (key, lab, cmap) in zip(axes, panels):
        field = per_cell_stat(E, ev, key, stat)
        vmin, vmax = np.nanpercentile(field, [1, 99])
        mesh = _draw_one(ax, E, field, vmin, vmax, cmap, tick_fontsize=tick_fontsize)
        cb = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06, aspect=28
        )
        cb.set_label(lab, fontsize=10.5)
        cb.ax.tick_params(labelsize=tick_fontsize)
        site = field[E.site_iy, E.site_ix]
        ax.set_title(
            f"domain median {np.nanmedian(field):,.1f}   |   " f"ARM cell {site:,.1f}",
            fontsize=10,
        )
    _site_legend_below(fig, legend_fontsize)
    fig.suptitle(
        f"{precip_banner(E.args)}\nMethod 1, per cell: the {stat} over "
        f"that cell's liquid-containing events   |   {E.args.region} "
        f"region, {_season_span(E)}\n{_subtitle(E)}",
        fontsize=11,
        y=1.02,
    )
    return _save(fig, E, out_dir, f"taylor_maps_{stat}", dpi)


def fig_snapshot_hist(
    E: ExtentAnalysis,
    comp: dict,
    out_dir=None,
    dpi=None,
    keys=("d_eq_km", "extent_wind_km", "feret_max_km", "feret_min_km"),
    interior_only: bool = False,
):
    """Histograms of the component size measures, one count per component-hour."""
    import matplotlib.pyplot as plt

    sel = ~comp["touches_edge"] if interior_only else np.ones(comp["t"].size, bool)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    bins = np.geomspace(1.0, 3000.0, 81)
    for key in keys:
        a = comp[key][sel]
        ax.hist(
            a,
            bins=bins,
            density=True,
            histtype="step",
            color=SNAPSHOT_COLORS[key],
            lw=2.0 if key == "d_eq_km" else 1.4,
            label=f"{SNAPSHOT_LABELS[key]}: median {np.median(a):,.0f} km, "
            f"IQR {_q(a, 25):,.0f}–{_q(a, 75):,.0f} km",
        )
    _cell_reference_lines(ax, E)
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("horizontal extent of a liquid-containing component [km]")
    ax.set_ylabel("probability density per component-hour")
    ax.legend(
        fontsize=8.3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        title=f"{int(sel.sum()):,} component-hours"
        + (", interior only" if interior_only else "")
        + f"   |   single-cell {100 * np.mean(comp['n_cells'][sel] == 1):.0f}%",
        title_fontsize=8.5,
    )
    _threshold_box(ax, E, loc="upper right")
    _tidy(ax)
    ax.set_title(
        f"{precip_banner(E.args)}\nMethod 2 — connected components "
        f"({comp['connectivity']}-connectivity) of the hourly field, "
        f"{E.args.region} region, {_season_span(E)}",
        fontsize=10.5,
    )
    stem = "snapshot_hist" + ("_interior" if interior_only else "")
    return _save(fig, E, out_dir, stem, dpi)


def fig_snapshot_monthly_box(
    E: ExtentAnalysis, comp: dict, out_dir=None, dpi=None, key: str = "extent_wind_km"
):
    """Component size by month: all components beside those holding the ARM cell."""
    import matplotlib.pyplot as plt

    months = E.months
    dom = [comp[key][comp["month"] == m] for m in months]
    site = [comp[key][(comp["month"] == m) & comp["has_site"]] for m in months]
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(months), 6.4))
    _month_boxes(
        ax,
        months,
        dom,
        site,
        "all components",
        "components containing the ARM cell",
        "km",
    )
    ax.set_yscale("log")
    ax.set_ylim(1.0, max(c.max() for c in dom if c.size) * 8.0)
    ax.set_ylabel(f"{SNAPSHOT_LABELS[key]} [km]", fontsize=11)
    ax.set_xticklabels(
        [
            f"{calendar.month_abbr[m]}\n{dom[j].size:,} component-h"
            f"\n{site[j].size:,} with ARM cell"
            for j, m in enumerate(months)
        ],
        fontsize=9,
    )
    _tidy(ax)
    _threshold_box(ax, E, loc="upper left")
    fig.suptitle(
        f"{precip_banner(E.args)}\nSize of liquid-containing components "
        f"by month, one sample per component per hour, ERA5, "
        f"{E.args.region} region, {_season_span(E)}",
        fontsize=11.5,
        y=0.97,
    )
    fig.subplots_adjust(top=0.87, bottom=0.16, left=0.08, right=0.985)
    return _save(fig, E, out_dir, f"snapshot_monthly_box_{key}", dpi)


def fig_snapshot_maps(
    E: ExtentAnalysis,
    comp: dict,
    out_dir=None,
    dpi=350,
    height: float = 6.0,
    legend_fontsize=None,
    tick_fontsize=None,
):
    """Per cell: the median size of the component the cell belonged to.

    One count per liquid-containing cell-hour, so a cell's value is the size
    of the cloud it is typically part of -- large near the middle of the
    domain's persistent decks, small where cloud is patchy. Edge cells are
    biased low: a component there is cut by the domain boundary.
    """
    import matplotlib.pyplot as plt

    legend_fontsize, tick_fontsize = _fonts(E, legend_fontsize, tick_fontsize)
    shape = E.liq.shape[1:]
    panels = [
        (
            cell_median_from_hist(comp["cell_hist_d_eq"], comp["size_bins_km"], shape),
            "median equivalent diameter of the containing component [km]",
        ),
        (
            cell_median_from_hist(comp["cell_hist_wind"], comp["size_bins_km"], shape),
            "median along-wind extent of the containing component [km]",
        ),
    ]
    aspect = projected_aspect(E)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * height * aspect * 1.5, height),
        subplot_kw={"projection": _projection(E)},
    )
    for ax, (field, lab) in zip(axes, panels):
        vmin, vmax = np.nanpercentile(field, [1, 99])
        mesh = _draw_one(
            ax, E, field, vmin, vmax, LENGTH_CMAP, tick_fontsize=tick_fontsize
        )
        cb = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06, aspect=28
        )
        cb.set_label(_wrap(lab, 44), fontsize=10)
        cb.ax.tick_params(labelsize=tick_fontsize)
        ax.set_title(
            f"domain median {np.nanmedian(field):,.0f} km   |   "
            f"ARM cell {field[E.site_iy, E.site_ix]:,.0f} km",
            fontsize=10,
        )
    _site_legend_below(fig, legend_fontsize)
    fig.suptitle(
        f"{precip_banner(E.args)}\nMethod 2, per cell: median over the "
        f"cell's liquid-containing hours of the size of its component "
        f"({comp['connectivity']}-conn.)   |   {E.args.region} region, "
        f"{_season_span(E)}\n{_subtitle(E)}",
        fontsize=11,
        y=1.02,
    )
    return _save(fig, E, out_dir, "snapshot_maps", dpi)


def fig_snapshot_example(
    E: ExtentAnalysis,
    comp: dict,
    t: int,
    out_dir=None,
    dpi=250,
    height: float = 7.0,
    legend_fontsize=None,
    tick_fontsize=None,
    quiver_stride: int = 3,
):
    """One hour: the labelled components, the 10 m wind, and their extents.

    The sanity figure for method 2. Every component is drawn in its own
    colour with its equivalent diameter and along-wind extent written at its
    centroid; the arrows are the 10 m wind. Edge-touching components are
    labelled with an asterisk.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap

    legend_fontsize, tick_fontsize = _fonts(E, legend_fontsize, tick_fontsize)
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1] = ndimage.generate_binary_structure(2, 2 if comp["connectivity"] == 8 else 1)
    lab, n = ndimage.label(E.liq[t : t + 1], structure=st)
    lab = lab[0].astype(float)
    lab[lab == 0] = np.nan
    sel = comp["t"] == t
    if int(sel.sum()) != n:
        raise AssertionError("component table disagrees with a fresh labelling")

    rng = np.random.default_rng(0)  # fixed so the colours repeat
    colors = plt.get_cmap("tab20")(rng.permutation(20) % 20)
    cmap = ListedColormap(np.tile(colors, (max(1, n // 20 + 1), 1))[: max(n, 1)])

    aspect = projected_aspect(E)
    fig, ax = plt.subplots(
        figsize=(max(height * aspect * 1.6, 5), height),
        subplot_kw={"projection": _projection(E)},
    )
    _draw_one(ax, E, lab, 1, max(n, 1), cmap, tick_fontsize=tick_fontsize)
    pc = ccrs.PlateCarree()
    s = quiver_stride
    LO, LA = np.meshgrid(E.lon[::s], E.lat[::s])
    # cartopy rotates the vectors into the projection itself; no angles="xy".
    ax.quiver(
        LO,
        LA,
        E.u10[t, ::s, ::s],
        E.v10[t, ::s, ::s],
        transform=pc,
        scale_units="width",
        scale=150,
        width=0.0025,
        color="black",
        alpha=0.65,
        zorder=5,
    )
    # Sort by label order in the table (raster order of first cell).
    for k in np.flatnonzero(sel):
        star = "*" if comp["touches_edge"][k] else ""
        ax.text(
            comp["lon_c"][k],
            comp["lat_c"][k],
            f"{comp['d_eq_km'][k]:.0f}{star}\n{comp['extent_wind_km'][k]:.0f}",
            transform=pc,
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
    _site_legend(fig, fontsize=legend_fontsize)
    ts = np.datetime_as_string(E.times[t], unit="h")
    ax.set_title(
        f"{precip_banner(E.args)}\n{ts} UTC: {n} liquid-containing "
        f"components ({comp['connectivity']}-connectivity)\n"
        f"labels: equivalent diameter / along-wind extent [km], "
        f"* touches the domain edge; arrows: 10 m wind",
        fontsize=10,
    )
    return _save(fig, E, out_dir, f"snapshot_example_{ts.replace(':', '')}", dpi)


def fig_method_comparison(
    E: ExtentAnalysis, ev: dict, comp: dict, out_dir=None, dpi=None
):
    """Both methods on one axis, and the one-to-one pairing at the ARM cell.

    (a) Distributions: Taylor L per event, Taylor L per cell-hour (duration-
        weighted, the snapshot method's sampling unit), and the snapshot
        along-wind extent and equivalent diameter per component-hour.
    (b) At the ARM cell, each event's Taylor L against the mean over its hours
        of the along-wind extent of the component containing the cell. Points
        above the 1:1 line are events Taylor calls wider than the field is at
        that moment -- the cloud was renewed or advected in, not frozen.
    """
    import matplotlib.pyplot as plt

    pairs = site_event_pairs(E, ev, comp)
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    bins = np.geomspace(1.0, 1e4, 81)
    series = [
        ("Taylor $L$, one sample per event", ev["L_km"], None, DOMAIN_COLOR, ":", 1.6),
        (
            "Taylor $L$, duration-weighted (per cell-hour)",
            ev["L_km"],
            ev["length_h"],
            DOMAIN_COLOR,
            "-",
            2.2,
        ),
        (
            "snapshot: extent along the wind, per component-hour",
            comp["extent_wind_km"],
            None,
            SNAPSHOT_COLORS["extent_wind_km"],
            "-",
            2.2,
        ),
        (
            "snapshot: equivalent diameter, per component-hour",
            comp["d_eq_km"],
            None,
            SNAPSHOT_COLORS["d_eq_km"],
            "-",
            1.6,
        ),
    ]
    for lab, a, w, c, ls, lw in series:
        med = np.median(a) if w is None else _weighted_median(a, w)
        ax_h.hist(
            a,
            bins=bins,
            weights=w,
            density=True,
            histtype="step",
            color=c,
            ls=ls,
            lw=lw,
            label=f"{lab}: median {med:,.0f} km",
        )
    _cell_reference_lines(ax_h, E)
    ax_h.set_xscale("log")
    ax_h.set_xlim(bins[0], bins[-1])
    ax_h.set_xlabel("horizontal extent [km]")
    ax_h.set_ylabel("probability density")
    ax_h.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
    )
    ax_h.set_title("(a)  the two methods over the whole region", loc="left")
    _tidy(ax_h)

    L, S = pairs["L_km"], pairs["snapshot_extent_wind_km"]
    cens = pairs["censored"]
    ax_s.scatter(
        S[~cens],
        L[~cens],
        s=14,
        color=SITE_COLOR,
        alpha=0.55,
        lw=0,
        label=f"{int((~cens).sum())} events",
    )
    ax_s.scatter(
        S[cens],
        L[cens],
        s=18,
        facecolor="none",
        edgecolor=SITE_COLOR,
        lw=0.8,
        label=f"{int(cens.sum())} censored events",
    )
    lim = (3.0, max(L.max(), S.max()) * 1.5)
    ax_s.plot(lim, lim, color="0.4", lw=1.0, ls="--", label="1:1")
    ratio = L / S
    ax_s.text(
        0.03,
        0.97,
        f"median $L$ / snapshot extent = {np.median(ratio):.2f}\n"
        f"Spearman $\\rho$ = {_spearman(np.log(S), np.log(L)):.2f}",
        transform=ax_s.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.8),
    )
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlim(lim)
    ax_s.set_ylim(lim)
    ax_s.set_xlabel(
        "snapshot extent along the wind of the component containing "
        "the ARM cell,\nmean over the event's hours [km]"
    )
    ax_s.set_ylabel("Taylor $L = U_{10}\\,\\Delta t$ of the same event [km]")
    ax_s.legend(
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
    )
    ax_s.set_title("(b)  the same events at the ARM cell, paired", loc="left")
    _tidy(ax_s)
    fig.suptitle(
        f"{precip_banner(E.args)}\nTaylor's hypothesis against the "
        f"snapshot geometry, ERA5, {E.args.region} region, "
        f"{_season_span(E)}",
        fontsize=11.5,
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, E, out_dir, "method_comparison", dpi)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)
