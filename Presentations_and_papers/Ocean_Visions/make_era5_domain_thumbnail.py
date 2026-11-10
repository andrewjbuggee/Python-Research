#!/usr/bin/env python3
"""A polar thumbnail map of the run's domain, with Utqiagvik marked by a gold star.

WHAT IT MAKES
=============
One PNG, ``<domain>_domain_thumbnail.png``: the Arctic seen from above, the
domain outlined in red and Utqiagvik shown as a gold star with a legend entry.
It is meant for the corner of a domain-wide figure, so a reader can see which
region "the whole domain" means.

This is an updated copy of ``map_liquid_hours.fig_domain_thumbnail`` (the last
code cell of ocean_visions_figures.ipynb). Changes from that version:

    * the domain outline is 30% thinner: 1.4 pt instead of 2.0 pt
    * a gold star marks Utqiagvik (the ARM NSA site), with a legend
    * the PNG is written at 500 dpi instead of 350
    * the domain comes from ov_config (any --region or --box), not from a
      prepared analysis object, so the script reads NO ERA5 data and runs on
      the laptop for any domain, including ones not on this disk

The star is drawn whether or not Utqiagvik is inside the domain: it is a
reference point, not a domain marker.

HOW THE OUTLINE IS FOUND
========================
The notebook version read the outline from the archive's own latitude and
longitude axes. Here the same grid is rebuilt from the domain box on the ERA5
0.25 deg grid (ov_config.GRID_DEG), which gives the same first and last grid
points as the archive for any box whose edges are on the grid, as every REGIONS
entry and every --box used so far is. Off-grid edges are snapped inward to the
grid, matching the CDS selection rule in aws_pipeline/era5_s3.make_window.

USAGE
=====
    python make_era5_domain_thumbnail.py                          # ov_config's DOMAIN
    python make_era5_domain_thumbnail.py --box 80 -180 65 -140 --region-name Rui-region
    python make_era5_domain_thumbnail.py --region arctic_circle --no-title
    python make_era5_domain_thumbnail.py --box 75 170 65 -170     # across the dateline

Output goes to ``<run out_dir>/domain_thumbnail/`` (beside the scripts for the
Barrow baseline, under OV_RESULTS_ROOT/<domain>_<span>/ otherwise), or to
--out-dir. --years only affects that folder name; the map depends on no season.

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

# First, before numpy: on Casper it sets thread limits, and it puts the
# analysis modules (ERA5/surface_energy_budget) on sys.path.
import ov_config as ov

import numpy as np

from map_liquid_hours import _central_lon, _lon_frame  # noqa: E402
from plot_surface_class_timeseries import SITE_LAT, SITE_LON  # noqa: E402  deg N, deg E

OUT_PRODUCT = "domain_thumbnail"

# ----------------------------------------------------------------------------
# Figure settings. Values the notebook version used are noted where changed.
# ----------------------------------------------------------------------------
DPI = 500  # dots per inch (notebook version: 350)
BOX_LINEWIDTH_PT = 1  # pt, 30% thinner than the notebook's 2.0 pt
BOX_COLOR = "red"
LAT_MIN_DEG = 66.0  # deg N, disc edge: about the Arctic Circle
SIZE_IN = 3.0  # in, width of the square map
STAR_COLOR = "gold"
STAR_EDGE_COLOR = "0.15"  # dark outline so the gold reads on pale land and sea
STAR_SIZE_PT = 8.0  # pt, marker size
SITE_LABEL = "Utqiaġvik"
TITLE = "averaging domain"


# ----------------------------------------------------------------------------
# The domain's grid
# ----------------------------------------------------------------------------
def domain_grid(domain: ov.Domain) -> tuple[np.ndarray, np.ndarray]:
    """Grid-point latitudes and longitudes (deg) inside the domain box.

    Longitudes are in -180..180. A box across the dateline (W > E) comes back
    in stored order, e.g. 170 .. 179.75, -180 .. -170, which is the order the
    archive gives and what map_liquid_hours._lon_frame expects.
    """
    step = ov.GRID_DEG
    eps = 1e-9
    # Snap inward to the grid: the first grid point at or inside each edge.
    lat_s = math.ceil(domain.south_deg / step - eps) * step
    lat_n = math.floor(domain.north_deg / step + eps) * step
    lat_deg = np.arange(lat_n, lat_s - step / 2, -step)  # north to south

    lon_w = math.ceil(domain.west_deg / step - eps) * step
    lon_e = math.floor(domain.east_deg / step + eps) * step
    if domain.crosses_dateline:
        # West edge to 179.75, then -180 to the east edge.
        lon_deg = np.concatenate(
            [
                np.arange(lon_w, 180.0 - step / 2, step),
                np.arange(-180.0, lon_e + step / 2, step),
            ]
        )
    else:
        lon_deg = np.arange(lon_w, lon_e + step / 2, step)
    return lat_deg, lon_deg


# ----------------------------------------------------------------------------
# The figure
# ----------------------------------------------------------------------------
def fig_domain_thumbnail(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    title: str | None = TITLE,
    lat_min: float = LAT_MIN_DEG,
    central_longitude: float | None = None,
    box_color: str = BOX_COLOR,
    box_linewidth: float = BOX_LINEWIDTH_PT,
    size_in: float = SIZE_IN,
):
    """The Arctic from above with the domain outlined and Utqiagvik starred.

    Same map as map_liquid_hours.fig_domain_thumbnail: a polar stereographic
    disc down to ``lat_min``, rotated so the domain sits at the bottom
    (``central_longitude`` defaults to the domain's central meridian), land
    grey, ocean pale blue, a light graticule, and the domain's grid bounds as
    an outline whose parallels curve as they should on this projection.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    lat0, lat1 = float(lat.min()), float(lat.max())
    # A smooth longitude frame, so a dateline box is outlined the short way round.
    frame, x_deg = _lon_frame(lon)
    lon0, lon1 = float(x_deg.min()), float(x_deg.max())
    if central_longitude is None:
        central_longitude = _central_lon(lon)
    south = 0.5 * (lat0 + lat1) < 0
    # Extend the disc if the domain reaches past the default edge (4 deg margin).
    if south:
        lat_edge = max(-lat_min, lat1 + 4.0)
        proj = ccrs.SouthPolarStereo(central_longitude=central_longitude)
    else:
        lat_edge = min(lat_min, lat0 - 4.0)
        proj = ccrs.NorthPolarStereo(central_longitude=central_longitude)

    fig = plt.figure(figsize=(size_in, size_in + (0.3 if title else 0.0)))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(
        [-180, 180, -90, lat_edge] if south else [-180, 180, lat_edge, 90],
        ccrs.PlateCarree(),
    )
    # Clip the square stereographic extent to its inscribed circle (the usual
    # polar-map recipe: a boundary path in axes coordinates).
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = mpath.Path(
        np.column_stack([0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta)])
    )
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="#dbe7f2",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.86",
        edgecolor="0.3",
        linewidth=0.5,
        zorder=1,
    )
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="0.55",
        alpha=0.7,
        linestyle="-",
        xlocs=range(-180, 180, 45),
        ylocs=range(-85, 90, 5),
        zorder=2,
    )

    # The domain as a ring of many points, so the two parallels draw as arcs
    # on this projection rather than as straight chords.
    n = 60
    ring_lon = np.concatenate(
        [
            np.linspace(lon0, lon1, n),
            np.full(n, lon1),
            np.linspace(lon1, lon0, n),
            np.full(n, lon0),
        ]
    )
    ring_lat = np.concatenate(
        [
            np.full(n, lat0),
            np.linspace(lat0, lat1, n),
            np.full(n, lat1),
            np.linspace(lat1, lat0, n),
        ]
    )
    ax.plot(
        ring_lon,
        ring_lat,
        transform=frame,
        color=box_color,
        linewidth=box_linewidth,
        solid_capstyle="round",
        zorder=5,
    )

    # Utqiagvik as a gold star. Only on a northern map: on a southern disc
    # it would fall outside the drawn area.
    if not south:
        ax.plot(
            SITE_LON,
            SITE_LAT,
            transform=ccrs.PlateCarree(),
            marker="*",
            markersize=STAR_SIZE_PT,
            color=STAR_COLOR,
            markeredgecolor=STAR_EDGE_COLOR,
            markeredgewidth=0.5,
            linestyle="none",
            zorder=6,
            label=SITE_LABEL,
        )
        # Legend in the top-left corner of the square axes, which lies outside
        # the clipped disc, so it never covers the map.
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.02),
            fontsize=7,
            frameon=False,
            handletextpad=0.2,
            borderaxespad=0.0,
            markerscale=0.9,
        )

    ax.spines["geo"].set_linewidth(0.8)
    ax.spines["geo"].set_edgecolor("0.3")
    if title:
        ax.set_title(title, fontsize=11, color="0.25", pad=4)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90 if title else 0.98)
    return fig


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ov.add_run_args(p)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="folder for the PNG (default: <run out_dir>/domain_thumbnail)",
    )
    p.add_argument(
        "--dpi", type=int, default=DPI, help=f"PNG resolution (default {DPI})"
    )
    p.add_argument(
        "--no-title", action="store_true", help="drop the caption above the map"
    )
    opts = p.parse_args(argv)

    # resolve_domain, not run_from_args: the latter stops when a domain's ERA5
    # files are not on this disk, and this script reads no ERA5 at all.
    domain = ov.resolve_domain(
        opts.region or os.environ.get("OV_REGION") or ov.DOMAIN,
        box=opts.box,
        name=opts.region_name,
    )
    lat_deg, lon_deg = domain_grid(domain)
    print(
        f"  domain  {domain.name}: {domain.describe()}, "
        f"{lat_deg.size} x {lon_deg.size} grid points"
    )

    if opts.out_dir is not None:
        out_dir = opts.out_dir
    else:
        from plot_surface_class_timeseries import parse_years  # the --years grammar

        raw_years = opts.years or os.environ.get("OV_YEARS") or ov.YEARS
        storage = opts.storage or os.environ.get("OV_STORAGE") or ov.DEFAULT_STORAGE
        run = ov.RunConfig(domain, tuple(parse_years(raw_years)), storage)
        out_dir = run.out_dir(OUT_PRODUCT)

    import matplotlib

    matplotlib.use("Agg")  # file output only; no display needed
    import matplotlib.pyplot as plt

    fig = fig_domain_thumbnail(lat_deg, lon_deg, title=None if opts.no_title else TITLE)
    path = out_dir / f"{domain.name}_domain_thumbnail.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {path}  ({opts.dpi} dpi)")
    return path


if __name__ == "__main__":
    main()
