"""Figure 1 site map for the EPCAPE overview paper (Russell et al., BAMS).

Draws a regional map of the eastern Pacific / Southern California Bight with a
gold star at La Jolla, and a zoomed-in aerial-imagery inset showing the three
locations named in the paper:

    * Scripps Pier (ARM AMF1 main site, M1)
    * Mt. Soledad (ARM supplementary site, S2)
    * Kearny Mesa (San Diego APCD air-quality monitor, "~10 km to the east")

Two styles, selected with --style:

    satellite : regional map AND inset both use true-colour imagery
    vector    : regional map uses cartopy/Natural Earth vector layers,
                inset uses true-colour imagery

and two layouts, selected with --version (see VERSIONS below):

    ver1 : 127-109 W, 23-39 N
    ver2 : 130-112 W, 28-43 N (the Fig. 3 map bounds), larger labels;
           output files carry a "_ver2" suffix

Usage (from the EPCAPE folder):

    python plotting/fig1_site_map.py                      # ver1, both styles
    python plotting/fig1_site_map.py --style vector
    python plotting/fig1_site_map.py --version ver2 --style satellite

Imagery: Esri World Imagery tiles (true-colour satellite/aerial mosaic).
Esri permits use in publications with the attribution printed in the inset;
Google tiles are deliberately NOT used, because Google's terms forbid fetching
its tiles outside the Google Maps APIs.
Tiles are fetched by fetch_mosaic() below with curl (hard timeouts and retries;
cartopy's own tile downloader and urllib both hung on this network) and cached in
data/basemap_tiles/ after the first run.
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

# ---------------------------------------------------------------------------
# Site coordinates (decimal degrees, WGS84; longitude negative = west)
# ---------------------------------------------------------------------------
# M1 and S2 are the lat/lon/alt global variables in the ARM files themselves
# (epcceilM1.b1 and epcldquantsS2.c1), so they match the data in the paper.
# Kearny Mesa is the EPA AQS site list entry for 06-073-1016,
# "San Diego - Kearny Villa Rd.", 6125A Kearny Villa Rd (active since 2010):
# https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip
SITES = {
    "pier": {"lat_deg": 32.866943, "lon_deg": -117.256943, "alt_m": 7.0},
    "soledad": {"lat_deg": 32.840378, "lon_deg": -117.249542, "alt_m": 217.0},
    "kearny": {"lat_deg": 32.845709, "lon_deg": -117.123964, "alt_m": 134.0},
}

# ---------------------------------------------------------------------------
# Figure versions
# ---------------------------------------------------------------------------
# Each version sets the regional extent [lon_min, lon_max, lat_min, lat_max]
# (deg), the graticule, the place labels, how much larger than the ver1 base
# sizes those labels are drawn, and how the inset is tied to its footprint.
#
# Label tuples are (lon_deg, lat_deg, text, kind, rotation_deg); "kind" picks
# the type style in LABEL_KIND below. Positions were placed by eye inside each
# polygon / water body for the given extent, so they are per-version.
#
# Leader tuples are ((inset_x, inset_y) in inset-axes fraction, corner of the
# inset footprint as "ll"/"lr"/"ul"/"ur") and draw the zoom lines.
VERSIONS = {
    # ver1: covers the Southern California Bight, Baja California and the
    # offshore stratocumulus region, and contains the 12 deg x 12 deg ERA5
    # clustering domain of Fig. 2 (Zheng et al. 2025).
    "ver1": {
        "extent_deg": [-127.0, -109.0, 23.0, 39.0],
        "xlocs_deg": np.arange(-130, -105, 4),
        "ylocs_deg": np.arange(20, 42, 4),
        "label_boost_pt": 0.0,
        "tick_label_pt": 8.0,
        "site_label_boost_pt": 0.0,
        "site_codes": True,  # "(M1)", "(S2)", "(SDAPCD)" under the site names
        "labels": [
            (-112.6, 36.6, "UNITED STATES", "country", 0),
            (-110.6, 30.6, "MEXICO", "country", 0),
            (-118.6, 36.4, "California", "state", 0),
            (-112.0, 34.3, "Arizona", "state", 0),
            (-115.9, 38.1, "Nevada", "state", 0),
            (-115.25, 30.6, "Baja\nCalifornia", "state_small", 0),
            (-112.2, 26.2, "Baja California\nSur", "state_small", -38),
            (-110.1, 29.0, "Sonora", "state", 0),
            (-123.6, 33.2, "PACIFIC\nOCEAN", "ocean", 0),
            (-111.75, 28.25, "Gulf of California", "sea", -47),
            (-119.35, 33.55, "Southern\nCalifornia\nBight", "sea_small", 0),
        ],
        "leaders": [((0.0, 1.0), "ll"), ((1.0, 1.0), "lr")],
        "suffix": "",
    },
    # ver2: same lon/lat bounds as the satellite maps of Fig. 3 (panels A-T),
    # measured from that figure's tick marks: 130-112 W, 28-43 N, with ticks
    # at 128/123/118/113 W and 30/35/40 N. Fig. 3 is drawn on a plate carree
    # grid; this map keeps Web Mercator (see MERC below), so the bounds match
    # but the panel is ~25 % taller relative to its width than a Fig. 3 panel.
    "ver2": {
        "extent_deg": [-130.0, -112.0, 28.0, 43.0],
        "xlocs_deg": np.array([-128, -123, -118, -113]),
        "ylocs_deg": np.array([30, 35, 40]),
        "label_boost_pt": 2.5,  # "a few points larger" than ver1
        "tick_label_pt": 10.0,  # lat/lon labels, "a bit larger" than ver1
        "site_label_boost_pt": 1.5,  # inset site names, 7.5 -> 9 pt
        "site_codes": False,  # names only
        "labels": [
            (-116.4, 41.3, "UNITED STATES", "country", 0),
            (-115.6, 31.75, "MEXICO", "country", 0),
            (-119.7, 37.2, "California", "state", 0),
            (-121.0, 42.5, "Oregon", "state", 0),
            (-116.6, 39.2, "Nevada", "state", 0),
            (-113.3, 34.6, "Arizona", "state", 0),
            (-114.35, 29.35, "Baja California", "state_small", -52),
            (-126.4, 37.6, "PACIFIC\nOCEAN", "ocean", 0),
            (-113.45, 29.85, "Gulf of\nCalifornia", "sea", -52),
            (-119.95, 33.85, "Southern\nCalifornia Bight", "sea_small", 0),
        ],
        # Inset sits beside La Jolla at about the same latitude, so the zoom
        # lines run from its right edge to the footprint's west side.
        "leaders": [((1.0, 1.0), "ul"), ((1.0, 0.0), "ll")],
        "suffix": "_ver2",
    },
}

# Base type sizes (pt) and styles for each label kind (ver1 sizes).
LABEL_KIND = {
    "country": {"size": 10.0, "weight": "bold", "style": "normal", "water": False},
    "state": {"size": 7.5, "weight": "normal", "style": "italic", "water": False},
    "state_small": {"size": 7.0, "weight": "normal", "style": "italic", "water": False},
    "ocean": {"size": 11.0, "weight": "bold", "style": "italic", "water": True},
    "sea": {"size": 7.5, "weight": "normal", "style": "italic", "water": True},
    "sea_small": {"size": 6.5, "weight": "normal", "style": "italic", "water": True},
}
CITY_SIZE_PT = 7.0  # Los Angeles
SITE_TOWN_SIZE_PT = 9.0  # "La Jolla" beside the regional-map star
INSET_OCEAN_SIZE_PT = 8.0  # "Pacific Ocean" in the inset
INSET_TOWN_SIZE_PT = 7.0  # "La Jolla" in the inset
INSET_SITE_SIZE_PT = 7.5  # Scripps Pier / Mt. Soledad / Kearny Mesa (bold)

# Inset extent: wide enough to hold the Pier, Mt. Soledad and Kearny Mesa,
# plus some ocean west of the Pier for labels.
INSET_EXTENT_DEG = [-117.335, -117.095, 32.790, 32.912]

# Esri World Imagery XYZ tile endpoint (note: Esri orders it z/y/x).
ESRI_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg"
)
ESRI_ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics, GIS User Community"

# Web-tile zoom levels. A tile is 256 px wide and covers 360/2**z deg of
# longitude, so the number of image pixels across each map is
#   region: 18 deg  * 2**8  * 256 / 360 ~ 3300 px  (map ~6.5 in * 400 dpi = 2600 px)
#   inset : 0.24 deg* 2**15 * 256 / 360 ~ 5600 px  (inset ~3.3 in * 400 dpi = 1300 px)
# i.e. both are at or above the printed resolution, so the imagery is never
# upsampled. Zoom 15 (~4.8 m/px at this latitude) is used rather than 14 so
# that the 300 m pier is resolved.
ZOOM_REGION = 8
ZOOM_INSET = 15

# Colours. Gold star for La Jolla / the Pier as requested; the two other sites
# get hues that stay distinct from gold and from each other on imagery.
GOLD = "#ffc20a"
SOLEDAD_COLOR = "#ff5a36"  # orange-red triangle (mountain-top site)
KEARNY_COLOR = "#33c3f0"  # sky-blue square (inland APCD monitor)
INK = "#1a1a1a"

FIG_WIDTH_IN = 7.0  # roughly a full page width (two-column) journal figure
DPI = 400

# Web Mercator: the native projection of the imagery tiles. Using it for both
# axes means the tiles are drawn without resampling (no reprojection blur).
MERC = ccrs.GOOGLE_MERCATOR
PC = ccrs.PlateCarree()


def halo(width: float = 2.0, color: str = "black") -> list:
    """Stroke behind text so labels stay legible on imagery."""
    return [pe.withStroke(linewidth=width, foreground=color)]


def merc_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Project lon/lat (deg) to Web-Mercator x/y (m)."""
    x_m, y_m = MERC.transform_point(lon_deg, lat_deg, PC)
    return x_m, y_m


def extent_aspect(extent_deg: list[float]) -> float:
    """Height/width ratio of a lon/lat box once projected to Web Mercator."""
    x0, y0 = merc_xy(extent_deg[0], extent_deg[2])
    x1, y1 = merc_xy(extent_deg[1], extent_deg[3])
    return (y1 - y0) / (x1 - x0)


# ---------------------------------------------------------------------------
# Imagery tiles
# ---------------------------------------------------------------------------
# Web-Mercator (EPSG:3857) constants: the projected world is a square of side
# 2*pi*R with R the WGS84 semi-major axis, split into 2**z x 2**z tiles at
# zoom z (OSM "Slippy map tilenames" convention, also used by Esri:
# https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames).
EARTH_RADIUS_M = 6378137.0
WORLD_HALF_M = np.pi * EARTH_RADIUS_M  # x, y range is [-WORLD_HALF_M, +WORLD_HALF_M]
TILE_PX = 256
TILE_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "basemap_tiles"


def _tile_index(lon_deg: float, lat_deg: float, zoom: int) -> tuple[int, int]:
    """Tile column x (west->east) and row y (north->south) holding a point."""
    n = 2**zoom
    lat_rad = np.deg2rad(lat_deg)
    x = int(np.floor((lon_deg + 180.0) / 360.0 * n))
    # Mercator northing normalised to [0, 1] from the north edge
    y = int(np.floor((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n))
    return x, y


def _fetch_tile(z: int, x: int, y: int, retries: int = 4) -> np.ndarray:
    """One 256x256 RGB tile from disk cache or the Esri server."""
    path = TILE_CACHE_DIR / str(z) / f"{x}_{y}.jpg"
    if not path.exists():
        # Download with curl rather than urllib: on this laptop Python's
        # sockets hung indefinitely in connect() against the Esri CDN while
        # curl from the same shell returned tiles in < 1 s. curl's -m is a
        # hard wall-clock limit per attempt, so a stalled tile cannot block
        # the run; --retry handles transient failures.
        url = ESRI_IMAGERY_URL.format(z=z, x=x, y=y)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")  # never leave a truncated .jpg behind
        cmd = ["curl", "-sSf", "-m", "20", "--retry", str(retries),
               "--retry-all-errors", "-A", "EPCAPE-fig1/1.0", "-o", str(tmp), url]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"tile {z}/{x}/{y} failed: {err}") from err
        tmp.rename(path)
    return np.asarray(Image.open(path).convert("RGB"))


def fetch_mosaic(extent_deg: list[float], zoom: int) -> tuple[np.ndarray, list[float]]:
    """Stitch every tile covering a lon/lat box into one RGB image.

    Returns the image (rows north->south) and its Web-Mercator bounds
    [x_min, x_max, y_min, y_max] in metres, ready for imshow(extent=...).
    The mosaic is slightly larger than the box (whole tiles); the axes extent
    crops it.
    """
    lon0, lon1, lat0, lat1 = extent_deg
    x_w, y_n = _tile_index(lon0, lat1, zoom)  # north-west corner tile
    x_e, y_s = _tile_index(lon1, lat0, zoom)  # south-east corner tile
    xs = range(x_w, x_e + 1)
    ys = range(y_n, y_s + 1)
    jobs = [(zoom, x, y) for y in ys for x in xs]
    print(f"  zoom {zoom}: {len(jobs)} tiles ({len(xs)} x {len(ys)})")
    with ThreadPoolExecutor(max_workers=4) as pool:
        tiles = list(pool.map(lambda j: _fetch_tile(*j), jobs))

    # Assemble row by row: tiles[i] is (row i // nx, column i % nx)
    nx = len(xs)
    rows = [np.concatenate(tiles[r * nx:(r + 1) * nx], axis=1) for r in range(len(ys))]
    mosaic = np.concatenate(rows, axis=0)  # shape (ny*256, nx*256, 3)

    # Mercator bounds of the tile block (tile edges are exact in metres)
    tile_m = 2.0 * WORLD_HALF_M / 2**zoom
    x_min_m = -WORLD_HALF_M + x_w * tile_m
    x_max_m = -WORLD_HALF_M + (x_e + 1) * tile_m
    y_max_m = WORLD_HALF_M - y_n * tile_m
    y_min_m = WORLD_HALF_M - (y_s + 1) * tile_m
    return mosaic, [x_min_m, x_max_m, y_min_m, y_max_m]


def add_imagery(ax, extent_deg: list[float], zoom: int) -> None:
    """Draw the Esri true-colour mosaic on a Web-Mercator GeoAxes.

    The axes and the tiles share the same projection, so imshow places the
    pixels directly with no reprojection/resampling beyond display scaling.
    """
    img, bounds_m = fetch_mosaic(extent_deg, zoom)
    ax.imshow(img, extent=bounds_m, transform=MERC, origin="upper",
              interpolation="lanczos", zorder=0)
    ax.set_extent(extent_deg, crs=PC)  # imshow resets limits; crop again


# ---------------------------------------------------------------------------
# Regional map
# ---------------------------------------------------------------------------
def draw_region_satellite(ax, extent_deg: list[float]) -> None:
    """True-colour imagery background plus thin political boundaries."""
    add_imagery(ax, extent_deg, ZOOM_REGION)
    # On imagery, the coastline is already visible; only add borders.
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"), edgecolor="white", linewidth=0.8
    )
    ax.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="white",
        linewidth=0.4,
        linestyle=(0, (4, 2)),
        alpha=0.8,
    )


def draw_region_vector(ax) -> None:
    """Standard cartopy / Natural Earth vector basemap."""
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d6e6f2")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f2efe9")
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#d6e6f2",
                   edgecolor="#7a8a99", linewidth=0.3)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"), edgecolor="#4d4d4d", linewidth=0.6
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"), edgecolor="#4d4d4d", linewidth=0.9
    )
    ax.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="#8c8c8c",
        linewidth=0.4,
        linestyle=(0, (4, 2)),
    )


def label_region(ax, on_imagery: bool, cfg: dict) -> None:
    """Names of countries, states, water bodies, and Los Angeles."""
    if on_imagery:
        text_color, water_color = "white", "#e6f2ff"
        effects = halo(1.6, "black")
    else:
        text_color, water_color = INK, "#2f5d80"
        effects = halo(1.6, "white")
    boost_pt = cfg["label_boost_pt"]

    # Countries upper-case bold, states small italic, water bodies italic
    # (blue on the vector map)
    for lon, lat, text, kind, rotation in cfg["labels"]:
        k = LABEL_KIND[kind]
        ax.text(lon, lat, text, transform=PC, ha="center", va="center",
                fontsize=k["size"] + boost_pt,
                color=water_color if k["water"] else text_color,
                style=k["style"], fontweight=k["weight"],
                rotation=rotation, rotation_mode="anchor",
                linespacing=1.0, path_effects=effects, zorder=6)

    # City for orientation
    lon_la, lat_la = -118.2437, 34.0522
    ax.plot(lon_la, lat_la, "o", ms=3.0, mfc=text_color, mec="black" if on_imagery
            else "white", mew=0.5, transform=PC, zorder=6)
    ax.text(lon_la + 0.25, lat_la + 0.18, "Los Angeles", transform=PC,
            fontsize=CITY_SIZE_PT + boost_pt, color=text_color,
            path_effects=effects, zorder=6)


def gridlines(ax, on_imagery: bool, cfg: dict) -> None:
    """Lat/lon graticule with labels on the left and bottom edges only."""
    gl = ax.gridlines(
        crs=PC, draw_labels=True, linewidth=0.4,
        color="white" if on_imagery else "#9a9a9a",
        alpha=0.5 if on_imagery else 0.6, linestyle=":",
        xlocs=cfg["xlocs_deg"], ylocs=cfg["ylocs_deg"],
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": cfg["tick_label_pt"], "color": INK}
    gl.ylabel_style = {"size": cfg["tick_label_pt"], "color": INK}


# ---------------------------------------------------------------------------
# Inset (aerial imagery)
# ---------------------------------------------------------------------------
def draw_inset(ax_in, boost_pt: float, site_boost_pt: float,
               site_codes: bool) -> None:
    ax_in.set_extent(INSET_EXTENT_DEG, crs=PC)
    add_imagery(ax_in, INSET_EXTENT_DEG, ZOOM_INSET)

    lab = dict(fontsize=INSET_SITE_SIZE_PT + site_boost_pt, color="white", path_effects=halo(2.0),
               zorder=8, linespacing=1.0)

    # Scripps Pier: gold star (matches the La Jolla star on the regional map)
    p = SITES["pier"]
    ax_in.plot(p["lon_deg"], p["lat_deg"], marker="*", ms=15, mfc=GOLD,
               mec="black", mew=0.8, transform=PC, zorder=9)
    ax_in.annotate("Scripps Pier" + ("\n(M1)" if site_codes else ""), xy=(p["lon_deg"], p["lat_deg"]),
                   xycoords=PC._as_mpl_transform(ax_in),
                   xytext=(-10, 6), textcoords="offset points",
                   ha="right", va="bottom", fontweight="bold", **lab)

    # Mt. Soledad: triangle (mountain-top site)
    s = SITES["soledad"]
    ax_in.plot(s["lon_deg"], s["lat_deg"], marker="^", ms=8.5,
               mfc=SOLEDAD_COLOR, mec="black", mew=0.8, transform=PC, zorder=9)
    ax_in.annotate("Mt. Soledad" + ("\n(S2)" if site_codes else ""), xy=(s["lon_deg"], s["lat_deg"]),
                   xycoords=PC._as_mpl_transform(ax_in),
                   xytext=(9, -4), textcoords="offset points",
                   ha="left", va="top", fontweight="bold", **lab)

    # Kearny Mesa APCD monitor: square
    k = SITES["kearny"]
    ax_in.plot(k["lon_deg"], k["lat_deg"], marker="s", ms=6.5,
               mfc=KEARNY_COLOR, mec="black", mew=0.8, transform=PC, zorder=9)
    ax_in.annotate("Kearny Mesa" + ("\n(SDAPCD)" if site_codes else ""), xy=(k["lon_deg"], k["lat_deg"]),
                   xycoords=PC._as_mpl_transform(ax_in),
                   # larger text would touch the inset's right frame, so
                   # shift it left in step with the size boost (0 in ver1)
                   xytext=(-4.0 * site_boost_pt, 8), textcoords="offset points",
                   ha="center", va="bottom", fontweight="bold", **lab)

    # Place names for orientation
    ax_in.text(-117.318, 32.822, "Pacific\nOcean", transform=PC, ha="left",
               va="center", fontsize=INSET_OCEAN_SIZE_PT + boost_pt,
               style="italic", color="#e6f2ff", linespacing=1.0,
               path_effects=halo(1.8), zorder=8)
    ax_in.text(-117.272, 32.8455, "La Jolla", transform=PC, ha="right",
               va="center", fontsize=INSET_TOWN_SIZE_PT + boost_pt,
               style="italic", color="white",
               path_effects=halo(1.8), zorder=8)

    scale_bar(ax_in, length_km=2.0)
    north_arrow(ax_in)

    ax_in.text(0.995, 0.008, ESRI_ATTRIBUTION, transform=ax_in.transAxes,
               ha="right", va="bottom", fontsize=4.5, color="white",
               path_effects=halo(1.0), zorder=8)

    # Double frame (white inside black) so the edge reads on any background
    for spine in ax_in.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2.2)
    ax_in.add_patch(Rectangle((0, 0), 1, 1, transform=ax_in.transAxes,
                              fill=False, ec="black", lw=0.8, zorder=10))


def scale_bar(ax_in, length_km: float) -> None:
    """Ground-distance scale bar in the lower-left corner of the inset.

    Web-Mercator distances are stretched by sec(latitude) relative to ground
    distance (Snyder 1987, Map Projections - A Working Manual, USGS PP 1395,
    eq. 7-4 with k = sec(phi)). The bar is therefore drawn with projected
    length = ground length / cos(lat) at the inset's central latitude. Over the
    inset's 0.12 deg of latitude the scale changes by < 0.2 %, so one bar is
    exact enough.
    """
    lat_c_deg = 0.5 * (INSET_EXTENT_DEG[2] + INSET_EXTENT_DEG[3])
    length_merc_m = length_km * 1e3 / np.cos(np.deg2rad(lat_c_deg))

    x0_m, x1_m, y0_m, y1_m = ax_in.get_extent(crs=MERC)
    xs_m = x0_m + 0.04 * (x1_m - x0_m)  # 4 % in from the left edge
    ys_m = y0_m + 0.07 * (y1_m - y0_m)  # 7 % up from the bottom edge
    tick_m = 0.012 * (y1_m - y0_m)

    # Alternating black/white segments, cartographic style
    n_seg = 2
    seg_m = length_merc_m / n_seg
    for i in range(n_seg):
        ax_in.add_patch(Rectangle(
            (xs_m + i * seg_m, ys_m), seg_m, tick_m * 1.6, transform=MERC,
            fc="white" if i % 2 == 0 else "black", ec="black", lw=0.6,
            zorder=9))
    # Tick labels centred on segment boundaries; the unit sits to the right
    # of the bar so it never collides with the last number.
    for frac in (0.0, 0.5, 1.0):
        ax_in.text(xs_m + frac * length_merc_m, ys_m + tick_m * 2.6,
                   f"{frac * length_km:g}", transform=MERC, ha="center",
                   va="bottom", fontsize=6.5, color="white",
                   path_effects=halo(1.5), zorder=9)
    ax_in.text(xs_m + length_merc_m * 1.06, ys_m + tick_m * 0.8, "km",
               transform=MERC, ha="left", va="center", fontsize=6.5,
               color="white", path_effects=halo(1.5), zorder=9)


def north_arrow(ax_in) -> None:
    """Simple north arrow (Web Mercator grid north = true north)."""
    ax_in.annotate("", xy=(0.955, 0.93), xytext=(0.955, 0.80),
                   xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                                   fc="white", ec="black", lw=0.8), zorder=9)
    ax_in.text(0.955, 0.75, "N", transform=ax_in.transAxes, ha="center",
               va="center", fontsize=8, fontweight="bold", color="white",
               path_effects=halo(1.6), zorder=9)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def make_figure(style: str, version: str, out_dir: Path) -> list[Path]:
    on_imagery = style == "satellite"
    cfg = VERSIONS[version]
    region_extent_deg = cfg["extent_deg"]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "savefig.dpi": DPI,
    })

    # Size the figure so the map axes fill it exactly at the map's projected
    # aspect ratio (no white bands from cartopy's equal-aspect constraint).
    # Margins leave room for the lat/lon labels, which grow with their size.
    pad = 0.0045 * (cfg["tick_label_pt"] - 8.0)
    left, right, bottom, top = 0.075 + pad, 0.985, 0.045 + pad, 0.99
    region_aspect = extent_aspect(region_extent_deg)  # height / width
    ax_w_in = FIG_WIDTH_IN * (right - left)
    ax_h_in = ax_w_in * region_aspect
    fig_h_in = ax_h_in / (top - bottom)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in))
    ax = fig.add_axes([left, bottom, right - left, top - bottom], projection=MERC)
    ax.set_extent(region_extent_deg, crs=PC)

    if on_imagery:
        draw_region_satellite(ax, region_extent_deg)
    else:
        draw_region_vector(ax)
    label_region(ax, on_imagery, cfg)
    gridlines(ax, on_imagery, cfg)

    # Gold star at La Jolla (Scripps Pier)
    p = SITES["pier"]
    ax.plot(p["lon_deg"], p["lat_deg"], marker="*", ms=17, mfc=GOLD,
            mec="black", mew=0.9, transform=PC, zorder=8)
    ax.text(p["lon_deg"] + 0.35, p["lat_deg"] + 0.15, "La Jolla", transform=PC,
            fontsize=SITE_TOWN_SIZE_PT + cfg["label_boost_pt"], fontweight="bold",
            color="white" if on_imagery else INK,
            path_effects=halo(1.8, "black" if on_imagery else "white"),
            ha="left", va="bottom", zorder=8)

    # Outline of the inset area on the regional map (true footprint)
    x0, y0 = merc_xy(INSET_EXTENT_DEG[0], INSET_EXTENT_DEG[2])
    x1, y1 = merc_xy(INSET_EXTENT_DEG[1], INSET_EXTENT_DEG[3])
    box_color = "white" if on_imagery else "black"
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=MERC,
                           fill=False, ec=box_color, lw=0.8, zorder=7))

    # Inset in the open ocean in the lower-left (south-west) corner. Its
    # height is set from the inset extent's projected aspect so the imagery
    # fills it edge to edge.
    inset_w_frac = 0.53
    inset_h_frac = inset_w_frac * extent_aspect(INSET_EXTENT_DEG) * ax_w_in / ax_h_in
    inset_bounds = [0.015, 0.018, inset_w_frac, inset_h_frac]  # axes fraction
    ax_in = ax.inset_axes(inset_bounds, projection=MERC, zorder=9)
    draw_inset(ax_in, cfg["label_boost_pt"], cfg["site_label_boost_pt"],
               cfg["site_codes"])

    # Leader (zoom) lines from inset corners to the footprint's corners
    corners_m = {"ll": (x0, y0), "lr": (x1, y0), "ul": (x0, y1), "ur": (x1, y1)}
    for (xa, ya), corner in cfg["leaders"]:
        con = ConnectionPatch(xyA=(xa, ya), coordsA=ax_in.transAxes,
                              xyB=corners_m[corner], coordsB=ax.transData,
                              color=box_color, lw=0.7, zorder=8)
        fig.add_artist(con)

    out_paths = []
    stem = f"EPCAPE_fig1_site_map_{style}{cfg['suffix']}"
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, facecolor="white")
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--style", choices=["satellite", "vector", "both"],
                        default="both")
    parser.add_argument("--version", choices=sorted(VERSIONS), default="ver1")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    styles = ["satellite", "vector"] if args.style == "both" else [args.style]
    for style in styles:
        for path in make_figure(style, args.version, args.out_dir):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
