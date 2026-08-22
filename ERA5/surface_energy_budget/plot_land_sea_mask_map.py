#!/usr/bin/env python3
"""Spatial map of the ERA5 land-sea mask for one region.

Every grid cell is coloured by ``lsm``, the land FRACTION of the box (0 =
open ocean, 1 = pure land; see ``analyze_land_sea_mask.py``), on a custom
two-colour ramp running from blue at 0 to brown at 1. Coastal cells -- the
boxes a real coastline runs through -- are neither, so they land on the
blend between the two and read visually as "mixed" without any extra
symbology. A black contour at lsm = 0.5, ECMWF's own recommended cut for a
binary land/sea decision, marks where the ramp crosses the midpoint.

Reads what ``download_era5_land_sea_mask.py`` writes, via the same
``load_land_sea_mask`` helper ``surface_classification.py`` uses, so this map
and the class-based figures elsewhere in this analysis are always looking at
the same file. The printed cell/area breakdown mirrors the core table in
``analyze_land_sea_mask.py``; run that script directly for the deeper
diagnostics (mixed-cell histogram, per-latitude-band breakdown) this one
leaves out.

Examples
========
  ./plot_land_sea_mask_map.py --region barrow
  ./plot_land_sea_mask_map.py --region arctic_circle
  ./plot_land_sea_mask_map.py --region barrow --projection platecarree --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analyze_land_sea_mask import BINARY_THRESHOLD, DEFAULT_TOL, area_weights_2d, classify, _pct
from download_era5_seb import STORAGE_ROOTS
from era5_seb_variables import get_region, region_names
from plot_monthly_longwave_maps import ARM_UTQIAGVIK_LAT, ARM_UTQIAGVIK_LON, draw_site_marker
from plot_turbulent_flux_maps import _make_circular, domain_aspect
from seb_analysis_common import resolve_data_root
from surface_classification import load_land_sea_mask

# Ocean at lsm=0, land at lsm=1. Same two hues surface_classification.py's
# CLASS_COLORS uses for "open_ocean" and "land", so this map and the
# surface-class figures elsewhere in this analysis read as one palette.
LAND_SEA_CMAP = LinearSegmentedColormap.from_list(
    "ocean_to_land", ["#1f5fa8", "#8c6d4f"]
)


def print_summary(lsm: np.ndarray, lat_deg: np.ndarray, tol: float) -> None:
    """Cell counts and area shares for sea/land/mixed.

    A trimmed version of ``analyze_land_sea_mask.report_one``'s core table --
    same categories, same area weighting -- without its histogram and
    latitude-band diagnostics.
    """
    n_lat, n_lon = lsm.shape
    finite = np.isfinite(lsm)
    is_sea, is_land, is_mixed = classify(lsm, tol)
    is_sea &= finite
    is_land &= finite
    is_mixed &= finite

    weights = area_weights_2d(lat_deg, n_lon)
    w_total = float(weights[finite].sum())
    n_finite = int(finite.sum())

    print(f"  Grid       : {n_lat} x {n_lon} = {n_finite:,} cells")
    print(f"  {'category':<12}{'cells':>10}{'% cells':>10}{'% area':>10}")
    print("  " + "-" * 42)
    for label, mask in (("sea only", is_sea), ("land only", is_land), ("mixed", is_mixed)):
        n = int(mask.sum())
        w = float(weights[mask].sum())
        print(f"  {label:<12}{n:>10,}{_pct(n, n_finite):>9.2f}%{_pct(w, w_total):>9.2f}%")

    land_frac = float(np.nansum(lsm * weights)) / w_total if w_total else 0.0
    print(f"  Land fraction of domain surface (area-weighted): {100 * land_frac:.2f}%")
    print("  (full breakdown, incl. mixed-cell histogram: ./analyze_land_sea_mask.py)")


def make_map(
    lsm_da,
    region: str,
    tol: float,
    show_coastline_contour: bool,
    projection: str,
    show_site: bool,
    output_path: Path | None = None,
    dpi: int = 150,
):
    """Draw the single-panel land-sea mask map and save it if asked."""
    import matplotlib.pyplot as plt

    use_cartopy = projection == "polar"
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature  # noqa: F401 (kept for parity with other map scripts)
        except ImportError:
            print("  cartopy unavailable; using a plain lat/lon grid.", file=sys.stderr)
            use_cartopy = False

    lat_deg = lsm_da["latitude"].values.astype(np.float64)
    lon_deg = lsm_da["longitude"].values.astype(np.float64)
    lsm = lsm_da.values.astype(np.float64)

    is_circumpolar = (lon_deg.max() - lon_deg.min()) >= 350.0

    proj_kw = {}
    data_crs = None
    if use_cartopy:
        central = 0.0 if is_circumpolar else float(np.mean([lon_deg.min(), lon_deg.max()]))
        proj_kw = {"projection": ccrs.NorthPolarStereo(central_longitude=central)}
        data_crs = ccrs.PlateCarree()

    if is_circumpolar and use_cartopy:
        figsize = (9.0, 8.0)
    else:
        panel_h = 6.5
        panel_w = float(np.clip(panel_h * domain_aspect(lat_deg, lon_deg), 4.0, 9.0))
        figsize = (panel_w + 2.2, panel_h + 1.0)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=proj_kw, constrained_layout=True)

    plot_kw = dict(cmap=LAND_SEA_CMAP, vmin=0.0, vmax=1.0, shading="auto")
    if use_cartopy:
        plot_kw["transform"] = data_crs
    mesh = ax.pcolormesh(lon_deg, lat_deg, lsm, **plot_kw)

    if show_coastline_contour:
        lo, hi = float(np.nanmin(lsm)), float(np.nanmax(lsm))
        if lo < BINARY_THRESHOLD < hi:
            ckw = dict(colors="black", linewidths=1.0, zorder=5)
            if use_cartopy:
                ckw["transform"] = data_crs
            ax.contour(lon_deg, lat_deg, lsm, levels=[BINARY_THRESHOLD], **ckw)

    if use_cartopy:
        if is_circumpolar:
            ax.set_extent([-180, 180, lat_deg.min(), 90], crs=data_crs)
            _make_circular(ax)
        else:
            ax.set_extent(
                [lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()], crs=data_crs
            )
        ax.coastlines(resolution="50m", linewidth=0.8, color="black", zorder=4)
        gl_kw = dict(linewidth=0.3, color="0.4", alpha=0.6, zorder=3)
        if not is_circumpolar:
            gl_kw["draw_labels"] = True
        ax.gridlines(**gl_kw)
    else:
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.tick_params(labelsize=8)

    site_in_domain = (
        lon_deg.min() <= ARM_UTQIAGVIK_LON <= lon_deg.max()
        and lat_deg.min() <= ARM_UTQIAGVIK_LAT <= lat_deg.max()
    )
    if show_site and site_in_domain:
        kw = {"transform": data_crs} if use_cartopy else {}
        draw_site_marker(ax, label=True, **kw)

    cb = fig.colorbar(mesh, ax=ax, location="right", pad=0.03, shrink=0.85, aspect=25)
    cb.set_label("Land fraction (lsm)", fontsize=10)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(
        ["0.0\nopen ocean", "0.25", f"{BINARY_THRESHOLD:g}\ncoastal cut", "0.75", "1.0\npure land"]
    )
    cb.ax.tick_params(labelsize=8)

    region_desc = get_region(region).description if region != "custom" else "custom region"
    fig.suptitle(
        f"ERA5 land-sea mask — {region}\n{region_desc}"
        + ("   |   coastline: lsm = 0.5 contour" if show_coastline_contour else ""),
        fontsize=12,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  -> {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--region", choices=region_names(), default="barrow",
        help="Region mask to map (default: barrow).",
    )
    parser.add_argument(
        "--storage", choices=sorted(STORAGE_ROOTS), default="local",
        help="Which root the mask was written to (default: local).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None, metavar="PATH",
        help="Explicit directory holding the mask file, overriding --storage.",
    )
    parser.add_argument(
        "--grid", type=float, default=None, metavar="DEG",
        help="Read the mask saved at this regridded resolution instead of native.",
    )
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL, metavar="TOL",
        help=f"How close lsm must be to 0 or 1 to count as pure sea/land in the "
             f"printed summary (default {DEFAULT_TOL:g}); packing round-off "
             "insurance, not physics.",
    )
    parser.add_argument(
        "--projection", choices=("polar", "platecarree"), default="polar",
        help="'polar' (default) uses cartopy's north polar stereographic "
             "projection with coastlines; 'platecarree' falls back to a plain "
             "lat/lon grid with no basemap.",
    )
    parser.add_argument(
        "--no-coastline-contour", action="store_true",
        help="Do not draw the lsm=0.5 contour line.",
    )
    parser.add_argument(
        "--no-site-marker", action="store_true",
        help="Do not mark the DOE ARM Utqiagvik site, even when it falls inside "
             "the mapped domain.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Land-sea mask map")
    print("=" * 72)

    root = resolve_data_root(args.storage, args.data_root)
    try:
        lsm_da = load_land_sea_mask(args.region, root, args.grid)
    except (FileNotFoundError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    print(f"  Region     : {args.region}")
    print(f"  Source     : {root}")
    print()
    print_summary(
        lsm_da.values.astype(np.float64), lsm_da["latitude"].values.astype(np.float64), args.tol
    )

    output_path = args.output
    if output_path is None:
        out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_tag = f"_{args.grid:.2f}deg".replace(".", "p") if args.grid is not None else ""
        output_path = out_dir / f"{args.region}_land_sea_mask_map{grid_tag}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_map(
        lsm_da, region=args.region, tol=args.tol,
        show_coastline_contour=not args.no_coastline_contour,
        projection=args.projection, show_site=not args.no_site_marker,
        output_path=output_path, dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
