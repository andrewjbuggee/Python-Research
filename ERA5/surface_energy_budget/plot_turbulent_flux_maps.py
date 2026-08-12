#!/usr/bin/env python3
"""Spatial maps of Arctic turbulent heat fluxes over open ocean, 1 row x 3 columns.

Panels, left to right: net turbulent flux (SH + LH), sensible heat flux, latent
heat flux. Each panel is the time mean over a user-specified date range.

Only ocean grid cells are shown. ERA5 leaves sea ice cover undefined over land,
so land is dropped automatically; sea-ice-covered water is dropped according to
``--mask`` and ``--max-siconc``.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**. A positive
value means the turbulent fluxes warm the surface; negative means the surface
loses heat to the atmosphere. This is the opposite of the Sledd et al. (2025)
Eq. (1) convention used in ``seb_terms.py``.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
These are the same two roots the downloader writes to, imported rather than
duplicated, so editing that constant moves both halves at once. ``--data-root``
overrides both with an explicit path.

Examples
--------
Time mean over the first week of January, open ocean only::

    python plot_turbulent_flux_maps.py --region barrow \
        --start 2026-01-01 --end 2026-01-07

Read a long record from the external drive::

    python plot_turbulent_flux_maps.py --storage external --region barrow \
        --start 2000-01-01 --end 2025-12-31 --mask all-ocean

Keep ice-covered ocean too, which in an Arctic winter is most of the domain::

    python plot_turbulent_flux_maps.py --region barrow \
        --start 2026-01-01 --end 2026-02-17 --mask all-ocean

Loosen the ice threshold instead of abandoning it::

    python plot_turbulent_flux_maps.py --region barrow --max-siconc 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    FLUX_PANELS,
    add_data_source_args,
    build_ocean_mask,
    compute_turbulent_fluxes,
    load_seb_data,
    parse_date,
    resolve_region_dir,
    warn_if_sparse,
)

# Diverging colormap with a neutral midpoint, appropriate for a signed quantity
# whose zero crossing is physically meaningful (surface gaining vs losing heat).
# RdBu_r is two-hue warm/cool and stays separable under the common forms of
# colour vision deficiency. A rainbow map would invent structure that is not
# in the data.
DIVERGING_CMAP = "RdBu_r"

# Robust percentile for the symmetric colour limits, so a handful of extreme
# cells cannot flatten the rest of the map.
COLOR_LIMIT_PERCENTILE = 98.0


def symmetric_limits(
    arrays: list[np.ndarray], percentile: float = COLOR_LIMIT_PERCENTILE
) -> float:
    """Symmetric colour limit ``v`` such that the scale runs -v .. +v."""
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if finite.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(finite), percentile))
    return v if v > 0 else float(np.max(np.abs(finite))) or 1.0


def domain_aspect(lat: np.ndarray, lon: np.ndarray) -> float:
    """Physical width/height of the domain, for sizing the figure sensibly.

    A lat/lon box is not square on the ground: 15 deg of longitude at 75N spans
    about 430 km while 10 deg of latitude spans about 1110 km. Sizing the figure
    from the degree extent alone gives tall panels stranded in white space.
    """
    height_km = (lat.max() - lat.min()) * 111.0
    mean_lat_rad = np.deg2rad(0.5 * (lat.max() + lat.min()))
    width_km = (lon.max() - lon.min()) * 111.0 * np.cos(mean_lat_rad)
    if height_km <= 0:
        return 1.0
    return float(width_km / height_km)


def figure_size(aspect: float, is_circumpolar: bool) -> tuple[float, float]:
    """Figure size in inches for a 1x3 panel row of the given panel aspect."""
    if is_circumpolar:
        return (15.0, 6.2)
    panel_height_in = 5.0
    # Clamp so a very thin or very wide domain still yields a usable figure.
    panel_width_in = float(np.clip(panel_height_in * aspect, 2.0, 6.0))
    fig_width = 3 * panel_width_in + 2.2  # margins plus colorbar breathing room
    fig_height = panel_height_in + 2.0  # title block and horizontal colorbar
    return (fig_width, fig_height)


def _make_circular(ax) -> None:
    """Clip a polar-stereographic axis to a circle, the usual circumpolar view."""
    import matplotlib.path as mpath

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.column_stack([0.5 + 0.5 * np.sin(theta),
                                         0.5 + 0.5 * np.cos(theta)]))
    ax.set_boundary(circle, transform=ax.transAxes)


def make_maps(
    means,
    region: str,
    time_label: str,
    mask_label: str,
    shared_scale: bool = True,
    projection: str = "polar",
    output_path: Path | None = None,
    dpi: int = 150,
    panels=FLUX_PANELS,
    quantity_noun: str = "turbulent heat fluxes",
    cbar_quantity: str = "Turbulent heat flux",
):
    """Draw the 1xN map figure and return the matplotlib Figure.

    ``panels`` is a sequence of ``(var_name, title, subtitle)`` naming variables
    in ``means``; the turbulent set is the default and the radiative maps script
    passes RADIATIVE_PANELS instead.
    """
    import matplotlib.pyplot as plt

    use_cartopy = projection == "polar"
    proj_kw = {}
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            print(
                "  cartopy unavailable; falling back to a plain lat/lon grid.",
                file=sys.stderr,
            )
            use_cartopy = False

    lat = means["latitude"].values
    lon = means["longitude"].values

    # A domain spanning essentially all longitudes (e.g. the arctic_circle
    # region) is a polar cap, not a box: draw it as a full circular view rather
    # than trying to fit a rectangular extent around the pole.
    is_circumpolar = (lon.max() - lon.min()) >= 350.0

    if use_cartopy:
        central_lon = 0.0 if is_circumpolar else float(np.mean([lon.min(), lon.max()]))
        map_proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        proj_kw = {"projection": map_proj}
        data_crs = ccrs.PlateCarree()

    aspect = domain_aspect(lat, lon)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=figure_size(aspect, is_circumpolar and use_cartopy),
        subplot_kw=proj_kw,
        constrained_layout=True,
    )

    fields = [means[name].values for name, _, _ in panels]
    if shared_scale:
        vmax = symmetric_limits(fields)
        limits = [vmax] * len(panels)
    else:
        limits = [symmetric_limits([f]) for f in fields]

    meshes = []
    for ax, (name, title, subtitle), field, vmax in zip(
        axes, panels, fields, limits
    ):
        plot_kw = dict(
            cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="auto"
        )
        if use_cartopy:
            plot_kw["transform"] = data_crs
        mesh = ax.pcolormesh(lon, lat, field, **plot_kw)
        meshes.append(mesh)

        if use_cartopy:
            if is_circumpolar:
                ax.set_extent([-180, 180, lat.min(), 90], crs=data_crs)
                _make_circular(ax)
            else:
                ax.set_extent(
                    [lon.min(), lon.max(), lat.min(), lat.max()], crs=data_crs
                )
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
            ax.coastlines(resolution="50m", linewidth=0.6, color="0.35", zorder=3)
            gl = ax.gridlines(
                draw_labels=not is_circumpolar,
                linewidth=0.4,
                color="0.7",
                alpha=0.7,
                zorder=4,
            )
            if not is_circumpolar:
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {"size": 8, "color": "0.35"}
                gl.ylabel_style = {"size": 8, "color": "0.35"}
        else:
            ax.set_xlabel("Longitude [deg]", fontsize=9)
            ax.set_ylabel("Latitude [deg]", fontsize=9)
            ax.tick_params(labelsize=8, colors="0.35")
            for spine in ax.spines.values():
                spine.set_color("0.8")

        n_valid = int(np.isfinite(field).sum())
        ax.set_title(
            f"{title}\n{subtitle}  |  {n_valid:,} of {field.size:,} cells",
            fontsize=11,
            pad=10,
        )

        if not shared_scale:
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05,
                              shrink=0.9)
            cb.set_label("W m$^{-2}$ (positive downward)", fontsize=9)
            cb.ax.tick_params(labelsize=8)

    if shared_scale:
        cb = fig.colorbar(
            meshes[0], ax=axes, orientation="horizontal", pad=0.04, shrink=0.55,
            aspect=40,
        )
        cb.set_label(
            f"{cbar_quantity} [W m$^{{-2}}$]   —   positive downward, into the surface",
            fontsize=10,
        )
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"ERA5 {quantity_noun} over ocean — {region}\n"
        f"{time_label}   |   {mask_label}",
        fontsize=13,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  Figure written to {output_path}")
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--start", type=parse_date, default=None,
                        help="First time, YYYY-MM-DD[THH]. Default: start of record.")
    parser.add_argument("--end", type=parse_date, default=None,
                        help="Last time, inclusive, YYYY-MM-DD[THH]. Default: end of record.")
    parser.add_argument("--mask", choices=("open-ocean", "all-ocean"),
                        default="open-ocean",
                        help="open-ocean drops land and sea ice (default); "
                             "all-ocean drops only land.")
    parser.add_argument("--max-siconc", type=float, default=DEFAULT_MAX_SICONC,
                        help=f"Sea ice fraction below which a cell is open water "
                             f"(default {DEFAULT_MAX_SICONC}). Only used with --mask open-ocean.")
    parser.add_argument("--per-panel-scale", action="store_true",
                        help="Give each panel its own colour scale instead of one shared scale.")
    parser.add_argument("--projection", choices=("polar", "platecarree"), default="polar",
                        help="polar = North Polar Stereographic with coastlines (default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path. Default: figures/<region>_turbulent_flux_maps.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 turbulent flux maps")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, args.start, args.end, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    print(f"  Source     : {region_dir}")
    t0 = str(ds.valid_time.values[0])[:16].replace("T", " ")
    t1 = str(ds.valid_time.values[-1])[:16].replace("T", " ")
    n_times = ds.sizes["valid_time"]
    print(f"  Region     : {args.region}")
    print(f"  Time range : {t0} to {t1} UTC  ({n_times} hourly steps)")
    print(f"  Grid       : {ds.sizes['latitude']} lat x {ds.sizes['longitude']} lon")

    mask, report = build_ocean_mask(ds, args.mask, args.max_siconc)
    print(f"\n  Masking ({args.mask}):")
    print(report.describe())
    warn_if_sparse(report)

    fluxes = compute_turbulent_fluxes(ds, mask)
    # Time mean at each grid cell, over the unmasked hours only. A cell that is
    # never open water stays NaN and is left blank on the map.
    means = fluxes.mean(dim="valid_time", skipna=True).compute()

    print("\n  Time-mean flux over retained cells [W m-2, positive downward]:")
    for name, title, _ in FLUX_PANELS:
        v = means[name].values
        f = v[np.isfinite(v)]
        if f.size:
            print(f"    {title:<22} mean={f.mean():>8.2f}  min={f.min():>8.2f}  "
                  f"max={f.max():>8.2f}  cells={f.size:,}")
        else:
            print(f"    {title:<22} no valid cells")

    time_label = f"time mean, {t0} to {t1} UTC ({n_times} h)"
    if args.mask == "open-ocean":
        mask_label = f"open ocean only (sea ice fraction < {args.max_siconc:g})"
    else:
        mask_label = "all ocean, including sea ice"

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        output_path = fig_dir / f"{args.region}_turbulent_flux_maps.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    fig = make_maps(
        means,
        region=args.region,
        time_label=time_label,
        mask_label=mask_label,
        shared_scale=not args.per_panel_scale,
        projection=args.projection,
        output_path=output_path,
        dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
