#!/usr/bin/env python3
"""Monthly maps of surface longwave radiation and sea ice, August to December.

A 3-row grid, one column per month:

    row 1   downwelling longwave    msdwlwrf   always positive
    row 2   net longwave            msnlwrf    almost always negative
    row 3   sea ice concentration   siconc     0 to 1

Unlike the flux-panel scripts, **each row gets its own colorbar and its own kind
of colour scale**, because the three quantities are not comparable:

* Downwelling longwave is a magnitude with no meaningful zero in its range
  (roughly 150-300 W m-2 in the Arctic), so it takes a *sequential* ramp.
  Rendering it on a zero-centred diverging map would invent a crossing that the
  data does not have.
* Net longwave is signed and its zero crossing is physical -- the surface is
  losing or gaining energy -- so it takes a *diverging* map centred on zero.
* Sea ice concentration is a bounded fraction, drawn on a fixed 0-1 scale so
  colours mean the same thing in every panel and across runs.

By default **no ocean mask is applied**: land is plotted along with ocean, which
is what you want when the question is about the radiation field itself rather
than about the surface energy budget of the water. Pass ``--ocean-only`` to drop
land, and ``--max-siconc`` with it to additionally drop ice-covered water.

One thing that no option can change: ERA5 leaves ``siconc`` undefined over land,
so row 3 is blank there whatever the mask setting. That is the archive, not a
plotting choice.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**. Net
longwave is therefore negative wherever the surface radiates away more than it
receives, which in the Arctic is nearly everywhere.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.

Examples
--------
All data including land, August to December::

    python plot_monthly_longwave_maps.py --storage external --region barrow

Ocean only, and only where the water is essentially ice free::

    python plot_monthly_longwave_maps.py --storage external --region barrow \
        --ocean-only --max-siconc 0.15

Each month of the record separately instead of a climatology::

    python plot_monthly_longwave_maps.py --storage external --region barrow \
        --by-year-month
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np

from plot_monthly_flux_maps import _month_list, monthly_means, period_coverage
from plot_turbulent_flux_maps import _make_circular, domain_aspect
from seb_analysis_common import (
    DEFAULT_MAX_SICONC,
    add_data_source_args,
    build_ocean_mask,
    load_seb_data,
    parse_date,
    resolve_region_dir,
)

# (variable, row title, subtitle, scale kind, colormap)
#   sequential  one-sided magnitude, limits from robust percentiles
#   diverging   signed, limits symmetric about zero
#   fraction    bounded 0-1, fixed limits so panels are comparable
ROWS = (
    ("msdwlwrf", "Downwelling longwave", "msdwlwrf", "sequential", "magma"),
    ("msnlwrf", "Net longwave", "msnlwrf", "diverging", "RdBu_r"),
    ("siconc", "Sea ice concentration", "siconc", "fraction", "Blues"),
)

DEFAULT_MONTHS = (8, 9, 10, 11, 12)
COLOR_PERCENTILE = 98.0

# DOE ARM North Slope of Alaska site, Utqiagvik (formerly Barrow).
ARM_UTQIAGVIK_LAT = 71.32315013408275
ARM_UTQIAGVIK_LON = -156.61143018302457


def draw_site_marker(ax, transform=None, label: bool = False) -> None:
    """Mark the ARM Utqiagvik site so it reads on any of the three colour ramps.

    Drawn as a bullseye -- white disc, black ring, black centre -- rather than a
    single coloured dot. The three rows use magma (near-black at the low end),
    RdBu_r (saturated blue) and Blues (white at the low end), so no single fill
    colour keeps contrast everywhere; a light shape with a dark outline does.
    """
    kw = {} if transform is None else {"transform": transform}
    ax.plot(ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT, marker="o", markersize=9,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8,
            linestyle="none", zorder=8, **kw)
    ax.plot(ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT, marker="o", markersize=3,
            markerfacecolor="black", markeredgecolor="black",
            linestyle="none", zorder=9, **kw)
    if label:
        ax.annotate("Utqiaġvik (ARM)", xy=(ARM_UTQIAGVIK_LON, ARM_UTQIAGVIK_LAT),
                    xytext=(4, 5), textcoords="offset points", fontsize=7.5,
                    color="black", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75), **kw)


def row_limits(fields: list[np.ndarray], kind: str) -> tuple[float, float]:
    """Colour limits for one row, by scale kind.

    Computed across every month in the row so a colour means the same thing in
    all columns -- the entire point of the figure is comparing months.
    """
    if kind == "fraction":
        return 0.0, 1.0
    finite = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
    if finite.size == 0:
        return 0.0, 1.0
    if kind == "diverging":
        v = float(np.percentile(np.abs(finite), COLOR_PERCENTILE)) or 1.0
        return -v, v
    lo, hi = np.percentile(finite, [100 - COLOR_PERCENTILE, COLOR_PERCENTILE])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def figure_size(n_cols: int, aspect: float, is_circumpolar: bool) -> tuple[float, float]:
    """Figure size for a 3-row grid, sized from the domain's physical aspect."""
    panel_h = 2.7 if not is_circumpolar else 2.5
    panel_w = float(np.clip(panel_h * (1.0 if is_circumpolar else aspect), 1.2, 4.2))
    # Extra width on the right for the three row colorbars.
    return (n_cols * panel_w + 3.4, 3 * panel_h + 2.0)


def make_longwave_maps(
    monthly,
    labels: list[str],
    region: str,
    source_label: str,
    mask_label: str,
    coverage: dict[str, float] | None = None,
    years_per_period: dict[str, int] | None = None,
    per_month_scale: bool = False,
    show_site: bool = True,
    projection: str = "polar",
    output_path: Path | None = None,
    dpi: int = 150,
):
    """Draw the 3-row longwave/sea-ice grid and return the Figure."""
    import matplotlib.pyplot as plt

    use_cartopy = projection == "polar"
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            print("  cartopy unavailable; using a plain lat/lon grid.", file=sys.stderr)
            use_cartopy = False

    lat = monthly["latitude"].values
    lon = monthly["longitude"].values
    is_circumpolar = (lon.max() - lon.min()) >= 350.0
    n_cols = monthly.sizes["period"]

    proj_kw = {}
    if use_cartopy:
        central = 0.0 if is_circumpolar else float(np.mean([lon.min(), lon.max()]))
        proj_kw = {"projection": ccrs.NorthPolarStereo(central_longitude=central)}
        data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        len(ROWS), n_cols,
        figsize=figure_size(n_cols, domain_aspect(lat, lon), is_circumpolar and use_cartopy),
        subplot_kw=proj_kw, squeeze=False, constrained_layout=True,
    )

    for i_row, (name, title, subtitle, kind, cmap) in enumerate(ROWS):
        fields = [monthly[name].isel(period=i).values for i in range(n_cols)]
        vmin, vmax = row_limits(fields, kind)

        mesh = None
        for i_col in range(n_cols):
            ax = axes[i_row][i_col]
            lo, hi = (vmin, vmax)
            if per_month_scale and kind != "fraction":
                # Each panel on its own limits: reveals within-month structure
                # that a shared scale crushes (downwelling longwave spans
                # ~180-310 W m-2 across the season, so December occupies only
                # the bottom sliver of a shared ramp). Costs comparability
                # BETWEEN columns, which is why it is not the default.
                lo, hi = row_limits([fields[i_col]], kind)
            plot_kw = dict(cmap=cmap, vmin=lo, vmax=hi, shading="auto")
            if use_cartopy:
                plot_kw["transform"] = data_crs
            mesh = ax.pcolormesh(lon, lat, fields[i_col], **plot_kw)

            if use_cartopy:
                if is_circumpolar:
                    ax.set_extent([-180, 180, lat.min(), 90], crs=data_crs)
                    _make_circular(ax)
                else:
                    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                                  crs=data_crs)
                # Land drawn UNDER the data so that where a field is defined over
                # land (rows 1 and 2) the values show, and where it is not
                # (row 3, sea ice) the grey shows through.
                ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
                ax.coastlines(resolution="50m", linewidth=0.45, color="0.35", zorder=3)
                ax.gridlines(linewidth=0.3, color="0.75", alpha=0.6, zorder=4)
                if show_site:
                    draw_site_marker(ax, transform=data_crs, label=(i_row == 0 and i_col == 0))
            else:
                ax.tick_params(labelsize=7, colors="0.4")
                if show_site:
                    draw_site_marker(ax, label=(i_row == 0 and i_col == 0))

            if i_row == 0:
                cov = (coverage or {}).get(labels[i_col], 1.0)
                head = labels[i_col]
                extra = []
                if years_per_period and labels[i_col] in years_per_period:
                    n_y = years_per_period[labels[i_col]]
                    extra.append(f"{n_y} yr" + ("s" if n_y != 1 else ""))
                if cov < 0.99:
                    extra.append(f"{cov*100:.0f}% of days")
                if extra:
                    head += "\n(" + ", ".join(extra) + ")"
                ax.set_title(head, fontsize=10, pad=6,
                             color="0.45" if cov < 0.99 else "black")
            if i_col == 0:
                ax.text(-0.13, 0.5, f"{title}\n{subtitle}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9.5)

        cb = fig.colorbar(mesh, ax=list(axes[i_row]), location="right",
                          pad=0.015, shrink=0.88, aspect=18)
        if kind == "fraction":
            cb.set_label("fraction (0\u20131)", fontsize=9)
        elif per_month_scale:
            # The bar shows only the LAST panel's range when each differs.
            cb.set_label("W m$^{-2}$ (per-panel scale)", fontsize=9)
        else:
            cb.set_label("W m$^{-2}$", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"ERA5 surface longwave radiation and sea ice — {region}\n"
        f"{source_label}   |   {mask_label}   |   fluxes positive downward",
        fontsize=12,
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
    parser.add_argument("--start", type=parse_date, default=None)
    parser.add_argument("--end", type=parse_date, default=None)
    parser.add_argument("--months", type=_month_list, default=DEFAULT_MONTHS,
                        metavar="M[,M...]",
                        help="Calendar months to show (default 8,9,10,11,12 = Aug-Dec). "
                             "Pass 'all' for every month present.")
    parser.add_argument("--ocean-only", action="store_true",
                        help="Drop land. OFF by default: land is plotted along with "
                             "ocean, since the longwave field is defined there too. "
                             "Note sea ice is undefined over land regardless, so row 3 "
                             "is blank there either way.")
    parser.add_argument("--max-siconc", type=float, default=None, metavar="F",
                        help="With --ocean-only, additionally drop water whose sea ice "
                             f"fraction is at or above F (e.g. {DEFAULT_MAX_SICONC} to "
                             "keep only open water). Ignored without --ocean-only.")
    parser.add_argument("--no-site-marker", action="store_true",
                        help="Do not mark the DOE ARM Utqiagvik site.")
    parser.add_argument("--per-month-scale", action="store_true",
                        help="Give each panel its own colour limits instead of one scale "
                             "per row. Reveals within-month structure that the shared "
                             "scale crushes (Nov/Dec downwelling longwave sits at the "
                             "dark end of a 180-310 W m-2 ramp), but panels are then no "
                             "longer comparable to each other.")
    parser.add_argument("--by-year-month", action="store_true",
                        help="One column per calendar month in the record instead of a "
                             "climatology averaged across years.")
    parser.add_argument("--projection", choices=("polar", "platecarree"), default="polar")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 monthly longwave and sea ice maps")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, args.start, args.end, region_dir.parent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    missing = [n for n, _, _, _, _ in ROWS if n not in ds]
    if missing:
        print(f"  Error: dataset is missing {missing}.", file=sys.stderr)
        return 1

    print(f"  Source     : {region_dir}")
    print(f"  Region     : {args.region}")
    print(f"  Grid       : {ds.sizes['latitude']} lat x {ds.sizes['longitude']} lon")

    # --- Month filter, applied before any averaging --------------------------
    if len(args.months) < 12:
        keep_t = ds["valid_time"].dt.month.isin(list(args.months))
        n_before = ds.sizes["valid_time"]
        ds = ds.sel(valid_time=keep_t)
        if ds.sizes["valid_time"] == 0:
            print(f"  Error: no data in months {list(args.months)}.", file=sys.stderr)
            return 1
        names = ", ".join(calendar.month_abbr[m] for m in sorted(args.months))
        print(f"  Months     : {names} "
              f"({ds.sizes['valid_time']:,} of {n_before:,} steps kept)")

    ds = ds.compute()

    # --- Masking (off by default) -------------------------------------------
    if args.ocean_only:
        mode = "open-ocean" if args.max_siconc is not None else "all-ocean"
        thr = args.max_siconc if args.max_siconc is not None else DEFAULT_MAX_SICONC
        mask, report = build_ocean_mask(ds, mode, thr)
        print(f"\n  Masking ({mode}):")
        print(report.describe())
        mask_label = ("ocean only, including sea ice" if mode == "all-ocean"
                      else f"open ocean only (sea ice fraction < {thr:g})")
        fields = ds[[n for n, _, _, _, _ in ROWS]].where(mask)
    else:
        print("\n  Masking    : none — land and ocean both plotted")
        print("               (sea ice is undefined over land, so row 3 is blank there)")
        mask_label = "all grid cells, land included"
        fields = ds[[n for n, _, _, _, _ in ROWS]]

    monthly, ice_m, labels = monthly_means(fields, ds["siconc"], args.by_year_month)
    monthly = monthly.compute()
    coverage = period_coverage(ds, args.by_year_month)

    # How many distinct years contribute to each column. A climatology whose
    # August averages three years and October two is not uniformly sampled, and
    # that is invisible from a day-coverage number alone.
    years_per_period: dict[str, int] = {}
    if not args.by_year_month:
        t = ds["valid_time"].values
        mo = t.astype("datetime64[M]").astype(int) % 12 + 1
        yr = t.astype("datetime64[Y]").astype(int) + 1970
        for m in np.unique(mo):
            years_per_period[calendar.month_abbr[int(m)]] = len(np.unique(yr[mo == m]))

    print(f"\n  Periods    : {len(labels)}  ({', '.join(labels)})")
    if years_per_period and len(set(years_per_period.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in years_per_period.items())
        print(f"  !! Uneven sampling across columns: years per month {detail}.")
        print("     Columns are not equally well constrained; each panel is labelled.")

    print("\n  Area-mean by period:")
    hdr = (f"    {'period':<9}{'LW down':>11}{'LW net':>10}{'sea ice':>10}"
           f"{'cells':>9}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for i, lab in enumerate(labels):
        row = f"    {lab:<9}"
        n_cells = 0
        for name, _, _, _, _ in ROWS:
            v = monthly[name].isel(period=i).values
            v = v[np.isfinite(v)]
            n_cells = max(n_cells, v.size)
            row += f"{v.mean():>10.2f} " if v.size else f"{'--':>10} "
        print(row + f"{n_cells:>8,}")

    yrs = sorted({int(y) + 1970
                  for y in ds["valid_time"].values.astype("datetime64[Y]").astype(int)})
    span = f"{yrs[0]}" if len(yrs) == 1 else f"{yrs[0]}–{yrs[-1]}"
    source_label = (f"individual months, {span}" if args.by_year_month
                    else f"climatology over {span}")

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        kind = "bymonth" if args.by_year_month else "climatology"
        suffix = "_oceanonly" if args.ocean_only else ""
        output_path = fig_dir / f"{args.region}_monthly_longwave_{kind}{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    make_longwave_maps(
        monthly, labels,
        region=args.region, source_label=source_label, mask_label=mask_label,
        coverage=coverage, years_per_period=years_per_period,
        per_month_scale=args.per_month_scale, show_site=not args.no_site_marker,
        projection=args.projection, output_path=output_path, dpi=args.dpi,
    )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
