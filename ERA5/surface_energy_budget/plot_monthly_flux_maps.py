#!/usr/bin/env python3
"""Monthly spatial maps of Arctic turbulent heat fluxes, one panel per month.

Builds a grid of maps showing net turbulent flux (SH + LH), sensible heat flux
and latent heat flux for every month in the record, so the seasonal progression
— including the open-ocean to sea-ice transition — is visible in one figure.

By default the months are a CLIMATOLOGY: every January in the record is averaged
together, every February, and so on. Use ``--by-year-month`` to keep each
individual month separate instead.

The sea-ice edge is drawn on every panel as a contour of the monthly-mean sea
ice fraction, which is what makes the freeze-up and melt-back progression legible
alongside the fluxes.

Works on hourly, daily, or monthly-mean input files: months are formed with a
groupby on the time axis, so any of the three resolutions gives the same answer
for an unmasked average.

SIGN CONVENTION: native ERA5, **positive downward (into the surface)**.

Data location
-------------
``--storage local`` (the default) reads ``data/`` beside this script;
``--storage external`` reads ``EXTERNAL_ROOT`` from ``download_era5_seb.py``.
``--data-root`` overrides both with an explicit path.

Examples
--------
Climatological months from whatever is in data/barrow::

    python plot_monthly_flux_maps.py --region barrow --mask all-ocean

A 26-year climatology from the external drive::

    python plot_monthly_flux_maps.py --storage external --region barrow \
        --start 2000-01-01 --end 2025-12-31 --mask all-ocean

Monthly means downloaded with --frequency monthly::

    python plot_monthly_flux_maps.py --region barrow_monthly --mask all-ocean

Each individual month rather than a climatology, in the original 3-column layout::

    python plot_monthly_flux_maps.py --region barrow --by-year-month \
        --layout months-as-rows --mask all-ocean
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np

from plot_turbulent_flux_maps import (
    COLOR_LIMIT_PERCENTILE,
    DIVERGING_CMAP,
    _make_circular,
    domain_aspect,
    symmetric_limits,
)
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

# Sea ice fraction contoured on every panel to mark the ice edge.
ICE_EDGE_LEVELS = (0.15, 0.80)

MAX_FIG_WIDTH_IN = 22.0
MAX_FIG_HEIGHT_IN = 24.0


def _month_list(text: str):
    """Parse ``8,9,10,11`` or ``all`` into a month tuple."""
    if text.strip().lower() == "all":
        return tuple(range(1, 13))
    try:
        months = tuple(int(t) for t in text.split(",") if t.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated month list")
    bad = [m for m in months if not 1 <= m <= 12]
    if bad or not months:
        raise argparse.ArgumentTypeError(f"months out of range: {bad or 'empty list'}")
    return months


def period_coverage(ds, by_year_month: bool) -> dict[str, float]:
    """Fraction of each period's calendar days that are actually present.

    A month at the edge of the record is usually incomplete. Averaging three
    days of March into a "March" climatology panel and drawing it next to a full
    January is misleading, so coverage is measured and any shortfall is written
    onto the panel label.

    Keyed by period label; 1.0 means every calendar day of the period is present.
    """
    raw_times = ds["valid_time"].values
    # Input that is ALREADY monthly carries one sample per month, which the
    # day-counting test below would score as ~3% coverage. Detect that case from
    # the sample spacing and treat every month as complete.
    if raw_times.size > 1:
        spacing_days = np.median(np.diff(raw_times).astype("timedelta64[h]").astype(float)) / 24.0
        if spacing_days >= 27.0:
            if by_year_month:
                return {
                    f"{t.astype(object).year}-{t.astype(object).month:02d}": 1.0
                    for t in raw_times.astype("datetime64[M]")
                }
            return {
                calendar.month_abbr[int(m)]: 1.0
                for m in np.unique(raw_times.astype("datetime64[M]").astype(int) % 12 + 1)
            }

    times = raw_times.astype("datetime64[D]")
    years = times.astype("datetime64[Y]").astype(int) + 1970
    months = times.astype("datetime64[M]").astype(int) % 12 + 1

    # Distinct days actually present, per (year, month).
    present: dict[tuple[int, int], set] = {}
    for y, m, t in zip(years, months, times):
        present.setdefault((int(y), int(m)), set()).add(t)

    if by_year_month:
        return {
            f"{y}-{m:02d}": len(days) / calendar.monthrange(y, m)[1]
            for (y, m), days in present.items()
        }

    # Climatology: pool every year that contributes to this calendar month.
    got: dict[int, int] = {}
    expected: dict[int, int] = {}
    for (y, m), days in present.items():
        got[m] = got.get(m, 0) + len(days)
        expected[m] = expected.get(m, 0) + calendar.monthrange(y, m)[1]
    return {
        calendar.month_abbr[m]: got[m] / expected[m] for m in sorted(got)
    }


def monthly_means(fluxes, siconc, by_year_month: bool):
    """Collapse the time axis to months.

    Returns ``(flux_means, ice_means, labels)`` where the first two are indexed
    by a single ``period`` dimension and ``labels`` names each period.
    """
    if by_year_month:
        flux_m = fluxes.resample(valid_time="1MS").mean()
        ice_m = siconc.resample(valid_time="1MS").mean()
        labels = [
            f"{np.datetime64(t, 'M').astype(object).year}-"
            f"{np.datetime64(t, 'M').astype(object).month:02d}"
            for t in flux_m["valid_time"].values
        ]
        flux_m = flux_m.rename({"valid_time": "period"})
        ice_m = ice_m.rename({"valid_time": "period"})
    else:
        flux_m = fluxes.groupby("valid_time.month").mean("valid_time")
        ice_m = siconc.groupby("valid_time.month").mean("valid_time")
        labels = [calendar.month_abbr[int(m)] for m in flux_m["month"].values]
        flux_m = flux_m.rename({"month": "period"})
        ice_m = ice_m.rename({"month": "period"})
    return flux_m, ice_m, labels


def figure_grid(
    n_periods: int, layout: str, aspect: float, is_circumpolar: bool
) -> tuple[int, int, tuple[float, float]]:
    """Return ``(nrows, ncols, figsize)`` for the requested layout."""
    if layout == "months-as-cols":
        nrows, ncols = 3, n_periods
    else:
        nrows, ncols = n_periods, 3

    panel_h = 2.6 if not is_circumpolar else 2.4
    panel_w = float(np.clip(panel_h * (1.0 if is_circumpolar else aspect), 1.15, 4.0))

    fig_w = ncols * panel_w + 2.4
    fig_h = nrows * panel_h + 2.2

    # Scale the whole figure down rather than letting a 12-month grid run off
    # the page in either direction.
    scale = min(1.0, MAX_FIG_WIDTH_IN / fig_w, MAX_FIG_HEIGHT_IN / fig_h)
    return nrows, ncols, (fig_w * scale, fig_h * scale)


def make_monthly_maps(
    flux_m,
    ice_m,
    labels: list[str],
    region: str,
    mask_label: str,
    source_label: str,
    coverage: dict[str, float] | None = None,
    layout: str = "months-as-cols",
    projection: str = "polar",
    output_path: Path | None = None,
    dpi: int = 150,
):
    """Draw the month-by-component grid and return the matplotlib Figure."""
    import matplotlib.pyplot as plt

    use_cartopy = projection == "polar"
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            print("  cartopy unavailable; using a plain lat/lon grid.", file=sys.stderr)
            use_cartopy = False

    lat = flux_m["latitude"].values
    lon = flux_m["longitude"].values
    is_circumpolar = (lon.max() - lon.min()) >= 350.0
    aspect = domain_aspect(lat, lon)
    n_periods = flux_m.sizes["period"]

    proj_kw = {}
    if use_cartopy:
        central_lon = 0.0 if is_circumpolar else float(np.mean([lon.min(), lon.max()]))
        proj_kw = {"projection": ccrs.NorthPolarStereo(central_longitude=central_lon)}
        data_crs = ccrs.PlateCarree()

    nrows, ncols, figsize = figure_grid(
        n_periods, layout, aspect, is_circumpolar and use_cartopy
    )
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, subplot_kw=proj_kw,
        squeeze=False, constrained_layout=True,
    )

    # One symmetric colour scale across every panel, so a month-to-month or
    # component-to-component comparison is meaningful. A per-panel scale would
    # make a quiet month look as dramatic as a stormy one.
    all_fields = [flux_m[name].values for name, _, _ in FLUX_PANELS]
    vmax = symmetric_limits(all_fields, COLOR_LIMIT_PERCENTILE)

    mesh = None
    for i_comp, (name, title, subtitle) in enumerate(FLUX_PANELS):
        for i_per in range(n_periods):
            if layout == "months-as-cols":
                ax = axes[i_comp][i_per]
            else:
                ax = axes[i_per][i_comp]

            field = flux_m[name].isel(period=i_per).values
            plot_kw = dict(cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="auto")
            if use_cartopy:
                plot_kw["transform"] = data_crs
            mesh = ax.pcolormesh(lon, lat, field, **plot_kw)

            # Sea-ice edge: the reason for looking at this month by month.
            ice = ice_m.isel(period=i_per).values
            if np.isfinite(ice).any():
                ct_kw = dict(levels=ICE_EDGE_LEVELS, colors="black",
                             linewidths=0.9, linestyles=["--", "-"], zorder=5)
                if use_cartopy:
                    ct_kw["transform"] = data_crs
                ax.contour(lon, lat, ice, **ct_kw)

            if use_cartopy:
                if is_circumpolar:
                    ax.set_extent([-180, 180, lat.min(), 90], crs=data_crs)
                    _make_circular(ax)
                else:
                    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                                  crs=data_crs)
                ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
                ax.coastlines(resolution="50m", linewidth=0.45, color="0.35", zorder=3)
                ax.gridlines(linewidth=0.3, color="0.75", alpha=0.6, zorder=4)
            else:
                ax.tick_params(labelsize=7, colors="0.4")

            # Flag a period built from an incomplete set of calendar days.
            cov = (coverage or {}).get(labels[i_per], 1.0)
            period_label = labels[i_per]
            if cov < 0.99:
                period_label = f"{period_label}\n({cov * 100:.0f}% of days)"

            # Label only the outer edge of the grid, so inner panels stay clean.
            if layout == "months-as-cols":
                if i_comp == 0:
                    ax.set_title(period_label, fontsize=10, pad=6,
                                 color="0.45" if cov < 0.99 else "black")
                if i_per == 0:
                    ax.text(-0.12, 0.5, f"{title}\n{subtitle}", transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=9.5)
            else:
                if i_per == 0:
                    ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=6)
                if i_comp == 0:
                    ax.text(-0.12, 0.5, period_label, transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=9.5,
                            color="0.45" if cov < 0.99 else "black")

    cb = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.02,
                      shrink=0.6, aspect=45)
    cb.set_label(
        "Turbulent heat flux [W m$^{-2}$], positive downward\n"
        f"contours: sea ice fraction {ICE_EDGE_LEVELS[0]} (dashed), "
        f"{ICE_EDGE_LEVELS[1]} (solid)",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=9)

    # Kept short deliberately: constrained_layout widens the whole figure to fit
    # the suptitle, so a long one strands the narrow map panels in white space.
    # The mask note lives here rather than as figtext at the bottom, where it
    # collided with the colorbar's second label line.
    fig.suptitle(
        f"ERA5 monthly turbulent fluxes — {region}\n"
        f"{source_label}   |   {mask_label}",
        fontsize=11,
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
    parser.add_argument("--mask", choices=("open-ocean", "all-ocean"), default="open-ocean",
                        help="open-ocean drops land and sea ice (default); all-ocean drops only land.")
    parser.add_argument("--max-siconc", type=float, default=DEFAULT_MAX_SICONC)
    parser.add_argument("--months", type=_month_list, default=(8, 9, 10, 11),
                        metavar="M[,M...]",
                        help="Calendar months to show, comma-separated (default 8,9,10,11 "
                             "= Aug-Nov, the freeze-up season). Outside these months the "
                             "Arctic is essentially fully ice covered and the panels carry "
                             "little information. Pass 'all' for every month present.")
    parser.add_argument("--by-year-month", action="store_true",
                        help="One panel per calendar month in the record instead of a "
                             "climatology averaged across years.")
    parser.add_argument("--layout", choices=("months-as-cols", "months-as-rows"),
                        default="months-as-cols",
                        help="months-as-cols puts time on the horizontal axis with the "
                             "three components as rows (default). months-as-rows keeps "
                             "the original 3-column component layout.")
    parser.add_argument("--projection", choices=("polar", "platecarree"), default="polar")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("ERA5 monthly turbulent flux maps")
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
    print(f"  Time range : {t0} to {t1} UTC  ({n_times} time steps)")

    mask, report = build_ocean_mask(ds, args.mask, args.max_siconc)
    print(f"\n  Masking ({args.mask}):")
    print(report.describe())
    warn_if_sparse(report)

    fluxes = compute_turbulent_fluxes(ds, mask)
    siconc = ds["siconc"]
    # Restrict to the requested calendar months BEFORE averaging, so a panel is
    # never built from months the caller excluded.
    if len(args.months) < 12:
        keep_t = ds["valid_time"].dt.month.isin(list(args.months))
        n_before = ds.sizes["valid_time"]
        ds = ds.sel(valid_time=keep_t)
        fluxes = fluxes.sel(valid_time=keep_t)
        siconc = siconc.sel(valid_time=keep_t)
        if ds.sizes["valid_time"] == 0:
            print(f"  Error: no data in months {list(args.months)}.", file=sys.stderr)
            return 1
        month_names = ", ".join(calendar.month_abbr[m] for m in sorted(args.months))
        print(f"\n  Months     : {month_names} "
              f"({ds.sizes['valid_time']:,} of {n_before:,} steps kept)")

    flux_m, ice_m, labels = monthly_means(fluxes, siconc, args.by_year_month)
    flux_m = flux_m.compute()
    ice_m = ice_m.compute()
    coverage = period_coverage(ds, args.by_year_month)

    partial = [f"{k} ({v * 100:.0f}%)" for k, v in coverage.items() if v < 0.99]
    print(f"\n  Periods    : {len(labels)}  ({', '.join(labels)})")
    if partial:
        print(f"  Incomplete : {', '.join(partial)} — these periods do not cover")
        print("               every calendar day and are labelled as such on the figure.")

    print("\n  Area-mean flux by period [W m-2, positive downward]:")
    hdr = (f"    {'period':<10}{'net':>10}{'sensible':>10}{'latent':>10}"
           f"{'mean ice':>10}{'coverage':>10}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for i, lab in enumerate(labels):
        vals = []
        for name, _, _ in FLUX_PANELS:
            f = flux_m[name].isel(period=i).values
            f = f[np.isfinite(f)]
            vals.append(f.mean() if f.size else np.nan)
        ice = ice_m.isel(period=i).values
        ice = ice[np.isfinite(ice)]
        ice_mean = ice.mean() if ice.size else np.nan
        print(f"    {lab:<10}{vals[0]:>10.2f}{vals[1]:>10.2f}{vals[2]:>10.2f}"
              f"{ice_mean:>10.3f}{coverage.get(lab, 1.0) * 100:>9.0f}%")

    output_path = args.output
    if output_path is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        kind = "bymonth" if args.by_year_month else "climatology"
        output_path = fig_dir / f"{args.region}_monthly_flux_maps_{kind}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    mask_label = (
        f"open ocean only (sea ice fraction < {args.max_siconc:g})"
        if args.mask == "open-ocean"
        else "all ocean, including sea ice"
    )
    # Report the years actually contributing, not the raw record span: after a
    # month filter, "2000-01 to 2026-01" would describe data that is not shown.
    yrs = sorted({int(y) + 1970
                  for y in ds["valid_time"].values.astype("datetime64[Y]").astype(int)})
    span = f"{yrs[0]}" if len(yrs) == 1 else f"{yrs[0]}–{yrs[-1]}"
    source_label = (
        f"individual months, {span}"
        if args.by_year_month
        else f"climatology over {span} ({len(yrs)} year{'s' if len(yrs) != 1 else ''})"
    )

    make_monthly_maps(
        flux_m, ice_m, labels,
        region=args.region,
        mask_label=mask_label,
        source_label=source_label,
        coverage=coverage,
        layout=args.layout,
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
