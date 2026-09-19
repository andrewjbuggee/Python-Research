#!/usr/bin/env python3
"""Run the Ocean Visions figure set for ANY box and season, from S3 or disk.

This is the batch twin of ``Presentations_and_papers/Ocean_Visions/
ocean_visions_figures.ipynb``: the same ``prepare()`` calls with the same
settings, but the region, seasons and storage come from the command line, and
each stage saves its figures plus a pickle of the reduced results (the
``Analysis`` objects minus their lazy dataset) so figures can be redrawn or
re-styled later without touching the data again.

The analysis modules are imported and called UNMODIFIED. The only thing
``--storage aws`` changes is where ``load_seb_data`` gets its bytes.

Examples
--------
Same domain and seasons as the talk, straight from the bucket::

    python run_analysis.py --storage aws --region barrow --years 2014-2024 --stages lwph maps

A bigger box (registered on the fly), two seasons, everything::

    python run_analysis.py --storage aws --box 82 -175 65 -110 --region-name beaufort_wide \\
        --years 2023-2024 --stages all --out results/beaufort_wide

On the EC2 instance the invocation is identical; ``remote/run_job.sh`` wraps
it with the environment and copies ``--out`` back.

Stages
------
  lwph    figures 1-3: seasonal liquid-containing hours vs the ARM record,
          monthly boxes, retention-by-knob sweep (site-dependent: 1-3)
  dlr     figure 7: DLR by sky state at the ARM cell (7a, site-dependent) and
          over the domain (7b, 7c)                      [needs lwph]
  extent  figure 4: cloud duration + wind; ``--cloud-level-wind`` adds the
          pressure-level wind for ``--pl-years``
  maps    figures 5-6: liquid fraction / LWP maps, one season month by month
  flux    figure 8: normalised surface-flux response
  thumb   the domain thumbnail                         [needs lwph]

Memory: the streaming passes hold ``block_hours x n_cells x ~15`` float64
temporaries. The default 720 h is sized for Barrow (2,501 cells); for a box
with N cells use ``--block-hours`` about ``720 * 2500 / N`` (the script does
this automatically unless you pass it).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SEB_DIR = Path(__file__).resolve().parent.parent
if str(SEB_DIR) not in sys.path:
    sys.path.insert(0, str(SEB_DIR))

from aws_pipeline import era5_s3, s3_storage  # noqa: E402

STAGES = ("lwph", "dlr", "extent", "maps", "flux", "thumb")

# Barrow's cell count, which the module defaults (block_hours=720) were sized for.
REFERENCE_CELLS = 41 * 61


def parse_years(text: str) -> tuple[int, ...]:
    """``2014-2024`` or ``2014,2016,2020`` -> season START years."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def parse_season(text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """``10-01:03-31`` -> ((10, 1), (3, 31))."""
    a, b = text.split(":")
    m0, d0 = (int(x) for x in a.split("-"))
    m1, d1 = (int(x) for x in b.split("-"))
    return (m0, d0), (m1, d1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source and domain")
    src.add_argument("--storage", default="aws", choices=["aws", "local", "external"])
    src.add_argument("--region", default="barrow",
                     help="a name in era5_seb_variables.REGIONS, or box:N,W,S,E")
    src.add_argument("--box", nargs=4, type=float, metavar=("N", "W", "S", "E"),
                     help="register this box under --region-name and use it")
    src.add_argument("--region-name", default=None, help="name for --box (default: box_N_W_S_E)")
    tm = p.add_argument_group("time")
    tm.add_argument("--years", default="2014-2024", help="season START years, e.g. 2014-2024 or 2020,2022")
    tm.add_argument("--season", default="10-01:03-31", help="MM-DD:MM-DD, wraps the new year if needed")
    tm.add_argument("--flux-years", default=None, help="season start years for the flux stage (default: --years)")
    tm.add_argument("--pl-years", default=None,
                    help="season start years for the cloud-level wind (default: last two of --years)")
    st = p.add_argument_group("what to run")
    st.add_argument("--stages", nargs="+", default=["lwph", "maps"], help=f"any of {STAGES} or 'all'")
    st.add_argument("--cloud-level-wind", action="store_true",
                    help="extent stage: add the pressure-level wind (heavy: ~40 MB per hour per variable)")
    st.add_argument("--out", type=Path, default=None, help="output directory (default: results/<region>_<years>)")
    st.add_argument("--no-pickle", action="store_true", help="save figures only")
    st.add_argument("--dpi", type=int, default=200)
    kn = p.add_argument_group("analysis knobs (defaults = the OV notebook)")
    kn.add_argument("--liquid-fraction-min", type=float, default=0.90)
    kn.add_argument("--ice-fraction-min", type=float, default=0.85,
                    help="0.85 for the domain figures in the OV notebook; 0.90 for its ARM-cell figures")
    kn.add_argument("--ice-fraction-min-site", type=float, default=0.90)
    kn.add_argument("--min-lwp", type=float, default=0.01, help="g m-2")
    kn.add_argument("--min-iwp", type=float, default=0.01, help="g m-2")
    kn.add_argument("--min-cloud-fraction", type=float, default=0.95)
    kn.add_argument("--precip-rate-max", type=float, default=0.05, help="mm hr-1; the OV filter")
    kn.add_argument("--all-sky", action="store_true", help="skip the precipitation filter everywhere")
    kn.add_argument("--block-hours", type=int, default=None, help="streaming block; auto-scaled by cell count")
    kn.add_argument("--flux-min-lwp", type=float, default=2.0, help="g m-2, the flux stage's liquid floor")
    return p


def main(argv=None) -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = build_parser().parse_args(argv)
    stages = list(STAGES) if "all" in args.stages else [s for s in STAGES if s in args.stages]
    bad = set(args.stages) - set(STAGES) - {"all"}
    if bad:
        raise SystemExit(f"unknown stage(s) {sorted(bad)}; choose from {STAGES}")
    if "dlr" in stages and "lwph" not in stages:
        stages.insert(0, "lwph")
    if "thumb" in stages and "lwph" not in stages:
        stages.insert(0, "lwph")

    if args.box:
        n, w, s, e = args.box
        name = args.region_name or f"box_{n:g}_{w:g}_{s:g}_{e:g}".replace("-", "m").replace(".", "p")
        s3_storage.register_region(name, n, w, s, e, "command-line box")
        args.region = name
    box = s3_storage.region_box(args.region)
    years = parse_years(args.years)
    flux_years = parse_years(args.flux_years) if args.flux_years else years
    pl_years = parse_years(args.pl_years) if args.pl_years else years[-2:]
    season_start, season_end = parse_season(args.season)

    out = args.out or (Path(__file__).resolve().parent / "results"
                       / f"{args.region}_{years[0]}-{years[-1]}_{args.storage}")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Block size scaled to the domain so memory stays at the Barrow footprint.
    win = era5_s3.make_window(*box)
    n_cells = win.n_lat * win.n_lon
    block_hours = args.block_hours or max(24, int(720 * REFERENCE_CELLS / n_cells) // 24 * 24)

    common = dict(
        region=args.region, years=years, season_start=season_start, season_end=season_end,
        phase_mode="fraction", liquid_fraction_min=args.liquid_fraction_min,
        min_lwp=args.min_lwp, min_iwp=args.min_iwp, min_cloud_fraction=args.min_cloud_fraction,
        lsm_tol=0.01, storage=args.storage, dpi=args.dpi,
    )
    precip = ({} if args.all_sky else
              dict(no_precip=True, precip_var="rate", precip_rate_max=args.precip_rate_max))

    print("=" * 78)
    print(f"Ocean Visions figure set  |  storage={args.storage}  region={args.region} "
          f"box={list(box)}  {n_cells:,} cells")
    print(f"  seasons {years[0]}..{years[-1]} ({len(years)}), window "
          f"{season_start[0]:02d}-{season_start[1]:02d} .. {season_end[0]:02d}-{season_end[1]:02d}; "
          f"stages {stages}; block_hours {block_hours}; out {out}")
    if args.storage == "aws":
        wins = era5_s3.season_windows(years, season_start, season_end)
        print(era5_s3.describe_cost(wins))
    print("=" * 78)

    # Import here so --help is instant and a local-only machine without
    # cartopy etc. can still print usage.
    import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
    import plot_dlr_by_phase as dlr  # noqa: E402
    import cloud_spatial_extent as cse  # noqa: E402
    import cloud_level_wind as clw  # noqa: E402
    import map_liquid_hours as mlh  # noqa: E402
    import turbulent_flux_response as tfr  # noqa: E402

    manifest = {"created": datetime.now().isoformat(timespec="seconds"),
                "argv": list(argv) if argv is not None else sys.argv[1:],
                "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "region": args.region, "box_NWSE": list(box), "years": list(years),
                "season": [list(season_start), list(season_end)], "storage": args.storage,
                "block_hours": block_hours, "stages": {}}

    def save(name: str, obj, drop=("ds",)) -> None:
        """Pickle the reduced result without its lazy dataset.

        The grid is kept as plain arrays (``grid_lat``/``grid_lon``) so a
        figure that only needs coordinates can be redrawn from the pickle.
        Passes that re-stream ``A.ds`` (``dlr.prepare_domain``,
        ``extract_site_table``, the retention-by-knob duration panel, the
        thumbnail) need a live ``prepare()`` -- rerun the stage for those.
        """
        if args.no_pickle:
            return
        payload = {k: v for k, v in vars(obj).items() if k not in drop} if hasattr(obj, "__dict__") else obj
        ds = getattr(obj, "ds", None)
        if isinstance(payload, dict) and ds is not None and "ds" in drop:
            payload["grid_lat"] = ds["latitude"].values.copy()
            payload["grid_lon"] = ds["longitude"].values.copy()
        if isinstance(payload, dict) and "args" in payload and isinstance(payload["args"], argparse.Namespace):
            payload["args"] = SimpleNamespace(**vars(payload["args"]))
        with open(out / f"{name}.pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved {out / (name + '.pkl')}")

    def figure(fn, *a, **kw):
        """Draw one figure; a missing observation/input file skips it with a
        message instead of discarding the stage's (expensive) prepare() --
        e.g. the gitignored genie_arm_monthly_hours.xlsx on an instance that
        was not fed with remote/sync_code.sh."""
        try:
            return fn(*a, **kw)
        except FileNotFoundError as exc:
            print(f"  !! skipped {fn.__name__}: {exc}", file=sys.stderr, flush=True)
            manifest.setdefault("skipped_figures", []).append({"figure": fn.__name__, "error": str(exc)})
            return None

    t_all = time.time()
    A_precip = None
    for stage in stages:
        t0 = time.time()
        print(f"\n##### stage {stage} #####")
        if stage == "lwph":
            A_precip = lwph.prepare(**common, **precip, block_hours=block_hours,
                                    ice_fraction_min=args.ice_fraction_min_site)
            lwph.print_report(A_precip)
            figure(lwph.fig_era5_vs_obs_simple_forOV, A_precip, out_dir=fig_dir, halo_alpha=0.7)
            figure(lwph.print_era5_vs_obs_mean_pct_diff, A_precip)
            save("A_precip", A_precip)
            A_allsky = lwph.prepare(**common, block_hours=block_hours,
                                    ice_fraction_min=args.ice_fraction_min_site)
            figure(lwph.fig_monthly_box_era5_vs_obs_build_forOV, A_allsky, out_dir=fig_dir,
                   category="liquid_containing")
            save("A_allsky", A_allsky)
            A_sweep = lwph.prepare(**common, block_hours=block_hours,
                                   ice_fraction_min=args.ice_fraction_min,
                                   sweep_lwp_min=0.1, sweep_lwp_max=20.0, sweep_points=21,
                                   sweep_spacing="linear", show_ice_only=False)
            figure(lwph.fig_liquid_retention_by_knob, A_sweep, out_dir=fig_dir)
            save("A_sweep", A_sweep)
        elif stage == "dlr":
            S = dlr.extract_site_table(A_precip)
            dlr.verify_against_fit(S, A_precip)
            D = dlr.prepare_domain(A_precip)
            dlr.fig_monthly_dlr_box(A_precip, S, out_dir=fig_dir)
            dlr.print_monthly_dlr_table(A_precip, S)
            dlr.fig_monthly_dlr_box_domain_forOV(A_precip, D, out_dir=fig_dir)
            dlr.fig_monthly_dlr_box_by_class_forOV(A_precip, D, out_dir=fig_dir)
            dlr.print_state_fraction_table(A_precip, D)
            dlr.print_monthly_dlr_table_domain(A_precip, D)
            dlr.fig_dlr_pdf(A_precip, D, out_dir=fig_dir)
            dlr.fig_dlr_pdf(A_precip, D, out_dir=fig_dir, show_states=True)
            save("S_precip", S, drop=())
            save("D_precip", D, drop=())
        elif stage == "extent":
            E = cse.prepare_extent(**common, **precip, block_hours=block_hours,
                                   ice_fraction_min=args.ice_fraction_min_site)
            ev = cse.taylor_events(E)
            cse.print_taylor_report(E, ev)
            cse.fig_taylor_hist(E, ev, out_dir=fig_dir)
            E_pl = CL = None
            if args.cloud_level_wind:
                E_pl = cse.prepare_extent(**{**common, "years": pl_years}, **precip,
                                          block_hours=block_hours,
                                          ice_fraction_min=args.ice_fraction_min)
                CL = clw.cloud_level_wind(E_pl)
                clw.fig_cloudDuration_andWind_forOV(E, ev, E_pl=E_pl, CL=CL, out_dir=fig_dir)
            clw.fig_cloudDuration_andWind_forOV(E, ev, wind_source="10m", out_dir=fig_dir)
            cse.print_site_liquid_hours_per_day(E, E_pl=E_pl, CL=CL, wind_source="10m")
            save("E", E)
            save("ev", ev, drop=())
            if CL is not None:
                save("CL", CL, drop=())
        elif stage == "maps":
            M_all = mlh.prepare_maps(**common, block_hours=block_hours,
                                     ice_fraction_min=args.ice_fraction_min)
            runs = [("all sky", M_all)]
            if precip:
                M_np = mlh.prepare_maps(**common, **precip, block_hours=block_hours,
                                        ice_fraction_min=args.ice_fraction_min)
                runs.append(("precipitation filtered", M_np))
                removed = 1.0 - M_np.counts["cloudy"].sum() / M_all.counts["cloudy"].sum()
                print(f"\n  the filter removes {100 * removed:.1f}% of overcast cell-hours across the domain")
            for tag, Mi in runs:
                mlh.fig_fraction_and_lwp(Mi, out_dir=fig_dir)
                mlh.fig_season_monthly_maps(Mi, season=years[-1], out_dir=fig_dir)
                save("M_" + tag.split()[0], Mi)
        elif stage == "flux":
            A_flux = tfr.prepare(**{**common, "years": flux_years, "min_lwp": args.flux_min_lwp},
                                 block_hours=block_hours, ice_fraction_min=args.ice_fraction_min,
                                 fit_mode="regression", open_ocean_max_siconc=0.05,
                                 sea_ice_min_siconc=0.95)
            tfr.fig_monthly_response_simple_normalized(A_flux, forcer="fnet", layout="row",
                                                       out_dir=fig_dir, dpi=args.dpi, signed=False)
            save("A_flux", A_flux)
        elif stage == "thumb":
            mlh.fig_domain_thumbnail(A_precip, out_dir=fig_dir, dpi=args.dpi)
        dt = time.time() - t0
        manifest["stages"][stage] = {"seconds": round(dt, 1)}
        print(f"##### stage {stage} done in {dt / 60:.1f} min #####")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nall stages done in {(time.time() - t_all) / 60:.1f} min; figures in {fig_dir}")
    era5_s3.shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
