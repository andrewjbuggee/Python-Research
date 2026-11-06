#!/usr/bin/env python3
"""One place that says WHICH domain, WHICH years and WHERE the ERA5 data is.

Every make_era5_*.py script and ocean_visions_figures.ipynb takes these from
here, so the same code runs unchanged on the laptop and on NCAR Casper. This is
the Ocean Visions counterpart of PRISM's ``prism_paths.py``.

WHERE THE ERA5 DATA COMES FROM
==============================
    Casper    storage="glade": the NCAR copy of ERA5 on disk at
              /gdex/data/d633000 (GDEX dataset d633000, formerly RDA ds633.0).
              It is read in place. Nothing is downloaded, and only decoded
              chunks are cached, in scratch.
    Laptop    storage="local": the files already in
              ERA5/surface_energy_budget/data/<region>/. A domain that is not on
              disk stops with an error; it never triggers a download.
              storage="s3" streams the public AWS mirror of the same archive,
              but its chunk cache grows on this disk (ERA5_CACHE moves it).

CHANGING THE DOMAIN OR THE YEARS
================================
Highest priority first:

    1. Command line   python make_era5_hour_count_csvs.py --box 82 -175 65 -110 --years 2014-2024
    2. Environment    OV_REGION, OV_YEARS, OV_STORAGE
    3. The settings block just below this docstring

A domain is any of:

    a name in era5_seb_variables.REGIONS   "barrow", "arctic_circle", ...
    an inline box "box:N,W,S,E"            "box:82,-175,65,-110"
    --box N W S E on the command line      (--region-name to name it)

Longitudes run from -180 to 180. West > east means the box crosses the dateline:
"box:75,170,65,-170" runs from 170 E across 180 to 170 W. To keep a domain under
a name, add one line to era5_seb_variables.REGIONS; the downloader,
run_analysis.py and submit.sh then all see it as well.

Years are season START years: 2014 means 1 Oct 2014 - 31 Mar 2015. The grammar
is that of --years everywhere: "2023", "2014-2024" or "2014,2016-2018".

WHERE THE OUTPUT GOES
=====================
Each run writes to its own folder, so a new domain can never overwrite the files
already delivered to collaborators:

    Barrow, 2014-2024, local files, on the laptop     beside this file, as always
    anything else on the laptop                       domain_runs/<domain>_<span>/
    anything on Casper                                /glade/work/$USER/era5_results/<domain>_<span>/

OV_RESULTS_ROOT moves the last two (RESULTS_ROOT, set by setup_casper.sh, is
honoured too).

Run ``python ov_config.py`` to print what the current settings resolve to.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Settings: change the domain and the years here
# ----------------------------------------------------------------------------
# Used when neither the command line nor the environment names them.
DOMAIN = "barrow"       # a REGIONS name, or "box:N,W,S,E"
YEARS = "2014-2024"     # season START years: 2014/15 .. 2024/25
# How class averages weight grid cells, with the same choices as --cell-weighting
# in plot_lwp_histogram_by_surface_class.py:
#   "area"    : cos(latitude), so an average is per unit AREA of the class
#   "uniform" : every grid cell equal, so an average is per GRID CELL
# Plain cell-hour COUNTS (make_era5_hour_count_csvs.py) are never weighted.
CELL_WEIGHTING = os.environ.get("OV_CELL_WEIGHTING") or "area"

# ----------------------------------------------------------------------------
# The record the existing Ocean Visions files describe. Only this combination
# writes beside this script; see RunConfig.is_baseline.
# ----------------------------------------------------------------------------
BASELINE_DOMAIN = "barrow"
BASELINE_YEARS = tuple(range(2014, 2025))

GRID_DEG = 0.25                 # deg, ERA5 single-level and pressure-level grid
REFERENCE_CELLS = 41 * 61       # Barrow, which block_hours=720 was sized for
MAX_BLOCK_HOURS = 720           # h, the analysis modules' default block


# ----------------------------------------------------------------------------
# Which machine are we on? Same rule as PRISM's prism_paths.py
# ----------------------------------------------------------------------------
_VALID_SYSTEMS = ("casper", "laptop")


def _detect_system() -> str:
    """Return "casper" or "laptop": OV_SYSTEM if set, else whether /glade exists."""
    override = os.environ.get("OV_SYSTEM", "").strip().lower()
    if override:
        if override not in _VALID_SYSTEMS:
            raise ValueError(f"OV_SYSTEM={override!r} is not one of {_VALID_SYSTEMS}")
        return override
    # /glade is mounted on every NCAR HPC node (login, compute, OnDemand Jupyter)
    return "casper" if Path("/glade").is_dir() else "laptop"


SYSTEM = _detect_system()
ON_CASPER = SYSTEM == "casper"
USER = getpass.getuser()

# ----------------------------------------------------------------------------
# Paths. REPO_ROOT comes from this file's location rather than a search for
# .git, because a tree copied by sync_to_casper.sh has no .git.
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OV_REPO_ROOT") or HERE.parents[1])
SEB_DIR = REPO_ROOT / "ERA5" / "surface_energy_budget"
if str(SEB_DIR) not in sys.path:
    sys.path.insert(0, str(SEB_DIR))

DEFAULT_STORAGE = "glade" if ON_CASPER else "local"
_default_results = (Path(f"/glade/work/{USER}/era5_results") if ON_CASPER
                    else HERE / "domain_runs")
RESULTS_ROOT = Path(os.environ.get("OV_RESULTS_ROOT")
                    or os.environ.get("RESULTS_ROOT")
                    or _default_results)

# ----------------------------------------------------------------------------
# Casper job environment. setup_casper.sh writes the same values to
# ~/.era5_casper_env for PBS jobs; setting them here as well means a script
# started from an OnDemand Jupyter session behaves the same way. setdefault, so
# anything already exported wins. This must run before numpy is imported, so
# the scripts import this module first.
# ----------------------------------------------------------------------------
if ON_CASPER:
    _scratch = Path(os.environ.get("SCRATCH") or f"/glade/derecho/scratch/{USER}")
    os.environ.setdefault("ERA5_SOURCE", "glade")           # never fall back to the network
    # The default ~/.cache would fill the 50 GB home quota.
    os.environ.setdefault("ERA5_CACHE", str(_scratch / "era5_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(_scratch / "mplconfig"))
    os.environ.setdefault("CARTOPY_DATA_DIR", f"/glade/work/{USER}/cartopy_data")
    # PBS exports NCPUS, the cores this job is charged for. Without a cap, numpy
    # starts one thread per physical core of a node shared with other jobs.
    _ncpus = os.environ.get("NCPUS")
    if _ncpus:
        for _var in ("ERA5_WORKERS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS"):
            os.environ.setdefault(_var, _ncpus)


# ----------------------------------------------------------------------------
# The spatial domain
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Domain:
    """A named latitude-longitude box, in the CDS [N, W, S, E] order."""

    name: str
    north_deg: float
    west_deg: float
    south_deg: float
    east_deg: float

    @property
    def crosses_dateline(self) -> bool:
        return self.west_deg > self.east_deg

    @property
    def n_lat(self) -> int:
        """Grid latitudes inside the box, assuming the box edges are on the grid."""
        return math.floor((self.north_deg - self.south_deg) / GRID_DEG + 1e-9) + 1

    @property
    def n_lon(self) -> int:
        """Grid longitudes inside the box; 1440 for a full circle."""
        span_deg = self.east_deg - self.west_deg
        if self.crosses_dateline:
            span_deg += 360.0
        return min(math.floor(span_deg / GRID_DEG + 1e-9) + 1, 1440)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    def describe(self) -> str:
        """'70-80 N, 165-150 W', or the mixed form when a box spans hemispheres."""

        def span(a: float, b: float, pos: str, neg: str) -> str:
            if a * b >= 0 and (a != 0 or b != 0):
                return f"{abs(a):g}-{abs(b):g} {pos if max(a, b) > 0 else neg}"
            return (f"{abs(a):g} {pos if a >= 0 else neg}"
                    f" to {abs(b):g} {pos if b >= 0 else neg}")

        return (f"{span(self.south_deg, self.north_deg, 'N', 'S')}, "
                f"{span(self.west_deg, self.east_deg, 'E', 'W')}")


def _slug(north_deg: float, west_deg: float, south_deg: float, east_deg: float) -> str:
    """box_82N_175W_65N_110W: safe in a filename and in a PBS -v list."""

    def one(value: float, pos: str, neg: str) -> str:
        return f"{abs(value):g}".replace(".", "p") + (pos if value >= 0 else neg)

    return (f"box_{one(north_deg, 'N', 'S')}_{one(west_deg, 'E', 'W')}"
            f"_{one(south_deg, 'N', 'S')}_{one(east_deg, 'E', 'W')}")


def _validate_box(north_deg: float, west_deg: float, south_deg: float, east_deg: float) -> None:
    """Fail here with a clear message rather than deep inside the reader."""
    if not -90.0 <= south_deg < north_deg <= 90.0:
        raise SystemExit(f"bad domain: need -90 <= S < N <= 90, got N={north_deg} S={south_deg}")
    if not (-180.0 <= west_deg <= 180.0 and -180.0 <= east_deg <= 180.0):
        raise SystemExit(f"bad domain: longitudes must be in -180..180, got W={west_deg} E={east_deg}")
    if west_deg == east_deg:
        raise SystemExit(f"bad domain: W and E are both {west_deg}")


def resolve_domain(spec: str | None = None, box=None, name: str | None = None) -> Domain:
    """Turn a domain request into a Domain the analysis modules can look up.

    An inline box is registered in ``era5_seb_variables.REGIONS`` under a plain
    name, so everything downstream (file names, ``region_box``, the
    ``<region>_pressure`` archives) handles it the same way as a named region.
    """
    import era5_seb_variables as v  # needs SEB_DIR on sys.path, done above

    if box is None and spec is not None and str(spec).startswith("box:"):
        try:
            box = [float(x) for x in str(spec)[4:].split(",")]
        except ValueError:
            raise SystemExit(f"bad domain {spec!r}: use box:N,W,S,E") from None
        if len(box) != 4:
            raise SystemExit(f"bad domain {spec!r}: use box:N,W,S,E")
    if box is not None:
        north_deg, west_deg, south_deg, east_deg = (float(x) for x in box)
        _validate_box(north_deg, west_deg, south_deg, east_deg)
        region_name = name or _slug(north_deg, west_deg, south_deg, east_deg)
        v.REGIONS[region_name] = v.Region(north_deg, west_deg, south_deg, east_deg,
                                          f"custom box {region_name}")
        return Domain(region_name, north_deg, west_deg, south_deg, east_deg)
    try:
        r = v.get_region(str(spec))
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    return Domain(str(spec), r.north_deg, r.west_deg, r.south_deg, r.east_deg)


# ----------------------------------------------------------------------------
# Output naming
# ----------------------------------------------------------------------------
def span_tag(seasons) -> str:
    """The years part of a file name: 2014_2025 for 2014/15 .. 2024/25.

    A non-contiguous selection also states its season count, so that
    "2016,2020" cannot be mistaken for every season from 2016 to 2021.
    """
    seasons = sorted(int(y) for y in seasons)
    tag = f"{seasons[0]}_{seasons[-1] + 1}"
    if seasons[-1] - seasons[0] + 1 != len(seasons):
        tag += f"_{len(seasons)}seasons"
    return tag


# ----------------------------------------------------------------------------
# One complete run: what, when, where from
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    domain: Domain
    years: tuple[int, ...]
    storage: str

    @property
    def is_baseline(self) -> bool:
        """The local Barrow record behind the delivered collaborator files.

        Remote storage does not count. tp there is rebuilt from mtpr and differs
        from the CDS field at GRIB packing precision, so a re-run could change
        those files slightly while appearing to reproduce them.
        """
        return (not ON_CASPER and self.storage == "local"
                and self.domain.name == BASELINE_DOMAIN and self.years == BASELINE_YEARS)

    @property
    def tag(self) -> str:
        return f"{self.domain.name}_{span_tag(self.years)}"

    @property
    def block_hours(self) -> int:
        """Hours streamed per block, scaled so memory stays at the Barrow footprint.

        One block holds block_hours x n_cells x ~15 float64 temporaries. This is
        the same rule as aws_pipeline/run_analysis.py, capped at the modules'
        default. OV_BLOCK_HOURS overrides.
        """
        raw = os.environ.get("OV_BLOCK_HOURS")
        if raw:
            return int(raw)
        scaled = int(MAX_BLOCK_HOURS * REFERENCE_CELLS / self.domain.n_cells) // 24 * 24
        return min(MAX_BLOCK_HOURS, max(24, scaled))

    def settings(self) -> dict:
        """The run-specific options to merge into a script's science settings."""
        return dict(region=self.domain.name, years=self.years, storage=self.storage,
                    block_hours=self.block_hours)

    def out_dir(self, product: str) -> Path:
        """Folder for one product, e.g. out_dir("era5_hour_count_csv")."""
        if self.is_baseline:
            return HERE / product
        return RESULTS_ROOT / self.tag / product

    def source_text(self) -> str:
        """Where the data came from, in words, for readME files."""
        if self.storage in ("local", "external"):
            return "the Copernicus Climate Data Store"
        return "the NSF NCAR copy of ERA5 (GDEX dataset d633000, formerly RDA ds633.0)"

    def report(self) -> str:
        d = self.domain
        return (f"  domain  {d.name}: {d.describe()}, {d.n_lat} x {d.n_lon} = {d.n_cells:,} cells\n"
                f"  seasons {self.years[0]}/{(self.years[0] + 1) % 100:02d} .. "
                f"{self.years[-1]}/{(self.years[-1] + 1) % 100:02d} ({len(self.years)})\n"
                f"  data    storage={self.storage} on {SYSTEM}; block_hours {self.block_hours}")


def warn_if_login_node() -> None:
    """Warn when analysis is started outside a PBS job on Casper.

    NCAR terminates login-node processes that use excessive CPU, more than a few
    GB of memory or excessive I/O, and emails the user. The policy advises batch
    or interactive jobs instead:
    https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/casper-use-policies/
    PBS sets PBS_JOBID inside a job, which includes OnDemand Jupyter sessions.
    """
    if ON_CASPER and not os.environ.get("PBS_JOBID"):
        print("  !! No PBS job found: this looks like a Casper login node, where heavy\n"
              "  !! processes are terminated automatically. GDEX is readable from a login\n"
              "  !! node, but run these passes with submit.sh or in an OnDemand session.",
              file=sys.stderr, flush=True)


def _check_local_data(domain: Domain, storage: str) -> None:
    """A domain that is not on disk is pointed at GDEX, never downloaded."""
    if storage == "local" and not (SEB_DIR / "data" / domain.name).is_dir():
        raise SystemExit(
            f"Domain {domain.name!r} ({domain.describe()}) is not on this disk under "
            f"{SEB_DIR / 'data'}.\nNothing is downloaded automatically. Either run on "
            "Casper, where storage=glade reads GDEX d633000 in place, or pass\n"
            "--storage s3 to stream the AWS mirror (its chunk cache grows on this disk).")


def resolve_run(region: str | None = None, box=None, region_name: str | None = None,
                years=None, storage: str | None = None) -> RunConfig:
    """Apply the precedence: arguments, then OV_* environment, then settings block."""
    from plot_surface_class_timeseries import parse_years  # the --years grammar

    domain = resolve_domain(region or os.environ.get("OV_REGION") or DOMAIN,
                            box=box, name=region_name)
    raw_years = years or os.environ.get("OV_YEARS") or YEARS
    year_tuple = tuple(raw_years) if isinstance(raw_years, (list, tuple, range)) \
        else tuple(parse_years(raw_years))
    chosen_storage = storage or os.environ.get("OV_STORAGE") or DEFAULT_STORAGE
    _check_local_data(domain, chosen_storage)
    warn_if_login_node()
    return RunConfig(domain, year_tuple, chosen_storage)


def add_weighting_arg(parser: argparse.ArgumentParser) -> None:
    """The cell-weighting toggle, on the scripts that average over a class."""
    parser.add_argument(
        "--cell-weighting", choices=("area", "uniform"), default=CELL_WEIGHTING,
        help="area = cos(latitude), an average per unit area of the class; uniform = "
             f"an average per grid cell (default {CELL_WEIGHTING}). Area-weighted output "
             "goes to its own folder and file names, so it never replaces uniform files.")


def weight_suffix(weighting: str) -> str:
    """Folder suffix: "_areaWeighted", or nothing for uniform."""
    return "_areaWeighted" if weighting == "area" else ""


# The file stems whose values depend on the weighting. Counts of cells or hours
# (nCells, hoursClassPresent, totalHours) do not, so they keep their names.
_WEIGHTED_STEMS = re.compile(
    r"\bERA5_(avgHoursPerCell|fractionOfTime|percentHoursPerCellLost|percentCellHoursLost"
    r"|cumulativeHoursPerCell|LWP|CloudTemp|DLR)_(?=per|\d|\{)")


def weighted_names(text: str, weighting: str) -> str:
    """Tag weighted file stems in a name or readME: ERA5_LWP_areaWeighted_perHour_..."""
    if weighting != "area":
        return text
    return _WEIGHTED_STEMS.sub(r"ERA5_\1_areaWeighted_", text)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """The same domain, year and storage options on every script."""
    g = parser.add_argument_group("domain, years, data source (see ov_config.py)")
    g.add_argument("--region", default=None,
                   help=f"a name in era5_seb_variables.REGIONS or box:N,W,S,E (default {DOMAIN})")
    g.add_argument("--box", nargs=4, type=float, default=None, metavar=("N", "W", "S", "E"),
                   help="custom box in degrees; W > E crosses the dateline")
    g.add_argument("--region-name", default=None, help="name for --box in folders and titles")
    g.add_argument("--years", default=None,
                   help=f"season START years, e.g. 2014-2024 or 2014,2016 (default {YEARS})")
    g.add_argument("--storage", default=None,
                   choices=["local", "external", "glade", "ncar", "s3", "aws"],
                   help=f"where ERA5 is read from (default here: {DEFAULT_STORAGE})")


def run_from_args(opts: argparse.Namespace) -> RunConfig:
    """The RunConfig for options parsed with add_run_args."""
    return resolve_run(opts.region, opts.box, opts.region_name, opts.years, opts.storage)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print what the current settings resolve to.")
    add_run_args(p)
    run = run_from_args(p.parse_args())
    print(f"machine   {SYSTEM}   repo {REPO_ROOT}")
    print(run.report())
    print(f"  output  {run.out_dir('<product>')}")
    print(f"  baseline record: {run.is_baseline}")
