# ERA5 from the NCAR S3 bucket — `--storage aws`

The same analysis code, the same `prepare()` calls, the same notebook — with the
data read on demand from the public NSF NCAR ERA5 mirror on AWS instead of
`data/<region>/`. Nothing in the analysis modules is duplicated; the only change
outside this directory is one `if storage == "aws"` branch in each of the three
functions every `prepare()` already calls.

```python
import plot_lwp_histogram_by_surface_class as lwph
from aws_pipeline import s3_storage

s3_storage.register_region("beaufort_wide", 82, -175, 65, -110)   # N, W, S, E; optional

A = lwph.prepare(region="beaufort_wide", storage="aws",
                 years=range(2014, 2025), season_start=(10, 1), season_end=(3, 31),
                 phase_mode="fraction", liquid_fraction_min=0.90, ice_fraction_min=0.85,
                 min_lwp=0.01, min_iwp=0.01, min_cloud_fraction=0.95,
                 no_precip=True, precip_var="rate", precip_rate_max=0.05)
```

That is the Ocean Visions notebook's call with two words changed. `region` may be
any name in `era5_seb_variables.REGIONS`, a box registered as above, or an inline
`"box:82,-175,65,-110"`. The time window comes from the `years` + season
arguments (or `--start/--end` in the map scripts), and only those hours are
fetched — on S3 "the whole archive" would be 1940–2026, so the window is
required rather than defaulted.

| | |
|---|---|
| Bucket | `s3://nsf-ncar-era5`, us-west-2, anonymous read — <https://registry.opendata.aws/nsf-ncar-era5/> |
| Coverage (2026-09-18) | 1940-01 → 2026-06 (surface analysis, pressure levels), → 2026-05 (fluxes); 3–4 month lag |
| Verified against the local CDS files | 26 of 33 single-level variables **bit-for-bit**; `skt t2m d2m sp tcwv istl1 istl2` within one float32 ULP; `tp` at GRIB packing precision (see below); `u`,`v` on 16 pressure levels bit-for-bit; land-sea mask bit-for-bit |

## Files

| File | Purpose |
|---|---|
| `era5_s3.py` | The reader: a lazy xarray backend over the bucket, windowed to a box, with concurrent chunk fetches and an on-disk chunk cache. Usable on its own: `era5_s3.open_dataset(N, W, S, E, windows, variables)`. |
| `s3_storage.py` | The adapter that makes `storage="aws"` work through `resolve_region_dir → load_seb_data → load_land_sea_mask`, plus the pressure-level datasets `cloud_level_wind` and `turbulent_flux_response` expect. |
| `run_analysis.py` | Batch driver: the Ocean Visions figure set (stages `lwph dlr extent maps flux thumb`) for any box/season/storage, saving figures and pickled reduced results. Identical invocation on the laptop and on EC2. |
| `verify_against_local.py` | `variables`: cell-by-cell comparison of every variable against `data/<region>/`; `analysis`: `lwph.prepare` local vs aws for a season. |
| `quickstart_aws.ipynb` | A short notebook: open the bucket, look at a window, run one figure, estimate a job. |
| `remote/` | EC2 (us-west-2): `launch_instance.sh`, `bootstrap.sh`, `sync_code.sh`, `run_job.sh`, `fetch_results.sh`, `jupyter_tunnel.sh`, `instance.sh`, `environment.yml`. |

Hooks outside this directory (each a few lines, inert unless `storage == "aws"`):
`seb_analysis_common.py` (`add_data_source_args`, `available_regions`,
`resolve_data_root`, `resolve_region_dir`, `region_time_index`, `load_seb_data`),
`surface_classification.py` (`load_land_sea_mask`), `cloud_level_wind.py`
(`wind_archive_available`, `cloud_level_wind`).

## How it works

**Nothing is downloaded up front.** `open_dataset` lists the months it needs
(listings cached on disk), predicts every file's hourly time axis from its name
(the archive is strictly regular: monthly surface-analysis files, half-month
forecast files on a `(forecast_initial_time, forecast_hour)` grid, daily
pressure-level files) and builds an `xarray.Dataset` whose variables are lazy.
Building a Dataset for 11 seasons × 33 variables costs ~1 s and zero data reads.
The prediction is checked against each file's own coordinate variables the first
time the file is opened.

**Reads are exact and concurrent.** The analysis modules stream through
`iter_time_blocks`, i.e. `ds[vars].isel(valid_time=slice(i0, i1)).load()`. The
backend maps that block to HDF5 chunks, looks up each chunk's byte range (h5py is
used for metadata only, and the offsets are cached per file), fetches the missing
chunks with one concurrent multi-range request, inflates them itself
(gzip + byte-shuffle; `zlib` and numpy release the GIL, h5py does not) and
assembles the window. Measured from this laptop: 8 flux initialisations in 9.4 s
this way against 78 s through h5py, bit-identical.

**Everything read is cached.** The decoded regional slab of every HDF5 chunk is
written to `~/.cache/era5_s3/chunks/` (`ERA5_S3_CACHE` moves it, e.g. to the
external drive). The OV notebook makes four or five passes over `A.ds`; only the
first touches S3. Re-running with different thresholds, or the same box next
week, reads nothing from the bucket. A Barrow cold season of every SEB variable
is ~1 GB; scale by the number of cells.

**The forecast time axis.** Fluxes are means over the hour ending at
`forecast_initial_time + forecast_hour` (ECMWF, [ERA5 data documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)),
which is how the CDS stamps the local `avg_*` fields — hence the bit-for-bit
match. Hours 00–06 UTC on the 1st of a month live in the previous month's second
half-month file; the backend lists it automatically. (The earlier `era5_aws.py`
missed this and silently dropped those hours.)

**`tp` is synthesised.** The bucket has no hourly-accumulation `tp`; it has
`mtpr` (mean total precipitation rate, kg m⁻² s⁻¹, mean over the same hour).
1 kg m⁻² is 1 mm, so `tp [m] = mtpr × 3600 / 1000 = mtpr × 3.6`. Against the
CDS `tp` the difference is ≤ 1.5 × 10⁻⁶ m/hr (median 1.9 × 10⁻⁷): both fields are
GRIB-packed independently. At the OV filter threshold (0.05 mm/hr) that flips the
precipitating flag on 1.15 % of the flagged cell-hours (216 CDS-only, 7 S3-only
of 19,413 over Barrow, 1–4 Oct 2024), which moves the seasonal hours in `A.col`
by < 0.1 %. `tcslw` has no S3 source; `liquid_var="tcslw"` raises.

**Variables.** Canonical names throughout (`siconc`, not NCAR's `ci`). The
declared set is `era5_s3.SEB_STANDARD` (33 single-level variables — everything
the six figure modules read plus the rest of the downloader's *recommended* set);
declaring is free, only what a module streams is fetched. Pressure levels:
`<region>_pressure` gives `t q clwc ciwc cc` on the *troposphere* level set,
`<region>_pressure_wind` gives `u v` on the *lower* set, same order as the local
archives (`era5_pressure_variables.LEVEL_SETS`).

## What it costs, and where to run it

The cost is set by HDF5 chunking, not by your box:

| group | chunk | per unit | what a 6-month season costs, per variable |
|---|---|---|---|
| surface analysis (`tcc tclw siconc …`) | 27 h × 139 × 277 tile | ~0.5 MB per tile-day | ~0.2 GB (Barrow) … ~2 GB (Arctic circle) |
| mean fluxes (`msdwlwrf mtpr msshf …`) | **whole globe** per 12 h init | ~18 MB | **6.5 GB regardless of box** |
| pressure levels (`clwc u v`) | **whole globe** per hour | 8 MB (`clwc`) – 65 MB (`u`,`v`) | 35–290 GB regardless of box |

Measured laptop throughput with concurrent range requests: ~25 MB/s. So from the
laptop, per cold season:

* figures 1–3, 5–6 (`tcc tclw tciw siconc` + `tp` for the filter): ~7 GB, **~5–10 min** → a full 11-season run overnight is realistic, and it is cached afterwards;
* figure 7 (`+ msdwlwrf msdwlwrfcs`): +13 GB/season;
* figure 8 (8 flux variables): +50 GB/season → hours per season; EC2 territory;
* the cloud-level wind (`clwc u v`, 23/16 levels): ~200 GB/season → EC2 only.

`era5_s3.describe_cost(windows, variables)` prints this for any request, and
`run_analysis.py` prints it before starting.

In us-west-2 the transfer is in-region (free and ~1 GB/s); decoding becomes the
limit at ~150–300 MB/s of compressed input per core, so a 16-vCPU instance does
the whole 11-season, all-stages Barrow set in well under an hour, and a wide box
costs the same flux/PL traffic plus proportionally more analysis tiles. Set
`ERA5_S3_WORKERS` to the vCPU count there (the laptop default is 8).

Memory: the streaming passes hold `block_hours × n_cells × ~15` doubles.
`run_analysis.py` scales `block_hours` down from 720 as the box grows; in a
notebook pass `block_hours=` yourself for a box much larger than Barrow.

## Running remotely (EC2)

You need an AWS account with an access key that can create EC2 instances
(UCSD/SIO may have institutional accounts — ask research IT; otherwise a personal
account with billing alerts). **Reading the bucket needs no account at all**; the
account is only for renting a machine next to it. Setup, once, on the laptop:

```bash
brew install awscli
aws configure            # access key, secret, region us-west-2, output json
```

Then:

```bash
cd ERA5/surface_energy_budget/aws_pipeline/remote
./launch_instance.sh                       # c7i.4xlarge, 300 GB; SPOT=1 for ~35% of the price
#   ... ~5 min while bootstrap.sh installs micromamba, the env, and clones the repo ...
./sync_code.sh                             # REQUIRED: rsync this working tree onto it (see below)
./instance.sh ssh
   ./Python-Research/ERA5/surface_energy_budget/aws_pipeline/remote/run_job.sh \
        --region beaufort_chukchi --years 2014-2024 --stages all
   tail -f /data/results/latest.log
./fetch_results.sh                         # rsync figures + pickles back
./instance.sh stop                         # or terminate; a stopped instance keeps its cache
```

**`sync_code.sh` is how "tweak locally, run remotely" works.** The instance
starts from a clone of GitHub `main`, which is missing two things: whatever
you have not committed (this directory itself, today's edits) and the
observation inputs the lwph figures read — `genie_arm_monthly_hours.xlsx` is
gitignored (`*.xlsx`) and `genie_arm_cloud_durations.txt` is untracked.
`sync_code.sh` rsyncs the whole `ERA5/surface_energy_budget` working tree
(code and small inputs; never `data/`, `figures/`, caches) onto the instance,
so no commit is needed between an edit and a remote run. Run it again after
every local change. `run_job.sh` then skips `git pull` while the tree has
synced changes. (If a figure's input file is still missing, `run_analysis.py`
skips that figure with a message and records it in `manifest.json` rather than
losing the stage's `prepare()`.)

`jupyter_tunnel.sh` instead starts JupyterLab on the instance and forwards it to
`http://localhost:8890`, so the Ocean Visions notebook runs there unchanged
with `storage="aws"` in `COMMON` (after `sync_code.sh`).

The scripts were written against the AWS CLI v2 command set and Canonical's
public Ubuntu 24.04 AMI parameter, but have not been executed against a live
account from this machine (no credentials here). Expect to adjust the security
group if you are behind a changing IP, and check the console's price for the
instance type before a long job.

## Caveats you should know about

* **The ARM site.** `site_cell_mask` picks the grid cell nearest Utqiaġvik with no
  containment check. For a box that does not contain 71.323 N, 156.609 W, every
  "ARM cell" quantity — OV figures 1, 2, 3, 7a, the site columns of 7b/7c — is an
  arbitrary edge cell. The adapter prints a loud warning; it does not raise,
  because the domain-wide figures stay valid.
* **Barrow-specific plotting constants.** `map_liquid_hours._draw_one` draws
  gridline labels only for 170–145 W, 70–80 N and anchors the site legend at 72 N;
  the figure-8 subtitle and several slide titles say "Oct–Mar" regardless of the
  season; the thumbnail clips at 66 N. These are cosmetic and unchanged here.
* **Dateline-crossing boxes** (`west > east`) come back with a non-monotonic
  longitude coordinate. The classification and streaming code copes (label-based
  reindexing); the map code's `lon.min()/max()` does not. Prefer two boxes.
* **Full-circle boxes** (`arctic_circle`, W = −180, E = 180) give 1440 columns
  −180 … 179.75 exactly as the CDS does.
* **Recent months** are ERA5T; NCAR replaces them with final ERA5 later. The
  listing and metadata caches re-check months younger than 200 days, and the
  chunk cache is keyed by file size so a replaced file is re-read.
* **Missing months** (a group lagging, or a request into the future) raise with
  the first and last missing hour; `open_dataset(..., missing="nan")` continues with NaN.
* **Filters**: only gzip / shuffle / fletcher32 are decoded — all the bucket uses.
  A file with another filter raises `NotImplementedError` rather than guessing.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `ERA5_S3_CACHE` | `~/.cache/era5_s3` | listings, per-file metadata, chunk cache |
| `ERA5_S3_CHUNK_CACHE` | `on` | `off` disables the decoded-chunk cache |
| `ERA5_S3_WORKERS` | min(8, CPUs) | concurrent fetch+decode threads |
| `ERA5_S3_DEBUG` | unset | `1` prints one line per read (bytes, seconds, cache hits) |

## Relation to `era5_aws.py`

`era5_aws.py` (Aug 2026) was the first pass: eager, h5py-only, small windows.
It is left in place; this package supersedes it. Bugs it had that are fixed here:
the first six hours of each month missing from every flux variable; `+180` where
the CDS writes `−180`; `.ll025uv.` pressure-level files invisible to its listing
regex; silent loss of a whole month when one group lags another.
