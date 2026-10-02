# Running the Ocean Visions analysis on NSF NCAR Casper

The same notebook and the same analysis modules, reading ERA5 straight off
NCAR's GLADE disk instead of downloading anything. On Casper the ERA5 archive is
*local*, so the cost is pure compute — which is what the Data Analysis
allocation pays for.

Nothing about the science changes. The only switch is `storage="aws"` →
the reader finds GLADE and uses it; `OV_REGION`/`OV_YEARS` move the domain and
the seasons. The notebook itself is unedited between laptop and cluster.

## The short version

```bash
# laptop — push the working tree (carries uncommitted edits and the gitignored
# ARM observation .xlsx that figures 2 and 3 read)
ERA5/surface_energy_budget/aws_pipeline/casper/sync_to_casper.sh <user>@casper.hpc.ucar.edu

# Casper login node — once
cd ~/Python-Research/ERA5/surface_energy_budget/aws_pipeline/casper
./setup_casper.sh
python preflight.py

# Casper — a cheap smoke test first, then the real run
./submit.sh --project <PROJECT_CODE> --region barrow --years 2023 --stages lwph
./submit.sh --project <PROJECT_CODE> --region arctic_circle --years 2014-2024 --all

# laptop — bring the figures home
ERA5/surface_energy_budget/aws_pipeline/casper/fetch_from_casper.sh <user>@casper.hpc.ucar.edu
```

## Files

| File | What it does |
|---|---|
| `preflight.py` | **Run this first.** Finds the archive, checks packages and modules, does one real read and sanity-checks the values, verifies write space and cartopy data. Changes nothing. |
| `setup_casper.sh` | One-time: locates ERA5 on GLADE, picks the python environment, makes the cache/results/cartopy directories, writes `~/.era5_casper_env`, runs preflight. |
| `submit.sh` | Submits the work, right-sized per stage. `--dry-run` prints the `qsub` lines without submitting. |
| `run_stage.pbs` | PBS job for one or more `run_analysis.py` stages. |
| `run_notebook.pbs` | PBS job that executes `ocean_visions_figures.ipynb` end to end via papermill (or nbconvert). |
| `sync_to_casper.sh` / `fetch_from_casper.sh` | Laptop-side rsync both ways. |
| `prefetch_cartopy.py` | Stages Natural Earth shapefiles on a login node, because compute nodes may have no internet. |
| `environment_casper.yml` | Only used if you ask `setup_casper.sh` for a private conda env instead of NCAR's NPL. |

## Where ERA5 actually is

The Research Data Archive relaunched as the **Geoscience Data Exchange (GDEX)**
in September 2026 and **`ds633.0` was renamed `d633000`**. The reader tries, in
order:

```
/gdex/data/d633000                                  <- current
/glade/campaign/collections/gdex/data/d633000       <- campaign-storage equivalent
/glade/campaign/collections/rda/data/ds633.0        <- pre-GDEX, may still exist
/glade/collections/rda/data/ds633.0
/gpfs/fs1/collections/rda/data/ds633.0
```

`ERA5_GLADE_ROOT` overrides all of them. If none is found, `preflight.py` says
so and prints how to look. Below the root the layout is **identical** to the AWS
mirror — `<group>/<YYYYMM>/<file>.nc` — which is why one reader serves both and
why a chunk cache built on the laptop is still valid here.

The backend was verified against the AWS mirror on all four file layouts
(tiled surface analysis, whole-globe forecast fluxes, pressure levels,
invariant land-sea mask) including full-circle and Greenwich-crossing boxes:
**bit-identical**, 10–35× faster from local disk.

## Job sizing, and why it is split

Casper is a shared queue charged as **wall-clock × cores requested**. Memory is
free, and the wall-clock ceiling is **24 hours**. A complete pan-Arctic
(136,800 cells) 11-season pass of the notebook is ~34 CPU-hours — too much for
one single-threaded job inside 24 h — so `submit.sh --all` splits it:

| stage | figures | ncpus | mem | walltime | why |
|---|---|---:|---:|---:|---|
| `lwph` (+`dlr`, `thumb`) | 1–3, 7 | 4 | 120 GB | 23 h | three `prepare` passes plus two re-streams ≈ 0.9 of a flux pass |
| `maps` | 5–6 | 4 | 120 GB | 12 h | light |
| `extent` | 4 | 4 | 250 GB | 16 h | holds 9 bytes per cell-hour ≈ 59 GB pan-Arctic |
| `flux` | 8 | 8 | 200 GB | 23 h | the single most expensive pass; already threads ~3× |

**Cores are few on purpose.** The charge is elapsed wall-clock × cores
*reserved*, and the analysis loops are serial numpy — only the chunk decode
threads. Reserving 16 cores for a stage that can use 3 doubles the bill and
shortens nothing. **Walltime, by contrast, is a ceiling and not a charge**: a
job that finishes early pays for what it used, while one killed at the ceiling
pays in full and produces nothing. Hence generous walltimes and modest cores.
The only cost of a long request is queue position, since PBS backfills short
jobs first — trim them once a real run has written its timings to
`manifest_<stages>.json`.

Keep `mem` under 350 GB: above that PBS routes to the `largemem` queue, which
has far fewer nodes and longer waits. Over 733 GB it *must* go there.

`--notebook` instead runs all 69 cells in one 24-hour job. Use it for a domain
that fits, or as a second pass once the staged run has filled the chunk cache —
at which point re-running is nearly free.

## What the env file sets, and why

`setup_casper.sh` writes `~/.era5_casper_env`; every job sources it.

| Variable | Set to | Why |
|---|---|---|
| `ERA5_SOURCE` | `glade` | Fail loudly rather than silently reading over the network from a compute node |
| `ERA5_GLADE_ROOT` | the detected path | Pins it, so a GDEX move doesn't silently change behaviour |
| `ERA5_CACHE` | `$SCRATCH/era5_cache` | The notebook makes ~8 passes over the same variables; decoding once instead of eight times saves ~15 CPU-hours per run |
| `ERA5_WORKERS` | `$NCPUS` (in the job) | Decode threads = cores actually reserved |
| `CARTOPY_DATA_DIR` | `$WORK/cartopy_data` | Map figures would otherwise try to download shapefiles from a compute node |
| `MPLCONFIGDIR`, `TMPDIR` | scratch | Keeps caches off the 50 GB home quota |

Results go to `$WORK/era5_results` (2 TB, not purged), **not** scratch, which is
purged after 180 days.

## Running the notebook interactively

NCAR's JupyterHub runs a Casper session as a PBS job, charged the same way —
cores × wall-clock — so a 4-hour session on 4 cores costs 16 core-hours whether
or not it is computing. Close sessions you aren't using.

In a notebook session, set the domain before running cell 4:

```python
import os
os.environ.update(OV_REGION="arctic_circle", OV_YEARS="2014-2024",
                  OV_STORAGE="aws", OV_PL_REGION="barrow")
```

or export them in the terminal before launching the session.

## How the notebook is parameterised

Cell 4 now reads its settings from the environment, using the project's own
parsers so the spellings match the command line (`2014-2024`, `MM-DD`). **With
the environment empty every default is exactly what it was**, so a local run is
unchanged — verified value by value.

| Variable | Default | Notes |
|---|---|---|
| `OV_REGION` | `barrow` | any name in `era5_seb_variables.REGIONS`, or `box:N,W,S,E` |
| `OV_YEARS` | `2014-2024` | season START years |
| `OV_SEASON_START` / `OV_SEASON_END` | `10-01` / `03-31` | |
| `OV_STORAGE` | `local` | `aws` = the remote archive (GLADE here) |
| `OV_PL_REGION` | same as `OV_REGION` | **set to `barrow` on a big run** — see below |
| `OV_PL_YEARS` | `2024,2025` | the cloud-level wind seasons |
| `OV_FLUX_YEARS` | `2014-2025` | figure 8 runs one season longer |
| `OV_FLUX_MIN_LWP` | `2.0` | g m⁻² |
| `OV_MAP_SEASON` | `2024` | which season figure 6 draws month by month |
| `OV_SAVE_DIR`, `OV_DPI`, `OV_REPO_ROOT` | beside the notebook, 350, walk up to `.git` | `OV_REPO_ROOT` matters after an rsync, which carries no `.git` |

**Keep `OV_PL_REGION=barrow` on a pan-Arctic run.** Figure 4's cloud-level wind
is a single-grid-cell diagnostic at the ARM site, but `cloud_level_wind`
allocates six full-domain float32 arrays and reads 23-level `clwc`, `u`, `v`.
Pan-Arctic that is ~30 GB of arrays and a terabyte of pressure-level decode to
produce a histogram at one cell. Restricting it to the Barrow box is the
scientifically correct scope, not a compromise.

## One science decision still open

`ice_fraction_min` (the IWP/CWP fraction above which an hour counts as ice-only)
is **0.90 in every notebook cell except figure 8**, which uses 0.85 — while the
notebook's own cell-4 comment describes a 0.90-for-figures-1,2,7 /
0.85-for-figures-3–6,8 split. The code and the prose disagree; this predates the
Casper work and is noted in the project's own history as unresolved.

`run_analysis.py` now mirrors the **code**, cell by cell, so `--all` and
`--notebook` produce the same numbers. But it is worth settling before the talk:
on one Barrow month, moving the maps from 0.85 to 0.90 shifts about 3% of
overcast cell-hours from ice-only into liquid-containing (80.2% → 83.1%) and the
mean LWP by ~0.8 g m⁻². Change `ICE_FRACTION_MIN_*` in cell 4, or override per
run with `OV_ICE_FRACTION_MIN_COMPARISON` / `OV_ICE_FRACTION_MIN_DOMAIN`.

## Known gaps

- **The ARM-site figures only mean something where the ARM site is.**
  `site_cell_mask` picks the *nearest* cell with no containment check. On a
  pan-Arctic box that is still Utqiaġvik, so figures 1–3 and 7a stay valid —
  but on a box that excludes it they would silently describe an edge cell. The
  adapter prints a warning.
- **Map gridline labels are hardcoded to the Barrow strip** in
  `map_liquid_hours._draw_one` (170–145 W, 70–80 N). Figures 5, 6 and the
  thumbnail will draw pan-Arctic but without a labelled graticule.
- **NPL ships numpy 1.26, the laptop runs 2.x.** No numpy-2-only call is used by
  these modules, but the smoke test above is what settles it — run it before a
  long job.
- **These scripts have not been executed against a live Casper account.** They
  follow NCAR's documented syntax (`-q casper`, `select=1:ncpus=N:mem=XGB`,
  `module load conda`) and `--dry-run` shows exactly what would be submitted,
  but expect to adjust details on first contact.
- **Analysis-ready Zarr and kerchunk copies exist on GLADE** under
  `d633000/e5.oper.an.sfc.zarr/` and `.../kerchunk/`, and would be faster still
  for the surface variables. They are not used here: they lag the netCDF archive
  (ending mid-2025 in the current catalog) and are rechunked derivatives, so
  they would need their own bit-for-bit verification before the figures could
  rely on them. Worth doing if the netCDF path proves too slow.
