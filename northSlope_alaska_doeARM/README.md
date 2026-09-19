# NSA ARM Mixed-Phase Cloud Pipeline

Data pipeline and plotting tools for ground-based observations from the DOE
Atmospheric Radiation Measurement (ARM) **North Slope of Alaska (NSA)**
Central Facility (C1, Utqiagvik/Barrow, 71.323 N, 156.609 W, ~8 m MSL).

## Scientific context

This code supports the Ocean Visions project **"Observation-Based Assessment
of Mixed-Phase Cloud Thinning for Reducing Sea Ice Loss in Northern Alaska
Communities"** (PI: Lynn M. Russell, Scripps;
https://oceanvisions.org/arctic-atmosphere_russell/). The project asks, using
the multi-year NSA observational record:

1. How often are supercooled liquid cloud layers present in winter?
2. Would converting that liquid to ice (Mixed-Phase Cloud Thinning, MPCT)
   substantially cool the surface?
3. Would that cooling increase sea ice?
4. Are Indigenous Peoples and local communities open to investigating MPCT?

The motivating modeling study is Villanueva et al. (2022, *Environ. Res.
Lett.* 17, 114057, doi:10.1088/1748-9326/aca16d), which estimates that
seeding ice-nucleating particles into polar mixed-phase-regime clouds in
winter ("MCT") could offset roughly 25% of the expected polar sea-surface
warming from CO2 doubling and increase Arctic sea-ice area by ~8%. The
physical lever is the strong longwave surface warming by supercooled
liquid-containing clouds: glaciating them reduces downwelling longwave
radiation and lets the surface cool and the ice grow.

Two observational papers define the pipeline's instrument sets and recipes:

**Hartig et al. (2026)**, "Cloud liquid water path at the North Slope of
Alaska is largely insensitive to local meteorology in Arctic winter"
(EGUsphere preprint, doi:10.5194/egusphere-2026-2426 — "Hartig26" throughout
the code). That paper found liquid-containing clouds present 60-70% of the
time in November-March at NSA (about half with LWP < 10 g/m^2), which is
precisely the population MPCT would target. Reproducing and extending their
sonde-coordinated, multi-instrument dataset is the natural starting point for
question (1).

**Bertrand et al. (2025)**, "Increasing wintertime cloud opacity increases
surface longwave radiation at a long-term Arctic observatory" (*Nat. Commun.*
16, 9135, doi:10.1038/s41467-025-64441-8 — "Bertrand25"). Using 26 years of
QCRAD broadband radiation and 13 years (~2004-2019) of the Shupe-Turner
multi-sensor cloud microphysics product, they show wintertime net surface
longwave flux at NSA *increases* with warming (+0.20 to +0.92 W/m^2/K) and
attribute it to increasing cloud opacity — driven equally by ice-only clouds
(thin-to-opaque transitions, 0.44 ± 0.06 W/m^2/K) and liquid-containing
clouds (opacity + ice-to-liquid phase shifts, 0.43 ± 0.60 W/m^2/K). Their
radiation-environment instrument set and their clear / ice-only /
mixed-phase / liquid-only scene decomposition are implemented here
(see "Cloud phase occurrence and per-phase radiation" below); the same
decomposition is the observational baseline MPCT perturbations would be
measured against.

Background on the NSA site: Verlinde, Zak, Shupe, Ivey, and Stamnes (2016),
"The ARM North Slope of Alaska (NSA) Sites," *Meteor. Monogr.* 57,
doi:10.1175/AMSMONOGRAPHS-D-15-0023.1 (operating since 1997; radiosondes,
cloud radar, microwave radiometers, broadband radiation, and a large aerosol
suite at Utqiagvik).

## Instruments and datastreams

Two core sets: Hartig26 Table 1 (cloud/thermodynamics) and the Bertrand25
radiation-environment set. Use `--datastreams sonde mwr ceil kazr` /
`--no-extensions` where you want the strict Hartig26 subset.

| Key        | ARM datastream(s)                    | Instrument / product                | Variables used                  | Native resolution   | Role |
|------------|--------------------------------------|-------------------------------------|---------------------------------|---------------------|------|
| sonde      | `nsasondewnpnC1.b1`                  | Vaisala radiosonde, 0-4 launch/day  | T, Td, RH, p, wind vs. altitude | 5-8 m vertical      | hartig26-core |
| kazr       | `nsakazrcorgeC1.c1` (~2011-2014), `nsakazrcorgeC1.c0` (~2014-2019), `nsakazrcfrcorgeC1.c0` (~2019-2023) | Ka-band (35 GHz) zenith Doppler cloud radar, general mode | reflectivity, SNR | 4-5 s, 30 m | hartig26-core |
| mwr        | `nsamwrret1liljclouC1.c2` **and** `.c1` (c2 preferred where both exist) | Microwave radiometer + MWRRET v1 retrieval (Turner et al. 2007) | best-estimate LWP, PWV | 20-30 s | hartig26-core |
| ceil       | `nsaceilC1.b1`                       | Vaisala CL31 ceilometer             | first cloud base height, status | 15 s, 10 m          | hartig26-core |
| met        | `nsametC1.b1`                        | Surface meteorology station (2003-) | 2-m T and RH, wind, pressure    | 1 min               | bertrand25-core |
| mettwr     | `nsamettwrC1.b1`                     | Tower met, MET predecessor (~1998-2003) | 2-m T and RH               | 1 min               | bertrand25-core |
| qcrad      | `nsaqcrad1longC1.c2` **and** `.c1`   | Broadband radiometers + QCRAD VAP (Bertrand25 merges c2 with c1 to fill gaps) | down/up SW and LW fluxes | 1 min | bertrand25-core |
| shupeturn  | `nsamicrobase2shupeturnC1.c1`        | Shupe-Turner multi-sensor cloud microphysics (Shupe 2007 classifier; Shupe et al. 2015) | time-height hydrometeor phase code, LWC/IWC, LWP/IWP | ~1 min, ~2004-2019 | bertrand25-core |
| mplcmask   | `nsamplcmask1zwangC1.c1`             | Micropulse lidar cloud mask (Wang)  | cloud mask profiles             | ~30 s               | extension (registered only) |
| interpsonde| `nsainterpolatedsondeC1.c1`          | Sonde profiles interpolated to continuous time (RRTM input in Bertrand25) | T, RH, p profiles | 1 min | extension (registered only) |

Analysis periods encoded in [arm_nsa/config.py](arm_nsa/config.py): Hartig26
uses **2011-11-12 through 2023-12-31**, extended winter **November-March**;
Bertrand25 uses 1998-2023 for radiation trends, **December-March**, with the
Shupe-Turner phase record covering approximately **2004-2019**.

## How to get the data: API vs. local download

**ARM has a proper API.** Everything below is free; you need an ARM user
account (https://adc.arm.gov/armuserreg/#/new).

1. **ARM Live Data Web Service** (https://adc.arm.gov/armlive/) — a REST API:
   `query` lists files for a datastream + date range, `saveData` downloads
   one file. Authentication is `user=<username>:<access_token>`; your token
   is shown after logging in at that page. This is what
   [arm_nsa/download.py](arm_nsa/download.py) uses (standard library only, no
   extra dependencies), and also what the ARM-supported Python toolkit **ACT**
   (`act-atmos`, https://arm-doe.github.io/ACT/) wraps in
   `act.discovery.download_arm_data(username, token, datastream, start, end)`.
   Either path yields identical files.
2. **Data Discovery web portal** (https://adc.arm.gov/discovery/) — manual
   ordering; fine for a first look, not for a pipeline.
3. **ARM co-located computing** (https://www.arm.gov/capabilities/computing-resources):
   the **ARM Data Workbench** (interactive tools next to the archive) and the
   **Cumulus HPC cluster** at the ARM Data Center, for ARM-approved projects
   that need to process large volumes without downloading. There is no
   general-purpose "compute over the API" service — for big data you either
   download it or move the computation to ARM.

**Recommendation for this project:**

- **Download locally via the API** for sonde, MWR, ceilometer, MET/METTWR,
  QCRAD, and the Shupe-Turner product. These are a few MB to a few tens of MB
  per day; even multi-decade records total tens of GB, and having them on
  disk (or the lab server / Alpine scratch) keeps analysis fast and
  reproducible.
- **KAZR is the one that hurts**: full-resolution general-mode moments run
  ~0.5-2 GB/day, i.e. of order 1 TB for the Hartig26 period. Options, in
  increasing order of commitment:
  (a) start with a few case-study weeks locally (the pipeline reads radar one
  sonde-hour at a time, so partial archives are fine);
  (b) download month-by-month, run `build_sonde_library.py`, and delete raw
  radar files after each month — the coordinated library keeps only the
  reduced per-sounding quantities;
  (c) request an ARM Data Workbench / Cumulus allocation and run this package
  there next to the archive. For the full multi-year reproduction, (c) is the
  intended path.

Store credentials once (see [arm_nsa/credentials.py](arm_nsa/credentials.py)):
either environment variables `ARM_LIVE_USERNAME` / `ARM_LIVE_TOKEN`, or

```python
from arm_nsa.credentials import save_credentials
save_credentials("your_username", "your_token")   # writes ~/.armlive_credentials.json, mode 600
```

## Installation

```bash
cd northSlope_alaska_doeARM
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate nsa-arm
pip install -e .                         # optional; scripts also run from a checkout
```

Core runtime needs only `numpy pandas xarray netCDF4`; `matplotlib` for the
plot scripts; `dask`/`scipy`/`act-atmos` are conveniences (see comments in
[environment.yml](environment.yml)).

By default data land in `data/` inside the repo (gitignored). Point
`ARM_NSA_DATA_ROOT` at scratch/external storage to put them elsewhere.

## Quickstart

```bash
# 1. one winter month of the light instruments
python scripts/download_nsa_data.py --datastreams sonde mwr ceil met qcrad \
    --start 2022-01-01 --end 2022-01-31

# 2. a couple of radar days for a case study (large files!)
python scripts/download_nsa_data.py --datastreams kazr \
    --start 2022-01-05 --end 2022-01-06

# 3. quicklook plots
python scripts/plot_mwr_lwp_pwv.py          --start 2022-01-01 --end 2022-01-31
python scripts/plot_downwelling_radiation.py --start 2022-01-01 --end 2022-01-31
python scripts/plot_surface_temperature.py   --start 2022-01-01 --end 2022-01-31
python scripts/plot_sonde_day.py    --date 2022-01-05
python scripts/plot_kazr_quicklook.py --date 2022-01-05
# 4. build the sonde-coordinated multi-instrument library (Hartig26 Sect. 2)
python scripts/build_sonde_library.py --start 2022-01-01 --end 2022-01-31
#    (add --skip-radar while the KAZR archive is still partial)

# 5. sky-state / cloud-phase occurrence statistics (sonde-anchored heuristic)
python scripts/plot_cloud_phase_stats.py \
    --library data/processed/nsa_sonde_coordinated_library.20220101_20220131.nc

# 6. THE phase/radiation analysis (Shupe-Turner + QCRAD; the phase product
#    exists ~2004-2019, so pick a period inside that window):
python scripts/download_nsa_data.py --datastreams shupeturn qcrad \
    --start 2015-01-01 --end 2015-03-31
python scripts/analyze_phase_radiation.py --start 2015-01-01 --end 2015-03-31
```

Figures are written to `figures/`; processed netCDF to `data/processed/`.

## Package layout

```
arm_nsa/
  config.py       site constants, datastream registry, ALL analysis constants
                  (grids, thresholds, IWC coefficients) with Hartig26 citations
  credentials.py  ARM Live credential lookup/storage
  download.py     ARM Live REST client: query + atomic, resumable downloads
  qc.py           ARM embedded QC bitmask decoding (new + old attribute styles)
  readers.py      generic file discovery + canonical-variable time-series reader
  sonde.py        profile gridding (8-12000 m @ 5 m) + saturated-layer detection
  radar.py        KAZR: SNR screen, common grid, cloud fraction, IWP, clear-sky flag
  mwr.py          MWRRET LWP/PWV with unit normalization
  ceilometer.py   CL31 first cloud base + detection status semantics
  surface.py      MET / METTWR / QCRAD readers (Bertrand25 radiation set)
  shupe_turner.py Shupe-Turner phase product: reader, scene classification
                  (clear / ice-only / mixed / liquid-only), flux pairing
  coordinate.py   sonde-coordinated hourly library builder (Hartig26 product)
  phase.py        sonde-anchored sky-state heuristic (full-period fallback
                  where Shupe-Turner is unavailable)
  seb.py          surface energy budget: SEB datastream tiers, readers, Eq. (1)
                  terms with ERA5-compatible names (see the SEB section)
  bulk_flux.py    bulk-aerodynamic SH/LH (MOST) for the un-instrumented terms
  ground_flux.py  ground heat flux estimates (residual / conduction / inertia)
  cloud_water.py  IWP from reflectivity, cloud temperature at ARSCL boundaries
scripts/          CLI entry points (download_nsa_data.py, download_nsa_seb_data.py,
                  download_nsa_wind_data.py, build, plot scripts, and
                  analyze_phase_radiation.py)
tests/            unit tests + synthetic end-to-end integration tests
```

## Processing recipe (and where it comes from)

All from Hartig26 Sect. 2 unless noted; constants live in `config.py`.

- **Soundings** interpolated to a shared 8-12,000 m grid at 5 m; launches
  topping out below 1,000 m dropped.
- **Saturated layers** (proxy for liquid-containing cloud): RH w.r.t. liquid
  >= 95%; sub-threshold gaps <= 30 m bounded by saturated air are filled;
  layers < 30 m thick discarded. The 95% (not 100%) threshold reflects the
  ~3% radiosonde RH uncertainty and lidar-validated practice at NSA (Silber
  et al. 2020, 2021).
- **KAZR**: general mode; gates with SNR < -13 dB treated as no detection;
  profiles aligned to a 105-12,000 m / 30-m grid.
  Radar "cloud" fraction includes precipitation by construction.
- **IWP**: IWC = 0.1 * Ze^0.63 (IWC in g/m^3, Ze in mm^6/m^3; prefactor from
  SHEBA winter, Shupe et al. 2005; exponent from Matrosov 1999), integrated
  over height, computed after hourly averaging.
- **Clear-sky flag** (per sonde-hour): >= 99% of bins below 10 km undetected
  and no contiguous echo deeper than 100 m.
- **Coordination**: for each launch, KAZR / MWR / ceilometer (and MET/QCRAD
  extensions) are averaged over the hour after launch; missing data ignored;
  radar hours kept only with >= 50% sample coverage.
- **LWP interpretation**: theoretical MWRRET uncertainty is ~25 g/m^2 but
  clear-sky retrievals cluster within a few g/m^2 of zero; values below
  10 g/m^2 are treated as clear-sky-ambiguous and split out, never discarded.
- **Sky-state classification** (`phase.py`, this project's synthesis —
  Hartig26 does not define these exact categories): clear / ice_probable /
  liquid_probable / liquid_confident from the radar clear flag, LWP >= 10
  g/m^2, and saturated-layer presence. `liquid_probable + liquid_confident`
  is the MPCT-relevant "seedable scene" frequency estimate.

### Deliberate implementation choices to be aware of

Documented here because Hartig26's text leaves them open; each is trivially
changeable in `config.py` / function keywords:

1. **IWP averaging order**: the paper says IWP is computed "after the hourly
   averaging"; `radar.iwp_g_m2(average_first=True)` therefore averages linear
   Ze first, then applies the power law. `average_first=False` gives the
   per-sample alternative (Jensen's inequality makes it strictly smaller for
   variable Ze).
2. **Layer depth convention**: a saturated run of N grid cells counts as
   N * 5 m deep (cell-inclusive), so the 30-m minimum = 6 cells and gap
   filling tolerates up to 6 sub-threshold cells.
3. **Radar regridding**: nearest-gate reindexing (tolerance 30 m) rather than
   linear interpolation, to avoid blending reflectivities in dBZ space.
4. **Clear vs. outage**: radar data presence is recorded before SNR
   screening, so a silent-but-running radar (clear sky, IWP = 0) is
   distinguished from a radar outage (IWP = NaN, clear flag not claimable).

### Assumptions to verify against the first real download

Variable names inside ARM files were set from ARM conventions and product
documentation, not by opening every product era (candidate lists in
`config.py` absorb renames; readers fail loudly listing the file's actual
variables if a candidate is missing). On your first real month, check:

- KAZR reflectivity/SNR names in each of the three product eras, and the
  exact era boundary dates (the downloader simply queries all three names, so
  boundaries only matter for bookkeeping).
- MWRRET `.c2` availability across the full period (the `.c2` metadata page
  shows a shorter listed range than Hartig26 uses; if a gap appears, add the
  `.c1` datastream name to the registry as a fallback).
- QCRAD level (`.c2` vs `.c1`) and the downwelling-LW variable name.
- CL31 `detection_status` codes (module docstring documents the assumed 0-5
  convention; status 4 = obscured, common in blowing snow).
- Shupe-Turner `CloudPhaseMask` flag values (taken from Bertrand25's code;
  see the phase-analysis section caveats) and the optional LWC/IWC/LWP/IWP
  variable names (`Avg_Retrieved_LWC` etc. — the reader silently skips
  optionals it cannot find, so check what actually loaded).
- METTWR 2-m temperature/humidity names (tower ingests use per-level naming;
  candidates are guesses pending a real file).

## Cloud phase occurrence and per-phase radiation (Bertrand25 analysis)

**Question**: how often are pure-liquid, pure-ice, and mixed-phase clouds
present at NSA in winter, and what is the downwelling surface radiation under
each? This is the first observational target of the MPCT project, and
`scripts/analyze_phase_radiation.py` answers it following Bertrand25.

### Phase source: the Shupe-Turner product

Phase cannot be read off a single instrument. The Shupe-Turner retrieval
suite (`nsamicrobase2shupeturnC1.c1`; Shupe 2007 classifier, Shupe et al.
2015 microphysics) synthesizes KAZR Doppler moments, micropulse-lidar
depolarization (liquid strongly backscatters and barely depolarizes; ice the
reverse), ceilometer, MWR LWP, and radiosonde temperature into a
`CloudPhaseMask(time, height)` hydrometeor code at ~1-min resolution:

| Code | Volume classification | Scene grouping (Bertrand25) |
|------|----------------------|------------------------------|
| 0    | clear                | —                            |
| 1    | ice                  | ice                          |
| 2    | snow                 | ice                          |
| 3    | liquid               | liquid                       |
| 4    | drizzle              | (ungrouped; see note)        |
| 5    | liquid + drizzle     | liquid                       |
| 6    | rain                 | (ungrouped; see note)        |
| 7    | mixed-phase          | mixed                        |

The code groupings are copied from Bertrand25's published analysis code
(Zenodo doi:10.5281/zenodo.15786066) and live in
[arm_nsa/config.py](arm_nsa/config.py) (`ST_LIQUID_CODES` etc.).

### Scene classification rules

Each ~1-min profile collapses to ONE scene type
([arm_nsa/shupe_turner.py](arm_nsa/shupe_turner.py) `classify_scene`):

- **mixed_phase** — any mixed volume in the column, OR liquid and ice volumes
  both present anywhere in the column (even at different heights);
- **liquid_only** — liquid volume(s) present and not mixed-phase;
- **ice_only** — ice volume(s) present, no liquid, not mixed-phase;
- **clear** — valid profile, no hydrometeor volumes;
- **other_hydrometeor** — only ungrouped codes present (drizzle/rain-only;
  essentially absent in Arctic winter);
- **missing** — no valid phase data (excluded from occurrence denominators).

Two documented subtleties, both reproduced from Bertrand25's code on purpose:
(a) the paper text describes a laxer liquid-only rule (ice at the boundaries
of the liquid layer), but the published code uses the stricter column rule
above — we match the code; (b) drizzle (4) and rain (6) are not in the liquid
group, so a hypothetical drizzle+ice profile classifies as ice_only. Both are
one-line changes in config.py if you want different conventions.

### Radiation pairing and outputs

Each classified minute is paired with the nearest QCRAD sample within a small
tolerance (default 3 min; both records are ~1-min, so this is clock
alignment, not averaging). Net longwave (down − up, downward positive — the
Bertrand25 convention) is computed when upwelling is present. Expect the
wintertime net-LW distribution to be bimodal (Stramler et al. 2011):
liquid-containing scenes populate the opaque mode (net LW near 0 W/m^2),
clear and thin-ice scenes the radiatively clear mode (around −40 to −50
W/m^2), with ice-only scenes spanning thin to opaque — the population whose
opacity change Bertrand25 identifies as the main feedback driver.

```bash
python scripts/analyze_phase_radiation.py --start 2015-01-01 --end 2015-03-31
# Bertrand25's season definition (December-March) instead of Nov-Mar:
python scripts/analyze_phase_radiation.py --start 2014-12-01 --end 2015-03-31 \
    --months 12,1,2,3
```

Prints occurrence tables (overall + by month) and per-scene LW statistics;
writes a 4-panel figure (occurrence; LW-down histograms; LW-down box summary;
net-LW histograms), the classified minute-by-minute series as netCDF, and the
statistics table as CSV.

### Caveats for this analysis

- Shupe-Turner covers roughly **2004-2019 with gaps** (radar/lidar upgrades);
  occurrence statistics should always be reported with their sample counts.
  For phase questions outside that window, fall back to the sonde-anchored
  heuristic (`arm_nsa/phase.py`) — weaker, but full-period.
- Phase-code numbering (table above) was taken from Bertrand25's code, not
  from a product file header; on first real download, check
  `CloudPhaseMask`'s flag attributes and reconcile config.py if they differ.
- Radar-only volumes (no lidar overlap, e.g. above lidar extinction) carry
  more phase uncertainty; the product internally tracks retrieval pathways,
  which this pipeline does not yet expose.
- Bertrand25 additionally merges QCRAD `.c2` with `.c1` records into one
  continuous flux series; this pipeline downloads both but the reader uses
  whichever files are present per period (`.c2` and `.c1` land in separate
  directories — for trend work across 1998-2023, audit which level covers
  which years).

## Testing

```bash
python tests/test_pipeline.py                # 21 unit tests, synthetic arrays
python tests/test_shupe_turner.py            # 7 tests: scene rules + pairing
python tests/test_seb.py                     # 19 tests: SEB terms, bulk flux, G, IWP
python tests/test_integration_synthetic.py   # end-to-end on fabricated ARM files
# or: pytest tests/ -v
```

The integration tests fabricate ARM-format days in a temp directory and run
the real readers against them: (1) a Hartig26-style day (sounding through a
400-700 m liquid cloud, MWR LWP = 60 g/m^2, co-located KAZR echo, ceilometer
base at 450 m) through `build_library()` with analytic value checks, ending
in `phase_code = liquid_confident`; and (2) a Shupe-Turner day with four 6-h
scene blocks (clear/liquid/mixed/ice) plus QCRAD, checked for exactly 25%
occurrence each and correct per-scene flux statistics.

## Surface energy budget from observations (the ERA5 SEB twin)

`scripts/download_nsa_seb_data.py` + `arm_nsa/seb.py` are the observational
counterpart of `ERA5/surface_energy_budget/download_era5_seb.py` +
`seb_terms.py`: the same Sledd et al. (2025) Equation (1), the same tiers
(`--var-set core / recommended / extended`), the same output variable names
and sign conventions (`lwd_W_m2`, `sh_up_W_m2` positive upward, ...), so an
ERA5 grid cell and the NSA C1 instruments can be compared term by term.

```
LWD - LWU + SWD - SWU - SWT - SH - LH + G = M                       (Eq. 1)
```

```bash
python scripts/download_nsa_seb_data.py --dry-run          # what the archive holds
python scripts/download_nsa_seb_data.py --var-set core     # < 1 GB, minutes
python scripts/download_nsa_seb_data.py                    # recommended, ~14 GB
python scripts/download_nsa_seb_data.py --season 2024      # a different cold season
python scripts/build_nsa_seb_hourly.py                     # -> data/processed/nsa_seb_hourly_<start>_<end>.nc
```

`build_nsa_seb_hourly.py` turns the 1-min streams into the hourly file the ERA5
comparison needs: every Eq. (1) term, the bulk SH/LH, two G estimates,
merged LWP with provenance, PWV, ARSCL cloud fraction / lowest base / highest
top, Z-based IWP and precipitation (when `cldtype` is on disk), and the
temperature at the lowest cloud base from the nearest sonde (when `sonde` is
on disk). Cloud phase is left to a follow-up step on top of this file (see the
script docstring for why).

Everything below was established by querying the ARM Live archive and opening
one real file per datastream on **2026-09-15**, for the cold season
**2025-10-01 .. 2026-03-31** (182 days). Variable names, units, mounting
heights and QC conventions in `config.py` come from those files, not from
documentation.

### What Barrow measures, term by term

| Term | Status at NSA C1 | Source (key) | Reliability |
|------|------------------|--------------|-------------|
| LWD, LWU, SWD, SWU | **measured**, 1-min | `qcrad` (QCRAD1LONG c1, Long & Shi 2008 QC); `sirs` carries the second LWD pyrgeometer and the case/dome temperatures | highest: calibrated, ventilated radiometers. QCRAD c1 LWD is SKYRAD pyrgeometer 1 verbatim; on 2025-12-15 pyrgeometer 2 read 4.6 W/m^2 lower on average (up to 12 W/m^2) -- that spread is the in-situ LWD uncertainty and is the size of the SH term. LWU is measured at **10 m** (10-m-radius footprint). |
| T_skin | measured as IR brightness T, 1-min | `gndirt` (looks down from 10 m); also inverted from LWU | needs emissivity (0.985 as in Sledd) and reflected-sky correction (`seb.skin_temperature_from_irt`); the two routes agreed to -0.5 +- 0.4 K on a test week |
| T, RH, U, p | measured, 1-min | `met` (2 m T/RH, 10 m wind), `twr` (2/10/20/40 m) | high |
| SH, LH | **not measured** -- no eddy-correlation system has ever existed at C1 | `arm_nsa/bulk_flux.py` from T_skin, T/RH, U, p | parameterization: MOST with SHEBA (Grachev et al. 2007) stable functions and Andreas (1987) scalar roughness; z_0 is a free parameter (default 3e-4 m; +-x3 changes SH by ~5%) |
| G | **not measured** -- no SEBS, STAMP, snow depth or snow temperature at C1 | `arm_nsa/ground_flux.py` | weakest: residual of Eq. (5), snow conduction (needs non-ARM snow data), or thermal inertia from the T_skin history (Wang & Bras 1999) |
| SWT, M | zero in polar night / for frozen tundra | -- | -- |

The measured turbulent and soil fluxes nearest to Barrow are at **NSA E10,
Oliktok Point (~250 km ESE)**: `ecor_e10` (nsaecorsfE10.b1, 2024-10 .. present,
EddyPro SH/LH/u*/L) and `sebs_e10` (nsasebsE10.b1, soil heat-flux plates,
2011 .. present; positive **upward**, per its `standard_name` and its own
closing `surface_energy_balance`). They are in the `extended` tier for testing
the bulk parameterization on the same tundra type in the same season and must
never be plotted as Barrow data. Two cautions from the 2025-12-15 sample:
the E10 ECOR reported SH of +200 to +1400 W/m^2 in polar night with EddyPro
flag 0 -- a rimed sonic anemometer -- so `seb.read_ecor_e10` applies a
plausibility screen (|flux| < 150 W/m^2) and the winter record should be
expected to be sparse; and the E10 SEBS read G = +7 W/m^2 upward with the
5-cm soil still at -3.9 degC (active layer refreezing), which is the kind of
magnitude to expect for G at Barrow in early winter. Every ECOR/SEBS/STAMP datastream name was queried at
C1, C2, E10-E14 and M1 over 1998-2026; the full list of absences, with the
evidence, is `seb.UNAVAILABLE_AT_C1` and is printed by every run.

### Cloud state (the ERA5 tclw / tciw / cloud-temperature / phase analogues)

| Quantity | Source (key) | Notes for 2025/26 |
|----------|--------------|-------------------|
| LWP | `mwr` (MWRRET c2 -> c1) **and** `mwr3c` (3-channel MWR, 90 GHz) | MWRRET is the standard physical retrieval but this season has no retrieval on Dec 15-16, no files Dec 14 / Feb 15, and a bias episode (median -47 g/m^2 on Nov 15). MWR3C ran continuously; its regression LWP looked sane on those days (5-11 g/m^2 medians) but has no clear-sky bias correction and no qc_lwp. `cloud_water.merge_lwp` keeps provenance. |
| IWP | `cldtype` 1-min ARSCL reflectivity profile + `cloud_water.iwp_from_reflectivity` (IWC = 0.1 Z^0.63, phase-masked) | factor-of-2 class uncertainty. The only retrieved IWC at NSA is `microbase` (nsamicrobaseC1.c1) at **670 MB/day**, so it is excluded from every tier -- request by key. |
| cloud base/top | `arsclbnd` (c1 -> c0) | complete; clear-sky sentinels (-1/-2) converted to NaN + `sky_flag` |
| cloud temperature | boundaries placed in the sonde profile: `thermocldphase` `sonde_temp` (through Jan 20), then `sonde` launches or `interpsonde` (61 MB/day) via `cloud_water.cloud_temperature_at_boundaries` | |
| phase | `thermocldphase` c0 | processed **only through 2026-01-20** at the time of writing; nothing later exists yet. `mplcmask` depolarization also ends Jan 20. Re-run `--dry-run` later. |
| precipitation | `met` PWD fields; `cldtype` radar precip rate | no stand-alone gauge at C1 |

### Data quality of the 2025/26 season (found while building this; act on it)

**Downwelling pyrgeometer 1 failed for much of Dec 2025 - Feb 2026, and
QCRAD c1 reports pyrgeometer 1 verbatim.** Monthly medians / 95th percentiles
of the two SKYRAD pyrgeometers (from `sirs`):

| month | pyrg 1 (= QCRAD c1 LWD) | pyrg 2 | LWU | QCRAD c1 LWD surviving its QC |
|-------|------------------------|--------|-----|-------------------------------|
| Oct | 284 / 315 | 283 / 315 | 296 / 320 | 100% |
| Nov | 250 / 291 | 249 / 291 | 266 / 297 | 100% |
| Dec | 225 / **419** | 208 / 269 | 230 / 275 | 88% |
| Jan | 193 / **363** | 162 / 289 | 203 / 292 | 81% |
| Feb | **290** / **508** | 165 / 271 | 202 / 270 | **44%** |
| Mar | 204 / 272 | 204 / 272 | 221 / 278 | 100% |

LWD above ~300 W/m^2 is impossible in Arctic winter; 25% of Dec-Feb
pyrgeometer-1 samples exceed sigma T_2m^4 + 25 W/m^2 (0.4% for pyrgeometer 2).
Ventilators ran and dome-case temperature differences were ~0 throughout, so
this is a sensor fault, not dome heating. QCRAD's tests removed only the
extreme part: on the samples it kept, c1 LWD is **12-13.5 W/m^2 too high in
Dec-Jan**, and in Feb its 56% rejection preferentially removed cloudy periods,
leaving a clear-sky-biased record. Consequences and remedy:

- `seb.best_estimate_lwdn()` builds LWD from both pyrgeometers with an
  explicit, flagged rule (mean when they agree within 10 W/m^2; the plausible
  one otherwise; pyrgeometer 2 when both are "plausible" but disagree), and
  `seb.compute_seb_terms(..., sirs=sirs)` uses it. Over the season: 74% mean of
  both, 13.5% pyrgeometer 2 only, 12% flagged disagreements, 99.8% coverage
  (vs 86% for QCRAD c1). Season-mean LWD 213 W/m^2 vs 223 from QCRAD c1.
- Re-check when QCRAD **c2** appears for these dates: its best-estimate logic
  will arbitrate the two sensors properly and should supersede this rule.

**Skin temperature: IRT vs LWU disagree in calm clear conditions.** Corrected
IRT minus LWU-inverted skin temperature is -0.5 +- 0.4 K in Oct-Nov and Mar
but -1.3 to -2.6 K (std up to 3 K, 5th percentile -9.5 K) in Dec-Feb, worst
when pyrgeometer 1 is also bad. In those cases (mean wind 2.9 m/s) the IRT
puts the surface 5.7 K colder than the 2-m air -- the expected clear-sky
inversion -- while the LWU inversion puts it 0.7 K *warmer* than 2-m air,
which is not physical. A frost-covered downward-facing dome emitting near the
10-m air temperature would produce exactly that. Moderately confident
interpretation: **in calm clear periods the IRT is the better skin
temperature and LWU may be biased high by ~10 W/m^2** (3 K at 250 K). Not
corrected in the code; `compute_seb_terms` reports both (`t_skin_K` from the
IRT, `t_skin_from_lwu_K`) so the difference can be used as a screen.

**MWRRET LWP** has gaps and a bias episode (see the cloud-state table);
**MWR3C** LWP shows a ~+8 g/m^2 clear-sky offset (5th percentile) because it
has no clear-sky bias correction -- estimate and remove it against MWRRET in
clear periods before comparing with ERA5 tclw.

### Flux responses to radiative forcing (Sledd et al. 2025 framework)

Two notebooks at the repository root are the observational twins of
`ERA5/surface_energy_budget/turbulent_flux_response_to_DLR.ipynb` and
`..._to_DLR_and_SW.ipynb`:

| notebook | forcer | what it answers |
|---|---|---|
| `seb_flux_response_to_DLR.ipynb` | LWD | where an extra W m^-2 of downwelling longwave goes: LWU, bulk SH, bulk LH, SW_net co-variation, and the subsurface remainder |
| `seb_flux_response_to_DLR_and_SW.ipynb` | LWD + SW_net (Sledd's forcer) | the same four-term partition, month by month against the digitised MOSAiC responses of Sledd et al. Fig. 2a, the December Fig. 1 panels, their winter window, closure |

Both are generated by `scripts/make_flux_response_notebooks.py` (`--execute`
runs them) and hold no analysis of their own: everything comes from
`arm_nsa/flux_response.py` -- plain and T_2m-controlled least-squares
responses, the partition and its sign-honest ledger, a moving-block bootstrap
(7-day blocks) for intervals, monthly responses with r^2-sized markers, the
sensitivity of the bulk-flux responses to z_0 and the stable-stability scheme,
the 10-min vs hourly comparison, and the ERA5 grid cell nearest the site on the
same Eq. (1) terms. They read the two products of
`scripts/build_nsa_seb_hourly.py` (`--average 10min` for the Sledd cadence,
`1h` for ERA5) and write PNGs to `figures/seb_flux_response/`.

**Read the variables section of either notebook first.** At MOSAiC every
responder was an independent measurement; at Barrow only the radiative terms
are, so f_LWU is comparable with Sledd's while f_SH, f_LH (bulk
parameterizations) and f_G (thermal inertia) are all downstream of the same
measured skin temperature. Their sum closing to one is therefore consistency,
not closure; the testable statement is d(NA)/dF against d(-G_TI)/dF.

### Tiers and sizes (measured per-day sizes x 182 days)

| Tier | Keys | Size |
|------|------|------|
| core | qcrad gndirt met twr mwr arsclbnd | ~0.8 GB |
| recommended | core + mwr3c cldtype sonde thermocldphase sirs | ~14 GB (thermocldphase 7.6 GB for its 111 days) |
| extended | recommended + interpsonde ceil mplcmask ecor_e10 sebs_e10 | ~+17 GB |
| by key only | microbase (61 GB for Oct-Dec), kazr, skyrad, gndrad | |

Each run writes `data/processed/seb_manifest_<start>_<end>.json` with archive
vs. on-disk file counts, sizes and the absence list.

### Analysis modules

```
arm_nsa/seb.py          tiers + sizing, readers (gndirt, sirs, mwr3c, arsclbnd,
                        cldtype, ecor_e10, sebs_e10), skin temperature from LWU
                        and from the IRT, compute_seb_terms() with ERA5 names
arm_nsa/bulk_flux.py    MOST bulk SH/LH: Grachev 2007 / Beljaars 1991 stable psi,
                        Paulson 1970 unstable, Andreas 1987 z_T/z_Q, Murphy &
                        Koop 2005 saturation over ice, iterated in L
arm_nsa/ground_flux.py  G: residual, snow conduction (Sturm 1997 k), thermal
                        inertia (Wang & Bras 1999; validated against the
                        analytic half-space solution in tests/test_seb.py)
arm_nsa/cloud_water.py  IWP from reflectivity, cloud temperature at ARSCL
                        boundaries, LWP merging with provenance
scripts/download_nsa_seb_data.py   the downloader (tiers, dry-run, manifest)
scripts/build_nsa_seb_hourly.py    the hourly product for the ERA5 comparison
```

Two reader fixes came out of this work and apply package-wide: `qc.py` now
reads the QC bit table from the file's *global* `qc_bit_N_assessment`
attributes when the `qc_` variable has none (GNDIRT, the 40-m tower), so those
streams no longer drop merely-Indeterminate samples; and `download.local_files`
keeps the spec's preferred-datastream order so QCRAD c2 (and ARSCL c1) win
over c1/c0 wherever two levels overlap, instead of the alphabetical order that
silently inverted the preference.

## Downwelling longwave vs cloud microphysics (`dlr_cloud_microphysics_barrow.ipynb`)

The notebook asks what sets the large spread of downwelling longwave (DLR) at low
liquid water path in the ERA5 DLR-vs-LWP regressions, and whether cloud
microphysics -- droplet size, ice-particle size, liquid/ice layering,
precipitation -- has a measurable role, with mixed-phase cloud thinning as the
motivation. Two complete cold seasons, **Sep 2023 - Apr 2024** and
**Sep 2024 - Apr 2025**, at 1-min resolution.

Why those seasons: `microbasepi2` (the ARM "Continuous Baseline Microphysical
Retrieval, Profile-Instantaneous") exists at NSA only for **2002-01 .. 2011-03**,
while `thermocldphase` starts in Nov 2011 -- they never overlap. Its successor
`microbase` (nsamicrobaseC1.c1: 2011-11 .. 2014-06 and 2020-10 .. 2025-12) does,
but at **670 MB/day**, and 2023/24 + 2024/25 are the latest seasons with complete
phase, radiation and MICROBASE coverage (2025/26 thermocldphase ends 2026-01-20,
QCRAD pyrgeometer 1 failed Dec 2025 - Feb 2026).

What MICROBASE's effective radii are (Wang et al. 2025, DOE/SC-ARM-TR-095, Sect.
4.2): liquid r_e = 1.358 r_mode with r_mode from LWC assuming N = 200 cm^-3 and a
log-normal width sigma = 0.35, i.e. r_e ~ LWC^(1/3); ice r_e = (75.3 + 0.5895 T[C])/2
(Ivanova et al. 2001), i.e. temperature only; phase by temperature (all ice below
-16 C, linear to 0 C). Neither carries size information independent of LWC or T.
The notebook therefore derives a droplet radius from KAZR reflectivity + MWR LWP
(Frisch et al. 1995, lognormal spectrum, no N assumption) for liquid-only
non-drizzling columns, and uses Doppler fall speed and depolarisation as ice
proxies.

```
scripts/download_dlr_microphysics_winters.sh   qcrad, met, mwr, gndirt, cldtype, thermocldphase
                                               for both seasons (~37 GB, resumable)
scripts/reduce_microbase.py                    MICROBASE download-reduce-delete sampler:
                                               1-min LWC/IWC/r_e profiles + column integrals,
                                               every N-th day  -> data/processed/microbase_1min/
scripts/build_dlr_columns.py                   per-day THERMOCLDPHASE column features
arm_nsa/column_features.py                     -> data/processed/dlr_columns/<lidar>/
arm_nsa/dlr_dataset.py                         1-min merge (columns + QCRAD + MET + GNDIRT skin T
                                               + CLDTYPE + MICROBASE) -> data/processed/
                                               dlr_microphysics_1min_<start>_<end>_<lidar>.nc
scripts/make_dlr_microphysics_notebook.py      generates (and --execute runs) the notebook
```

Column features per 30-s profile (all on the VAP's 30 m grid by indexing, no
interpolation): phase pixel counts and a whole-column class (clear / ice_only /
liquid_only / mixed_column), liquid and ice base/top heights, ice pixels
below/above/within the liquid, ARSCL layer count and per-layer phase, sonde
temperatures at every boundary, low-level inversion strength, KAZR Ze / mean
Doppler velocity (positive up) / spectral width / LDR summarised over liquid and
ice pixels, a Z-based IWP proxy (IWC = 0.1 Ze^0.63) split below/above the liquid,
MPL depolarisation, MWR LWP/PWV. See the module docstring for the code groups.

Notebook sections: data inventory (what microphysical information exists at NSA
at all), coverage, MWR clear-sky LWP correction, the ERA5 figure reproduced with
ARM data, single-predictor correlations with autocorrelation-corrected sample
sizes, nested OLS (LWP -> cloud-base sigma T^4 -> PWV -> ice -> layering; skin
temperature only as a reference because it is a *response* to DLR), emission
temperature and a Brunt-type clear-sky baseline -> effective emissivity vs LWP,
surface-cloud temperature contrast, ice effects, liquid/ice layering classes,
multi-layer clouds, Frisch droplet radius, MICROBASE r_e (shown to be LWC^(1/3)
and T by construction), Doppler fall speed, and the MCT analogues (matched-bin
glaciation estimate, precipitating vs non-precipitating supercooled clouds).
Every interval is a day-block bootstrap. Figures go to `figures/dlr_microphysics/`,
key numbers to `figures/dlr_microphysics/summary_numbers.json`.

```bash
nohup bash scripts/download_dlr_microphysics_winters.sh > data/download_logs/dl_winters.log 2>&1 &
python scripts/reduce_microbase.py --start 2023-09-01 --end 2024-04-30 --every 5
python scripts/build_dlr_columns.py --start 2023-09-01 --end 2025-04-30
python scripts/make_dlr_microphysics_notebook.py --execute
# a quick test on any month already on disk:
DLR_NB_PERIODS="2025-11-01:2025-11-30" python scripts/make_dlr_microphysics_notebook.py --execute
```

## Not implemented yet (natural next steps)

- **Adiabatic LWP** (Hartig26 Eqs. 1-4, after Eytan et al. 2021) for the
  observed-vs-adiabatic comparison.
- **ERA5 + self-organizing maps** for large-scale circulation regimes
  (Hartig26 uses a 4x3 SOM on sea-level-pressure anomalies; not an ARM
  instrument, so out of scope for this pipeline stage). ERA5 tooling exists
  elsewhere in this repo under `ERA5/`.
- **RRTM-LW radiative closure and driver attribution** (Bertrand25 Methods):
  feeding Shupe-Turner microphysics + interpolated-sonde profiles to RRTM-LW
  to compute per-driver flux sensitivities and cloud radiative effect. This
  is the machinery that turns the per-phase occurrence/radiation baselines
  (implemented) into an MPCT perturbation estimate — the direct bridge from
  project question (1) to question (2). The `interpsonde` datastream is
  already registered for it.
- Shupe-Turner LWC/IWC *profiles* (read but unused): per-phase water-path
  distributions, e.g. Bertrand25's thin-to-opaque transition histograms.
- AERI retrievals and richer lidar products beyond the current set.

## References

- Hartig, K., J. J. Cassano, M. D. Shupe, A. Solomon (2026): Cloud liquid
  water path at the North Slope of Alaska is largely insensitive to local
  meteorology in Arctic winter. EGUsphere preprint,
  doi:10.5194/egusphere-2026-2426.
- Bertrand, L., J. E. Kay, G. de Boer (2025): Increasing wintertime cloud
  opacity increases surface longwave radiation at a long-term Arctic
  observatory. *Nat. Commun.* 16, 9135, doi:10.1038/s41467-025-64441-8.
  Analysis code: doi:10.5281/zenodo.15786066.
- Shupe, M. D. (2007): A ground-based multisensor cloud phase classifier.
  *Geophys. Res. Lett.* 34, L22809, doi:10.1029/2007GL031008.
- Shupe, M. D., et al. (2015): Deriving Arctic cloud microphysics at Barrow,
  Alaska: algorithms, results, and radiative closure. *J. Appl. Meteor.
  Climatol.* 54, 1675-1689, doi:10.1175/JAMC-D-15-0054.1.
- Villanueva, D., A. Possner, D. Neubauer, B. Gasparini, U. Lohmann,
  M. Tesche (2022): Mixed-phase regime cloud thinning could help restore sea
  ice. *Environ. Res. Lett.* 17, 114057, doi:10.1088/1748-9326/aca16d.
- Verlinde, J., B. Zak, M. D. Shupe, M. Ivey, K. Stamnes (2016): The ARM
  North Slope of Alaska (NSA) Sites. *Meteor. Monogr.* 57,
  doi:10.1175/AMSMONOGRAPHS-D-15-0023.1.
- Turner, D. D., et al. (2007): Retrieving liquid water path and precipitable
  water vapor from the Atmospheric Radiation Measurement (ARM) microwave
  radiometers. *IEEE TGRS* 45, 3680-3690, doi:10.1109/TGRS.2007.903703.
- Shupe, M. D., et al. (2005): Arctic mixed-phase cloud properties derived
  from surface-based sensors at SHEBA. *J. Atmos. Sci.* 63, 697-711.
- Matrosov, S. Y. (1999): Retrievals of vertical profiles of ice cloud
  microphysics from radar and IR measurements. *J. Appl. Meteor.* 38, 1245-1254.
- Stramler, K., A. D. Del Genio, W. B. Rossow (2011): Synoptically driven
  Arctic winter states. *J. Climate* 24, 1747-1762.
- ARM Live Data Web Service: https://adc.arm.gov/armlive/ — ACT toolkit:
  https://arm-doe.github.io/ACT/ — Computing resources:
  https://www.arm.gov/capabilities/computing-resources
