# ERA5 downloader — Arctic surface energy budget

Hourly ERA5 single-level data for a surface energy budget (SEB) analysis following
Equation (1) of:

> Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy balance
> responses to radiative forcing in the central Arctic from MOSAiC and models.
> *JGR Atmospheres*, 130, e2024JD042578. <https://doi.org/10.1029/2024JD042578>

## Files

| File | Purpose |
| --- | --- |
| `download_era5_seb.py` | CLI downloader, hourly/daily/monthly. Chunking, resume, retries, manifest. |
| `era5_seb_variables.py` | Variable registry (units + Eq. 1 role) and Arctic regions. |
| `seb_terms.py` | Post-processing: maps ERA5 fields onto Eq. (1), fixing the sign flip. |
| `seb_analysis_common.py` | Shared loading, ocean/ice masking, flux terms, area weights (analysis side). |
| `plot_turbulent_flux_maps.py` | 1×3 spatial maps of net / sensible / latent flux, time-mean over a range. |
| `plot_turbulent_flux_pdfs.py` | 1×3 probability density functions of the same three quantities. |
| `plot_radiative_flux_maps.py` | Radiative counterpart of the maps: net radiative / net LW / net SW. |
| `plot_radiative_flux_pdfs.py` | Radiative counterpart of the PDFs. |
| `plot_monthly_flux_maps.py` | Month-by-component grid of maps, with the sea-ice edge contoured. |
| `plot_monthly_longwave_maps.py` | 3-row monthly grid: downwelling LW, net LW, sea ice. No ocean mask by default. |
| `plot_fall_seb_timeseries.py` | Freeze-up season: climatological net SEB over open ocean (median + IQR) with the region's ice-free fraction beneath it. |
| `era5_aws.py` | Read ERA5 straight from the public NSF NCAR S3 bucket, no download. |
| `era5_aws_analysis.ipynb` | Notebook running the same analysis against that remote data. |

Note the analysis scripts keep the **native ERA5 positive-downward** convention,
unlike `seb_terms.py` which flips the turbulent terms to Sledd's positive-upward.
Both are documented in their module docstrings; do not mix their outputs.

All three plotting scripts take the same data-source options as the downloader:

```bash
python plot_turbulent_flux_maps.py --storage external --region barrow --mask all-ocean
```

`--storage local` (default) reads `data/` beside the scripts; `--storage external`
reads `EXTERNAL_ROOT` from `download_era5_seb.py`. Those roots are imported from
the downloader rather than duplicated, so editing that one constant moves both
the writing and the reading side. `--data-root PATH` overrides both. If a region
is missing from the chosen disk but present on the other, the error says so and
names the flag to use.

### The freeze-up time series

`plot_fall_seb_timeseries.py` targets the question of what the ocean's energy
balance is doing in the run-up to ice growth:

```bash
python plot_fall_seb_timeseries.py --storage external --region barrow
```

The net SEB, positive downward, needs no sign flips because all four ERA5 terms
share the convention:

```
SEB_net = msnlwrf + msnswrf + msshf + mslhf     [W m-2]
```

For each time-of-season step (1 Sep 12:00, 1 Sep 13:00, …) it pools every year
and every open-ocean cell (`siconc < --max-siconc`), then draws the
cos(lat)-weighted median with the 25th–75th percentile band. A second panel
below shows the ice-free fraction of the region's ocean area, so the reader can
see how much open water the statistics stand on. Steps with fewer than
`--min-cells` surviving samples are left blank — as freeze-up completes, the SEB
curve going blank *is* the signal that the region has closed. `--group day`
pools each calendar day's 24 hours for a smoother curve without the solar
diurnal cycle; `--season-start`/`--season-end` (MM-DD) move the window. The
"N years" label counts only years contributing data inside the season window.

## Status

Verified against a real download: 1–7 January 2026 over the `barrow` region,
`recommended` set, 7 daily files, ~2.0 MB each, on the external drive at
`SCRIPPS/ERA5/surface_energy_budget/barrow/`. All 34 variables present,
`expver = '0001'` (final ERA5), 92.7% of grid cells at sea-ice fraction > 0.9.

Sanity of the retrieved fields, Barrow strip, that week:

| Quantity | Value | Expected |
| --- | --- | --- |
| LWD | 152 W m⁻² mean (121–219) | plausible Arctic January |
| LWU | 204 W m⁻² mean | — |
| net LW | −52 W m⁻² mean | surface losing heat, correct sign |
| SWD, SWN | exactly 0 | polar night at 70–80° N in January |
| T_skin from LWU vs ERA5 `skt` | +0.014 K mean difference | independent consistency check on the LWU inversion |

## Quick start

Estimate size without contacting the CDS:

```bash
python download_era5_seb.py --dry-run
```

Small smoke test — Barrow strip, one day, ~8 MB:

```bash
python download_era5_seb.py --region barrow --start 2026-01-01 --end 2026-01-01
```

The configured default run — first week of January 2026, north of the Arctic Circle:

```bash
python download_era5_seb.py --storage external
```

## ⚠️ Sign convention — read this before using the data

**ERA5 defines every surface flux as positive DOWNWARD (into the surface).**
Sledd et al. Eq. (1) defines the turbulent fluxes as positive **UPWARD** and writes
them with leading minus signs. The mapping is therefore:

```
SH_sledd = -msshf          LH_sledd = -mslhf
```

so the group `- SH - LH` in Eq. (1) becomes `+ msshf + mslhf` in ERA5 variables.
Getting this backwards silently reverses both turbulent terms. `seb_terms.py`
applies the conversion in one place — use it rather than repeating the flip.

Source: [ECMWF, surface fluxes of sensible heat — positive downwards](https://sites.ecmwf.int/era/40-atlas/docs/section_B/parameter_sfoshpd.html),
[latent heat](https://sites.ecmwf.int/era/40-atlas/docs/section_B/parameter_sfolhpd.html).

## Reading from AWS instead of downloading

ERA5 is also on a public NSF NCAR bucket — `s3://nsf-ncar-era5`, us-west-2,
anonymous, no account: <https://registry.opendata.aws/nsf-ncar-era5/>.
`era5_aws.py` reads it directly and returns a Dataset **shaped exactly like a
file from `download_era5_seb.py`**, so every analysis function works unchanged.

```python
from era5_aws import load_named_region
ds = load_named_region("barrow", "2000-09-05", "2000-09-06")
```

Verified against the local CDS files over the same window: values match
**bit-for-bit** (`max|diff| = 0.00000`) and the land masks agree at 7.2%.

**It is the same data, but it is not faster from a laptop.** Reading from S3
avoids *storing* the data; the bytes still travel to wherever Python runs.
"Analysis in the cloud" only avoids that if the compute also runs in us-west-2.
Two structural facts make laptop-side reads expensive:

- **One variable per file**, each costing 12–15 s of HDF5 header reading over HTTP.
- **Mean-flux files are chunked `(1, 12, 721, 1440)`** — one chunk spans the whole
  globe, so a 0.24% spatial subset still inflates entire global chunks. (Analysis
  files are tiled `(27, 139, 279)` and subset well.)

Measured: **~500–700 s for 5 variables × 2 days**. The CDS returns the same
window with all 34 variables, server-side subset, in ~2.5 MB per day. So use AWS
for exploratory work with no queue to wait on, or on an EC2 instance in
us-west-2; keep the CDS downloader for the multi-decade local archive.

Differences to know about: short names are the original GRIB ones (`ci` not
`siconc`, `2t` not `t2m` — `NCAR_VARS` maps them all), longitude is stored
0–360, mean-flux time is a 2-D `(init_time, forecast_hour)` grid, `tcslw` is
**absent**, and `tp` is not archived as an accumulation (use `mtpr`).

One trap worth naming: raw `h5py` does **not** apply CF `_FillValue`, and these
files store land as `9.999e+20` rather than NaN. Left unhandled, every land cell
becomes a large finite number, `build_ocean_mask` sees no land, and an
`all-ocean` average silently includes the Alaskan North Slope. `_apply_cf()`
handles it; a values-only comparison masked with `isfinite` on both sides cannot
detect the mistake, so the notebook's check compares NaN patterns and land
fractions too.

## Choosing a temporal resolution

`--frequency` selects one of three CDS datasets. All three carry the same
45-variable catalogue, so any `--var-set` works at any frequency.

| `--frequency` | Dataset | Extra options |
| --- | --- | --- |
| `hourly` (default) | `reanalysis-era5-single-levels` | — |
| `daily` | `derived-era5-single-levels-daily-statistics` | `--daily-statistic` |
| `monthly` | `reanalysis-era5-single-levels-monthly-means` | `--monthly-product` |

```bash
python download_era5_seb.py --frequency monthly --region arctic_circle --start 2000-01-01 --end 2025-12-31
```

Files land in `<region>_<frequency>/` (hourly stays in plain `<region>/`, so
existing downloads still resume). Chunking defaults to one file per day, month,
and year respectively.

### Monthly means are ~260× smaller and numerically equivalent

Verified on the Barrow strip, January–March 2026, by comparing the monthly-means
product against the monthly average of the hourly files:

| Field | Agreement (complete months) |
| --- | --- |
| `skt`, `t2m`, `siconc` | within 0.003% |
| `msdwlwrf` | within 0.1% |
| `msshf`, `mslhf` | within 0.4 W m⁻² per cell |
| On-disk size | 159 MB hourly vs 0.61 MB monthly — **262× smaller** |

Residuals are at the level of ERA5's archived packing precision. For an
**unmasked** monthly average, downloading hourly data and averaging it yourself
buys nothing.

### But a monthly mean cannot be masked to ice-free conditions

This is the real constraint, and it decides the question for you:

- A cell whose **monthly-mean** `siconc` is 0.5 was ice-covered for roughly half
  the month. Its monthly-mean flux already blends open-water and ice-covered
  hours, and no post-hoc filter can separate them.
- Masking has to happen **before** averaging. Masking and averaging do not
  commute.

So:

| What you want | Use |
| --- | --- |
| Monthly maps over all ocean, ice included | **monthly** — same answer, 260× less data |
| Seasonal cycle, sea-ice edge as a continuous field | **monthly** |
| Fluxes conditioned on ice-free water only | **daily** or hourly — the mask must precede the average |
| Distributions, extremes, PDFs, polynya events | **daily** keeps most of it; hourly for the far tail |

### Daily is a strong middle ground, not just a compromise

Measured on the Barrow hourly record (Jan–Mar 2026) by aggregating it to daily
means and comparing against the hourly truth:

| Property | Retained at daily resolution |
| --- | --- |
| Standard deviation of net turbulent flux | 94.6% |
| 1st-percentile (cold) tail | 97.0% |
| Most extreme single value | 91.8% |
| Ice-mask agreement vs hourly (`siconc` < 0.15) | 99.997% of cell-days |

Two reasons daily costs so little here:

1. **Arctic turbulent flux variability is synoptic, not diurnal.** It is driven by
   multi-day weather systems, and the test period is polar night with no solar
   diurnal cycle at all. Daily averaging removes variance that is largely absent.
2. **Sea ice concentration evolves over days, not hours.** The mean absolute
   change is 0.0004 per hour against 0.009 per day, so a daily-mean ice mask
   reproduces an hourly mask almost exactly. This is the key result: unlike
   monthly means, daily means *can* carry a meaningful ice-free mask.

Caveat: that test covers January–March. In summer the solar cycle is present even
at 70–80° N, so daily averaging will lose more diurnal variance then. Re-check
before applying the same reasoning to a melt-season analysis.

### Storage, daily means, 2000-01-01 to 2025-12-31

9,497 days (26 calendar years). Projected from a measured **1.768 bytes per
cell-day-variable** — the compressed size of this repository's own hourly Barrow
data aggregated to daily means and written with the downloader's zlib settings
(2.26× compression).

| Region | grid cells | `core` (17) | `recommended` (34) | `extended` (45) |
| --- | ---: | ---: | ---: | ---: |
| `barrow` | 2,501 | 0.7 GB | **1.3 GB** | 1.8 GB |
| `arctic_circle` | 136,800 | 36 GB | **73 GB** | 96 GB |

Land cells barely help: only `cbh` and `siconc` are NaN over land, and the flux
variables are defined there too, so the pan-Arctic figure is a fair estimate
rather than a ceiling.

For reference, the same span at **hourly** resolution would be roughly 24× these
numbers — about 31 GB for Barrow and 1.7 TB for the Arctic Circle.

### Request count is the real cost, not bytes

The CDS costs a request by how much **hourly** ERA5 it must touch —
`variables × days × 24` — not by the size of the output. Probed ceilings
(largest request verified accepted, and the smallest observed rejection):

| frequency | verified accepted | rejected with 403 |
| --- | ---: | ---: |
| hourly | 12,648 fields | 25,296 fields |
| daily | 8,160 fields | 12,240 fields |

Two consequences that are easy to get backwards:

**The daily product has the tighter ceiling**, because it still reads all 24
hourly steps to form each daily mean. Its output is 24× smaller; its request cost
is not. For the same span and variable set, daily therefore needs **more**
requests than hourly, not fewer.

**Request count drives wall time, and the CDS throttles high-volume users.** An
overnight run of 9,497 single-day requests degraded from 1.8 min per request to
34.3 min per request over roughly 280 requests — a 19× slowdown — which puts the
remaining work at over 200 days.

For 2000–2025 over one region, with the 34-variable `recommended` set:

| configuration | cost/request | requests | |
| --- | ---: | ---: | --- |
| hourly, `--chunk day` | 816 | 9,497 | what not to do |
| hourly, `--chunk-days 15` | 12,240 | **806** | verified accepted |
| hourly, `--var-set core --chunk month` | 12,648 | **312** | verified; only 17 variables |
| hourly, `--chunk month` | 25,296 | 312 | rejected (403) |
| daily, `--chunk-days 10` | 8,160 | 1,118 | verified; *more* requests than hourly |
| monthly | — | 26 | verified |

`--chunk-days N` sets the days per request directly; chunks never span a month
boundary and each month is split into near-equal groups. The plan output reports
the per-request field count and warns when it exceeds the verified ceiling.

### The daily product works, but it is computed on demand and is slow

Unlike hourly and monthly, which serve pre-computed archive fields,
`derived-era5-single-levels-daily-statistics` calculates the statistic per
request. A verified minimal request (3 days, 2 variables, Barrow) took **68
minutes** from `accepted` to `successful`, against under a minute for monthly and
about 1.5 minutes for an hourly day-chunk.

26 years is 312 requests. **Time a single month before committing to a full
download**, and be aware that requesting hourly data and averaging it locally may
finish sooner despite moving 24× more bytes.

An earlier test in which all 12 chunks failed turned out to be a local DNS outage
(`Failed to resolve 'cds.climate.copernicus.eu'`), not a problem with the request
or the dataset licence. The request shape is confirmed correct against the
dataset's live schema.

## What the CDS actually returns

Two behaviours of the current CDS backend that the download handles for you.
Both were found by running a real request, not by reading the docs.

**1. It returns a ZIP, not a netCDF.** When a request mixes GRIB `stepType`s —
and ours always does, combining instantaneous fields, time-mean fluxes, and
accumulated precipitation — the CDS ignores `download_format: "unarchived"` and
returns a zip of three files:

```
data_stream-oper_stepType-instant.nc   23 vars: skt, siconc, t2m, tcc, istl1-4, ...
data_stream-oper_stepType-avg.nc       10 vars: the mean flux terms
data_stream-oper_stepType-accum.nc      1 var:  tp
```

`consolidate_to_netcdf()` merges these onto their shared
`(valid_time, latitude, longitude)` grid and writes one compressed netCDF per
chunk. The merge uses `join="exact"`, so a coordinate mismatch between streams
raises rather than silently broadcasting.

**2. The flux variables are not named what the ERA5 docs say.** The netCDF
backend writes `avg_*` names, not the GRIB short names:

| CDS netCDF name | canonical | paramId | |
| --- | --- | --- | --- |
| `avg_sdlwrf` | `msdwlwrf` | 235036 | LWD |
| `avg_snlwrf` | `msnlwrf` | 235038 | net LW |
| `avg_sdswrf` | `msdwswrf` | 235035 | SWD |
| `avg_snswrf` | `msnswrf` | 235037 | net SW |
| `avg_ishf` | `msshf` | 235033 | sensible heat |
| `avg_slhtf` | `mslhf` | 235034 | latent heat |
| `avg_sdlwrfcs` / `avg_snlwrfcs` | `msdwlwrfcs` / `msnlwrfcs` | 235069 / 235052 | clear-sky LW |
| `avg_sdswrfcs` / `avg_snswrfcs` | `msdwswrfcs` / `msnswrfcs` | 235068 / 235051 | clear-sky SW |

Note `avg_ishf` is the **sensible** heat flux and `avg_slhtf` the **latent** one —
the abbreviations do not make that obvious, so each mapping was verified against
the `GRIB_paramId` attribute in a downloaded file. Files written by this script
are normalised to the canonical names. `compute_seb_terms()` calls
`normalise_names()` itself, so it also accepts a file pulled straight from the CDS.

**3. Monthly means timestamp their streams at different hours.** The monthly
product splits into `avgad`/`avgid`/`avgua` streams; the time-mean flux streams
land on **06:00** of the first of the month while the instantaneous stream lands
on **00:00**. Both describe the same calendar month, but the offset makes an
exact-join merge fail. `_floor_time_to_month()` discards the hour before merging,
and is applied only at monthly resolution — at hourly or daily resolution a time
offset is a real difference and must not be flattened.

Note also that a failure *after* a successful transfer is never retried: the
payload is already on disk and will not merge differently on a second attempt, so
it is kept as `.raw_unmerged` for inspection instead of burning another CDS queue
wait. Only transfer failures back off and retry.

## Equation (1) mapping

Sledd et al. Eq. (1), turbulent fluxes positive upward:

```
LWD − LWU + SWD − SWU − SWT − SH − LH + G = M
```

In ERA5 variables:

```
msnlwrf + msnswrf − SWT + msshf + mslhf + G = M
```

| Eq. (1) term | ERA5 source | Notes |
| --- | --- | --- |
| LWD | `msdwlwrf` | direct |
| LWU | `msdwlwrf − msnlwrf` | ERA5 archives the net, not the upwelling |
| SWD | `msdwswrf` | direct |
| SWU | `msdwswrf − msnswrf` | as above |
| SWN = SWD − SWU | `msnswrf` | this is the SW half of the forcing term |
| SH | `−msshf` | **sign flip** |
| LH | `−mslhf` | **sign flip** |
| SWT | — | **not in ERA5**, see below |
| G | `istl1…istl4` (indirect) | **approximate only**, see below |
| M | — | **not in ERA5** for sea ice, see below |

Radiative forcing of the Miller et al. (2017) framework used throughout the paper
(their Section 3.2): `forcing = LWD + SWN = msdwlwrf + msnswrf`.

Eq. (5), the net atmospheric flux: `NA = msnlwrf + msnswrf + msshf + mslhf = M − SWT − G`.

## Known limitations

These are real constraints on how far ERA5 can reproduce the Sledd et al. analysis.

**SWT is not an ERA5 output.** Sledd et al. derive it from Beer's law (their Eq. 6)
using MOSAiC-specific snow and ice optical properties — an extinction coefficient
that changes through the melt season, plus a visible/near-IR split. Nothing
equivalent exists in ERA5. This is harmless for a January analysis, since SWT is
zero in polar night, but it blocks a direct summer application.

**M (melt energy over sea ice) is not archived.** ERA5 exposes `mean_snowmelt_rate`
for the *land* tile only. In Sledd et al. M is a residual anyway, so it can be
computed as one — but only once every other term is trustworthy.

**G is approximate at best, and ERA5's sea ice is structurally different from
MOSAiC's.** ERA5's sea ice is a fixed 1.5 m slab with **no snow layer on top**.
Sledd et al.'s own Table 2 lists exactly this for the ECMWF IFS, which is ERA5's
forecast model. Since snow is the dominant thermal insulator over Arctic sea ice,
an ERA5-derived conduction term is not comparable to the MOSAiC IMB-based G in
their Eqs. (2)–(4). Treat any ERA5 G as a model diagnostic, not an observational
analogue.

*Confidence note:* I confirmed the 1.5 m no-snow configuration from the Sledd
paper's own Table 2. I could **not** find authoritative documentation for the
individual `ice_temperature_layer_1…4` thicknesses — commonly quoted as
0.07 / 0.21 / 0.72 / 0.50 m, summing to 1.5 m, and one ECMWF source corroborates
0.07 m for the top layer, but I did not verify the full set. **Confirm the layer
depths against the IFS documentation before computing any temperature gradient
from them.**

**ERA5 snow variables are land-only.** `snow_depth` and `snow_density` (in the
`extended` set) describe the land tile. They are *not* snow on sea ice. Included
for coastal contrast only.

**Check `expver` in the output.** Data within roughly three months of real time is
preliminary ERA5T rather than final ERA5, and the two can appear merged in one
file. The January 2026 test download came back `expver = '0001'` throughout, i.e.
final ERA5, so this is not a concern for the current analysis period.

**`total_precipitation` (`tp`) is accumulated in metres over the preceding hour**,
unlike the `mean_*` flux variables which are already rates. The `extended` set adds
`mtpr` and `msr` (kg m⁻² s⁻¹) if you prefer consistent rate units and a snow/rain split.

## Variable sets

Select with `--var-set`.

- **`core`** (17) — exactly the originally requested list: the six flux terms, four
  cloud-cover fields, five column hydrometeor fields, total precipitation, cloud base height.
- **`recommended`** (34, default) — `core` plus what Eq. (1) actually needs to close
  and what restricts the analysis to Arctic Ocean sea ice:
  - `skin_temperature` — drives LWU; the growth-vs-melt regime distinction that
    organises the whole paper depends on whether T_skin is free or pinned at melting.
  - `sea_ice_cover` — the sea-ice mask. The analysis is over ice; without this you
    cannot separate ice from open ocean.
  - `ice_temperature_layer_1…4` — the only route to a subsurface term (with the caveats above).
  - `forecast_albedo` — the control on SW absorption emphasised throughout the paper.
  - four clear-sky flux fields — give surface cloud radiative effect, the natural
    decomposition of a cloud-driven forcing term.
  - `total_column_water_vapour` — the other first-order control on Arctic LWD besides cloud.
  - `2m_temperature`, `2m_dewpoint_temperature`, `10m_u/v_wind`, `surface_pressure` —
    near-surface stability and wind, which the paper invokes to explain the SH response.
- **`extended`** (45) — adds boundary layer height, friction velocity, precipitation
  rates, land snow properties, SST, and TOA fluxes.

## Regions

`--region` accepts:

| Name | Area [N, W, S, E] | ~Size, recommended set |
| --- | --- | --- |
| `arctic_circle` | 90, −180, 66.5, 180 | 426 MB/day, 2.9 GB/week |
| `arctic_70n` | 90, −180, 70, 180 | 363 MB/day, 2.5 GB/week |
| `central_arctic` | 90, −180, 80, 180 | 184 MB/day, 1.3 GB/week |
| `barrow` | 80, −165, 70, −150 | 7.8 MB/day, 0.05 GB/week |
| `beaufort_chukchi` | 80, −170, 68, −120 | 31 MB/day, 0.21 GB/week |
| `custom` | edit `CUSTOM_REGION` in `era5_seb_variables.py` | — |

Sizes are uncompressed upper bounds; netCDF output is typically 2–4× smaller.

For a one-off box, skip `custom` and pass the corner coordinates directly — this
overrides `--region`:

```bash
python download_era5_seb.py --area 82 -60 72 -10
```

## Storage

`--storage local` (default) writes to `./data/<region>/` beside the script. The
repository `.gitignore` already excludes `*.nc`, so downloads are not committed.

`--storage external` writes to `EXTERNAL_ROOT` at the top of `download_era5_seb.py`,
currently:

```
/Volumes/My Passport/SCRIPPS/ERA5/surface_energy_budget/
```

The volume had 677 GB free at setup. It is a case-insensitive filesystem, so
`Scripps` and `SCRIPPS` both resolve; the on-disk casing is used in the constant.

If the drive is not mounted, the script refuses to run rather than quietly filling
the boot drive at that path. `--out-dir` overrides both.

## Behaviour worth knowing

- **Resume works in date space, not filename space.** Before planning, the run
  scans the output directory and decodes which days each existing file covers from
  its name, then drops those days. A range already downloaded as 275 one-day files
  is therefore recognised by a later `--chunk-days 15` run. Holes left by earlier
  failures are picked up automatically as their own small chunks. `--overwrite`
  bypasses the scan. The raw payload lands on a `.raw.part` scratch path and the
  merged netCDF is moved into place only once complete.
- **Chunks are always contiguous day runs**, so a `DD-DD` filename honestly
  describes its contents — a gapped chunk named `05-23` holding only the 5th and
  23rd would make the next resume skip the 6th through 22nd.
- **Ctrl-C does not cancel the CDS request.** The job keeps running server-side and
  holds one of your few concurrent queue slots. Abandoned jobs throttle everything
  after them; clear them at <https://cds.climate.copernicus.eu/requests>.
- **`--jobs N` does not make downloads faster.** The CDS enforces *"the maximum
  number of per-user requests that access the CDS-MARS data is 1"* — extra
  submissions only queue behind the running one. An earlier claim here that
  `--jobs 3` gave a 3× speedup was wrong: that measurement (44.8 min for 3 chunks
  = 14.9 min/chunk) matches the *sequential* rate observed later (7.5–14.4
  min/chunk), so the requests had been running one at a time all along. Default
  is 1. N>1 also puts the netCDF merge on worker threads, where HDF5 is not
  thread-safe — this environment links *two* copies of it (h5py against libhdf5
  1.14.5, netCDF4 against 1.14.6), which is what produced the
  `HDF5-DIAG ... thread 1` noise. A lock guards the merge, but staying at
  `--jobs 1` avoids the situation entirely.

### How large can one request be?

The ceiling is on **fields = variables × days × 24**, probed empirically at
~12,648 accepted / 25,296 rejected (403). Since only one request runs per user,
total wall time is `requests × minutes-per-request`, and the only lever on
request count is the variable count:

| variable set | vars | max days/request | requests for 2019–2022 |
| --- | ---: | ---: | ---: |
| `core` | 17 | 31 (a full month) | 48 |
| `recommended` | 35 | 15 | 98 |
| `extended` | 45 | 11 | 133 |

So with the 35-variable `recommended` set: **one year is 24× over the limit, one
month is 2× over, and 15 days fits** — `--chunk-days 15` is already at the
ceiling. At ~10 min/request, 2019–2022 is roughly 17 hours. Halving the variable
count halves the request count; nothing else does.
- **`--cds-retries N` bounds connection retries** (default 10, roughly N × 2 min).
  cdsapi's own default is **500**, so a dropped connection is retried for over
  **16 hours** while making zero progress — an overnight run was found stuck at
  "attempt 248 of 500". Failing fast is strictly better here because resume
  restarts exactly where it stopped.
- **Retries.** Four attempts per chunk with exponential backoff (30 s, 60 s, 120 s).
  A chunk that exhausts its retries is recorded as failed and the run continues.
- **Manifest.** Each run writes `manifest_<start>_<end>.json` recording the region,
  area, period, full variable list with units and roles, grid, and per-file status.
- **Chunking.** `--chunk day` (default) or `--chunk month`. Daily keeps individual
  CDS requests small and makes resume fine-grained; monthly means fewer queue waits.

## Credentials

Requires `cdsapi` and `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your personal access token>
```

The current endpoint has **no** `/v2` suffix, and the key is the bare token, not the
old `UID:KEY` pair. The script warns if it finds the retired URL. Your existing
`~/.cdsapirc` is already in the correct format.

## Analysis example

```python
import glob
import numpy as np
from seb_terms import compute_seb_terms, open_seb_files

ds = open_seb_files(sorted(glob.glob("data/barrow/era5_seb_barrow_2026*.nc")))
seb = compute_seb_terms(ds)          # Sledd sign convention applied

# Restrict to consolidated sea ice, as in the paper
ice = seb.where(ds["siconc"] > 0.9)

# The Miller et al. regression: response of LWU to radiative forcing
x = ice["forcing_W_m2"].values.ravel()
y = ice["lwu_W_m2"].values.ravel()
m = np.isfinite(x) & np.isfinite(y)
slope, intercept = np.polyfit(x[m], y[m], 1)   # compare against ~0.5 in winter
```
