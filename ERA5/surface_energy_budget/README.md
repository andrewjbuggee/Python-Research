# ERA5 downloader — Arctic surface energy budget

Hourly ERA5 single-level data for a surface energy budget (SEB) analysis following
Equation (1) of:

> Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy balance
> responses to radiative forcing in the central Arctic from MOSAiC and models.
> *JGR Atmospheres*, 130, e2024JD042578. <https://doi.org/10.1029/2024JD042578>

## Files

| File | Purpose |
| --- | --- |
| `download_era5_seb.py` | CLI downloader. Chunking, resume, retries, run manifest. |
| `era5_seb_variables.py` | Variable registry (units + Eq. 1 role) and Arctic regions. |
| `seb_terms.py` | Post-processing: maps ERA5 fields onto Eq. (1), fixing the sign flip. |

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

- **Resume.** Existing output files are skipped unless `--overwrite` is passed.
  The raw CDS payload lands on a `.raw.part` scratch path and the merged netCDF is
  moved into place only once complete, so an interrupted run never leaves a partial
  file that a later resume would treat as finished.
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
