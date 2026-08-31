# ERA5 vs. SASSIE surface fluxes

Compares ERA5 surface radiative and turbulent fluxes against shipboard measurements from the
SASSIE fall 2022 field campaign in the Beaufort Sea, using the ERA5 grid cell closest to each
recorded ship position.

Reference for the campaign and its Figure 8:

> Drushka, K., et al. (2024). Salinity and Stratification at the Sea Ice Edge (SASSIE): an
> oceanographic field campaign in the Beaufort Sea. *Earth Syst. Sci. Data*, 16, 4209-4242.
> <https://doi.org/10.5194/essd-16-4209-2024>

## Files

| File | Purpose |
| --- | --- |
| `download_era5_sassie_gap.py` | **Primary ERA5 source.** Downloads only the three CDS tiles the `barrow` archive does not already cover. |
| `era5_sassie_mosaic.py` | Stitches `barrow` and the three gap tiles into one dataset, and proves the tiling is exact. |
| `fetch_era5_sassie_box.py` | Fallback: single-box cache read straight from the NSF NCAR S3 mirror via `../era5_aws.py`. No CDS account needed, but slow (~1 h) and fewer variables. |
| `compare_era5_sassie_fluxes.ipynb` | The comparison: collocation, four flux figures with residual panels, summary statistics, and a net-longwave supplement. |
| `data/` | Downloaded ERA5 (`sassie_gap_*/`) and the S3 fallback cache. |
| `figures/` | PNG output from the notebook. |

## Why the ERA5 has to be extended

The `barrow` archive under `../data/` covers 70-80 N, 165-150 W. The SASSIE ship track runs
69.20-73.52 N, 165.96-144.89 W, so **42 % of the shipboard records fall outside it** - mostly the
eastern half of the campaign.

Enclosing the whole track on the 0.25 deg grid with one cell of margin takes 69.00-73.75 N,
166.25-144.50 W: 20 x 88 = 1760 cells. `barrow` already supplies 976 of them. Three tiles supply
the rest, abutting the Barrow strip without overlapping it or each other:

|  | 166.25-165.25 W | 165.00-150.00 W | 149.75-144.50 W |
| --- | --- | --- | --- |
| **73.75-70.00 N** | `sassie_gap_west` (80) | `barrow` (976) | `sassie_gap_east` (352) |
| **69.75-69.00 N** | `sassie_gap_south` -- 352 cells across the full width | | |

`976 + 80 + 352 + 352 = 1760`, exactly, with nothing counted twice.
`era5_sassie_mosaic.check_tiling()` asserts this, so a later edit to any region box cannot
silently open a seam.

### Downloading the gap

```bash
python download_era5_sassie_gap.py --dry-run   # plan only, no CDS contact
python download_era5_sassie_gap.py             # do it
```

Three tiles x half-monthly chunks over 2022-09-08 to 2022-10-02 = **9 CDS requests**, ~20 MB. The
default one-file-per-day chunking would have submitted 75. All of it delegates to
`../download_era5_seb.py`, so the tiles get the same 35-variable `recommended` set, the same
`avg_*` name normalisation, and the same resume-in-date-space behaviour as the rest of the
archive - re-running skips whatever is already on disk.

The three boxes are registered as `sassie_gap_south`, `sassie_gap_west` and `sassie_gap_east` in
`../era5_seb_variables.py`, so they are also reachable from the plain downloader with
`--region sassie_gap_east`.

### Using the mosaic

```python
from era5_sassie_mosaic import load_sassie_box
era5 = load_sassie_box("2022-09-08", "2022-10-02")
```

The four rectangles form a U rather than a hypercube, so `combine_by_coords` cannot infer the
layout. The loader uses an outer-join `xr.merge` with `compat="no_conflicts"` instead, then checks
the merged grid is the expected 20 x 88 and that no cell came back all-NaN.

### Fallback without a CDS account

```bash
python fetch_era5_sassie_box.py
```

Reads eight variables over a single 68.5-74.0 N / 167.0-144.0 W box from the NSF NCAR S3 mirror.
Roughly an hour, dominated by HDF5 header reads over HTTP (see the cost model in
`../era5_aws.py`). The notebook uses it automatically if the gap tiles are missing.

## SASSIE input

`SASSIE_Fall_2022_shipboard_MET.nc` from PO.DAAC (`SASSIE_L2_SHIPBOARD_METEOROLOGY_V2`,
DOI [10.5067/SASSIE-MET2](https://doi.org/10.5067/SASSIE-MET2)), 20 min averages along the R/V
*Woldstad* track, currently at
`/Users/andrewbuggee/Documents/Scripps - UCSD/Ocean Visions/SASSIE/`. The path is set at the top
of the notebook.

## What is compared

| Flux | SASSIE | ERA5 | Convention in the figure |
| --- | --- | --- | --- |
| Shortwave | `shortwave_radiation` (CMP21 pyranometer) | `msdwswrf` | positive downward |
| Longwave | `downwelling_longwave` (CGR4 pyrgeometer) | `msdwlwrf` | positive downward |
| Sensible heat | `bulk_sensible_heat_flux` (COARE 3.5) | `-msshf` | positive out of the ocean |
| Latent heat | `bulk_latent_heat_flux` (COARE 3.5) | `-mslhf` | positive out of the ocean |

Radiation is compared as the **downwelling** component rather than the net shown in the paper's
Figure 8c. SASSIE had no upward-looking radiometers, so its net fluxes need an assumed ocean albedo
and emissivity; the downwelling fluxes are direct measurements. A net-longwave comparison in the
Figure 8c convention is included at the end of the notebook with its assumptions spelled out.

ERA5 hourly `mean_*` fluxes are averages over the hour **ending** at their valid time, so each
20 min shipboard record is assigned to the hour that contains it and the records sharing an hour
are averaged together.

## Two defects in the PO.DAAC MET file, found here

1. `bulk_sensible_heat_flux` and `bulk_latent_heat_flux` are labelled *"flux into ocean"* but the
   stored values are positive **out of** the ocean. The notebook verifies this from the sign of
   `SST - T_air` and it agrees with Figure 8b of the paper.
2. `bulk_upwelling_IR`, documented as *"estimated net longwave radiation"*, holds values of
   0.08-16.5 and correlates with `bulk_U10` at r = 1.000. It is a copy of the 10 m wind speed, not
   a longwave flux. It is not used.
