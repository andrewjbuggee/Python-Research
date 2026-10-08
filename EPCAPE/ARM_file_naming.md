# ARM file names: a quick reference

Source for the naming pattern and the data-level definitions:
[ARM, "Formatting and file naming protocols"](https://www.arm.gov/guidance/datause/formatting-and-file-naming-protocols)
(quoted 2026-10-07). The EPCAPE examples and header attributes were read from files in this project's `data/arm/`.

## Anatomy of a file name

ARM's pattern is `(sss)(nn)(inst)(qqq)(Fn).(dl).YYYYMMDD.hhmmss.(nc|cdf)`.

| Part | Meaning | Example: `epcmwrret1liljclouM1.c2.20230215.000000.nc` |
|---|---|---|
| `sss` | site | `epc` = EPCAPE (La Jolla, CA) |
| `nn` | integration period (optional) | none here; see `epc30smplcmask1zwangM1.c1` (`30s`) |
| `inst` | instrument or value-added product (VAP) | `mwrret` = MWR retrievals |
| `qqq` | optional qualifier | `1liljclou` (see note 1) |
| `Fn` | facility | `M1` = Scripps Pier |
| `dl` | **data level** (table below) | `c2` |
| `YYYYMMDD.hhmmss` | date and time of the first record in the file, UTC | 2023-02-15 00:00:00 UTC |
| extension | both are netCDF | `.nc` or `.cdf` |

**Note 1:** for VAPs the qualifier usually reads *version number + developer or algorithm*. Examples: `pblhtsonde1mcfarl` (McFarlane), `arsclkazr1kollias` (Kollias), `mplcmask1zwang` (Z. Wang), `radflux1long` (Long), `mwrret1liljclou` vs `mwrret2turn` (two generations of the MWR retrieval). This is a pattern seen in the names, not a rule I found written down by ARM.

**Note 2:** the integration period does not always come first. `skyrad60s` (1-min SKYRAD) puts `60s` after the instrument name.

## Data levels

| Level | ARM definition (quoted) | In plain terms | EPCAPE examples used here |
|---|---|---|---|
| `00` | "raw data – primary raw data stream collected directly from instrument" | instrument's own files | — |
| `01` | "raw data – redundant data stream or sneakernet data" | backup copy of raw data | — |
| `a0` | "converted to netCDF" | raw values, netCDF format. ARM Live refused `epcceilpblhtM1.a0` with HTTP 403 | `ceilpblht.a0` (not downloadable) |
| `a1` | "calibration factors applied and converted to geophysical units" | physical units, no QC yet | — |
| `a2`–`a9` | "further processing on a1 level data that does not merit b1 classification" | | — |
| **`b1`** | **"QC checks applied to measurements"** | **instrument data in physical units with `qc_` fields. The usual starting point.** | `ceil.b1`, `mwrlos.b1`, `ld.b1`, `sondewnpn.b1`, `skyrad60s.b1` |
| `b2`–`b9` | "further processing on b1 level data that does not merit c1 classification" | | — |
| `c0` | "intermediate value-added data product; this data level is always used as input" | an intermediate VAP stage; use the c1 instead | `arsclkazr1kollias.c0` (we use its `c1`) |
| **`c1`** | **"derived or calculated value-added data product (VAP)"** | **a retrieval or derived product built from one or more b1 streams** | `arsclkazr1kollias.c1`, `pblhtsonde1mcfarl.c1`, `pblhtthermo.c1`, `sondeparam.c1`, `ldquants.c1`, `vdisquants.c1` |
| `c2`–`c9` | "further processing applied to a 'c1' level data stream" | a further-processed VAP stage | `radflux1long.c2`, `qcrad1long.c2`, `mwrret1liljclou.c2` (note 3) |
| `s1` | "summary file consisting of a subset of the parent .c1 file" | a smaller extract of a c1 product | `pblhtsonde1mcfarl.s1` |
| `s2` | "summary file consisting of a further – processed s1 data" | | — |

**Note 3:** `epcmwrret1liljclouM1.c2` lists only b1 inputs, and its `command_line` contains `--data-level 2`. So at EPCAPE its c2 looks like the same code run in a level-2 configuration rather than a reprocessing of a c1 file. ARM Live has no MWRRET1 c1 files for EPCAPE. This is an inference from the header, not from ARM documentation.

## The level is not a version

The data level says how far along the processing chain a file is. Versions are recorded in each file's global attributes. If ARM reprocesses a dataset, the file names stay the same and these attributes change.

| Attribute | What it records | Example (`epcradflux1longM1.c2`) |
|---|---|---|
| `data_level` | the level, again | `c2` |
| `process_version` | the software that made the file | `radflux1long-3.16.0` |
| `dod_version` | the data object design, i.e. the file-format version | `radflux1long-c2-1.7` |
| `input_datastreams` | which streams (and their versions and dates) were used | `epcqcrad1longM1.c2 : 6.8 : 20221209.000000-20240214.000000` |
| `history` | who made it, where, and when | `... at 2024-08-02 16:36:41, using radflux1long-3.16.0` |
| `doi` | the citation DOI for the datastream | `10.5439/1395157` |

To read them: `ncdisp(f)` in MATLAB, or `xr.open_dataset(f).attrs` in Python.

## EPCAPE facilities

| Code | Location |
|---|---|
| `M1` | Scripps Pier (AMF1 main site) |
| `S2` | Mt. Soledad |

ARM Live has no `S1` streams for EPCAPE, so the seasonal-averages sheet's "LDS1" can only mean the S2 disdrometer.
