# Was Barrow assimilated into ERA5?

A definitive check, using ECMWF's ERA5 observation feedback archive (ODB) in
MARS. This answers, per station, per variable, per report: was the observation
**actively assimilated**, **passively monitored**, **rejected**, or
**blacklisted**.

---

## Read this before running anything

Two parts of the question are already settled, and knowing which is which will
save you from over-interpreting the output.

**1. Nothing from ARM/NSA or NOAA-BRW radiation instrumentation is in ERA5.**
ERA5 assimilates no surface broadband irradiance, no cloud radar reflectivity,
and no microwave-radiometer LWP. There is no observation operator for
downwelling longwave in the IFS. This is not something the ODB query can
overturn — it is a property of the observing-system configuration. The useful
consequence for a surface-energy-budget study is the good one: **the ARM Barrow
radiation record is independent of ERA5**, so comparing ERA5 fluxes against it
is not circular. (High confidence.)

**2. The Barrow WMO station is assimilated, but not in the way you might
assume.** Surface pressure from SYNOP and the radiosonde profile (T, u, v, q)
enter the 4D-Var atmospheric analysis. **2 m temperature and 2 m relative
humidity do not.** ERA5 routes screen-level observations into a separate 2-D
optimal-interpolation screen-level analysis and the soil-moisture simplified
EKF — see Hersbach et al. (2020), *QJRMS* 146:1999–2049, §3. So a non-active
flag on `varno=39` (t2m) or `varno=58` (rh2m) in this feedback means "not used
by the atmospheric analysis", **not** "unused by ERA5". The analysis script
labels those rows explicitly rather than letting the table imply otherwise.
(High confidence on the mechanism.)

What the ODB query actually adds is the specifics for *your* period: which
station identifiers exist, whether reports were present continuously or
episodically, how many were QC-rejected, and how large the model's departures
from them were.

---

## What you need

Full MARS access, which a CDS account does **not** provide. Either:

- **an ECMWF platform login** (Atos HPC / ecgate), where the `mars` client is
  installed — much the easier route, no credentials to manage; or
- **an ECMWF web-API account** with the ERA5 feedback entitlement, plus
  `~/.ecmwfapirc`.

A permissions error (as opposed to an empty result) means an entitlement
problem, not a query problem — that goes to ECMWF user support.

```bash
pip install -r requirements-odb.txt
```

---

## Step 0 — verify the MARS keys (do not skip)

`class=ea`, `type=ofb`, `obsgroup=conv` and `expver=1` are certain. **`stream`
and `time` are not**, and I have not verified the defaults in
`fetch_era5_obs_feedback.py` against a live archive. ERA5 runs 12-hour 4D-Var
windows (09–21 and 21–09 UTC), so `time` here keys the *window*, not the
observation hour.

This matters more than it sounds: a wrong window key returns "no data found",
which is indistinguishable from a genuine negative result. A `list` request is
free and settles it:

```bash
python fetch_era5_obs_feedback.py --probe --start 2015-01-15
```

Run the emitted `probe_list.req` with `mars`, read off the archived
`stream`/`time`, and pass them to the retrieval below.

## Step 1 — retrieve

On an ECMWF platform:

```bash
python fetch_era5_obs_feedback.py --start 2014-01-01 --end 2016-12-31 --emit-only
```

then `for f in data/obs_feedback/*.req; do mars "$f"; done`.

Remotely through the web API, drop `--emit-only`. Requests are chunked by month
and skip targets that already exist, so an interrupted multi-year pull resumes.

## Step 2 — summarise

```bash
python read_era5_obs_feedback.py --in-dir data/obs_feedback
```

If you converted to CSV on the ECMWF side instead
(`odb sql 'select *' -i f.odb -o f.csv --no_alignment`), add `--csv`.

Outputs `station_census.csv` and `usage_summary.csv`, plus a printed verdict.

## Step 3 — test the plumbing without ECMWF access

```bash
python make_synthetic_feedback.py
python read_era5_obs_feedback.py --csv --in-dir data/obs_feedback_synthetic
```

Fabricated numbers in the real column layout. It proves the analysis path runs;
it says nothing about Barrow.

---

## On 70026 vs 70027

70026 is the WMO index for BARROW/W. POST-W. ROGERS AP (PABR). **I have not
verified what 70027 is**, so the code does not rely on either guess: the
primary selector is a lat/lon box around 71.28 N, 156.79 W, and
`station_census.csv` reports every `statid` the archive actually uses there,
with its position and report count. Read the identifiers off the archive.

This is also a robustness point, not just convenience. ODB `statid` is a
fixed-width character field that may arrive blank-padded or right-justified;
an equality test against the wrong padding returns zero rows, which again looks
exactly like "never assimilated". The box cannot fail that way.

## Interpreting the output

| column | meaning |
|---|---|
| `n_active` | data used in the minimisation. **This is assimilation.** |
| `n_passive` | ingested, departures computed, given zero weight — monitoring only |
| `n_rejected` | failed QC (background departure, duplicate, gross error) |
| `n_blacklisted` | excluded a priori by station/variable/period |
| `bias_fg_depar_active` | mean (observation − first guess), o−b |
| `rms_an_depar_active` | rms (observation − analysis), o−a |
| `depar_shrinkage` | rms(o−a) / rms(o−b) |

`depar_shrinkage < 1` is the quantitative signature of assimilation: the
analysis moved toward the observation. A station flagged active whose shrinkage
sits at ~1.0 was nominally used but carried almost no weight. Corroborating the
flag with the departures is worth doing — the flag alone tells you the intent,
the shrinkage tells you the effect.

## Reliability of the code tables

`era5_odb_config.py` transcribes ODB code tables (`varno`, `obstype`,
`reportype`, status bit offsets) from ECMWF documentation. These are the most
likely thing here to be stale. Every lookup degrades gracefully — an
unrecognised code is reported as `varno_247` rather than dropped or
mislabelled, and the analysis prints a census of unnamed codes. Counts are
never affected, only labels.

The status bit offsets are used **only** as a fallback. The retrieval asks ODB
to expand the named bitfield members in SQL (`datum_status.active`, …), so the
flags normally arrive as plain 0/1 columns with no dependence on bit positions.
The analysis prints which path it took (`usage flags from: …`). To check the
offsets against your own file: `odb header file.odb`.

## Files

| file | role |
|---|---|
| `era5_odb_config.py` | station geometry, ODB code tables, box definition |
| `fetch_era5_obs_feedback.py` | builds and submits/emits the MARS requests |
| `read_era5_obs_feedback.py` | reads ODB, decodes flags, writes the summary |
| `make_synthetic_feedback.py` | offline test fixture |
| `requirements-odb.txt` | extra dependencies |
