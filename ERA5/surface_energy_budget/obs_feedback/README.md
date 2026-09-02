# Was Barrow assimilated into ERA5?

A definitive check, using ECMWF's ERA5 observation feedback archive (ODB) in
MARS. This answers, per station, per variable, per report: was the observation
**actively assimilated**, **passively monitored**, **rejected**, or
**blacklisted**.

---

## Read this before running anything

Two parts of the question are already settled, and knowing which is which will
save you from over-interpreting the output.

**1. No ARM/NSA or NOAA-BRW *cloud or radiation* instrumentation is in ERA5.**
ERA5 assimilates no surface broadband irradiance, no cloud radar reflectivity,
no ceilometer cloud base and no microwave-radiometer LWP. See "Evidence, and
its limits" below for the basis and for what this does NOT cover -- notably ARM
radiosondes, which ARE on the GTS under WMO 70026 and, since February 2019,
ARE the Barrow sounding that ERA5 ingests. The useful
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

## Evidence, and its limits

The claims above are argued from ECMWF's own enumeration of the ERA5 observing
system, not from proof. Recording the basis here so it can be checked.

**Hersbach et al. (2020), *QJRMS* 146:1999-2049, section 5.1** lists what ERA5
assimilates:

- in-situ: 10 m wind over sea; 2 m humidity over land; pressure over land and
  sea; upper-air wind/temperature/humidity from radiosonde, PILOT, dropsonde
  and aircraft
- satellite: AMVs; temperature/humidity/ozone sounders; microwave imagers
  (SSMI, SSMIS, TMI, AMSR-E, AMSR-2, GMI) in all-sky; hyperspectral infrared
  (IASI, AIRS, CrIS); GNSS-RO bending angles; scatterometer wind and soil
  moisture; altimeter wave height; level-2 ozone
- ground-based radar-gauge composite rain rate, from 2009
- LDAS: SYNOP screen-level T/RH feeding the soil-moisture analysis, separate
  from 4D-Var

Absent from that list: ground-based cloud radar, ground-based microwave
radiometer LWP, surface radiation, ceilometer cloud base, ground-based GNSS
zenith total delay.

**What that supports (high confidence).** No LWP retrieval, cloud-fraction
product, surface irradiance measurement or cloud-radar reflectivity is
assimilated anywhere in ERA5. LWP is not an assimilated quantity; cloud liquid
is touched only indirectly, via all-sky microwave radiances.

**Moderate confidence, and it matters at this site.** Those all-sky microwave
radiances are largely an ocean-surface product -- Hersbach describes extending
their use over land and sea ice as an advance, implying a more restricted
baseline. Over Utqiagvik, land and sea ice, ERA5 cloud liquid is close to pure
model output.

**RESOLVED, AND IT REVERSES AN EARLIER CLAIM IN THIS README: ARM sondes at
Utqiagvik ARE on the GTS, under WMO 70026, and are very probably assimilated.**

The documented chain:

1. NWS Service Change Notice 18-89 (issued 2019-03-22): effective
   **12 February 2019**, the Barrow radiosonde program, "World Meteorological
   Organization (WMO) # 70026, Station ID PABR", moved from manual launches to
   an automated launcher "4.6 miles northeast of the legacy release point".
   The notice gives the new release point as **71.32267 N, 156.61784 W,
   8.4 m**.
2. Those coordinates are the DOE ARM North Slope of Alaska C1 site (ARM lists
   NSA C1 at 71.3 N, 156.6 W, 8 m).
3. ARM operates the autosonde there -- a Vaisala AS15, replaced by an AS41 in
   September 2022 (ARM ASR STM presentation, 2025-04-04, slide 10).
4. The same presentation, slide 11, lists ARM's current GTS products as
   including **"NSA C1 (WMO 70026)"**.
5. ERA5 assimilates radiosondes (Hersbach et al. 2020, section 5.1).

So from February 2019 onward, the Barrow 70026 sounding that ERA5 ingests is
launched and processed by ARM. Any statement that ARM data never entered ERA5
is wrong for the post-2019 sonde record.

**Unchanged by this.** The reversal covers soundings only. ARM's radiation,
cloud radar and MWR instrumentation remains outside ERA5's observing system,
so the surface-energy-budget comparison this package was written to support is
still non-circular.

**Consequence for any query on 70026.** The identifier spans two physically
different launch points across 2019-02-12. ``position_history()`` in
``read_era5_obs_feedback.py`` detects the move and warns; split statistics on
that date rather than averaging through it.

**Still not established.** ERA5 assimilates ground-based radar-gauge rain-rate
composites from 2009, so "no ground-based remote sensing" is false as a
category statement. It has no North Slope coverage and concerns precipitation
rather than cloud, so it does not affect the conclusion here -- but the
category is not empty.

**The limit.** This is an argument from documented absence: strong, but it
cannot exclude a stream that entered under a category label not recognised
here. The empirical test is the station census this package produces -- it
lists every statid reporting near 71.3 N, so an ARM identifier would show up
under some variable if one exists. That is the strongest reason to pursue MARS
access.

Corroboration worth noting: Yuan et al. (2025), *GRL*, report ERA5 zenith total
delay discontinuities at 09:00 and 21:00 UTC attributed to the assimilation
window transition -- independent support for the 12-hour window boundaries
behind the ``--mars-time 0900/2100`` default, though not for the MARS key
encoding itself.

References
- Hersbach, H. et al. (2020), The ERA5 global reanalysis, QJRMS 146:1999-2049.
  https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803
- Yuan, P. et al. (2025), A Global Assessment of Diurnal Discontinuities in ERA5
  Tropospheric Zenith Total Delays Using 10 Years of GNSS Data, GRL.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL113140
- NWS Service Change Notice 18-89 (updated 2019-03-22), Transition of Manual
  Radiosonde Observations to Automated Radiosonde Observations at Barrow, AK.
  https://www.weather.gov/media/notification/pdfs/scn18-89upper_air_barrow_aaa.pdf
- ARM, Advancements in ARM's Instrumentation and Measurement Strategies,
  ASR STM, 2025-04-04, slides 10-11.
  https://www.asr.arm.gov/meetings/stm/presentations/2025/1853.pdf

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
python check_setup.py
```

`check_setup.py` tests every prerequisite for both routes and prints the
specific next action for each failure. Run it first; it distinguishes the four
different missing pieces that otherwise all present as the same error.

```bash
pip install -r requirements-odb.txt
```

**Access is the hard gate, not the code.** Full MARS is provisioned through
ECMWF Member and Co-operating States. The United States is neither, so a
CU Boulder affiliation does not itself grant access, and self-registration at
ecmwf.int gives an account without the MARS entitlement. I do not know the
current route for a US-based academic to obtain ERA5 feedback access — ask
ECMWF user support (https://support.ecmwf.int) directly, naming the ERA5
observation feedback (ODB) archive specifically. A collaborator with an Atos
login is usually the faster path.

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
| `check_setup.py` | verifies prerequisites, reports which route is closest |
| `era5_odb_config.py` | station geometry, ODB code tables, box definition |
| `fetch_era5_obs_feedback.py` | builds and submits/emits the MARS requests |
| `read_era5_obs_feedback.py` | reads ODB, decodes flags, writes the summary |
| `make_synthetic_feedback.py` | offline test fixture |
| `requirements-odb.txt` | extra dependencies |
