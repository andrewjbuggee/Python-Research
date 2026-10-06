# EPCAPE data tools

Scripts that download EPCAPE data from the DOE ARM archive and merge them into
single netCDF files for analysis in Python or MATLAB. The same code runs on a
laptop, the UCSD Research Cluster, and ARM's JupyterHub/Cumulus. Everything
machine-specific lives in `config.yaml`.

## One-time setup

1. **Python environment** (from this folder):

   ```bash
   conda env create -f environment.yml
   conda activate epcape
   ```

   Without conda: `pip install -r requirements.txt` (Python 3.9 or newer).

2. **ARM access token.** Log in at <https://adc.arm.gov/armlive/> with your
   ARM account; the page shows your access token. The first download asks for
   your username and token and offers to save them to `~/.arm_credentials`
   (readable only by you, outside this folder). Alternatively, set the
   `ARM_USERNAME` and `ARM_TOKEN` environment variables.

## Test run: cloud-base height for the whole campaign

```bash
python download_data/download_arm.py cbh_ceil_M1      # ~365 daily files, cloud-base variables only
python download_data/combine_product.py cbh_ceil_M1   # one file: data/processed/cbh_ceil_M1_20230215_20240214.nc
```

`download_data/download_arm.py` asks ARM's server to extract only the product's variables
from each daily file (`first_cbh`, `second_cbh`, `third_cbh`,
`detection_status`, `status_flag`, `vertical_visibility`, their `qc_`
companions, time and location). The 16 s × 770-gate backscatter profiles, which
make up most of each complete file, are never transferred. The script prints
the actual sizes when it finishes.

Rerunning either command is safe. Downloads resume, and files already on disk
are skipped. `download_data/combine_product.py` ends with a coverage and statistics summary
(days without data, percent of samples with a cloud base, median heights) as a
sanity check.

**Heights are above the instrument, not sea level.** ARM's ceilometer handbook
says cloud heights "are measured above the optics assembly and are not adjusted
for altitude". Add the `alt` variable for height above sea level. This matters
at Mt. Soledad (S2, ~250 m MSL).

## Reading the output in MATLAB

```matlab
f = 'data/processed/cbh_ceil_M1_20230215_20240214.nc';
t      = datetime(ncread(f,'time'), 'ConvertFrom','posixtime', 'TimeZone','UTC');
cbh1   = ncread(f,'first_cbh');         % m above the instrument; NaN = no cloud base
status = ncread(f,'detection_status');  % int16 codes 0-5; -9999 = missing
ncdisp(f, 'detection_status')           % code meanings, units, QC bit descriptions
```

Output conventions: `time` is seconds since 1970-01-01 UTC. Float variables use
NaN for missing values. Integer flags keep their type, with -9999 for missing.
Variable attributes and the ARM citation are copied into the file.

## Common commands

```bash
python download_data/download_arm.py --list                          # products, active machine, data folder
python download_data/download_arm.py cbh_ceil_S2                     # same variables at Mt. Soledad
python download_data/download_arm.py cbh_ceil_M1 --dry-run           # what would be downloaded
python download_data/download_arm.py cbh_ceil_M1 --start 2023-07-01 --end 2023-07-31
python download_data/download_arm.py cbh_ceil_M1 --full              # complete files for that datastream
python download_data/download_arm.py --datastream epcdlfptS2.b1 --start 2023-07-01 --end 2023-07-01   # any datastream, complete files
python download_data/combine_product.py cbh_ceil_M1 --start 2023-06-01 --end 2023-08-31
```

To add a product, add an entry under `products:` in `config.yaml` with a
datastream name (from ARM Data Discovery) and the variables you need. A
misspelled variable stops the run before any bulk download and lists the
variables that exist.

## Running on other machines

Clone or copy this folder to the machine. Then set which `machines:` entry of
`config.yaml` applies, for example in `~/.bashrc`:

```bash
export EPCAPE_MACHINE=ucsd            # or arm_jupyterhub, arm_cumulus; default is local
export EPCAPE_DATA_ROOT=/some/path    # optional: overrides data_root for one machine
```

- **UCSD Research Cluster:** data go to `~/epcape_data` (100 GB default
  quota). Subsets like the cloud-base heights are small, so this is enough for
  processed products.
- **ARM JupyterHub / Cumulus:** if the ARM archive is mounted at the configured
  `arm_archive` path, the scripts read files where they are and download
  nothing. That path (`/data/archive`) is a guess to confirm on first login;
  if it doesn't exist, the scripts download as usual. On Cumulus, replace
  `PROJECT_ID` in `data_root`. Cumulus scratch space is purged periodically, so
  keep only reproducible outputs there.

## Data layout

```
data/
  arm/<datastream>/          complete ARM files, archive names
  arm_subset/<product>/      server-extracted variable subsets, plus manifest.json
  processed/                 combined netCDF files
  processed/<UCSD object>/   Russell-group files from the UC San Diego Library
                             (listed under ucsd_library: in config.yaml), plus ucsd_manifest.json
```

`manifest.json` records each file's size and download time, the variable list,
every run, and ARM's citation for the datastream. ARM asks that publications
cite each datastream's DOI.

## Behavior worth knowing

- If ARM's server can't extract the variables from a particular file (for
  example, a QC variable added partway through the campaign), that one file is
  downloaded whole and subset locally. The summary counts these, and
  `manifest.json` records the reason.
- If extraction fails for the very first file, the run stops and suggests
  `--full`. This prevents a broken service from silently turning into complete
  downloads of every file.
- A busy server (HTTP 429/502/503/504 after all retries) is never answered
  with a complete-file download; the file is listed as failed and the next run
  retries it. On 2026-10-06 ARM Live's subset service answered 502/503 to four
  parallel requests and worked with one (`--workers 1`).
- Files that ARM Live can't serve are listed at the end. Those have to be
  ordered through ARM Data Discovery.

## Analysis code layout

The `EPCAPE/` folder is itself the Python package. Every module is imported
by its full path, e.g. `from EPCAPE.analysis_tools import qc`, with the folder
*containing* `EPCAPE/` (Python-Research) on `sys.path`. The scripts, the tests
and the notebooks add it themselves. The top level holds only configuration
and documentation; code lives in sub-folders so it is easy to find later:

```
EPCAPE/                       the package and the repository folder
  config.yaml, README.md        machine paths and products; this file
  environment.yml, requirements.txt
  data/                         downloaded + processed data (not in git)
download_data/                getting ARM data
  download_arm.py               command line: python download_data/download_arm.py <product>
  combine_product.py            command line: python download_data/combine_product.py <product>
  config.py                     reads config.yaml (campaign dates, machine paths, products)
  sync.py, armlive.py, arm_files.py, credentials.py   resumable ARM Live downloads
  combine.py                    daily files -> data/processed/<product>_<start>_<end>.nc
analysis_tools/               shared analysis code
  products.py                   load_product(): one Dataset per product (builds the combined file if needed)
  qc.py                         decode ARM qc_ bit fields into "bad sample" masks
  filters.py                    named selection criteria and the filter "funnel" table
  units.py, stats.py            unit conversion from file attributes; paired-comparison statistics
  derived.py                    save_derived(): MATLAB-ready netCDF with settings + git commit in attributes
plotting/style.py             fixed instrument colours and figure style (from EPCAPE.plotting import COLORS)
instruments/<instrument>/     products that come from ONE instrument
  mfrsr/mfrsrcldod.py           MFRSRCLDOD tau and r_e (load, standardize, selection criteria)
  sunphotometer/sphotcod.py     SPHOTCOD tau, r_e, LWP
  mwr/mwrlos.py                 MWRLOS LWP and PWV
  */plots.py                    one-day quicklooks
variables/<variable>/         products that estimate ONE variable from SEVERAL instruments (none yet)
comparisons/<topic>/          analyses that set several products side by side
  cloud_optical_properties/     MFRSR vs sunphotometer vs MWR: tau, r_e, LWP
    compare_cloud_optical_properties.ipynb   <- start here
  seasonal_averages/            re-derives the seasonal-averages table rows attributed to Kavin
    download_data.py              ARM products + UCSD Library files the check needs
    check_seasonal_averages.ipynb <- start here (QC = 0 only; outputs in processed/derived/)
    seasonal.py, quantities.py, sources.py   seasons + table parsing; one function per quantity; data access
```

Rules the code follows:

- Notebooks never read daily ARM files directly; they call
  `load_product`, so the same notebook runs on a laptop (downloaded files),
  on ARM JupyterHub/Cumulus (`EPCAPE_MACHINE=arm_jupyterhub`, archive read in
  place), or on the UCSD cluster.
- Each instrument module documents its retrieval assumptions with
  citations, and returns its selection criteria as named masks. The
  notebook prints how many samples each criterion removes.
- Derived files go to `<data folder>/processed/derived/` with every setting
  recorded in their attributes. They are not in git, so back them up;
  everything else can be re-downloaded.

The three cloud-property products in `config.yaml` are `cod_mfrsr_M1`,
`cod_sphot_M1` and `lwp_mwr_M1`. Their ancillary fields are listed under
`optional_variables` (kept when present), because their exact names could not
be checked against an EPCAPE file when the products were added. Variable
names are matched case-insensitively.

What the real EPCAPE files showed (checked 2026-10-06), and how the code handles it:

- **SPHOTCOD** retrievals have a `gain` dimension (0 = aureole gain A, 1 = sky
  gain K, 2 = mean of A and K). `sphotcod.standardize` uses the mean by default.
  Its `retrieval_flag` is a bit-packed QC field with 7 tests, all "Bad". Only
  about 14% of retrievals pass every test, because the spectral-signature
  tests 2-5 fail often at the pier. `sphotcod.Criteria(ignore_flag_bits=...)`
  relaxes them.
- **SPHOTCOD** assumes a MODIS albedo stored once per daily file. The combine
  step now repeats such per-file variables along time instead of keeping only
  the first file's values.
- **MFRSRCLDOD** `ir_temp` is empty for the whole campaign. A criterion whose
  input is missing or all-NaN is reported as *skipped* rather than rejecting
  every sample. The LWP source is `source_lwp`, mostly MWRRET.
- **MWRLOS** `wet_window` (window heater on) is 1 for 39% of the campaign,
  mostly at night in the marine layer. Following TR-016, it is not used by
  default; Tb > 100 K and LWP > 1000 g m-2 screen rain and window water.
  `mwrlos.Criteria(reject_wet_window=True)` gives the strict screen.

**ARM orders delivered as symbolic links.** An order staged for ARM's own
computing contains symlinks into `/data/archive/...`. These resolve on ARM
JupyterHub/Cumulus and nowhere else. On a laptop such a folder looks full in
Finder but holds no data (`ls -l` shows `->`). Use `download_data/download_arm.py` (or a
standard Data Discovery download) to get real files.

To test the analysis without real data:

```bash
python tests/synthetic_cloud_vaps.py /tmp/epcape_synth
EPCAPE_DATA_ROOT=/tmp/epcape_synth jupyter lab comparisons/cloud_optical_properties/
```

## Tests

```bash
python -m pytest tests -q
```

The tests run offline against a mock ARM Live server with synthetic files that
follow ARM's `ceil.b1` data object design. They cover subsetting, resume,
retries on busy servers and dropped connections, wrong tokens, unavailable
files, a mid-campaign variable change, a mounted-archive mode, and the MATLAB
view of the output. `tests/test_cloud_products.py` covers the analysis layer
(QC decoding, units, window statistics, an end-to-end run on synthetic files).
`tests/test_seasonal_averages.py` covers the seasonal-averages check (reading
the sheet's cells, season windows, QC = 0, the Romps LCL, rain events, GCVI
residuals).
