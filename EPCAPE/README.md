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
python download_arm.py cbh_ceil_M1      # ~365 daily files, cloud-base variables only
python combine_product.py cbh_ceil_M1   # one file: data/processed/cbh_ceil_M1_20230215_20240214.nc
```

`download_arm.py` asks ARM's server to extract only the product's variables
from each daily file (`first_cbh`, `second_cbh`, `third_cbh`,
`detection_status`, `status_flag`, `vertical_visibility`, their `qc_`
companions, time and location). The 16 s × 770-gate backscatter profiles, which
make up most of each complete file, are never transferred. The script prints
the actual sizes when it finishes.

Rerunning either command is safe. Downloads resume, and files already on disk
are skipped. `combine_product.py` ends with a coverage and statistics summary
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
python download_arm.py --list                          # products, active machine, data folder
python download_arm.py cbh_ceil_S2                     # same variables at Mt. Soledad
python download_arm.py cbh_ceil_M1 --dry-run           # what would be downloaded
python download_arm.py cbh_ceil_M1 --start 2023-07-01 --end 2023-07-31
python download_arm.py cbh_ceil_M1 --full              # complete files for that datastream
python download_arm.py --datastream epcdlfptS2.b1 --start 2023-07-01 --end 2023-07-01   # any datastream, complete files
python combine_product.py cbh_ceil_M1 --start 2023-06-01 --end 2023-08-31
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
- Files that ARM Live can't serve are listed at the end. Those have to be
  ordered through ARM Data Discovery.

## Tests

```bash
python -m pytest tests -q
```

The tests run offline against a mock ARM Live server with synthetic files that
follow ARM's `ceil.b1` data object design. They cover subsetting, resume,
retries on busy servers and dropped connections, wrong tokens, unavailable
files, a mid-campaign variable change, a mounted-archive mode, and the MATLAB
view of the output.
