# Mie Table Combination Scripts

These scripts combine NetCDF Mie scattering tables across wavelengths for each alpha value.

## Overview

**Input Structure:**
- Base directory: `/projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/`
- 221 subdirectories: `wavelength_300nm/` through `wavelength_2500nm/` (every 10 nm)
- Each subdirectory contains 48 files: `wc_gamma_XXX_0_mie.cdf` (where XXX is alpha value)

**Alpha Values:**
- 1-40 (continuous)
- 45, 50, 55, 60
- 80, 100, 125, 150
- Total: 48 unique alpha values

**Output:**
- 48 combined files (one per alpha value)
- Each combines all 221 wavelengths
- Output directory: `combined_by_alpha/` (will be created automatically)
- Output files: `wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_XXX.cdf`
- File format: NETCDF4 (same as input files, compatible with libRadtran)

## Files in This Package

1. **combine_alpha_netcdf.py** - Main Python script that combines files for one alpha value
2. **submit_combine.sh** - SLURM job array script for parallel processing
3. **test_single_alpha.sh** - Test script to verify everything works
4. **run_submission.sh** - Interactive helper script to submit jobs
5. **README.md** - This file

## Quick Start

### Step 1: Copy Scripts to Alpine

Copy these scripts to your working directory on Alpine:

```bash
# SSH to Alpine
ssh anbu8374@login.rc.colorado.edu

# Create a working directory
mkdir -p ~/mie_combination
cd ~/mie_combination

# Copy all scripts here
# (You can use scp or create them directly)
```

### Step 2: Test with Single Alpha Value

Before processing all 48 alpha values, test with one:

```bash
chmod +x test_single_alpha.sh
bash test_single_alpha.sh
```

This will:
- Find and load an available Python 3 module
- Install netCDF4 if needed
- Process alpha = 14 (combines 221 wavelength files)
- Show you the output file location

**Check the output:**
```bash
# View NetCDF header
module load netcdf
ncdump -h /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/combined_by_alpha/wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_014.cdf
```

### Step 3: Submit Full Job Array

If the test looks good, submit all 48 jobs:

```bash
chmod +x run_submission.sh
bash run_submission.sh
```

Or submit directly:
```bash
chmod +x submit_combine.sh
sbatch --array=1-48 submit_combine.sh
```

## Monitoring Jobs

**Check job status:**
```bash
squeue -u anbu8374
```

**Watch output in real-time:**
```bash
tail -f logs/combine_alpha_*.out
```

**Check for errors:**
```bash
grep -i error logs/combine_alpha_*.err
```

**Check specific alpha value:**
```bash
cat logs/combine_alpha_14.out
```

## Output Structure

Each combined file contains:

**Dimensions:**
- `nlam` (221) - Number of wavelengths
- `nreff` (35) - Number of effective radii
- `nthetamax` (1000) - Maximum phase function angles
- `nmommax` (1000) - Maximum Legendre moments
- `nphamat` (1) - Phase matrix components

**Variables:**
- `wavelen(nlam)` - Wavelength array [µm]
- `reff(nreff)` - Effective radius array [µm]
- `ext(nlam, nreff)` - Extinction efficiency
- `ssa(nlam, nreff)` - Single scattering albedo
- `gg(nlam, nreff)` - Asymmetry parameter
- `refre(nlam)` - Real refractive index
- `refim(nlam)` - Imaginary refractive index
- `rho(nlam, nreff)` - Density
- `theta(nlam, nreff, nphamat, nthetamax)` - Scattering angles
- `phase(nlam, nreff, nphamat, nthetamax)` - Phase function
- `pmom(nlam, nreff, nphamat, nmommax)` - Legendre coefficients
- `ntheta(nlam, nreff, nphamat)` - Number of angles used
- `nmom(nlam, nreff, nphamat)` - Number of moments used

## Using the Combined Files

### In MATLAB:

```matlab
% Read combined NetCDF file
filename = '/projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/combined_by_alpha/wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_014.cdf';

% Read variables
wavelengths = ncread(filename, 'wavelen');  % (221,1)
reff = ncread(filename, 'reff');            % (35,1)
ext = ncread(filename, 'ext');              % (35,221) - note: transposed!
ssa = ncread(filename, 'ssa');              % (35,221)
gg = ncread(filename, 'gg');                % (35,221)

% MATLAB reads as (nreff, nlam) so you may need to transpose:
ext = ext';  % Now (221,35) - wavelength x radius
ssa = ssa';
gg = gg';

% Plot extinction vs wavelength for first radius
plot(wavelengths, ext(:,1))
xlabel('Wavelength [\mum]')
ylabel('Extinction Efficiency')
title('Extinction vs Wavelength (r_{eff} = 1 \mum)')
```

### In Python:

```python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Open file
nc = Dataset('combined_by_alpha/wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_014.cdf', 'r')

# Read variables
wavelengths = nc.variables['wavelen'][:]  # (221,)
reff = nc.variables['reff'][:]            # (35,)
ext = nc.variables['ext'][:]              # (221, 35)
ssa = nc.variables['ssa'][:]              # (221, 35)
gg = nc.variables['gg'][:]                # (221, 35)

# Plot
plt.plot(wavelengths, ext[:, 0])
plt.xlabel('Wavelength [µm]')
plt.ylabel('Extinction Efficiency')
plt.title(f'Extinction vs Wavelength (reff = {reff[0]} µm)')
plt.show()

nc.close()
```

## Troubleshooting

**Problem: Python module not found**
```bash
# Find available Python modules on Alpine
module spider python

# Load a specific version (example)
module load python/3.11.6
```

The scripts will automatically try to find an available Python 3 module. If this fails, you can edit the scripts to load a specific version available on your system.

**Problem: netCDF4 module not found**
```bash
# After loading a Python module
pip3 install --user netCDF4
```

**Problem: Files not found**
- Check the base directory path in `combine_alpha_netcdf.py`
- Verify file naming convention matches (look at an actual file)
- Make sure you're on a login node or compute node with access to `/projects/`

**Problem: Job fails**
- Check error logs: `cat logs/combine_alpha_*.err`
- Verify one file manually exists:
  ```bash
  ls /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/wavelength_300nm/wc_gamma_014_0_mie.cdf
  ```

**Problem: Out of memory**
- Increase memory in submit script: Add `#SBATCH --mem=8G`
- Each job should only need ~2-4 GB

**Problem: Timeout**
- Increase time limit: Change `#SBATCH --time=01:00:00` to `02:00:00`
- Each alpha should complete in 10-30 minutes

## Performance Notes

- Each job processes 221 files and writes one output file
- Expected runtime: 10-30 minutes per alpha value
- All 48 jobs run in parallel (limited by available compute nodes)
- Total processing time: ~30 minutes to 1 hour for all 48 alpha values

## File Sizes

- Input: Each file is ~1-2 MB
- Output: Each combined file is ~250-450 MB
- Total output: ~12-20 GB for all 48 alpha values

## Questions or Issues?

Contact: anbu8374@colorado.edu
