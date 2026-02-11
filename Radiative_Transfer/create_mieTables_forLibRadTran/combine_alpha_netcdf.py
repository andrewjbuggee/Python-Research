#!/usr/bin/env python3
"""
Combine NetCDF Mie table files across wavelengths for a single alpha value.

Usage: python combine_alpha_netcdf.py <alpha_value>
Example: python combine_alpha_netcdf.py 14
"""

from netCDF4 import Dataset
import numpy as np
import sys
import os
from glob import glob

def get_alpha_string(alpha_value):
    """Convert alpha value to the string format used in filenames."""
    return f"{alpha_value:03d}"

def combine_netcdf_for_alpha(alpha_value, base_dir, output_dir):
    """
    Combine NetCDF files across all wavelengths for a single alpha value.
    
    Parameters:
    -----------
    alpha_value : int
        The alpha value (e.g., 1, 14, 80, etc.)
    base_dir : str
        Base directory containing wavelength subdirectories
    output_dir : str
        Directory to save combined NetCDF file
    """
    
    # Format alpha for filename (e.g., 1 -> '001', 14 -> '014', 80 -> '080')
    alpha_str = get_alpha_string(alpha_value)
    
    print(f"=" * 70)
    print(f"Processing alpha = {alpha_value} (filename: {alpha_str})")
    print(f"=" * 70)
    
    # Get all wavelength directories (sorted by wavelength number)
    wavelength_dirs = sorted(
        glob(os.path.join(base_dir, 'wavelength_*nm')),
        key=lambda x: int(x.split('_')[-1].replace('nm', ''))
    )
    
    if len(wavelength_dirs) == 0:
        print(f"ERROR: No wavelength directories found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(wavelength_dirs)} wavelength directories")
    
    # Extract wavelengths from directory names
    wavelengths = []
    valid_files = []
    
    for wdir in wavelength_dirs:
        wl_nm = int(wdir.split('_')[-1].replace('nm', ''))
        wavelengths.append(wl_nm)
        
        # Construct filename - note the underscore after alpha string
        nc_file = os.path.join(wdir, f'wc.gamma_{alpha_str}.0.mie.cdf')
        
        if not os.path.exists(nc_file):
            print(f"WARNING: File not found: {nc_file}")
            continue
        
        valid_files.append(nc_file)
    
    if len(valid_files) == 0:
        print(f"ERROR: No valid files found for alpha = {alpha_value}")
        sys.exit(1)
    
    print(f"Found {len(valid_files)} valid files")
    
    # Read first file to get structure
    first_file = valid_files[0]
    print(f"\nReading template file: {os.path.basename(first_file)}")
    
    nc_sample = Dataset(first_file, 'r')
    
    # Get dimensions
    nreff = len(nc_sample.dimensions['nreff'])
    nthetamax = len(nc_sample.dimensions['nthetamax'])
    nmommax = len(nc_sample.dimensions['nmommax'])
    nphamat = len(nc_sample.dimensions['nphamat'])
    
    # Get reff array (should be same for all files)
    reff = nc_sample.variables['reff'][:]
    
    print(f"\nDimensions:")
    print(f"  nreff (radii): {nreff}")
    print(f"  nthetamax: {nthetamax}")
    print(f"  nmommax: {nmommax}")
    print(f"  nphamat: {nphamat}")
    print(f"  Wavelengths to combine: {len(valid_files)}")
    
    nc_sample.close()
    
    # Initialize arrays
    nlam = len(valid_files)
    wavelengths_arr = np.zeros(nlam)
    
    # Variables with shape (nlam, nreff)
    ext_all = np.zeros((nlam, nreff))
    ssa_all = np.zeros((nlam, nreff))
    rho_all = np.zeros((nlam, nreff))
    
    # Variables with shape (nlam)
    refre_all = np.zeros(nlam)
    refim_all = np.zeros(nlam)
    
    # Variables with shape (nlam, nreff, nphamat)
    ntheta_all = np.zeros((nlam, nreff, nphamat), dtype=np.int32)
    nmom_all = np.zeros((nlam, nreff, nphamat), dtype=np.int32)
    
    # Variables with shape (nlam, nreff, nphamat, nthetamax)
    theta_all = np.zeros((nlam, nreff, nphamat, nthetamax), dtype=np.float32)
    phase_all = np.zeros((nlam, nreff, nphamat, nthetamax), dtype=np.float32)
    
    # Variables with shape (nlam, nreff, nphamat, nmommax)
    pmom_all = np.zeros((nlam, nreff, nphamat, nmommax), dtype=np.float32)
    
    # Read all files
    print("\nReading all NetCDF files...")
    for i, nc_file in enumerate(valid_files):
        if i % 20 == 0:
            print(f"  Processing file {i+1}/{nlam}")
        
        nc = Dataset(nc_file, 'r')
        
        # Read scalar wavelength
        wavelengths_arr[i] = nc.variables['wavelen'][0]
        
        # Read 2D variables (squeeze out nlam=1 dimension)
        ext_all[i, :] = np.squeeze(nc.variables['ext'][:])
        ssa_all[i, :] = np.squeeze(nc.variables['ssa'][:])
        rho_all[i, :] = np.squeeze(nc.variables['rho'][:])
        
        # Read scalar refractive indices
        refre_all[i] = nc.variables['refre'][0]
        refim_all[i] = nc.variables['refim'][0]
        
        # Read 3D variables
        ntheta_all[i, :, :] = np.squeeze(nc.variables['ntheta'][:], axis=0)
        nmom_all[i, :, :] = np.squeeze(nc.variables['nmom'][:], axis=0)
        
        # Read 4D variables
        theta_all[i, :, :, :] = np.squeeze(nc.variables['theta'][:], axis=0)
        phase_all[i, :, :, :] = np.squeeze(nc.variables['phase'][:], axis=0)
        pmom_all[i, :, :, :] = np.squeeze(nc.variables['pmom'][:], axis=0)
        
        nc.close()
    
    print(f"  Completed reading {nlam} files")
    
    # Create output file with custom name
    output_file = os.path.join(output_dir, f'wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_{alpha_str}.cdf')
    print(f"\nWriting combined file: {output_file}")
    
    # Use NETCDF4 format (same as input files) with .cdf extension
    nc_out = Dataset(output_file, 'w', format='NETCDF4')

    # Create dimensions
    nc_out.createDimension('nlam', nlam)
    nc_out.createDimension('nreff', nreff)
    nc_out.createDimension('nthetamax', nthetamax)
    nc_out.createDimension('nmommax', nmommax)
    nc_out.createDimension('nphamat', nphamat)
    nc_out.createDimension('nrho', 1)

    # Global attributes - version is CRITICAL for libRadTran compatibility.
    # Without it, libRadTran assumes "old format" (single wavelength per file)
    # and shifts all array indexing by one dimension, causing read errors.
    nc_out.version = np.int64(20090626)
    nc_out.file_info = f'Custom Mie table for gamma distribution with alpha = {alpha_value}'
    nc_out.parameterization = 'mie'
    nc_out.size_distr = 'Gamma distribution.'
    nc_out.param_alpha = float(alpha_value)

    # Create and write variables
    # 1D variables
    var_wavelen = nc_out.createVariable('wavelen', 'f8', ('nlam',))
    var_wavelen[:] = wavelengths_arr

    var_reff = nc_out.createVariable('reff', 'f8', ('nreff',))
    var_reff[:] = reff

    var_refre = nc_out.createVariable('refre', 'f8', ('nlam',))
    var_refre[:] = refre_all

    var_refim = nc_out.createVariable('refim', 'f8', ('nlam',))
    var_refim[:] = refim_all

    # rho: libRadTran expects shape (nrho,), not (nlam, nreff)
    var_rho = nc_out.createVariable('rho', 'f8', ('nrho',))
    var_rho[:] = [rho_all.flat[0]]

    # 2D variables
    var_ext = nc_out.createVariable('ext', 'f8', ('nlam', 'nreff'))
    var_ext[:] = ext_all

    var_ssa = nc_out.createVariable('ssa', 'f8', ('nlam', 'nreff'))
    var_ssa[:] = ssa_all

    # Note: 'gg' is intentionally omitted. libRadTran computes the asymmetry
    # parameter from pmom internally; the reference files do not include 'gg'.

    # 3D variables
    var_ntheta = nc_out.createVariable('ntheta', 'i4', ('nlam', 'nreff', 'nphamat'))
    var_ntheta[:] = ntheta_all

    # nmom: libRadTran expects shape (nlam, nreff), not (nlam, nreff, nphamat)
    var_nmom = nc_out.createVariable('nmom', 'i4', ('nlam', 'nreff'))
    var_nmom[:] = nmom_all[:, :, 0]

    # 4D variables with _FillValue for unused entries
    var_theta = nc_out.createVariable('theta', 'f4', ('nlam', 'nreff', 'nphamat', 'nthetamax'),
                                       fill_value=np.float32(-999.0))
    theta_masked = np.ma.masked_values(theta_all, -999.0)
    var_theta[:] = theta_masked

    var_phase = nc_out.createVariable('phase', 'f4', ('nlam', 'nreff', 'nphamat', 'nthetamax'),
                                       fill_value=np.float32(-999.0))
    phase_masked = np.ma.masked_values(phase_all, -999.0)
    var_phase[:] = phase_masked

    # pmom: _FillValue=0.0 (matching reference wc.sol.mie.cdf)
    var_pmom = nc_out.createVariable('pmom', 'f4', ('nlam', 'nreff', 'nphamat', 'nmommax'),
                                      fill_value=np.float32(0.0))
    pmom_masked = np.ma.masked_values(pmom_all, 0.0)
    var_pmom[:] = pmom_masked
    
    nc_out.close()
    
    print(f"\nSUCCESS!")
    print(f"  Created: {output_file}")
    print(f"  Dimensions: nlam={nlam}, nreff={nreff}")
    print(f"  Wavelength range: {wavelengths_arr[0]:.3f} - {wavelengths_arr[-1]:.3f} µm")
    print(f"  Reff range: {reff[0]:.1f} - {reff[-1]:.1f} µm")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python combine_alpha_netcdf.py <alpha_value>")
        print("Example: python combine_alpha_netcdf.py 14")
        sys.exit(1)
    
    alpha = int(sys.argv[1])
    
    # Validate alpha value
    valid_alphas = list(range(1, 41)) + list(range(45, 61, 5)) + [80, 100, 125, 150]
    if alpha not in valid_alphas:
        print(f"ERROR: Invalid alpha value {alpha}")
        print(f"Valid alpha values: {valid_alphas}")
        sys.exit(1)
    
    base_dir = '/projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha'
    output_dir = os.path.join(base_dir, 'combined_by_alpha')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    combine_netcdf_for_alpha(alpha, base_dir, output_dir)
