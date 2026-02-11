#!/usr/bin/env python3
"""
Fix existing combined CDF mie table files to be compatible with libRadTran.

The main issues:
1. Missing 'version' global attribute - without it, libRadTran assumes
   "old format" (nolam=1) and shifts all array indexing by one dimension,
   causing the "theta must begin with 180 degrees" error.
2. 'nmom' has dimensions (nlam, nreff, nphamat) but libRadTran expects (nlam, nreff).
3. 'rho' has dimensions (nlam, nreff) but libRadTran expects (nrho,).
4. 'gg' variable is present but libRadTran doesn't read it (computes from pmom).
5. theta/phase should use _FillValue=-999.0 for unused entries.

Usage:
    python fix_cdf_for_libradtran.py <input.cdf> [output.cdf]

If output is not specified, the input file is overwritten.
To fix all files in a directory:
    python fix_cdf_for_libradtran.py --all <directory>
"""

from netCDF4 import Dataset
import numpy as np
import sys
import os
import shutil
from glob import glob


def fix_cdf_file(input_path, output_path=None):
    """
    Read an existing combined CDF file and rewrite it in the format
    that libRadTran expects.
    """
    if output_path is None:
        output_path = input_path

    print(f"Fixing: {input_path}")

    # Read all data from the input file
    nc_in = Dataset(input_path, 'r')

    # Read dimensions
    nlam = len(nc_in.dimensions['nlam'])
    nreff = len(nc_in.dimensions['nreff'])
    nthetamax = len(nc_in.dimensions['nthetamax'])
    nmommax = len(nc_in.dimensions['nmommax'])
    nphamat = len(nc_in.dimensions['nphamat'])

    # Read all variables
    wavelen = nc_in.variables['wavelen'][:]
    reff = nc_in.variables['reff'][:]
    refre = nc_in.variables['refre'][:]
    refim = nc_in.variables['refim'][:]
    ext = nc_in.variables['ext'][:]
    ssa = nc_in.variables['ssa'][:]
    rho_data = nc_in.variables['rho'][:]

    ntheta = nc_in.variables['ntheta'][:]
    nmom = nc_in.variables['nmom'][:]
    theta = nc_in.variables['theta'][:]
    phase = nc_in.variables['phase'][:]
    pmom = nc_in.variables['pmom'][:]

    # Get alpha from global attributes if available
    alpha_value = getattr(nc_in, 'alpha_parameter', None)

    nc_in.close()

    # If overwriting, write to a temp file first
    if output_path == input_path:
        tmp_path = input_path + '.tmp'
    else:
        tmp_path = output_path

    # Create the fixed output file
    nc_out = Dataset(tmp_path, 'w', format='NETCDF4')

    # Create dimensions (same as before)
    nc_out.createDimension('nlam', nlam)
    nc_out.createDimension('nreff', nreff)
    nc_out.createDimension('nthetamax', nthetamax)
    nc_out.createDimension('nmommax', nmommax)
    nc_out.createDimension('nphamat', nphamat)
    nc_out.createDimension('nrho', 1)

    # --- FIX 1: Add version global attribute ---
    # This is the critical fix. Without 'version', libRadTran sets nolam=1
    # and shifts all array indexing, causing the theta error.
    nc_out.version = np.int64(20090626)

    # Add other global attributes matching the reference format
    if alpha_value is not None:
        nc_out.param_alpha = float(alpha_value)
    nc_out.parameterization = 'mie'
    nc_out.size_distr = 'Gamma distribution.'
    nc_out.file_info = 'Custom Mie table for gamma distribution'

    # --- Write 1D variables ---
    var_wavelen = nc_out.createVariable('wavelen', 'f8', ('nlam',))
    var_wavelen[:] = wavelen

    var_reff = nc_out.createVariable('reff', 'f8', ('nreff',))
    var_reff[:] = reff

    var_refre = nc_out.createVariable('refre', 'f8', ('nlam',))
    var_refre[:] = refre

    var_refim = nc_out.createVariable('refim', 'f8', ('nlam',))
    var_refim[:] = refim

    # --- FIX 3: rho should be (nrho,) not (nlam, nreff) ---
    var_rho = nc_out.createVariable('rho', 'f8', ('nrho',))
    # Use a single density value (water = 1.0 g/cm³)
    var_rho[:] = [rho_data.flat[0]]

    # --- Write 2D variables ---
    var_ext = nc_out.createVariable('ext', 'f8', ('nlam', 'nreff'))
    var_ext[:] = ext

    var_ssa = nc_out.createVariable('ssa', 'f8', ('nlam', 'nreff'))
    var_ssa[:] = ssa

    # --- FIX 4: Do NOT write 'gg' variable ---
    # libRadTran computes gg from pmom internally. The reference files
    # do not have a 'gg' variable.

    # --- Write 3D variables ---
    var_ntheta = nc_out.createVariable('ntheta', 'i4', ('nlam', 'nreff', 'nphamat'))
    var_ntheta[:] = ntheta

    # --- FIX 2: nmom should be (nlam, nreff) not (nlam, nreff, nphamat) ---
    var_nmom = nc_out.createVariable('nmom', 'i4', ('nlam', 'nreff'))
    # Take the first (and only, since nphamat=1) phamat component
    if nmom.ndim == 3:
        var_nmom[:] = nmom[:, :, 0]
    else:
        var_nmom[:] = nmom

    # --- FIX 5: Write theta/phase/pmom with proper _FillValue and masking ---
    # Mask entries beyond the valid count (ntheta/nmom) for each (lam, reff)
    # rather than relying on matching specific sentinel values.

    # Build theta/phase mask from ntheta
    theta_mask = np.ones_like(theta, dtype=bool)
    for i in range(nlam):
        for j in range(nreff):
            for k in range(nphamat):
                nt = ntheta[i, j, k]
                theta_mask[i, j, k, :nt] = False

    var_theta = nc_out.createVariable('theta', 'f4', ('nlam', 'nreff', 'nphamat', 'nthetamax'),
                                       fill_value=np.float32(-999.0))
    theta_clean = np.where(theta_mask, -999.0, theta)
    var_theta[:] = np.ma.array(theta_clean, mask=theta_mask)

    var_phase = nc_out.createVariable('phase', 'f4', ('nlam', 'nreff', 'nphamat', 'nthetamax'),
                                       fill_value=np.float32(-999.0))
    phase_clean = np.where(theta_mask, -999.0, phase)
    var_phase[:] = np.ma.array(phase_clean, mask=theta_mask)

    # pmom: _FillValue=0.0 (matching reference wc.sol.mie.cdf)
    # Build mask from nmom (which is now 2D after the fix above)
    nmom_2d = nmom[:, :, 0] if nmom.ndim == 3 else nmom
    pmom_mask = np.ones((nlam, nreff, nphamat, nmommax), dtype=bool)
    for i in range(nlam):
        for j in range(nreff):
            nm = nmom_2d[i, j]
            pmom_mask[i, j, :, :nm] = False

    var_pmom = nc_out.createVariable('pmom', 'f4', ('nlam', 'nreff', 'nphamat', 'nmommax'),
                                      fill_value=np.float32(0.0))
    pmom_clean = np.where(pmom_mask, 0.0, pmom)
    var_pmom[:] = np.ma.array(pmom_clean, mask=pmom_mask)

    nc_out.close()

    # If overwriting, move tmp file over original
    if output_path == input_path:
        shutil.move(tmp_path, output_path)

    print(f"  -> Fixed: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fix_cdf_for_libradtran.py <input.cdf> [output.cdf]")
        print("  python fix_cdf_for_libradtran.py --all <directory>")
        sys.exit(1)

    if sys.argv[1] == '--all':
        if len(sys.argv) < 3:
            print("ERROR: --all requires a directory argument")
            sys.exit(1)
        directory = sys.argv[2]
        cdf_files = sorted(glob(os.path.join(directory, '*.cdf')))
        if not cdf_files:
            print(f"No .cdf files found in {directory}")
            sys.exit(1)
        print(f"Found {len(cdf_files)} CDF files to fix")
        for f in cdf_files:
            fix_cdf_file(f)
        print(f"\nDone! Fixed {len(cdf_files)} files.")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        fix_cdf_file(input_path, output_path)
        print("\nDone!")


if __name__ == "__main__":
    main()
