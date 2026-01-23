#!/bin/bash
# Test script to process a single alpha value before submitting full job array

echo "=================================================="
echo "Test Run - Single Alpha Value"
echo "=================================================="
echo ""

# Check if scripts exist
if [ ! -f "combine_alpha_netcdf.py" ]; then
    echo "ERROR: combine_alpha_netcdf.py not found"
    exit 1
fi

# Load Python
module purge
module load python/3.10.2

# Check/install netCDF4
python -c "import netCDF4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing netCDF4..."
    pip install --user netCDF4
    echo ""
fi

# Test with alpha = 14 (should match your uploaded example file)
ALPHA_TEST=14

echo "Testing with alpha = $ALPHA_TEST"
echo "This will combine 221 wavelength files into one file"
echo "Expected output: wc_gamma_014_combined.nc"
echo ""
echo "Starting test run..."
echo "=================================================="
echo ""

python combine_alpha_netcdf.py $ALPHA_TEST

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST SUCCESSFUL!"
    echo ""
    echo "Output file should be at:"
    echo "  /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/"
    echo "  mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/"
    echo "  combined_by_alpha/wc_gamma_014_combined.nc"
    echo ""
    echo "You can inspect the output file with:"
    echo "  ncdump -h <output_file>"
    echo "  or load it in MATLAB/Python"
    echo ""
    echo "If the test looks good, submit the full job array:"
    echo "  bash run_submission.sh"
    echo "  or"
    echo "  sbatch --array=1-48 submit_combine.sh"
else
    echo "TEST FAILED!"
    echo "Exit code: $EXIT_CODE"
    echo "Please check the error messages above"
    echo ""
    echo "Common issues:"
    echo "  - Check that base directory path is correct"
    echo "  - Verify file naming convention matches"
    echo "  - Ensure netCDF4 is properly installed"
fi
echo "=================================================="
