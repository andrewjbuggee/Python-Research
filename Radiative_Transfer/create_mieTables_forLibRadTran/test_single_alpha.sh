#!/bin/bash
# Test script to process a single alpha value before submitting full job array

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=00:30:00     # Longer time for multiple files
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --job-name=test_single_alpha_combine_mieTables
#SBATCH --output=test_single_alpha_combine_mieTables.out
#SBATCH --error=test_single_alpha_combine_mieTables.err
#SBATCH --mail-user=anbu8374@colorado.edu
#SBATCH --mail-type=ALL



echo "=================================================="
echo "Test Run - Single Alpha Value"
echo "=================================================="
echo ""

# Check if scripts exist
if [ ! -f "combine_alpha_netcdf.py" ]; then
    echo "ERROR: combine_alpha_netcdf.py not found"
    exit 1
fi

# Load Python module
module purge

echo "Searching for Python 3 module..."
if module load python 2>/dev/null; then
    echo "Loaded default python module"
elif module load python/3.11.6 2>/dev/null; then
    echo "Loaded python/3.11.6"
elif module load python/3.10.10 2>/dev/null; then
    echo "Loaded python/3.10.10"
elif module load python/3.9.10 2>/dev/null; then
    echo "Loaded python/3.9.10"
elif module load anaconda 2>/dev/null; then
    echo "Loaded anaconda"
else
    echo "ERROR: Could not find Python 3"
    echo ""
    echo "Please run 'module spider python' to see available versions"
    echo "Then edit this script to load the correct module"
    exit 1
fi

# Show Python version
python3 --version
echo ""

# Check/install netCDF4
python3 -c "import netCDF4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing netCDF4..."
    pip3 install --user netCDF4
    echo ""
fi

# Test with alpha = 14 (should match your uploaded example file)
ALPHA_TEST=14

echo "Testing with alpha = $ALPHA_TEST"
echo "This will combine 221 wavelength files into one file"
echo "Expected output: wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_014.cdf"
echo ""
echo "Starting test run..."
echo "=================================================="
echo ""

python3 combine_alpha_netcdf.py $ALPHA_TEST

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST SUCCESSFUL!"
    echo ""
    echo "Output file should be at:"
    echo "  /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/"
    echo "  mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/"
    echo "  combined_by_alpha/wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_014.cdf"
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
