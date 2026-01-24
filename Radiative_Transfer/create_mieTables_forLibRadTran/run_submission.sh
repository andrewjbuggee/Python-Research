#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=00:30:00     # Longer time for multiple files
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --job-name=alpha_combine_mieTables  
#SBATCH --output=alpha_combine_mieTables.out
#SBATCH --error=alpha_combine_mieTables.err
#SBATCH --mail-user=anbu8374@colorado.edu
#SBATCH --mail-type=ALL


# Helper script to submit the Mie table combination job array

echo "=================================================="
echo "Mie Table Combination - Job Submission"
echo "=================================================="
echo ""
echo "This will combine 221 wavelength files for each of 48 alpha values"
echo "Total files to process: 10,608 NetCDF files"
echo "Output: 48 combined files (one per alpha value)"
echo ""
echo "Alpha values to process:"
echo "  1-40 (continuous)"
echo "  45, 50, 55, 60, 80, 100, 125, 150"
echo ""
echo "Output directory:"
echo "  /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/"
echo "  mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/combined_by_alpha/"
echo ""
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "combine_alpha_netcdf.py" ]; then
    echo "ERROR: combine_alpha_netcdf.py not found in current directory"
    echo "Please cd to the directory containing the scripts"
    exit 1
fi

if [ ! -f "submit_combine.sh" ]; then
    echo "ERROR: submit_combine.sh not found in current directory"
    echo "Please cd to the directory containing the scripts"
    exit 1
fi

# Check if Python module is available
module purge

if module load python 2>/dev/null; then
    echo "Loaded default python module"
elif module load python/3.11.6 2>/dev/null; then
    echo "Loaded python/3.11.6"
elif module load python/3.10.10 2>/dev/null; then
    echo "Loaded python/3.10.10"
elif module load anaconda 2>/dev/null; then
    echo "Loaded anaconda"
else
    echo "ERROR: Could not load Python 3"
    echo "Please check available modules with: module spider python"
    exit 1
fi

# Check if netCDF4 is installed
python3 -c "import netCDF4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: netCDF4 module not found"
    echo "Installing netCDF4 for Python..."
    pip3 install --user netCDF4
    echo ""
fi

# Create logs directory
mkdir -p logs

echo "Ready to submit job array!"
echo ""
echo "Command to submit:"
echo "  sbatch --array=1-48 submit_combine.sh"
echo ""
read -p "Submit now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting job array..."
    sbatch --array=1-48 submit_combine.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Job submitted successfully!"
        echo ""
        echo "To check job status:"
        echo "  squeue -u $USER"
        echo ""
        echo "To view output logs:"
        echo "  tail -f logs/combine_alpha_*.out"
        echo ""
        echo "To check for errors:"
        echo "  grep -i error logs/combine_alpha_*.err"
    else
        echo "ERROR: Job submission failed"
        exit 1
    fi
else
    echo "Job not submitted. To submit later, run:"
    echo "  sbatch --array=1-48 submit_combine.sh"
fi

echo ""
echo "=================================================="
