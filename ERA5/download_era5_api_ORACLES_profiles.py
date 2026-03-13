# if needed, install the cdsapi package using:
# pip install cdsapi
# Also need scipy: pip install scipy
import cdsapi
import os
from datetime import datetime, timedelta
import scipy.io

# Configuration - MODIFY THESE
download_dir = "/Users/anbu8374/Documents/MATLAB/Matlab-Research/Hyperspectral_Cloud_Retrievals/ERA5_reanalysis/ERA5_data/ORACLES"  # Change this to your desired directory

# MAT file path
mat_path = "/Users/anbu8374/Documents/MATLAB/Matlab-Research/Hyperspectral_Cloud_Retrievals/ORACLES/oracles_data/ensemble_profiles_with_precip_from_33_files_LWC-threshold_0.05_Nc-threshold_10_13-Mar-2026.mat"

# Create directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Dataset configuration
dataset = "reanalysis-era5-pressure-levels"

# Load MAT file
mat = scipy.io.loadmat(mat_path)
ensemble_profiles = mat['ensemble_profiles']  # (1, 243) cell array

# Function to convert MATLAB datenum to Python datetime
def matlab_datenum_to_datetime(datenum):
    from datetime import datetime, timedelta
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

# Collect unique ERA5 timestamps
unique_downloads = {}
for i in range(ensemble_profiles.shape[1]):
    profile = ensemble_profiles[0, i]
    
    # Extract timevec (MATLAB datenum for each level)
    timevec = profile['timevec'][0, 0]  # array of datenums
    datenum = timevec[0][0]  # Use the first (start) time
    
    # Convert to Python datetime
    dt = matlab_datenum_to_datetime(datenum)
    
    # Round to nearest hour (>= 30 min rounds up)
    minute = dt.minute
    second = dt.second
    if minute * 60 + second >= 1800:
        dt_rounded = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt_rounded = dt.replace(minute=0, second=0, microsecond=0)
    
    key = (dt_rounded.year, dt_rounded.month, dt_rounded.day, dt_rounded.hour)
    if key not in unique_downloads:
        unique_downloads[key] = []
    unique_downloads[key].append(i)  # Store profile index

print(f"Found {ensemble_profiles.shape[1]} profiles mapping to "
      f"{len(unique_downloads)} unique ERA5 downloads.")
print()
print("Date/time mapping (ERA5 timestamp ← profile indices):")
for (year, month, day, hour), indices in sorted(unique_downloads.items()):
    print(f"  {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC  ←  profiles {', '.join(map(str, indices))}")
print()

# Common ERA5 request parameters
base_request = {
    "product_type": ["reanalysis"],
    "variable": [
        "fraction_of_cloud_cover",
        "geopotential",
        "relative_humidity",
        "specific_humidity",
        "specific_cloud_liquid_water_content",
        "temperature",
    ],
    "pressure_level": [
        "1", "2", "3",
        "5", "7", "10",
        "20", "30", "50",
        "70", "100", "125",
        "150", "175", "200",
        "225", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000",
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [0, -10, -30, 5],  # [North, West, South, East] - Southeast Atlantic for ORACLES
}

# Initialize CDS API client
client = cdsapi.Client()

sorted_downloads = sorted(unique_downloads.items())
total = len(sorted_downloads)
print(f"Starting download of {total} ERA5 files...")
print(f"Files will be saved to: {download_dir}")
print("-" * 60)

for i, ((year, month, day, hour), indices) in enumerate(sorted_downloads, 1):
    filename = f"era5_{year}_{month:02d}_{day:02d}_{hour:02d}00UTC.nc"
    output_path = os.path.join(download_dir, filename)

    print(f"[{i}/{total}] {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC")
    print(f"      Profiles       : {', '.join(map(str, indices))}")
    print(f"      Output file    : {filename}")

    if os.path.exists(output_path):
        print(f"      File already exists — skipping.")
        print("-" * 60)
        continue

    # Build request for this specific date and hour
    request = {**base_request,
               "year": [str(year)],
               "month": [f"{month:02d}"],
               "day": [f"{day:02d}"],
               "time": [f"{hour:02d}:00"]}

    try:
        client.retrieve(dataset, request).download(output_path)
        print(f"      Complete!")
    except Exception as e:
        print(f"      Error: {e}")
        print(f"      Continuing with next download...")

    print("-" * 60)

print("All downloads complete!")