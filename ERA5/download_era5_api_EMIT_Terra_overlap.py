# if needed, install the cdsapi package using:
# pip install cdsapi
import cdsapi
import os
from datetime import datetime, timedelta

# Configuration - MODIFY THESE
download_dir = "/Users/anbu8374/Documents/MATLAB/Matlab-Research/Hyperspectral_Cloud_Retrievals/ERA5_reanalysis/ERA5_data/"  # Change this to your desired directory

# Create directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Dataset configuration
dataset = "reanalysis-era5-pressure-levels"

# Subdirectory names to process.
# Format: YYYY_M_D_THHMMSS (UTC time after 'T')
subdirectories = [
    "2023_3_6_T151922",
    "2023_3_6_T151946",
    "2023_3_6_T151958",
    "2023_3_6_T152010",
    "2024_11_26_T144359",
    "2024_11_26_T144435",
    "2024_11_4_T144644",
    "2024_1_27_T150332",
    "2024_3_9_T140852",
    "2024_9_5_T135602",
    "2025_11_7_T132422",
    "2025_11_7_T132434",
    "2025_11_7_T132446",
    "2025_1_26_T140757",
    "2025_1_26_T140809",
    "2025_1_4_T141002",
    "2025_3_6_T135149",
    "2025_9_3_T132322",
]


def parse_subdirectory(subdir_name):
    """
    Parse a subdirectory name to extract year, month, day, and closest ERA5 hour.

    Subdirectory format: YYYY_M_D_THHMMSS
    ERA5 data is available every hour on the hour (00:00 - 23:00 UTC).
    The closest hour is found by standard rounding (>= 30 min rounds up).

    Returns: (year, month, day, closest_hour) as integers, accounting for
             midnight rollovers when rounding up from 23:xx.
    """
    parts = subdir_name.split("_")
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    time_str = parts[3][1:]  # Strip leading 'T' → HHMMSS

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])

    # Round to nearest hour (>= 30 min 0 sec rounds up)
    total_seconds_past_hour = minute * 60 + second
    if total_seconds_past_hour >= 1800:  # 30 minutes in seconds
        # Use datetime to safely handle midnight rollovers (e.g., 23:45 → next day 00:00)
        rounded_dt = datetime(year, month, day, hour) + timedelta(hours=1)
        return rounded_dt.year, rounded_dt.month, rounded_dt.day, rounded_dt.hour
    else:
        return year, month, day, hour


# Parse all subdirectories and collect unique (year, month, day, hour) combinations.
# Multiple subdirectories that round to the same ERA5 timestamp share one download.
unique_downloads = {}
for subdir in subdirectories:
    key = parse_subdirectory(subdir)
    if key not in unique_downloads:
        unique_downloads[key] = []
    unique_downloads[key].append(subdir)

print(f"Found {len(subdirectories)} subdirectories mapping to "
      f"{len(unique_downloads)} unique ERA5 downloads.")
print()
print("Date/time mapping (ERA5 timestamp ← source subdirectories):")
for (year, month, day, hour), subdirs in sorted(unique_downloads.items()):
    print(f"  {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC  ←  {', '.join(subdirs)}")
print()

# Common ERA5 request parameters (same variables, pressure levels, and area as ver3)
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
    "area": [-16, -93, -45, -68],  # [North, West, South, East]
}

# Initialize CDS API client
client = cdsapi.Client()

sorted_downloads = sorted(unique_downloads.items())
total = len(sorted_downloads)
print(f"Starting download of {total} ERA5 files...")
print(f"Files will be saved to: {download_dir}")
print("-" * 60)

for i, ((year, month, day, hour), subdirs) in enumerate(sorted_downloads, 1):
    filename = f"era5_{year}_{month:02d}_{day:02d}_{hour:02d}00UTC.nc"
    output_path = os.path.join(download_dir, filename)

    print(f"[{i}/{total}] {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC")
    print(f"      Source subdirs : {', '.join(subdirs)}")
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
