# if needed, install the cdsapi package using:
# pip install cdsapi
import cdsapi
import os

# Configuration - MODIFY THESE
download_dir = "/Users/andrewbuggee/Documents/MATLAB/Matlab-Research/Hyperspectral_Cloud_Retrievals/ERA5_reanalysis/ERA5_data/VOCALS_REx_overlap/November_2008/"  # Change this to your desired directory

base_filename = "era5_vocalsrex_november_2008"  # Base filename (will add date suffix)

# Create directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Dataset configuration
dataset = "reanalysis-era5-pressure-levels"

# Days to download
days = ["2", "4", "6", "9", "11", "13", "15"]

# Common request parameters
base_request = {
    "product_type": ["reanalysis"],
    "variable": [
        "fraction_of_cloud_cover",
        "relative_humidity",
        "specific_humidity",
        "specific_cloud_liquid_water_content",
        "temperature"
    ],
    "year": ["2008"],
    "month": ["10"],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
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
        "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [-16, -93, -45, -68]  # [North, West, South, East]
}

# Initialize client
client = cdsapi.Client()

# Loop through each day
print(f"Starting download of {len(days)} days...")
print(f"Files will be saved to: {download_dir}")
print("-" * 60)

for i, day in enumerate(days, 1):
    # Create filename for this day
    filename = f"{base_filename}_day{day}.nc"
    output_path = os.path.join(download_dir, filename)
    
    # Create request for this specific day
    request = base_request.copy()
    request["day"] = [day]
    
    # Submit request and download
    print(f"[{i}/{len(days)}] Downloading data for 2008-10-{day}...")
    print(f"      Output: {filename}")
    
    try:
        client.retrieve(dataset, request).download(output_path)
        print(f"      ✓ Complete!")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        print(f"      Continuing with next day...")
    
    print("-" * 60)

print("All downloads complete!")