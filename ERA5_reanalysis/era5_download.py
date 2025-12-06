# %% [markdown]
# # ERA5 Reanalysis Data Download
# 
# This notebook downloads ERA5 pressure level data from the Copernicus Climate Data Store (CDS).
# 
# **Requirements:**
# - Install cdsapi: `pip install cdsapi`
# - Set up CDS credentials in `~/.cdsapirc`

# %% [markdown]
# ## 1. Import Libraries

# %%
import cdsapi
import os
from datetime import datetime
import time

# %% [markdown]
# ## 2. Verify CDS API Credentials

# %%
def verify_credentials():
    """Check if CDS API credentials are configured"""
    cdsapirc_path = os.path.expanduser("~/.cdsapirc")
    
    if os.path.exists(cdsapirc_path):
        print(f"✓ Found credentials file: {cdsapirc_path}")
        return True
    else:
        print(f"✗ Credentials file not found: {cdsapirc_path}")
        print("\nTo set up your credentials:")
        print("1. Register at https://cds.climate.copernicus.eu/")
        print("2. Get your API key from your profile page")
        print("3. Create ~/.cdsapirc with:")
        print("   url: https://cds.climate.copernicus.eu/api/v2")
        print("   key: YOUR_UID:YOUR_API_KEY")
        return False

# Run credential check
verify_credentials()

# %% [markdown]
# ## 3. Define Download Parameters

# %%
# Dataset to download
dataset = "reanalysis-era5-pressure-levels"

# Request parameters
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "fraction_of_cloud_cover",
        "geopotential",
        "ozone_mass_mixing_ratio",
        "relative_humidity",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "specific_rain_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    "year": ["2008"],
    "month": ["11"],
    "day": ["11"],
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
    "download_format": "zip",
    "area": [-15, -80, -25, -70]  # [North, West, South, East]
}

# Output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"era5_data_{timestamp}.zip"

print(f"Dataset: {dataset}")
print(f"Date: 2008-11-11")
print(f"Variables: {len(request['variable'])} variables")
print(f"Pressure levels: {len(request['pressure_level'])} levels")
print(f"Time steps: {len(request['time'])} hours")
print(f"Area: {request['area']}")
print(f"Output file: {output_file}")

# %% [markdown]
# ## 4. Download Function with Error Handling

# %%
def download_era5_data(dataset, request, output_file, max_retries=3):
    """
    Download ERA5 reanalysis data with error handling and progress tracking.
    
    Args:
        dataset: CDS dataset name
        request: Request parameters dictionary
        output_file: Output filename
        max_retries: Number of retry attempts if download fails
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Initialize client
    try:
        client = cdsapi.Client()
        print("✓ CDS API client initialized successfully\n")
    except Exception as e:
        print(f"✗ Error initializing CDS API client: {e}")
        print("Make sure you have a ~/.cdsapirc file with your credentials")
        return False
    
    # Attempt download with retries
    for attempt in range(1, max_retries + 1):
        try:
            print(f"{'='*60}")
            print(f"Attempt {attempt}/{max_retries}")
            print(f"{'='*60}")
            print(f"Submitting request to CDS...")
            
            # Submit request and download
            result = client.retrieve(dataset, request)
            print(f"\nDownloading to: {output_file}")
            print("This may take several minutes...")
            result.download(output_file)
            
            # Verify download
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"\n{'='*60}")
                print(f"✓ Download successful!")
                print(f"{'='*60}")
                print(f"  File: {output_file}")
                print(f"  Size: {file_size:.2f} MB")
                return True
            else:
                raise Exception("Download completed but file not found")
                
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Download failed: {e}")
            print(f"{'='*60}")
            
            if attempt < max_retries:
                wait_time = 30 * attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts")
                return False
    
    return False

# %% [markdown]
# ## 5. Execute Download
# 
# **Note:** This cell will take several minutes to complete. CDS requests are queued and processed on their servers.

# %%
# Start download
success = download_era5_data(dataset, request, output_file)

if not success:
    print("\n⚠️  Download failed. Check error messages above.")

# %% [markdown]
# ## 6. Verify Downloaded File

# %%
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ File exists: {output_file}")
    print(f"  Size: {file_size:.2f} MB")
    
    # Optional: Load and inspect the data
    print("\nTo extract and load the data:")
    print(f"  !unzip {output_file}")
    print("  import xarray as xr")
    print("  ds = xr.open_dataset('your_netcdf_file.nc')")
    print("  print(ds)")
else:
    print(f"✗ File not found: {output_file}")

# %% [markdown]
# ## Optional: Extract and Load Data

# %%
# Uncomment to extract the zip file
# !unzip -o {output_file}

# %%
# Uncomment to load with xarray (requires: pip install xarray netcdf4)
# import xarray as xr
# 
# # List extracted files
# nc_files = [f for f in os.listdir('.') if f.endswith('.nc')]
# print(f"NetCDF files: {nc_files}")
# 
# # Load the first file
# if nc_files:
#     ds = xr.open_dataset(nc_files[0])
#     print(ds)
#     
#     # Display basic info
#     print(f"\nVariables: {list(ds.data_vars)}")
#     print(f"Coordinates: {list(ds.coords)}")
#     print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")

# %%