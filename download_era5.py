import cdsapi
import xarray as xr
import os
import tempfile

# 1. Designate output directory
output_dir = 'data/era5'
os.makedirs(output_dir, exist_ok=True)

# Global parameters that remain the same
times = [f'{h:02d}:00' for h in range(24)]
pressure_levels = ['850', '300']  # 300 hPa for cruise altitude, 850 hPa for top of boundary layer 
vars_single = ['10u', '10v', 'cape', 'hcct'] # Single-level variables
vars_pressure = ['u', 'v', 'w'] # Pressure-level variables
area = [72, -15, 30, 41] # N, W, S, E

def download_era5_data(year: str, month: str, day: str):
    """
    Downloads ERA5 data for a specific date, processes it, and saves it.

    Args:
        year (str): The year (e.g., '2024').
        month (str): The month (e.g., '04').
        day (str): The day (e.g., '01').
    """
    # 2. Define dates/times and variables are now passed or global

    # Designate a temporary directory for intermediate files
    temp_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)

    base_filename = f"{year}-{month}-{day}"
    single_level_temp_path = os.path.join(temp_dir, f"{base_filename}_single.nc")
    pressure_level_temp_path = os.path.join(temp_dir, f"{base_filename}_pressure.nc")
    final_output_path = os.path.join(output_dir, f"{base_filename}.nc")

    # 3. Initialize CDS client
    c = cdsapi.Client()

    # 4. Retrieve single-level data (skip if file already exists)
    if os.path.exists(single_level_temp_path):
        print(f"Single-level data already exists: {single_level_temp_path}")
    else:
        print(f"Downloading single-level data to: {single_level_temp_path}")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': vars_single,
                'year': year,
                'month': month,
                'day': day,
                'time': times,
                'area': area,
            },
            single_level_temp_path
        )

    # 5. Retrieve pressure-level data (skip if file already exists)
    if os.path.exists(pressure_level_temp_path):
        print(f"Pressure-level data already exists: {pressure_level_temp_path}")
    else:
        print(f"Downloading pressure-level data to: {pressure_level_temp_path}")
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': vars_pressure,
                'pressure_level': pressure_levels,
                'year': year,
                'month': month,
                'day': day,
                'time': times,
                'area': area,
            },
            pressure_level_temp_path
        )

    # 6. Load into xarray
    ds_single   = xr.open_dataset(single_level_temp_path)
    ds_pressure = xr.open_dataset(pressure_level_temp_path)

    # Handle the expver dimension if it exists by merging ERA5 and ERA5T data
    if 'expver' in ds_single.dims:
        if ds_single.dims['expver'] > 1:
            # Try to combine ERA5 (expver=1) and ERA5T (expver=5) data
            try:
                ds_single = ds_single.sel(expver=1).combine_first(ds_single.sel(expver=5))
            except KeyError:
                # If specific expver values don't exist, just take the first one
                ds_single = ds_single.isel(expver=0)
        else:
            # If only one expver value, remove the dimension
            ds_single = ds_single.squeeze('expver', drop=True)
    
    if 'expver' in ds_pressure.dims:
        if ds_pressure.dims['expver'] > 1:
            # Try to combine ERA5 (expver=1) and ERA5T (expver=5) data
            try:
                ds_pressure = ds_pressure.sel(expver=1).combine_first(ds_pressure.sel(expver=5))
            except KeyError:
                # If specific expver values don't exist, just take the first one
                ds_pressure = ds_pressure.isel(expver=0)
        else:
            # If only one expver value, remove the dimension
            ds_pressure = ds_pressure.squeeze('expver', drop=True)

    # Clean up singleton dimensions and ambiguous coordinates
    for var in ['number', 'expver']:
        if var in ds_single.coords and var not in ds_single.dims:
            ds_single = ds_single.drop_vars(var, errors='ignore')
        if var in ds_pressure.coords and var not in ds_pressure.dims:
            ds_pressure = ds_pressure.drop_vars(var, errors='ignore')

    # 7. Merge and save
    ds = xr.merge([ds_single, ds_pressure])
    ds.to_netcdf(final_output_path)

    print(f'Combined dataset saved to: {final_output_path}')

    # Clean up temporary files
    try:
        os.remove(single_level_temp_path)
        os.remove(pressure_level_temp_path)
        # Attempt to remove the temporary directory if it's empty
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except OSError as e:
        print(f"Error cleaning up temporary files: {e}")

if __name__ == '__main__':
    # Example of how to call the function
    # You can modify this to take command-line arguments or loop through dates
    from datetime import datetime, timedelta
    
    # Download data from April 1, 2023 to December 1, 2023
    start_date = datetime(2023, 4, 1)
    end_date = datetime(2023, 12, 1)
    
    current_date = start_date
    while current_date <= end_date:
        year = str(current_date.year)
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"
        
        print(f"Downloading data for {year}-{month}-{day}")
        download_era5_data(year, month, day)
        
        # Move to next day
        current_date += timedelta(days=1)
