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
vars_single = ['10u', '10v', 'cape', 'cin', 'hcct'] # Single-level variables
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

    # 4. Retrieve single-level data
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

    # 5. Retrieve pressure-level data
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
    example_year, example_month, example_day = '2024', '04', '01'
    download_era5_data(example_year, example_month, example_day)
