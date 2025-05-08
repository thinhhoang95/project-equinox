# Wind Model Documentation

This document provides documentation and examples for the `WindModel` class found in `src/equinox/wind/wind_model.py`.

## Overview

The `WindModel` class is designed to retrieve meteorological data, specifically wind components (u and v), CAPE (Convective Available Potential Energy), and CIN (Convective Inhibition), from ERA5 NetCDF data files. It handles interpolation across spatial dimensions (latitude, longitude), time, and altitude to provide data at arbitrary query points.

## Installation

This class depends on the following Python libraries:

- `xarray`
- `numpy`
- `scipy`
- `pandas`

You can install them using pip:

```bash
pip install xarray numpy scipy pandas
```

Additionally, you will need ERA5 data files in NetCDF format (`.nc`). The `WindModel` expects these files to be named after the date they represent (e.g., `2023-01-01.nc`) and located in a specified data directory.

## Usage

### Initializing the Model

To use the `WindModel`, you need to initialize it with the date of the data you want to use and the path to the directory containing your NetCDF files.

```python
from equinox.wind.wind_model import WindModel
from datetime import datetime

# Specify the date for the data you want to load
model_date = "2023-01-01" # Make sure you have data/era5/2023-01-01.nc

# Specify the directory where your ERA5 .nc files are stored
data_dir = "data/era5"

# Initialize the model
try:
    wind_model = WindModel(date_str=model_date, data_dir=data_dir)
    print(f"WindModel initialized successfully for date: {model_date}")
except FileNotFoundError as e:
    print(f"Error initializing WindModel: {e}")
    print("Please ensure the data file exists at the specified path.")
except ValueError as e:
    print(f"Error initializing WindModel: {e}")
    print("The data file is missing required variables.")
```

### Retrieving Wind Components

Use the `get_wind_components` method to get the u and v wind components at a specific latitude, longitude, altitude, and time. Altitude is provided in feet, and the time is a `datetime` object.

The method supports spatial and temporal interpolation (`interpolate_spatial_time=True`, default) or nearest neighbor lookup (`interpolate_spatial_time=False`). Altitude is always linearly interpolated or clamped.

```python
# Define query parameters
query_lat = 50.0     # Latitude in degrees
query_lon = 0.0      # Longitude in degrees
query_alt_ft = 30000.0 # Altitude in feet
query_time = datetime(2023, 1, 1, 12, 0) # Datetime object (Year, Month, Day, Hour, Minute)

# Retrieve wind components
u_wind, v_wind = wind_model.get_wind_components(
    lat=query_lat,
    lon=query_lon,
    alt_ft=query_alt_ft,
    time=query_time,
    interpolate_spatial_time=True # Use interpolation
)

if not np.isnan(u_wind) and not np.isnan(v_wind):
    print(f"Wind components at lat={query_lat}, lon={query_lon}, alt={query_alt_ft}ft, time={query_time}:")
    print(f"u = {u_wind:.2f} m/s, v = {v_wind:.2f} m/s")
else:
    print("Could not retrieve wind components for the specified parameters.")

# Example using nearest neighbor for spatial/time
u_wind_nn, v_wind_nn = wind_model.get_wind_components(
    lat=query_lat,
    lon=query_lon,
    alt_ft=query_alt_ft,
    time=query_time,
    interpolate_spatial_time=False # Use nearest neighbor
)

if not np.isnan(u_wind_nn) and not np.isnan(v_wind_nn):
    print(f"Wind components (nearest neighbor) at lat={query_lat}, lon={query_lon}, alt={query_alt_ft}ft, time={query_time}:")
    print(f"u = {u_wind_nn:.2f} m/s, v = {v_wind_nn:.2f} m/s")
else:
    print("Could not retrieve wind components (nearest neighbor) for the specified parameters.")
```

### Retrieving CAPE and CIN

Use the `get_cape_cin` method to get the CAPE and CIN values at a specific latitude, longitude, and time. These are surface variables, so altitude is not a parameter.

Similar to `get_wind_components`, this method supports interpolation (`interpolate=True`, default) or nearest neighbor (`interpolate=False`).

```python
# Define query parameters
query_lat = 50.0 # Latitude in degrees
query_lon = 0.0  # Longitude in degrees
query_time = datetime(2023, 1, 1, 12, 0) # Datetime object

# Retrieve CAPE and CIN
cape_val, cin_val = wind_model.get_cape_cin(
    lat=query_lat,
    lon=query_lon,
    time=query_time,
    interpolate=True # Use interpolation
)

if not np.isnan(cape_val) and not np.isnan(cin_val):
    print(f"CAPE and CIN at lat={query_lat}, lon={query_lon}, time={query_time}:")
    print(f"CAPE = {cape_val:.2f} J/kg, CIN = {cin_val:.2f} J/kg")
else:
     print("Could not retrieve CAPE/CIN for the specified parameters.")

# Example using nearest neighbor
cape_val_nn, cin_val_nn = wind_model.get_cape_cin(
    lat=query_lat,
    lon=query_lon,
    time=query_time,
    interpolate=False # Use nearest neighbor
)

if not np.isnan(cape_val_nn) and not np.isnan(cin_val_nn):
    print(f"CAPE and CIN (nearest neighbor) at lat={query_lat}, lon={query_lon}, time={query_time}:")
    print(f"CAPE = {cape_val_nn:.2f} J/kg, CIN = {cin_val_nn:.2f} J/kg")
else:
     print("Could not retrieve CAPE/CIN (nearest neighbor) for the specified parameters.")
```

## Error Handling

The `WindModel` constructor will raise `FileNotFoundError` if the specified data file does not exist and `ValueError` if essential variables are missing from the dataset. The retrieval methods (`get_wind_components`, `get_cape_cin`) will return `(np.nan, np.nan)` if data cannot be retrieved due to issues like interpolation failures or missing altitude levels.

## Internal Details

- Pressure-to-altitude conversion is performed using the International Standard Atmosphere (ISA) formula.
- Surface wind components (`u10`, `v10`) are assumed to be at 10 meters altitude (`_SURFACE_ALTITUDE_M`).
- High-altitude winds are interpolated across available pressure levels, which are converted to altitudes using the ISA formula.
- Spatial and temporal interpolation uses `xarray`'s `interp` method with linear or nearest neighbor methods. Altitude interpolation for high-altitude winds uses `scipy.interpolate.interp1d` with linear interpolation and clamping to boundary values for extrapolation.
- Time clipping ensures that query times outside the data's valid time range are clamped to the nearest available time within the data.

```python
# ... existing code ...
```
