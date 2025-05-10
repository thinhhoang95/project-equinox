import xarray as xr
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd

def time_clip(time_coord: datetime, time_min: datetime, time_max: datetime) -> datetime:
    if time_coord < time_min:
        return time_min
    elif time_coord > time_max:
        return time_max
    else:
        return time_coord

class WindModel:
    """
    A model to retrieve wind components (u, v), CAPE, and CIN from ERA5 NetCDF data.
    It handles interpolation in space, time, and altitude.
    """

    # International Standard Atmosphere constants for pressure-to-altitude conversion
    _P0 = 1013.25  # Standard sea level pressure (hPa)
    _T0 = 288.15   # Standard sea level temperature (K)
    _L = 0.0065    # Standard temperature lapse rate (K/m)
    _R_SPECIFIC = 287.058  # Specific gas constant for dry air (J/(kg·K))
    _G = 9.80665   # Gravitational acceleration (m/s^2)
    _ISA_EXPONENT = (_R_SPECIFIC * _L) / _G  # Approx. 0.190263

    _SURFACE_ALTITUDE_M = 10.0 # Altitude for u10/v10 variables (m)

    def _pressure_to_altitude(self, p_hpa_array: np.ndarray) -> np.ndarray:
        """Converts pressure (hPa) to altitude (m) using the ISA formula."""
        # Ensure pressure is positive to avoid issues with power of negative numbers
        p_ratio = np.maximum(p_hpa_array, 1e-3) / self._P0
        return (self._T0 / self._L) * (1 - p_ratio**self._ISA_EXPONENT)

    def __init__(self, date_str: str, data_dir: str = "data/era5"):
        """
        Initializes the WindModel by loading and preprocessing data for a specific date.

        Args:
            date_str: The date string in 'YYYY-MM-DD' format.
            data_dir: The directory containing the NetCDF data files.
        
        Raises:
            FileNotFoundError: If the NetCDF file for the given date is not found.
            ValueError: If essential variables are missing from the dataset.
        """
        file_path = f"{data_dir}/{date_str}.nc"
        try:
            self.data = xr.open_dataset(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")

        required_vars = ['u10', 'v10', 'u', 'v', 'cape']
        for var in required_vars:
            if var not in self.data:
                raise ValueError(f"Variable '{var}' not found in dataset {file_path}.")
            
        # Ensure coordinates are sorted for robust interpolation
        self.data = self.data.sortby("latitude")
        self.data = self.data.sortby("longitude")
        self.data = self.data.sortby("valid_time")


        # Prepare pressure levels and their corresponding altitudes for u, v variables
        if 'pressure_level' in self.data['u'].coords:
            # Pressure levels (e.g., [50, 100, 200, 300] hPa). Sort descending for typical representation.
            pressure_levels_hpa_orig = self.data['u'].pressure_level.sortby(self.data['u'].pressure_level, ascending=False).values
            model_altitudes_m = self._pressure_to_altitude(pressure_levels_hpa_orig)

            # Sort by altitude (ascending) for interpolation
            sort_indices = np.argsort(model_altitudes_m)
            self.pressure_levels_hpa = pressure_levels_hpa_orig[sort_indices]
            self.model_altitudes_at_pressure_levels_m = model_altitudes_m[sort_indices]

            if len(self.pressure_levels_hpa) < 2 and len(self.pressure_levels_hpa) > 0:
                print(f"Warning: Only {len(self.pressure_levels_hpa)} pressure level(s) found for u/v. "
                      f"Altitude interpolation for high-altitude winds will be limited to this/these level(s).")
            elif len(self.pressure_levels_hpa) == 0:
                 print("Warning: No pressure levels found for u/v. High-altitude wind components will not be available.")
                 self.pressure_levels_hpa = np.array([])
                 self.model_altitudes_at_pressure_levels_m = np.array([])
        else:
            print("Warning: 'pressure_level' coordinate not found for 'u' component. "
                  "High-altitude wind components will not be available.")
            self.pressure_levels_hpa = np.array([])
            self.model_altitudes_at_pressure_levels_m = np.array([])
        
        # Min/max for coordinate clipping
        self._lat_min = self.data.latitude.min().item()
        self._lat_max = self.data.latitude.max().item()
        self._lon_min = self.data.longitude.min().item()
        self._lon_max = self.data.longitude.max().item()
        # Convert min/max valid_time from numpy datetime64[ns] to Python datetime
        self._time_min = pd.to_datetime(self.data.valid_time.min().item()).to_pydatetime()
        self._time_max = pd.to_datetime(self.data.valid_time.max().item()).to_pydatetime()


    def _datetime_to_time_coord(self, dt: datetime) -> datetime:
        """
        Converts a datetime object to the coordinate type used in the dataset.
        Ensures dt is a datetime object for interpolation.
        """
        # TODO: Add check if dt.date() matches the date of the loaded self.data
        return dt

    def get_wind_components(self, lat: float, lon: float, alt_ft: float, time: datetime, 
                            interpolate_spatial_time: bool = True) -> tuple[float, float]:
        """
        Retrieves u and v wind components for a given location, altitude, and time.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            alt_ft: Altitude in feet.
            time: Datetime object for the query time.
            interpolate_spatial_time: If True (default), use linear interpolation for lat, lon, and time.
                                      If False, use nearest neighbor. Altitude is always linearly interpolated
                                      or clamped to boundary values.

        Returns:
            A tuple (u_component, v_component) in m/s. Returns (np.nan, np.nan) if data cannot be retrieved.
        """
        time_coord = self._datetime_to_time_coord(time)
        method_st = "linear" if interpolate_spatial_time else "nearest"
        alt_m = alt_ft * 0.3048 # Convert feet to meters

        # Clip spatial coordinates to ensure they are within data bounds
        interp_lat = np.clip(lat, self._lat_min, self._lat_max)
        interp_lon = np.clip(lon, self._lon_min, self._lon_max)

        # Clip time coordinate to ensure it is within data bounds
        interp_time = time_clip(time_coord, self._time_min, self._time_max)

        coords_to_interp = {'latitude': interp_lat, 'longitude': interp_lon, 'valid_time': interp_time}

        if alt_m <= self._SURFACE_ALTITUDE_M: # Surface or near-surface winds
            try:
                u_val = self.data['u10'].interp(coords_to_interp, method=method_st, kwargs={"fill_value": None}).item()
                v_val = self.data['v10'].interp(coords_to_interp, method=method_st, kwargs={"fill_value": None}).item()
                return u_val, v_val
            except Exception: # Broad exception if interpolation fails for any reason
                return np.nan, np.nan
        else: # High-altitude winds
            if not self.model_altitudes_at_pressure_levels_m.size:
                # No pressure levels loaded (e.g., 'pressure_level' coord missing or empty)
                print("Warning: No high-altitude pressure levels available. Cannot provide wind components.")
                return np.nan, np.nan

            try:
                # Interpolate u, v spatially and temporally, keeping pressure_level dimension
                u_profile = self.data['u'].interp(coords_to_interp, method=method_st, kwargs={"fill_value": None})
                v_profile = self.data['v'].interp(coords_to_interp, method=method_st, kwargs={"fill_value": None})

                # Select profiles at the model's pressure levels (sorted by altitude)
                u_profile_ordered_values = u_profile.sel(pressure_level=self.pressure_levels_hpa).data.squeeze()
                v_profile_ordered_values = v_profile.sel(pressure_level=self.pressure_levels_hpa).data.squeeze()
            except Exception:
                 return np.nan, np.nan


            # Filter out NaNs that might have resulted from spatial/time interpolation before altitude interpolation
            valid_mask = ~np.isnan(u_profile_ordered_values) & ~np.isnan(v_profile_ordered_values)
            
            current_altitudes = self.model_altitudes_at_pressure_levels_m[valid_mask]
            current_u_values = u_profile_ordered_values[valid_mask]
            current_v_values = v_profile_ordered_values[valid_mask]

            if len(current_altitudes) == 0:
                return np.nan, np.nan # No valid data points after spatial/temporal interpolation
            
            if len(current_altitudes) == 1:
                # Only one valid pressure level/altitude after filtering. Return its values.
                return current_u_values.item() if current_u_values.ndim == 0 else current_u_values[0], \
                       current_v_values.item() if current_v_values.ndim == 0 else current_v_values[0]

            # Perform linear interpolation for altitude using scipy.interpolate.interp1d
            # fill_value=(low_bound_val, high_bound_val) handles extrapolation by clamping to boundary values.
            try:
                u_interpolator = interp1d(current_altitudes, current_u_values,
                                          kind='linear', bounds_error=False,
                                          fill_value=(current_u_values[0], current_u_values[-1]))
                v_interpolator = interp1d(current_altitudes, current_v_values,
                                          kind='linear', bounds_error=False,
                                          fill_value=(current_v_values[0], current_v_values[-1]))
                
                u_final = u_interpolator(alt_m).item()
                v_final = v_interpolator(alt_m).item()
                return u_final, v_final
            except Exception: # If scipy interpolation fails
                return np.nan, np.nan


    def get_cape_cin(self, lat: float, lon: float, time: datetime, 
                     interpolate: bool = True) -> tuple[float, float]:
        """
        Retrieves CAPE and CIN values for a given location and time.
        These are surface variables, so no altitude interpolation is performed.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            time: Datetime object for the query time.
            interpolate: If True (default), use linear interpolation. If False, use nearest neighbor.

        Returns:
            A tuple (cape_value, cin_value). Units are J kg**-1. Returns (np.nan, np.nan) if data cannot be retrieved.
        """
        time_coord = self._datetime_to_time_coord(time)
        method = "linear" if interpolate else "nearest"

        # Clip spatial coordinates to ensure they are within data bounds
        interp_lat = np.clip(lat, self._lat_min, self._lat_max)
        interp_lon = np.clip(lon, self._lon_min, self._lon_max)

        # Clip time coordinate to ensure it is within data bounds
        interp_time = time_clip(time_coord, self._time_min, self._time_max)

        coords_to_interp = {'latitude': interp_lat, 'longitude': interp_lon, 'valid_time': interp_time}
        
        try:
            cape_val = self.data['cape'].interp(coords_to_interp, method=method, kwargs={"fill_value": None}).item()
            # cin_val = self.data['cin'].interp(coords_to_interp, method=method, kwargs={"fill_value": None}).item()
            return cape_val, np.nan
        except Exception: # Broad exception if interpolation fails
            return np.nan, np.nan

if __name__ == '__main__':
    # Example usage (assuming you have a sample data file)
    # Create a dummy NetCDF file for testing if needed
    # For example: create_dummy_era5_netcdf("2024-04-01.nc") in data/era5/
    
    print("WindModel class implemented. Example usage (requires a data file):")
    try:
        # Replace with a valid date for which you have a .nc file
        model_date = "2023-01-01" # Example date
        # Create a dummy file for this example to run without erroring out immediately
        # This part would normally be run if you have a script to generate test data
        
        # --- Minimal dummy file creation for demonstration ---
        def create_dummy_era5_for_testing(date_str, dir_path="data/era5"):
            import os
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path, f"{date_str}.nc")
            if os.path.exists(file_path):
                print(f"Dummy file {file_path} already exists.")
                return

            lat_coords = np.arange(30, 72.1, 0.25) # Example: 30 to 72 N
            lon_coords = np.arange(-15, 41.1, 0.25) # Example: -15 to 41 E
            time_coords = np.arange(0, 24, 1) # 24 hours
            # pressure_levels = np.array([300.0, 200.0, 100.0, 50.0]) # Multiple levels
            pressure_levels = np.array([300.0]) # Single level as per era5.md initially

            data_vars = {
                'u10': (('valid_time', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(lat_coords), len(lon_coords)) * 10),
                'v10': (('valid_time', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(lat_coords), len(lon_coords)) * 10),
                'u': (('valid_time', 'pressure_level', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(pressure_levels), len(lat_coords), len(lon_coords)) * 20),
                'v': (('valid_time', 'pressure_level', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(pressure_levels), len(lat_coords), len(lon_coords)) * 20),
                'cape': (('valid_time', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(lat_coords), len(lon_coords)) * 1000),
                'cin': (('valid_time', 'latitude', 'longitude'), np.random.rand(len(time_coords), len(lat_coords), len(lon_coords)) * -200),
            }
            coords = {
                'valid_time': time_coords,
                'latitude': lat_coords,
                'longitude': lon_coords,
                'pressure_level': pressure_levels
            }
            ds = xr.Dataset(data_vars, coords=coords)
            ds.to_netcdf(file_path)
            print(f"Created dummy file: {file_path}")
            
        # Create the dummy file for the example date if it doesn't exist
        create_dummy_era5_for_testing(model_date)
        # --- End of dummy file creation ---

        wind_model = WindModel(date_str=model_date, data_dir="data/era5")
        
        query_time = datetime(int(model_date[:4]), int(model_date[5:7]), int(model_date[8:10]), 12, 30) # Year, Month, Day, Hour, Minute

        # Test surface wind
        u_sfc, v_sfc = wind_model.get_wind_components(lat=50.0, lon=0.0, alt_m=10.0, time=query_time)
        print(f"Surface wind (10m) at lat=50, lon=0, alt=10m, time={query_time}: u={u_sfc:.2f} m/s, v={v_sfc:.2f} m/s")

        # Test high-altitude wind (e.g., ~9000m which is around 300 hPa)
        # Altitude for 300 hPa is approx 9164m by ISA
        # Altitude for 50 hPa is approx 20576m
        alt_300hpa_approx = wind_model._pressure_to_altitude(np.array([300.0])).item()
        u_high, v_high = wind_model.get_wind_components(lat=50.0, lon=0.0, alt_m=alt_300hpa_approx + 100, time=query_time) # Slightly above 300hpa level
        print(f"High-altitude wind at lat=50, lon=0, alt={alt_300hpa_approx + 100:.0f}m, time={query_time}: u={u_high:.2f} m/s, v={v_high:.2f} m/s")

        u_high_interp, v_high_interp = wind_model.get_wind_components(lat=50.0, lon=0.0, alt_m=15000, time=query_time) # Interpolated altitude
        print(f"High-altitude wind at lat=50, lon=0, alt=15000m, time={query_time}: u={u_high_interp:.2f} m/s, v={v_high_interp:.2f} m/s (Note: result depends on available pressure levels in dummy data)")


        # Test CAPE/CIN
        cape, cin = wind_model.get_cape_cin(lat=50.0, lon=0.0, time=query_time)
        print(f"CAPE/CIN at lat=50, lon=0, time={query_time}: CAPE={cape:.2f} J/kg, CIN={cin:.2f} J/kg")

        # Test edge case interpolation (nearest neighbor)
        u_near, v_near = wind_model.get_wind_components(lat=20.0, lon=-30.0, alt_m=10.0, time=query_time, interpolate_spatial_time=False) # Outside typical bounds from era5.md but clipped
        print(f"Nearest surface wind (10m) at lat=20 (clipped), lon=-30 (clipped), alt=10m: u={u_near:.2f} m/s, v={v_near:.2f} m/s")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure you have the NetCDF data file in the 'data/era5' directory or update the path.")
        print("A dummy file creation attempt was made in the example; check its output.")
    except ValueError as e:
        print(e)
    except ImportError:
        print("This example requires xarray, numpy, and scipy. Please install them.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

