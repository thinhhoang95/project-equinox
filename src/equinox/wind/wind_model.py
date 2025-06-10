import xarray as xr
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd
import torch

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

    def get_wind_components_batched(
        self, 
        lats_pt: torch.Tensor, 
        lons_pt: torch.Tensor, 
        alts_ft_pt: torch.Tensor, 
        etas_sec_pt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves u and v wind components for a batch of locations, altitudes, and times.

        Args:
            lats_pt (torch.Tensor): Latitudes in degrees, shape [N].
            lons_pt (torch.Tensor): Longitudes in degrees, shape [N].
            alts_ft_pt (torch.Tensor): Altitudes in feet, shape [N].
            etas_sec_pt (torch.Tensor): ETA in seconds relative to self._time_min, shape [N].

        Returns:
            A tuple of PyTorch Tensors (u_components_mps, v_components_mps), both shape [N],
            on the same device and dtype as input tensors.
            Returns (nan, nan) for points where data cannot be retrieved.
        """
        original_device = lats_pt.device
        original_dtype = lats_pt.dtype

        # Convert PyTorch tensors to NumPy arrays
        lats_np = lats_pt.cpu().numpy()
        lons_np = lons_pt.cpu().numpy()
        alts_ft_np = alts_ft_pt.cpu().numpy()
        etas_sec_np = etas_sec_pt.cpu().numpy()

        num_pts = lats_np.shape[0]
        alts_m_np = alts_ft_np * 0.3048

        # Initialize output arrays
        u_final_np = np.full(num_pts, np.nan, dtype=np.float32) # Use float32 for np calculations
        v_final_np = np.full(num_pts, np.nan, dtype=np.float32)

        # Convert ETA seconds to numpy.datetime64 array
        # self._time_min is a python datetime, convert to datetime64
        time_min_np = np.datetime64(self._time_min)
        query_datetimes_np = time_min_np + etas_sec_np.astype('timedelta64[s]')
        
        # Clip all coordinates
        interp_lats_np = np.clip(lats_np, self._lat_min, self._lat_max)
        interp_lons_np = np.clip(lons_np, self._lon_min, self._lon_max)
        
        # Ensure self.data.valid_time.min/max are numpy.datetime64 for comparison
        data_time_min_np = self.data.valid_time.min().values
        data_time_max_np = self.data.valid_time.max().values
        interp_times_np = np.clip(query_datetimes_np, data_time_min_np, data_time_max_np)

        # Create xarray DataArrays for coordinates for interpolation
        # These will be used for indexing into the xarray dataset
        xr_lats = xr.DataArray(interp_lats_np, dims="points")
        xr_lons = xr.DataArray(interp_lons_np, dims="points")
        xr_times = xr.DataArray(interp_times_np, dims="points")
        
        # --- Handle Surface Winds (altitude <= _SURFACE_ALTITUDE_M) ---
        surface_mask = alts_m_np <= self._SURFACE_ALTITUDE_M
        if np.any(surface_mask):
            coords_sfc = {
                'latitude': xr_lats[surface_mask],
                'longitude': xr_lons[surface_mask],
                'valid_time': xr_times[surface_mask]
            }
            try:
                u10_vals = self.data['u10'].interp(coords_sfc, method="linear", kwargs={"fill_value": np.nan}).data
                v10_vals = self.data['v10'].interp(coords_sfc, method="linear", kwargs={"fill_value": np.nan}).data
                u_final_np[surface_mask] = u10_vals
                v_final_np[surface_mask] = v10_vals
            except Exception: # Broad exception for safety
                 # Already initialized to NaN, so just pass
                pass 

        # --- Handle High-Altitude Winds (altitude > _SURFACE_ALTITUDE_M) ---
        high_alt_mask = alts_m_np > self._SURFACE_ALTITUDE_M
        if np.any(high_alt_mask):
            if not self.model_altitudes_at_pressure_levels_m.size:
                # No pressure levels, u_final_np[high_alt_mask] and v_final_np[high_alt_mask] remain NaN
                pass
            else:
                coords_high = {
                    'latitude': xr_lats[high_alt_mask],
                    'longitude': xr_lons[high_alt_mask],
                    'valid_time': xr_times[high_alt_mask]
                }
                try:
                    # Interpolate u, v spatially and temporally, keeping pressure_level dimension
                    # .data converts to numpy array. Shape: [num_high_alt_pts, num_pressure_levels]
                    u_profiles_batch = self.data['u'].interp(coords_high, method="linear", kwargs={"fill_value": np.nan}).sel(pressure_level=self.pressure_levels_hpa).data
                    v_profiles_batch = self.data['v'].interp(coords_high, method="linear", kwargs={"fill_value": np.nan}).sel(pressure_level=self.pressure_levels_hpa).data

                    target_alts_m_for_high = alts_m_np[high_alt_mask]
                    
                    u_interp_for_high = np.full(target_alts_m_for_high.shape[0], np.nan, dtype=np.float32)
                    v_interp_for_high = np.full(target_alts_m_for_high.shape[0], np.nan, dtype=np.float32)

                    for i in range(target_alts_m_for_high.shape[0]):
                        current_u_profile_values = u_profiles_batch[i, :]
                        current_v_profile_values = v_profiles_batch[i, :]
                        
                        # Filter out NaNs that might have resulted from spatial/time interpolation
                        valid_prof_mask = ~np.isnan(current_u_profile_values) & ~np.isnan(current_v_profile_values)
                        
                        current_altitudes_for_interp = self.model_altitudes_at_pressure_levels_m[valid_prof_mask]
                        current_u_values_for_interp = current_u_profile_values[valid_prof_mask]
                        current_v_values_for_interp = current_v_profile_values[valid_prof_mask]

                        if len(current_altitudes_for_interp) == 0:
                            continue # Remains NaN
                        if len(current_altitudes_for_interp) == 1:
                            u_interp_for_high[i] = current_u_values_for_interp[0]
                            v_interp_for_high[i] = current_v_values_for_interp[0]
                            continue
                        
                        try:
                            u_interpolator = interp1d(current_altitudes_for_interp, current_u_values_for_interp,
                                                      kind='linear', bounds_error=False,
                                                      fill_value=(current_u_values_for_interp[0], current_u_values_for_interp[-1]))
                            v_interpolator = interp1d(current_altitudes_for_interp, current_v_values_for_interp,
                                                      kind='linear', bounds_error=False,
                                                      fill_value=(current_v_values_for_interp[0], current_v_values_for_interp[-1]))
                            
                            u_interp_for_high[i] = u_interpolator(target_alts_m_for_high[i])
                            v_interp_for_high[i] = v_interpolator(target_alts_m_for_high[i])
                        except Exception: # If scipy interpolation fails for a point
                            # Remains NaN
                            pass
                    
                    u_final_np[high_alt_mask] = u_interp_for_high
                    v_final_np[high_alt_mask] = v_interp_for_high
                except Exception: # Broad exception for xarray interp or sel
                    # u_final_np[high_alt_mask] etc remain NaN
                    pass

        # Convert final NumPy arrays back to PyTorch tensors on the original device and dtype
        u_torch = torch.from_numpy(u_final_np).to(device=original_device, dtype=original_dtype)
        v_torch = torch.from_numpy(v_final_np).to(device=original_device, dtype=original_dtype)
        
        return u_torch, v_torch

    def get_average_tailwind_on_edges_knots(
        self,
        transitions: list[tuple],
        node_coords_deg: torch.Tensor,
        min_wall_clock_time_sec: float,
        delta_t_wall_clock_sec: float,
        num_integration_steps: int = 3,
    ) -> torch.Tensor:
        """
        Computes the average tailwind component in knots over a list of flight edges.

        For each transition, it samples points along the great-circle-approximated path
        in space and time, queries the wind at these points, projects the wind onto the
        flight path to get the tailwind at each sample point, and returns the
        averaged tailwind for each transition.

        Args:
            transitions (list[tuple]): A list of state transitions. Each tuple is expected
                to be in the format: (u_idx, k_u_idx, rho_u_idx, u_alt_ft, phase_u,
                                      v_idx, k_v_idx, rho_v_idx, v_alt_ft, phase_v).
            node_coords_deg (torch.Tensor): A tensor of shape [num_nodes, 2] containing
                the latitude and longitude for each node index.
            min_wall_clock_time_sec (float): The absolute start time for k_idx=0.
            delta_t_wall_clock_sec (float): The duration of each wall-clock time bin in seconds.
            num_integration_steps (int): The number of points to sample along each edge.

        Returns:
            torch.Tensor: A tensor of shape [num_transitions] containing the
                averaged tailwind in knots for each transition.
        """
        if not transitions:
            return torch.empty((0,), device=node_coords_deg.device)
        
        if num_integration_steps < 1:
            raise ValueError("num_integration_steps must be at least 1.")

        original_device = node_coords_deg.device
        original_dtype = node_coords_deg.dtype
        MPS_TO_KNOTS = 1.94384

        # 1. Unpack transitions
        transitions_np = np.array(transitions, dtype=object)
        u_indices = transitions_np[:, 0].astype(int)
        k_u_indices = transitions_np[:, 1].astype(int)
        u_alts_ft = transitions_np[:, 3].astype(float)
        v_indices = transitions_np[:, 5].astype(int)
        k_v_indices = transitions_np[:, 6].astype(int)
        v_alts_ft = transitions_np[:, 8].astype(float)

        num_transitions = len(transitions)

        # 2. Get start/end coordinates and times
        u_coords = node_coords_deg[u_indices]
        v_coords = node_coords_deg[v_indices]

        u_lats_rad = torch.deg2rad(u_coords[:, 0])
        v_lats_rad = torch.deg2rad(v_coords[:, 0])
        dlon_rad = torch.deg2rad(v_coords[:, 1] - u_coords[:, 1])

        # 3. Calculate bearing unit vector for each transition
        x_bearing = torch.sin(dlon_rad) * torch.cos(v_lats_rad)
        y_bearing = torch.cos(u_lats_rad) * torch.sin(v_lats_rad) - torch.sin(u_lats_rad) * torch.cos(v_lats_rad) * torch.cos(dlon_rad)
        bearing_rad = torch.atan2(x_bearing, y_bearing)
        e_track = torch.sin(bearing_rad) # East component of track vector
        n_track = torch.cos(bearing_rad) # North component of track vector

        # 4. Generate integration points
        if num_integration_steps == 1:
            weights = torch.tensor([0.5], device=original_device, dtype=original_dtype)
        else:
            weights = torch.linspace(0, 1, num_integration_steps, device=original_device, dtype=original_dtype)

        # Use broadcasting to create sample points
        u_coords_expanded = u_coords.unsqueeze(1)
        v_coords_expanded = v_coords.unsqueeze(1)
        weights_expanded = weights.view(1, -1, 1)

        interp_coords = u_coords_expanded * (1 - weights.view(1,-1))[:,:,None] + v_coords_expanded * weights.view(1,-1)[:,:,None]
        interp_lats = interp_coords[:, :, 0]
        interp_lons = interp_coords[:, :, 1]
        
        u_alts_pt = torch.tensor(u_alts_ft, device=original_device, dtype=original_dtype)
        v_alts_pt = torch.tensor(v_alts_ft, device=original_device, dtype=original_dtype)
        interp_alts_ft = u_alts_pt.unsqueeze(1) * (1 - weights) + v_alts_pt.unsqueeze(1) * weights
        
        time_min_sec_midnight = self._time_min.hour * 3600 + self._time_min.minute * 60 + self._time_min.second
        u_times_sec_midnight = min_wall_clock_time_sec + torch.tensor(k_u_indices, device=original_device, dtype=original_dtype) * delta_t_wall_clock_sec
        v_times_sec_midnight = min_wall_clock_time_sec + torch.tensor(k_v_indices, device=original_device, dtype=original_dtype) * delta_t_wall_clock_sec
        u_etas = u_times_sec_midnight - time_min_sec_midnight
        v_etas = v_times_sec_midnight - time_min_sec_midnight
        interp_etas = u_etas.unsqueeze(1) * (1 - weights) + v_etas.unsqueeze(1) * weights

        # 5. Get wind components for all flattened points
        u_components, v_components = self.get_wind_components_batched(
            interp_lats.flatten(), interp_lons.flatten(), interp_alts_ft.flatten(), interp_etas.flatten()
        )

        u_unflattened = u_components.view(num_transitions, num_integration_steps)
        v_unflattened = v_components.view(num_transitions, num_integration_steps)
        
        # 6. Project wind onto track for each integration point
        # e_track and n_track have shape [num_transitions], need to expand for broadcasting
        tailwind_mps_per_point = u_unflattened * e_track.unsqueeze(1) + v_unflattened * n_track.unsqueeze(1)

        # 7. Average tailwind and convert to knots
        avg_tailwind_mps = torch.nan_to_num(tailwind_mps_per_point, nan=0.0).mean(axis=1)
        avg_tailwind_knots = avg_tailwind_mps * MPS_TO_KNOTS

        return avg_tailwind_knots.to(device=original_device, dtype=original_dtype)

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

