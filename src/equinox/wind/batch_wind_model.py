from equinox.wind.wind_model import WindModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import os
import xarray as xr
import torch


def get_flight_batches(routes_csv_path: str, batch_size: int) -> list[pd.DataFrame]:
    """
    Reads flight routes from a CSV and divides them into batches,
    preserving the original order from the file.

    Args:
        routes_csv_path (str): Path to the CSV file with flight routes.
        batch_size (int): The maximum number of flights per batch.

    Returns:
        list[pd.DataFrame]: A list of DataFrames, where each DataFrame is a batch.
    """
    df = pd.read_csv(routes_csv_path)
    
    batches = []
    num_flights = len(df)
    for i in range(0, num_flights, batch_size):
        batches.append(df.iloc[i:i+batch_size])
    return batches

def get_typical_flight_duration(routes_csv_path: str, percentile: float = 0.95) -> float:
    """
    Calculates the typical flight duration in seconds from the routes CSV.

    Args:
        routes_csv_path (str): Path to the CSV file with flight routes.
        percentile (float): The percentile to use for "typical" duration.

    Returns:
        float: The typical flight duration in seconds.
    """
    df = pd.read_csv(routes_csv_path)
    if 'flight_time_s' not in df.columns:
        raise ValueError("CSV must contain 'flight_time_s' column.")
    return df['flight_time_s'].quantile(percentile)


class BatchWindModel(WindModel):
    """
    A wind model that handles batches of flights or a single flight,
    loading the necessary time-disparate ERA5 data snapshots. It inherits
    from WindModel to reuse interpolation logic but overrides the data loading part.
    """
    def __init__(
        self,
        typical_flight_duration_s: float,
        flight_batch_df: pd.DataFrame = None,
        takeoff_time: datetime = None,
        data_dir: str = "data/era5",
        time_buffer_h: float = 1.0
    ):
        """
        Initializes the BatchWindModel by loading and preprocessing data
        for a specific batch of flights or a single flight.

        Args:
            typical_flight_duration_s: The typical duration of a flight in seconds.
            flight_batch_df: A pandas DataFrame containing the flights in this batch.
                             Must contain 'takeoff' column with Unix timestamps.
                             Mutually exclusive with takeoff_time.
            takeoff_time: The takeoff time for a single flight.
                          Mutually exclusive with flight_batch_df.
            data_dir: Directory containing the NetCDF data files.
                      Files are expected to be named '{unix_timestamp}.nc'.
            time_buffer_h: A buffer in hours to add before the earliest and after
                           the latest time to ensure data coverage for interpolation.
        """
        if flight_batch_df is None and takeoff_time is None:
            raise ValueError("Either flight_batch_df or takeoff_time must be provided.")
        if flight_batch_df is not None and takeoff_time is not None:
            raise ValueError("Provide either flight_batch_df or takeoff_time, not both.")

        # 1. Find all available NetCDF files and their timestamps
        all_files_paths = glob.glob(os.path.join(data_dir, "*.nc"))
        if not all_files_paths:
            raise FileNotFoundError(f"No NetCDF files found in directory: {data_dir}")

        file_timestamps_map = {}
        for f_path in all_files_paths:
            try:
                # Extract date from filename like '2023-04-01.nc'
                filename = os.path.splitext(os.path.basename(f_path))[0]
                # Parse the date and convert to unix timestamp
                date_obj = datetime.strptime(filename, '%Y-%m-%d')
                timestamp = int(date_obj.timestamp())
                file_timestamps_map[timestamp] = f_path
            except (ValueError, IndexError):
                # Ignore files that do not have a valid date format as a name
                continue

        if not file_timestamps_map:
            raise FileNotFoundError(f"No valid timestamped NetCDF files found in {data_dir}")

        sorted_timestamps = sorted(file_timestamps_map.keys())
        sorted_timestamps_np = np.array(sorted_timestamps)

        # 2. Determine the required minimal set of files
        required_file_paths = set()
        buffer_s = time_buffer_h * 3600

        takeoff_times_unix = []
        if flight_batch_df is not None:
            if 'takeoff' not in flight_batch_df.columns:
                raise ValueError("flight_batch_df must contain a 'takeoff' column.")
            if not isinstance(flight_batch_df, pd.DataFrame) or flight_batch_df.empty:
                raise ValueError("flight_batch_df must be a non-empty pandas DataFrame.")
            takeoff_times_unix = flight_batch_df['takeoff'].tolist()
        else:  # takeoff_time is not None
            takeoff_times_unix = [takeoff_time.timestamp()]


        for takeoff_unix in takeoff_times_unix:
            # Determine time window for this specific flight
            start_time_unix = takeoff_unix - buffer_s
            end_time_unix = takeoff_unix + typical_flight_duration_s + buffer_s

            # Find file indices that bracket this flight's time window.
            # We need the file immediately before the start and after the end for interpolation.
            start_idx = np.searchsorted(sorted_timestamps_np, start_time_unix, side='right')
            start_idx = max(0, start_idx - 1)  # Get file before or at the start time

            end_idx = np.searchsorted(sorted_timestamps_np, end_time_unix, side='left')
            end_idx = min(end_idx, len(sorted_timestamps_np) - 1) # Get file at or after the end time

            # Add all files within this range to our set of required files
            for i in range(start_idx, end_idx + 1):
                ts = sorted_timestamps[i]
                required_file_paths.add(file_timestamps_map[ts])

        files_to_load = sorted(list(required_file_paths))

        if not files_to_load:
            min_t = min(takeoff_times_unix)
            max_t = max(takeoff_times_unix)
            min_t_dt = datetime.fromtimestamp(min_t)
            max_t_dt = datetime.fromtimestamp(max_t + typical_flight_duration_s)
            raise FileNotFoundError(f"No suitable data files found for flight(s) "
                                    f"(covering time range approx. {min_t_dt} to {max_t_dt}) in {data_dir}")

        # 3. Load data using xarray.open_mfdataset
        try:
            # combine='by_coords' is robust for time series, especially non-contiguous ones.
            self.data = xr.open_mfdataset(files_to_load, combine='by_coords')
        except Exception as e:
            raise IOError(f"Failed to load or combine NetCDF files: {files_to_load}. Error: {e}")

        # 4. Run post-initialization steps from parent WindModel
        required_vars = ['u10', 'v10', 'u', 'v', 'cape']
        for var in required_vars:
            if var not in self.data:
                raise ValueError(f"Variable '{var}' not found in the combined dataset.")
        
        self.data = self.data.sortby("latitude")
        self.data = self.data.sortby("longitude")
        self.data = self.data.sortby("valid_time")

        if 'pressure_level' in self.data['u'].coords:
            pressure_levels_hpa_orig = self.data['u'].pressure_level.sortby(self.data['u'].pressure_level, ascending=False).values
            model_altitudes_m = self._pressure_to_altitude(pressure_levels_hpa_orig)
            sort_indices = np.argsort(model_altitudes_m)
            self.pressure_levels_hpa = pressure_levels_hpa_orig[sort_indices]
            self.model_altitudes_at_pressure_levels_m = model_altitudes_m[sort_indices]

            if len(self.pressure_levels_hpa) < 2:
                print(f"Warning: Only {len(self.pressure_levels_hpa)} pressure level(s) found.")
        else:
            print("Warning: 'pressure_level' coordinate not found for 'u' component.")
            self.pressure_levels_hpa = np.array([])
            self.model_altitudes_at_pressure_levels_m = np.array([])
        
        self._lat_min, self._lat_max = self.data.latitude.min().item(), self.data.latitude.max().item()
        self._lon_min, self._lon_max = self.data.longitude.min().item(), self.data.longitude.max().item()
        self._time_min = pd.to_datetime(self.data.valid_time.min().item()).to_pydatetime()
        self._time_max = pd.to_datetime(self.data.valid_time.max().item()).to_pydatetime()

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

        This method overrides the parent implementation to correctly handle absolute
        time calculations required when data spans multiple days, as is common
        for BatchWindModel.

        Args:
            transitions (list[tuple]): A list of state transitions.
            node_coords_deg (torch.Tensor): A tensor of shape [num_nodes, 2] with lat/lon.
            min_wall_clock_time_sec (float): The absolute start time as a Unix timestamp for k_idx=0.
            delta_t_wall_clock_sec (float): The duration of each time bin in seconds.
            num_integration_steps (int): The number of points to sample along each edge.

        Returns:
            torch.Tensor: A tensor of shape [num_transitions] with the
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

        # 2. Get start/end coordinates
        u_coords = node_coords_deg[u_indices]
        v_coords = node_coords_deg[v_indices]

        # 3. Calculate bearing unit vector for each transition
        u_lats_rad = torch.deg2rad(u_coords[:, 0])
        v_lats_rad = torch.deg2rad(v_coords[:, 0])
        dlon_rad = torch.deg2rad(v_coords[:, 1] - u_coords[:, 1])
        x_bearing = torch.sin(dlon_rad) * torch.cos(v_lats_rad)
        y_bearing = torch.cos(u_lats_rad) * torch.sin(v_lats_rad) - torch.sin(u_lats_rad) * torch.cos(v_lats_rad) * torch.cos(dlon_rad)
        bearing_rad = torch.atan2(x_bearing, y_bearing)
        e_track = torch.sin(bearing_rad)
        n_track = torch.cos(bearing_rad)

        # 4. Generate integration points
        if num_integration_steps == 1:
            weights = torch.tensor([0.5], device=original_device, dtype=original_dtype)
        else:
            weights = torch.linspace(0, 1, num_integration_steps, device=original_device, dtype=original_dtype)

        # Interpolate spatial coordinates
        interp_coords = u_coords.unsqueeze(1) * (1 - weights.view(1, -1, 1)) + \
                        v_coords.unsqueeze(1) * weights.view(1, -1, 1)
        interp_lats = interp_coords[:, :, 0]
        interp_lons = interp_coords[:, :, 1]
        
        # Interpolate altitudes
        u_alts_pt = torch.tensor(u_alts_ft, device=original_device, dtype=original_dtype)
        v_alts_pt = torch.tensor(v_alts_ft, device=original_device, dtype=original_dtype)
        interp_alts_ft = u_alts_pt.unsqueeze(1) * (1 - weights) + v_alts_pt.unsqueeze(1) * weights
        
        # --- Corrected ETA Calculation ---
        # Calculate absolute query times as Unix timestamps
        u_times_unix = min_wall_clock_time_sec + torch.tensor(k_u_indices, device=original_device, dtype=original_dtype) * delta_t_wall_clock_sec
        v_times_unix = min_wall_clock_time_sec + torch.tensor(k_v_indices, device=original_device, dtype=original_dtype) * delta_t_wall_clock_sec

        # Get the start time of the loaded data as a Unix timestamp
        time_min_unix = self._time_min.timestamp()

        # Calculate ETAs in seconds relative to the start of the loaded data, as
        # expected by get_wind_components_batched.
        u_etas = u_times_unix - time_min_unix
        v_etas = v_times_unix - time_min_unix
        interp_etas = u_etas.unsqueeze(1) * (1 - weights) + v_etas.unsqueeze(1) * weights
        # --- End of Correction ---

        # 5. Get wind components for all flattened points
        u_components, v_components = self.get_wind_components_batched(
            interp_lats.flatten(), interp_lons.flatten(), interp_alts_ft.flatten(), interp_etas.flatten()
        )

        u_unflattened = u_components.view(num_transitions, num_integration_steps)
        v_unflattened = v_components.view(num_transitions, num_integration_steps)
        
        # 6. Project wind onto track for each integration point
        tailwind_mps_per_point = u_unflattened * e_track.unsqueeze(1) + v_unflattened * n_track.unsqueeze(1)

        # 7. Average tailwind and convert to knots
        avg_tailwind_mps = torch.nan_to_num(tailwind_mps_per_point, nan=0.0).mean(axis=1)
        avg_tailwind_knots = avg_tailwind_mps * MPS_TO_KNOTS

        return avg_tailwind_knots.to(device=original_device, dtype=original_dtype)
