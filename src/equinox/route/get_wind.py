import torch
from equinox.wind.wind_model import WindModel
from datetime import timedelta

def get_wind(
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    altitude: torch.Tensor,
    eta_src: torch.Tensor,
    wind_model: WindModel,
) -> torch.Tensor:
    """Calculates the along-track wind component for a batch of flight segments using vectorized operations.

    For each segment defined by source and target coordinates, this function queries
    a wind model to obtain the wind components (eastward and northward) at the
    segment's source location, altitude, and estimated time of arrival. It then
    projects the wind vector onto the segment's track (bearing from source to target)
    to determine the wind speed component acting along the direction of flight.
    A positive value indicates a tailwind (wind assisting forward movement), and a
    negative value indicates a headwind (wind opposing forward movement).

    If wind data is missing for a specific point (propagated as NaNs from wind_model), 
    the function will convert these NaNs to zero wind for that segment.

    Args:
        coords_src (torch.Tensor): Source coordinates for each segment, shape `[E, 2]`. 
                                   Format is (latitude, longitude) in degrees.
        coords_tgt (torch.Tensor): Target coordinates for each segment, shape `[E, 2]`. 
                                   Format is (latitude, longitude) in degrees.
        altitude (torch.Tensor): Altitude for each segment's source point, shape `[E]`. 
                                 Altitude is in feet.
        eta_src (torch.Tensor): Estimated Time of Arrival (ETA) at the source point 
                                for each segment, shape `[E]`. Time is in seconds, 
                                interpretable by the wind_model (e.g., relative to _time_min or absolute).
        wind_model (WindModel): An instance of a WindModel providing wind data, 
                                expected to have a `get_wind_components_batched` method.

    Returns:
        torch.Tensor: Along-track wind speed for each segment, shape `[E]`. The speed
                      is in meters per second (m/s). Positive values indicate tailwind,
                      negative values indicate headwind.

    Example:
        >>> import torch
        >>> from equinox.wind.wind_model import WindModel # Assuming WindModel is available
        >>> from datetime import datetime, timedelta
        >>> # Mock WindModel for demonstration
        >>> class MockWindModel(WindModel):
        ...     def __init__(self, date_str="2023-01-01", data_dir="dummy"): # Satisfy WindModel constructor
        ...         self._time_min = pd.Timestamp("2023-01-01T00:00:00").to_pydatetime() # Example, pandas might not be imported here
        ...         # Minimal init to avoid errors. A real WindModel instance would be passed.
        ...         pass 
        ...     def get_wind_components_batched(self, lats_pt, lons_pt, alts_ft_pt, etas_sec_pt):
        ...         # Example: Constant 10 m/s eastward wind (tailwind for eastbound)
        ...         # Ensure output is on the same device and dtype as inputs
        ...         u = torch.full_like(lats_pt, 10.0, device=lats_pt.device, dtype=lats_pt.dtype)
        ...         v = torch.zeros_like(lats_pt, device=lats_pt.device, dtype=lats_pt.dtype)
        ...         return u, v
        >>> # Need pandas for MockWindModel internal _time_min if it was used directly.
        >>> # For this example, let's assume it's handled or not strictly needed for the mock's batched call.
        >>> # import pandas as pd # Would be needed if MockWindModel used pd.Timestamp
        >>>
        >>> wind_model_instance = MockWindModel()
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> dtype = torch.float32
        >>> coords_src = torch.tensor([[34.0522, -118.2437], [40.7128, -74.0060]], device=device, dtype=dtype) # LA, NYC
        >>> coords_tgt = torch.tensor([[36.7783, -119.4179], [41.8781, -87.6298]], device=device, dtype=dtype) # Near Fresno, Chicago
        >>> altitude_ft = torch.tensor([35000.0, 30000.0], device=device, dtype=dtype) # in feet
        >>> eta_seconds = torch.tensor([0.0, 3600.0], device=device, dtype=dtype) # in seconds
        >>>
        >>> along_track_wind = get_wind(coords_src, coords_tgt, altitude_ft, eta_seconds, wind_model_instance)
        >>> print(along_track_wind.shape)
        torch.Size([2])
        >>> print(along_track_wind.dtype == altitude_ft.dtype)
        True
        >>> print(along_track_wind.device == altitude_ft.device)
        True
        # Expected output depends on mock model (e.g., positive for eastbound if u=10, v=0)
        # print(along_track_wind) 
    """
    # Ensure all inputs are on the same device and have compatible dtypes.
    # The device and dtype of `altitude` will be used for the output tensor.
    # It's assumed coords_src, coords_tgt, altitude, eta_src are already on the correct device.

    lat_src = coords_src[:, 0]
    lon_src = coords_src[:, 1]
    lat_tgt = coords_tgt[:, 0]
    lon_tgt = coords_tgt[:, 1]

    # Calculate bearing (vectorized)
    # Convert degrees to radians
    phi1 = torch.deg2rad(lat_src)
    phi2 = torch.deg2rad(lat_tgt)
    dlon = torch.deg2rad(lon_tgt - lon_src)

    # Bearing calculation components
    x_bearing = torch.sin(dlon) * torch.cos(phi2)
    y_bearing = torch.cos(phi1) * torch.sin(phi2) - torch.sin(phi1) * torch.cos(phi2) * torch.cos(dlon)
    
    # Bearing in radians from North, clockwise
    bearing_rad = torch.atan2(x_bearing, y_bearing)

    # Unit vector components for the track direction
    # East component: sin(bearing)
    # North component: cos(bearing)
    e_track = torch.sin(bearing_rad)
    n_track = torch.cos(bearing_rad)

    # Get wind components (u, v) from the wind model using the new batched method
    # It's assumed wind_model.get_wind_components_batched will handle eta_src (seconds tensor)
    # and altitude (in feet) correctly, and return u, v in m/s on the same device/dtype.
    u_wind_mps, v_wind_mps = wind_model.get_wind_components_batched(
        lat_src, lon_src, altitude, eta_src
    )

    # Project wind vector onto the track
    # wind_along_track = u_wind * e_track + v_wind * n_track
    wind_along_track_mps = u_wind_mps * e_track + v_wind_mps * n_track
    
    # Handle cases where wind data might be missing (NaNs returned by model)
    # Original code appended 0.0 if u or v was None or NaN.
    # torch.nan_to_num will replace NaNs with 0.0.
    wind_along_track_mps = torch.nan_to_num(wind_along_track_mps, nan=0.0)

    return wind_along_track_mps.to(dtype=altitude.dtype, device=altitude.device)
