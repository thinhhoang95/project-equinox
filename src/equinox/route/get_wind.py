import torch
import math
from equinox.wind.wind_model import WindModel
from datetime import timedelta

def get_wind(
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    altitude: torch.Tensor,
    eta_src: torch.Tensor,
    wind_model: WindModel,
) -> torch.Tensor:
    """Calculates the along-track wind component for a batch of flight segments.

    For each segment defined by source and target coordinates, this function queries
    a wind model to obtain the wind components (eastward and northward) at the
    segment's source location, altitude, and estimated time of arrival. It then
    projects the wind vector onto the segment's track (bearing from source to target)
    to determine the wind speed component acting along the direction of flight.
    A positive value indicates a tailwind (wind assisting forward movement), and a
    negative value indicates a headwind (wind opposing forward movement).

    If wind data is missing for a specific point, the function assumes zero wind
    for that segment.

    Args:
        coords_src (torch.Tensor): Source coordinates for each segment, shape `[E, 2]`. 
                                   Format is (latitude, longitude) in degrees.
        coords_tgt (torch.Tensor): Target coordinates for each segment, shape `[E, 2]`. 
                                   Format is (latitude, longitude) in degrees.
        altitude (torch.Tensor): Altitude for each segment's source point, shape `[E]`. 
                                 Altitude is in feet.
        eta_src (torch.Tensor): Estimated Time of Arrival (ETA) at the source point 
                                for each segment, shape `[E]`. Time is in seconds, 
                                relative to the wind model's minimum time (`wind_model._time_min`).
        wind_model (WindModel): An instance of a WindModel providing wind data.

    Returns:
        torch.Tensor: Along-track wind speed for each segment, shape `[E]`. The speed
                      is in meters per second (m/s). Positive values indicate tailwind,
                      negative values indicate headwind.

    Example:
        >>> import torch
        >>> from equinox.wind.wind_model import WindModel # Assuming WindModel is available
        >>> # Mock WindModel for demonstration
        >>> class MockWindModel(WindModel):
        ...     def __init__(self):
        ...         self._time_min = datetime.timedelta(seconds=0) # Assume start time is epoch
        ...     def get_wind_components(self, lat, lon, alt, query_time):
        ...         # Example: Constant 10 m/s eastward wind (tailwind for eastbound)
        ...         return 10.0, 0.0 # u=10 m/s (east), v=0 m/s (north)
        >>> wind_model = MockWindModel()
        >>> coords_src = torch.tensor([[34.0522, -118.2437], [40.7128, -74.0060]]) # LA, NYC
        >>> coords_tgt = torch.tensor([[36.7783, -119.4179], [41.8781, -87.6298]]) # Near Fresno, Chicago
        >>> altitude = torch.tensor([35000.0, 30000.0]) # in feet
        >>> eta_src = torch.tensor([0.0, 3600.0]) # in seconds
        >>> along_track_wind = get_wind(coords_src, coords_tgt, altitude, eta_src, wind_model)
        >>> print(along_track_wind.shape)
        torch.Size([2])
        >>> print(along_track_wind.dtype == altitude.dtype)
        True
        >>> print(along_track_wind.device == altitude.device)
        True
        # Example output (depends on mock wind model and geometry):
        # tensor([..., ...], dtype=..., device=...)
    """
    # Move data to CPU numpy arrays
    lat_src = coords_src[:, 0].cpu().numpy()
    lon_src = coords_src[:, 1].cpu().numpy()
    lat_tgt = coords_tgt[:, 0].cpu().numpy()
    lon_tgt = coords_tgt[:, 1].cpu().numpy()
    alt_ft = altitude.cpu().numpy()
    eta_sec = eta_src.cpu().numpy()

    wind_along = []
    for la, lo, la2, lo2, altv, eta in zip(
        lat_src, lon_src, lat_tgt, lon_tgt, alt_ft, eta_sec
    ):
        # compute query time relative to model's earliest time
        query_time = wind_model._time_min + timedelta(seconds=float(eta))
        u, v = wind_model.get_wind_components(
            float(la), float(lo), float(altv), query_time
        )
        # if data missing, assume zero wind
        if u is None or v is None or math.isnan(u) or math.isnan(v):
            wind_along.append(0.0)
            continue
        # bearing from source to target (radians, from north clockwise)
        phi1 = math.radians(la)
        phi2 = math.radians(la2)
        dlon = math.radians(lo2 - lo)
        x = math.sin(dlon) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
            phi2
        ) * math.cos(dlon)
        bearing = math.atan2(x, y)
        # unit vector components: east = sin(bearing), north = cos(bearing)
        e = math.sin(bearing)
        n = math.cos(bearing)
        # project wind vector onto track
        wind_along.append(u * e + v * n)
    # return as tensor on original device/dtype
    return torch.tensor(wind_along, dtype=altitude.dtype, device=altitude.device)
