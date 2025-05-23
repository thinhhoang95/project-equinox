from .wind_model import WindModel
from datetime import datetime
import torch
import numpy as np

class WindFree(WindModel):
    def __init__(self):
        # super().__init__(date_str="2024-04-01", data_dir="data/era5")
        self._time_min = datetime(2024, 4, 1, 0, 0, 0)
        self._time_max = datetime(2024, 4, 1, 23, 59, 59)
        self._lat_min = -90
        self._lat_max = 90
        self._lon_min = -180
        self._lon_max = 180 # these are not important at all for wind-free

        # Initialize attributes that might be expected by WindModel methods
        self._SURFACE_ALTITUDE_M = 10.0  # Standard surface altitude in meters
        self.model_altitudes_at_pressure_levels_m = np.array([], dtype=np.float32)
        self.pressure_levels_hpa = np.array([], dtype=np.float32)
        self.data = None # No data for wind-free model

    def get_wind_components(self, lat, lon, alt_ft, time, interpolate_spatial_time=True):
        return 0, 0

    def get_wind_components_batched(
        self,
        lats_pt: torch.Tensor,
        lons_pt: torch.Tensor, # Unused, but part of the signature
        alts_ft_pt: torch.Tensor, # Unused, but part of the signature
        etas_sec_pt: torch.Tensor # Unused, but part of the signature
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns zero wind components for a batch of locations, altitudes, and times.
        Ensures output is on the same device and dtype as input tensors.
        """
        num_pts = lats_pt.shape[0]
        original_device = lats_pt.device
        original_dtype = lats_pt.dtype

        u_zeros = torch.zeros(num_pts, device=original_device, dtype=original_dtype)
        v_zeros = torch.zeros(num_pts, device=original_device, dtype=original_dtype)
        return u_zeros, v_zeros

    def get_cape_cin(self, lat, lon, time, interpolate=True):
        return 0, 0
