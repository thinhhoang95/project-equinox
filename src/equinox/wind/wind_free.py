from .wind_model import WindModel
from datetime import datetime
class WindFree(WindModel):
    def __init__(self):
        # super().__init__(date_str="2024-04-01", data_dir="data/era5")
        self._time_min = datetime(2024, 4, 1, 0, 0, 0)
        self._time_max = datetime(2024, 4, 1, 23, 59, 59)
        pass # do nothing to speed up testing

    def get_wind_components(self, lat, lon, alt_ft, time, interpolate_spatial_time=True):
        return 0, 0

    def get_cape_cin(self, lat, lon, time, interpolate=True):
        return 0, 0
