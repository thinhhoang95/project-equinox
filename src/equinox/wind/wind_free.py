from .wind_model import WindModel

class WindFree(WindModel):
    def __init__(self):
        super().__init__(date_str="2024-04-01", data_dir="data/era5")

    def get_wind_components(self, lat, lon, alt_ft, time, interpolate_spatial_time=True):
        return 0, 0

    def get_cape_cin(self, lat, lon, time, interpolate=True):
        return 0, 0
