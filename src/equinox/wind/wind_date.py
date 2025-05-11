from equinox.wind.wind_model import WindModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WindDate(WindModel):
    def __init__(self, date_str: str, data_dir: str = "data/era5"):
        super().__init__(date_str, data_dir)

        # Convert min/max valid_time from numpy datetime64[ns] to Python datetime
        # Set _time_min to midnight of the average date, _time_max to midnight of next day
        # Compute the average valid_time (in nanoseconds since epoch)
        valid_times = self.data.valid_time.values
        avg_time_ns = np.mean(valid_times.astype('datetime64[ns]').astype('int64'))
        avg_time = pd.to_datetime(avg_time_ns)
        avg_date = avg_time.date()
        self._time_min = datetime.combine(avg_date, datetime.min.time())
        self._time_max = self._time_min + timedelta(days=1)
