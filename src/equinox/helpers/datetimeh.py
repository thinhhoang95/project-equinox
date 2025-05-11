from datetime import datetime, timedelta
import numpy as np
def datestr_to_seconds_since_midnight(datestr: str) -> int:
    # Convert time_at_departure to seconds from midnight
    date_dt = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    seconds_since_midnight = (date_dt - date_dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_since_midnight

def seconds_since_midnight_to_datestr(datestr: str, seconds: int) -> str:
    # Convert seconds from midnight to a datetime object
    try:
        # Try to parse with time component first
        date_obj = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # If that fails, parse with just date component
        date_obj = datetime.strptime(datestr, "%Y-%m-%d")
    
    # Create midnight datetime from the date portion only
    midnight = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    if type(seconds) == np.float32 or type(seconds) == np.float64:
        seconds = int(seconds)
    date_dt = midnight + timedelta(seconds=seconds)
    return date_dt.strftime("%Y-%m-%d %H:%M:%S")

