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

def seconds_since_midnight_to_datetime(datestr: str, seconds: int) -> datetime:
    """
    Converts seconds since midnight for a given date string to a datetime object.

    Args:
        datestr (str): The base date string (e.g., "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS").
        seconds (int): The number of seconds since midnight of that day.

    Returns:
        datetime: The resulting datetime object.
    """
    try:
        # Try to parse with time component first
        date_obj = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # If that fails, parse with just date component
        date_obj = datetime.strptime(datestr, "%Y-%m-%d")
    
    # Create midnight datetime from the date portion only
    midnight = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    if type(seconds) == np.float32 or type(seconds) == np.float64:
        seconds = int(seconds) # Ensure seconds is int for timedelta
    
    final_datetime = midnight + timedelta(seconds=seconds)
    return final_datetime

def seconds_to_hhmmss(seconds: float) -> str:
    """
    Convert seconds (int or float) to HH:mm:ss string format.
    """
    if isinstance(seconds, float):
        seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
