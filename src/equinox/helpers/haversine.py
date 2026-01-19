import math
import numpy as np
import torch

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the haversine formula.
    Returns the distance in nautical miles.
    Supports both scalar inputs (int/float) and array-like/vector inputs.
    """
    # Earth radius in nautical miles
    R_NM = 3440.065

    # Scalar path: all inputs are plain Python numbers
    if isinstance(lat1, (int, float)) and isinstance(lon1, (int, float)) \
       and isinstance(lat2, (int, float)) and isinstance(lon2, (int, float)):
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R_NM * c

    # Vectorized path: at least one input is array-like
    lat1_arr, lon1_arr, lat2_arr, lon2_arr = map(np.asarray, (lat1, lon1, lat2, lon2))
    dlat = np.radians(lat2_arr - lat1_arr)
    dlon = np.radians(lon2_arr - lon1_arr)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1_arr)) *
         np.cos(np.radians(lat2_arr)) *
         np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R_NM * c

def bearing(point1, point2):
    """
    Calculate the initial bearing between two points in degrees.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    initial_bearing = math.atan2(y, x)
    return (math.degrees(initial_bearing) + 360) % 360


def destination_point(point, bearing_deg, distance_nm):
    """
    Calculate destination point given start, bearing, and distance (nm).
    """
    R_NM = 3440.065
    lat1, lon1 = point

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    bearing_rad = math.radians(bearing_deg)

    angular_distance = distance_nm / R_NM

    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(angular_distance)
        + math.cos(lat1_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    lon2_rad = lon1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1_rad),
        math.cos(angular_distance) - math.sin(lat1_rad) * math.sin(lat2_rad),
    )

    return (math.degrees(lat2_rad), math.degrees(lon2_rad))

# Here is a Torch version of the same haversine function
def haversinet(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the haversine formula.
    Returns the distance in nautical miles.
    All inputs must be torch tensors (can be broadcastable shapes).
    """
    R_NM = 3440.065
    # Convert degrees to radians by multiplying by pi/180
    dlat = (lat2 - lat1) * (torch.pi / 180.0)
    dlon = (lon2 - lon1) * (torch.pi / 180.0)
    a = (torch.sin(dlat / 2) ** 2 +
         torch.cos(lat1 * (torch.pi / 180.0)) *
         torch.cos(lat2 * (torch.pi / 180.0)) *
         torch.sin(dlon / 2) ** 2)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R_NM * c
