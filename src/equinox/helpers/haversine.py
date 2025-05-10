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
