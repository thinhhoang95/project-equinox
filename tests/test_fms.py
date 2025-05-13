from typing import Tuple, List
import networkx as nx
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.route.forward_state import get_next_state_fw
from equinox.route.backward_state import get_next_state_bw
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
from equinox.wind.wind_model import WindModel
from equinox.wind.wind_date import WindDate
from datetime import datetime
from equinox.helpers.haversine import haversinet # torch version
from equinox.helpers.datetimeh import seconds_to_hhmmss
import time
from equinox.route.fms import get_4d_trajectory

def test_fms():
    # Load the airac graph
    time_start = time.time()
    airac_graph = nx.read_gml("data/wp/airac_2502.gml")
    print(f"Time taken to load the airac graph: {time.time() - time_start} seconds")

    # Prepare the performance tables
    performance = Performance(
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    # Get the climb_performance table (the time and distance profiles)
    climb_performance_table = get_eta_and_distance_climb(performance, 1000) # return [(altitude (ft), eta (s), along_track_distance (nm))]

    # Get the descent_performance table
    descent_performance_table = get_eta_and_distance_descent(performance, 1000) # ADDED

    # Load the wind_model
    wind_model = WindDate("2024-04-01", "data/era5")

    # Waypoint string
    fpl_str = "LFBO FISTO POI PEPAX NIMER KEPER LUMAN ROMGO NERKI BANOX LFPG"

    # Take-off time
    takeoff_time = "2024-04-01 06:00:00"
    takeoff_time_s = datestr_to_seconds_since_midnight(takeoff_time)

    time_start = time.time()
    # Get the 4D trajectory
    flight_plan_phase_fw, flight_plan_eta_fw, flight_plan_dist_fw, flight_plan_alt_ft_fw = get_4d_trajectory(
        fpl_str, airac_graph, 1000, 1000, takeoff_time_s, 
        climb_performance_table, 
        descent_performance_table, # ADDED
        wind_model
    )
    print(f"Time taken to get the 4D trajectory: {time.time() - time_start} seconds")
    # Format and print the results as a table
    import pandas as pd

    # Convert tensors to numpy arrays
    phase_np = flight_plan_phase_fw.cpu().numpy()
    eta_np = flight_plan_eta_fw.cpu().numpy()
    dist_np = flight_plan_dist_fw.cpu().numpy()
    alt_np = flight_plan_alt_ft_fw.cpu().numpy()

    # Map phase integers to names
    phase_map = {0: "CLIMB", 1: "CRUISE", 2: "DESCENT"}
    phase_str = [phase_map.get(int(p), str(int(p))) for p in phase_np]

    # Build DataFrame
    df = pd.DataFrame({
        "Waypoint #": range(len(phase_np)),
        "Phase": phase_str,
        # "ETA (s)": eta_np,
        "TTLT": [seconds_to_hhmmss(e) for e in eta_np],
        "Distance (nm)": dist_np,
        "Altitude (ft)": alt_np,
    })

    # Format and print
    print(df.to_string(index=False, float_format="%.2f"))