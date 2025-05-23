import torch
import numpy as np
import networkx as nx
from datetime import datetime
from equinox.dp.phasing.forward_dp import run_forward_dp
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_since_midnight_to_datetime
from equinox.cost.cost_rev1 import CostRev1
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE

def forward_heuristic_pass():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")

    node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
    node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)}
    num_actual_nodes = len(node_list_for_matrix)

    # Load the distance matrix
    dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")
    
    # Load the airspace charges matrix
    ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")

    # Load the wind model
    # Consistent with test_forward_dp, using a 2024 date for wind data,
    # while flight times (landing time here) are for 2023.
    wind_model = WindDate(date_str="2024-04-01", data_dir="data/era5")

    # Load the performance model for a typical narrow body jet
    performance_model = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000.0,
        cruise_speed_kts=450.0,
    )

    # Load the cost model
    cost_model_instance = cost_model_1
    
    # Estimated landing time
    heuristic_takeoff_time_str = "2023-04-01 12:00:00" # Similar to takeoff time in forward DP for wind consistency
    heuristic_takeoff_eta = datestr_to_seconds_since_midnight(heuristic_takeoff_time_str) # seconds since midnight

    # Run the forward heuristic pass
    V, active_eta, active_alt, active_phase, active_elapsed_time = run_forward_dp(
        graph=G,
        source_node_id="LEMD",
        takeoff_time_str=heuristic_takeoff_time_str,
        source_elevation_ft=0.0,
        goal_elevation_ft=0.0,
        cost_model=cost_model_instance,
        wind_model=wind_model,
        performance_model=performance_model,
        dist_matrix_np=dist_matrix,
        ac_matrix_np=ac_matrix,
        initial_alt_ft=0.0, # Start at 0ft AMSL if source_elevation_ft is 0 for profile alignment
        delta_t_seconds=600, # time window length
        max_flight_duration_hours=5, # max duration to consider for time bins
        device=device
    )

    return V, active_eta, active_alt, active_phase, active_elapsed_time, node_to_idx_for_matrix



if __name__ == "__main__":
    V, active_eta, active_alt, active_phase, active_elapsed_time, node_to_idx_for_matrix = forward_heuristic_pass()

    print(f'')

    print('Stop here')
