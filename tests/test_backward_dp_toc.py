# Load the transitions from the file
import pickle 
transitions_list = pickle.load(open("../data/graph/LEMD_EGLL_2023_04_01_climb_transitions.pkl", "rb")) # from test_toc_forward_dp.ipynb, to be rewritten into a more comprehensive package
print(f"Transitions list loaded with {len(transitions_list)} transitions")

import networkx as nx
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.wind.wind_free import WindFree
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
import numpy as np
import torch

# Load the route graph
G = nx.read_gml("../data/graph/LEMD_EGLL_2023_04_01.gml")
node_to_idx = {node: i for i, node in enumerate(G.nodes())}
idx_to_node = {i: node for i, node in enumerate(G.nodes())}

node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
# node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)} # Not directly used in this test script main flow
num_actual_nodes = len(node_list_for_matrix)

# Load the distance matrix
dist_matrix = np.load("../data/graph/LEMD_EGLL_2023_04_01_distances.npy")

# Load the airspace charges matrix
ac_matrix = np.load("../data/graph/LEMD_EGLL_2023_04_01_charges.npy")

# Load the wind model
# Consistent with test_forward_dp, using a 2024 date for wind data,
# while flight times (landing time here) are for 2023.
wind_model = WindDate(date_str="2024-04-01", data_dir="../data/era5")
wind_model = WindFree()

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
estimated_landing_time_str = "2023-04-01 12:00:00"
# Estimated takeoff time
estimated_takeoff_time_str = "2023-04-01 10:15:00"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from equinox.dp.pretoc.backward_dp_vec3 import run_backward_dp
import importlib
import equinox.dp.pretoc.backward_dp_vec3
importlib.reload(equinox.dp.pretoc.backward_dp_vec3)
from equinox.dp.pretoc.backward_dp_vec3 import run_backward_dp

def perform_backward_dp():
    V, active_eta, active_alt, active_phase_return = run_backward_dp(
        graph = G,
        goal_node_id="EGLL",
        estimated_landing_time_str=estimated_landing_time_str,
        origin_elevation_ft=0.0,
        destination_elevation_ft=0.0,
        cost_model=cost_model_instance,
        wind_model=wind_model,
        performance_model=performance_model,
        dist_matrix_np=dist_matrix,
        ac_matrix_np=ac_matrix,
        transitions_list=transitions_list,
        eta_takeoff_str=estimated_takeoff_time_str,
        max_eps_bin=36,
        final_alt_ft=0.0,
        delta_t_seconds_wall_clock=300,
        delta_t_seconds_climb=30,
        max_flight_duration_hours=5,
        climb_phase_switch_allowance_climb_time_bins=10,
        device=device,
        temperature=5e-3
    )

    return V, active_eta, active_alt, active_phase_return

if __name__ == "__main__":
    V, active_eta, active_alt, active_phase_return = perform_backward_dp()
    