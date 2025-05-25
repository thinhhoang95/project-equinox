import torch
import numpy as np
import networkx as nx
from datetime import datetime
from equinox.dp.forward_dp_vec2 import run_forward_dp
from equinox.helpers.datetimeh import (
    datestr_to_seconds_since_midnight,
    seconds_since_midnight_to_datetime,
)
from equinox.cost.cost_rev1 import CostRev1
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import (
    NARROW_BODY_JET_CLIMB_PROFILE,
    NARROW_BODY_JET_DESCENT_PROFILE,
    NARROW_BODY_JET_CLIMB_VS_PROFILE,
    NARROW_BODY_JET_DESCENT_VS_PROFILE,
)
from equinox.dp.backward_dp_vec2 import run_backward_dp
from equinox.dp.helpers.align_time_bins import calculate_aligned_time_parameters


def get_configuration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")

    node_list_for_matrix = list(G.nodes())  # Consistent order for matrix indexing
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
    estimated_landing_time_str = "2023-04-01 12:00:00"
    estimated_landing_eta = datestr_to_seconds_since_midnight(
        estimated_landing_time_str
    )  # seconds since midnight
    destination_elevation_ft = 0.0

    # Estimated takeoff time
    heuristic_takeoff_time_str = "2023-04-01 10:20:00"  # Similar to takeoff time in forward DP for wind consistency
    heuristic_takeoff_eta = datestr_to_seconds_since_midnight(
        heuristic_takeoff_time_str
    )  # seconds since midnight
    origin_elevation_ft = 0.0

    return {
        "device": device,
        "G": G,
        "node_to_idx_for_matrix": node_to_idx_for_matrix,
        "num_actual_nodes": num_actual_nodes,
        "dist_matrix": dist_matrix,
        "ac_matrix": ac_matrix,
        "wind_model": wind_model,
        "performance_model": performance_model,
        "cost_model": cost_model_instance,
        "estimated_landing_time_str": estimated_landing_time_str,
        "estimated_landing_eta": estimated_landing_eta,
        "origin_elevation_ft": origin_elevation_ft,
        "destination_elevation_ft": destination_elevation_ft,
        "heuristic_takeoff_time_str": heuristic_takeoff_time_str,
        "heuristic_takeoff_eta": heuristic_takeoff_eta,
        "source_node_id": "LEMD",
        "goal_node_id": "EGLL",
    }

import sys

if __name__ == "__main__":
    sys.exit()
    config = get_configuration()
    device = config["device"]
    G = config["G"]
    node_to_idx_for_matrix = config["node_to_idx_for_matrix"]
    dist_matrix = config["dist_matrix"]
    ac_matrix = config["ac_matrix"]
    wind_model = config["wind_model"]
    performance_model = config["performance_model"]
    cost_model_instance = config["cost_model"]
    estimated_landing_time_str = config["estimated_landing_time_str"]
    origin_elevation_ft = config["origin_elevation_ft"]
    destination_elevation_ft = config["destination_elevation_ft"]
    source_node_id = config["source_node_id"]
    goal_node_id = config["goal_node_id"]

    # --- Time Alignment ---
    delta_t_seconds_val = 600  # 10 minutes time window
    # This global window should be large enough to contain the flight plus any desired padding.
    global_window_duration_hours_val = 5.0

    aligned_time_params = calculate_aligned_time_parameters(
        takeoff_time_str=config["heuristic_takeoff_time_str"],
        estimated_landing_time_str=estimated_landing_time_str,
        global_window_duration_hours=global_window_duration_hours_val,
        delta_t_seconds=delta_t_seconds_val,
    )

    min_time_aligned = aligned_time_params["min_time_overall_seconds"]
    num_bins_aligned = aligned_time_params["num_time_bins"]
    takeoff_bin_aligned = aligned_time_params["takeoff_bin_idx"]
    landing_bin_aligned = aligned_time_params["landing_bin_idx"]
    takeoff_s_val = aligned_time_params["takeoff_seconds_since_midnight"]
    landing_s_val = aligned_time_params["landing_seconds_since_midnight"]

    print(f"Aligned Time Parameters: {aligned_time_params}")

    # --- Forward pass ---
    print("Running Forward DP...")
    V_f, eta_f, alt_f, phase_f = run_forward_dp(
        graph=G,
        source_node_id=source_node_id,
        source_elevation_ft=origin_elevation_ft,
        goal_elevation_ft=destination_elevation_ft,  # Unused, but kept for signature consistency for now
        cost_model=cost_model_instance,
        wind_model=wind_model,
        performance_model=performance_model,
        dist_matrix_np=dist_matrix,
        ac_matrix_np=ac_matrix,
        initial_alt_ft=0.0,  # Assuming start at 0ft AMSL as origin_elevation_ft is 0
        delta_t_seconds=delta_t_seconds_val,
        min_time_overall_seconds_aligned=min_time_aligned,
        num_time_bins_aligned=num_bins_aligned,
        takeoff_bin_idx_aligned=takeoff_bin_aligned,
        takeoff_seconds_since_midnight_val=takeoff_s_val,
        device=device,
    )
    print("Forward DP finished.")

    # --- Backward pass ---
    print("Running Backward DP...")
    (
        V_b,
        eta_b,
        alt_b,
        phase_b,
        edge_costs_time_binned,
        edge_cost_gradients_time_binned,
        edge_to_canonical_idx,
    ) = run_backward_dp(
        graph=G,
        goal_node_id=goal_node_id,
        origin_elevation_ft=origin_elevation_ft,  # For climb profile from origin if needed by get_next_state_bw
        destination_elevation_ft=destination_elevation_ft,
        cost_model=cost_model_instance,
        wind_model=wind_model,
        performance_model=performance_model,
        dist_matrix_np=dist_matrix,
        ac_matrix_np=ac_matrix,
        final_alt_ft=0.0,  # Assuming landing at 0ft AMSL as destination_elevation_ft is 0
        delta_t_seconds=delta_t_seconds_val,
        min_time_overall_seconds_aligned=min_time_aligned,
        num_time_bins_aligned=num_bins_aligned,
        landing_bin_idx_aligned=landing_bin_aligned,
        landing_seconds_since_midnight_val=landing_s_val,
        device=device,
        temperature=1.0,  # Default temperature
    )
    print("Backward DP finished.")

    # Now V_f, eta_f, alt_f, phase_f from forward pass and
    # V_b, eta_b, alt_b, phase_b from backward pass share the same time bin indexing.
    # For example, eta_f[node, k] and eta_b[node, k] correspond to the same time interval.

    # Example: Check shapes (should be [num_nodes, num_bins_aligned])
    if V_f is not None and V_b is not None:
        print(f"V_f shape: {V_f.shape}, V_b shape: {V_b.shape}")
        print(f"eta_f shape: {eta_f.shape}, eta_b shape: {eta_b.shape}")
        if V_f.shape[1] != num_bins_aligned or V_b.shape[1] != num_bins_aligned:
            print("ERROR: Mismatch in number of time bins!")

    print("Stop here for debugging or further processing")
