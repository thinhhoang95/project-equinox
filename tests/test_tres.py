import torch
import numpy as np
import networkx as nx
from datetime import datetime
from equinox.dp.trespass.tres_forward import tres_forward, save_transitions
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_since_midnight_to_datetime
from equinox.cost.cost_rev1 import CostRev1
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE

def forward_tres():
    
    print('Test: Forward Dynamic Programming (forward_dp_vec2)')

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
    
    # Takeoff time
    takeoff_time_str = "2023-04-01 12:00:00"

    try:
        eta_final, alt_final, phase_final, transitions_list = tres_forward(
            graph=G,
            source_node_id="LEMD",
            takeoff_time_str=takeoff_time_str,
            source_elevation_ft=0.0, # Assuming SFO at sea level
            goal_elevation_ft=0.0,   # Not used during forward DP
            cost_model=cost_model_instance,
            wind_model=wind_model,
            performance_model=performance_model,
            dist_matrix_np=dist_matrix,
            ac_matrix_np=ac_matrix,
            initial_alt_ft=0.0, # Start at 0ft AMSL if source_elevation_ft is 0 for profile alignment
            delta_t_seconds=600, # time window length
            max_flight_duration_hours=5, # max duration to consider for time bins
            etto_delta_t_seconds=30, # ETTO bin width
            max_elapsed_time_since_takeoff_hours=0.75, # max elapsed time to consider for ETTO bins
            device=device
        )
        # transitions_list = [(node_1, eps_1, alt_1, node_2, eps_2, alt_2)]
        # where eps_1 and eps_2 are ETTO values (not ETTO bins)
        # node_1 and node_2 are node IDs (integers, not node names)
        # alt_1 and alt_2 are altitudes (float, in feet)

        print(f'Number of nodes: {num_actual_nodes}')

        print("\n--- Results ---")
        print(f'Total number of unique nodes_from in transitions_list: {len(set([node_1 for node_1, _, _, node_2, _, _ in transitions_list]))}')
        print(f'Total number of unique nodes_to in transitions_list: {len(set([node_2 for _, _, _, node_2, _, _ in transitions_list]))}')
        print(f'Total number of unique transitions in transitions_list: {len(transitions_list)}')
        # base_output_time_str = f"{takeoff_time_str}"

        # for node_idx_res in range(num_actual_nodes):
        #     for time_idx_res in range(V_final.shape[1]):
        #         if not torch.isinf(V_final[node_idx_res, time_idx_res]):
        #             node_id_res = node_list_for_matrix[node_idx_res] # Use the consistent list
                    
        #             # Calculate the approximate start time of this bin for display
        #             # min_time_overall_seconds is takeoff_seconds_since_midnight from inside run_forward_dp
        #             # This was the min_time_overall_seconds used to calculate k_v
        #             takeoff_ssm_ref = datestr_to_seconds_since_midnight(takeoff_time_str) # ssm: seconds since midnight
        #             bin_start_time_ssm = takeoff_ssm_ref + time_idx_res * 600 # 600 is delta_t_seconds from example

        #             time_dt_display = seconds_since_midnight_to_datetime(base_output_time_str, bin_start_time_ssm)
                    
        #             print(f"Node {node_id_res} ({node_idx_res}), Time Bin {time_idx_res} (approx arrival by {time_dt_display.strftime('%Y-%m-%d %H:%M:%S')}):")
        #             print(f"  V = {V_final[node_idx_res, time_idx_res].item():.2f}")
        #             active_eta_val = eta_final[node_idx_res, time_idx_res].item()
        #             active_alt_val = alt_final[node_idx_res, time_idx_res].item()
        #             active_phase_val = phase_final[node_idx_res, time_idx_res].item()
        #             print(f"  Exact ETA: {seconds_since_midnight_to_datetime(base_output_time_str, active_eta_val).strftime('%Y-%m-%d %H:%M:%S') if not np.isnan(active_eta_val) else 'N/A'}")
        #             print(f"  Altitude (ft): {active_alt_val if not np.isnan(active_alt_val) else 'N/A'}")
        #             print(f"  Phase: {active_phase_val if active_phase_val != -1 else 'N/A'}")

        # Dump the value function V to a file
        import os

        output_dir = "data/graph/transitions"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the transitions_list to a file
        save_transitions(transitions_list, output_dir, "LEMD_EGLL_2023_04_01_CLB")

        return transitions_list

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()

import pickle
import networkx as nx
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.wind.wind_free import WindFree
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
import numpy as np
import torch

def backward_tres():
    transitions_list = pickle.load(open("data/graph/transitions/LEMD_EGLL_2023_04_01_CLB.pkl", "rb")) # from test_toc_forward_dp.ipynb, to be rewritten into a more comprehensive package
    
    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for i, node in enumerate(G.nodes())}

    node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
    # node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)} # Not directly used in this test script main flow
    num_actual_nodes = len(node_list_for_matrix)

    # Load the distance matrix
    dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")

    # Load the airspace charges matrix
    ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")

    # Load the wind model
    # Consistent with test_forward_dp, using a 2024 date for wind data,
    # while flight times (landing time here) are for 2023.
    wind_model = WindDate(date_str="2024-04-01", data_dir="data/era5")
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
    from equinox.dp.trespass.tres_backward import tres_backward

    state_closure_list = tres_backward(
        graph=G,
        goal_node_id="EGLL",
        estimated_landing_time_str=estimated_landing_time_str,
        origin_elevation_ft=0.0,
        destination_elevation_ft=0.0,
        wind_model=wind_model,
        performance_model=performance_model,
        transitions_list=transitions_list,
        eta_takeoff_str=estimated_takeoff_time_str,
        final_alt_ft=0.0,
        delta_t_seconds_wall_clock=300,
        delta_t_seconds_climb=30,
        max_flight_duration_hours=5,
        climb_phase_switch_allowance_climb_time_bins=10,
        device=device
    )

    save_transitions(state_closure_list, "data/graph/transitions", "LEMD_EGLL_2023_04_01_CLSR")

    return state_closure_list

import pickle 
from equinox.dp.trespass.forward_svi_log import forward_soft_value_iteration

def forward_svi():
    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for i, node in enumerate(G.nodes())}

    node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
    # node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)} # Not directly used in this test script main flow
    num_actual_nodes = len(node_list_for_matrix)

    # Load the distance matrix
    dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")

    # Load the airspace charges matrix
    ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")

    # Load the wind model
    # Consistent with test_forward_dp, using a 2024 date for wind data,
    # while flight times (landing time here) are for 2023.
    wind_model = WindDate(date_str="2024-04-01", data_dir="data/era5")
    wind_model = WindFree()

    # Load the performance model for a typical narrow body jet
    # performance_model = Performance(
    #     climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
    #     descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
    #     climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
    #     descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
    #     cruise_altitude_ft=35000.0,
    #     cruise_speed_kts=450.0,
    # )

    # Load the cost model
    cost_model_instance = cost_model_1

    # Estimated landing time
    # estimated_landing_time_str = "2023-04-01 12:00:00"
    # Estimated takeoff time
    estimated_takeoff_time_str = "2023-04-01 10:15:00"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transitions = pickle.load(open("data/graph/transitions/LEMD_EGLL_2023_04_01_REACHABLE.pkl", "rb"))
    origin_node_idx = node_to_idx["LEMD"]
    # goal_node_idx = node_to_idx["EGLL"] # Not used in forward_svi
    cost_model = cost_model_1 # Already defined as cost_model_instance
    num_nodes = len(G.nodes())

    # Extract node coordinates from the graph - REMOVED, will be done inside forward_soft_value_iteration
    # node_coords_deg = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    # for node_name, node_idx_val in node_to_idx.items(): # Use node_to_idx for correct mapping
    #     node_data = G.nodes[node_name]
    #     lat = float(node_data['lat'])
    #     lon = float(node_data['lon'])
    #     node_coords_deg[node_idx_val, 0] = lat  # latitude
    #     node_coords_deg[node_idx_val, 1] = lon  # longitude

    # Convert matrices to torch tensors
    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    # Set up time parameters
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    # landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str) # This variable is not used in the current function scope
    
    # Time bin parameters (consistent with other functions)
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    max_flight_duration_hours = 5.0
    num_time_bins_wall_clock = int(max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1
    
    # Climb time parameters
    delta_t_climb_sec = 30.0  # 30 seconds for climb bins
    max_climb_time_hours = 0.75
    num_rho_bins = 37 # this number is from the tres_forward pass: maximum is 36 + 1 for the NUMBER OF BINS (0 to 36, which is 37 bins)
    
    # Phase parameters
    num_phases = 3  # CLIMB=0, CRUISE=1, DESCENT=2
    
    # Set min_wall_clock_time_sec to takeoff time
    min_wall_clock_time_sec = float(takeoff_ssm)

    print(f"Calling forward_soft_value_iteration with:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  num_time_bins_wall_clock: {num_time_bins_wall_clock}")
    print(f"  num_rho_bins: {num_rho_bins}")
    print(f"  num_phases: {num_phases}")
    print(f"  transitions count: {len(transitions)}")
    print(f"  origin_node_idx: {origin_node_idx}")
    print(f"  min_wall_clock_time_sec: {min_wall_clock_time_sec}")
    print(f"  delta_t_wall_clock_sec: {delta_t_wall_clock_sec}")

    # Call forward_soft_value_iteration
    import time
    time_start = time.time()
    V_soft = forward_soft_value_iteration(
        state_transitions=transitions,
        G=G, 
        idx_to_node=idx_to_node,
        origin_node_idx=origin_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        # node_coords_deg=node_coords_deg, # REMOVED
        wind_model=wind_model,
        min_wall_clock_time_sec=min_wall_clock_time_sec,
        delta_t_wall_clock_sec=delta_t_wall_clock_sec,
        device=device,
        verbose=True
    )
    time_end = time.time()
    print(f"Forward SVI completed successfully in {time_end - time_start:.2f} seconds")
    
    print(f"\nForward SVI completed successfully!")
    print(f"V_soft shape: {V_soft.shape}")
    print(f"Number of finite values: {torch.isfinite(V_soft).sum().item()}")
    print(f"Number of infinite values: {torch.isinf(V_soft).sum().item()}")
    
    # Print some sample values at the origin
    print(f"\nSample V_soft values at origin node {origin_node_idx}:")
    for k in range(min(5, num_time_bins_wall_clock)):
        for rho in range(min(3, num_rho_bins)):
            for phase in range(num_phases):
                val = V_soft[origin_node_idx, k, rho, phase].item()
                if torch.isfinite(V_soft[origin_node_idx, k, rho, phase]):
                    print(f"  V_soft[{origin_node_idx}, {k}, {rho}, {phase}] = {val:.4f}")

    # Convert V_soft to a numpy array
    V_soft_np = V_soft.cpu().numpy()
    # Save V_soft to a file
    # Create the directory if it doesn't exist
    import os
    os.makedirs("data/graph/V_soft", exist_ok=True)
    np.save("data/graph/V_soft/LEMD_EGLL_2023_04_01_CLB_V_FWD.npy", V_soft_np)
    return V_soft_np

if __name__ == '__main__':
    # forward_tres()
    # backward_tres()
    forward_svi()