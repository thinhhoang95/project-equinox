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
from equinox.dp.trespass.sparse_io_utils import save_sparse_coo_tensor_with_convention, load_sparse_coo_tensor_with_convention

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

def forward_svi(headless=False):
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
    
    # Time bin parameters
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    # max_flight_duration_hours = 5.0 # Original static definition
    # num_time_bins_wall_clock = int(max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1 # Original static definition
    
    # Climb time parameters (original static definitions, now derived)
    # delta_t_climb_sec = 30.0  # 30 seconds for climb bins
    # max_climb_time_hours = 0.75
    # num_rho_bins = 37 # this number is from the tres_forward pass: maximum is 36 + 1 for the NUMBER OF BINS (0 to 36, which is 37 bins)
    
    # Phase parameters (original static definition, now derived)
    # num_phases = 3  # CLIMB=0, CRUISE=1, DESCENT=2

    # Dynamically determine num_time_bins_wall_clock, num_rho_bins, and num_phases from transitions
    max_k_val = 0
    max_rho_val = 0
    max_phase_val = 0
    if transitions:
        # Structure of t: u_idx, k_u, rho_u, u_alt_ft, phase_u, v_idx, k_v, rho_v, v_alt_ft, phase_v, ...
        max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions))
        max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions))
        max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions))

    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1

    # Check against a configured max duration, if necessary (optional, similar to backward_svi's warning)
    # This part is illustrative; you might want to adjust max_flight_duration_hours if it's still a relevant concept
    # or rely solely on transition data.
    configured_max_flight_duration_hours = 5.0 
    max_bins_from_config = int(configured_max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1
    if num_time_bins_wall_clock > max_bins_from_config:
        print(f"Warning: max_k_val from transitions ({max_k_val}) suggests more time bins ({num_time_bins_wall_clock}) than a typical configured max duration would imply ({max_bins_from_config}). Using derived value from transitions.")
    elif not transitions:
        print("Warning: No transitions loaded. num_time_bins_wall_clock, num_rho_bins, num_phases derived as 1. This might be too small if transitions are expected.")
    
    # Set min_wall_clock_time_sec to takeoff time
    min_wall_clock_time_sec = float(takeoff_ssm)

    print(f"Calling forward_soft_value_iteration with:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  num_time_bins_wall_clock: {num_time_bins_wall_clock} (derived from transitions, max_k_val: {max_k_val})")
    print(f"  num_rho_bins: {num_rho_bins} (derived from transitions, max_rho_val: {max_rho_val})")
    print(f"  num_phases: {num_phases} (derived from transitions, max_phase_val: {max_phase_val})")
    print(f"  transitions count: {len(transitions)}")
    print(f"  origin_node_idx: {origin_node_idx}")
    print(f"  min_wall_clock_time_sec: {min_wall_clock_time_sec}")
    print(f"  delta_t_wall_clock_sec: {delta_t_wall_clock_sec}")

    # Ask for confirmation before proceeding with the forward SVI computation
    if not headless:
        print("\nAbout to run forward soft value iteration with the above parameters.")
        confirmation = input("Proceed? (Y/n): ").strip().lower()
        if confirmation in ['n', 'no']:
            print("Aborted by user.")
            return None
        elif confirmation not in ['', 'y', 'yes']:
            print("Invalid input. Assuming 'no' and aborting.")
            return None
        print("Proceeding with forward SVI computation...")

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
    import os
    os.makedirs("data/graph/V_soft", exist_ok=True)
    
    # Create sparse tensor for saving: store only finite values from V_soft
    finite_mask = torch.isfinite(V_soft)
    # nonzero(as_tuple=False) returns (N, D) tensor, transpose to (D, N) for sparse_coo_tensor
    sparse_indices = finite_mask.nonzero(as_tuple=False).transpose(0, 1)
    sparse_values = V_soft[finite_mask]

    V_soft_sparse = torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=V_soft.shape,
        dtype=V_soft.dtype,
        device=V_soft.device
    ).coalesce()

    output_dir_path = "data/graph/V_soft" 
    sparse_file_name = "LEMD_EGLL_2023_04_01_V_FWD_SPRSE.pt" 
    sparse_file_path = os.path.join(output_dir_path, sparse_file_name)
    
    # save_sparse_coo_tensor_with_convention is imported at the top of test_trespass.py
    save_sparse_coo_tensor_with_convention(
        V_soft_sparse, 
        sparse_file_path,
        "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
    )
    print(f"Saved sparse V_soft (finite values only) to {sparse_file_path}")

    return V_soft_np

from equinox.dp.trespass.backward_svi_log_cost import backward_soft_value_iteration

def backward_svi(headless=False):
    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for i, node in enumerate(G.nodes())}

    num_nodes = len(G.nodes())

    # Load the distance matrix
    dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")

    # Load the airspace charges matrix
    ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")

    # Load the wind model (using WindFree for simplicity and consistency with forward_svi example)
    wind_model = WindFree()

    # Load the cost model
    cost_model = cost_model_1

    # Estimated takeoff time (used for defining the time window start for k_idx)
    estimated_takeoff_time_str = "2023-04-01 10:15:00"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load transitions - these define the state space and connections
    # Ensure this is the correct transitions file. The forward SVI uses REACHABLE.
    transitions = pickle.load(open("data/graph/transitions/LEMD_EGLL_2023_04_01_REACHABLE.pkl", "rb"))
    
    goal_node_idx = node_to_idx["EGLL"] 

    # Convert matrices to torch tensors
    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    # Set up time parameters
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # Time bin parameters (should be consistent with those used to generate transitions)
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    max_flight_duration_hours = 5.0
    num_time_bins_wall_clock = int(max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1
    
    # Profile time bins (rho_bins)
    # num_rho_bins = 37 # From forward_svi example, should match transition generation logic
    # Let's try to infer num_rho_bins and num_phases from transitions data if possible, or set explicitly.
    # Max rho_u_idx and rho_v_idx from transitions + 1
    # Max phase_u and phase_v from transitions + 1
    max_k_val = 0
    max_rho_val = 0
    max_phase_val = 0
    if transitions:
        max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions))
        # Correct indices for rho and phase based on backward_svi_log.py unpacking:
        # u_idx, k_u, rho_u, u_alt_ft, phase_u
        # v_idx, k_v, rho_v, v_alt_ft, phase_v
        # trans[2] = rho_u, trans[4] = phase_u
        # trans[7] = rho_v, trans[9] = phase_v
        max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions)) 
        max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions))

    num_time_bins_wall_clock_actual = max_k_val + 1 # Ensure this is large enough for all k in transitions
    if num_time_bins_wall_clock_actual > num_time_bins_wall_clock:
        print(f"Warning: max_k_val from transitions ({max_k_val}) suggests more time bins than configured ({num_time_bins_wall_clock}). Using {num_time_bins_wall_clock_actual}.")
        num_time_bins_wall_clock = num_time_bins_wall_clock_actual
    elif not transitions:
         print("Warning: No transitions loaded. num_time_bins_wall_clock might be too small.")

    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1
    
    # Set min_wall_clock_time_sec to takeoff time (anchor for k_idx calculations)
    min_wall_clock_time_sec = float(takeoff_ssm)

    print(f"Calling backward_soft_value_iteration with:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  num_time_bins_wall_clock: {num_time_bins_wall_clock} (derived: {max_k_val+1})")
    print(f"  num_rho_bins: {num_rho_bins} (derived: {max_rho_val+1})")
    print(f"  num_phases: {num_phases} (derived: {max_phase_val+1})")
    print(f"  transitions count: {len(transitions)}")
    print(f"  goal_node_idx: {goal_node_idx}")
    print(f"  min_wall_clock_time_sec (for k_idx interpretation): {min_wall_clock_time_sec}")
    print(f"  delta_t_wall_clock_sec: {delta_t_wall_clock_sec}")

    # Ask for confirmation before proceeding with the backward SVI computation
    if not headless:
        print("\nAbout to run backward soft value iteration with the above parameters.")
        confirmation = input("Proceed? (Y/n): ").strip().lower()
        if confirmation in ['n', 'no']:
            print("Aborted by user.")
            return None
        elif confirmation not in ['', 'y', 'yes']:
            print("Invalid input. Assuming 'no' and aborting.")
            return None
        print("Proceeding with backward SVI computation...")

    # Call backward_soft_value_iteration
    import time
    time_start = time.time()
    V_soft_bwd, edge_costs = backward_soft_value_iteration(
        state_transitions=transitions,
        G=G, 
        idx_to_node=idx_to_node,
        goal_node_idx=goal_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        wind_model=wind_model,
        min_wall_clock_time_sec=min_wall_clock_time_sec,
        delta_t_wall_clock_sec=delta_t_wall_clock_sec,
        device=device,
        verbose=True
    )
    time_end = time.time()
    print(f"Backward SVI completed successfully in {time_end - time_start:.2f} seconds")
    
    print(f"\nBackward SVI (cost-to-go) completed successfully!")
    print(f"V_soft_bwd shape: {V_soft_bwd.shape}")
    print(f"Number of finite values: {torch.isfinite(V_soft_bwd).sum().item()}")
    print(f"Number of infinite values: {torch.isinf(V_soft_bwd).sum().item()}")
    
    # Print some sample values at the goal node
    print(f"\nSample V_soft_bwd values at goal node {goal_node_idx}:")
    for k in range(min(5, num_time_bins_wall_clock)):
        for rho in range(min(3, num_rho_bins)):
            for phase in range(num_phases):
                val = V_soft_bwd[goal_node_idx, k, rho, phase].item()
                if torch.isfinite(V_soft_bwd[goal_node_idx, k, rho, phase]):
                    print(f"  V_soft_bwd[{goal_node_idx}, {k}, {rho}, {phase}] = {val:.4f}")

    V_soft_bwd_np = V_soft_bwd.cpu().numpy()
    import os
    os.makedirs("data/graph/V_soft", exist_ok=True)
    
    # Create sparse tensor for saving: store only finite values from V_soft_bwd
    finite_mask = torch.isfinite(V_soft_bwd)
    # nonzero(as_tuple=False) returns (N, D) tensor, transpose to (D, N) for sparse_coo_tensor
    sparse_indices = finite_mask.nonzero(as_tuple=False).transpose(0, 1)
    sparse_values = V_soft_bwd[finite_mask]

    V_soft_bwd_sparse = torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=V_soft_bwd.shape,
        dtype=V_soft_bwd.dtype,
        device=V_soft_bwd.device
    ).coalesce()

    output_dir_path = "data/graph/V_soft" 
    sparse_file_name = "LEMD_EGLL_2023_04_01_V_BWD_SPRSE.pt" 
    sparse_file_path = os.path.join(output_dir_path, sparse_file_name)
    
    # save_sparse_coo_tensor_with_convention is imported at the top of test_trespass.py
    save_sparse_coo_tensor_with_convention(
        V_soft_bwd_sparse, 
        sparse_file_path,
        "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
    )
    print(f"Saved sparse V_soft_bwd (finite values only) to {sparse_file_path}")
    
    save_sparse_coo_tensor_with_convention(edge_costs, "data/graph/V_soft/LEMD_EGLL_2023_04_01_CLB_COST.pt")
    return V_soft_bwd_np

from equinox.sampling.trespass.sampler import sample_tres_trajectory

def test_tres_sampler(headless=True):
    print("\n--- Test TRes Sampler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the route graph
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for i, node in enumerate(G.nodes())}
    num_nodes = len(G.nodes())

    # Load backward soft value function (log values) from sparse format
    try:
        V_soft_bwd_sparse, interpretation_note = load_sparse_coo_tensor_with_convention(
            "data/graph/V_soft/LEMD_EGLL_2023_04_01_V_BWD_SPRSE.pt",
            target_device=device
            # interpretation_note="Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
        )
        print(f"Loaded sparse backward value function. Interpretation: {interpretation_note}")
        # V_soft_bwd_np = V_soft_bwd_sparse.to_dense().cpu().numpy() # Original line

        # New logic to fill unspecified sparse entries with -inf when converting to dense
        # 1. Coalesce the sparse tensor (good practice, ensures unique indices).
        V_soft_bwd_sparse_coalesced = V_soft_bwd_sparse.coalesce()

        # 2. Create a dense tensor filled with -infinity.
        #    Use the sparse tensor's dtype and device.
        V_soft_bwd_dense_filled = torch.full(
            V_soft_bwd_sparse_coalesced.shape,
            -float('inf'),
            dtype=V_soft_bwd_sparse_coalesced.dtype,
            device=V_soft_bwd_sparse_coalesced.device
        )

        # 3. Get indices and values from the coalesced sparse tensor.
        indices = V_soft_bwd_sparse_coalesced.indices()
        values = V_soft_bwd_sparse_coalesced.values()

        # 4. Place the explicit values from the sparse tensor into the dense tensor.
        #    This is done only if there are any explicit values.
        if values.numel() > 0:
            V_soft_bwd_dense_filled[tuple(indices)] = values
        
        # 5. Convert to numpy array on CPU.
        V_soft_bwd_np = V_soft_bwd_dense_filled.cpu().numpy()

    except FileNotFoundError:
        print("Backward value function file not found. Please run backward_svi first.")
        print("Skipping TRes Sampler test.")
        return

    # Convert log values V_bwd to Z_b values for the sampler
    Z_b_values = torch.exp(torch.from_numpy(V_soft_bwd_np).to(device))

    # Load edge costs
    try:
        edge_costs_tensor, interpretation_note = load_sparse_coo_tensor_with_convention(
            "data/graph/V_soft/LEMD_EGLL_2023_04_01_CLB_COST.pt",
            target_device=device
        ) # actually, sparse values are defaulted to 0 here (which is wrong, should be inf instead), but it does not matter because we never select these values during the sampling process
        print(f"Edge costs loaded. Interpretation Note: {interpretation_note}")
    except FileNotFoundError:
        print("Edge costs file not found. Please run backward_svi first.")
        print("Skipping TRes Sampler test.")
        return

    origin_node_id = "LEMD"
    goal_node_id = "EGLL"

    # Time parameters (consistent with backward_svi)
    estimated_takeoff_time_str = "2023-04-01 10:15:00" # From backward_svi test
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight # Already imported
    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    min_wall_clock_time_sec = float(takeoff_ssm)
    delta_t_wall_clock_sec = 300.0  # 5 minutes, from backward_svi test

    # Initial state parameters
    initial_k = 0 # Assuming k=0 corresponds to min_wall_clock_time_sec (takeoff time)
    
    # num_rho_bins is Z_b_values.shape[2]
    # rho_idx is 0 for "no climb time remaining". Max index is for full climb time remaining.
    if Z_b_values.shape[2] > 0:
        initial_rho = Z_b_values.shape[2] - 1 
    else:
        print("Error: Number of rho bins is 0. Cannot determine initial_rho.")
        return
        
    initial_phase = 0 # 0: CLIMB

    print(f"Starting sampling from {origin_node_id} (idx {node_to_idx[origin_node_id]}) to {goal_node_id} (idx {node_to_idx[goal_node_id]})")
    print(f"Initial state: k={initial_k}, rho={initial_rho}, phase={initial_phase}")
    print(f"Z_b shape: {Z_b_values.shape}")
    print(f"Edge costs indices shape: {edge_costs_tensor.indices().shape}, values shape: {edge_costs_tensor.values().shape}")


    num_samples = 100
    successful_samples = 0
    failed_samples = 0
    trajectories = []

    for i in range(num_samples):
        print(f"\nSampling trajectory {i+1}/{num_samples}...")
        trajectory = sample_tres_trajectory(
            G=G,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            origin_node_id=origin_node_id,
            goal_node_id=goal_node_id,
            initial_rho=initial_rho,
            initial_phase=initial_phase,
            backward_values=Z_b_values, # This is Z_b = exp(V_bwd)
            edge_costs_uv=edge_costs_tensor,
            max_steps=200 # Max steps per trajectory
        )
        if trajectory:
            successful_samples += 1
            trajectories.append(trajectory)
            print(f"  Successfully sampled trajectory {i+1} with {len(trajectory)} steps.")
            # print(f"  Trajectory: {trajectory}") # Can be very verbose
            # Print first and last few states
            if len(trajectory) > 5:
                print(f"    Start: {trajectory[:3]}")
                print(f"    End: {trajectory[-3:]}")
            else:
                print(f"    Trajectory: {trajectory}")
        else:
            failed_samples += 1
            print(f"  Failed to sample trajectory {i+1}.")

    print("\n--- Sampler Test Results ---")
    print(f"Total attempts: {num_samples}")
    print(f"Successful trajectories: {successful_samples}")
    print(f"Failed trajectories: {failed_samples}")

    if trajectories:
        avg_len = np.mean([len(t) for t in trajectories])
        min_len = np.min([len(t) for t in trajectories])
        max_len = np.max([len(t) for t in trajectories])
        print(f"Average trajectory length: {avg_len:.2f}")
        print(f"Min trajectory length: {min_len}")
        print(f"Max trajectory length: {max_len}")

    # Save trajectories to a file
    import os
    os.makedirs("data/graph/trajectories", exist_ok=True)
    
    # Save trajectories as text file with waypoint names separated by whitespace
    with open("data/graph/trajectories/LEMD_EGLL_2023_04_01_CLB_trajectories.txt", "w") as f:
        for i, trajectory in enumerate(trajectories):
            # Extract waypoint names from trajectory tuples (waypoint_name, k, rho, phase)
            waypoint_names = [str(step[0]) for step in trajectory]
            trajectory_line = " ".join(waypoint_names)
            f.write(f"{trajectory_line}\n")
    
    print(f"Saved {len(trajectories)} trajectories to data/graph/trajectories/LEMD_EGLL_2023_04_01_CLB_trajectories.txt")
    
    # Example of checking a specific state if needed for debugging
    # origin_idx_val = node_to_idx[origin_node_id]
    # initial_Z_b_val = Z_b_values[origin_idx_val, initial_k, initial_rho, initial_phase].item()
    # print(f"Initial Z_b({origin_node_id}, k={initial_k}, rho={initial_rho}, phase={initial_phase}) = {initial_Z_b_val}")
    # if not np.isfinite(initial_Z_b_val) or initial_Z_b_val == 0:
    #    print("Warning: Initial Z_b value is not suitable for starting sampling.")


if __name__ == '__main__':
    forward_tres()
    backward_tres()
    backward_svi(headless=True)
    forward_svi(headless=True) # headless = false: ask for confirmation
    print('CAUTION: The forward SVI contains a hard-coded initial log-mass. This should be corrected in the future.')
    test_tres_sampler(headless=False)