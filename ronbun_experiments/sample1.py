
flight_takeoff_str = '2023-04-01 06:26:25'
flight_landing_str = '2023-04-01 08:23:59'

import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from equinox.dp.trespass.tres_forward import tres_forward, save_transitions
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.dp.trespass.sparse_io_utils import save_sparse_coo_tensor_with_convention, load_sparse_coo_tensor_with_convention
import pickle

from equinox.config import RunConfiguration

def forward_tres(config: RunConfiguration, components: dict):
    
    print('Test: Forward Dynamic Programming (forward_dp_vec2)')

    device = components['device']
    print(f"Using device: {device}")

    G = components['graph']
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    wind_model = components['wind_model']
    performance_model = components['performance_model']
    cost_model_instance = components['cost_model']
    num_actual_nodes = components['num_nodes']
    
    # Takeoff time
    takeoff_time_str = config.takeoff_time_str

    try:
        eta_final, alt_final, phase_final, transitions_list = tres_forward(
            graph=G,
            source_node_id=config.origin_node,
            takeoff_time_str=takeoff_time_str,
            source_elevation_ft=config.source_elevation_ft,
            goal_elevation_ft=config.goal_elevation_ft,
            cost_model=cost_model_instance,
            wind_model=wind_model,
            performance_model=performance_model,
            dist_matrix_np=dist_matrix,
            ac_matrix_np=ac_matrix,
            initial_alt_ft=config.initial_alt_ft,
            delta_t_seconds=config.delta_t_seconds,
            max_flight_duration_hours=config.max_flight_duration_hours,
            etto_delta_t_seconds=config.etto_delta_t_seconds,
            max_elapsed_time_since_takeoff_hours=config.max_elapsed_time_since_takeoff_hours,
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
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the transitions_list to a file
        save_transitions(transitions_list, output_dir, f"{config.file_prefix}_{config.tres_forward_output_file_name}")

        return transitions_list

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()

import pickle
import networkx as nx
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
import numpy as np
import torch

def backward_tres(config: RunConfiguration, components: dict):
    forward_pass_output_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.tres_forward_output_file_name}.pkl")
    transitions_list = pickle.load(open(forward_pass_output_path, "rb"))
    
    # Load from components
    G = components['graph']
    wind_model = components['wind_model']
    performance_model = components['performance_model']
    device = components['device']

    # Estimated landing time
    estimated_landing_time_str = config.estimated_landing_time_str
    # Estimated takeoff time
    estimated_takeoff_time_str = config.estimated_takeoff_time_str

    from equinox.dp.trespass.tres_backward import tres_backward

    state_closure_list = tres_backward(
        graph=G,
        goal_node_id=config.goal_node,
        estimated_landing_time_str=estimated_landing_time_str,
        origin_elevation_ft=config.source_elevation_ft,
        destination_elevation_ft=config.goal_elevation_ft,
        wind_model=wind_model,
        performance_model=performance_model,
        transitions_list=transitions_list,
        eta_takeoff_str=estimated_takeoff_time_str,
        final_alt_ft=0.0,
        delta_t_seconds_wall_clock=300,
        delta_t_seconds_climb=30,
        max_flight_duration_hours=config.max_flight_duration_hours,
        climb_phase_switch_allowance_climb_time_bins=config.climb_phase_switch_allowance_climb_time_bins,
        device=device
    )

    save_transitions(state_closure_list, config.output_dir, f"{config.file_prefix}_{config.tres_backward_output_file_name}")

    return state_closure_list

def thinning(config: RunConfiguration, components: dict):
    from equinox.dp.trespass.thinning import thin_closures
    from equinox.dp.trespass.tres_forward import save_transitions
    # Load closure_list from the backward tres pass
    backward_pass_output_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.tres_backward_output_file_name}.pkl")
    closure_list = pickle.load(open(backward_pass_output_path, "rb"))
    print(f"Loaded {len(closure_list)} closures from backward tres pass")

    G = components['graph']
    node_to_idx = components['node_to_idx']

    thinned_closures = thin_closures(node_to_idx[config.origin_node], node_to_idx[config.goal_node], 36, G, closure_list)
    save_transitions(thinned_closures, config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}")

def amortize_wind_average(config: RunConfiguration, components: dict):
    print("Pre-computing wind averages on all transititions")
    # Calculate the average wind for all transitions
    wind_model = components['wind_model']

    # Load the transitions
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    delta_t_wall_clock_sec = 300.0  # 5 minutes
    max_flight_duration_hours = config.max_flight_duration_hours
    num_time_bins_wall_clock = int(max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1
    # CRITICAL FIX: Use same time reference as tres_backward
    estimated_landing_time_str = config.estimated_landing_time_str
    estimated_landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str)
    min_wall_clock_time_sec = float(estimated_landing_ssm - max_flight_duration_hours * 3600)

    node_coords_deg = components['node_coords_deg']

    wind_avg = wind_model.get_average_tailwind_on_edges_knots(transitions, node_coords_deg, min_wall_clock_time_sec, delta_t_wall_clock_sec, num_integration_steps=3)
    # Save the wind averages to a file for later use
    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    os.makedirs(os.path.dirname(wind_avg_file_path), exist_ok=True)
    torch.save(wind_avg, wind_avg_file_path)
    print(f"Saved wind averages tensor of shape {wind_avg.shape} to {wind_avg_file_path}")

def forward_svi(config: RunConfiguration, components: dict, headless=False):
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    G = components['graph']
    idx_to_node = components['idx_to_node']
    cost_model_instance = components['cost_model']
    origin_node_idx = components['origin_node_idx']
    device = components['device']
    num_nodes = components['num_nodes']

    # Load pre-computed wind averages
    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    try:
        avg_tailwind_knots_per_transition = torch.load(wind_avg_file_path)
        print(f"Loaded pre-computed wind averages from {wind_avg_file_path} with shape {avg_tailwind_knots_per_transition.shape}")
    except FileNotFoundError:
        print(f"Wind averages file not found at {wind_avg_file_path}.")
        print("Please run the `amortize_wind_average` function first.")
        return

    # Estimated takeoff time
    estimated_takeoff_time_str = config.estimated_takeoff_time_str

    device = components['device']
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    # Convert matrices to torch tensors
    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    # Set up time parameters
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # Time bin parameters
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    
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
    configured_max_flight_duration_hours = config.max_flight_duration_hours
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
        avg_tailwind_knots_per_transition=avg_tailwind_knots_per_transition.to(device),
        G=G,
        idx_to_node=idx_to_node,
        origin_node_idx=origin_node_idx,
        cost_model=cost_model_instance,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        verbose=True,
        gamma=config.gamma
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
    output_dir_path = config.output_dir
    os.makedirs(output_dir_path, exist_ok=True)
    
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

    sparse_file_name = f"{config.file_prefix}_V_FWD_SPRSE_WIND.pt"
    sparse_file_path = os.path.join(output_dir_path, sparse_file_name)
    
    # save_sparse_coo_tensor_with_convention is imported at the top of test_trespass.py
    save_sparse_coo_tensor_with_convention(
        V_soft_sparse, 
        sparse_file_path,
        "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
    )
    print(f"Saved sparse V_soft (finite values only) to {sparse_file_path}")

    return V_soft_np

# from equinox.dp.trespass.amorwin.backward_svi_log_cost import backward_soft_value_iteration
# from equinox.dp.trespass.amorwin.backward_svi_log_cost_hardmin import backward_hard_value_iteration
from equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass

def backward_svi(config: RunConfiguration, components: dict, headless=False):
    num_nodes = components['num_nodes']
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    G = components['graph']
    idx_to_node = components['idx_to_node']
    goal_node_idx = components['goal_node_idx']
    cost_model_instance = components['cost_model']
    device = components['device']

    # Load pre-computed wind averages
    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    try:
        avg_tailwind_knots_per_transition = torch.load(wind_avg_file_path)
        print(f"Loaded pre-computed wind averages from {wind_avg_file_path}")
    except FileNotFoundError:
        print(f"Wind averages file not found at {wind_avg_file_path}. Please run `amortize_wind_average` first.")
        return

    # Estimated takeoff time (used for defining the time window start for k_idx)
    estimated_takeoff_time_str = config.estimated_takeoff_time_str
    
    # Load transitions - these define the state space and connections
    # Ensure this is the correct transitions file. The forward SVI uses REACHABLE.
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    # Convert matrices to torch tensors
    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    # Set up time parameters
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # Time bin parameters (should be consistent with those used to generate transitions)
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    max_flight_duration_hours = config.max_flight_duration_hours
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
    # V_soft_bwd, edge_costs = backward_hard_value_iteration(
    V_soft_bwd, edge_costs = backward_soft_value_iteration(
        state_transitions=transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots_per_transition.to(device),
        G=G,
        idx_to_node=idx_to_node,
        goal_node_idx=goal_node_idx,
        cost_model=cost_model_instance,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        verbose=True,
        gamma=config.gamma # smaller gamma means more greedy towards the shortest path
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
    output_dir_path = config.output_dir
    os.makedirs(output_dir_path, exist_ok=True)
    
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

    sparse_file_name = f"{config.file_prefix}_V_BWD_SPRSE_WIND.pt"
    sparse_file_path = os.path.join(output_dir_path, sparse_file_name)
    
    # save_sparse_coo_tensor_with_convention is imported at the top of test_trespass.py
    save_sparse_coo_tensor_with_convention(
        V_soft_bwd_sparse, 
        sparse_file_path,
        "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
    )
    print(f"Saved sparse V_soft_bwd (finite values only) to {sparse_file_path}")
    
    edge_costs_path = os.path.join(output_dir_path, f"{config.file_prefix}_COST_WIND.pt")
    save_sparse_coo_tensor_with_convention(edge_costs, edge_costs_path)
    return V_soft_bwd_np


# Implement the backward gradient pass HERE
# Updated for CostRev4 - works with PLM parameters instead of preference matrix
def backward_gradient_pass_test(config: RunConfiguration, components: dict, headless=True):
    print("\n--- Running Backward Gradient Pass Test (CostRev4) ---")
    device = components['device']

    # --- 1. Load all necessary data ---
    print("Loading data for gradient pass...")
    
    # Load transitions to determine dimensions
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    try:
        transitions = pickle.load(open(thinned_transitions_path, "rb"))
        print(f"Loaded {len(transitions)} transitions.")
    except FileNotFoundError:
        print(f"Transitions file not found at {thinned_transitions_path}. Aborting.")
        return

    # Dynamically determine dimensions from transitions
    max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions))
    max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions))
    max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions))
    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1
    num_nodes = components['num_nodes']
    v_shape = (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases)
    print(f"Value function shape determined from transitions: {v_shape}")

    def load_dense_v(path):
        sparse_v, _ = load_sparse_coo_tensor_with_convention(path, target_device=device)
        sparse_v = sparse_v.coalesce()
        dense_v = torch.full(v_shape, float('inf'), dtype=sparse_v.dtype, device=device)
        indices = sparse_v.indices()
        values = sparse_v.values()
        if values.numel() > 0:
            dense_v[tuple(indices)] = values
        return dense_v

    # Load Forward and Backward Value Functions
    try:
        v_fwd_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_FWD_SPRSE_WIND.pt")
        V_f = load_dense_v(v_fwd_path)
        print("Loaded forward value function.")
        
        v_bwd_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_BWD_SPRSE_WIND.pt")
        V_b = load_dense_v(v_bwd_path)
        print("Loaded backward value function.")
    except FileNotFoundError as e:
        print(f"Value function file not found: {e}. Please run both SVI passes first. Aborting.")
        return

    # Load other required components
    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    avg_tailwind_knots = torch.load(wind_avg_file_path).to(device)
    dist_matrix = torch.tensor(components['dist_matrix'], dtype=torch.float64, device=device)
    ac_matrix = torch.tensor(components['ac_matrix'], dtype=torch.float64, device=device)
    cost_model = components['cost_model']
    cost_model.to(torch.float64) # Ensure model is float64 for calculations
    origin_node_idx = components['origin_node_idx']

    # Load or create empirical counts
    empirical_counts_path = os.path.join(config.output_dir, f"{config.file_prefix}_empirical_counts.pt")
    try:
        empirical_counts = torch.load(empirical_counts_path).to(device, dtype=torch.float64)
        print(f"Loaded empirical counts from {empirical_counts_path}")
    except FileNotFoundError:
        print(f"Empirical counts not found. Creating a dummy count matrix.")
        empirical_counts = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float64)
        # Create a dummy trajectory for testing
        if len(transitions) > 20:
            # Create a simple path from origin to somewhere
            u,v = -1, origin_node_idx
            for _ in range(20):
                # Find a transition starting from v
                found = False
                for t in transitions:
                    if t[0] == v:
                        u, v = t[0], t[5]
                        empirical_counts[u, v] = 1
                        found = True
                        break
                if not found:
                    break # No more segments in path
        print("Dummy empirical counts created.")


    # --- 2. Run the gradient pass ---
    if not headless:
        print("\nAbout to run gradient pass.")
        confirmation = input("Proceed? (Y/n): ").strip().lower()
        if confirmation in ['n', 'no']:
            print("Aborted by user.")
            return None
    print("\nExecuting backward_gradient_pass...")
    
    # Ensure cost model parameters require grad for CostRev4
    # CostRev4 doesn't have a preference matrix, so we enable gradients for PLM parameters
    for param in cost_model.parameters():
        if param.requires_grad:
            param.requires_grad = True

    likelihoods, grad = backward_gradient_pass(
        state_transitions=transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        V_f=V_f,
        V_b=V_b,
        cost_model=cost_model,
        empirical_counts=empirical_counts,
        origin_node_idx=origin_node_idx,
        num_nodes=num_nodes,
        distance_matrix_d=dist_matrix,
        airspace_charge_matrix_ac=ac_matrix,
        device=device,
        gamma=config.gamma,
        verbose=True
    )
    # --- 2.1. Display the likelihood pass results ---
    # Please write your code here
    print("\n--- Top 10 Links by Expected Traversal Likelihood ---")
    idx_to_node = components['idx_to_node']

    # Move likelihoods tensor to CPU for processing with NumPy
    likelihoods_np = likelihoods.cpu().numpy()

    # Flatten the matrix and get indices that would sort it in descending order
    flat_sorted_indices = np.argsort(likelihoods_np, axis=None)[::-1]

    top_k = 10
    count = 0
    for flat_idx in flat_sorted_indices:
        if count >= top_k:
            break
        likelihood_val = likelihoods_np.flat[flat_idx]
        if likelihood_val <= 0.0:
            break  # Stop if likelihoods are zero or negative

        u_idx = flat_idx // num_nodes
        v_idx = flat_idx % num_nodes

        u_name = idx_to_node[u_idx]
        v_name = idx_to_node[v_idx]

        print(f"{count + 1}. {u_name} -> {v_name}: {likelihood_val:.6f}")
        count += 1

    if count == 0:
        print("No links with positive traversal likelihood were found.")

    # --- 3. Display results and perform a test optimizer step ---
    print("\n--- Gradient Pass Results ---")
    print(f"Shape of traversal likelihoods: {likelihoods.shape}")
    print(f"Total traversal likelihood: {likelihoods.sum().item():.4f} (should be close to 1.0)")
    print(f"Shape of final gradient: {grad.shape}")
    print(f"Gradient norm: {torch.norm(grad).item()}")
    print(f"Gradient (first 10 values): {grad[:10].tolist()}")

    print("\n--- Testing SGD Step ---")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cost_model.parameters()), lr=0.01)
    
    # Manually assign gradients
    param_idx = 0
    for param in cost_model.parameters():
        if param.requires_grad:
            if param.grad is not None:
                param.grad.zero_()
            num_param_elements = param.numel()
            grad_slice = grad[param_idx : param_idx + num_param_elements].view(param.shape).to(param.dtype)
            param.grad = grad_slice
            param_idx += num_param_elements

    # For CostRev4, check changes in PLM parameters instead of preference matrix
    params_before = {}
    for name, param in cost_model.named_parameters():
        if param.requires_grad:
            params_before[name] = param.detach().clone()
    
    optimizer.step()
    
    total_change = 0.0
    for name, param in cost_model.named_parameters():
        if param.requires_grad and name in params_before:
            param_change = torch.norm(param.detach() - params_before[name]).item()
            total_change += param_change
            print(f"Change in {name}: {param_change:.6f}")
    
    print(f"Total norm of parameter changes after one SGD step: {total_change:.6f}")
    if total_change > 1e-9:
        print("OK: Parameters were updated.")
    else:
        print("WARNING: Parameters did not update. Check gradient calculation or optimizer setup.")

    print("\nBackward gradient pass test finished.")
    return grad, likelihoods




from equinox.sampling.trespass.sampler_log import sample_tres_trajectory

def test_tres_sampler(config: RunConfiguration, components: dict, headless=True):
    print("\n--- Test TRes Sampler ---")
    device = components['device']
    print(f"Using device: {device}")

    # Load the route graph
    G = components['graph']
    node_to_idx = components['node_to_idx']
    idx_to_node = components['idx_to_node']

    # Load backward soft value function (log values) from sparse format
    try:
        v_bwd_sparse_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_BWD_SPRSE_WIND.pt")
        V_soft_bwd_sparse, interpretation_note = load_sparse_coo_tensor_with_convention(
            v_bwd_sparse_path,
            target_device=device
        )
        print(f"Loaded sparse backward value function. Interpretation: {interpretation_note}")

        V_soft_bwd_sparse_coalesced = V_soft_bwd_sparse.coalesce()
        V_soft_bwd_dense_filled = torch.full(
            V_soft_bwd_sparse_coalesced.shape,
            float('inf'),
            dtype=V_soft_bwd_sparse_coalesced.dtype,
            device=V_soft_bwd_sparse_coalesced.device
        )

        indices = V_soft_bwd_sparse_coalesced.indices()
        values = V_soft_bwd_sparse_coalesced.values()

        if values.numel() > 0:
            V_soft_bwd_dense_filled[tuple(indices)] = values
        
        V_soft_bwd_np = V_soft_bwd_dense_filled.cpu().numpy()

    except FileNotFoundError:
        print("Backward value function file not found. Please run backward_svi first.")
        print("Skipping TRes Sampler test.")
        return

    # Load edge costs
    try:
        edge_costs_path = os.path.join(config.output_dir, f"{config.file_prefix}_COST_WIND.pt")
        edge_costs_tensor, interpretation_note = load_sparse_coo_tensor_with_convention(
            edge_costs_path,
            target_device=device
        )
        print(f"Edge costs loaded. Interpretation Note: {interpretation_note}")
    except FileNotFoundError:
        print("Edge costs file not found. Please run backward_svi first.")
        print("Skipping TRes Sampler test.")
        return

    origin_node_id = config.origin_node
    goal_node_id = config.goal_node

    # Time parameters (consistent with backward_svi)
    # estimated_takeoff_time_str = config.estimated_takeoff_time_str
    # from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    # takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # min_wall_clock_time_sec = float(takeoff_ssm)
    # delta_t_wall_clock_sec = 300.0  # 5 minutes, from backward_svi test

    # Initial state parameters
    initial_k = 0
    
    if V_soft_bwd_dense_filled.shape[2] > 0:
        initial_rho = V_soft_bwd_dense_filled.shape[2] - 1 
    else:
        print("Error: Number of rho bins is 0. Cannot determine initial_rho.")
        return
        
    initial_phase = 0 # 0: CLIMB

    print(f"Starting sampling from {origin_node_id} (idx {node_to_idx[origin_node_id]}) to {goal_node_id} (idx {node_to_idx[goal_node_id]})")
    print(f"Initial state: k={initial_k}, rho={initial_rho}, phase={initial_phase}")
    print(f"V_bwd shape: {V_soft_bwd_dense_filled.shape}")
    print(f"Edge costs indices shape: {edge_costs_tensor.indices().shape}, values shape: {edge_costs_tensor.values().shape}")


    num_samples = 100
    successful_samples = 0
    failed_samples = 0
    trajectories = []
    all_trajectory_costs = []

    for i in range(num_samples):
        print(f"\nSampling trajectory {i+1}/{num_samples}...")
        trajectory, trajectory_costs = sample_tres_trajectory(
            G=G,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            origin_node_id=origin_node_id,
            goal_node_id=goal_node_id,
            initial_rho=initial_rho,
            initial_phase=initial_phase,
            soft_cost_to_go=V_soft_bwd_dense_filled, # This is V_bwd
            edge_costs_uv=edge_costs_tensor,
            max_steps=200, # Max steps per trajectory
            gamma = config.gamma
        )
        if trajectory:
            successful_samples += 1
            trajectories.append(trajectory)
            all_trajectory_costs.append(trajectory_costs)
            print(f"  Successfully sampled trajectory {i+1} with {len(trajectory)} steps.")
            total_cost = np.sum(trajectory_costs)
            print(f"    Total cost: {total_cost:.4f}")
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

        all_total_costs = [np.sum(c) for c in all_trajectory_costs]
        if all_total_costs:
            avg_cost = np.mean(all_total_costs)
            min_cost = np.min(all_total_costs)
            max_cost = np.max(all_total_costs)
            print(f"Average trajectory cost: {avg_cost:.2f}")
            print(f"Min trajectory cost: {min_cost:.2f}")
            print(f"Max trajectory cost: {max_cost:.2f}")

    # Save trajectories to a file
    trajectories_dir = config.output_dir
    os.makedirs(trajectories_dir, exist_ok=True)
    
    # Save trajectories as text file with waypoint names separated by whitespace
    trajectories_file_path = os.path.join(trajectories_dir, f"{config.file_prefix}_CLB_trajectories.txt")
    with open(trajectories_file_path, "w") as f:
        for i, trajectory in enumerate(trajectories):
            waypoint_names = [str(step[0]) for step in trajectory]
            trajectory_line = " ".join(waypoint_names)
            total_cost = np.sum(all_trajectory_costs[i])
            f.write(f"{total_cost},{trajectory_line}\n")
    
    print(f"Saved {len(trajectories)} trajectories to {trajectories_file_path}")


def run_backward_svi_wrapper(config, components):
    """Wrapper function for backward_svi to run in separate process"""
    print("Starting backward SVI in parallel process...")
    return backward_svi(config, components, headless=True)


def run_forward_svi_wrapper(config, components):
    """Wrapper function for forward_svi to run in separate process"""
    print("Starting forward SVI in parallel process...")
    return forward_svi(config, components, headless=True)


def run_forward_tres_wrapper(config, components):
    """Wrapper function for forward_tres to run in separate process"""
    print("Starting forward tres in parallel process...")
    return forward_tres(config, components)


def run_backward_tres_wrapper(config, components):
    """Wrapper function for backward_tres to run in separate process"""
    print("Starting backward tres in parallel process...")
    return backward_tres(config, components)


def run_tres_parallel(config, components):
    """Run both forward and backward tres passes in parallel using multiprocessing"""
    print("Running forward and backward tres passes in parallel...")
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_forward = executor.submit(run_forward_tres_wrapper, config, components)
        future_backward = executor.submit(run_backward_tres_wrapper, config, components)
        
        # Wait for both to complete and get results
        results = {}
        for future in as_completed([future_forward, future_backward]):
            try:
                if future == future_forward:
                    results['forward'] = future.result()
                    print("Forward tres completed successfully!")
                elif future == future_backward:
                    results['backward'] = future.result()
                    print("Backward tres completed successfully!")
            except Exception as exc:
                if future == future_forward:
                    print(f"Forward tres generated an exception: {exc}")
                elif future == future_backward:
                    print(f"Backward tres generated an exception: {exc}")
                raise
    
    print("Both tres computations completed!")
    return results


def run_svi_parallel(config, components):
    """Run both backward and forward SVI in parallel using multiprocessing"""
    print("Running backward and forward SVI in parallel...")
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_backward = executor.submit(run_backward_svi_wrapper, config, components)
        future_forward = executor.submit(run_forward_svi_wrapper, config, components)
        
        # Wait for both to complete and get results
        results = {}
        for future in as_completed([future_backward, future_forward]):
            try:
                if future == future_backward:
                    results['backward'] = future.result()
                    print("Backward SVI completed successfully!")
                elif future == future_forward:
                    results['forward'] = future.result()
                    print("Forward SVI completed successfully!")
            except Exception as exc:
                if future == future_backward:
                    print(f"Backward SVI generated an exception: {exc}")
                elif future == future_forward:
                    print(f"Forward SVI generated an exception: {exc}")
                raise
    
    print("Both SVI computations completed!")
    return results


if __name__ == '__main__':
    import time
    time_start = time.time()
    
    # Load configuration
    CONFIG_PATH = "data/profiles/lemd_egll_flight0.yaml"
    config = RunConfiguration.load_from_yaml(CONFIG_PATH)
    
    # Initialize components - CostRev4 will be used based on configuration
    components = config.initialize_all_components(manual_cost_model_init = True)
    # Because we are using disable_config_wind_model in the settings yaml file, we need to manually initialize the wind model
    components['wind_model'] = WindDate(date_str=config.wind_date, data_dir=config.wind_data_dir)
    # Initialize the cost model using CostRev4Lite as specified in the configuration
    cost_model = config.initialize_cost_model(num_waypoints=components['num_nodes'])

    # ATTENTION: For CostRev4Lite, we do not need to load the checkpoint because the weights are fixed in the model.
    # Load the checkpoint
    # checkpoint_path = config.checkpoint_path
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path, map_location=components['device'], weights_only=False)
    #     cost_model.load_state_dict(checkpoint['model_state_dict'])
    #     print("Cost model state loaded from checkpoint.")
    # else:
    #     raise ValueError(f"Checkpoint file not found at {checkpoint_path}")
    
    components['cost_model'] = cost_model

    # ================================
    # Run tres passes (they CANNOT be run in parallel because backward_tres depends on the forward passes)
    run_forward_tres_wrapper(config, components)
    run_backward_tres_wrapper(config, components)

    thinning(config, components)
    amortize_wind_average(config, components)
    # ================================
    # # Run both SVI functions in parallel
    run_svi_parallel(config, components)
    # # forward_svi(config, components, headless=False)
    # backward_svi(config, components, headless=False)
    
    # print('CAUTION: The forward SVI contains a hard-coded initial log-mass. This should be corrected in the future.')
    test_tres_sampler(config, components, headless=False)

    # backward_gradient_pass_test(config, components, headless=True)

    time_end = time.time()
    print(f"Total clock time: {time_end - time_start} seconds")