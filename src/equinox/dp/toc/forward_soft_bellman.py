import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from equinox.route.forward_state import get_next_state_fw, CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.wind.wind_model import WindModel
from typing import List, Tuple

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts

# It's good practice to define default narrow body jet profiles here or pass them
# For now, assuming they might be part of performance_model_params or loaded similarly
# from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE


def run_forward_soft_bellman(
    graph: nx.DiGraph,
    source_node_id: str, # Graph node ID (e.g., 'LEMD')
    takeoff_time_str: str, # e.g., "2023-04-01 12:00:00"
    source_elevation_ft: float, # Elevation of the source airport/node
    goal_elevation_ft: float, # Elevation of the goal airport/node (unused in fwd pass)
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    initial_alt_ft: float = 1000.0, # Initial altitude at source node relative to its elevation after takeoff
    delta_t_seconds: int = 300, # 5 minutes time window for the clock time bin
    max_flight_duration_hours: int = 10, # Max duration to consider for time bins
    etto_delta_t_seconds: int = 30, # the time window length for elapsed time since takeoff, to store 
    max_elapsed_time_since_takeoff_hours: float = 0.75, # the maximum elapsed time since takeoff to consider for the elapsed time bins
    device: torch.device = None
):
    """
    Implements the forward dynamic programming algorithm for soft Bellman updates,
    processing nodes in topological generations for enhanced batching.

    Args:
        graph (nx.DiGraph): The route graph. Nodes should have a 'coords' attribute (lat, lon).
        source_node_id (str): The ID of the source node in the graph.
        takeoff_time_str (str): ISO format takeoff time string (e.g., "2023-04-01 12:00:00").
        source_elevation_ft (float): Elevation of the source airport in feet.
        goal_elevation_ft (float): Elevation of the destination airport in feet (unused in forward pass).
        cost_model (CostRev1): Instantiated cost model.
        wind_model (WindModel): Instantiated wind model.
        performance_model (Performance): Instantiated aircraft performance model.
        dist_matrix_np (np.ndarray): 2D array of distances between node indices (in nautical miles).
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        initial_alt_ft (float, optional): Initial altitude at the source node (in feet AMSL) at takeoff_time_str.
        delta_t_seconds (int, optional): Duration of each time bin in seconds.
        max_flight_duration_hours (int, optional): Maximum flight duration to define the number of time bins.
        device (torch.device, optional): PyTorch device to run computations on.

    Returns:
        torch.Tensor: Value function V[node_idx, time_bin_idx, etto_bin_idx].
        torch.Tensor: active_eta[node_idx, time_bin_idx, etto_bin_idx] (exact seconds since midnight)
        torch.Tensor: active_alt[node_idx, time_bin_idx, etto_bin_idx] (altitude in ft AMSL).
        torch.Tensor: active_phase[node_idx, time_bin_idx, etto_bin_idx] (flight phase).
        List[Tuple[int, int, int, int]]: transitions_list containing (u_node_idx, eps_u_idx, v_node_idx, eps_v_idx)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialization ---
    node_list = list(graph.nodes()) # Keep original node IDs for graph access
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if source_node_id not in node_to_idx:
        raise ValueError(f"Source node {source_node_id} not found in graph.")
    s_idx = node_to_idx[source_node_id]

    takeoff_seconds_since_midnight = datestr_to_seconds_since_midnight(takeoff_time_str)
    
    min_time_overall_seconds = takeoff_seconds_since_midnight
    max_time_overall_seconds = min_time_overall_seconds + max_flight_duration_hours * 3600
    num_time_bins = int((max_time_overall_seconds - min_time_overall_seconds) / delta_t_seconds) + 1

    # ETTO (Elapsed Time Since TakeOff) bins
    # Climb performance table is (time_sec, dist_nm, alt_ft, tas_kts, vs_fpm, fuel_flow_pph)
    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=source_elevation_ft)
    actual_max_climb_time_sec = climb_perf_table[-1][1] # Time to reach TOC

    # Define num_etto_bins based on the user-provided max_elapsed_time_since_takeoff_hours
    # The actual ETTO values will be clamped by actual_max_climb_time_sec later for bin assignment.
    max_etto_config_sec = max_elapsed_time_since_takeoff_hours * 3600
    num_etto_bins = int(max_etto_config_sec / etto_delta_t_seconds) + 1
    
    # Determine the ETTO bin index that corresponds to the actual_max_climb_time_sec
    # This bin and any subsequent bins (due to max_etto_config_sec being larger) effectively represent cruise ETTO.
    if etto_delta_t_seconds > 0:
        max_climb_etto_bin_idx = int(torch.round(torch.tensor(actual_max_climb_time_sec / etto_delta_t_seconds)).item())
        max_climb_etto_bin_idx = min(max_climb_etto_bin_idx, num_etto_bins - 1) # Clamp to max defined bins
    else: # Avoid division by zero if etto_delta_t_seconds is 0
        max_climb_etto_bin_idx = 0 if actual_max_climb_time_sec == 0 else num_etto_bins -1


    V = torch.full((num_nodes, num_time_bins, num_etto_bins), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins, num_etto_bins), float('nan'), dtype=torch.float64, device=device)
    active_phase = torch.full((num_nodes, num_time_bins, num_etto_bins), -1, dtype=torch.long, device=device)
    active_eta = torch.full((num_nodes, num_time_bins, num_etto_bins), float('nan'), dtype=torch.float64, device=device) # in seconds since midnight (ssm)
    
    transitions_list: List[Tuple[int, int, int, int]] = []


    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    initial_time_bin = 0
    initial_eps_sec = 0.0 # Elapsed time at takeoff is 0
    initial_eps_idx = 0 # Corresponds to initial_eps_sec
    if not (0 <= initial_eps_idx < num_etto_bins): # Should always be true if num_etto_bins > 0
        raise ValueError(f"Initial ETTO bin index {initial_eps_idx} is out of bounds [0, {num_etto_bins-1}]")


    V[s_idx, initial_time_bin, initial_eps_idx] = 0.0
    active_alt[s_idx, initial_time_bin, initial_eps_idx] = float(initial_alt_ft) 
    active_phase[s_idx, initial_time_bin, initial_eps_idx] = CLIMB 
    active_eta[s_idx, initial_time_bin, initial_eps_idx] = float(takeoff_seconds_since_midnight)

    # --- 2. Topological Generations for Node Processing Order ---
    try:
        # nx.topological_generations returns an iterator of sets of nodes.
        # Each set contains nodes where all predecessors are in previous sets.
        topo_generations_node_ids = list(nx.topological_generations(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph is not a DAG, cannot perform topological sort for generations.")

    # --- 3. Main DP Loop ---
    for generation_node_ids in tqdm(topo_generations_node_ids, desc="Topological Generations"):
        # Batch lists for all transitions from the current generation of nodes
        batch_coords_src_list = []
        batch_alts_src_list = []
        batch_eta_src_list = []
        batch_phase_src_list = []
        batch_coords_tgt_list = []
        batch_v_node_indices_list = [] 
        batch_u_indices_for_cost_list = [] 
        batch_v_indices_for_cost_list = [] 
        batch_V_u_k_eps_list = [] 
        batch_eps_u_idx_list = []

        for u_node_graph_id in generation_node_ids:
            u_node_idx = node_to_idx[u_node_graph_id]
            successors = list(graph.successors(u_node_graph_id)) # Use graph_id for graph ops
            if not successors:
                continue

            for k_u in range(num_time_bins):
                for eps_u_idx in range(num_etto_bins):
                    if not torch.isinf(V[u_node_idx, k_u, eps_u_idx]):
                        current_alt_u = active_alt[u_node_idx, k_u, eps_u_idx]
                        current_phase_u = active_phase[u_node_idx, k_u, eps_u_idx]
                        current_eta_u = active_eta[u_node_idx, k_u, eps_u_idx]
                        
                        if torch.isnan(current_alt_u) or current_phase_u == -1 or torch.isnan(current_eta_u):
                            continue 

                        u_coords_tuple = (graph.nodes[u_node_graph_id].get('lat'), graph.nodes[u_node_graph_id].get('lon'))
                        u_coords = [u_coords_tuple[0].item() if isinstance(u_coords_tuple[0], torch.Tensor) else float(u_coords_tuple[0]),
                                    u_coords_tuple[1].item() if isinstance(u_coords_tuple[1], torch.Tensor) else float(u_coords_tuple[1])]
                        
                        for v_node_id_succ in successors:
                            v_node_idx_succ = node_to_idx[v_node_id_succ]
                            v_coords_tuple = (graph.nodes[v_node_id_succ].get('lat'), graph.nodes[v_node_id_succ].get('lon'))
                            v_coords = [v_coords_tuple[0].item() if isinstance(v_coords_tuple[0], torch.Tensor) else float(v_coords_tuple[0]),
                                        v_coords_tuple[1].item() if isinstance(v_coords_tuple[1], torch.Tensor) else float(v_coords_tuple[1])]

                            batch_coords_src_list.append(u_coords)
                            batch_alts_src_list.append(current_alt_u.item())
                            batch_eta_src_list.append(current_eta_u.item())
                            batch_phase_src_list.append(current_phase_u.item())
                            batch_coords_tgt_list.append(v_coords)
                            batch_v_node_indices_list.append(v_node_idx_succ)
                            batch_u_indices_for_cost_list.append(u_node_idx) 
                            batch_v_indices_for_cost_list.append(v_node_idx_succ)
                            batch_V_u_k_eps_list.append(V[u_node_idx, k_u, eps_u_idx].item())
                            batch_eps_u_idx_list.append(eps_u_idx)
        
        # Process the accumulated batch for the current generation
        if not batch_coords_src_list:
            continue

        coords_src_tensor = torch.tensor(batch_coords_src_list, dtype=torch.float64, device=device)
        alts_src_tensor = torch.tensor(batch_alts_src_list, dtype=torch.float64, device=device)
        eta_src_tensor = torch.tensor(batch_eta_src_list, dtype=torch.float64, device=device) # seconds since midnight
        phase_src_tensor = torch.tensor(batch_phase_src_list, dtype=torch.long, device=device)
        coords_tgt_tensor = torch.tensor(batch_coords_tgt_list, dtype=torch.float64, device=device)
        
        # import time
        # time_start = time.time()
        alt_v_new_batch, eta_v_new_batch, phase_v_new_batch = get_next_state_fw(
            coords_src_tensor, alts_src_tensor, eta_src_tensor, phase_src_tensor,
            coords_tgt_tensor, climb_perf_table, wind_model
        )
        # time_end = time.time()
        # print(f"Time taken for get_next_state_fw: {time_end - time_start} seconds")
        
        tailwind_mps_batch = get_wind(
            coords_src_tensor, coords_tgt_tensor, alts_src_tensor, eta_src_tensor, wind_model
        )
        tailwind_kts_batch = tailwind_mps_batch * MPS_TO_KNOTS
        
        # u_indices for cost model are from batch_u_indices_for_cost_list
        u_indices_cost_tensor = torch.tensor(batch_u_indices_for_cost_list, dtype=torch.long, device=device)
        v_indices_cost_tensor = torch.tensor(batch_v_indices_for_cost_list, dtype=torch.long, device=device)
        
        cost_uv_batch = cost_model(
            (u_indices_cost_tensor, v_indices_cost_tensor),
            dist_matrix, ac_matrix, tailwind_kts_batch.to(dtype=torch.float32) 
        )
        
        V_u_k_eps_tensor = torch.tensor(batch_V_u_k_eps_list, dtype=torch.float64, device=device)
        
        for i in range(len(alt_v_new_batch)): # i is the index of the transition from one (u, k_u) to one of its successor nodes (v, _)
            v_node_idx = batch_v_node_indices_list[i] # This is the successor's index
            u_node_idx_for_trans = batch_u_indices_for_cost_list[i] # original u_node_idx for this transition
            eps_u_idx_for_trans = batch_eps_u_idx_list[i]

            alt_v_new = alt_v_new_batch[i]
            eta_v_new = eta_v_new_batch[i]
            phase_v_new = phase_v_new_batch[i]
            cost_uv = cost_uv_batch[i]
            V_u_val = V_u_k_eps_tensor[i] # This is V[u_original, k_u_original, eps_u_original]

            if torch.isinf(cost_uv) or torch.isinf(V_u_val):
                continue

            # Elapsed time since takeoff for the successor state v
            elapsed_time_v_sec = eta_v_new.item() - takeoff_seconds_since_midnight
            if elapsed_time_v_sec < 0:
                 continue
            
            # Clock time bin index k_v for the successor v
            clock_time_since_first_bin_sec = eta_v_new.item() - min_time_overall_seconds # min_time_overall is takeoff_seconds_since_midnight
            k_v = int(torch.round(torch.tensor(clock_time_since_first_bin_sec / delta_t_seconds)).item())


            if not (0 <= k_v < num_time_bins):
                continue
            
            # ETTO bin index eps_v_idx for the successor v
            # Clamp the ETTO to actual_max_climb_time_sec for bin calculation
            effective_etto_v_for_bin_idx_calc = min(elapsed_time_v_sec, actual_max_climb_time_sec)
            if etto_delta_t_seconds > 0:
                eps_v_idx = int(torch.round(torch.tensor(effective_etto_v_for_bin_idx_calc / etto_delta_t_seconds)).item())
            else: # Avoid division by zero
                eps_v_idx = 0
            eps_v_idx = max(0, min(eps_v_idx, num_etto_bins - 1)) # Ensure within bounds [0, num_etto_bins-1]


            val_to_add_in_exp = V_u_val + cost_uv.double() # V_u_val is V[u, k_u, eps_u_idx]
            current_V_v_k_eps = V[v_node_idx, k_v, eps_v_idx]
            
            if torch.isinf(current_V_v_k_eps):
                V[v_node_idx, k_v, eps_v_idx] = val_to_add_in_exp
                active_alt[v_node_idx, k_v, eps_v_idx] = alt_v_new
                active_phase[v_node_idx, k_v, eps_v_idx] = phase_v_new
                active_eta[v_node_idx, k_v, eps_v_idx] = eta_v_new
            else:
                V[v_node_idx, k_v, eps_v_idx] = -torch.logaddexp(
                    -current_V_v_k_eps,
                    -val_to_add_in_exp
                )

            # Store transition if not a cruise-to-cruise based on ETTO bins
            # A state is considered "cruise ETTO" if its ETTO bin is at or beyond max_climb_etto_bin_idx
            is_u_cruise_etto = eps_u_idx_for_trans >= max_climb_etto_bin_idx
            is_v_cruise_etto = eps_v_idx >= max_climb_etto_bin_idx
            
            if not (is_u_cruise_etto and is_v_cruise_etto):
                transitions_list.append((u_node_idx_for_trans, eps_u_idx_for_trans, v_node_idx, eps_v_idx))

    return V, active_eta, active_alt, active_phase, transitions_list