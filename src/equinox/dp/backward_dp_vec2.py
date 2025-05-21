import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from equinox.route.backward_state import get_next_state_bw
from equinox.route.forward_state import CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_descent
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.wind.wind_model import WindModel

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts

# It's good practice to define default narrow body jet profiles here or pass them
# For now, assuming they might be part of performance_model_params or loaded similarly
# from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE


def run_backward_dp(
    graph: nx.DiGraph,
    goal_node_id: str,
    estimated_landing_time_str: str,
    origin_elevation_ft: float,
    destination_elevation_ft: float,
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    final_alt_ft: float = 0.0,
    delta_t_seconds: int = 300,
    max_flight_duration_hours: int = 10,
    device: torch.device = None,
    temperature: float = 1.0
):
    """
    Implements the backward dynamic programming algorithm for soft Bellman updates,
    processing nodes in topological generations (of the reversed graph) for enhanced batching.

    Args:
        graph (nx.DiGraph): The route graph. Nodes should have a 'coords' attribute (lat, lon).
        goal_node_id (str): The ID of the goal node in the graph.
        estimated_landing_time_str (str): ISO format estimated landing time string (e.g., "2023-04-01 18:00:00").
        origin_elevation_ft (float): Elevation of the origin airport in feet (used for descent profile context).
        destination_elevation_ft (float): Elevation of the destination airport in feet.
        cost_model (CostRev1): Instantiated cost model.
        wind_model (WindModel): Instantiated wind model.
        performance_model (Performance): Instantiated aircraft performance model.
        dist_matrix_np (np.ndarray): 2D array of distances between node indices (in nautical miles).
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        final_alt_ft (float, optional): Altitude at the goal node (in feet AMSL) at estimated_landing_time_str.
                                       Defaults to destination_elevation_ft if not specified, but param allows override.
        delta_t_seconds (int, optional): Duration of each time bin in seconds.
        max_flight_duration_hours (int, optional): Maximum flight duration to define the number of time bins.
        device (torch.device, optional): PyTorch device to run computations on.

    Returns:
        torch.Tensor: Value function V[node_idx, time_bin_idx] (cost-to-go).
        torch.Tensor: active_eta[node_idx, time_bin_idx] (exact seconds since midnight) - the ETA of the path from (i, k_i) to goal.
        torch.Tensor: active_alt[node_idx, time_bin_idx] (altitude in ft AMSL).
        torch.Tensor: active_phase[node_idx, time_bin_idx] (flight phase).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialization ---
    node_list = list(graph.nodes()) # Keep original node IDs for graph access
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if goal_node_id not in node_to_idx:
        raise ValueError(f"Goal node {goal_node_id} not found in graph.")
    g_idx = node_to_idx[goal_node_id] # Goal node index

    estimated_landing_seconds_since_midnight = datestr_to_seconds_since_midnight(estimated_landing_time_str)
    
    # Time window calculated backwards from landing time
    max_time_overall_seconds = estimated_landing_seconds_since_midnight
    min_time_overall_seconds = max_time_overall_seconds - max_flight_duration_hours * 3600
    num_time_bins = int((max_time_overall_seconds - min_time_overall_seconds) / delta_t_seconds) + 1

    V = torch.full((num_nodes, num_time_bins), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device)
    active_phase = torch.full((num_nodes, num_time_bins), -1, dtype=torch.long, device=device)
    active_eta = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device) # in seconds since midnight (ssm)

    # Descent performance table relative to destination airport elevation.
    # Altitudes in get_next_state_bw and stored in active_alt should be AMSL.
    # The `destination_airport_elevation_ft` for get_eta_and_distance_descent is the actual landing field elevation.
    descent_perf_table = get_eta_and_distance_descent(performance_model, destination_airport_elevation_ft=destination_elevation_ft)
    
    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    # Initialize at goal node, at the last effective time bin
    # The time bin corresponding to estimated_landing_seconds_since_midnight
    goal_node_time_since_min_overall = estimated_landing_seconds_since_midnight - min_time_overall_seconds
    landing_time_bin = int(round(goal_node_time_since_min_overall / delta_t_seconds))
    
    # Ensure landing_time_bin is within bounds, typically num_time_bins - 1 if max_flight_duration makes sense
    if not (0 <= landing_time_bin < num_time_bins):
        # This might happen if estimated_landing_time is outside the window defined by max_flight_duration
        # Forcing it to the last bin, or could raise error.
        # If landing_time_bin is num_time_bins, it implies it's exactly at max_time_overall_seconds + delta_t_seconds/2 effectively due to rounding.
        # if it's num_time_bins, it's out of bounds for 0-indexed.
        landing_time_bin = num_time_bins -1 
        # We should also ensure estimated_landing_seconds_since_midnight aligns with this bin's center or start.
        # For simplicity, we assume it's correctly placed at this bin.

    V[g_idx, landing_time_bin] = 0.0
    # Altitude at goal node. If final_alt_ft was 0.0 (default), use destination_elevation_ft.
    current_final_alt = float(final_alt_ft if final_alt_ft != 0.0 else destination_elevation_ft)
    active_alt[g_idx, landing_time_bin] = current_final_alt
    active_phase[g_idx, landing_time_bin] = DESCENT # Assuming aircraft is in descent or landed phase at goal
    active_eta[g_idx, landing_time_bin] = float(estimated_landing_seconds_since_midnight)

    # --- 2. Topological Generations for Node Processing Order (Reversed Graph) ---
    try:
        graph_reversed = graph.reverse(copy=True)
        # nx.topological_generations returns an iterator of sets of nodes.
        # Each set contains nodes where all "predecessors" (in reversed graph, so original successors) are in previous sets.
        # This processes from goal backwards to source.
        topo_generations_node_ids = list(nx.topological_generations(graph_reversed))
    except nx.NetworkXUnfeasible:
        raise ValueError("Reversed graph is not a DAG, cannot perform topological sort for generations.")

    # --- 3. Main DP Loop ---
    # Iterate from goal backwards. `v_node_graph_id` is the current node being processed.
    # `u_node_id_pred` are its predecessors in the original graph (successors in reversed graph).
    for generation_node_ids in tqdm(topo_generations_node_ids, desc="Topological Generations (Backward)"):
        batch_coords_pred_list = [] # Coordinates of predecessor u
        batch_alts_v_curr_list = [] # Altitude at current node v
        batch_eta_v_curr_list = []  # ETA at current node v
        batch_phase_v_curr_list = []# Phase at current node v
        batch_coords_v_curr_list = []# Coordinates of current node v
        batch_u_node_indices_list = [] # Indices of predecessor u (for updating V[u,k_u])
        batch_u_indices_for_cost_list = [] 
        batch_v_indices_for_cost_list = [] 
        batch_V_v_kv_list = [] # V[v, k_v]

        for v_node_graph_id in generation_node_ids: # v is current node in backward pass
            v_node_idx = node_to_idx[v_node_graph_id]
            # Predecessors in original graph are successors in graph_reversed
            predecessors_orig_graph = list(graph_reversed.successors(v_node_graph_id)) 
            if not predecessors_orig_graph:
                continue

            for k_v in range(num_time_bins): # Time bin at current node v
                if not torch.isinf(V[v_node_idx, k_v]):
                    current_alt_v = active_alt[v_node_idx, k_v]
                    current_phase_v = active_phase[v_node_idx, k_v]
                    current_eta_v = active_eta[v_node_idx, k_v]
                    
                    if torch.isnan(current_alt_v) or current_phase_v == -1 or torch.isnan(current_eta_v):
                        continue 

                    v_coords_tuple = (graph.nodes[v_node_graph_id].get('lat'), graph.nodes[v_node_graph_id].get('lon'))
                    v_coords = [v_coords_tuple[0].item() if isinstance(v_coords_tuple[0], torch.Tensor) else v_coords_tuple[0],
                                v_coords_tuple[1].item() if isinstance(v_coords_tuple[1], torch.Tensor) else v_coords_tuple[1]]
                    
                    for u_node_id_pred in predecessors_orig_graph: # u is predecessor of v
                        u_node_idx_pred = node_to_idx[u_node_id_pred]
                        u_coords_tuple = (graph.nodes[u_node_id_pred].get('lat'), graph.nodes[u_node_id_pred].get('lon'))
                        u_coords = [u_coords_tuple[0].item() if isinstance(u_coords_tuple[0], torch.Tensor) else u_coords_tuple[0],
                                    u_coords_tuple[1].item() if isinstance(u_coords_tuple[1], torch.Tensor) else u_coords_tuple[1]]

                        batch_coords_pred_list.append(u_coords) # p_s in get_next_state_bw
                        batch_alts_v_curr_list.append(current_alt_v.item()) # alt_t in get_next_state_bw
                        batch_eta_v_curr_list.append(current_eta_v.item()) # eta_t in get_next_state_bw
                        batch_phase_v_curr_list.append(current_phase_v.item()) # phase_t in get_next_state_bw
                        batch_coords_v_curr_list.append(v_coords) # p_t in get_next_state_bw
                        
                        batch_u_node_indices_list.append(u_node_idx_pred) # For updating V[u, k_u]
                        batch_u_indices_for_cost_list.append(u_node_idx_pred) 
                        batch_v_indices_for_cost_list.append(v_node_idx) 
                        batch_V_v_kv_list.append(V[v_node_idx, k_v].item())
        
        if not batch_coords_pred_list:
            continue

        # Tensors for get_next_state_bw call
        coords_pred_tensor = torch.tensor(batch_coords_pred_list, dtype=torch.float64, device=device) # p_s
        alts_v_curr_tensor = torch.tensor(batch_alts_v_curr_list, dtype=torch.float64, device=device) # alt_t
        eta_v_curr_tensor = torch.tensor(batch_eta_v_curr_list, dtype=torch.float64, device=device)   # eta_t
        phase_v_curr_tensor = torch.tensor(batch_phase_v_curr_list, dtype=torch.long, device=device)   # phase_t
        coords_v_curr_tensor = torch.tensor(batch_coords_v_curr_list, dtype=torch.float64, device=device) # p_t
        
        # get_next_state_bw returns state at predecessor u (alt_s_out, eta_s_out, phase_s_out)
        alt_u_new_batch, eta_u_new_batch, phase_u_new_batch = get_next_state_bw(
            coords_src=coords_pred_tensor,  # p_s (predecessor u)
            alts_t=alts_v_curr_tensor,      # alt_v (current node v)
            eta_t=eta_v_curr_tensor,        # eta_v (current node v)
            phase_t=phase_v_curr_tensor,    # phase_v (current node v)
            coords_tgt=coords_v_curr_tensor,# p_t (current node v)
            descent_performance=descent_perf_table,
            wind_model=wind_model
        )
        
        # Wind for cost calculation of leg u -> v
        # Wind should be evaluated at the state of u (coords_pred_tensor, alt_u_new_batch, eta_u_new_batch)
        # for the segment from u (coords_pred_tensor) to v (coords_v_curr_tensor)
        tailwind_mps_batch = get_wind(
            coords_pred_tensor, coords_v_curr_tensor, # From u to v
            alt_u_new_batch,    # Altitude at u
            eta_u_new_batch,    # ETA at u (time for wind)
            wind_model
        )
        tailwind_kts_batch = tailwind_mps_batch * MPS_TO_KNOTS
        
        u_indices_cost_tensor = torch.tensor(batch_u_indices_for_cost_list, dtype=torch.long, device=device)
        v_indices_cost_tensor = torch.tensor(batch_v_indices_for_cost_list, dtype=torch.long, device=device)
        
        # Cost of traversing edge u -> v
        cost_uv_batch = cost_model(
            (u_indices_cost_tensor, v_indices_cost_tensor), # (u,v)
            dist_matrix, ac_matrix, tailwind_kts_batch.to(dtype=torch.float32) 
        )
        
        V_v_kv_tensor = torch.tensor(batch_V_v_kv_list, dtype=torch.float64, device=device)
        
        for i in range(len(alt_u_new_batch)): # i is the index of the transition from u to v
            u_node_idx = batch_u_node_indices_list[i] # This is the predecessor's index
            alt_u_new = alt_u_new_batch[i]
            eta_u_new = eta_u_new_batch[i] # This is ETA at node u
            phase_u_new = phase_u_new_batch[i]
            cost_uv = cost_uv_batch[i] # Cost from u to v
            V_v_val = V_v_kv_tensor[i] # V[v, k_v]

            if torch.isinf(cost_uv) or torch.isinf(V_v_val) or torch.isnan(eta_u_new) or alt_u_new < 0: # Check for invalid states from get_next_state_bw
                continue

            # Time bin for state at u
            time_at_u_since_min_overall = eta_u_new - min_time_overall_seconds
            if time_at_u_since_min_overall < -delta_t_seconds: # Allow some slack for rounding near min_time_overall_seconds
                 continue 
            
            k_u = int(torch.round(time_at_u_since_min_overall / delta_t_seconds).item())

            if not (0 <= k_u < num_time_bins):
                continue
            
            # Backward DP update: V(u) = cost(u,v) + V(v)
            val_to_add_in_exp = V_v_val + cost_uv.double() # Cost-to-go from u via v
            current_V_u_ku = V[u_node_idx, k_u]
            
            if torch.isinf(current_V_u_ku):
                V[u_node_idx, k_u] = val_to_add_in_exp
                active_alt[u_node_idx, k_u] = alt_u_new
                active_phase[u_node_idx, k_u] = phase_u_new
                active_eta[u_node_idx, k_u] = eta_u_new
            else:
                # Soft Bellman update (min convention for costs)
                V[u_node_idx, k_u] = -temperature * torch.logaddexp(
                    -current_V_u_ku / temperature,
                    -val_to_add_in_exp / temperature
                )
                # TODO: How to update active_alt, active_phase, active_eta with soft updates?
                # For now, if a new path contributes, we could check if it's "better" (lower cost component)
                # or average, or keep the one from the dominant term.
                # Simplest: if val_to_add_in_exp is lower than current_V_u_ku before logaddexp, update.
                # This isn't strictly correct for soft-min path reconstruction.
                # For now, let's assume the first path that sets it or a path that significantly lowers the raw sum is chosen.
                # A more rigorous approach might store multiple path attributes or use a policy.
                # Let's update if the new path is "better" (smaller value before softmin conversion)
                if val_to_add_in_exp < current_V_u_ku: # This comparison is not direct with logaddexp
                                                       # but gives a heuristic for state tracking.
                                                       # A true soft-min path would require more complex tracking.
                    active_alt[u_node_idx, k_u] = alt_u_new
                    active_phase[u_node_idx, k_u] = phase_u_new
                    active_eta[u_node_idx, k_u] = eta_u_new


    return V, active_eta, active_alt, active_phase
