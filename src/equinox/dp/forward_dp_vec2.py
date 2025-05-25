import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from equinox.route.forward_state import get_next_state_fw, CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.wind.wind_model import WindModel

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts

# It's good practice to define default narrow body jet profiles here or pass them
# For now, assuming they might be part of performance_model_params or loaded similarly
# from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE

# VEC2 VERSION ALLOWS ALIGNMENT OF TIME GRID BETWEEN FORWARD AND BACKWARD PASSES

def run_forward_dp(
    graph: nx.DiGraph,
    source_node_id: str, # Graph node ID (e.g., 'LEMD')
    source_elevation_ft: float, # Elevation of the source airport/node
    goal_elevation_ft: float, # Elevation of the goal airport/node (unused in fwd pass)
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    initial_alt_ft: float, # Initial altitude at source node AMSL
    delta_t_seconds: int,
    # Aligned time parameters
    min_time_overall_seconds_aligned: float,
    num_time_bins_aligned: int,
    takeoff_bin_idx_aligned: int,
    takeoff_seconds_since_midnight_val: float, # Renamed for clarity
    device: torch.device = None
):
    """
    Implements the forward dynamic programming algorithm for soft Bellman updates,
    processing nodes in topological generations for enhanced batching,
    using a pre-calculated common time grid for alignment with backward pass.

    Args:
        graph (nx.DiGraph): The route graph. Nodes should have a 'coords' attribute (lat, lon).
        source_node_id (str): The ID of the source node in the graph.
        source_elevation_ft (float): Elevation of the source airport in feet.
        goal_elevation_ft (float): Elevation of the destination airport in feet (unused in forward pass).
        cost_model (CostRev1): Instantiated cost model.
        wind_model (WindModel): Instantiated wind model.
        performance_model (Performance): Instantiated aircraft performance model.
        dist_matrix_np (np.ndarray): 2D array of distances between node indices (in nautical miles).
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        initial_alt_ft (float): Initial altitude at the source node (in feet AMSL) at takeoff_time_str.
        delta_t_seconds (int): Duration of each time bin in seconds.
        min_time_overall_seconds_aligned (float): Common reference start time (seconds since midnight) for the shared time grid.
        num_time_bins_aligned (int): Common number of time bins for DP arrays.
        takeoff_bin_idx_aligned (int): The bin index in the common grid for the takeoff time.
        takeoff_seconds_since_midnight_val (float): The takeoff time in seconds since midnight.
        device (torch.device, optional): PyTorch device to run computations on.

    Returns:
        torch.Tensor: Value function V[node_idx, time_bin_idx].
        torch.Tensor: active_eta[node_idx, time_bin_idx] (exact seconds since midnight) - the ETA of the first path arriving at state (i, k_i) (node i, time bin k_i)
        torch.Tensor: active_alt[node_idx, time_bin_idx] (altitude in ft AMSL).
        torch.Tensor: active_phase[node_idx, time_bin_idx] (flight phase).
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

    # Use aligned time parameters directly
    min_time_overall_seconds = min_time_overall_seconds_aligned
    num_time_bins = num_time_bins_aligned

    V = torch.full((num_nodes, num_time_bins), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device)
    active_phase = torch.full((num_nodes, num_time_bins), -1, dtype=torch.long, device=device)
    active_eta = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device) # in seconds since midnight (ssm)

    # Climb performance table is relative to origin airport elevation.
    # Altitudes in get_next_state_fw and stored in active_alt should be AMSL.
    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=source_elevation_ft) 
    
    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    # Use takeoff_bin_idx_aligned for initialization
    initial_time_bin = takeoff_bin_idx_aligned
    V[s_idx, initial_time_bin] = 0.0
    # Ensure initial_alt_ft is AMSL. If it was given as AGL for source, it should be adjusted before this call.
    # Assuming initial_alt_ft is already AMSL as per updated docstring.
    active_alt[s_idx, initial_time_bin] = float(initial_alt_ft) 
    active_phase[s_idx, initial_time_bin] = CLIMB 
    active_eta[s_idx, initial_time_bin] = float(takeoff_seconds_since_midnight_val) # Use passed takeoff time in ssm

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
        batch_V_u_ku_list = [] 

        for u_node_graph_id in generation_node_ids:
            u_node_idx = node_to_idx[u_node_graph_id]
            successors = list(graph.successors(u_node_graph_id)) # Use graph_id for graph ops
            if not successors:
                continue

            for k_u in range(num_time_bins):
                if not torch.isinf(V[u_node_idx, k_u]):
                    current_alt_u = active_alt[u_node_idx, k_u]
                    current_phase_u = active_phase[u_node_idx, k_u]
                    current_eta_u = active_eta[u_node_idx, k_u]
                    
                    if torch.isnan(current_alt_u) or current_phase_u == -1 or torch.isnan(current_eta_u):
                        continue 

                    u_coords_tuple = (graph.nodes[u_node_graph_id].get('lat'), graph.nodes[u_node_graph_id].get('lon'))
                    u_coords = [u_coords_tuple[0].item() if isinstance(u_coords_tuple[0], torch.Tensor) else u_coords_tuple[0],
                                u_coords_tuple[1].item() if isinstance(u_coords_tuple[1], torch.Tensor) else u_coords_tuple[1]]
                    
                    for v_node_id_succ in successors:
                        v_node_idx_succ = node_to_idx[v_node_id_succ]
                        v_coords_tuple = (graph.nodes[v_node_id_succ].get('lat'), graph.nodes[v_node_id_succ].get('lon'))
                        v_coords = [v_coords_tuple[0].item() if isinstance(v_coords_tuple[0], torch.Tensor) else v_coords_tuple[0],
                                    v_coords_tuple[1].item() if isinstance(v_coords_tuple[1], torch.Tensor) else v_coords_tuple[1]]

                        batch_coords_src_list.append(u_coords)
                        batch_alts_src_list.append(current_alt_u.item())
                        batch_eta_src_list.append(current_eta_u.item())
                        batch_phase_src_list.append(current_phase_u.item())
                        batch_coords_tgt_list.append(v_coords)
                        batch_v_node_indices_list.append(v_node_idx_succ)
                        batch_u_indices_for_cost_list.append(u_node_idx) # Store original u_node_idx for cost matrix
                        batch_v_indices_for_cost_list.append(v_node_idx_succ) # Store v_node_idx for cost matrix
                        batch_V_u_ku_list.append(V[u_node_idx, k_u].item())
        
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
        
        V_u_ku_tensor = torch.tensor(batch_V_u_ku_list, dtype=torch.float64, device=device)
        
        for i in range(len(alt_v_new_batch)): # i is the index of the transition from one (u, k_u) to one of its successor nodes (v, _)
            v_node_idx = batch_v_node_indices_list[i] # This is the successor's index
            alt_v_new = alt_v_new_batch[i]
            eta_v_new = eta_v_new_batch[i]
            phase_v_new = phase_v_new_batch[i]
            cost_uv = cost_uv_batch[i]
            # V_u_val corresponds to V[original_u_node_idx, original_k_u] for this transition
            V_u_val = V_u_ku_tensor[i]

            if torch.isinf(cost_uv) or torch.isinf(V_u_val):
                continue

            # time_since_takeoff_sec is now time_at_v_since_min_overall for consistency
            time_at_v_since_min_overall = eta_v_new - min_time_overall_seconds # Use the common min_time_overall_seconds
            if time_at_v_since_min_overall < -delta_t_seconds: # Allow some slack
                 continue
            
            # The time bin index k_v for the successor v
            k_v = int(torch.round(time_at_v_since_min_overall / delta_t_seconds).item())

            if not (0 <= k_v < num_time_bins):
                continue

            val_to_add_in_exp = V_u_val + cost_uv.double()
            current_V_v_kv = V[v_node_idx, k_v]
            
            if torch.isinf(current_V_v_kv):
                V[v_node_idx, k_v] = val_to_add_in_exp
                active_alt[v_node_idx, k_v] = alt_v_new
                active_phase[v_node_idx, k_v] = phase_v_new
                active_eta[v_node_idx, k_v] = eta_v_new
            else:
                V[v_node_idx, k_v] = -torch.logaddexp(
                    -current_V_v_kv,
                    -val_to_add_in_exp
                )

    return V, active_eta, active_alt, active_phase
