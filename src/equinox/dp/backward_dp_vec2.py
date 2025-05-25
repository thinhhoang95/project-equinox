import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from equinox.route.backward_state import get_next_state_bw
from equinox.route.forward_state import CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.wind.wind_model import WindModel
from typing import Dict, Tuple, List

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts

# It's good practice to define default narrow body jet profiles here or pass them
# For now, assuming they might be part of performance_model_params or loaded similarly
# from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE

# VEC2 VERSION: RETURNS THE GRADIENTS, EDGE COSTS, AND EDGE TO CANONICAL INDEX MAPPING
# VEC2 VERSION ALSO ALLOWS ALIGNMENT OF TIME GRID BETWEEN FORWARD AND BACKWARD PASSES

def run_backward_dp(
    graph: nx.DiGraph,
    goal_node_id: str,
    origin_elevation_ft: float,
    destination_elevation_ft: float,
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    final_alt_ft: float,
    delta_t_seconds: int,
    min_time_overall_seconds_aligned: float,
    num_time_bins_aligned: int,
    landing_bin_idx_aligned: int,
    landing_seconds_since_midnight_val: float,
    device: torch.device = None,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[Tuple[int, int], int]]:
    """
    Implements the backward dynamic programming algorithm for soft Bellman updates,
    processing nodes in topological generations (of the reversed graph) for enhanced batching,
    using a pre-calculated common time grid for alignment with forward pass.
    Also computes and returns edge costs and their gradients with respect to cost function parameters.

    Args:
        graph (nx.DiGraph): The route graph. Nodes should have a 'coords' attribute (lat, lon).
        goal_node_id (str): The ID of the goal node in the graph.
        origin_elevation_ft (float): Elevation of the origin airport in feet (used for descent profile context).
        destination_elevation_ft (float): Elevation of the destination airport in feet.
        cost_model (CostRev1): Instantiated cost model. Its parameters intended for learning should have \`requires_grad=True\`.
        wind_model (WindModel): Instantiated wind model.
        performance_model (Performance): Instantiated aircraft performance model.
        dist_matrix_np (np.ndarray): 2D array of distances between node indices (in nautical miles).
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        final_alt_ft (float): Altitude at the goal node (in feet AMSL) at estimated_landing_time_str.
                                       Defaults to destination_elevation_ft if not specified, but param allows override.
        delta_t_seconds (int): Duration of each time bin in seconds.
        min_time_overall_seconds_aligned (float): Common reference start time (seconds since midnight) for the shared time grid.
        num_time_bins_aligned (int): Common number of time bins for DP arrays.
        landing_bin_idx_aligned (int): The bin index in the common grid for the landing time.
        landing_seconds_since_midnight_val (float): The landing time in seconds since midnight.
        device (torch.device, optional): PyTorch device to run computations on.
        temperature (float, optional): Temperature parameter for the soft Bellman update.
    Returns:
        torch.Tensor: Value function V[node_idx, time_bin_idx] (cost-to-go).
        torch.Tensor: active_eta[node_idx, time_bin_idx] (exact seconds since midnight) - the ETA of the path from (i, k_i) to goal.
        torch.Tensor: active_alt[node_idx, time_bin_idx] (altitude in ft AMSL).
        torch.Tensor: active_phase[node_idx, time_bin_idx] (flight phase).
        torch.Tensor: edge_costs_time_binned[edge_canonical_idx, time_bin_idx] (cost of traversing edge at a time bin).
        torch.Tensor: edge_cost_gradients_time_binned[edge_canonical_idx, time_bin_idx, param_idx] (gradients of edge costs).
        Dict[Tuple[int, int], int]: edge_to_canonical_idx: Maps (u_node_idx, v_node_idx) to edge_canonical_idx.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_model.to(device) # Ensure cost_model is on the correct device

    # --- 0. Prepare Edge Mappings and Cost Parameter Info ---
    node_list = list(graph.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    # Create a canonical mapping for edges present in the graph
    # Edges are defined by (u_node_idx, v_node_idx)
    edge_list_with_indices: List[Tuple[int, int]] = []
    for u_str, v_str in graph.edges():
        if u_str in node_to_idx and v_str in node_to_idx:
            u_idx, v_idx = node_to_idx[u_str], node_to_idx[v_str]
            # Ensure u_idx and v_idx are valid indices for dist_matrix and ac_matrix
            if 0 <= u_idx < num_nodes and 0 <= v_idx < num_nodes:
                 edge_list_with_indices.append((u_idx, v_idx))
            # else:
            #     print(f"Warning: Edge ({u_str}, {v_str}) with indices ({u_idx}, {v_idx}) out of bounds for num_nodes {num_nodes}. Skipping.")


    unique_edges = sorted(list(set(edge_list_with_indices))) # Sort for consistent indexing
    edge_to_canonical_idx: Dict[Tuple[int, int], int] = {edge: i for i, edge in enumerate(unique_edges)}
    num_unique_edges = len(unique_edges)
    
    if num_unique_edges == 0 and graph.number_of_edges() > 0:
        print("Warning: num_unique_edges is 0, but graph has edges. Check node_to_idx mapping or graph structure.")
    elif num_unique_edges == 0 and graph.number_of_edges() == 0:
        print("Info: Graph has no edges. Edge cost and gradient tensors will be empty.")


    learnable_cost_params = [p for p in cost_model.parameters() if p.requires_grad]
    num_cost_params = sum(p.numel() for p in learnable_cost_params)
    
    if not learnable_cost_params:
        print("Warning: No learnable parameters (with requires_grad=True) found in cost_model. Gradients will be zero or empty.")


    # --- 1. Initialization ---
    if goal_node_id not in node_to_idx:
        raise ValueError(f"Goal node {goal_node_id} not found in graph.")
    g_idx = node_to_idx[goal_node_id] # Goal node index

    # Use aligned time parameters directly
    min_time_overall_seconds = min_time_overall_seconds_aligned
    num_time_bins = num_time_bins_aligned
    # estimated_landing_seconds_since_midnight is now landing_seconds_since_midnight_val
    # goal_node_time_since_min_overall and landing_time_bin calculation removed, use landing_bin_idx_aligned

    V = torch.full((num_nodes, num_time_bins), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device)
    active_phase = torch.full((num_nodes, num_time_bins), -1, dtype=torch.long, device=device)
    active_eta = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device) # in seconds since midnight (ssm)

    # Initialize tensors for edge costs and gradients
    # Handle cases with no edges or no time bins
    if num_unique_edges > 0 and num_time_bins > 0:
        edge_costs_time_binned = torch.full((num_unique_edges, num_time_bins), float('nan'), dtype=torch.float64, device=device)
        if num_cost_params > 0:
            edge_cost_gradients_time_binned = torch.full((num_unique_edges, num_time_bins, num_cost_params), float('nan'), dtype=torch.float64, device=device)
        else: # No learnable parameters
            edge_cost_gradients_time_binned = torch.empty((num_unique_edges, num_time_bins, 0), dtype=torch.float64, device=device)
    else: # No edges or no time bins, create empty tensors with correct number of dimensions
        edge_costs_time_binned = torch.empty((num_unique_edges, num_time_bins), dtype=torch.float64, device=device)
        edge_cost_gradients_time_binned = torch.empty((num_unique_edges, num_time_bins, num_cost_params if num_cost_params > 0 else 0), dtype=torch.float64, device=device)


    # Descent performance table relative to destination airport elevation.
    # Altitudes in get_next_state_bw and stored in active_alt should be AMSL.
    # The \`destination_airport_elevation_ft\` for get_eta_and_distance_descent is the actual landing field elevation.
    descent_perf_table = get_eta_and_distance_descent(performance_model, destination_airport_elevation_ft=destination_elevation_ft)
    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=origin_elevation_ft)
    
    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device) # Ensure float64 for consistency with V
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)   # Ensure float64 for consistency with V

    # Initialize at goal node using landing_bin_idx_aligned
    landing_time_bin = landing_bin_idx_aligned
    
    # Ensure landing_time_bin is within bounds (already validated by calculate_aligned_time_parameters)
    # if not (0 <= landing_time_bin < num_time_bins): 
    #     landing_time_bin = num_time_bins -1 

    V[g_idx, landing_time_bin] = 0.0
    # Altitude at goal node. If final_alt_ft was 0.0 (default), use destination_elevation_ft.
    current_final_alt = float(final_alt_ft if final_alt_ft != 0.0 else destination_elevation_ft)
    active_alt[g_idx, landing_time_bin] = current_final_alt
    active_phase[g_idx, landing_time_bin] = DESCENT # Assuming aircraft is in descent or landed phase at goal
    active_eta[g_idx, landing_time_bin] = float(landing_seconds_since_midnight_val) # Use passed landing time in ssm

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
    # Iterate from goal backwards. \`v_node_graph_id\` is the current node being processed.
    # \`u_node_id_pred\` are its predecessors in the original graph (successors in reversed graph).
    for generation_node_ids in tqdm(topo_generations_node_ids, desc="Topological Generations (Backward)"):
        batch_coords_pred_list = [] # Coordinates of predecessor u
        batch_alts_v_curr_list = [] # Altitude at current node v
        batch_eta_v_curr_list = []  # ETA at current node v
        batch_phase_v_curr_list = []# Phase at current node v
        batch_coords_v_curr_list = []# Coordinates of current node v
        batch_u_node_indices_list = [] # Indices of predecessor u (for updating V[u,k_u])
        batch_v_node_indices_list = [] # Storing v_node_idx corresponding to each transition
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
                        batch_v_node_indices_list.append(v_node_idx) # v_node_idx for this u->v transition
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
        
        # Use the u and v indices directly from the batch lists
        u_indices_for_cost_tensor = torch.tensor(batch_u_node_indices_list, dtype=torch.long, device=device)
        v_indices_for_cost_tensor = torch.tensor(batch_v_node_indices_list, dtype=torch.long, device=device)
        
        # Cost of traversing edge u -> v
        # Ensure cost model inputs (dist_matrix, ac_matrix) are float32 if cost_model expects that
        # Cost model internal params are float32. tailwind_kts_batch also made float32
        cost_uv_batch = cost_model(
            (u_indices_for_cost_tensor, v_indices_for_cost_tensor), # (u,v)
            dist_matrix.to(torch.float32), # cost_model expects float32 for matrices if params are float32
            ac_matrix.to(torch.float32),   # cost_model expects float32
            tailwind_kts_batch.to(torch.float32) 
        ) # cost_uv_batch will be float32
        
        V_v_kv_tensor = torch.tensor(batch_V_v_kv_list, dtype=torch.float64, device=device)
        
        for i in range(len(alt_u_new_batch)): # i is the index of the transition from u to v
            u_node_idx = batch_u_node_indices_list[i] # This is the predecessor's index
            v_node_idx = batch_v_node_indices_list[i] # v_node_idx for this specific transition
            
            alt_u_new = alt_u_new_batch[i]
            eta_u_new = eta_u_new_batch[i] # This is ETA at node u, as a result of get_next_state_bw
            phase_u_new = phase_u_new_batch[i]
            cost_uv_scalar = cost_uv_batch[i] # This is a scalar tensor, float32
            V_v_val = V_v_kv_tensor[i] # V[v, k_v]

            # Check for invalid states from get_next_state_bw or invalid costs
            if torch.isinf(cost_uv_scalar) or torch.isnan(cost_uv_scalar) or \
               torch.isinf(V_v_val) or torch.isnan(eta_u_new) or alt_u_new < 0: # Check for invalid states from get_next_state_bw
                continue

            # Time bin for state at u
            time_at_u_since_min_overall = eta_u_new - min_time_overall_seconds # Use common min_time_overall_seconds
            if time_at_u_since_min_overall < -delta_t_seconds: # Allow some slack for rounding near min_time_overall_seconds
                 continue 
            
            # Calculate the time bin index for the node we need to know, from eta_u_new above
            k_u = int(torch.round(time_at_u_since_min_overall / delta_t_seconds).item())

            if not (0 <= k_u < num_time_bins):
                continue
            
            # Store cost and gradient
            edge_tuple = (u_node_idx, v_node_idx) # Use the actual integer indices
            current_canonical_edge_idx = edge_to_canonical_idx.get(edge_tuple)

            if current_canonical_edge_idx is not None and num_unique_edges > 0 : # Ensure edge is in our map and tensors exist
                edge_costs_time_binned[current_canonical_edge_idx, k_u] = cost_uv_scalar.item() # Store as float64
                
                if learnable_cost_params: # This implies num_cost_params > 0
                    # Ensure cost_uv_scalar requires grad for this specific calculation.
                    # It should if it's an output of nn.Module and learnable_cost_params were used in its computation graph.
                    if cost_uv_scalar.requires_grad:
                        grads_for_edge_cost = torch.autograd.grad(
                            outputs=cost_uv_scalar,
                            inputs=learnable_cost_params,
                            retain_graph=True, # Retain graph for other items in batch or for V updates
                            allow_unused=True, # Some params might not influence this specific cost if PLMs are complex
                            create_graph=False # We need concrete values, not graph for further diff
                        )
                        
                        # Store these gradients, matching the flattened order of learnable_cost_params
                        temp_grads_storage = torch.zeros(num_cost_params, device=device, dtype=torch.float64)
                        param_flat_idx = 0
                        for p_idx, p_grad in enumerate(grads_for_edge_cost):
                            num_p_elements = learnable_cost_params[p_idx].numel()
                            if p_grad is not None:
                                temp_grads_storage[param_flat_idx : param_flat_idx + num_p_elements] = \
                                    p_grad.detach().clone().flatten().to(torch.float64)
                            # else, it remains zero (already initialized in temp_grads_storage)
                            param_flat_idx += num_p_elements
                        edge_cost_gradients_time_binned[current_canonical_edge_idx, k_u, :] = temp_grads_storage
                    # else:
                        # If cost_uv_scalar does not require grad, but learnable_params exist,
                        # it implies these params were not part of its computation graph for some reason
                        # (e.g., detached intermediate, or logic error). Gradients are effectively zero.
                        # The tensor is already init with NaN, could fill with 0 explicitly if preferred:
                        # edge_cost_gradients_time_binned[current_canonical_edge_idx, k_u, :] = torch.zeros(num_cost_params, device=device, dtype=torch.float64)
            
            # Backward DP update: V(u) = cost(u,v) + V(v)
            val_to_add_in_exp = V_v_val + cost_uv_scalar.double() # Cost-to-go from u via v. Ensure cost is float64 for sum.
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


    return V, active_eta, active_alt, active_phase, edge_costs_time_binned, edge_cost_gradients_time_binned, edge_to_canonical_idx
