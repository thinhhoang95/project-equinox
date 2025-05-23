import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple

# Equinox imports
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.helpers.haversine import haversinet
from equinox.cost.cost_rev1 import CostRev1
from equinox.wind.wind_model import WindModel
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.route.forward_state import get_next_state_fw, CLIMB, CRUISE, DESCENT # Assuming CLIMB=0, CRUISE=1, DESCENT=2
from equinox.route.get_wind import get_wind

MPS_TO_KNOTS = 1.9438444924406 # From equinox.dp.forward_dp_vec2

def _get_node_to_idx_and_vice_versa(graph: nx.DiGraph) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Helper to create node to index mappings."""
    node_list = list(graph.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}
    return node_to_idx, idx_to_node

def _sample_single_route_with_details(
    graph: nx.DiGraph,
    node_to_idx: Dict[str, int],
    idx_to_node: Dict[int, str],
    source_node_id: str,
    goal_node_id: str,
    V_bwd: torch.Tensor, # Backward value function tensor
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_tensor: torch.Tensor, # Already a tensor
    ac_matrix_tensor: torch.Tensor,   # Already a tensor
    source_elevation_ft: float,
    initial_alt_ft_amsl: float, # Altitude AMSL at source node at takeoff
    delta_t_seconds: int, # time bin length/size
    min_time_overall_seconds: float, # Reference start time for time bins (landing time - max flight duration)
    takeoff_ssm: float,
    device: torch.device,
    max_segments: int = 100,
    temperature: float = 1.0,
    verbose: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Samples a single route using the backward value function and cost model.
    Returns the route as a list of node IDs and a list of leg distances in NM.
    """

    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=source_elevation_ft)

    current_node_id = source_node_id
    current_node_idx = node_to_idx[source_node_id]
    
    current_alt_amsl = torch.tensor(initial_alt_ft_amsl, dtype=torch.float64, device=device)
    current_eta_ssm = torch.tensor(takeoff_ssm, dtype=torch.float64, device=device) 
    current_phase = torch.tensor(CLIMB, dtype=torch.long, device=device)

    sampled_route_ids = [current_node_id]
    leg_distances_nm = [] # Initialize list to store leg distances

    for seg_count in range(max_segments):
        if current_node_id == goal_node_id:
            break

        time_offset_from_start_u = current_eta_ssm.item() - min_time_overall_seconds
        if time_offset_from_start_u < 0: time_offset_from_start_u = 0.0
        
        k_u = int(round(time_offset_from_start_u / delta_t_seconds))
        k_u = max(0, min(k_u, V_bwd.shape[1] - 1))
        
        V_bwd_u_ku = V_bwd[current_node_idx, k_u]
        if verbose:
            print(f'Node {current_node_id} (idx {current_node_idx}), time bin is {k_u}: backward val V_bwd_u_ku: {V_bwd_u_ku}')

        if torch.isinf(V_bwd_u_ku) and V_bwd_u_ku > 0:
             print(f"Warning: V_bwd at {current_node_id} (idx {current_node_idx}), t_bin {k_u} is +inf. Terminating.")
             break

        successors_graph_ids = list(graph.successors(current_node_id))
        if not successors_graph_ids:
            print(f"Warning: Node {current_node_id} has no successors. Terminating route.")
            break

        batch_coords_src_list, batch_alts_src_list, batch_eta_src_list, batch_phase_src_list = [], [], [], []
        batch_coords_tgt_list = []
        
        successor_node_indices_list = [] 
        u_indices_for_cost_model = [] 
        v_indices_for_cost_model = [] 

        current_u_coords_data = graph.nodes[current_node_id]
        current_u_coords = [
            current_u_coords_data['lat'].item() if isinstance(current_u_coords_data['lat'], torch.Tensor) else current_u_coords_data['lat'],
            current_u_coords_data['lon'].item() if isinstance(current_u_coords_data['lon'], torch.Tensor) else current_u_coords_data['lon']
        ]
        current_u_coords_tensor = torch.tensor([current_u_coords], dtype=torch.float64, device=device)

        for v_node_g_id in successors_graph_ids:
            v_node_g_idx = node_to_idx[v_node_g_id]
            v_coords_data = graph.nodes[v_node_g_id]
            v_coords = [
                v_coords_data['lat'].item() if isinstance(v_coords_data['lat'], torch.Tensor) else v_coords_data['lat'],
                v_coords_data['lon'].item() if isinstance(v_coords_data['lon'], torch.Tensor) else v_coords_data['lon']
            ]

            batch_coords_src_list.append(current_u_coords)
            batch_alts_src_list.append(current_alt_amsl.item())
            batch_eta_src_list.append(current_eta_ssm.item())
            batch_phase_src_list.append(current_phase.item())
            batch_coords_tgt_list.append(v_coords)
            
            successor_node_indices_list.append(v_node_g_idx)
            u_indices_for_cost_model.append(current_node_idx)
            v_indices_for_cost_model.append(v_node_g_idx)

        if not batch_coords_src_list: break 

        coords_src_b = torch.tensor(batch_coords_src_list, dtype=torch.float64, device=device)
        alts_src_b = torch.tensor(batch_alts_src_list, dtype=torch.float64, device=device)
        eta_src_b = torch.tensor(batch_eta_src_list, dtype=torch.float64, device=device)
        phase_src_b = torch.tensor(batch_phase_src_list, dtype=torch.long, device=device)
        coords_tgt_b = torch.tensor(batch_coords_tgt_list, dtype=torch.float64, device=device)

        alt_v_new_b, eta_v_new_b, phase_v_new_b = get_next_state_fw(
            coords_src_b, alts_src_b, eta_src_b, phase_src_b,
            coords_tgt_b, climb_perf_table, wind_model
        )

        tailwind_mps_b = get_wind(coords_src_b, coords_tgt_b, alts_src_b, eta_src_b, wind_model)
        tailwind_kts_b = tailwind_mps_b * MPS_TO_KNOTS
        
        u_idc_cost_b = torch.tensor(u_indices_for_cost_model, dtype=torch.long, device=device)
        v_idc_cost_b = torch.tensor(v_indices_for_cost_model, dtype=torch.long, device=device)
        
        cost_uv_b = cost_model(
            (u_idc_cost_b, v_idc_cost_b),
            dist_matrix_tensor, ac_matrix_tensor, tailwind_kts_b.to(dtype=torch.float32)
        )

        log_probs_list = []
        valid_successor_options = [] 

        for i in range(len(successors_graph_ids)):
            v_g_id = successors_graph_ids[i]
            v_g_idx = successor_node_indices_list[i]
            
            cost_uv_val = cost_uv_b[i]
            eta_v_new_val = eta_v_new_b[i]

            time_offset_from_start_v = eta_v_new_val.item() - min_time_overall_seconds
            
            if time_offset_from_start_v < -1e-3: 
                print(f"Warning: ETA for {v_g_id} ({eta_v_new_val.item()}) is before takeoff. Skipping.")
                continue
            if time_offset_from_start_v < 0: time_offset_from_start_v = 0.0

            k_v = int(round(time_offset_from_start_v / delta_t_seconds))
            k_v_clipped = max(0, min(k_v, V_bwd.shape[1] - 1))

            if k_v_clipped != k_v:
                k_v = k_v_clipped
                print(f"Warning: k_v clipped from {k_v} to {k_v_clipped} for {v_g_id}.")
                raise ValueError(f"k_v clipped from {k_v} to {k_v_clipped} for {v_g_id}.")

            V_bwd_v_kv = V_bwd[v_g_idx, k_v_clipped]

            if torch.isinf(cost_uv_val) or (torch.isinf(V_bwd_v_kv) and V_bwd_v_kv > 0):
                log_prob_val = torch.tensor(float('-inf'), device=device, dtype=torch.float64)
            else:
                log_prob_val = -cost_uv_val.double() - V_bwd_v_kv.double() + V_bwd_u_ku.double()
            
            # For debugging
            # if v_g_id in ['LETP', 'LFDA', 'KOVAK']:
            #     print(f'{v_g_id}, alt: {alt_v_new_b[i]}, eta: {eta_v_new_b[i]}, phase: {phase_v_new_b[i]}, log_prob_val: {log_prob_val}')
            #     print(f'k_v: {k_v}/{V_bwd.shape[1]}, V_bwd_u_ku: {V_bwd_u_ku}, V_bwd_v_kv: {V_bwd_v_kv}')
            #     print(f'cost_uv_val: {cost_uv_val}')
            #     print('---')
            
            log_probs_list.append(log_prob_val)
            valid_successor_options.append({
                "id": v_g_id, "idx": v_g_idx,
                "alt": alt_v_new_b[i], "eta": eta_v_new_b[i], "phase": phase_v_new_b[i],
                "k_v": k_v, "k_v_clipped": k_v_clipped,
                "V_bwd_u_ku": V_bwd_u_ku, "V_bwd_v_kv": V_bwd_v_kv
            })
        
        if not valid_successor_options:
            print(f"Warning: No valid successors from {current_node_id} after state/cost calculation. Terminating.")
            break
        
        log_probs_tensor = torch.stack(log_probs_list)
        
        # Apply temperature scaling
        if temperature <= 0: # Avoid division by zero or negative, treat as T->0+ (greedy) or T=1 (original)
            print(f"Warning: Temperature is {temperature}. Using T=1e-6 for near-greedy or T=1 if issues persist.")
            # Heuristic: if user means greedy, use a very small positive number.
            # If it's an error, default to 1. For now, let's use a small number.
            # A more robust solution might raise an error or have a clearer policy.
            effective_temperature = 1e-6
        else:
            effective_temperature = temperature
        
        scaled_log_probs_tensor = log_probs_tensor / effective_temperature

        if torch.all(torch.isneginf(scaled_log_probs_tensor)) and not torch.all(torch.isneginf(log_probs_tensor)):
            print(f"Warning: All log_probs became -inf after T-scaling from {current_node_id}. Original: {log_probs_tensor}, Scaled: {scaled_log_probs_tensor}. Using original.")
            distribution_logits = log_probs_tensor # Fallback to original if scaling causes all -inf
        elif torch.all(torch.isneginf(scaled_log_probs_tensor)):
             print(f"Warning: All successor log_probs are -inf from {current_node_id} (even before T-scaling or T-scaling made them all -inf). Terminating.")
             break
        else:
            distribution_logits = scaled_log_probs_tensor

        try:
            distribution = torch.distributions.Categorical(logits=distribution_logits)
            chosen_successor_list_idx = distribution.sample().item()
        except RuntimeError as e:
            print(f"Categorical distribution error: {e}. Logits: {distribution_logits}. Terminating.")
            break
            
        chosen_data = valid_successor_options[chosen_successor_list_idx]

        next_node_id = chosen_data["id"]
        # Calculate leg distance before updating current_node_id
        next_node_coords_data = graph.nodes[next_node_id]
        next_node_coords = [
            next_node_coords_data['lat'].item() if isinstance(next_node_coords_data['lat'], torch.Tensor) else next_node_coords_data['lat'],
            next_node_coords_data['lon'].item() if isinstance(next_node_coords_data['lon'], torch.Tensor) else next_node_coords_data['lon']
        ]
        next_node_coords_tensor = torch.tensor([next_node_coords], dtype=torch.float64, device=device)

        leg_dist_nm = haversinet(
            current_u_coords_tensor[:, 0], current_u_coords_tensor[:, 1],
            next_node_coords_tensor[:, 0], next_node_coords_tensor[:, 1]
        ).item()
        leg_distances_nm.append(leg_dist_nm)

        current_node_id = chosen_data["id"]
        current_node_idx = chosen_data["idx"]
        current_alt_amsl = chosen_data["alt"] 
        current_eta_ssm = chosen_data["eta"]
        current_phase = chosen_data["phase"]
        
        sampled_route_ids.append(current_node_id)

    else: 
        if current_node_id != goal_node_id:
            print(f"Warning: Sampling stopped after {max_segments} segments, goal {goal_node_id} not reached. Current: {current_node_id}")

    return sampled_route_ids, leg_distances_nm


def sample_routes(
    graph: nx.DiGraph,
    landing_time_str: str,
    max_flight_duration_hours: int,
    takeoff_time_str: str,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    wind_model: WindModel,
    V_bwd: torch.Tensor,
    source_node_id: str,
    goal_node_id: str, 
    cost_model: CostRev1,
    performance_model: Performance,
    source_elevation_ft: float,
    initial_alt_ft_amsl: float = None, 
    delta_t_seconds: int = 300, 
    device_str: str = "cpu",
    max_segments_in_route: int = 100,
    num_samples: int = 1,
    temperature: float = 1.0,
    max_attempts: int = 200,
    verbose: bool = False
) -> List[Tuple[List[str], List[float]]]:
    """
    Samples routes from a source to a goal node using a backward value function.
    Args:
        graph (nx.DiGraph): The route graph. Nodes must have 'lat', 'lon' attributes.
        landing_time_str (str): ISO format of the landing time string (e.g., "2023-04-01 12:00:00").
        dist_matrix_np (np.ndarray): 2D array of distances between node indices (NM).
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        wind_model (WindModel): Instantiated wind model.
        V_bwd (torch.Tensor): Backward value function V[node_idx, time_bin_idx].
        source_node_id (str): The ID of the source node in the graph.
        goal_node_id (str): The ID of the goal node in the graph.
        cost_model (CostRev1): Instantiated cost model.
        performance_model (Performance): Instantiated aircraft performance model.
        source_elevation_ft (float): Elevation of the source airport in feet.
        initial_alt_ft_amsl (float, optional): Initial altitude (AMSL) at source node at takeoff. 
                                               Defaults to source_elevation_ft.
        delta_t_seconds (int, optional): Duration of each time bin in seconds. Must match V_bwd.
        device_str (str, optional): "cpu" or "cuda".
        max_segments_in_route (int, optional): Max number of segments before terminating a sample.
        num_samples (int, optional): Number of routes to sample.
        temperature (float, optional): Sampling temperature. Lower values (e.g., 0.1) make sampling
                                       more greedy (closer to min cost path). Higher values
                                       (e.g., 2.0) make it more random. Defaults to 1.0.
    Returns:
        List[Tuple[List[str], List[float]]]: A list of tuples. Each tuple contains:
            - A list of node IDs representing the sampled route.
            - A list of floats representing the leg distances in nautical miles for that route.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    if initial_alt_ft_amsl is None:
        initial_alt_ft_amsl = source_elevation_ft
        print(f"Note: initial_alt_ft_amsl not provided, defaulting to source_elevation_ft: {source_elevation_ft} ft.")

    node_to_idx, idx_to_node = _get_node_to_idx_and_vice_versa(graph)

    if source_node_id not in node_to_idx:
        raise ValueError(f"Source node {source_node_id} not in graph.")
    if goal_node_id not in node_to_idx:
        raise ValueError(f"Goal node {goal_node_id} not in graph.")
    if V_bwd.shape[0] != len(node_to_idx):
        raise ValueError(f"V_bwd num_nodes ({V_bwd.shape[0]}) mismatch with graph nodes ({len(node_to_idx)}).")

    dist_matrix_tensor = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix_tensor = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)
    
    landing_ssm = datestr_to_seconds_since_midnight(landing_time_str)
    min_time_ref_for_bins = landing_ssm - (max_flight_duration_hours * 3600)
    takeoff_ssm = datestr_to_seconds_since_midnight(takeoff_time_str)

    sampled_routes_list = []
    sampled_distances_list = []
    completed_routes_count = 0 # Renamed for clarity

    for attempt_num in range(max_attempts): # Added attempt_num for potential debugging
        if completed_routes_count >= num_samples:
            break
            

        try: 
            route_ids, leg_distances = _sample_single_route_with_details(
                graph=graph,
                node_to_idx=node_to_idx,
                idx_to_node=idx_to_node,
                source_node_id=source_node_id,
                goal_node_id=goal_node_id,
                V_bwd=V_bwd.clone().to(dtype=torch.float64, device=device),
                cost_model=cost_model,
                wind_model=wind_model,
                performance_model=performance_model,
                dist_matrix_tensor=dist_matrix_tensor,
                ac_matrix_tensor=ac_matrix_tensor,
                source_elevation_ft=source_elevation_ft,
                initial_alt_ft_amsl=initial_alt_ft_amsl,
                delta_t_seconds=delta_t_seconds,
                min_time_overall_seconds=min_time_ref_for_bins,
                takeoff_ssm=takeoff_ssm,
                device=device,
                max_segments=max_segments_in_route,
                temperature=temperature,
                verbose=verbose
            )
        except ValueError as e:
            continue # Skip this attempt

        if route_ids and route_ids[-1] == goal_node_id: # Check if route_ids is not empty
            sampled_routes_list.append(route_ids)
            sampled_distances_list.append(leg_distances)
            completed_routes_count += 1

        elif verbose:
            print(f"Attempt {attempt_num + 1}: Route did not reach goal or was empty. Last node: {route_ids[-1] if route_ids else 'N/A'}")

    if completed_routes_count < num_samples:
        print(f"Warning: Only {completed_routes_count} routes reached the goal out of {num_samples} desired, after {max_attempts} attempts.")

    return sampled_routes_list, sampled_distances_list



if __name__ == '__main__':
    print("Illustrative example for sample_routes (requires actual data and models to run fully):")
    pass