import torch
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Union, List, Tuple, Dict, Optional

from equinox.route.forward_state import get_next_state_fw, CLIMB, CRUISE, DESCENT
from equinox.cost.cost_rev1 import CostRev1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.route.get_wind import get_wind as get_wind_scalar_component


def forward_dp_adj(
    graph_gml_path: str,
    dist_matrix_path: str,
    ac_matrix_path: str,
    wind_model_data_dir: str,
    departure_datetime_str: str,
    source_node_gml_id: Union[str, int],
    
    climb_profile_tensor: torch.Tensor,
    descent_profile_tensor: torch.Tensor,
    climb_vs_profile_tensor: torch.Tensor,
    descent_vs_profile_tensor: torch.Tensor,
    cruise_altitude_ft: float,
    cruise_speed_kts: float,
    
    cost_beta0: float,
    cost_beta1: float,
    cost_beta2: float,
    cost_knots_ac_dist: Optional[List[float]] = None,
    cost_knots_wind: Optional[List[float]] = None,
    
    time_delta_t_minutes: int = 5,
    max_flight_duration_hours: int = 10,
    default_source_alt_ft: float = 0.0, # Used if source node has no elevation attribute
    device_str: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[Union[str, int], int], Dict[int, Union[str, int]]]:
    """
    Performs forward dynamic programming using a soft Bellman update on a flight route graph.

    Args:
        graph_gml_path: Path to the graph GML file.
        dist_matrix_path: Path to the .npy file for the distance matrix D.
        ac_matrix_path: Path to the .npy file for the airspace charge matrix AC.
        wind_model_data_dir: Directory containing wind data (e.g., ERA5).
        departure_datetime_str: Departure datetime ("YYYY-MM-DD HH:MM:SS").
        source_node_gml_id: GML ID of the source node in the graph.
        
        climb_profile_tensor, descent_profile_tensor, 
        climb_vs_profile_tensor, descent_vs_profile_tensor: Aircraft performance profile tensors.
        cruise_altitude_ft: Cruise altitude in feet.
        cruise_speed_kts: Cruise speed in knots.
        
        cost_beta0, cost_beta1, cost_beta2: Coefficients for the cost model.
        cost_knots_ac_dist, cost_knots_wind: Knot points for the PLM in the cost model.
        
        time_delta_t_minutes: Duration of each time bin in minutes.
        max_flight_duration_hours: Maximum flight duration to consider for time bins.
        default_source_alt_ft: Default altitude for the source node if not found in graph attributes.
        device_str: Device to use ("cpu" or "cuda").

    Returns:
        Tuple containing:
            - V (torch.Tensor): Value function V[node_idx, time_bin_idx].
            - current_alts_at_V (torch.Tensor): Altitudes associated with V.
            - current_etas_sec_at_V (torch.Tensor): ETAs (seconds since midnight) associated with V.
            - current_phases_at_V (torch.Tensor): Phases associated with V.
            - node_to_idx (dict): Mapping from GML node ID to integer index.
            - idx_to_node (dict): Mapping from integer index to GML node ID.
    """
    device = torch.device(device_str)

    # 1. Load Graph and Data
    G = nx.read_gml(graph_gml_path)
    
    node_list = list(G.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    idx_to_node = {i: node_id for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if source_node_gml_id not in node_to_idx:
        raise ValueError(f"Source node GML ID {source_node_gml_id} not found in the graph.")
    source_idx = node_to_idx[source_node_gml_id]
    
    dist_matrix_d = torch.from_numpy(np.load(dist_matrix_path)).float().to(device)
    airspace_charge_matrix_ac = torch.from_numpy(np.load(ac_matrix_path)).float().to(device)

    if dist_matrix_d.shape != (num_nodes, num_nodes) or \
       airspace_charge_matrix_ac.shape != (num_nodes, num_nodes):
        raise ValueError(f"Mismatch in D/AC matrix dimensions ({dist_matrix_d.shape}, {airspace_charge_matrix_ac.shape}) "
                         f"and graph node count ({num_nodes}). Ensure matrices are indexed 0 to N-1 "
                         "corresponding to the order in `list(G.nodes())`.")

    # 2. Initialize Models
    departure_date_obj = datetime.strptime(departure_datetime_str.split(" ")[0], "%Y-%m-%d")
    wind_model = WindDate(
        date_str=departure_date_obj.strftime("%Y-%m-%d"),
        data_dir=wind_model_data_dir,
        device=device 
    )

    performance_model = Performance(
        climb_profile_tensor, descent_profile_tensor, 
        climb_vs_profile_tensor, descent_vs_profile_tensor,
        cruise_altitude_ft, cruise_speed_kts
    )
    climb_perf_table_torch = get_eta_and_distance_climb(performance_model, alt_step_ft=1000)
    climb_perf_table_tuples = [
        (float(r[0]), float(r[1]), float(r[2])) for r in climb_perf_table_torch.cpu().numpy()
    ]

    cost_model = CostRev1(
        beta0=cost_beta0, beta1=cost_beta1, beta2=cost_beta2,
        knots_ac_dist=cost_knots_ac_dist, knots_wind=cost_knots_wind,
        device=device
    )

    # 3. DP Initialization
    delta_t_seconds = time_delta_t_minutes * 60
    num_time_bins = (max_flight_duration_hours * 3600) // delta_t_seconds + 1

    V = torch.full((num_nodes, num_time_bins), float('inf'), dtype=torch.float32, device=device)
    current_alts_at_V = torch.full_like(V, float('nan')) 
    current_etas_sec_at_V = torch.full_like(V, float('nan')) 
    current_phases_at_V = torch.full_like(V, -1, dtype=torch.long, device=device)

    departure_seconds_since_midnight = float(datestr_to_seconds_since_midnight(departure_datetime_str))
    initial_time_bin_k_s = int(departure_seconds_since_midnight // delta_t_seconds)

    source_node_attrs = G.nodes[source_node_gml_id]
    initial_alt_ft_src = float(source_node_attrs.get('alt', source_node_attrs.get('elevation_ft', default_source_alt_ft)))
    
    if not (0 <= initial_time_bin_k_s < num_time_bins):
        raise ValueError(f"Initial departure time bin {initial_time_bin_k_s} is out of bounds [0, {num_time_bins-1}]. "
                         f"Max flight duration or departure time might be problematic.")

    V[source_idx, initial_time_bin_k_s] = 0.0
    current_alts_at_V[source_idx, initial_time_bin_k_s] = initial_alt_ft_src
    current_etas_sec_at_V[source_idx, initial_time_bin_k_s] = departure_seconds_since_midnight
    current_phases_at_V[source_idx, initial_time_bin_k_s] = CLIMB

    # 4. Topological Sort for DAG processing
    try:
        topo_order_gml_ids = list(nx.topological_sort(G))
        topo_order_indices = [node_to_idx[node_id] for node_id in topo_order_gml_ids]
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph is not a DAG. Topological sort failed.")

    # 5. Main DP Loop
    for i_idx in topo_order_indices:
        reachable_time_bins_for_i = torch.where(torch.isfinite(V[i_idx, :]))[0]

        if reachable_time_bins_for_i.numel() == 0:
            continue

        i_node_gml_id = idx_to_node[i_idx]
        successors_of_i_gml_ids = list(G.successors(i_node_gml_id))
        if not successors_of_i_gml_ids:
            continue
            
        num_successors = len(successors_of_i_gml_ids)
        # Ensure j_indices are long for indexing if needed, but mostly used to get gml_ids
        j_indices_list = [node_to_idx[succ_id] for succ_id in successors_of_i_gml_ids]

        for k_idx_loop_var in reachable_time_bins_for_i:
            k_idx = k_idx_loop_var.item()

            V_i_k = V[i_idx, k_idx] # Scalar cost to reach (i,k)
            #unsqueeze(0) to make them [1] for repeat, then squeeze if get_next_state_fw needs scalar for non-batch
            alt_i_tensor = current_alts_at_V[i_idx, k_idx].reshape(1) 
            eta_i_sec_tensor = current_etas_sec_at_V[i_idx, k_idx].reshape(1)
            phase_i_tensor = current_phases_at_V[i_idx, k_idx].reshape(1)

            i_node_attrs = G.nodes[i_node_gml_id]
            coords_i_tensor = torch.tensor([[float(i_node_attrs['lat']), float(i_node_attrs['lon'])]], 
                                           dtype=torch.float64, device=device)

            coords_src_batch = coords_i_tensor.repeat(num_successors, 1)
            alts_src_batch = alt_i_tensor.repeat(num_successors)
            eta_src_batch = eta_i_sec_tensor.repeat(num_successors)
            phase_src_batch = phase_i_tensor.repeat(num_successors)
            
            lats_j_list = [float(G.nodes[succ_gml_id]['lat']) for succ_gml_id in successors_of_i_gml_ids]
            lons_j_list = [float(G.nodes[succ_gml_id]['lon']) for succ_gml_id in successors_of_i_gml_ids]
            coords_tgt_batch = torch.tensor(list(zip(lats_j_list, lons_j_list)), dtype=torch.float64, device=device)

            alt_j_new_batch, eta_j_new_sec_batch, phase_j_new_batch = get_next_state_fw(
                coords_src_batch, alts_src_batch, eta_src_batch, phase_src_batch,
                coords_tgt_batch, climb_perf_table_tuples, wind_model
            )
            
            # Convert j_indices_list to tensor for indexing D and AC matrices
            j_indices_tensor = torch.tensor(j_indices_list, device=device, dtype=torch.long)
            dist_ij_batch = dist_matrix_d[i_idx, j_indices_tensor]
            # ac_ij_batch = airspace_charge_matrix_ac[i_idx, j_indices_tensor] # Not directly used if CostRev1 fetches

            # Wind component for cost function (along track, kts)
            # Using alts_src_batch, eta_src_batch for wind calculation at segment start
            tailwind_mps_ij_batch = get_wind_scalar_component(
                coords_src_batch, coords_tgt_batch, alts_src_batch, eta_src_batch, wind_model
            )
            KNOTS_PER_MPS = 1.94384
            tailwind_kts_ij_batch = tailwind_mps_ij_batch * KNOTS_PER_MPS

            for succ_iter_idx in range(num_successors):
                j_target_idx = j_indices_list[succ_iter_idx]
                
                current_dist_ij = dist_ij_batch[succ_iter_idx]
                if torch.isinf(current_dist_ij): # No edge in D matrix or true infinite distance
                    cost_val_ij = torch.tensor(float('inf'), device=device)
                else:
                    cost_val_ij = cost_model(
                        edge_indices=(i_idx, j_target_idx), # tuple of int indices
                        distance_matrix_d=dist_matrix_d,
                        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
                        tailwind_value_w=tailwind_kts_ij_batch[succ_iter_idx] # scalar tensor
                    )

                cost_path_to_j_via_i = V_i_k + cost_val_ij
                if torch.isinf(cost_path_to_j_via_i) or torch.isnan(cost_path_to_j_via_i):
                    continue

                alt_j_new = alt_j_new_batch[succ_iter_idx]
                eta_j_new_sec = eta_j_new_sec_batch[succ_iter_idx]
                phase_j_new = phase_j_new_batch[succ_iter_idx]

                if torch.isnan(eta_j_new_sec) or torch.isinf(eta_j_new_sec) or \
                   torch.isnan(alt_j_new) or torch.isinf(alt_j_new): # check for invalid propagated state
                    continue
                
                k_new_idx_for_j = int(eta_j_new_sec.item() // delta_t_seconds)

                if not (0 <= k_new_idx_for_j < num_time_bins):
                    continue

                V_old_jk_new = V[j_target_idx, k_new_idx_for_j]
                
                # Softmin update for V: V_new = -log(exp(-V_old) + exp(-V_path_contrib))
                # If V_old is inf, exp(-V_old) is 0. V_new = V_path_contrib.
                # If V_path_contrib is inf, exp(-V_path_contrib) is 0. V_new = V_old.
                neg_costs_for_lse = torch.stack([-V_old_jk_new, -cost_path_to_j_via_i])
                updated_V_jk_new = -torch.logsumexp(neg_costs_for_lse, dim=0)
                
                # Heuristic for updating auxiliary state variables (alt, eta, phase):
                # If this path's cost contribution is better (in a hard-min sense) than the
                # V value before this soft update, then this path "dominates" the state.
                # This is a simplification for associating a single state with the soft V.
                if cost_path_to_j_via_i < V_old_jk_new : # Checks if new path is "better"
                    current_alts_at_V[j_target_idx, k_new_idx_for_j] = alt_j_new
                    current_etas_sec_at_V[j_target_idx, k_new_idx_for_j] = eta_j_new_sec
                    current_phases_at_V[j_target_idx, k_new_idx_for_j] = phase_j_new
                
                V[j_target_idx, k_new_idx_for_j] = updated_V_jk_new
                
    return V, current_alts_at_V, current_etas_sec_at_V, current_phases_at_V, node_to_idx, idx_to_node
