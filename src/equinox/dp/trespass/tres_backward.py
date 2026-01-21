from math import floor
import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from equinox.route.backward_state_f1 import get_next_state_bw
from equinox.route.forward_state import CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.wind.wind_model import WindModel

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts
EPS_BIN_EPS = 1e-9  # small guard against float error

# Phase constants as indices
PHASE_CLIMB = CLIMB # Typically 0
PHASE_CRUISE = CRUISE # Typically 1
PHASE_DESCENT = DESCENT # Typically 2
NUM_PHASES = 3


def _round_to_bin_idx(value: float, max_bin_idx: int) -> int:
    """Safely rounds a float value to a valid bin index."""
    if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
        return -1 # Invalid index
    idx = int(floor(value + EPS_BIN_EPS))
    if not (0 <= idx <= max_bin_idx): # Max bin index is inclusive
        return -1
    return idx


def tres_backward(
    graph: nx.DiGraph,
    goal_node_id: str,
    estimated_landing_time_str: str,
    origin_elevation_ft: float, # Used for climb performance
    destination_elevation_ft: float,
    wind_model: WindModel, # Still needed for get_next_state_bw
    performance_model: Performance,
    transitions_list: list, # List of (u_idx, eps_u_bins, alt_u_ft, v_idx, eps_v_bins, alt_v_ft)
    eta_takeoff_str: str,
    final_alt_ft: float = 0.0,
    delta_t_seconds_wall_clock: int = 300,
    delta_t_seconds_climb: int = 30,
    max_flight_duration_hours: int = 10,
    climb_phase_switch_allowance_climb_time_bins: int = 40,
    device: torch.device = None
):
    """
    Implements a backward pass to identify feasible flight transitions in a 4D state space:
    (waypoint, wall-clock time bin, remaining climb time bin, phase).

    This function works backward from the destination, identifying all kinematically
    feasible transitions between states. It does not compute an optimal value function
    but rather records the connections that are possible according to flight performance
    and rules. The algorithm handles three flight phases (CLIMB, CRUISE, DESCENT)
    and processes nodes in topological generations.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph representing the flight route network. Each node should have 'lat' and 'lon'
        attributes representing waypoint coordinates.
    goal_node_id : str
        Identifier of the destination node.
    estimated_landing_time_str : str
        Expected landing time in format "YYYY-MM-DD HH:MM:SS".
    origin_elevation_ft : float
        Elevation of origin airport in feet, used for climb performance calculations.
    destination_elevation_ft : float
        Elevation of destination airport in feet, used for descent performance calculations.
    wind_model : WindModel
        Wind model providing wind data (used by get_next_state_bw).
    performance_model : Performance
        Aircraft performance model.
    transitions_list : list
        List of climb transition tuples: (u_idx, eps_u_bins, alt_u_ft, v_idx, eps_v_bins, alt_v_ft).
    eta_takeoff_str : str
        Estimated takeoff time in format "YYYY-MM-DD HH:MM:SS".
    max_eps_bin : int, default=36
        Maximum climb time in number of bins from climb performance table.
    final_alt_ft : float, default=0.0
        Final altitude at destination in feet. If 0.0, uses destination_elevation_ft.
    delta_t_seconds_wall_clock : int, default=300
        Time discretization for wall-clock time bins in seconds.
    delta_t_seconds_climb : int, default=30
        Time discretization for climb time bins in seconds.
    max_flight_duration_hours : int, default=10
        Maximum allowed flight duration in hours.
    climb_phase_switch_allowance_climb_time_bins : int, default=40
        Additional time bins allowed for switching from cruise to climb phase (in climb time bins).
    device : torch.device, optional
        PyTorch device for computations. If None, automatically selects CUDA if available.

    Returns
    -------
    list[tuple]
        A list of feasible transition tuples. Each tuple is in the format:
        (node_id_from, eta_bin_from, rho_bin_from, alt_from_rounded_ft, phase_from,
         node_id_to, eta_bin_to, rho_bin_to, alt_to_rounded_ft, phase_to,
         eta_from_abs_s, eta_to_abs_s)
        where:
        - node_id_from/to: str, waypoint identifier
        - eta_bin_from/to: int, wall-clock time bin index
        - rho_bin_from/to: int, remaining climb time bin index
        - alt_from/to_rounded_ft: int, altitude in feet, rounded
        - phase_from/to: int, flight phase (0:CLIMB, 1:CRUISE, 2:DESCENT)
        - eta_from/to_abs_s: float, seconds since midnight (wall-clock absolute)

    Raises
    ------
    ValueError
        If goal_node_id is not found, landing time is outside window, or graph is not a DAG.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialization ---
    feasible_transitions_list = [] # Stores the output transitions

    node_list = list(graph.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    idx_to_node = {i: node_id for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if goal_node_id not in node_to_idx:
        raise ValueError(f"Goal node {goal_node_id} not found in graph.")
    g_idx = node_to_idx[goal_node_id]

    estimated_landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str)
    eta_takeoff_ssm = datestr_to_seconds_since_midnight(eta_takeoff_str)

    max_time_overall_seconds = estimated_landing_ssm
    min_time_overall_seconds = max_time_overall_seconds - max_flight_duration_hours * 3600
    num_time_bins = int((max_time_overall_seconds - min_time_overall_seconds) / delta_t_seconds_wall_clock) + 1

    descent_perf_table = get_eta_and_distance_descent(performance_model, destination_airport_elevation_ft=destination_elevation_ft)
    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=origin_elevation_ft)

    # Calculating max_eps_bin from climb performance table, which is the elapsed time of the final row
    climb_time_max = climb_perf_table[-1][1]
    max_eps_bin = round(climb_time_max / delta_t_seconds_climb) # should be the same as in tres_forward's num_etto_bins

    num_rho_bins = max_eps_bin + 1 # Remaining climb time bins from 0 to max_eps_bin

    # V tensor is removed. active_alt, active_eta, active_phase_return are used for reachability.
    # V = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('nan'), dtype=torch.float64, device=device)
    active_eta = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('nan'), dtype=torch.float64, device=device)
    active_phase_return = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), -1, dtype=torch.long, device=device)
    
    # dist_matrix and ac_matrix are removed as cost_model is removed
    # dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    # ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    transitions_map_by_indices = {(t[0], t[3]): (t[1], t[2], t[4]) for t in transitions_list} # (u_idx, v_idx) -> (eps_u_bins, alt_u_ft, eps_v_bins)
    # Note in the case that there are multiple transitions from the same u_idx to the same v_idx,
    # the last transition in the list will be used.
    
    # Initialize at goal node
    goal_node_time_since_min_overall = estimated_landing_ssm - min_time_overall_seconds
    landing_time_bin_idx = _round_to_bin_idx(goal_node_time_since_min_overall / delta_t_seconds_wall_clock, num_time_bins -1)
    
    if landing_time_bin_idx == -1:
        raise ValueError("Estimated landing time is outside the defined time window.")

    rho_g_idx = 0 # At landing, remaining climb time is 0
    phase_g_idx = PHASE_DESCENT
    
    current_final_alt = float(final_alt_ft if final_alt_ft != 0.0 else destination_elevation_ft)
    
    goal_state_tuple = (g_idx, landing_time_bin_idx, rho_g_idx, phase_g_idx)
    # V[goal_state_tuple] = 0.0 # V removed
    active_alt[goal_state_tuple] = current_final_alt
    active_eta[goal_state_tuple] = float(estimated_landing_ssm)
    active_phase_return[goal_state_tuple] = phase_g_idx

    # --- 2. Topological Generations ---
    try:
        graph_reversed = graph.reverse(copy=True)
        topo_generations_node_ids = list(nx.topological_generations(graph_reversed))
    except nx.NetworkXUnfeasible:
        raise ValueError("Reversed graph is not a DAG, cannot perform topological sort.")

    # --- 3. Main DP Loop ---
    for generation_node_ids in tqdm(topo_generations_node_ids, desc="Topological Generations (Backward)"):
        # Batch lists for get_next_state_bw
        batch_coords_src_list = []  # u_coords
        batch_alts_t_list = []      # alt_v
        batch_eta_t_list = []       # eta_v
        batch_phase_t_list = []     # phi_v
        batch_coords_tgt_list = []  # v_coords
        # Identifiers to map results back
        batch_identifiers_list = [] # ( (v_idx, k_v_idx, rho_v_idx, phi_v_idx), u_idx )

        for v_node_graph_id in generation_node_ids:
            v_idx = node_to_idx[v_node_graph_id]
            v_coords_tuple = (graph.nodes[v_node_graph_id].get('lat'), graph.nodes[v_node_graph_id].get('lon'))
            v_coords_for_pred = [v_coords_tuple[0].item() if isinstance(v_coords_tuple[0], torch.Tensor) else v_coords_tuple[0],
                                 v_coords_tuple[1].item() if isinstance(v_coords_tuple[1], torch.Tensor) else v_coords_tuple[1]]


            predecessors_orig_graph = list(graph_reversed.successors(v_node_graph_id))
            if not predecessors_orig_graph:
                continue

            for k_v_idx in range(num_time_bins):
                for rho_v_idx in range(num_rho_bins):
                    for phi_v_idx in range(NUM_PHASES):
                        current_S_v_tuple = (v_idx, k_v_idx, rho_v_idx, phi_v_idx)
                        # if not torch.isinf(V[current_S_v_tuple]): # V removed
                        if not np.isnan(active_eta[current_S_v_tuple].item()): # Check reachability via active_eta
                            alt_v_amsl = active_alt[current_S_v_tuple].item()
                            eta_v_ssm = active_eta[current_S_v_tuple].item()

                            if np.isnan(alt_v_amsl) or np.isnan(eta_v_ssm): # Should not happen if active_eta was not NaN
                                continue
                            
                            for u_node_id_pred in predecessors_orig_graph:
                                u_idx_pred = node_to_idx[u_node_id_pred]
                                u_coords_tuple = (graph.nodes[u_node_id_pred].get('lat'), graph.nodes[u_node_id_pred].get('lon'))
                                u_coords_for_pred = [u_coords_tuple[0].item() if isinstance(u_coords_tuple[0], torch.Tensor) else u_coords_tuple[0],
                                                     u_coords_tuple[1].item() if isinstance(u_coords_tuple[1], torch.Tensor) else u_coords_tuple[1]]

                                batch_coords_src_list.append(u_coords_for_pred) # p_s (predecessor u)
                                batch_alts_t_list.append(alt_v_amsl)            # alt_v
                                batch_eta_t_list.append(eta_v_ssm)              # eta_v
                                batch_phase_t_list.append(phi_v_idx)            # phase_v
                                batch_coords_tgt_list.append(v_coords_for_pred) # p_t (current node v)
                                batch_identifiers_list.append((current_S_v_tuple, u_idx_pred))
        
        if not batch_coords_src_list:
            continue

        # --- Batch call to get_next_state_bw for standard propagation ---
        coords_src_tensor = torch.tensor(batch_coords_src_list, dtype=torch.float64, device=device)
        alts_t_tensor = torch.tensor(batch_alts_t_list, dtype=torch.float64, device=device)
        eta_t_tensor = torch.tensor(batch_eta_t_list, dtype=torch.float64, device=device)
        phase_t_tensor = torch.tensor(batch_phase_t_list, dtype=torch.long, device=device)
        coords_tgt_tensor = torch.tensor(batch_coords_tgt_list, dtype=torch.float64, device=device)

        alt_u_std_batch, eta_u_std_ssm_batch, phase_u_std_batch = get_next_state_bw(
            coords_src=coords_src_tensor,
            alts_t=alts_t_tensor,
            eta_t=eta_t_tensor,
            phase_t=phase_t_tensor,
            coords_tgt=coords_tgt_tensor,
            descent_performance=descent_perf_table,
            wind_model=wind_model # Note: get_next_state_bw internally calls wind model for its own calculation
        )

        # --- Process results from the get_next_state_bw call and apply DP updates ---
        # Recall: get_next_state_bw always returns CRUISE or DESCENT phase at u
        # The cruise state is always "cruised" to the next node, but for the switch to CLIMB, we have to handle them separately.
        for i in range(len(batch_identifiers_list)):
            S_v_tuple, u_idx = batch_identifiers_list[i]
            v_idx, k_v_idx, rho_v_idx, phi_v_idx = S_v_tuple
            
            # current_V_S_v = V[S_v_tuple].item() # V removed
            eta_v_ssm_val = active_eta[S_v_tuple].item() 
            alt_v_val = active_alt[S_v_tuple].item() # Get alt_v for transition tuple

            alt_u_std = alt_u_std_batch[i].item()
            eta_u_std_ssm = eta_u_std_ssm_batch[i].item()
            phase_u_std = phase_u_std_batch[i].item()
                
                # BUG FIX: The problem lies in what eta_v_ssm_val represents. It's the total elapsed time from the start of the flight to reaching waypoint v. If there was a cruise segment between u and v, eta_v_ssm_val is effectively:
                # eta_v_ssm_val = time_to_reach_u + cruise_time_from_u_to_v
                # The code calculates eta_u_trans_ssm_val by only subtracting a potential edge_climb_time_seconds. It completely ignores the cruise_time_from_u_to_v.
                
            edge_cruise_time_seconds = 0.0
            if not np.isnan(eta_u_std_ssm):
                # This is the travel time assuming cruise/descent between u and v
                edge_cruise_time_seconds = eta_v_ssm_val - eta_u_std_ssm
                if edge_cruise_time_seconds < 0:
                    edge_cruise_time_seconds = 0.0 # Should not happen in backward pass


            # --- Path 1: Standard Propagation (CRZ-CRZ, DES-DES, CRZ-DES) ---
            if alt_u_std >= 0 and not np.isnan(eta_u_std_ssm) and phase_u_std != PHASE_CLIMB and eta_u_std_ssm >= eta_takeoff_ssm:
                k_u_std_idx = _round_to_bin_idx((eta_u_std_ssm - min_time_overall_seconds) / delta_t_seconds_wall_clock, num_time_bins -1)
                rho_u_std_idx = 0 # For CRUISE or DESCENT at u, remaining climb is 0
                
                if k_u_std_idx != -1 :
                    # Cost calculation removed
                    # tailwind_mps_std = get_wind(...)
                    # cost_uv_std = cost_model(...)

                    # if not np.isinf(cost_uv_std): # Cost check removed
                    S_u_std_tuple = (u_idx, k_u_std_idx, rho_u_std_idx, phase_u_std)

                    if not (
                        np.isnan(eta_v_ssm_val) or
                        np.isnan(alt_v_val) or
                        np.isnan(alt_u_std) or
                        np.isnan(eta_u_std_ssm) or
                        phase_u_std == -1 # -1 is returned when phase_t (input argument) for get_next_state_bw is CLIMB (0). The get_next_state_bw only handles CRUISE/DESCENT (i.e., before the Top of Climb) so it does not handle that. 
                        # The climb case will be handled separately in Path 2 below.
                    ): # If any of the values are NaN, meaning that the get_next_state_bw returned an invalid state, skip appending to feasible_transitions_list
                        # Record feasible transition
                        transition = (
                            u_idx, k_u_std_idx, rho_u_std_idx, round(alt_u_std), phase_u_std,
                            v_idx, k_v_idx, rho_v_idx, round(alt_v_val), phi_v_idx,
                            eta_u_std_ssm, eta_v_ssm_val,
                        )
                        feasible_transitions_list.append(transition)

                        # Update active states if S_u_std_tuple is newly reached
                        if np.isnan(active_eta[S_u_std_tuple].item()):
                            active_alt[S_u_std_tuple] = alt_u_std
                            active_eta[S_u_std_tuple] = eta_u_std_ssm
                            active_phase_return[S_u_std_tuple] = phase_u_std
                                
                                # DEBUGGING REMOVED
                                # if u_idx == 331: # LERM
                                #     v_node = idx_to_node[v_idx]
                                #     S_v_tuple_debug = (v_idx, k_v_idx, rho_v_idx, phi_v_idx) # Renamed to avoid conflict
                                #     print(f'1> {v_node} <- LERM: {S_v_tuple_debug} <- {S_u_std_tuple}, ALT@u: {int(alt_u_std)}, ETA: {int(eta_v_ssm_val)} <- {int(eta_u_std_ssm)}')
            
            # --- Path 2: Transitions-based Propagation (CLIMB focus) ---
            if (u_idx, v_idx) in transitions_map_by_indices:
                eps_u_bins, alt_u_ft_from_trans, eps_v_bins = transitions_map_by_indices[(u_idx, v_idx)]
                v_actual_elapsed_bins_from_takeoff = (eta_v_ssm_val - eta_takeoff_ssm) / delta_t_seconds_climb
                alt_u_trans = alt_u_ft_from_trans
                
                phase_u_trans = -1
                rho_u_trans_idx = -1 # remaining climb time at u, which is roughly eps_bin_max - eps_u_bins
                eta_u_trans_ssm_val = float('nan') # wall-clock 
                k_u_trans_idx = -1
                
                valid_transition_to_climb_path = False # could be a transition from CLIMB or CRUISE

                # DEBUGGING
                # if u_idx == 185 and phi_v_idx == PHASE_CLIMB:
                    # print(f'POTENTIAL TRANSITION CLB-CLB TO LEMD')

                if phi_v_idx == PHASE_CLIMB: # <--- CLIMB TO CLIMB TRANSITION
                    phase_u_trans = PHASE_CLIMB
                    v_node = idx_to_node[v_idx]
                    u_node = idx_to_node[u_idx]
                    # Remaining climb time at u, based on elapsed climb time up to u from transitions
                    rho_u_trans_idx = _round_to_bin_idx(max(0.0, float(max_eps_bin) - eps_u_bins), max_eps_bin) # roughly max_eps_bin - eps_u_bins

                    # --- BUG FIX: Directly compute wall-clock time from elapsed climb time (epsilon) ---
                    # OLD: eta_u_trans_ssm_val = eta_v_ssm_val - edge_climb_time_seconds
                    eta_u_trans_ssm_val = eta_takeoff_ssm + (eps_u_bins * delta_t_seconds_climb)
                    valid_transition_to_climb_path = True

                    # DEBUGGING
                    # if u_idx == 331: # LERM
                    #     print(f'2.1> {v_node} <- LERM ADMT PTH 2 CLB/CLB ETAV = {eta_v_ssm_val:.0f} > {eta_takeoff_ssm:.0f}')
                    #     pass
                
                # SWITCHING FROM CRUISE TO CLIMB <--- THIS IS THE ONLY PLACE WHERE WE SWITCH FROM CRUISE TO CLIMB
                elif phi_v_idx == PHASE_CRUISE and \
                     max_eps_bin <= v_actual_elapsed_bins_from_takeoff <= max_eps_bin + climb_phase_switch_allowance_climb_time_bins:
                    phase_u_trans = PHASE_CLIMB
                    u_node = idx_to_node[u_idx]
                    v_node = idx_to_node[v_idx]
                    
                    effective_eps_v_bins_for_switch = float(max_eps_bin)
                    rho_u_trans_idx = _round_to_bin_idx(max(0.0, effective_eps_v_bins_for_switch - eps_u_bins), max_eps_bin) # basically max_eps_bin - eps_u_bins
                    edge_climb_time_seconds = (effective_eps_v_bins_for_switch - eps_u_bins) * delta_t_seconds_climb # effective_eps_v_bins_for_switch = max_eps_bin
                    
                    if edge_climb_time_seconds >= 0: # Must be a forward progression in climb time
                        # --- BUG FIX: Directly compute wall-clock time from elapsed climb time (epsilon) ---
                        # OLD: eta_u_trans_ssm_val = eta_v_ssm_val - edge_climb_time_seconds
                        eta_u_trans_ssm_val = eta_takeoff_ssm + (eps_u_bins * delta_t_seconds_climb)
                        valid_transition_to_climb_path = True

                        # # DEBUGGING
                        # if u_idx == 331: # LERM
                        #     print(f'2.2> {v_node} <- {u_node} ADMT PTH 2 CRZ/CLB ETAV = {eta_v_ssm_val:.0f} > {eta_takeoff_ssm:.0f}')
                        #     pass

                else:
                    pass
                    # print(f'X> PTH 2 NOT PRCD! phi_v_idx = {phi_v_idx}, phi_u_trans = {phase_u_trans}, rho_u_trans_idx = {rho_u_trans_idx}, v_actual_elapsed_bins_from_takeoff = {v_actual_elapsed_bins_from_takeoff}')
                    # print(f'{"v CRZ" if phi_v_idx == PHASE_CRUISE else "v DES" if phi_v_idx == PHASE_DESCENT else "v CLB"}')
                
                # THESE CONDITIONS WILL VERIFY THE TRANSITION PATH ONE MORE TIME WITH MORE CONDITIONS
                # BUT THE SWITCHING LOGIC IS ALREADY HANDLED ABOVE!
                if valid_transition_to_climb_path and alt_u_trans >= 0 and not np.isnan(eta_u_trans_ssm_val) and rho_u_trans_idx != -1 and eta_u_trans_ssm_val >= eta_takeoff_ssm:
                    k_u_trans_idx = _round_to_bin_idx((eta_u_trans_ssm_val - min_time_overall_seconds) / delta_t_seconds_wall_clock, num_time_bins-1)

                    if k_u_trans_idx != -1:
                        # Cost calculation removed
                        # tailwind_mps_trans = get_wind(...)
                        # cost_uv_trans = cost_model(...)

                        # if not np.isinf(cost_uv_trans): # Cost check removed
                        S_u_trans_tuple = (u_idx, k_u_trans_idx, rho_u_trans_idx, phase_u_trans)

                        # Record feasible transition
                        transition = (
                            u_idx, k_u_trans_idx, rho_u_trans_idx, round(alt_u_trans), phase_u_trans,
                            v_idx, k_v_idx, rho_v_idx, round(alt_v_val), phi_v_idx,
                            eta_u_trans_ssm_val, eta_v_ssm_val,
                        )
                        feasible_transitions_list.append(transition)
                        
                        # Update active states if S_u_trans_tuple is newly reached
                        if np.isnan(active_eta[S_u_trans_tuple].item()):
                            active_alt[S_u_trans_tuple] = alt_u_trans
                            active_eta[S_u_trans_tuple] = eta_u_trans_ssm_val
                            active_phase_return[S_u_trans_tuple] = phase_u_trans
                                
                                # DEBUGGING REMOVED
                                # if u_idx == 331: # LERM
                                #     v_node_debug = idx_to_node[v_idx] # Renamed
                                #     u_node_debug = idx_to_node[u_idx] # Renamed
                                #     S_v_tuple_debug = (v_idx, k_v_idx, rho_v_idx, phi_v_idx) # Renamed
                                #     print(f'2> {v_node_debug} <- {u_node_debug}: {S_v_tuple_debug}, {S_u_trans_tuple}, ALT: {int(alt_u_trans)}, ETA: {int(eta_u_trans_ssm_val)}')
                                
                else:
                    # print(f'2x> valid_transition_path = {valid_transition_to_climb_path}, alt_u_trans = {alt_u_trans}, eta_u_trans_ssm_val = {eta_u_trans_ssm_val} >? {eta_takeoff_ssm}, rho_u_trans_idx = {rho_u_trans_idx}')
                    pass
    # The old active_phase_return construction loop is removed as it's populated on the fly.
    # For returning active_phase, create a tensor indicating the phase if V is finite
    # active_phase_return = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), -1, dtype=torch.long, device=device)
    # for n in range(num_nodes):
    #     for t in range(num_time_bins):
    #         for r in range(num_rho_bins):
    #             for p in range(NUM_PHASES):
    #                 if not torch.isinf(V[n,t,r,p]):
    #                     active_phase_return[n,t,r,p] = p


    return feasible_transitions_list # Return the list of feasible transitions
