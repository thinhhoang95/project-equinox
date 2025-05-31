from math import floor
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
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.wind.wind_model import WindModel

MPS_TO_KNOTS = 1.9438444924406 # 1.9438444924406 m/s to kts

# Phase constants as indices
PHASE_CLIMB = CLIMB # Typically 0
PHASE_CRUISE = CRUISE # Typically 1
PHASE_DESCENT = DESCENT # Typically 2
NUM_PHASES = 3


def _round_to_bin_idx(value: float, max_bin_idx: int) -> int:
    """Safely rounds a float value to a valid bin index."""
    if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
        return -1 # Invalid index
    idx = int(floor(value))
    if not (0 <= idx <= max_bin_idx): # Max bin index is inclusive
        return -1
    return idx


def run_backward_dp(
    graph: nx.DiGraph,
    goal_node_id: str,
    estimated_landing_time_str: str,
    origin_elevation_ft: float, # Currently unused in backward pass but kept for API consistency
    destination_elevation_ft: float,
    cost_model: CostRev1,
    wind_model: WindModel,
    performance_model: Performance,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    transitions_list: list, # List of (u_idx, eps_u_bins, alt_u_ft, v_idx, eps_v_bins, alt_v_ft), retrieved from equinox.dp.pretoc.forward_soft_bellman; see the test_toc_forward_dp.ipynb notebook. 
    eta_takeoff_str: str,
    max_eps_bin: int = 36, # Max climb time in number of bins, according to climb performance table
    final_alt_ft: float = 0.0,
    delta_t_seconds_wall_clock: int = 300,
    delta_t_seconds_climb: int = 30,
    max_flight_duration_hours: int = 10,
    climb_phase_switch_allowance_climb_time_bins: int = 40, # N_bin_max, in wall-clock bins
    device: torch.device = None,
    temperature: float = 1.0
):
    """
    Implements the backward dynamic programming algorithm for aircraft trajectory optimization
    with a 4D state space: (waypoint, wall-clock time bin, remaining climb time bin, phase).
    
    This function solves the optimal trajectory problem by working backwards from the destination,
    computing minimum costs to reach the goal from each possible state. The algorithm handles
    three flight phases (CLIMB, CRUISE, DESCENT) and processes nodes in topological generations
    for efficient batching. It supports both standard flight propagation and explicit climb
    transitions with soft Bellman updates for robust optimization.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph representing the flight route network. Each node should have 'lat' and 'lon'
        attributes representing waypoint coordinates.
    goal_node_id : str
        Identifier of the destination node (e.g., "EGLL" for London Heathrow).
    estimated_landing_time_str : str
        Expected landing time in format "YYYY-MM-DD HH:MM:SS" (e.g., "2023-04-01 12:00:00").
    origin_elevation_ft : float
        Elevation of origin airport in feet. Currently unused in backward pass but kept for
        API consistency with forward pass algorithms.
    destination_elevation_ft : float
        Elevation of destination airport in feet, used for descent performance calculations.
    cost_model : CostRev1
        Cost model instance that computes flight costs based on distance, airspace charges,
        and wind conditions.
    wind_model : WindModel
        Wind model providing wind data for trajectory calculations. Can be WindDate for
        historical data or WindFree for no-wind scenarios.
    performance_model : Performance
        Aircraft performance model containing climb/descent profiles, cruise parameters,
        and vertical speed profiles for trajectory calculations.
    dist_matrix_np : np.ndarray
        Distance matrix between waypoints, shape (num_nodes, num_nodes).
    ac_matrix_np : np.ndarray
        Airspace charges matrix between waypoints, shape (num_nodes, num_nodes).
    transitions_list : list
        List of climb transition tuples in format:
        (u_idx, eps_u_bins, alt_u_ft, v_idx, eps_v_bins, alt_v_ft)
        where eps_*_bins are elapsed climb time bins and alt_*_ft are altitudes.
        Retrieved from forward DP preprocessing.
    eta_takeoff_str : str
        Estimated takeoff time in format "YYYY-MM-DD HH:MM:SS" (e.g., "2023-04-01 10:15:00").
    max_eps_bin : int, default=36
        Maximum climb time in number of bins, derived from climb performance table.
    final_alt_ft : float, default=0.0
        Final altitude at destination in feet. If 0.0, uses destination_elevation_ft.
    delta_t_seconds_wall_clock : int, default=300
        Time discretization for wall-clock time bins in seconds (5 minutes).
    delta_t_seconds_climb : int, default=30
        Time discretization for climb time bins in seconds (30 seconds).
    max_flight_duration_hours : int, default=10
        Maximum allowed flight duration in hours, defines the time window.
    climb_phase_switch_allowance_climb_time_bins : int, default=40
        Additional time bins allowed for switching from cruise to climb phase.
    device : torch.device, optional
        PyTorch device for computations. If None, automatically selects CUDA if available.
    temperature : float, default=1.0
        Temperature parameter for soft Bellman updates. Lower values make updates more greedy.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A 4-tuple containing:
        
        V : torch.Tensor
            Value function tensor of shape (num_nodes, num_time_bins, num_rho_bins, NUM_PHASES).
            Contains minimum costs to reach the goal from each state. Infinite values indicate
            unreachable states.
            
        active_eta : torch.Tensor
            Active estimated time of arrival tensor of same shape as V. Contains the wall-clock
            time (seconds since midnight) for each reachable state.
            
        active_alt : torch.Tensor
            Active altitude tensor of same shape as V. Contains the altitude in feet for each
            reachable state.
            
        active_phase_return : torch.Tensor
            Active phase tensor of same shape as V with dtype=torch.long. Contains the flight
            phase index (0=CLIMB, 1=CRUISE, 2=DESCENT) for each reachable state, or -1 for
            unreachable states.

    Raises
    ------
    ValueError
        If goal_node_id is not found in the graph, if estimated landing time is outside the
        defined time window, or if the reversed graph is not a DAG.

    Examples
    --------
    Basic usage with LEMD to EGLL route:

    >>> import networkx as nx
    >>> from equinox.cost.cost_model_1 import cost_model_1
    >>> from equinox.wind.wind_free import WindFree
    >>> from equinox.vnav.vnav_performance import Performance
    >>> import numpy as np
    >>> import torch
    >>> 
    >>> # Load route graph and matrices
    >>> G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    >>> dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")
    >>> ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")
    >>> 
    >>> # Load transitions from forward DP preprocessing
    >>> import pickle
    >>> transitions_list = pickle.load(open("data/graph/LEMD_EGLL_2023_04_01_climb_transitions.pkl", "rb"))
    >>> 
    >>> # Setup models
    >>> wind_model = WindFree()
    >>> performance_model = Performance(
    ...     climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
    ...     descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
    ...     climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
    ...     descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
    ...     cruise_altitude_ft=35000.0,
    ...     cruise_speed_kts=450.0,
    ... )
    >>> 
    >>> # Run backward DP
    >>> V, active_eta, active_alt, active_phase = run_backward_dp(
    ...     graph=G,
    ...     goal_node_id="EGLL",
    ...     estimated_landing_time_str="2023-04-01 12:00:00",
    ...     origin_elevation_ft=0.0,
    ...     destination_elevation_ft=0.0,
    ...     cost_model=cost_model_1,
    ...     wind_model=wind_model,
    ...     performance_model=performance_model,
    ...     dist_matrix_np=dist_matrix,
    ...     ac_matrix_np=ac_matrix,
    ...     transitions_list=transitions_list,
    ...     eta_takeoff_str="2023-04-01 10:15:00",
    ...     max_flight_duration_hours=5,
    ...     temperature=5e-3
    ... )

    Advanced usage with custom parameters:

    >>> # Use CUDA device and custom time discretization
    >>> device = torch.device("cuda")
    >>> V, active_eta, active_alt, active_phase = run_backward_dp(
    ...     graph=G,
    ...     goal_node_id="EGLL",
    ...     estimated_landing_time_str="2023-04-01 12:00:00",
    ...     origin_elevation_ft=0.0,
    ...     destination_elevation_ft=0.0,
    ...     cost_model=cost_model_1,
    ...     wind_model=wind_model,
    ...     performance_model=performance_model,
    ...     dist_matrix_np=dist_matrix,
    ...     ac_matrix_np=ac_matrix,
    ...     transitions_list=transitions_list,
    ...     eta_takeoff_str="2023-04-01 10:15:00",
    ...     delta_t_seconds_wall_clock=180,  # 3-minute time bins
    ...     delta_t_seconds_climb=15,        # 15-second climb bins
    ...     climb_phase_switch_allowance_climb_time_bins=20,
    ...     device=device,
    ...     temperature=1e-2
    ... )

    Notes
    -----
    The algorithm implements two propagation paths:
    
    1. **Standard Propagation**: Handles CRUISE and DESCENT phases using the get_next_state_bw
       function for backward state transitions.
       
    2. **Transitions-based Propagation**: Handles CLIMB phase and phase transitions using
       precomputed climb transitions from forward DP preprocessing.
    
    The soft Bellman update mechanism allows for robust optimization by combining multiple
    paths using the logaddexp function with temperature scaling.
    
    State space dimensions:
    - Waypoint: Graph nodes (airports, waypoints, etc.)
    - Wall-clock time: Discretized time from takeoff to landing
    - Remaining climb time: Time remaining in climb phase
    - Phase: CLIMB (0), CRUISE (1), or DESCENT (2)
    
    The algorithm processes nodes in reverse topological order to ensure that when computing
    the value for a node, all its successors have already been processed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialization ---
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
    max_eps_bin = _round_to_bin_idx(climb_time_max / delta_t_seconds_climb, max_eps_bin) + 1

    num_rho_bins = max_eps_bin + 1 # Remaining climb time bins from 0 to max_eps_bin

    V = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('nan'), dtype=torch.float64, device=device)
    active_eta = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), float('nan'), dtype=torch.float64, device=device)
    # active_phase is implicitly the 4th dimension index
    
    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    transitions_map_by_indices = {(t[0], t[3]): (t[1], t[2], t[4]) for t in transitions_list} # (u_idx, v_idx) -> (eps_u_bins, alt_u_ft, eps_v_bins)

    # Initialize at goal node
    goal_node_time_since_min_overall = estimated_landing_ssm - min_time_overall_seconds
    landing_time_bin_idx = _round_to_bin_idx(goal_node_time_since_min_overall / delta_t_seconds_wall_clock, num_time_bins -1)
    
    if landing_time_bin_idx == -1:
        raise ValueError("Estimated landing time is outside the defined time window.")

    rho_g_idx = 0 # At landing, remaining climb time is 0
    phase_g_idx = PHASE_DESCENT
    
    current_final_alt = float(final_alt_ft if final_alt_ft != 0.0 else destination_elevation_ft)
    
    goal_state_tuple = (g_idx, landing_time_bin_idx, rho_g_idx, phase_g_idx)
    V[goal_state_tuple] = 0.0
    active_alt[goal_state_tuple] = current_final_alt
    active_eta[goal_state_tuple] = float(estimated_landing_ssm)

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
                        if not torch.isinf(V[current_S_v_tuple]):
                            alt_v_amsl = active_alt[current_S_v_tuple].item()
                            eta_v_ssm = active_eta[current_S_v_tuple].item()

                            if np.isnan(alt_v_amsl) or np.isnan(eta_v_ssm):
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
            
            current_V_S_v = V[S_v_tuple].item()
            eta_v_ssm_val = active_eta[S_v_tuple].item() # Needed for Path 2 logic, this is different from eta_v_ssm, which is the batch tensor used for vectorized get_next_state_bw
            
            # v_coords for cost model (not needed directly as we use indices, but for context)
            # u_coords for cost model (similarly)

            alt_u_std = alt_u_std_batch[i].item()
            eta_u_std_ssm = eta_u_std_ssm_batch[i].item()
            phase_u_std = phase_u_std_batch[i].item()

            # --- Path 1: Standard Propagation (CRZ/DES-CRZ/DES) ---
            if alt_u_std >= 0 and not np.isnan(eta_u_std_ssm) and phase_u_std != PHASE_CLIMB and eta_u_std_ssm >= eta_takeoff_ssm: # Standard path not for climb phase at u
                # note phase_u_std is from get_next_state_bw, and always CRUISE or DESCENT so this conditions always holds.
                # this is expected, because we ALWAYS "cruise" to the next predecessor to account for a "late switch to climb" from cruise phase.
                # DEBUGGING
                # if u_idx == 331: # LERM
                #   print(f'0.1> LERM ADMT PTH 1 ETA = {eta_u_std_ssm:.0f} > {eta_takeoff_ssm:.0f}')
                #   pass
                k_u_std_idx = _round_to_bin_idx((eta_u_std_ssm - min_time_overall_seconds) / delta_t_seconds_wall_clock, num_time_bins -1)
                rho_u_std_idx = 0 # For CRUISE or DESCENT at u, remaining climb is 0
                
                if k_u_std_idx != -1 : # rho_u_std_idx is always valid if phase is CRUISE/DESCENT
                    # Wind for cost: evaluated at u_std for segment u->v
                    tailwind_mps_std = get_wind(
                        coords_src_tensor[i:i+1], coords_tgt_tensor[i:i+1], # u_coords, v_coords
                        torch.tensor([alt_u_std], device=device, dtype=torch.float64),
                        torch.tensor([eta_u_std_ssm], device=device, dtype=torch.float64),
                        wind_model
                    )
                    cost_uv_std = cost_model(
                        (torch.tensor([u_idx], device=device), torch.tensor([v_idx], device=device)),
                        dist_matrix, ac_matrix, tailwind_mps_std * MPS_TO_KNOTS
                    ).item()

                    if not np.isinf(cost_uv_std):
                        val_to_add_std = current_V_S_v + cost_uv_std
                        S_u_std_tuple = (u_idx, k_u_std_idx, rho_u_std_idx, phase_u_std)
                        current_V_S_u_std = V[S_u_std_tuple].item()

                        if np.isinf(current_V_S_u_std) or val_to_add_std < current_V_S_u_std : # Heuristic update for soft DP
                             # Soft Bellman update (min convention for costs)
                            if np.isinf(current_V_S_u_std):
                                V[S_u_std_tuple] = val_to_add_std
                            else:
                                V[S_u_std_tuple] = -temperature * torch.logaddexp(
                                    torch.tensor(-current_V_S_u_std / temperature, device=device),
                                    torch.tensor(-val_to_add_std / temperature, device=device)
                                ).item()
                            
                            # Update active states if this path is chosen by heuristic
                            active_alt[S_u_std_tuple] = alt_u_std
                            active_eta[S_u_std_tuple] = eta_u_std_ssm

                            # DEBUGGING
                            if u_idx == 331: # LERM
                                v_node = idx_to_node[v_idx]
                                S_v_tuple = (v_idx, k_v_idx, rho_v_idx, phi_v_idx)
                                print(f'1> {v_node} <- LERM: {S_v_tuple} <- {S_u_std_tuple}, ALT@u: {int(alt_u_std)}, ETA: {int(eta_v_ssm_val)} <- {int(eta_u_std_ssm)}')
            
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

                if phi_v_idx == PHASE_CLIMB:
                    phase_u_trans = PHASE_CLIMB
                    v_node = idx_to_node[v_idx]
                    # Remaining climb time at u, based on elapsed climb time up to u from transitions
                    rho_u_trans_idx = _round_to_bin_idx(max(0.0, float(max_eps_bin) - eps_u_bins), max_eps_bin)

                    edge_climb_time_seconds = (eps_v_bins - eps_u_bins) * delta_t_seconds_climb
                    eta_u_trans_ssm_val = eta_v_ssm_val - edge_climb_time_seconds
                    valid_transition_to_climb_path = True

                    # DEBUGGING
                    if u_idx == 331: # LERM
                        print(f'2.1> {v_node} <- LERM ADMT PTH 2 CLB/CLB ETAV = {eta_v_ssm_val:.0f} > {eta_takeoff_ssm:.0f}')
                        pass
                
                # SWITCHING FROM CRUISE TO CLIMB <--- THIS IS THE ONLY PLACE WHERE WE SWITCH TO CLIMB
                elif phi_v_idx == PHASE_CRUISE and \
                     max_eps_bin <= v_actual_elapsed_bins_from_takeoff <= max_eps_bin + climb_phase_switch_allowance_climb_time_bins:
                    phase_u_trans = PHASE_CLIMB
                    u_node = idx_to_node[u_idx]
                    v_node = idx_to_node[v_idx]
                    
                    effective_eps_v_bins_for_switch = float(max_eps_bin)
                    rho_u_trans_idx = _round_to_bin_idx(max(0.0, effective_eps_v_bins_for_switch - eps_u_bins), max_eps_bin) # basically max_eps_bin - eps_u_bins
                    edge_climb_time_seconds = (effective_eps_v_bins_for_switch - eps_u_bins) * delta_t_seconds_climb # effective_eps_v_bins_for_switch = max_eps_bin
                    
                    if edge_climb_time_seconds >= 0: # Must be a forward progression in climb time
                        eta_u_trans_ssm_val = eta_v_ssm_val - edge_climb_time_seconds
                        valid_transition_to_climb_path = True

                        # DEBUGGING
                        if u_idx == 331: # LERM
                            print(f'2.2> {v_node} <- {u_node} ADMT PTH 2 CRZ/CLB ETAV = {eta_v_ssm_val:.0f} > {eta_takeoff_ssm:.0f}')
                            pass

                else:
                    pass
                    # print(f'X> PTH 2 NOT PRCD! phi_v_idx = {phi_v_idx}, phi_u_trans = {phase_u_trans}, rho_u_trans_idx = {rho_u_trans_idx}, v_actual_elapsed_bins_from_takeoff = {v_actual_elapsed_bins_from_takeoff}')
                    # print(f'{"v CRZ" if phi_v_idx == PHASE_CRUISE else "v DES" if phi_v_idx == PHASE_DESCENT else "v CLB"}')
                
                # THESE CONDITIONS WILL VERIFY THE TRANSITION PATH ONE MORE TIME WITH MORE CONDITIONS
                # BUT THE SWITCHING LOGIC IS ALREADY HANDLED ABOVE!
                if valid_transition_to_climb_path and alt_u_trans >= 0 and not np.isnan(eta_u_trans_ssm_val) and rho_u_trans_idx != -1 and eta_u_trans_ssm_val >= eta_takeoff_ssm:
                    k_u_trans_idx = _round_to_bin_idx((eta_u_trans_ssm_val - min_time_overall_seconds) / delta_t_seconds_wall_clock, num_time_bins-1)

                    if k_u_trans_idx != -1:
                        # Wind for cost: evaluated at u_trans for segment u->v
                        tailwind_mps_trans = get_wind(
                            coords_src_tensor[i:i+1], coords_tgt_tensor[i:i+1], # u_coords, v_coords
                            torch.tensor([alt_u_trans], device=device, dtype=torch.float64),
                            torch.tensor([eta_u_trans_ssm_val], device=device, dtype=torch.float64),
                            wind_model
                        )
                        cost_uv_trans = cost_model(
                            (torch.tensor([u_idx], device=device), torch.tensor([v_idx], device=device)),
                            dist_matrix, ac_matrix, tailwind_mps_trans * MPS_TO_KNOTS
                        ).item()

                        if not np.isinf(cost_uv_trans):
                            val_to_add_trans = current_V_S_v + cost_uv_trans
                            S_u_trans_tuple = (u_idx, k_u_trans_idx, rho_u_trans_idx, phase_u_trans)
                            current_V_S_u_trans = V[S_u_trans_tuple].item()

                            if np.isinf(current_V_S_u_trans) or val_to_add_trans < current_V_S_u_trans: # Heuristic update
                                if np.isinf(current_V_S_u_trans):
                                    V[S_u_trans_tuple] = val_to_add_trans
                                else:
                                    V[S_u_trans_tuple] = -temperature * torch.logaddexp(
                                        torch.tensor(-current_V_S_u_trans / temperature, device=device),
                                        torch.tensor(-val_to_add_trans / temperature, device=device)
                                    ).item()
                                
                                active_alt[S_u_trans_tuple] = alt_u_trans
                                active_eta[S_u_trans_tuple] = eta_u_trans_ssm_val

                                # DEBUGGING
                                if u_idx == 331: # LERM
                                    v_node = idx_to_node[v_idx]
                                    u_node = idx_to_node[u_idx]
                                    S_v_tuple = (v_idx, k_v_idx, rho_v_idx, phi_v_idx)
                                    print(f'2> {v_node} <- {u_node}: {S_v_tuple}, {S_u_trans_tuple}, ALT: {int(alt_u_trans)}, ETA: {int(eta_u_trans_ssm_val)}')
                                
                else:
                    # print(f'2x> valid_transition_path = {valid_transition_to_climb_path}, alt_u_trans = {alt_u_trans}, eta_u_trans_ssm_val = {eta_u_trans_ssm_val} >? {eta_takeoff_ssm}, rho_u_trans_idx = {rho_u_trans_idx}')
                    pass
    # For returning active_phase, create a tensor indicating the phase if V is finite
    active_phase_return = torch.full((num_nodes, num_time_bins, num_rho_bins, NUM_PHASES), -1, dtype=torch.long, device=device)
    for n in range(num_nodes):
        for t in range(num_time_bins):
            for r in range(num_rho_bins):
                for p in range(NUM_PHASES):
                    if not torch.isinf(V[n,t,r,p]):
                        active_phase_return[n,t,r,p] = p


    return V, active_eta, active_alt, active_phase_return
