import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

from equinox.route.forward_state import get_next_state_fw, CLIMB, CRUISE, DESCENT
from equinox.route.get_wind import get_wind
from equinox.cost.cost_rev1 import CostRev1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_since_midnight_to_datetime
from equinox.helpers.haversine import MPS_TO_KNOTS

# It's good practice to define default narrow body jet profiles here or pass them
# For now, assuming they might be part of performance_model_params or loaded similarly
# from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE


def run_forward_dp(
    graph: nx.DiGraph,
    source_node_id: str, # Graph node ID (e.g., 'LEMD')
    takeoff_time_str: str, # e.g., "2023-04-01 12:00:00"
    source_elevation_ft: float, # Elevation of the source airport/node
    goal_elevation_ft: float, # Elevation of the goal airport/node (unused in fwd pass)
    cost_model: CostRev1,
    wind_model_date_str: str, # e.g., "2023-04-01" (for WindDate)
    wind_data_dir: str, # e.g., "data/era5"
    climb_profile: list, 
    descent_profile: list, 
    climb_vs_profile: list, 
    descent_vs_profile: list, 
    cruise_alt_ft: float,
    cruise_spd_kts: float,
    dist_matrix_np: np.ndarray,
    ac_matrix_np: np.ndarray,
    initial_alt_ft: float = 1000.0, # Initial altitude at source node relative to its elevation after takeoff
    delta_t_seconds: int = 300, # 5 minutes time window
    max_flight_duration_hours: int = 10, # Max duration to consider for time bins
    device: torch.device = None
):
    """
    Implements the forward dynamic programming algorithm for soft Bellman updates,
    processing nodes in topological generations for enhanced batching.

    Args:
        graph (nx.DiGraph): The route graph. Nodes should have 'coords' attribute (lat, lon).
        source_node_id (str): The ID of the source node in the graph.
        takeoff_time_str (str): ISO format takeoff time string.
        source_elevation_ft (float): Elevation of the source airport in ft.
        goal_elevation_ft (float): Elevation of the destination airport in ft (unused in forward pass).
        cost_model (CostRev1): Instantiated cost model.
        wind_model_date_str (str): Date string for initializing WindDate.
        wind_data_dir (str): Directory for wind data.
        climb_profile, descent_profile, climb_vs_profile, descent_vs_profile: Aircraft performance profiles.
        cruise_alt_ft (float): Cruise altitude in feet.
        cruise_spd_kts (float): Cruise speed in knots.
        dist_matrix_np (np.ndarray): 2D array of distances between node indices.
        ac_matrix_np (np.ndarray): 2D array of airspace charges between node indices.
        initial_alt_ft (float): Altitude at the source node (above its elevation) at takeoff_time_str.
                                This should be the aircraft's altitude AMSL.
        delta_t_seconds (int): Duration of each time bin in seconds.
        max_flight_duration_hours (int): Maximum flight duration to define the number of time bins.
        device (torch.device): PyTorch device to run computations on.

    Returns:
        torch.Tensor: Value function V[node_idx, time_bin_idx].
        torch.Tensor: active_eta[node_idx, time_bin_idx] (exact seconds since midnight).
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

    takeoff_seconds_since_midnight = datestr_to_seconds_since_midnight(takeoff_time_str)
    
    min_time_overall_seconds = takeoff_seconds_since_midnight
    max_time_overall_seconds = min_time_overall_seconds + max_flight_duration_hours * 3600
    num_time_bins = int((max_time_overall_seconds - min_time_overall_seconds) / delta_t_seconds) + 1

    V = torch.full((num_nodes, num_time_bins), float('inf'), dtype=torch.float64, device=device)
    active_alt = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device)
    active_phase = torch.full((num_nodes, num_time_bins), -1, dtype=torch.long, device=device)
    active_eta = torch.full((num_nodes, num_time_bins), float('nan'), dtype=torch.float64, device=device)

    performance_model = Performance(
        climb_profile, descent_profile,
        climb_vs_profile, descent_vs_profile,
        cruise_altitude_ft=cruise_alt_ft,
        cruise_speed_kts=cruise_spd_kts
    )
    # Climb performance table is relative to origin airport elevation.
    # Altitudes in get_next_state_fw and stored in active_alt should be AMSL.
    climb_perf_table = get_eta_and_distance_climb(performance_model, origin_airport_elevation_ft=source_elevation_ft) 

    wind_model = WindDate(date_str=wind_model_date_str, data_dir=wind_data_dir)
    
    dist_matrix = torch.from_numpy(dist_matrix_np).to(dtype=torch.float64, device=device)
    ac_matrix = torch.from_numpy(ac_matrix_np).to(dtype=torch.float64, device=device)

    initial_time_bin = 0
    V[s_idx, initial_time_bin] = 0.0
    # Ensure initial_alt_ft is AMSL. If it was given as AGL for source, it should be adjusted before this call.
    # Assuming initial_alt_ft is already AMSL as per updated docstring.
    active_alt[s_idx, initial_time_bin] = float(initial_alt_ft) 
    active_phase[s_idx, initial_time_bin] = CLIMB 
    active_eta[s_idx, initial_time_bin] = float(takeoff_seconds_since_midnight)

    # --- 2. Topological Generations for Node Processing Order ---
    try:
        # nx.topological_generations returns an iterator of sets of nodes.
        # Each set contains nodes where all predecessors are in previous sets.
        topo_generations_node_ids = list(nx.topological_generations(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph is not a DAG, cannot perform topological sort for generations.")

    # --- 3. Main DP Loop ---
    for generation_node_ids in topo_generations_node_ids:
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

                    u_coords_tuple = graph.nodes[u_node_graph_id].get('coords')
                    if u_coords_tuple is None:
                        print(f"Warning: Node {u_node_graph_id} has no coords. Skipping.")
                        continue
                    u_coords = [u_coords_tuple[0].item() if isinstance(u_coords_tuple[0], torch.Tensor) else u_coords_tuple[0],
                                u_coords_tuple[1].item() if isinstance(u_coords_tuple[1], torch.Tensor) else u_coords_tuple[1]]
                    
                    for v_node_id_succ in successors:
                        v_node_idx_succ = node_to_idx[v_node_id_succ]
                        v_coords_tuple = graph.nodes[v_node_id_succ].get('coords')
                        if v_coords_tuple is None:
                            print(f"Warning: Node {v_node_id_succ} has no coords. Skipping.")
                            continue
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
        eta_src_tensor = torch.tensor(batch_eta_src_list, dtype=torch.float64, device=device)
        phase_src_tensor = torch.tensor(batch_phase_src_list, dtype=torch.long, device=device)
        coords_tgt_tensor = torch.tensor(batch_coords_tgt_list, dtype=torch.float64, device=device)
        
        alt_v_new_batch, eta_v_new_batch, phase_v_new_batch = get_next_state_fw(
            coords_src_tensor, alts_src_tensor, eta_src_tensor, phase_src_tensor,
            coords_tgt_tensor, climb_perf_table, wind_model
        )
        
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
        
        for i in range(len(alt_v_new_batch)):
            v_node_idx = batch_v_node_indices_list[i] # This is the successor's index
            alt_v_new = alt_v_new_batch[i]
            eta_v_new = eta_v_new_batch[i]
            phase_v_new = phase_v_new_batch[i]
            cost_uv = cost_uv_batch[i]
            # V_u_val corresponds to V[original_u_node_idx, original_k_u] for this transition
            V_u_val = V_u_ku_tensor[i]

            if torch.isinf(cost_uv) or torch.isinf(V_u_val):
                continue

            time_since_takeoff_sec = eta_v_new - min_time_overall_seconds
            if time_since_takeoff_sec < 0:
                 continue

            k_v = int(torch.round(time_since_takeoff_sec / delta_t_seconds).item())

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


# Example Usage (Illustrative - requires actual data and models)
if __name__ == '__main__':
    print("Setting up illustrative example for forward_dp_vec2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G = nx.DiGraph()
    G.add_node("N0", coords=[37.7749, -122.4194]) 
    G.add_node("N1", coords=[38.123, -121.021])   
    G.add_node("N2", coords=[38.407, -117.179])  
    G.add_node("N3", coords=[40.7128, -74.0060])  
    G.add_edges_from([("N0", "N1"), ("N0", "N2"), ("N1", "N3"), ("N2", "N3")]) # Make N3 a common sink
    # For topological_generations to work well, N0 is gen0, N1,N2 are gen1, N3 is gen2
    
    node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
    node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)}
    num_actual_nodes = len(node_list_for_matrix)

    from equinox.helpers.haversine import haversine
    dummy_dist_matrix = np.full((num_actual_nodes, num_actual_nodes), float('inf'))
    dummy_ac_matrix = np.zeros((num_actual_nodes, num_actual_nodes))
    for u_name, v_name in G.edges():
        u_i, v_i = node_to_idx_for_matrix[u_name], node_to_idx_for_matrix[v_name]
        coords_u = G.nodes[u_name]['coords']
        coords_v = G.nodes[v_name]['coords']
        dummy_dist_matrix[u_i, v_i] = haversine(coords_u[0], coords_u[1], coords_v[0], coords_v[1])
        dummy_ac_matrix[u_i, v_i] = np.random.uniform(5, 20) 
    np.fill_diagonal(dummy_dist_matrix, 0)

    cost_model_instance = CostRev1(beta0=0.1, beta1=1.0, beta2=0.5, device=device)
    wind_date = "2023-04-01"
    import os
    dummy_wind_dir = "data/era5_dummy_test"
    os.makedirs(dummy_wind_dir, exist_ok=True)
    dummy_era5_file_path = os.path.join(dummy_wind_dir, f"era5_{wind_date.replace('-','')}.nc")
    
    if not os.path.exists(dummy_era5_file_path):
        try:
            import xarray as xr
            ds = xr.Dataset(
                {
                    "u10": (("time", "latitude", "longitude"), np.zeros((1,1,1))),
                    "v10": (("time", "latitude", "longitude"), np.zeros((1,1,1))),
                    "u": (("time", "level", "latitude", "longitude"), np.zeros((1,1,1,1))),
                    "v": (("time", "level", "latitude", "longitude"), np.zeros((1,1,1,1))),
                    "z": (("time", "level", "latitude", "longitude"), np.zeros((1,1,1,1))), 
                },
                coords={
                    "time": [datetime.fromisoformat(f"{wind_date}T00:00:00")],
                    "latitude": [0.0], "longitude": [0.0], "level": [1000] 
                }
            )
            ds.to_netcdf(dummy_era5_file_path)
            print(f"Created dummy ERA5 file: {dummy_era5_file_path}")
        except ImportError:
            print(f"Skipping dummy ERA5 file creation: xarray or dependencies not installed.")
            print(f"Please ensure {dummy_era5_file_path} exists or provide a valid wind_data_dir.")
    
    dummy_climb_profile = [(0, 0, 0), (10000, 300, 20), (20000, 700, 50), (35000, 1200, 100)]
    dummy_descent_profile = [(35000, 1200, 100), (20000, 700, 50), (10000, 300, 20), (0, 0, 0)]
    dummy_climb_vs_profile = [(0, 3000), (10000, 2500), (20000, 2000), (35000, 1500)]
    dummy_descent_vs_profile = [(35000, -1500), (20000, -2000), (10000, -2500), (0, -1000)]

    print("Running forward DP (illustrative)...")
    try:
        V_final, eta_final, alt_final, phase_final = run_forward_dp(
            graph=G,
            source_node_id="N0",
            takeoff_time_str=f"{wind_date} 12:00:00",
            source_elevation_ft=0.0, # Assuming SFO at sea level for dummy profile alignment
            goal_elevation_ft=0.0,   # Dummy
            cost_model=cost_model_instance,
            wind_model_date_str=wind_date,
            wind_data_dir=dummy_wind_dir, 
            climb_profile=dummy_climb_profile, 
            descent_profile=dummy_descent_profile, 
            climb_vs_profile=dummy_climb_vs_profile, 
            descent_vs_profile=dummy_descent_vs_profile, 
            cruise_alt_ft=35000.0,
            cruise_spd_kts=450.0,
            dist_matrix_np=dummy_dist_matrix,
            ac_matrix_np=dummy_ac_matrix,
            initial_alt_ft=0.0, # Start at 0ft AMSL if source_elevation_ft is 0 for profile alignment
            delta_t_seconds=600, 
            max_flight_duration_hours=5,
            device=device
        )

        print("\n--- Results ---")
        print(f"V function shape: {V_final.shape}")
        base_output_time_str = f"{wind_date} 00:00:00"
        # takeoff_ssm = datestr_to_seconds_since_midnight(f"{wind_date} 12:00:00") # Already available as min_time_val inside

        for node_idx_res in range(num_actual_nodes):
            for time_idx_res in range(V_final.shape[1]):
                if not torch.isinf(V_final[node_idx_res, time_idx_res]):
                    node_id_res = node_list_for_matrix[node_idx_res] # Use the consistent list
                    
                    # Calculate the approximate start time of this bin for display
                    # min_time_overall_seconds is takeoff_seconds_since_midnight from inside run_forward_dp
                    # This was the min_time_overall_seconds used to calculate k_v
                    takeoff_ssm_ref = datestr_to_seconds_since_midnight(f"{wind_date} 12:00:00")
                    bin_start_time_ssm = takeoff_ssm_ref + time_idx_res * 600 # 600 is delta_t_seconds from example

                    time_dt_display = seconds_since_midnight_to_datetime(base_output_time_str, bin_start_time_ssm)
                    
                    print(f"Node {node_id_res} ({node_idx_res}), Time Bin {time_idx_res} (approx arrival by {time_dt_display.strftime('%Y-%m-%d %H:%M:%S')}):")
                    print(f"  V = {V_final[node_idx_res, time_idx_res].item():.2f}")
                    active_eta_val = eta_final[node_idx_res, time_idx_res].item()
                    active_alt_val = alt_final[node_idx_res, time_idx_res].item()
                    active_phase_val = phase_final[node_idx_res, time_idx_res].item()
                    print(f"  Exact ETA: {seconds_since_midnight_to_datetime(base_output_time_str, active_eta_val).strftime('%Y-%m-%d %H:%M:%S') if not np.isnan(active_eta_val) else 'N/A'}")
                    print(f"  Altitude (ft): {active_alt_val if not np.isnan(active_alt_val) else 'N/A'}")
                    print(f"  Phase: {active_phase_val if active_phase_val != -1 else 'N/A'}")

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'dummy_era5_file_path' in locals() and os.path.exists(dummy_era5_file_path) and "dummy_test" in dummy_era5_file_path:
             try:
                 os.remove(dummy_era5_file_path)
                 if dummy_wind_dir == "data/era5_dummy_test": 
                    try:
                        os.rmdir(dummy_wind_dir)
                        print(f"Cleaned up dummy wind data directory: {dummy_wind_dir}")
                    except OSError as e_dir:
                        print(f"Could not remove dummy directory {dummy_wind_dir}: {e_dir} (might not be empty or created by this run if cleanup failed before)")
                 else: 
                    print(f"Cleaned up dummy wind file: {dummy_era5_file_path}") 

             except OSError as e_file:
                 print(f"Error cleaning up dummy file {dummy_era5_file_path}: {e_file}")
