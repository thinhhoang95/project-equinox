import torch
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import collections
import math

# Imports from the equinox project (adjust paths if necessary)
from equinox.cost.cost_rev1 import CostRev1 # Assuming CostRev1 is in this path
from equinox.wind.wind_model import WindModel # Assuming WindModel is in this path
from equinox.route.forward_state import get_next_state_fw 
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE # Example profile

# Phase identifiers (consistent with forward_state.py)
CLIMB, CRUISE, DESCENT = 0, 1, 2

DEFAULT_DELTA_T_SECONDS = 60
DEFAULT_MAX_FLIGHT_DURATION_HOURS = 6
DEFAULT_INITIAL_PHASE = CLIMB # CLIMB

def calculate_bearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    Calculates the initial bearing (degrees clockwise from North) from point 1 to point 2.
    """
    lat1_rad = math.radians(lat1_deg)
    lon1_rad = math.radians(lon1_deg)
    lat2_rad = math.radians(lat2_deg)
    lon2_rad = math.radians(lon2_deg)

    delta_lon = lon2_rad - lon1_rad

    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    
    initial_bearing_rad = math.atan2(y, x)
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    
    # Normalize to 0-360
    compass_bearing_deg = (initial_bearing_deg + 360) % 360
    return compass_bearing_deg


def soft_bellman_forward_pass(
    graph_gml_path: str,
    s_node_id: str,
    ts_datetime: datetime,
    cost_model: CostRev1,
    dist_matrix_path: str,
    charge_matrix_path: str,
    s_node_elevation_ft: float,
    cruise_altitude_ft: float,
    cruise_speed_kts: float,
    wind_model_data_dir: str = "data/era5", # Default from WindModel notes
    delta_t_seconds: int = DEFAULT_DELTA_T_SECONDS,
    max_flight_duration_hours: int = DEFAULT_MAX_FLIGHT_DURATION_HOURS,
    s_node_initial_phase: int = DEFAULT_INITIAL_PHASE,
    performance_model: Performance = None
):
    """
    Implements the soft Bellman forward pass for Max-Ent IRL.

    Args:
        graph_gml_path: Path to the GML graph file.
        s_node_id: ID of the source node in the graph.
        ts_datetime: Initial takeoff datetime object.
        cost_model: An initialized instance of the CostRev1 model.
        dist_matrix_path: Path to the .npy file for the distance matrix D.
        charge_matrix_path: Path to the .npy file for the airspace charge matrix AC.
        s_node_elevation_ft: Elevation of the source node in feet. Used for climb profile generation.
        cruise_altitude_ft: Cruise altitude in feet for performance model.
        cruise_speed_kts: Cruise speed in knots for performance model.
        wind_model_data_dir: Directory containing ERA5 NetCDF data files.
        delta_t_seconds: Duration of each time bin in seconds.
        max_flight_duration_hours: Maximum expected flight duration to determine time bins.
        s_node_initial_phase: Initial flight phase at the source node (CLIMB, CRUISE, DESCENT).

    Returns:
        torch.Tensor: The value function V(i, k) as a 2D tensor [num_nodes, num_time_bins].
                      V(i, k) = -log sum_{paths s->i at time bin k} exp(-cost(path)).
        dict: Mapping from node ID string to integer index used in the V_table.
        datetime: The reference start datetime used for binning (ts_datetime).
        int: delta_t_seconds used for binning.
    """
    device = cost_model.beta0.device # Get device from cost_model

    # 1. Load Graph
    G_nx = nx.read_gml(graph_gml_path)
    if not nx.is_directed_acyclic_graph(G_nx):
        print("Warning: Graph is not a DAG. Topological sort might fail or behavior is undefined.")

    node_list = list(G_nx.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    idx_to_node = {i: node_id for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if s_node_id not in node_to_idx:
        raise ValueError(f"Source node {s_node_id} not found in the graph.")

    # 2. Load Distance and Charge Matrices
    # Ensure they are loaded as torch tensors and moved to the correct device
    try:
        distance_matrix_np = np.load(dist_matrix_path)
        # Assuming D[i,j] where i,j are indices from node_to_idx
        # Need to ensure GML node IDs map correctly to these indices if they are not already 0..N-1 based
        # For now, assume matrices are indexed 0..N-1 corresponding to node_list order.
        dist_matrix = torch.from_numpy(distance_matrix_np).to(dtype=torch.float32, device=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Distance matrix not found: {dist_matrix_path}")
    
    try:
        airspace_charge_matrix_np = np.load(charge_matrix_path)
        charge_matrix = torch.from_numpy(airspace_charge_matrix_np).to(dtype=torch.float32, device=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Airspace charge matrix not found: {charge_matrix_path}")

    # 3. Initialize WindModel
    # WindModel uses date string from ts_datetime
    wind_model_date_str = ts_datetime.strftime("%Y-%m-%d")
    try:
        wind_model = WindModel(date_str=wind_model_date_str, data_dir=wind_model_data_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not initialize WindModel. Data for {wind_model_date_str} not found in {wind_model_data_dir}. Error: {e}")
    except ValueError as e: # Handles missing variables in dataset
        raise ValueError(f"Could not initialize WindModel due to missing variables. Error: {e}")


    # 4. Prepare Climb Performance Table for get_next_state_fw
    # NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, etc. are placeholders for actual profile data.
    # The Performance class expects specific profile structures.
    # For now, using NARROW_BODY_JET_CLIMB_PROFILE as specified in test_state_prediction for climb
    # The other profiles (descent, vs) are not directly used by get_eta_and_distance_climb.
    try:
        # These profiles would typically be loaded from a configuration or constants file
        # from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, \
        #                                                 NARROW_BODY_JET_DESCENT_PROFILE, \
        #                                                 NARROW_BODY_JET_CLIMB_VS_PROFILE, \
        #                                                 NARROW_BODY_JET_DESCENT_VS_PROFILE
        
        if performance_model is None:
            # If performance_model is not provided, use the example Narrow Body Jet profile.
            performance_model = Performance(
                climb_profile=NARROW_BODY_JET_CLIMB_PROFILE, # List of (alt_ft, tas_kts, vs_fpm, rocd_fpm)
                descent_profile=NARROW_BODY_JET_CLIMB_PROFILE, # Dummy, not used by get_eta_and_distance_climb
                climb_vs_profile=NARROW_BODY_JET_CLIMB_PROFILE, # Dummy
                descent_vs_profile=NARROW_BODY_JET_CLIMB_PROFILE, # Dummy
                cruise_altitude_ft=cruise_altitude_ft,
                cruise_speed_kts=cruise_speed_kts
            )
        climb_performance_table = get_eta_and_distance_climb(performance_model, initial_altitude_ft=s_node_elevation_ft)
        # climb_performance_table is List[Tuple[float, float, float]] (altitude_ft, eta_sec_from_profile_start, dist_nm_from_profile_start)
    except ImportError:
        raise ImportError("Ensure vnav_profiles_rev1.py and vnav_performance.py are accessible and contain necessary data/classes.")
    except Exception as e:
        raise RuntimeError(f"Failed to create climb performance table: {e}")


    # 5. Initialize Z_table (stores sum of exp(-cost))
    # Z_table will be indexed by [node_idx, time_bin_idx]
    num_time_bins = math.ceil((max_flight_duration_hours * 3600) / delta_t_seconds)
    Z_table = torch.full((num_nodes, num_time_bins), 0.0, dtype=torch.float32, device=device)

    # 6. Initial State and Propagator Queue
    s_idx = node_to_idx[s_node_id]
    
    # Time binning: ts_datetime is t=0 for binning purposes.
    # So, the first bin is index 0.
    initial_time_bin_idx = 0 # int(((ts_datetime - ts_datetime).total_seconds()) / delta_t_seconds)
    
    if not (0 <= initial_time_bin_idx < num_time_bins):
        raise ValueError(f"Initial time bin {initial_time_bin_idx} is out of configured range [0, {num_time_bins-1}]")

    Z_table[s_idx, initial_time_bin_idx] = 1.0 # Cost of path to source at ts is 0, so exp(-0)=1

    # Propagator queue stores: (u_idx, time_u_dt, alt_u_ft, phase_u, accumulated_exp_neg_cost_to_u)
    # accumulated_exp_neg_cost_to_u = exp(- sum of costs of edges from s to u)
    propagators = collections.deque()
    propagators.append((
        s_idx,
        ts_datetime,
        s_node_elevation_ft,
        s_node_initial_phase,
        torch.tensor(1.0, device=device) # exp(-cost_to_source)
    )) # all probability mass is located at the source node
    
    # Set to keep track of (u_idx, time_u_dt_rounded_to_sec, alt_u_ft, phase_u) to avoid redundant processing for very similar states
    # This is primarily to handle floating point issues with datetime if they lead to micro-variations.
    # Using a simpler tuple for visited: (u_idx, time_offset_seconds, alt_u_ft, phase_u)
    # A true cycle in detailed states should not happen if time always progresses via get_next_state_fw.
    # This queue approach naturally handles the DAG traversal without explicit topological sort if time always increases.

    processed_count = 0 # For monitoring

    # 7. Main Propagation Loop
    while propagators:
        u_idx, time_u_dt, alt_u_ft, phase_u, exp_neg_cost_s_u = propagators.popleft()
        
        processed_count += 1
        # if processed_count % 1000 == 0:
        #     print(f"Processed {processed_count} propagator states...")

        u_node_id = idx_to_node[u_idx]
        
        # Current time offset from ts_datetime for eta_src in get_next_state_fw
        time_offset_u_sec = (time_u_dt - ts_datetime).total_seconds()

        # Node coordinates for u (ensure lat, lon order for get_next_state_fw)
        # Graph usually has 'lon', 'lat' or 'x', 'y'. Assuming 'lat', 'lon' keys.
        try:
            lat_u, lon_u = G_nx.nodes[u_node_id]['lat'], G_nx.nodes[u_node_id]['lon']
        except KeyError:
            raise KeyError(f"Node {u_node_id} in graph is missing 'lat' or 'lon' attributes.")
        
        coords_u_tensor = torch.tensor([[lat_u, lon_u]], device=device, dtype=torch.float32)
        alts_u_tensor = torch.tensor([alt_u_ft], device=device, dtype=torch.float32)
        eta_u_tensor = torch.tensor([time_offset_u_sec], device=device, dtype=torch.float32)
        phase_u_tensor = torch.tensor([phase_u], device=device, dtype=torch.int64)


        for v_node_id in G_nx.successors(u_node_id):
            v_idx = node_to_idx[v_node_id]
            try:
                lat_v, lon_v = G_nx.nodes[v_node_id]['lat'], G_nx.nodes[v_node_id]['lon']
            except KeyError:
                # print(f"Warning: Successor node {v_node_id} is missing 'lat' or 'lon'. Skipping edge ({u_node_id}, {v_node_id}).")
                continue # Skip this successor
            coords_v_tensor = torch.tensor([[lat_v, lon_v]], device=device, dtype=torch.float32)

            # 7.1 State Propagation using get_next_state_fw
            try:
                alt_v_ft_tensor, time_offset_v_sec_tensor, phase_v_tensor = get_next_state_fw(
                    coords_src=coords_u_tensor,
                    alts_src=alts_u_tensor,
                    eta_src=eta_u_tensor,
                    phase_src=phase_u_tensor,
                    coords_tgt=coords_v_tensor,
                    climb_performance=climb_performance_table,
                    wind_model=wind_model # Pass the WindModel instance
                )
            except Exception as e:
                # print(f"Error in get_next_state_fw for edge ({u_node_id}, {v_node_id}): {e}. Skipping.")
                continue

            alt_v_ft = alt_v_ft_tensor.item()
            time_offset_v_sec = time_offset_v_sec_tensor.item()
            phase_v = phase_v_tensor.item()

            if torch.isnan(alt_v_ft_tensor) or torch.isinf(alt_v_ft_tensor) or \
               torch.isnan(time_offset_v_sec_tensor) or torch.isinf(time_offset_v_sec_tensor):
                # print(f"Invalid state from get_next_state_fw for edge ({u_node_id}, {v_node_id}). Skipping.")
                continue
            
            if time_offset_v_sec < time_offset_u_sec: # Should not happen if eta is cumulative
                 # print(f"Warning: Time did not progress for edge ({u_node_id}, {v_node_id}). {time_offset_u_sec} -> {time_offset_v_sec}. Skipping to avoid loops.")
                 continue

            time_v_dt = ts_datetime + timedelta(seconds=time_offset_v_sec)

            # 7.2 Calculate Cost c_uv
            # Wind at u (current alt_u_ft, time_u_dt)
            # WindModel's get_wind_components uses (lat, lon, alt_ft, time_datetime)
            u_wind_ms_u_comp, u_wind_ms_v_comp = wind_model.get_wind_components(lat_u, lon_u, alt_u_ft, time_u_dt)
            
            # Wind at v (predicted alt_v_ft, time_v_dt)
            v_wind_ms_u_comp, v_wind_ms_v_comp = wind_model.get_wind_components(lat_v, lon_v, alt_v_ft, time_v_dt)

            if np.isnan(u_wind_ms_u_comp) or np.isnan(u_wind_ms_v_comp) or \
               np.isnan(v_wind_ms_u_comp) or np.isnan(v_wind_ms_v_comp):
                # print(f"NaN wind components for edge ({u_node_id}, {v_node_id}). Assuming inf cost. Skipping.")
                continue

            avg_u_wind_ms = (u_wind_ms_u_comp + v_wind_ms_u_comp) / 2.0
            avg_v_wind_ms = (u_wind_ms_v_comp + v_wind_ms_v_comp) / 2.0
            
            bearing_deg = calculate_bearing(lat_u, lon_u, lat_v, lon_v) # Degrees clockwise from North
            bearing_rad = math.radians(bearing_deg)

            # Tailwind = u_wind * sin(bearing_rad_from_east_ccw) + v_wind * cos(bearing_rad_from_east_ccw)
            # If bearing_rad is clockwise from North:
            # u_wind (Eastward) corresponds to sin(bearing_rad) component of track vector (if track vector is unit North)
            # v_wind (Northward) corresponds to cos(bearing_rad) component of track vector
            # So, tailwind = avg_u_wind_ms * math.sin(bearing_rad) + avg_v_wind_ms * math.cos(bearing_rad)
            # This is component of wind in direction of travel.
            tailwind_ms = avg_u_wind_ms * math.sin(bearing_rad) + avg_v_wind_ms * math.cos(bearing_rad)
            tailwind_kts = tailwind_ms * 1.94384 # Convert m/s to knots

            # Get cost from cost model
            cost_uv = cost_model(
                edge_indices=(u_idx, v_idx), # Assumes cost_model can take integer indices
                distance_matrix_d=dist_matrix,
                airspace_charge_matrix_ac=charge_matrix,
                tailwind_value_w=torch.tensor(tailwind_kts, device=device, dtype=torch.float32)
            )

            if torch.isinf(cost_uv) or torch.isnan(cost_uv):
                # print(f"Infinite or NaN cost for edge ({u_node_id}, {v_node_id}). Skipping.")
                continue

            # 7.3 Update Z_table for v
            exp_neg_cost_uv = torch.exp(-cost_uv)
            exp_neg_cost_s_v_via_path = exp_neg_cost_s_u * exp_neg_cost_uv

            time_bin_v_idx = math.floor(time_offset_v_sec / delta_t_seconds) # math.floor for non-negative

            if 0 <= time_bin_v_idx < num_time_bins:
                Z_table[v_idx, time_bin_v_idx] += exp_neg_cost_s_v_via_path.detach() # Detach if exp_neg_cost_s_u has grad

                # 7.4 Add new propagator for v
                # The exp_neg_cost_s_v_via_path is specific to this one path, so it's the one to propagate
                propagators.append((
                    v_idx,
                    time_v_dt,
                    alt_v_ft,
                    phase_v,
                    exp_neg_cost_s_v_via_path 
                ))
            # else:
                # print(f"Time bin {time_bin_v_idx} for node {v_node_id} (time {time_v_dt}) is out of range [0, {num_time_bins-1}]. Path discarded.")

    # 8. Calculate V_table from Z_table
    # V = -log(Z). Handle Z=0 (log(0) = -inf)
    V_table = torch.full_like(Z_table, torch.inf) # Initialize with inf (for -log(0))
    valid_Z_mask = Z_table > 1e-9 # Small epsilon to avoid log(0) issues from very small Z
    V_table[valid_Z_mask] = -torch.log(Z_table[valid_Z_mask])
    
    # print(f"Completed forward pass. Processed {processed_count} propagator states.")
    return V_table, node_to_idx, ts_datetime, delta_t_seconds

if __name__ == '__main__':
    # This is a placeholder for a proper test or example usage.
    # To run this, you would need:
    # 1. A graph file (e.g., 'data/graph/LEMD_EGLL_2023_04_01.gml')
    # 2. Cost model parameters (beta0, beta1, beta2)
    # 3. Distance and charge matrices (.npy files)
    # 4. Source node details (ID, elevation)
    # 5. Cruise performance parameters
    # 6. Wind data for the specified date and directory
    
    print("Soft Bellman Forward Pass definition complete.")
    print("To use, call soft_bellman_forward_pass with appropriate arguments.")

    # Example (minimal, won't run without data and model setup):
    # try:
    #     # Dummy Cost Model (replace with actual initialization)
    #     dummy_cost_model = CostRev1(beta0=0.1, beta1=1.0, beta2=0.5) 
    #     dummy_cost_model.to(torch.device("cpu")) # Ensure device consistency
        
    #     # Dummy graph and data paths (replace with actual paths)
    #     # Need to create dummy files for this example to be runnable standalone
    #     # For instance, create a simple GML, and dummy NPY arrays.
        
    #     # Create dummy graph file
    #     G = nx.DiGraph()
    #     G.add_node("A", lat=40.0, lon=-3.0)
    #     G.add_node("B", lat=40.5, lon=-2.5)
    #     G.add_node("C", lat=41.0, lon=-2.0)
    #     G.add_edge("A", "B")
    #     G.add_edge("B", "C")
    #     nx.write_gml(G, "dummy_graph.gml")

    #     # Create dummy distance and charge matrices
    #     num_dummy_nodes = len(G.nodes)
    #     dummy_dist = np.random.rand(num_dummy_nodes, num_dummy_nodes).astype(np.float32) * 100
    #     dummy_charges = np.random.rand(num_dummy_nodes, num_dummy_nodes).astype(np.float32) * 10
    #     np.save("dummy_dist.npy", dummy_dist)
    #     np.save("dummy_charges.npy", dummy_charges)
        
    #     # Create dummy wind data directory and a file for WindModel to load
    #     # This is more complex; WindModel expects a specific NetCDF structure.
    #     # For a true test, a minimal valid .nc file would be needed.
    #     # Skipping full WindModel setup for this basic __main__ example.
    #     # For now, this example will likely fail at WindModel initialization if not handled.

    #     start_time = datetime(2023, 4, 1, 12, 0, 0)

    #     print(f"Attempting to run with dummy data (WindModel will likely fail without proper data)...")
    #     # This part will fail unless WindModel can handle missing data or you provide dummy NetCDF.
    #     # V, node_map, _, _ = soft_bellman_forward_pass(
    #     #     graph_gml_path="dummy_graph.gml",
    #     #     s_node_id="A",
    #     #     ts_datetime=start_time,
    #     #     cost_model=dummy_cost_model,
    #     #     dist_matrix_path="dummy_dist.npy",
    #     #     charge_matrix_path="dummy_charges.npy",
    #     #     s_node_elevation_ft=1000.0, # e.g., LEMD elevation
    #     #     cruise_altitude_ft=35000.0,
    #     #     cruise_speed_kts=450.0,
    #     #     wind_model_data_dir="data/era5_dummy", # Point to a dir that might exist or be created
    #     #     delta_t_seconds=600, # 10 min bins
    #     #     max_flight_duration_hours=3 
    #     # )
    #     # print("V_table (sample):")
    #     # print(V[node_map["C"], :])
        
    # except FileNotFoundError as e:
    #     print(f"Example run failed due to FileNotFoundError: {e}")
    #     print("This is expected if dummy data files/directories are not correctly set up, especially for WindModel.")
    # except Exception as e:
    #     print(f"An error occurred during the example run: {e}")
    # finally:
    #     # Clean up dummy files
    #     import os
    #     if os.path.exists("dummy_graph.gml"): os.remove("dummy_graph.gml")
    #     if os.path.exists("dummy_dist.npy"): os.remove("dummy_dist.npy")
    #     if os.path.exists("dummy_charges.npy"): os.remove("dummy_charges.npy")

    pass
