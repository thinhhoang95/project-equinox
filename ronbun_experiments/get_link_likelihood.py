"""
Link Likelihood Computation Module

This module implements the computation of link traversal likelihood based on 
forward and backward soft value functions obtained from soft Bellman message passing.

The module computes the expected probability of traversing each link between waypoints
by aggregating the probabilities of all state transitions within that link.
"""

import os
import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
from collections import defaultdict

# Import required modules
from equinox.dp.trespass.sparse_io_utils import load_sparse_coo_tensor_with_convention


def load_value_function_files(base_directory: str, flight_id: str) -> Tuple[str, str, str, str]:
    """
    Load file paths for transitions, value functions, and cost tensor.
    
    Args:
        base_directory: Base directory containing the sample data
        flight_id: Flight identifier
    
    Returns:
        Tuple of (transitions_file_path, forward_v_path, backward_v_path, cost_file_path)
    """
    sample_dir = os.path.join(base_directory, flight_id)
    
    transitions_file = os.path.join(sample_dir, f"{flight_id}_REACHABLE_WIND.pkl")
    forward_v_file = os.path.join(sample_dir, f"{flight_id}_V_FWD_SPRSE_WIND.pt")
    backward_v_file = os.path.join(sample_dir, f"{flight_id}_V_BWD_SPRSE_WIND.pt")
    cost_file = os.path.join(sample_dir, f"{flight_id}_COST_WIND.pt")
    
    return transitions_file, forward_v_file, backward_v_file, cost_file


def load_dense_v(sparse_file_path: str, v_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Load sparse value function tensor and convert to dense format.
    
    This function loads a sparse tensor from file and converts it to a dense tensor
    with infinite values for implicit zeros, matching the convention used in the codebase.
    
    Args:
        sparse_file_path: Path to the sparse tensor file
        v_shape: Expected shape of the dense tensor
        device: PyTorch device for tensor operations
    
    Returns:
        Dense tensor with finite values from sparse tensor and inf for implicit zeros
    """
    # Load sparse tensor
    sparse_v, interpretation_note = load_sparse_coo_tensor_with_convention(
        sparse_file_path, target_device=device
    )
    sparse_v = sparse_v.coalesce()
    
    # Create dense tensor filled with infinity
    dense_v = torch.full(v_shape, float('inf'), dtype=sparse_v.dtype, device=device)
    
    # Fill in the finite values from sparse tensor
    indices = sparse_v.indices()
    values = sparse_v.values()
    if values.numel() > 0:
        dense_v[tuple(indices)] = values
    
    return dense_v


def determine_value_function_shape(state_transitions: list) -> Tuple[int, int, int, int]:
    """
    Determine the shape of value function tensors from state transitions.
    
    Args:
        state_transitions: List of state transitions
        
    Returns:
        Tuple of (num_nodes, num_time_bins, num_rho_bins, num_phases)
    """
    if not state_transitions:
        raise ValueError("No state transitions provided")
    
    # Extract maximum indices from transitions
    # Transition format: (u_wp, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp, k_v_idx, rho_v_idx, alt_v, phase_v)
    max_node_idx = 0
    max_k_idx = 0
    max_rho_idx = 0
    max_phase_idx = 0
    
    for trans in state_transitions:
        u_wp, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp, k_v_idx, rho_v_idx, alt_v, phase_v = trans
        
        max_node_idx = max(max_node_idx, u_wp, v_wp)
        max_k_idx = max(max_k_idx, k_u_idx, k_v_idx)
        max_rho_idx = max(max_rho_idx, rho_u_idx, rho_v_idx)
        max_phase_idx = max(max_phase_idx, phase_u, phase_v)
    
    # Add 1 because indices are 0-based
    num_nodes = max_node_idx + 1
    num_time_bins = max_k_idx + 1
    num_rho_bins = max_rho_idx + 1
    num_phases = max_phase_idx + 1
    
    return num_nodes, num_time_bins, num_rho_bins, num_phases


def compute_partition_function(V: torch.Tensor, node_idx: int, gamma: float) -> torch.Tensor:
    """
    Compute the partition function Z from a value function at a specific node.
    
    The partition function represents the sum of probabilities of all possible paths
    through the state-space graph.
    
    Args:
        V: Value function (cost-to-go V_b, or cost-from-start V_f)
        node_idx: Index of the waypoint (origin for V_b, destination for V_f)
        gamma: Temperature parameter
        
    Returns:
        -gamma * log(Z)
    """
    # Get values at the specified node across all states
    v_node = V[node_idx]
    
    # Filter out infinite values
    finite_mask = torch.isfinite(v_node)
    non_inf_v = v_node[finite_mask]
    
    if non_inf_v.numel() == 0:
        return torch.tensor(float('inf'), device=V.device)
    
    # Compute log partition function: log(sum(exp(-V/gamma)))
    # Using logsumexp for numerical stability: log_partition_z = -gamma * log(Z)
    log_partition_z = -gamma * torch.logsumexp(-non_inf_v / gamma, dim=0)
    
    return log_partition_z


def compute_link_likelihood(
    base_directory: str,
    flight_id: str,
    graph_file_path: str,
    goal_node: str,
    source_node: str = None,
    gamma: float = 1.0,
    device: Optional[torch.device] = None,
    save: bool = True,
    debug_coordinates: Optional[list] = None,
    debug_cell_size_nm: float = 10.0,
    debug_log_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute link traversal likelihood from forward and backward value functions.
    
    This function implements the link likelihood computation procedure described in
    the background documentation, which calculates the expected probability of
    traversing each link between waypoints.
    
    The computation follows three main steps:
    1. Compute the partition function Z from forward values at the goal node
    2. Compute state transition probabilities using the formula:
       P(s_u -> s_v) = (1/Z) * exp(-(V_f(s_u) + c(s_u,s_v) + V_b(s_v))/gamma)
    3. Aggregate probabilities by link: N_expected(u,v) = sum over all s_u in u, s_v in v of P(s_u -> s_v)
    
    Args:
        base_directory: Base directory containing sample data 
        flight_id: Flight identifier
        graph_file_path: Path to the NetworkX graph file (.gml)
        goal_node: ID of the goal waypoint.
        gamma: Temperature parameter for soft operations (default: 1.0)
        device: PyTorch device for computations (default: auto-detect)
        save: Whether to save the likelihood tensor to file (default: True)
        debug_coordinates: List of (lat, lon) tuples for debug logging.
        debug_cell_size_nm: Cell size for debug grid.
        debug_log_dir: Directory to save debug logs.
        
    Returns:
        2D tensor of shape (num_nodes, num_nodes) with element (i,j) indicating
        the likelihood for the link between waypoint i and waypoint j
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Computing link likelihood for flight {flight_id}")
    print(f"Using device: {device}")
    
    # Load graph to get node mappings
    print(f"Loading graph from {graph_file_path}")
    graph = nx.read_gml(graph_file_path)
    node_id_to_idx = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    if goal_node not in node_id_to_idx:
        raise ValueError(f"Goal node '{goal_node}' not found in graph.")
    goal_node_idx = node_id_to_idx[goal_node]

    if source_node is not None:
        if source_node not in node_id_to_idx:
            raise ValueError(f"Source node '{source_node}' not found in graph.")
        source_node_idx = node_id_to_idx[source_node]

    # --- Debugging setup ---
    debug_cell_contributions = defaultdict(list)
    debug_cells_map = {}
    lat_bins, lon_bins = None, None
    idx_to_node_id = {}
    node_coords = {}

    if debug_coordinates and debug_log_dir:
        print("--- Setting up debug logging for likelihood contribution ---")
        idx_to_node_id = {i: node_id for i, node_id in enumerate(graph.nodes())}
        for node_id, node_data in graph.nodes(data=True):
            if 'lat' in node_data and 'lon' in node_data:
                node_coords[node_id] = (node_data['lat'], node_data['lon'])

        lats = [coord[0] for coord in node_coords.values()]
        lons = [coord[1] for coord in node_coords.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        avg_lat = (min_lat + max_lat) / 2
        lat_deg_per_nm = 1.0 / 60.0
        lon_deg_per_nm = 1.0 / (60.0 * np.cos(np.radians(avg_lat)))
        
        cell_size_lat = debug_cell_size_nm * lat_deg_per_nm
        cell_size_lon = debug_cell_size_nm * lon_deg_per_nm
        
        lat_bins = np.arange(min_lat - cell_size_lat, max_lat + 2*cell_size_lat, cell_size_lat)
        lon_bins = np.arange(min_lon - cell_size_lon, max_lon + 2*cell_size_lon, cell_size_lon)
        
        for lat, lon in debug_coordinates:
            lat_idx = np.digitize(lat, lat_bins) - 1
            lon_idx = np.digitize(lon, lon_bins) - 1
            if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
                debug_cells_map[(lat_idx, lon_idx)] = (lat, lon)
        print(f"Tracking {len(debug_cells_map)} debug cells.")

    # Load file paths
    transitions_file, forward_v_file, backward_v_file, cost_file = load_value_function_files(
        base_directory, flight_id
    )
    
    # Load state transitions
    print("Loading state transitions...")
    with open(transitions_file, 'rb') as f:
        state_transitions = pickle.load(f)
    print(f"Loaded {len(state_transitions)} state transitions")
    
    # Determine value function shape from transitions
    num_nodes, num_time_bins, num_rho_bins, num_phases = determine_value_function_shape(
        state_transitions
    )
    v_shape = (num_nodes, num_time_bins, num_rho_bins, num_phases)
    print(f"Value function shape: {v_shape}")
    
    # Load forward and backward value functions
    print("Loading forward value function...")
    V_f = load_dense_v(forward_v_file, v_shape, device)
    
    print("Loading backward value function...")
    V_b = load_dense_v(backward_v_file, v_shape, device)
    
    # Load cost tensor (sparse format) and create a lookup map
    print("Loading cost tensor...")
    cost_map = None
    if os.path.exists(cost_file):
        try:
            sparse_cost, _ = load_sparse_coo_tensor_with_convention(
                cost_file, target_device=device
            )
            sparse_cost = sparse_cost.coalesce()
            print(f"Loaded sparse cost tensor.")

            print("Creating cost map from sparse tensor...")
            indices = sparse_cost.indices().T
            values = sparse_cost.values()
            cost_map = {tuple(index.tolist()): value.item() for index, value in zip(indices, values)}
            print("Cost map created.")
        except Exception as e:
            print(f"Warning: Could not load or process cost file: {e}")
            print("Using zero costs as fallback.")
            cost_map = None
    else:
        print(f"Warning: Cost file not found: {cost_file}")
        print("Using zero costs as fallback")
        cost_map = None
    
    # Step 1: Compute partition function
    print("Computing partition function...")
    log_partition_z = compute_partition_function(V_f, goal_node_idx, gamma)
    if source_node is not None:
        log_partition_z2 = compute_partition_function(V_b, source_node_idx, gamma)
    if torch.isinf(log_partition_z):
        print("Warning: Partition function is infinite. No valid paths found.")
        return torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float64)
    
    print(f"Log partition function: {log_partition_z.item():.6f}")
    if source_node is not None:
        print(f"Backup Log partition function: {log_partition_z2.item():.6f}")
        print(f"Difference: {log_partition_z - log_partition_z2:.6f}")
    
    # Step 2 & 3: Compute state transition probabilities and aggregate by link
    print("Computing link traversal likelihoods...")
    link_traversal_likelihoods = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float64, device=device
    )
    
    # Process each state transition
    with torch.no_grad():  # No gradients needed for this computation
        for i, trans in enumerate(state_transitions):
            # Unpack transition: (u_wp, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp, k_v_idx, rho_v_idx, alt_v, phase_v)
            u_idx, k_u, rho_u, _, phase_u, v_idx, k_v, rho_v, _, phase_v = trans
            
            # Get forward and backward values for this transition
            v_f_u = V_f[u_idx, k_u, rho_u, phase_u]
            v_b_v = V_b[v_idx, k_v, rho_v, phase_v]
            
            # Skip transitions with infinite values
            if torch.isinf(v_f_u) or torch.isinf(v_b_v):
                continue
            
            # Get actual cost from the cost map
            if cost_map is not None:
                cost_key = (u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v)
                cost_uv = cost_map.get(cost_key, float('inf'))
                if cost_uv == float('inf'):
                    raise ValueError(f"Cost is infinite for transition {trans}???")
                
                # Skip transitions with infinite cost (not reachable)
                if not torch.isfinite(torch.tensor(cost_uv)):
                    raise ValueError(f"Cost is not finite for transition {trans}???")
                    continue
            else:
                # Fallback to zero cost if tensor not available
                cost_uv = 0.0
                raise ValueError(f"Cost map is not available for transition {trans}???")
            
            # Compute transition probability using the formula:
            # P(s_u -> s_v) = (1/Z) * exp(-(V_f(s_u) + c(s_u,s_v) + V_b(s_v))/gamma)
            # In log space: log_p = (-V_f(s_u) - c(s_u,s_v) - V_b(s_v) + log_partition_z) / gamma
            log_p_transition = (-v_f_u - cost_uv - v_b_v + log_partition_z) / gamma
            p_transition = torch.exp(log_p_transition)
            
            # Accumulate probability for this link (u_idx, v_idx)
            link_traversal_likelihoods[u_idx, v_idx] += p_transition
            
            # --- Debug logging ---
            if debug_coordinates and debug_log_dir and p_transition > 0:
                u_node = idx_to_node_id.get(u_idx)
                v_node = idx_to_node_id.get(v_idx)

                if u_node in node_coords and v_node in node_coords:
                    u_lat, u_lon = node_coords[u_node]
                    v_lat, v_lon = node_coords[v_node]
                    
                    dlat = np.radians(v_lat - u_lat)
                    dlon = np.radians(v_lon - u_lon)
                    a = (np.sin(dlat/2)**2 + np.cos(np.radians(u_lat)) * np.cos(np.radians(v_lat)) * np.sin(dlon/2)**2)
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance_nm = 3440.065 * c

                    cells_passed = set()
                    if distance_nm > 0:
                        num_samples = max(100, int(distance_nm / debug_cell_size_nm * 2))
                        for s_idx in range(num_samples + 1):
                            t = s_idx / num_samples
                            sample_lat = u_lat + t * (v_lat - u_lat)
                            sample_lon = u_lon + t * (v_lon - u_lon)
                            
                            lat_idx = np.digitize(sample_lat, lat_bins) - 1
                            lon_idx = np.digitize(sample_lon, lon_bins) - 1
                            
                            if 0 <= lat_idx < (len(lat_bins) - 1) and 0 <= lon_idx < (len(lon_bins) - 1):
                                cells_passed.add((lat_idx, lon_idx))
                    else:
                        lat_idx = np.digitize(u_lat, lat_bins) - 1
                        lon_idx = np.digitize(u_lon, lon_bins) - 1
                        if 0 <= lat_idx < (len(lat_bins) - 1) and 0 <= lon_idx < (len(lon_bins) - 1):
                            cells_passed.add((lat_idx, lon_idx))

                    if cells_passed:
                        weight = p_transition.item() / len(cells_passed)
                        for lat_idx, lon_idx in cells_passed:
                            if (lat_idx, lon_idx) in debug_cells_map:
                                u_label = graph.nodes[u_node].get('label', str(u_node))
                                v_label = graph.nodes[v_node].get('label', str(v_node))
                                link_label = f"{u_label} - {v_label}"
                                debug_cell_contributions[(lat_idx, lon_idx)].append(
                                    (link_label, weight)
                                )

            # Progress reporting
            if (i + 1) % 10000 == 0 or i == len(state_transitions) - 1:
                print(f"  Processed {i + 1}/{len(state_transitions)} transitions...")
    
    print("Link likelihood computation completed!")
    print(f"Total likelihood: {link_traversal_likelihoods.sum().item():.6f}")
    print(f"Number of non-zero links: {(link_traversal_likelihoods > 0).sum().item()}")

    # --- Write debug log file ---
    if debug_coordinates and debug_log_dir:
        os.makedirs(debug_log_dir, exist_ok=True)
        log_path = os.path.join(debug_log_dir, f"{flight_id}.log")
        with open(log_path, 'w') as f:
            for cell_indices, contributions in sorted(debug_cell_contributions.items()):
                orig_lat, orig_lon = debug_cells_map[cell_indices]
                total_likelihood = sum(c[1] for c in contributions)
                
                contrib_str_parts = [f"{label} = {likelihood:.4f}" for label, likelihood in contributions]
                contrib_str = ", ".join(contrib_str_parts)

                lat_idx, lon_idx = cell_indices
                num_lon_bins = len(lon_bins) - 1
                cell_id = lat_idx * num_lon_bins + lon_idx
                
                log_line = (f"{orig_lat}, {orig_lon}, cell_id = {cell_id}: "
                            f"{contrib_str}, Total = {total_likelihood:.4f}\n")
                f.write(log_line)
        print(f"Debug log with likelihood contributions saved to {log_path}")
    
    # Optional: Save results to file
    if save:
        sample_dir = os.path.join(base_directory, flight_id)
        output_file = os.path.join(sample_dir, f"{flight_id}_LINK_LIKELIHOOD_WIND.pt")
        torch.save(link_traversal_likelihoods.cpu(), output_file)
        print(f"Saved link likelihood tensor to: {output_file}")
    
    return link_traversal_likelihoods


def render_likelihood_map(
    link_likelihood_tensor: torch.Tensor,
    graph_file_path: str,
    size_nm: float = 10.0,
    save_path: Optional[str] = None,
    smoothen: bool = False,
    route_string: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    flight_id: Optional[str] = None,
    debug_coordinates: Optional[list] = None
) -> None:
    """
    Render a map showing likelihood values discretized into cells by summing up the likelihoods of all links passing through that cell.
    
    This function creates a geographical map where each cell represents a discretized
    area of size_nm x size_nm nautical miles. For each cell, the function computes
    the maximum likelihood per nautical mile of all links passing through that cell.
    
    Args:
        link_likelihood_tensor: 2D tensor of shape (num_nodes, num_nodes) with 
                               link traversal likelihoods
        graph_file_path: Path to the NetworkX graph file (.gml) containing waypoint
                        coordinates
        size_nm: Cell size in nautical miles (default: 10.0)
        save_path: Optional path to save the map as PNG file
        smoothen: Whether to apply Gaussian smoothing to the grid (default: True)
        route_string: Optional route string to overlay on the map (space-separated waypoints)
        origin: Optional origin airport code for labeling
        destination: Optional destination airport code for labeling
        flight_id: Optional flight ID for title
        debug_coordinates: Optional list of (lat, lon) tuples to highlight on the map
    """
    # Load the graph to get waypoint coordinates
    print(f"Loading graph from {graph_file_path}")
    graph = nx.read_gml(graph_file_path)

    idx_to_node_id = {i: node_id for i, node_id in enumerate(graph.nodes())}
    node_id_to_idx = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    # Extract node coordinates
    node_coords = {}
    waypoint_name_to_coords = {}
    for node_id, node_data in graph.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            node_coords[node_id] = (node_data['lat'], node_data['lon'])
            # Also create mapping by waypoint label/name for route string lookup
            waypoint_name = node_data.get('label', str(node_id))
            waypoint_name_to_coords[waypoint_name] = (node_data['lat'], node_data['lon'])
    
    print(f"Loaded {len(node_coords)} waypoints with coordinates")
    
    # Convert tensor to numpy for easier processing
    likelihood_np = link_likelihood_tensor.cpu().numpy()
    
    # Parse route string if provided
    route_waypoints = []
    route_lats, route_lons = [], []
    if route_string:
        route_waypoints = [wp.strip() for wp in route_string.split() if wp.strip()]
        for waypoint in route_waypoints:
            if waypoint in waypoint_name_to_coords:
                lat, lon = waypoint_name_to_coords[waypoint]
                route_lats.append(lat)
                route_lons.append(lon)
            else:
                print(f"Warning: Waypoint '{waypoint}' not found in graph")
    
    # Find bounds of the area
    lats = [coord[0] for coord in node_coords.values()]
    lons = [coord[1] for coord in node_coords.values()]
    
    # Include route coordinates in bounds calculation if available
    if route_lats:
        lats.extend(route_lats)
        lons.extend(route_lons)
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Convert nautical miles to degrees (approximate)
    # 1 nautical mile ≈ 1/60 degree latitude
    # 1 nautical mile longitude ≈ 1/(60 * cos(lat)) degrees
    avg_lat = (min_lat + max_lat) / 2
    lat_deg_per_nm = 1.0 / 60.0
    lon_deg_per_nm = 1.0 / (60.0 * np.cos(np.radians(avg_lat)))
    
    cell_size_lat = size_nm * lat_deg_per_nm
    cell_size_lon = size_nm * lon_deg_per_nm
    
    # Create grid
    lat_bins = np.arange(min_lat - cell_size_lat, max_lat + 2*cell_size_lat, cell_size_lat)
    lon_bins = np.arange(min_lon - cell_size_lon, max_lon + 2*cell_size_lon, cell_size_lon)
    
    # Initialize likelihood grid
    likelihood_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    print("Computing likelihood per cell using summation...")
    
    # Process each link
    for u_idx in range(likelihood_np.shape[0]):
        for v_idx in range(likelihood_np.shape[1]):
            u_node = idx_to_node_id.get(u_idx)
            v_node = idx_to_node_id.get(v_idx)

            if likelihood_np[u_idx, v_idx] > 0 and u_node in node_coords and v_node in node_coords:
                # Get coordinates
                u_lat, u_lon = node_coords[u_node]
                v_lat, v_lon = node_coords[v_node]
                
                # Calculate link distance in nautical miles
                # Using haversine formula approximation
                dlat = np.radians(v_lat - u_lat)
                dlon = np.radians(v_lon - u_lon)
                a = (np.sin(dlat/2)**2 + 
                     np.cos(np.radians(u_lat)) * np.cos(np.radians(v_lat)) * np.sin(dlon/2)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                distance_nm = 3440.065 * c  # Earth radius in nautical miles
                
                link_likelihood = likelihood_np[u_idx, v_idx]
                
                # Determine all unique cells this link passes through
                cells_passed = set()
                # Sample points along the link to find all cells it passes through
                if distance_nm > 0:
                    # The number of samples should be enough to not miss any cells
                    num_samples = max(100, int(distance_nm / size_nm * 2))
                    for i in range(num_samples + 1):
                        t = i / num_samples
                        sample_lat = u_lat + t * (v_lat - u_lat)
                        sample_lon = u_lon + t * (v_lon - u_lon)
                        
                        lat_idx = np.digitize(sample_lat, lat_bins) - 1
                        lon_idx = np.digitize(sample_lon, lon_bins) - 1
                        
                        if 0 <= lat_idx < likelihood_grid.shape[0] and 0 <= lon_idx < likelihood_grid.shape[1]:
                            cells_passed.add((lat_idx, lon_idx))
                else:
                    # Handle self-loops (distance is zero)
                    lat_idx = np.digitize(u_lat, lat_bins) - 1
                    lon_idx = np.digitize(u_lon, lon_bins) - 1
                    if 0 <= lat_idx < likelihood_grid.shape[0] and 0 <= lon_idx < likelihood_grid.shape[1]:
                        cells_passed.add((lat_idx, lon_idx))

                # Distribute the link's likelihood proportionally among the cells it passes through
                if cells_passed:
                    weight = link_likelihood / len(cells_passed)
                    for lat_idx, lon_idx in cells_passed:
                        likelihood_grid[lat_idx, lon_idx] += weight
    
    print(f"Created likelihood grid with shape {likelihood_grid.shape}")
    print(f"Max likelihood value: {likelihood_grid.max():.6f}")
    print(f"Non-zero cells: {(likelihood_grid > 0).sum()}")

    if smoothen:
        likelihood_grid = gaussian_filter(likelihood_grid, sigma=0.1)
        print(f"Smoothed grid. New max likelihood: {likelihood_grid.max():.6f}")
    
    # Exponential scaling for visualization
    exp_likelihood_grid = np.exp(likelihood_grid)
    
    # Use the original likelihood_grid to find non-zero cells for statistics,
    # as exp(0) = 1.
    non_zero_mask = likelihood_grid > 0
    non_zero_values = exp_likelihood_grid[non_zero_mask]

    # Calculate color bounds using percentiles to be robust to outliers
    vmin, vmax = None, None
    if non_zero_values.size > 0:
        # Using percentiles to ignore extreme outliers for better color scale
        vmin = np.percentile(non_zero_values, 5)  # 5th percentile
        vmax = np.percentile(non_zero_values, 95) # 95th percentile
        print(f"Using exponential scaling with percentile-based color range.")
        print(f"Color range (5th-95th percentile): ({vmin:.4f}, {vmax:.4f})")

    # Create the map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    margin = 1.0  # degrees
    ax.set_extent([min_lon - margin, max_lon + margin, 
                   min_lat - margin, max_lat + margin], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Highlight debug cells if provided
    if debug_coordinates:
        print("Highlighting debug cells on map...")
        for lat, lon in debug_coordinates:
            lat_idx = np.digitize(lat, lat_bins) - 1
            lon_idx = np.digitize(lon, lon_bins) - 1

            if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
                # Get cell boundaries
                lon_min, lon_max = lon_bins[lon_idx], lon_bins[lon_idx + 1]
                lat_min, lat_max = lat_bins[lat_idx], lat_bins[lat_idx + 1]
                
                # Draw a rectangle for the cell
                rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                     fill=False, edgecolor='red', linewidth=2, 
                                     transform=ccrs.PlateCarree(), zorder=15)
                ax.add_patch(rect)

    # Create meshgrid for plotting
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Plot likelihood heatmap
    masked_likelihood = np.ma.masked_where(likelihood_grid == 0, exp_likelihood_grid)
    im = ax.pcolormesh(lon_mesh, lat_mesh, masked_likelihood, 
                       cmap='YlGn', alpha=0.7, transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Exp(Likelihood per Cell)', rotation=270, labelpad=15)
    
    # Plot original route if provided (same style as plot_routes.py)
    legend_elements = []
    if route_string and len(route_lats) > 1:
        ax.plot(route_lons, route_lats, 'b:', linewidth=2, 
               transform=ccrs.PlateCarree(), label='Original Route', zorder=4)
        
        # Plot waypoints on the route
        ax.scatter(route_lons, route_lats, c='blue', s=50, marker='o', 
                  transform=ccrs.PlateCarree(), zorder=5)
        
        legend_elements.append(
            plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='Original Route')
        )
    
    # Add origin and destination markers if provided
    if origin and origin in waypoint_name_to_coords:
        origin_coords = waypoint_name_to_coords[origin]
        ax.scatter(origin_coords[1], origin_coords[0], c='green', s=200, 
                  marker='^', transform=ccrs.PlateCarree(), zorder=10, 
                  edgecolors='black', linewidth=2)
        ax.text(origin_coords[1], origin_coords[0] + 0.5, origin, 
               transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                      markersize=10, label='Origin')
        )
    
    if destination and destination in waypoint_name_to_coords:
        dest_coords = waypoint_name_to_coords[destination]
        ax.scatter(dest_coords[1], dest_coords[0], c='red', s=200, 
                  marker='s', transform=ccrs.PlateCarree(), zorder=10,
                  edgecolors='black', linewidth=2)
        ax.text(dest_coords[1], dest_coords[0] + 0.5, destination, 
               transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='Destination')
        )
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Create title
    title_text = f''
    if flight_id:
        title_text = f'Flight {flight_id}\n' + title_text
        if origin and destination:
            title_text = f'{origin} - {destination}\n'
    
    plt.title(title_text)
    
    # Add legend if we have elements
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage and testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute link likelihood and optionally render map")
    parser.add_argument("--base-directory", type=str, 
                       default="ronbun_experiments/runs_LGAV_LFPG/samples",
                       help="Base directory containing sample data")
    parser.add_argument("--flight-id", type=str, 
                       default="45CAB5AFR36HN_1681912194",
                       help="Flight identifier")
    parser.add_argument("--graph-file", type=str,
                       default="data/cases/LGAV_LFPG/graphs/routes.gml",
                       help="Path to the route graph file")
    parser.add_argument("--goal-node", type=str, required=False, default="LFPG",
                       help="ID of the goal waypoint (must be present in the graph)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Temperature parameter for soft operations")
    parser.add_argument("--cell-size-nm", type=float, default=10.0,
                       help="Cell size in nautical miles for map rendering")
    parser.add_argument("--save-map", type=str, default=None,
                       help="Path to save the likelihood map as PNG file")
    parser.add_argument("--load-existing", action="store_true",
                       help="Load existing likelihood tensor instead of computing")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Smooth the likelihood map")
    parser.add_argument("--route-string", type=str, default=None,
                       help="Route string to overlay on the map (space-separated waypoints)")
    parser.add_argument("--origin", type=str, default=None,
                       help="Origin airport code for labeling")
    parser.add_argument("--destination", type=str, default=None,
                       help="Destination airport code for labeling")
    parser.add_argument("--debug-coordinates", type=float, nargs=2, action='append',
                       help="Lat/lon coordinates to debug, specify multiple times for multiple points. E.g. --debug-coordinates 42.22 1.06")
    parser.add_argument("--debug-log-dir", type=str, default="log_contribution",
                       help="Directory to save debug contribution logs.")
    parser.add_argument("--source-node", type=str, default="LGAV",
                       help="Source airport code for double checking the partition function")
    
    args = parser.parse_args()

    print(f"CAUTION: Current temperature is {args.gamma}! A mismatch between the temperature in the config file and the temperature in the command line will lead to incorrect results.")
    print(f"If you are using the default temperature, you can ignore this message.")
    
    # Simple version without cost model
    print("=== Link likelihood computation and mapping ===")
    try:
        if args.load_existing:
            # Try to load existing likelihood tensor
            sample_dir = os.path.join(args.base_directory, args.flight_id)
            likelihood_file = os.path.join(sample_dir, f"{args.flight_id}_LINK_LIKELIHOOD_WIND.pt")
            if os.path.exists(likelihood_file):
                print(f"Loading existing likelihood tensor from: {likelihood_file}")
                likelihood_tensor = torch.load(likelihood_file)
                print(f"Loaded likelihood tensor with shape: {likelihood_tensor.shape}")
            else:
                print(f"Existing likelihood file not found: {likelihood_file}")
                print("Computing new likelihood tensor...")
                likelihood_tensor = compute_link_likelihood(
                    base_directory=args.base_directory,
                    flight_id=args.flight_id,
                    graph_file_path=args.graph_file,
                    goal_node=args.goal_node,
                    source_node=args.source_node,
                    gamma=args.gamma,
                    save=True,
                    debug_coordinates=args.debug_coordinates,
                    debug_cell_size_nm=args.cell_size_nm,
                    debug_log_dir=args.debug_log_dir
                )
        else:
            likelihood_tensor = compute_link_likelihood(
                base_directory=args.base_directory,
                flight_id=args.flight_id,
                graph_file_path=args.graph_file,
                goal_node=args.goal_node,
                source_node=args.source_node,
                gamma=args.gamma,
                save=True,
                debug_coordinates=args.debug_coordinates,
                debug_cell_size_nm=args.cell_size_nm,
                debug_log_dir=args.debug_log_dir
            )
        
        print(f"Successfully computed/loaded link likelihood tensor with shape: {likelihood_tensor.shape}")
        
        # Display top links by likelihood
        print("\nTop 10 links by likelihood:")
        likelihood_np = likelihood_tensor.cpu().numpy()
        flat_indices = likelihood_np.argsort(axis=None)[::-1]
        
        for i, flat_idx in enumerate(flat_indices[:10]):
            if likelihood_np.flat[flat_idx] <= 0:
                break
            u_idx = flat_idx // likelihood_tensor.shape[1]
            v_idx = flat_idx % likelihood_tensor.shape[1]
            likelihood_val = likelihood_np.flat[flat_idx]
            print(f"  {i+1}. Link {u_idx} -> {v_idx}: {likelihood_val:.6f}")
        
        # Render map if requested
        if args.save_map:
            print(f"\n=== Rendering likelihood map ===")
            render_likelihood_map(
                link_likelihood_tensor=likelihood_tensor,
                graph_file_path=args.graph_file,
                size_nm=args.cell_size_nm,
                save_path=args.save_map,
                smoothen=args.smooth,
                route_string=args.route_string,
                origin=args.origin,
                destination=args.destination,
                flight_id=args.flight_id,
                debug_coordinates=args.debug_coordinates
            )
            
    except Exception as e:
        print(f"Error during computation: {e}")
        print("This is expected if the sample data files don't exist in the specified directory.")