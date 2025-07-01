"""
Node Likelihood Computation Module

This module implements the computation of node traversal likelihood based on 
forward and backward soft value functions obtained from soft Bellman message passing.

The module computes the expected probability of traversing each node (waypoint)
by aggregating the probabilities of all states within that node.
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


def compute_node_likelihood(
    base_directory: str,
    flight_id: str,
    graph_file_path: str,
    goal_node: str,
    source_node: Optional[str] = None,
    gamma: float = 1.0,
    device: Optional[torch.device] = None,
    save: bool = True,
) -> torch.Tensor:
    """
    Compute node traversal likelihood from forward and backward value functions.
    
    This function implements the node likelihood computation, which calculates 
    the probability of passing through each waypoint (node).
    
    The computation follows these steps:
    1. Compute the partition function Z from forward values at the goal node.
    2. Compute state probabilities using the formula:
       P(s_i) = (1/Z) * exp(-(V_f(s_i) + V_b(s_i)) / gamma)
    3. Aggregate probabilities by node: P(u) = sum over all states s_i in node u of P(s_i)
    
    Args:
        base_directory: Base directory containing sample data
        flight_id: Flight identifier
        graph_file_path: Path to the NetworkX graph file (.gml)
        goal_node: ID of the goal waypoint.
        source_node: ID of the source waypoint (optional, for validation).
        gamma: Temperature parameter for soft operations (default: 1.0)
        device: PyTorch device for computations (default: auto-detect)
        save: Whether to save the likelihood tensor to file (default: True)
        
    Returns:
        1D tensor of shape (num_nodes,) with element i indicating the
        likelihood for waypoint i.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Computing node likelihood for flight {flight_id}")
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

    # Load file paths
    transitions_file, forward_v_file, backward_v_file, _ = load_value_function_files(
        base_directory, flight_id
    )

    # Load state transitions to determine shape
    print("Loading state transitions to determine shape...")
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

    # Step 1: Compute partition function
    print("Computing partition function...")
    log_partition_z = compute_partition_function(V_f, goal_node_idx, gamma)
    if source_node is not None:
        log_partition_z2 = compute_partition_function(V_b, source_node_idx, gamma)

    if torch.isinf(log_partition_z):
        print("Warning: Partition function is infinite. No valid paths found.")
        return torch.zeros(num_nodes, device=device, dtype=torch.float64)

    print(f"Log partition function: {log_partition_z.item():.6f}")
    if source_node is not None:
        print(f"Backup Log partition function: {log_partition_z2.item():.6f}")
        print(f"Difference: {log_partition_z - log_partition_z2:.6f}")

    # Step 2 & 3: Compute state probabilities and aggregate by node
    print("Computing node traversal likelihoods...")
    node_likelihoods = torch.zeros(num_nodes, dtype=torch.float64, device=device)

    with torch.no_grad():
        for u_idx in range(num_nodes):
            for k in range(num_time_bins):
                for rho in range(num_rho_bins):
                    for phase in range(num_phases):
                        v_f_u = V_f[u_idx, k, rho, phase]
                        v_b_u = V_b[u_idx, k, rho, phase]

                        if torch.isinf(v_f_u) or torch.isinf(v_b_u):
                            continue
                        
                        # Compute state probability using the formula:
                        # P(s_i) = exp(-(V_f(s_i) + V_b(s_i) - log_Z) / gamma)
                        log_p_state = (-v_f_u - v_b_u + log_partition_z) / gamma
                        p_state = torch.exp(log_p_state)
                        
                        # Accumulate probability for this node
                        node_likelihoods[u_idx] += p_state
            
            if (u_idx + 1) % 100 == 0 or u_idx == num_nodes - 1:
                print(f"  Processed {u_idx + 1}/{num_nodes} nodes...")

    print("Node likelihood computation completed!")
    print(f"Total likelihood summed over all nodes: {node_likelihoods.sum().item():.6f}")
    print(f"Number of non-zero nodes: {(node_likelihoods > 0).sum().item()}")

    # Optional: Save results to file
    if save:
        sample_dir = os.path.join(base_directory, flight_id)
        output_file = os.path.join(sample_dir, f"{flight_id}_NODE_LIKELIHOOD_WIND.pt")
        torch.save(node_likelihoods.cpu(), output_file)
        print(f"Saved node likelihood tensor to: {output_file}")

    return node_likelihoods


def render_node_likelihood_map(
    node_likelihood_tensor: torch.Tensor,
    graph_file_path: str,
    save_path: Optional[str] = None,
    route_string: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    flight_id: Optional[str] = None,
) -> None:
    """
    Render a map showing node likelihoods by coloring the nodes in the graph.
    
    Args:
        node_likelihood_tensor: 1D tensor of shape (num_nodes,) with node likelihoods.
        graph_file_path: Path to the NetworkX graph file (.gml) containing waypoint coordinates.
        save_path: Optional path to save the map as a PNG file.
        route_string: Optional route string to overlay on the map (space-separated waypoints).
        origin: Optional origin airport code for labeling.
        destination: Optional destination airport code for labeling.
        flight_id: Optional flight ID for title.
    """
    # Load the graph to get waypoint coordinates
    print(f"Loading graph from {graph_file_path}")
    graph = nx.read_gml(graph_file_path)

    node_id_to_idx = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    # Extract node coordinates
    node_coords = {}
    waypoint_name_to_coords = {}
    for node_id, node_data in graph.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            node_coords[node_id] = (node_data['lat'], node_data['lon'])
            waypoint_name = node_data.get('label', str(node_id))
            waypoint_name_to_coords[waypoint_name] = (node_data['lat'], node_data['lon'])
    
    print(f"Loaded {len(node_coords)} waypoints with coordinates")
    
    # Convert tensor to numpy for easier processing
    likelihood_np = node_likelihood_tensor.cpu().numpy()
    
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
    all_lats = [coord[0] for coord in node_coords.values()]
    all_lons = [coord[1] for coord in node_coords.values()]
    if route_lats:
        all_lats.extend(route_lats)
        all_lons.extend(route_lons)
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Prepare data for scatter plot
    plot_lats, plot_lons, plot_likelihoods = [], [], []
    for node_id, coords in node_coords.items():
        if node_id in node_id_to_idx:
            idx = node_id_to_idx[node_id]
            plot_lats.append(coords[0])
            plot_lons.append(coords[1])
            plot_likelihoods.append(likelihood_np[idx])

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
    
    # Plot node likelihoods as a scatter plot
    sc = ax.scatter(plot_lons, plot_lats, c=plot_likelihoods, cmap='viridis', s=50,
                    transform=ccrs.PlateCarree(), zorder=10, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Node Likelihood', rotation=270, labelpad=15)
    
    # Plot original route if provided
    legend_elements = []
    if route_string and len(route_lats) > 1:
        ax.plot(route_lons, route_lats, 'r:', linewidth=2, 
               transform=ccrs.PlateCarree(), label='Original Route', zorder=4)
        
        ax.scatter(route_lons, route_lats, c='red', s=30, marker='o', 
                  transform=ccrs.PlateCarree(), zorder=5, edgecolors='black')
        
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Original Route')
        )
    
    # Add origin and destination markers
    if origin and origin in waypoint_name_to_coords:
        origin_coords = waypoint_name_to_coords[origin]
        ax.scatter(origin_coords[1], origin_coords[0], c='green', s=200, 
                  marker='^', transform=ccrs.PlateCarree(), zorder=12, 
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
                  marker='s', transform=ccrs.PlateCarree(), zorder=12,
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
    title_text = "Node Likelihood Map"
    if origin and destination:
        title_text = f'{origin} - {destination}\n' + title_text
    if flight_id:
        title_text = f'Flight {flight_id}\n' + title_text
    
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
    parser = argparse.ArgumentParser(description="Compute node likelihood and optionally render map")
    parser.add_argument("--base-directory", type=str, 
                       default="ronbun_experiments/runs_LGAV_LFPG/samples",
                       help="Base directory containing sample data")
    parser.add_argument("--flight-id", type=str, 
                       default="45CAB5AFR53PR_1682306604",
                       help="Flight identifier")
    parser.add_argument("--graph-file", type=str,
                       default="data/cases/LGAV_LFPG/graphs/routes.gml",
                       help="Path to the route graph file")
    parser.add_argument("--goal-node", type=str, required=False, default="LFPG",
                       help="ID of the goal waypoint (must be present in the graph)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Temperature parameter for soft operations")
    parser.add_argument("--save-map", type=str, default=None,
                       help="Path to save the likelihood map as PNG file")
    parser.add_argument("--load-existing", action="store_true",
                       help="Load existing likelihood tensor instead of computing")
    parser.add_argument("--route-string", type=str, default=None,
                       help="Route string to overlay on the map (space-separated waypoints)")
    parser.add_argument("--origin", type=str, default=None,
                       help="Origin airport code for labeling")
    parser.add_argument("--destination", type=str, default=None,
                       help="Destination airport code for labeling")
    parser.add_argument("--source-node", type=str, default="LGAV",
                       help="Source airport code for double checking the partition function")
    
    args = parser.parse_args()

    print(f"CAUTION: Current temperature is {args.gamma}! A mismatch between the temperature in the config file and the temperature in the command line will lead to incorrect results.")
    print(f"If you are using the default temperature, you can ignore this message.")
    
    print("=== Node likelihood computation and mapping ===")
    try:
        if args.load_existing:
            # Try to load existing likelihood tensor
            sample_dir = os.path.join(args.base_directory, args.flight_id)
            likelihood_file = os.path.join(sample_dir, f"{args.flight_id}_NODE_LIKELIHOOD_WIND.pt")
            if os.path.exists(likelihood_file):
                print(f"Loading existing likelihood tensor from: {likelihood_file}")
                likelihood_tensor = torch.load(likelihood_file)
                print(f"Loaded likelihood tensor with shape: {likelihood_tensor.shape}")
            else:
                print(f"Existing likelihood file not found: {likelihood_file}")
                print("Computing new likelihood tensor...")
                likelihood_tensor = compute_node_likelihood(
                    base_directory=args.base_directory,
                    flight_id=args.flight_id,
                    graph_file_path=args.graph_file,
                    goal_node=args.goal_node,
                    source_node=args.source_node,
                    gamma=args.gamma,
                    save=True
                )
        else:
            likelihood_tensor = compute_node_likelihood(
                base_directory=args.base_directory,
                flight_id=args.flight_id,
                graph_file_path=args.graph_file,
                goal_node=args.goal_node,
                source_node=args.source_node,
                gamma=args.gamma,
                save=True
            )
        
        print(f"Successfully computed/loaded node likelihood tensor with shape: {likelihood_tensor.shape}")
        
        # Display top nodes by likelihood
        print("\nTop 10 nodes by likelihood:")
        likelihood_np = likelihood_tensor.cpu().numpy()
        sorted_indices = likelihood_np.argsort()[::-1]
        
        graph = nx.read_gml(args.graph_file)
        idx_to_node_id = {i: node_id for i, node_id in enumerate(graph.nodes())}

        for i, node_idx in enumerate(sorted_indices[:10]):
            if likelihood_np[node_idx] <= 1e-9: # Threshold for meaningful likelihood
                break
            node_id = idx_to_node_id.get(node_idx, f"Unknown_idx_{node_idx}")
            likelihood_val = likelihood_np[node_idx]
            node_label = graph.nodes[node_id].get('label', node_id)
            print(f"  {i+1}. Node {node_label} (idx {node_idx}): {likelihood_val:.6f}")
        
        # Render map if requested
        if args.save_map:
            print(f"\n=== Rendering likelihood map ===")
            render_node_likelihood_map(
                node_likelihood_tensor=likelihood_tensor,
                graph_file_path=args.graph_file,
                save_path=args.save_map,
                route_string=args.route_string,
                origin=args.origin,
                destination=args.destination,
                flight_id=args.flight_id,
            )
            
    except Exception as e:
        print(f"Error during computation: {e}")
        import traceback
        traceback.print_exc()
        print("This is expected if the sample data files don't exist in the specified directory.")