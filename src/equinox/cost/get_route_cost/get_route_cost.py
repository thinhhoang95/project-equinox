#!/usr/bin/env python3
"""
Route Cost Calculation using CostRev4Ronbun1 and Backward Closures
================================================================

This module provides functionality to calculate route costs using the cost_rev4_ronbun1 
model with backward closure transitions and wind data integration.
"""

import torch
import numpy as np
import networkx as nx
import pickle
from typing import List, Tuple, Dict, Optional, Any

# Equinox imports
from equinox.cost.cost_rev4_ronbun1 import CostRev4Ronbun1
from equinox.route.get_wind import get_wind
from equinox.wind.wind_model import WindModel


def get_route_cost(
    backward_closures_pkl_path: str,
    cost_model: CostRev4Ronbun1,
    original_route_str: str,
    takeoff_time_unix: int,
    route_graph: nx.Graph,
    delta_t_seconds_wall_clock: int,
    delta_t_seconds_climb: int,
    max_flight_duration_hours: float,
    climb_phase_switch_allowance_climb_time_bins: int,
    wind_model: WindModel,
    distance_matrix: Optional[np.ndarray] = None,
    airspace_charge_matrix: Optional[np.ndarray] = None,
    estimated_landing_time_unix: Optional[int] = None,
    device: Optional[torch.device] = None
) -> float:
    """
    Calculate route cost using backward closures and cost_rev4_ronbun1 model.
    
    Args:
        backward_closures_pkl_path: Path to pickle file containing backward closure transitions
        cost_model: Initialized CostRev4Ronbun1 cost model instance
        original_route_str: Route string like "LGAV NIKOL LFGX OKIPA LFPE LFPG"
        takeoff_time_unix: Takeoff time as unix timestamp (e.g., 1680440202)
        route_graph: NetworkX graph with waypoint nodes containing lat/lon attributes
        delta_t_seconds_wall_clock: Time bin duration for wall clock time (e.g., 600)
        delta_t_seconds_climb: Time bin duration for climb phase
        max_flight_duration_hours: Maximum flight duration in hours
        climb_phase_switch_allowance_climb_time_bins: Allowance for climb phase switching
        wind_model: WindModel instance for wind data retrieval
        distance_matrix: Optional distance matrix, if None will try to extract from cost model usage
        airspace_charge_matrix: Optional airspace charge matrix
        estimated_landing_time_unix: Optional estimated landing time, if None will estimate
        device: Optional torch device, if None will use cost model's device
        
    Returns:
        Total route cost as float
        
    Raises:
        FileNotFoundError: If backward closures pickle file doesn't exist
        ValueError: If route waypoints not found in graph or invalid parameters
        RuntimeError: If no valid transitions found for route segments
    """
    
    # Set device
    if device is None:
        device = cost_model.device
    
    # Parse route waypoints
    waypoints = _parse_route_string(original_route_str)
    if len(waypoints) < 2:
        raise ValueError(f"Route must contain at least 2 waypoints, got: {waypoints}")
    
    # Validate waypoints exist in graph
    _validate_waypoints_in_graph(waypoints, route_graph)
    
    # Load backward closures
    backward_closures = _load_backward_closures(backward_closures_pkl_path)
    
    # Create node-to-index mapping
    node_to_idx = {i: node for i, node in enumerate(route_graph.nodes())}
    idx_to_node = {node: i for i, node in enumerate(route_graph.nodes())}
    
    # Calculate time parameters
    time_params = _calculate_time_parameters(
        takeoff_time_unix, estimated_landing_time_unix, 
        max_flight_duration_hours, delta_t_seconds_wall_clock
    )
    
    # Calculate cost for each route segment
    total_cost = 0.0
    route_segments = []
    
    for i in range(len(waypoints) - 1):
        u_wp = waypoints[i]
        v_wp = waypoints[i + 1]
        
        # Find matching transition in backward closures
        transition = _find_matching_transition(u_wp, v_wp, backward_closures, idx_to_node)
        if transition is None:
            print(f"Warning: No transition found for {u_wp} -> {v_wp}, skipping segment")
            continue
            
        # Extract transition parameters
        u_wp_id, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp_id, k_v_idx, rho_v_idx, alt_v, phase_v = transition[:10]
        
        # Calculate ETA values
        eta_u = _calculate_eta_from_time_bin(k_u_idx, time_params)
        eta_v = _calculate_eta_from_time_bin(k_v_idx, time_params)
        
        # Get coordinates
        coords_src, coords_tgt = _get_waypoint_coordinates(u_wp, v_wp, route_graph, device)
        
        # Get wind (convert m/s to match cost model expectations)
        altitude_tensor = torch.tensor([alt_u], dtype=torch.float32, device=device)
        eta_tensor = torch.tensor([eta_u], dtype=torch.float32, device=device)
        
        tailwind_mps = get_wind(coords_src, coords_tgt, altitude_tensor, eta_tensor, wind_model)
        # CostRev4Ronbun1 expects wind in m/s based on the forward() method
        tailwind_tensor = tailwind_mps.to(device=device, dtype=torch.float32)
        
        # Get edge indices
        u_idx = idx_to_node[u_wp]
        v_idx = idx_to_node[v_wp]
        u_indices = torch.tensor([u_idx], dtype=torch.long, device=device)
        v_indices = torch.tensor([v_idx], dtype=torch.long, device=device)
        
        # Calculate segment cost using cost model
        segment_cost_tensor = cost_model.forward(
            edge_indices=(u_indices, v_indices),
            distance_matrix_d=distance_matrix,
            airspace_charge_matrix_ac=airspace_charge_matrix,
            tailwind_values_w=tailwind_tensor
        )
        
        segment_cost = segment_cost_tensor.item()
        total_cost += segment_cost
        
        route_segments.append({
            'u_wp': u_wp,
            'v_wp': v_wp,
            'eta_u': eta_u,
            'eta_v': eta_v,
            'alt_u': alt_u,
            'alt_v': alt_v,
            'tailwind_mps': tailwind_mps.item(),
            'cost': segment_cost
        })
        
        print(f"Segment {u_wp} -> {v_wp}: ETA_u={eta_u:.0f}s, Alt_u={alt_u:.0f}ft, "
              f"Tailwind={tailwind_mps.item():.2f}m/s, Cost={segment_cost:.6f}")
    
    if not route_segments:
        raise RuntimeError("No valid route segments found for cost calculation")
    
    print(f"Total route cost: {total_cost:.6f}")
    return total_cost


def _parse_route_string(route_str: str) -> List[str]:
    """Parse route string into list of waypoints."""
    return [wp.strip() for wp in route_str.split() if wp.strip()]


def _validate_waypoints_in_graph(waypoints: List[str], graph: nx.Graph) -> None:
    """Validate that all waypoints exist in the graph."""
    missing_waypoints = [wp for wp in waypoints if wp not in graph.nodes()]
    if missing_waypoints:
        raise ValueError(f"Waypoints not found in graph: {missing_waypoints}")


def _load_backward_closures(pkl_path: str) -> List[Tuple]:
    """Load backward closures from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            closures = pickle.load(f)
        return closures
    except FileNotFoundError:
        raise FileNotFoundError(f"Backward closures file not found: {pkl_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load backward closures: {e}")


def _calculate_time_parameters(
    takeoff_time_unix: int,
    estimated_landing_time_unix: Optional[int],
    max_flight_duration_hours: float,
    delta_t_seconds: int
) -> Dict[str, float]:
    """Calculate time-related parameters for ETA computation."""
    
    # Convert unix timestamps to seconds since midnight
    takeoff_ssm = takeoff_time_unix % 86400  # seconds in a day
    
    if estimated_landing_time_unix is None:
        # Estimate landing time as takeoff + max flight duration
        estimated_landing_ssm = takeoff_ssm + (max_flight_duration_hours * 3600)
    else:
        estimated_landing_ssm = estimated_landing_time_unix % 86400
    
    # Calculate time window parameters
    max_time_overall_seconds = estimated_landing_ssm
    min_time_overall_seconds = max_time_overall_seconds - max_flight_duration_hours * 3600
    num_time_bins = int((max_time_overall_seconds - min_time_overall_seconds) / delta_t_seconds) + 1
    
    return {
        'min_time_overall_seconds': min_time_overall_seconds,
        'max_time_overall_seconds': max_time_overall_seconds,
        'num_time_bins': num_time_bins,
        'delta_t_seconds': delta_t_seconds,
        'takeoff_ssm': takeoff_ssm,
        'estimated_landing_ssm': estimated_landing_ssm
    }


def _calculate_eta_from_time_bin(k_idx: int, time_params: Dict[str, float]) -> float:
    """Calculate ETA in seconds since midnight from time bin index."""
    return time_params['min_time_overall_seconds'] + k_idx * time_params['delta_t_seconds']


def _find_matching_transition(
    u_wp: str, 
    v_wp: str, 
    backward_closures: List[Tuple],
    idx_to_node: Dict[str, int]
) -> Optional[Tuple]:
    """
    Find the first transition that matches the waypoint pair.
    
    Backward closures contain transitions like:
    (u_wp_idx, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp_idx, k_v_idx, rho_v_idx, alt_v, phase_v)
    
    We need to match u_wp and v_wp with the waypoint indices.
    """
    
    # Get waypoint indices
    u_wp_idx = idx_to_node.get(u_wp)
    v_wp_idx = idx_to_node.get(v_wp)
    
    if u_wp_idx is None or v_wp_idx is None:
        return None
    
    # Search for matching transition
    for transition in backward_closures:
        if len(transition) >= 10:  # Ensure transition has expected format
            trans_u_idx, k_u_idx, rho_u_idx, alt_u, phase_u, trans_v_idx, k_v_idx, rho_v_idx, alt_v, phase_v = transition[:10]
            
            if trans_u_idx == u_wp_idx and trans_v_idx == v_wp_idx:
                return (u_wp, k_u_idx, rho_u_idx, alt_u, phase_u, v_wp, k_v_idx, rho_v_idx, alt_v, phase_v)
    
    return None


def _get_waypoint_coordinates(
    u_wp: str, 
    v_wp: str, 
    graph: nx.Graph, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get coordinate tensors for waypoints."""
    
    u_data = graph.nodes[u_wp]
    v_data = graph.nodes[v_wp]
    
    coords_src = torch.tensor([[u_data['lat'], u_data['lon']]], dtype=torch.float32, device=device)
    coords_tgt = torch.tensor([[v_data['lat'], v_data['lon']]], dtype=torch.float32, device=device)
    
    return coords_src, coords_tgt


def _round_to_bin_idx(value: float, max_idx: int) -> int:
    """Round value to bin index with bounds checking."""
    bin_idx = int(round(value))
    return max(0, min(bin_idx, max_idx))


# Example usage function
def example_usage():
    """Example of how to use the get_route_cost function."""
    print("Example usage of get_route_cost function:")
    print("1. Load your cost model: cost_model = CostRev4Ronbun1(device=device)")
    print("2. Load your graph: G = nx.read_gml('path/to/graph.gml')")
    print("3. Initialize wind model: wind_model = WindModel(date_str='2023-04-01', data_dir='data/era5')")
    print("4. Call get_route_cost with appropriate parameters")
    print("\nNote: This function requires backward closure data from the TRES backward pass.")


if __name__ == "__main__":
    example_usage()
