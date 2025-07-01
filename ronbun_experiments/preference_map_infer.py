"""
Preference Map Inference Module

This module infers flight preferences by comparing the original route with the lowest cost trajectory.
For each cell that the original route passes through, it finds the closest cell traversed by the 
lowest cost trajectory and computes the preference delta based on likelihood differences.

The resulting preference map shows where the original route is preferred over the optimal path,
indicating potential inefficiencies or external constraints not captured in the cost model.
"""

import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

# Import utility functions from get_link_likelihood
from get_link_likelihood import render_likelihood_map


def load_flight_data(flight_id: str, samples_dir: Path, trajectories_dir: Path, 
                     csv_file: Path) -> Tuple[Optional[torch.Tensor], Optional[List[Tuple[float, List[str]]]], 
                                           Optional[Dict]]:
    """
    Load likelihood tensor, trajectories, and flight metadata for a given flight.
    
    Args:
        flight_id: Unique flight identifier
        samples_dir: Directory containing sample data files
        trajectories_dir: Directory containing trajectory files
        csv_file: Path to CSV file with flight metadata
        
    Returns:
        Tuple of (likelihood_tensor, trajectories, flight_metadata)
    """
    # Load likelihood tensor
    likelihood_file = samples_dir / flight_id / f"{flight_id}_LINK_LIKELIHOOD_WIND.pt"
    likelihood_tensor = None
    
    if likelihood_file.exists():
        try:
            likelihood_tensor = torch.load(likelihood_file)
            print(f"Loaded likelihood tensor for {flight_id}: shape {likelihood_tensor.shape}")
        except Exception as e:
            print(f"Error loading likelihood tensor for {flight_id}: {e}")
            return None, None, None
    else:
        print(f"Likelihood tensor not found for {flight_id}: {likelihood_file}")
        return None, None, None
    
    # Load trajectories
    trajectory_file = trajectories_dir / f"{flight_id}.txt"
    trajectories = []
    
    if trajectory_file.exists():
        try:
            with open(trajectory_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            cost = float(parts[0])
                            waypoints = parts[1].split()
                            trajectories.append((cost, waypoints))
            print(f"Loaded {len(trajectories)} trajectories for {flight_id}")
        except Exception as e:
            print(f"Error loading trajectories for {flight_id}: {e}")
            return None, None, None
    else:
        print(f"Trajectory file not found for {flight_id}: {trajectory_file}")
        return None, None, None
    
    # Load flight metadata
    flight_metadata = None
    try:
        import pandas as pd
        flights_df = pd.read_csv(csv_file)
        flights_df['unique_flight_id'] = flights_df['flight_id'].astype(str) + '_' + flights_df['takeoff_time'].astype(str)
        
        flight_data = flights_df[flights_df['unique_flight_id'] == flight_id]
        if not flight_data.empty:
            flight_metadata = flight_data.iloc[0].to_dict()
            print(f"Loaded flight metadata for {flight_id}")
        else:
            print(f"Flight metadata not found for {flight_id}")
    except Exception as e:
        print(f"Error loading flight metadata for {flight_id}: {e}")
    
    return likelihood_tensor, trajectories, flight_metadata


def find_lowest_cost_trajectory(trajectories: List[Tuple[float, List[str]]]) -> Tuple[float, List[str]]:
    """
    Find the trajectory with the lowest cost.
    
    Args:
        trajectories: List of (cost, waypoints) tuples
        
    Returns:
        Tuple of (best_cost, best_waypoints)
    """
    if not trajectories:
        raise ValueError("No trajectories provided")
    
    best_cost = float('inf')
    best_trajectory = None
    
    for cost, waypoints in trajectories:
        if cost < best_cost:
            best_cost = cost
            best_trajectory = waypoints
    
    return best_cost, best_trajectory


def create_cell_grid(graph_file_path: str, cell_size_nm: float = 10.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create the same cell grid as used in render_likelihood_map.
    
    Args:
        graph_file_path: Path to the graph file
        cell_size_nm: Cell size in nautical miles
        
    Returns:
        Tuple of (lat_bins, lon_bins, waypoint_coords)
    """
    # Load graph
    graph = nx.read_gml(graph_file_path)
    
    # Extract node coordinates
    waypoint_coords = {}
    for node_id, node_data in graph.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            waypoint_coords[node_id] = (node_data['lat'], node_data['lon'])
            # Also create mapping by waypoint label/name
            # waypoint_name = node_data.get('label', str(node_id))
            # waypoint_coords[waypoint_name] = (node_data['lat'], node_data['lon'])
    
    # Find bounds
    lats = [coord[0] for coord in waypoint_coords.values()]
    lons = [coord[1] for coord in waypoint_coords.values()]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Convert nautical miles to degrees
    avg_lat = (min_lat + max_lat) / 2
    lat_deg_per_nm = 1.0 / 60.0
    lon_deg_per_nm = 1.0 / (60.0 * np.cos(np.radians(avg_lat)))
    
    cell_size_lat = cell_size_nm * lat_deg_per_nm
    cell_size_lon = cell_size_nm * lon_deg_per_nm
    
    # Create grid
    lat_bins = np.arange(min_lat - cell_size_lat, max_lat + 2*cell_size_lat, cell_size_lat)
    lon_bins = np.arange(min_lon - cell_size_lon, max_lon + 2*cell_size_lon, cell_size_lon)
    
    return lat_bins, lon_bins, waypoint_coords


def get_route_cells(route_waypoints: List[str], waypoint_coords: Dict, 
                   lat_bins: np.ndarray, lon_bins: np.ndarray, 
                   cell_size_nm: float = 10.0) -> Set[Tuple[int, int]]:
    """
    Get all cells that a route passes through.
    
    Args:
        route_waypoints: List of waypoint names
        waypoint_coords: Dictionary mapping waypoint names to (lat, lon)
        lat_bins: Latitude bin edges
        lon_bins: Longitude bin edges
        cell_size_nm: Cell size in nautical miles
        
    Returns:
        Set of (lat_idx, lon_idx) tuples representing cells
    """
    cells_passed = set()
    
    # Process each leg of the route
    for i in range(len(route_waypoints) - 1):
        u_waypoint = route_waypoints[i]
        v_waypoint = route_waypoints[i + 1]
        
        if u_waypoint not in waypoint_coords or v_waypoint not in waypoint_coords:
            print(f"Warning: Waypoint not found in coordinates: {u_waypoint} or {v_waypoint}")
            continue
            
        u_lat, u_lon = waypoint_coords[u_waypoint]
        v_lat, v_lon = waypoint_coords[v_waypoint]
        
        # Calculate distance in nautical miles
        dlat = np.radians(v_lat - u_lat)
        dlon = np.radians(v_lon - u_lon)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(u_lat)) * np.cos(np.radians(v_lat)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        distance_nm = 3440.065 * c  # Earth radius in nautical miles
        
        # Sample points along the leg to find all cells it passes through
        if distance_nm > 0:
            num_samples = max(100, int(distance_nm / cell_size_nm * 2))
            for s_idx in range(num_samples + 1):
                t = s_idx / num_samples
                sample_lat = u_lat + t * (v_lat - u_lat)
                sample_lon = u_lon + t * (v_lon - u_lon)
                
                lat_idx = np.digitize(sample_lat, lat_bins) - 1
                lon_idx = np.digitize(sample_lon, lon_bins) - 1
                
                if 0 <= lat_idx < (len(lat_bins) - 1) and 0 <= lon_idx < (len(lon_bins) - 1):
                    cells_passed.add((lat_idx, lon_idx))
        else:
            # Handle case where waypoints are at the same location
            lat_idx = np.digitize(u_lat, lat_bins) - 1
            lon_idx = np.digitize(u_lon, lon_bins) - 1
            if 0 <= lat_idx < (len(lat_bins) - 1) and 0 <= lon_idx < (len(lon_bins) - 1):
                cells_passed.add((lat_idx, lon_idx))
    
    return cells_passed


def get_cell_likelihood_from_tensor(likelihood_tensor: torch.Tensor, graph_file_path: str,
                                   lat_bins: np.ndarray, lon_bins: np.ndarray,
                                   cell_size_nm: float = 10.0) -> np.ndarray:
    """
    Convert the link likelihood tensor to cell-based likelihood grid.
    This replicates the logic from render_likelihood_map.
    
    Args:
        likelihood_tensor: 2D tensor of link likelihoods
        graph_file_path: Path to the graph file
        lat_bins: Latitude bin edges
        lon_bins: Longitude bin edges
        cell_size_nm: Cell size in nautical miles
        
    Returns:
        2D numpy array representing likelihood per cell
    """
    # Load graph
    graph = nx.read_gml(graph_file_path)
    idx_to_node_id = {i: node_id for i, node_id in enumerate(graph.nodes())}
    
    # Extract node coordinates
    node_coords = {}
    for node_id, node_data in graph.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            node_coords[node_id] = (node_data['lat'], node_data['lon'])
    
    # Initialize likelihood grid
    likelihood_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    # Convert tensor to numpy
    likelihood_np = likelihood_tensor.cpu().numpy()
    
    # Process each link (replicating render_likelihood_map logic)
    for u_idx in range(likelihood_np.shape[0]):
        for v_idx in range(likelihood_np.shape[1]):
            u_node = idx_to_node_id.get(u_idx)
            v_node = idx_to_node_id.get(v_idx)
            
            if likelihood_np[u_idx, v_idx] > 0 and u_node in node_coords and v_node in node_coords:
                # Get coordinates
                u_lat, u_lon = node_coords[u_node]
                v_lat, v_lon = node_coords[v_node]
                
                # Calculate link distance in nautical miles
                dlat = np.radians(v_lat - u_lat)
                dlon = np.radians(v_lon - u_lon)
                a = (np.sin(dlat/2)**2 + 
                     np.cos(np.radians(u_lat)) * np.cos(np.radians(v_lat)) * np.sin(dlon/2)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                distance_nm = 3440.065 * c  # Earth radius in nautical miles
                
                link_likelihood = likelihood_np[u_idx, v_idx]
                
                # Determine all unique cells this link passes through
                cells_passed = set()
                if distance_nm > 0:
                    num_samples = max(100, int(distance_nm / cell_size_nm * 2))
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
    
    return likelihood_grid


def find_closest_cell(target_cell: Tuple[int, int], candidate_cells: Set[Tuple[int, int]],
                     lat_bins: np.ndarray, lon_bins: np.ndarray) -> Tuple[int, int]:
    """
    Find the closest cell from a set of candidate cells.
    
    Args:
        target_cell: (lat_idx, lon_idx) of the target cell
        candidate_cells: Set of candidate cells
        lat_bins: Latitude bin edges
        lon_bins: Longitude bin edges
        
    Returns:
        Closest cell (lat_idx, lon_idx)
    """
    if not candidate_cells:
        raise ValueError("No candidate cells provided")
    
    if target_cell in candidate_cells:
        return target_cell
    
    # Convert cell indices to coordinates for distance calculation
    target_lat_idx, target_lon_idx = target_cell
    target_lat = (lat_bins[target_lat_idx] + lat_bins[target_lat_idx + 1]) / 2
    target_lon = (lon_bins[target_lon_idx] + lon_bins[target_lon_idx + 1]) / 2
    
    min_distance = float('inf')
    closest_cell = None
    
    for candidate_lat_idx, candidate_lon_idx in candidate_cells:
        candidate_lat = (lat_bins[candidate_lat_idx] + lat_bins[candidate_lat_idx + 1]) / 2
        candidate_lon = (lon_bins[candidate_lon_idx] + lon_bins[candidate_lon_idx + 1]) / 2
        
        # Calculate distance using haversine formula
        dlat = np.radians(candidate_lat - target_lat)
        dlon = np.radians(candidate_lon - target_lon)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(target_lat)) * np.cos(np.radians(candidate_lat)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 3440.065 * c  # Earth radius in nautical miles
        
        if distance < min_distance:
            min_distance = distance
            closest_cell = (candidate_lat_idx, candidate_lon_idx)
    
    return closest_cell


def compute_preference_map(flight_id: str, samples_dir: Path, trajectories_dir: Path, 
                          csv_file: Path, graph_file_path: str, 
                          cell_size_nm: float = 10.0) -> Optional[np.ndarray]:
    """
    Compute the preference map for a given flight.
    
    Args:
        flight_id: Unique flight identifier
        samples_dir: Directory containing sample data files
        trajectories_dir: Directory containing trajectory files
        csv_file: Path to CSV file with flight metadata
        graph_file_path: Path to the graph file
        cell_size_nm: Cell size in nautical miles
        
    Returns:
        2D numpy array representing preference values per cell, or None if computation fails
    """
    print(f"Computing preference map for flight {flight_id}")
    
    # Load flight data
    likelihood_tensor, trajectories, flight_metadata = load_flight_data(
        flight_id, samples_dir, trajectories_dir, csv_file
    )
    
    if likelihood_tensor is None or not trajectories or flight_metadata is None:
        print(f"Failed to load required data for flight {flight_id}")
        return None
    
    # Find lowest cost trajectory
    best_cost, best_trajectory = find_lowest_cost_trajectory(trajectories)
    print(f"Lowest cost trajectory: {best_cost:.6f} with {len(best_trajectory)} waypoints")
    
    # Parse original route
    original_route = [wp.strip() for wp in flight_metadata['route'].split() if wp.strip()]
    print(f"Original route: {len(original_route)} waypoints")
    
    # Create cell grid
    lat_bins, lon_bins, waypoint_coords = create_cell_grid(graph_file_path, cell_size_nm)
    print(f"Created cell grid: {len(lat_bins)-1} x {len(lon_bins)-1} cells")
    
    # Get likelihood grid from tensor
    likelihood_grid = get_cell_likelihood_from_tensor(
        likelihood_tensor, graph_file_path, lat_bins, lon_bins, cell_size_nm
    )
    print(f"Generated likelihood grid with {(likelihood_grid > 0).sum()} non-zero cells")
    
    # Get cells passed by original route and best trajectory
    original_cells = get_route_cells(original_route, waypoint_coords, lat_bins, lon_bins, cell_size_nm)
    best_cells = get_route_cells(best_trajectory, waypoint_coords, lat_bins, lon_bins, cell_size_nm)
    
    print(f"Original route passes through {len(original_cells)} cells")
    print(f"Best trajectory passes through {len(best_cells)} cells")
    
    # Initialize preference map
    preference_map = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    # Compute preference for each cell in the original route
    for original_cell in original_cells:
        lat_idx, lon_idx = original_cell
        
        # Get likelihood of the original cell
        original_likelihood = likelihood_grid[lat_idx, lon_idx]
        
        # Find closest cell in the best trajectory
        if best_cells:
            closest_best_cell = find_closest_cell(original_cell, best_cells, lat_bins, lon_bins)
            closest_lat_idx, closest_lon_idx = closest_best_cell
            
            # Get likelihood of the closest best cell
            best_likelihood = likelihood_grid[closest_lat_idx, closest_lon_idx]
            
            # Calculate preference delta
            delta = best_likelihood - original_likelihood
            
            # Add to preference map
            preference_map[lat_idx, lon_idx] += delta
            
            # Debug output for extreme values
            if abs(delta) > 0.1:  # Threshold for significant preference difference
                print(f"  Cell ({lat_idx}, {lon_idx}): orig={original_likelihood:.6f}, "
                      f"best={best_likelihood:.6f}, delta={delta:.6f}")
    
    print(f"Preference map computed with {(preference_map != 0).sum()} non-zero cells")
    print(f"Preference range: [{preference_map.min():.6f}, {preference_map.max():.6f}]")
    
    return preference_map


def save_preference_map(preference_map: np.ndarray, flight_id: str, 
                       samples_dir: Path) -> Path:
    """
    Save the preference map as a tensor file.
    
    Args:
        preference_map: 2D numpy array of preference values
        flight_id: Unique flight identifier
        samples_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Convert to tensor and save
    preference_tensor = torch.from_numpy(preference_map).float()
    
    # Create flight-specific directory if it doesn't exist
    flight_dir = samples_dir / flight_id
    flight_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    output_file = flight_dir / f"{flight_id}_PREFERENCE_MAP.pt"
    torch.save(preference_tensor, output_file)
    
    print(f"Saved preference map to: {output_file}")
    return output_file


def render_preference_map(preference_map: np.ndarray, lat_bins: np.ndarray, lon_bins: np.ndarray,
                         graph_file_path: str, flight_id: str, flight_metadata: Dict,
                         save_path: Optional[str] = None, smoothen: bool = False) -> None:
    """
    Render the preference map as a visual plot.
    
    Args:
        preference_map: 2D numpy array of preference values
        lat_bins: Latitude bin edges
        lon_bins: Longitude bin edges
        graph_file_path: Path to the graph file
        flight_id: Unique flight identifier
        flight_metadata: Flight metadata dictionary
        save_path: Optional path to save the plot
        smoothen: Whether to apply Gaussian smoothing
    """
    # Apply smoothing if requested
    if smoothen:
        preference_map = gaussian_filter(preference_map, sigma=1.0)
        print("Applied Gaussian smoothing to preference map")
    
    # Load graph for waypoint coordinates
    graph = nx.read_gml(graph_file_path)
    waypoint_coords = {}
    for node_id, node_data in graph.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            waypoint_name = node_data.get('label', str(node_id))
            waypoint_coords[waypoint_name] = (node_data['lat'], node_data['lon'])
    
    # Parse original route
    original_route = [wp.strip() for wp in flight_metadata['route'].split() if wp.strip()]
    route_lats, route_lons = [], []
    for waypoint in original_route:
        if waypoint in waypoint_coords:
            lat, lon = waypoint_coords[waypoint]
            route_lats.append(lat)
            route_lons.append(lon)
    
    # Create the map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    min_lat = lat_bins.min()
    max_lat = lat_bins.max()
    min_lon = lon_bins.min()
    max_lon = lon_bins.max()
    margin = 0.5  # degrees
    ax.set_extent([min_lon - margin, max_lon + margin, 
                   min_lat - margin, max_lat + margin], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Create meshgrid for plotting
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Plot preference heatmap
    # Use a diverging colormap where positive values are red (preferred) and negative are blue (avoided)
    non_zero_mask = preference_map != 0
    if non_zero_mask.sum() > 0:
        # Calculate symmetric color limits based on the 5th and 95th percentiles to prevent outliers
        # from hiding details in the map.
        non_zero_vals = preference_map[non_zero_mask]
        p5, p95 = np.percentile(non_zero_vals, [5, 95])
        max_abs = max(abs(p5), abs(p95))
        vmin, vmax = -max_abs, max_abs
        
        masked_preference = np.ma.masked_where(preference_map == 0, preference_map)
        im = ax.pcolormesh(lon_mesh, lat_mesh, masked_preference, 
                          cmap='RdBu_r', alpha=0.8, transform=ccrs.PlateCarree(),
                          vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Preference Delta\n(Positive = Preferred)', rotation=270, labelpad=15)
    
    # Plot original route
    legend_elements = []
    if len(route_lats) > 1:
        ax.plot(route_lons, route_lats, 'b:', linewidth=2, 
               transform=ccrs.PlateCarree(), label='Original Route', zorder=4)
        
        # Plot waypoints
        ax.scatter(route_lons, route_lats, c='blue', s=50, marker='o', 
                  transform=ccrs.PlateCarree(), zorder=5)
        
        legend_elements.append(
            plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='Original Route')
        )
    
    # Add origin and destination markers
    origin = flight_metadata.get('origin')
    destination = flight_metadata.get('destination')
    
    if origin and origin in waypoint_coords:
        origin_coords = waypoint_coords[origin]
        ax.scatter(origin_coords[1], origin_coords[0], c='green', s=200, 
                  marker='^', transform=ccrs.PlateCarree(), zorder=10, 
                  edgecolors='black', linewidth=2)
        ax.text(origin_coords[1], origin_coords[0] + 0.3, origin, 
               transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                      markersize=10, label='Origin')
        )
    
    if destination and destination in waypoint_coords:
        dest_coords = waypoint_coords[destination]
        ax.scatter(dest_coords[1], dest_coords[0], c='red', s=200, 
                  marker='s', transform=ccrs.PlateCarree(), zorder=10,
                  edgecolors='black', linewidth=2)
        ax.text(dest_coords[1], dest_coords[0] + 0.3, destination, 
               transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='Destination')
        )
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Create title
    title_text = f'Preference Map: {flight_id}\n{origin} - {destination}'
    plt.title(title_text, pad=20)
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Preference map saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def render_aggregate_preference_map(preference_map: np.ndarray, lat_bins: np.ndarray, lon_bins: np.ndarray,
                                   case_name: str, save_path: Optional[str] = None, 
                                   smoothen: bool = False) -> None:
    """
    Render the aggregate preference map as a visual plot.
    
    Args:
        preference_map: 2D numpy array of aggregate preference values
        lat_bins: Latitude bin edges
        lon_bins: Longitude bin edges
        case_name: The case name for the title
        save_path: Optional path to save the plot
        smoothen: Whether to apply Gaussian smoothing
    """
    # Apply smoothing if requested
    if smoothen:
        preference_map = gaussian_filter(preference_map, sigma=1.0)
        print("Applied Gaussian smoothing to aggregate preference map")
    
    # Create the map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    min_lat = lat_bins.min()
    max_lat = lat_bins.max()
    min_lon = lon_bins.min()
    max_lon = lon_bins.max()
    margin = 0.5  # degrees
    ax.set_extent([min_lon - margin, max_lon + margin, 
                   min_lat - margin, max_lat + margin], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Create meshgrid for plotting
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Plot preference heatmap
    non_zero_mask = preference_map != 0
    if non_zero_mask.sum() > 0:
        # Calculate symmetric color limits based on the 5th and 95th percentiles to prevent outliers
        # from hiding details in the map.
        non_zero_vals = preference_map[non_zero_mask]
        p5, p95 = np.percentile(non_zero_vals, [5, 95])
        max_abs = max(abs(p5), abs(p95))
        vmin, vmax = -max_abs, max_abs
        
        masked_preference = np.ma.masked_where(preference_map == 0, preference_map)
        im = ax.pcolormesh(lon_mesh, lat_mesh, masked_preference, 
                          cmap='RdBu_r', alpha=0.8, transform=ccrs.PlateCarree(),
                          vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Normalized Preference Delta\n(Positive = Preferred)', rotation=270, labelpad=15)
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Create title
    title_text = f'Aggregate Preference Map: {case_name}'
    plt.title(title_text, pad=20)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aggregate preference map saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def process_single_flight(flight_id: str, samples_dir: Path, trajectories_dir: Path, 
                         csv_file: Path, graph_file_path: str, plots_dir: Path,
                         cell_size_nm: float = 10.0, smoothen: bool = False) -> bool:
    """
    Process a single flight to compute and save its preference map.
    
    Args:
        flight_id: Unique flight identifier
        samples_dir: Directory containing sample data files
        trajectories_dir: Directory containing trajectory files
        csv_file: Path to CSV file with flight metadata
        graph_file_path: Path to the graph file
        plots_dir: Directory to save plots
        cell_size_nm: Cell size in nautical miles
        smoothen: Whether to apply smoothing to the preference map
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n=== Processing flight {flight_id} ===")
        
        # Load flight metadata
        import pandas as pd
        flights_df = pd.read_csv(csv_file)
        flights_df['unique_flight_id'] = flights_df['flight_id'].astype(str) + '_' + flights_df['takeoff_time'].astype(str)
        
        flight_data = flights_df[flights_df['unique_flight_id'] == flight_id]
        if flight_data.empty:
            print(f"Flight metadata not found for {flight_id}")
            return False
        
        flight_metadata = flight_data.iloc[0].to_dict()
        
        # Compute preference map
        preference_map = compute_preference_map(
            flight_id, samples_dir, trajectories_dir, csv_file, 
            graph_file_path, cell_size_nm
        )
        
        if preference_map is None:
            print(f"Failed to compute preference map for {flight_id}")
            return False
        
        # Save preference map
        save_preference_map(preference_map, flight_id, samples_dir)
        
        # Create cell grid for visualization
        lat_bins, lon_bins, _ = create_cell_grid(graph_file_path, cell_size_nm)
        
        # Render and save preference map visualization
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = f"{flight_id}_preference_map.png"
        save_path = plots_dir / plot_filename
        
        render_preference_map(
            preference_map, lat_bins, lon_bins, graph_file_path, 
            flight_id, flight_metadata, str(save_path), smoothen
        )
        
        print(f"Successfully processed flight {flight_id}")
        return True
        
    except Exception as e:
        print(f"Error processing flight {flight_id}: {e}")
        return False


def batch_process_preference_maps(case_name: str = "LGAV_LFPG", 
                                 cell_size_nm: float = 10.0, 
                                 max_flights: Optional[int] = None,
                                 smoothen: bool = False) -> Tuple[int, int]:
    """
    Batch process preference maps for all flights in a case.
    
    Args:
        case_name: Case name to process
        cell_size_nm: Cell size in nautical miles
        max_flights: Maximum number of flights to process
        smoothen: Whether to apply smoothing to preference maps
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "cases" / case_name
    results_dir = project_root / "ronbun_experiments" / f"runs_{case_name}"
    
    csv_file = data_dir / "all_routes_sculpted.csv"
    graph_file_path = str(data_dir / "graphs" / "routes.gml")
    trajectories_dir = results_dir / "trajectories"
    samples_dir = results_dir / "samples"
    plots_dir = results_dir / "plots"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of flights with trajectory data
    trajectory_files = list(trajectories_dir.glob("*.txt"))
    flight_ids = [f.stem for f in trajectory_files]
    
    if max_flights:
        flight_ids = flight_ids[:max_flights]
    
    total_flights = len(flight_ids)
    if total_flights == 0:
        print(f"No trajectory files found in {trajectories_dir}")
        return 0, 0
        
    successful_count = 0
    failed_count = 0
    
    print(f"Processing preference maps for {total_flights} flights in case {case_name}")
    print(f"Cell size: {cell_size_nm} nm")
    print(f"Smoothing: {smoothen}")
    
    # Create cell grid and initialize aggregate maps
    lat_bins, lon_bins, _ = create_cell_grid(graph_file_path, cell_size_nm)
    aggregate_preference_map = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    contribution_count_map = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    for i, flight_id in enumerate(flight_ids, 1):
        print(f"\nProcessing flight {i}/{total_flights}: {flight_id}")
        
        preference_map = compute_preference_map(
            flight_id, samples_dir, trajectories_dir, csv_file, 
            graph_file_path, cell_size_nm
        )
        
        if preference_map is not None:
            # Add to aggregate maps
            aggregate_preference_map += preference_map
            contribution_count_map[preference_map != 0] += 1
            successful_count += 1
        else:
            print(f"Failed to compute preference map for {flight_id}, skipping.")
            failed_count += 1
        
        # Progress update
        if i % 5 == 0:
            print(f"Progress: {i}/{total_flights} flights processed "
                  f"({successful_count} successful, {failed_count} failed)")
    
    print(f"\n=== Batch processing completed ===")
    print(f"Total flights processed: {total_flights}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if successful_count > 0:
        # Normalize the aggregate map
        # To avoid division by zero, set count to 1 where it's 0.
        # This won't affect the result as the preference is 0 there anyway.
        contribution_count_map[contribution_count_map == 0] = 1
        normalized_preference_map = aggregate_preference_map / contribution_count_map
        
        # Save the aggregate preference map
        agg_map_tensor = torch.from_numpy(normalized_preference_map).float()
        agg_map_file = results_dir / f"aggregate_preference_map_{case_name}.pt"
        torch.save(agg_map_tensor, agg_map_file)
        print(f"Saved aggregate preference map to: {agg_map_file}")
        
        # Render and save aggregate map visualization
        plot_filename = f"aggregate_preference_map_{case_name}.png"
        save_path = plots_dir / plot_filename
        
        render_aggregate_preference_map(
            normalized_preference_map, lat_bins, lon_bins, 
            case_name, str(save_path), smoothen
        )
    
    return successful_count, failed_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute preference maps for flights")
    parser.add_argument("--case", default="LGAV_LFPG", help="Case name to process")
    parser.add_argument("--flight", help="Specific flight ID to process")
    parser.add_argument("--cell-size-nm", type=float, default=10.0, 
                       help="Cell size in nautical miles")
    parser.add_argument("--max-flights", type=int, 
                       help="Maximum number of flights to process")
    parser.add_argument("--test-10-flights", action="store_true",
                       help="Process only the first 10 flights for testing.")
    parser.add_argument("--smooth", action="store_true", 
                       help="Apply Gaussian smoothing to preference maps")
    
    args = parser.parse_args()

    if args.test_10_flights:
        args.max_flights = 10
    
    if args.flight:
        # Process single flight
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "cases" / args.case
        results_dir = project_root / "ronbun_experiments" / f"runs_{args.case}"
        
        csv_file = data_dir / "all_routes_sculpted.csv"
        graph_file_path = str(data_dir / "graphs" / "routes.gml")
        trajectories_dir = results_dir / "trajectories"
        samples_dir = results_dir / "samples"
        plots_dir = results_dir / "plots"
        
        success = process_single_flight(
            args.flight, samples_dir, trajectories_dir, csv_file, 
            graph_file_path, plots_dir, args.cell_size_nm, args.smooth
        )
        
        if not success:
            print(f"Failed to process flight {args.flight}")
            exit(1)
    else:
        # Batch process all flights
        successful, failed = batch_process_preference_maps(
            args.case, args.cell_size_nm, args.max_flights, args.smooth
        )
        
        if failed > 0:
            print(f"Warning: {failed} flights failed to process")
            exit(1)