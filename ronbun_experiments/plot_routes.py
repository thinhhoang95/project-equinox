#!/usr/bin/env python3
"""
Route Visualization Script
=========================

This script visualizes original flight routes and sampled trajectories using cartopy.
It plots:
- Original routes from all_routes_sculpted.csv in red
- Sampled trajectories from trajectory files in various colors
- Waypoints and airports on a geographic map

Features:
- Interactive preview mode
- Batch PNG export mode
- Automatic map bounds calculation
- Customizable styling and colors
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
import argparse

# Try importing cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    CARTOPY_AVAILABLE = True
except ImportError:
    print("Warning: cartopy not available. Install with: pip install cartopy")
    CARTOPY_AVAILABLE = False

# Global configuration
CASE_NAME = "LGAV_LFPG"
PROJECT_ROOT = Path(__file__).parent.parent

# Try importing equinox modules for cost calculation
try:
    # Add project root to Python path for imports
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    
    import torch
    from equinox.cost.cost_rev4_ronbun1 import CostRev4Ronbun1
    from equinox.cost.get_route_cost.get_route_cost import get_route_cost
    from equinox.wind.wind_model import WindModel
    EQUINOX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Equinox modules not available for cost calculation: {e}")
    EQUINOX_AVAILABLE = False
DATA_DIR = PROJECT_ROOT / "data" / "cases" / CASE_NAME
RESULTS_DIR = PROJECT_ROOT / "ronbun_experiments" / f"runs_{CASE_NAME}"

class RouteVisualizer:
    """Handles visualization of flight routes and trajectories."""
    
    def __init__(self, case_name: str = CASE_NAME):
        self.case_name = case_name
        self.data_dir = PROJECT_ROOT / "data" / "cases" / case_name
        self.results_dir = PROJECT_ROOT / "ronbun_experiments" / f"runs_{case_name}"
        
        # Data files
        self.csv_file = self.data_dir / "all_routes_sculpted.csv"
        self.graph_file = self.data_dir / "graphs" / "routes.gml"
        self.distance_matrix_file = self.data_dir / "graphs" / "routes_distances.npy"
        self.charges_matrix_file = self.data_dir / "graphs" / "routes_charges.npy"
        self.trajectories_dir = self.results_dir / "trajectories"
        self.plots_dir = self.results_dir / "plots"
        self.samples_dir = self.results_dir / "samples"
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.waypoint_coords = {}
        self.flights_df = None
        self.route_graph = None
        self.distance_matrix = None
        self.charges_matrix = None
        self.cost_model = None
        self.wind_model = None
        
        self.load_graph_data()
        self.load_flights_data()
        self.initialize_cost_calculation()
        
        # Default: don't calculate original costs unless explicitly requested
        self._calculate_original_costs = False
        
    def load_graph_data(self):
        """Load waypoint coordinates from the graph file."""
        if not self.graph_file.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_file}")
            
        try:
            self.route_graph = nx.read_gml(str(self.graph_file))
            
            for node_id, data in self.route_graph.nodes(data=True):
                if 'lat' in data and 'lon' in data:
                    waypoint_name = data.get('label', str(node_id))
                    self.waypoint_coords[waypoint_name] = {
                        'lat': float(data['lat']),
                        'lon': float(data['lon'])
                    }
                    
            print(f"Loaded {len(self.waypoint_coords)} waypoints from graph")
            
        except Exception as e:
            raise Exception(f"Failed to load graph data: {e}")
            
    def load_flights_data(self):
        """Load flights data from CSV file."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Flights CSV not found: {self.csv_file}")
            
        try:
            self.flights_df = pd.read_csv(self.csv_file)
            # Create unique flight IDs to match the automation script
            self.flights_df['unique_flight_id'] = self.flights_df['flight_id'].astype(str) + '_' + self.flights_df['takeoff_time'].astype(str)
            print(f"Loaded {len(self.flights_df)} flights from CSV")
            
        except Exception as e:
            raise Exception(f"Failed to load flights data: {e}")
    
    def initialize_cost_calculation(self):
        """Initialize cost model and load matrices for route cost calculation."""
        if not EQUINOX_AVAILABLE:
            print("Equinox modules not available, skipping cost calculation initialization")
            return
            
        try:
            # Load distance and charges matrices
            if self.distance_matrix_file.exists():
                self.distance_matrix = np.load(str(self.distance_matrix_file))
                print(f"Loaded distance matrix: {self.distance_matrix.shape}")
            else:
                print(f"Warning: Distance matrix not found: {self.distance_matrix_file}")
                
            if self.charges_matrix_file.exists():
                self.charges_matrix = np.load(str(self.charges_matrix_file))
                print(f"Loaded charges matrix: {self.charges_matrix.shape}")
            else:
                print(f"Warning: Charges matrix not found: {self.charges_matrix_file}")
                
            # Initialize cost model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cost_model = CostRev4Ronbun1(device=device)
            print(f"Initialized CostRev4Ronbun1 model on device: {device}")
            
            # Initialize wind model (using a default date - will be updated per flight)
            try:
                wind_data_dir = PROJECT_ROOT / "data" / "era5"
                if wind_data_dir.exists():
                    self.wind_model = WindModel(date_str="2023-04-01", data_dir=str(wind_data_dir))
                    print("Initialized WindModel")
                else:
                    print(f"Warning: Wind data directory not found: {wind_data_dir}")
            except Exception as e:
                print(f"Warning: Could not initialize WindModel: {e}")
                
        except Exception as e:
            print(f"Warning: Cost calculation initialization failed: {e}")
            self.cost_model = None
            self.wind_model = None
    
    def enable_original_cost_calculation(self):
        """Enable calculation of original route costs."""
        self._calculate_original_costs = True
            
    def parse_route_waypoints(self, route_str: str) -> List[str]:
        """Parse waypoint names from route string."""
        if pd.isna(route_str) or not route_str.strip():
            return []
        return [wp.strip() for wp in route_str.split() if wp.strip()]
        
    def get_route_coordinates(self, waypoints: List[str]) -> Tuple[List[float], List[float]]:
        """Get lat/lon coordinates for a list of waypoints."""
        lats, lons = [], []
        
        for waypoint in waypoints:
            if waypoint in self.waypoint_coords:
                coords = self.waypoint_coords[waypoint]
                lats.append(coords['lat'])
                lons.append(coords['lon'])
            else:
                print(f"Warning: Waypoint '{waypoint}' not found in graph")
                
        return lats, lons
        
    def load_sampled_trajectories(self, flight_id: str) -> List[Tuple[float, List[str]]]:
        """Load sampled trajectories for a flight."""
        trajectory_file = self.trajectories_dir / f"{flight_id}.txt"
        
        if not trajectory_file.exists():
            print(f"Warning: Trajectory file not found for flight {flight_id}")
            return []
            
        trajectories = []
        
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
                            
        except Exception as e:
            print(f"Error loading trajectories for flight {flight_id}: {e}")
            
        return trajectories
        
    def analyze_trajectory_costs(self, flight_id: str) -> Optional[Dict]:
        """Analyze cost statistics for sampled trajectories."""
        trajectories = self.load_sampled_trajectories(flight_id)
        
        if not trajectories:
            return None
            
        costs = [cost for cost, _ in trajectories]
        
        analysis = {
            'total_trajectories': len(trajectories),
            'min_cost': min(costs),
            'max_cost': max(costs),
            'mean_cost': sum(costs) / len(costs),
            'cost_range': max(costs) - min(costs)
        }
        
        # Find percentiles
        sorted_costs = sorted(costs)
        n = len(sorted_costs)
        
        analysis['median_cost'] = sorted_costs[n//2] if n % 2 == 1 else (sorted_costs[n//2-1] + sorted_costs[n//2]) / 2
        analysis['p25_cost'] = sorted_costs[n//4]
        analysis['p75_cost'] = sorted_costs[3*n//4]
        
        return analysis
    
    def calculate_original_route_cost(self, unique_flight_id: str, flight_row: pd.Series) -> Optional[float]:
        """Calculate the cost of the original route using the cost model."""
        if not EQUINOX_AVAILABLE or self.cost_model is None:
            return None
            
        try:
            # Find backward closures file
            closure_file = self.samples_dir / unique_flight_id / f"{unique_flight_id}_CLSR_WIND.pkl"
            if not closure_file.exists():
                print(f"Warning: Backward closures file not found: {closure_file}")
                return None
            
            # Parse original route
            original_route = self.parse_route_waypoints(flight_row['route'])
            if len(original_route) < 2:
                print(f"Warning: Original route too short for {unique_flight_id}")
                return None
                
            # Get takeoff time
            takeoff_time_unix = int(flight_row['takeoff_time'])
            
            # Default parameters (these could be made configurable)
            delta_t_seconds_wall_clock = 600  # 10 minutes
            delta_t_seconds_climb = 600
            max_flight_duration_hours = 8.0
            climb_phase_switch_allowance_climb_time_bins = 5
            
            # Calculate route cost
            route_cost = get_route_cost(
                backward_closures_pkl_path=str(closure_file),
                cost_model=self.cost_model,
                original_route_str=' '.join(original_route),
                takeoff_time_unix=takeoff_time_unix,
                route_graph=self.route_graph,
                delta_t_seconds_wall_clock=delta_t_seconds_wall_clock,
                delta_t_seconds_climb=delta_t_seconds_climb,
                max_flight_duration_hours=max_flight_duration_hours,
                climb_phase_switch_allowance_climb_time_bins=climb_phase_switch_allowance_climb_time_bins,
                wind_model=self.wind_model,
                distance_matrix=self.distance_matrix,
                airspace_charge_matrix=self.charges_matrix,
                estimated_landing_time_unix=int(flight_row['landing_time']) if 'landing_time' in flight_row else None
            )
            
            return route_cost
            
        except Exception as e:
            print(f"Error calculating original route cost for {unique_flight_id}: {e}")
            return None
        
    def calculate_map_bounds(self, all_lats: List[float], all_lons: List[float]) -> Tuple[float, float, float, float]:
        """Calculate appropriate map bounds with padding."""
        if not all_lats or not all_lons:
            # Default bounds for LGAV-LFPG route
            return (35, 50, 2, 30)
            
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        
        # Add 10% padding
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        padding_lat = max(lat_range * 0.1, 1.0)  # Minimum 1 degree padding
        padding_lon = max(lon_range * 0.1, 1.0)
        
        return (
            lat_min - padding_lat,
            lat_max + padding_lat,
            lon_min - padding_lon,
            lon_max + padding_lon
        )
        
    def plot_flight_routes(self, flight_id: str, preview: bool = True, save_png: bool = False) -> bool:
        """Plot original route and sampled trajectories for a single flight."""
        
        if not CARTOPY_AVAILABLE:
            print("Error: cartopy not available for plotting")
            return False
            
        # Get flight data - try both original and unique flight ID formats
        flight_data = self.flights_df[self.flights_df['unique_flight_id'] == flight_id]
        if flight_data.empty:
            # Try with original flight_id for backward compatibility
            flight_data = self.flights_df[self.flights_df['flight_id'] == flight_id]
            if flight_data.empty:
                print(f"Flight {flight_id} not found in CSV data")
                return False
            # If found by original ID, use the first match and get its unique ID
            unique_flight_id = flight_data.iloc[0]['unique_flight_id']
        else:
            unique_flight_id = flight_id
            
        flight_row = flight_data.iloc[0]
        original_route = self.parse_route_waypoints(flight_row['route'])
        
        # Get original route coordinates
        orig_lats, orig_lons = self.get_route_coordinates(original_route)
        
        if not orig_lats:
            print(f"No valid coordinates found for original route of flight {flight_id}")
            return False
            
        # Load sampled trajectories using unique flight ID
        sampled_trajectories = self.load_sampled_trajectories(unique_flight_id)
        
        # Calculate original route cost if requested
        original_route_cost = None
        if hasattr(self, '_calculate_original_costs') and self._calculate_original_costs:
            original_route_cost = self.calculate_original_route_cost(unique_flight_id, flight_row)
            if original_route_cost is not None:
                print(f"Original route cost for {unique_flight_id}: {original_route_cost:.6f}")
        
        # Print cost analysis
        cost_analysis = self.analyze_trajectory_costs(unique_flight_id)
        if cost_analysis:
            print(f"Cost Analysis for {unique_flight_id}:")
            print(f"  Total trajectories: {cost_analysis['total_trajectories']}")
            print(f"  Cost range: {cost_analysis['min_cost']:.3f} - {cost_analysis['max_cost']:.3f}")
            print(f"  Mean cost: {cost_analysis['mean_cost']:.3f}")
            print(f"  Median cost: {cost_analysis['median_cost']:.3f}")
            print(f"  25th percentile: {cost_analysis['p25_cost']:.3f}")
            print(f"  75th percentile: {cost_analysis['p75_cost']:.3f}")
            
            # Compare with original route cost if available
            if original_route_cost is not None:
                print(f"  Original route cost: {original_route_cost:.6f}")
                if cost_analysis['min_cost'] < original_route_cost:
                    improvement = ((original_route_cost - cost_analysis['min_cost']) / original_route_cost) * 100
                    print(f"  Best sampled trajectory is {improvement:.1f}% better than original route")
        
        # Collect all coordinates for map bounds
        all_lats = orig_lats.copy()
        all_lons = orig_lons.copy()
        
        # Add trajectory coordinates to bounds calculation
        for _, waypoints in sampled_trajectories[:10]:  # Limit to first 10 for bounds
            traj_lats, traj_lons = self.get_route_coordinates(waypoints)
            all_lats.extend(traj_lats)
            all_lons.extend(traj_lons)
            
        # Calculate map bounds
        lat_min, lat_max, lon_min, lon_max = self.calculate_map_bounds(all_lats, all_lons)
        
        # Create the plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot original route in red
        original_route_label = 'Original Route'
        if original_route_cost is not None:
            original_route_label += f' (cost: {original_route_cost:.3f})'
            
        if len(orig_lats) > 1:
            ax.plot(orig_lons, orig_lats, 'r-', linewidth=3, 
                   transform=ccrs.PlateCarree(), label=original_route_label)
            
        # Plot waypoints
        ax.scatter(orig_lons, orig_lats, c='red', s=50, marker='o', 
                  transform=ccrs.PlateCarree(), zorder=5)
        
        # Find the lowest cost trajectory
        best_trajectory = None
        best_cost = float('inf')
        
        if sampled_trajectories:
            for cost, waypoints in sampled_trajectories:
                if cost < best_cost:
                    best_cost = cost
                    best_trajectory = waypoints
        
        # Plot the best trajectory first with special styling (always show it)
        best_trajectory_plotted = False
        if best_trajectory is not None:
            best_lats, best_lons = self.get_route_coordinates(best_trajectory)
            if len(best_lats) > 1:
                ax.plot(best_lons, best_lats, color='darkgreen', linewidth=3, 
                       linestyle='--', alpha=0.9, transform=ccrs.PlateCarree(),
                       label=f'Lowest Cost Path (cost: {best_cost:.3f})')
                best_trajectory_plotted = True
        
        # Plot other sampled trajectories in different colors
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        plotted_trajectories = 0
        
        for i, (cost, waypoints) in enumerate(sampled_trajectories):
            if plotted_trajectories >= 15:  # Limit number of other trajectories to avoid clutter
                break
                
            # Skip the best trajectory since we already plotted it
            if waypoints == best_trajectory:
                continue
                
            traj_lats, traj_lons = self.get_route_coordinates(waypoints)
            
            if len(traj_lats) > 1:
                color = colors[plotted_trajectories % len(colors)]
                alpha = 0.5 if plotted_trajectories < 5 else 0.3  # Make first few more visible
                
                ax.plot(traj_lons, traj_lats, color=color, linewidth=1.5, alpha=alpha,
                       transform=ccrs.PlateCarree())
                
                plotted_trajectories += 1
                
        # Add origin and destination markers
        origin = flight_row['origin']
        destination = flight_row['destination']
        
        if origin in self.waypoint_coords:
            origin_coords = self.waypoint_coords[origin]
            ax.scatter(origin_coords['lon'], origin_coords['lat'], c='green', s=200, 
                      marker='^', transform=ccrs.PlateCarree(), zorder=10, 
                      edgecolors='black', linewidth=2)
            ax.text(origin_coords['lon'], origin_coords['lat'] + 0.5, origin, 
                   transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
                   
        if destination in self.waypoint_coords:
            dest_coords = self.waypoint_coords[destination]
            ax.scatter(dest_coords['lon'], dest_coords['lat'], c='red', s=200, 
                      marker='s', transform=ccrs.PlateCarree(), zorder=10,
                      edgecolors='black', linewidth=2)
            ax.text(dest_coords['lon'], dest_coords['lat'] + 0.5, destination, 
                   transform=ccrs.PlateCarree(), ha='center', fontweight='bold')
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='red', label=original_route_label),
            mpatches.Patch(color='blue', alpha=0.6, label=f'Sampled Trajectories ({len(sampled_trajectories)} total)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                      markersize=10, label='Origin'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='Destination')
        ]
        
        # Add lowest cost path to legend if available
        if best_trajectory_plotted:
            legend_elements.insert(2, plt.Line2D([0], [0], color='darkgreen', linewidth=3, 
                                               linestyle='--', label=f'Lowest Cost Path ({best_cost:.3f})'))
        
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Set title
        title_text = f'Flight Routes: {unique_flight_id}\n{origin} → {destination}\n'
        title_text += f'Original: {len(original_route)} waypoints'
        if original_route_cost is not None:
            title_text += f' (cost: {original_route_cost:.3f})'
        title_text += f', Sampled: {plotted_trajectories} trajectories shown'
        
        if best_trajectory is not None:
            title_text += f'\nLowest cost trajectory: {best_cost:.3f} ({len(best_trajectory)} waypoints)'
            if original_route_cost is not None and best_cost < original_route_cost:
                improvement = ((original_route_cost - best_cost) / original_route_cost) * 100
                title_text += f' ({improvement:.1f}% improvement)'
        
        plt.title(title_text, fontsize=12, pad=20)
        
        # Save or show
        if save_png:
            png_file = self.plots_dir / f"{unique_flight_id}_routes.png"
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {png_file}")
            
        if preview:
            plt.show()
        else:
            plt.close()
            
        return True
        
    def plot_all_flights(self, preview: bool = False, save_png: bool = True, max_flights: Optional[int] = None):
        """Plot routes for all flights with trajectory data."""
        
        # Get list of flights with trajectory data (these should now be unique flight IDs)
        trajectory_files = list(self.trajectories_dir.glob("*.txt"))
        unique_flight_ids = [f.stem for f in trajectory_files]
        
        if max_flights:
            unique_flight_ids = unique_flight_ids[:max_flights]
            
        total_flights = len(unique_flight_ids)
        successful_plots = 0
        
        print(f"Plotting routes for {total_flights} flights...")
        
        for i, unique_flight_id in enumerate(unique_flight_ids, 1):
            print(f"Processing flight {i}/{total_flights}: {unique_flight_id}")
            
            try:
                if self.plot_flight_routes(unique_flight_id, preview=preview and i <= 3, save_png=save_png):
                    successful_plots += 1
                    
            except Exception as e:
                print(f"Error plotting flight {unique_flight_id}: {e}")
                
            # Progress update
            if i % 5 == 0:
                print(f"Progress: {i}/{total_flights} flights processed "
                      f"({successful_plots} successful)")
                      
        print(f"\nPlotting completed: {successful_plots}/{total_flights} flights successful")
        
        if save_png:
            print(f"PNG files saved to: {self.plots_dir}")
            
        return successful_plots, total_flights - successful_plots
        
    def generate_summary_plot(self, save_png: bool = True):
        """Generate a summary plot showing all original routes on one map."""
        
        if not CARTOPY_AVAILABLE:
            print("Error: cartopy not available for plotting")
            return False
            
        print("Generating summary plot of all original routes...")
        
        # Collect all route coordinates
        all_routes = []
        all_lats, all_lons = [], []
        
        for _, flight_row in self.flights_df.iterrows():
            flight_id = flight_row['flight_id']
            route_waypoints = self.parse_route_waypoints(flight_row['route'])
            
            if route_waypoints:
                route_lats, route_lons = self.get_route_coordinates(route_waypoints)
                if route_lats:
                    all_routes.append((flight_id, route_lats, route_lons))
                    all_lats.extend(route_lats)
                    all_lons.extend(route_lons)
                    
        if not all_routes:
            print("No valid routes found for summary plot")
            return False
            
        # Calculate map bounds
        lat_min, lat_max, lon_min, lon_max = self.calculate_map_bounds(all_lats, all_lons)
        
        # Create the plot
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.2, color='lightblue')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot all routes
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_routes)))
        
        for i, (flight_id, route_lats, route_lons) in enumerate(all_routes):
            if len(route_lats) > 1:
                ax.plot(route_lons, route_lats, color=colors[i], linewidth=1.5, alpha=0.7,
                       transform=ccrs.PlateCarree())
                       
        # Add origin and destination points
        origins = set()
        destinations = set()
        
        for _, flight_row in self.flights_df.iterrows():
            origins.add(flight_row['origin'])
            destinations.add(flight_row['destination'])
            
        # Plot origins
        for origin in origins:
            if origin in self.waypoint_coords:
                coords = self.waypoint_coords[origin]
                ax.scatter(coords['lon'], coords['lat'], c='green', s=150, 
                          marker='^', transform=ccrs.PlateCarree(), zorder=10,
                          edgecolors='black', linewidth=1)
                ax.text(coords['lon'], coords['lat'] + 0.5, origin, 
                       transform=ccrs.PlateCarree(), ha='center', fontweight='bold', fontsize=8)
                       
        # Plot destinations
        for dest in destinations:
            if dest in self.waypoint_coords:
                coords = self.waypoint_coords[dest]
                ax.scatter(coords['lon'], coords['lat'], c='red', s=150, 
                          marker='s', transform=ccrs.PlateCarree(), zorder=10,
                          edgecolors='black', linewidth=1)
                ax.text(coords['lon'], coords['lat'] + 0.5, dest, 
                       transform=ccrs.PlateCarree(), ha='center', fontweight='bold', fontsize=8)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, 
                      label=f'Original Routes ({len(all_routes)} flights)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                      markersize=10, label=f'Origins ({len(origins)})'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label=f'Destinations ({len(destinations)})')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        plt.title(f'Summary: All Original Flight Routes - Case {self.case_name}\n'
                 f'{len(all_routes)} flights plotted', fontsize=14, pad=20)
        
        # Save
        if save_png:
            png_file = self.plots_dir / f"summary_all_routes_{self.case_name}.png"
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {png_file}")
            
        plt.show()
        return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize flight routes and sampled trajectories")
    parser.add_argument("--case", default=CASE_NAME, help="Case name to process")
    parser.add_argument("--flight", help="Specific flight ID to plot")
    parser.add_argument("--preview", action="store_true", help="Show interactive preview")
    parser.add_argument("--save", action="store_true", help="Save PNG files")
    parser.add_argument("--max-flights", type=int, help="Maximum number of flights to plot")
    parser.add_argument("--summary", action="store_true", help="Generate summary plot of all routes")
    parser.add_argument("--all", action="store_true", help="Plot all flights (same as --save)")
    parser.add_argument("--analyze-costs", action="store_true", help="Show cost analysis for all flights")
    parser.add_argument("--calculate-original-cost", action="store_true", help="Calculate original route costs using cost model")
    
    args = parser.parse_args()
    
    if not CARTOPY_AVAILABLE:
        print("Error: cartopy is required for plotting. Install with: pip install cartopy")
        return 1
    
    if args.all:
        args.save = True
    
    try:
        visualizer = RouteVisualizer(args.case)
        
        # Enable original cost calculation if requested
        if args.calculate_original_cost:
            visualizer.enable_original_cost_calculation()
        
        if args.summary:
            visualizer.generate_summary_plot(save_png=args.save)
            
        elif args.analyze_costs:
            # Show cost analysis for all flights
            trajectory_files = list(visualizer.trajectories_dir.glob("*.txt"))
            unique_flight_ids = [f.stem for f in trajectory_files]
            
            if args.max_flights:
                unique_flight_ids = unique_flight_ids[:args.max_flights]
                
            print(f"Cost Analysis Summary for {len(unique_flight_ids)} flights:")
            print("=" * 60)
            
            all_analyses = []
            for unique_flight_id in unique_flight_ids:
                analysis = visualizer.analyze_trajectory_costs(unique_flight_id)
                if analysis:
                    all_analyses.append((unique_flight_id, analysis))
                    print(f"\n{unique_flight_id}:")
                    print(f"  Trajectories: {analysis['total_trajectories']}")
                    print(f"  Cost range: {analysis['min_cost']:.3f} - {analysis['max_cost']:.3f}")
                    print(f"  Mean: {analysis['mean_cost']:.3f}, Median: {analysis['median_cost']:.3f}")
                    
            # Summary statistics across all flights
            if all_analyses:
                print("\n" + "=" * 60)
                print("OVERALL SUMMARY:")
                all_min_costs = [a[1]['min_cost'] for a in all_analyses]
                all_mean_costs = [a[1]['mean_cost'] for a in all_analyses]
                print(f"Best trajectory cost across all flights: {min(all_min_costs):.3f}")
                print(f"Worst best-trajectory cost: {max(all_min_costs):.3f}")
                print(f"Average of mean costs: {sum(all_mean_costs)/len(all_mean_costs):.3f}")
                
        elif args.flight:
            # Plot specific flight
            success = visualizer.plot_flight_routes(
                args.flight, 
                preview=args.preview, 
                save_png=args.save
            )
            if not success:
                print(f"Failed to plot flight {args.flight}")
                return 1
                
        else:
            # Plot all flights with trajectories
            successful, failed = visualizer.plot_all_flights(
                preview=args.preview,
                save_png=args.save,
                max_flights=args.max_flights
            )
            
            if failed > 0:
                print(f"Warning: {failed} flights failed to plot")
                
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())