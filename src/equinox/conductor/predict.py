"""
Equinox Conductor - Complete Prediction Pipeline

This module implements a complete prediction pipeline that processes flight requests
and generates trajectory ensembles using the Equinox framework.
"""

import os
import sys
import tempfile
import shutil
import time
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import networkx as nx

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from equinox.training.prep.prep_graph import process_and_save_graph
from equinox.feateng.airspace_charges import compute_charges_for_graph
from equinox.feateng.laplace import enumerate_nodes, node_names_to_ids
from equinox.feateng.distance import haversine_distance_matrix
from equinox.training.prep.resculpt_viterbi import viterbi_match, haversine_nm
from equinox.training.prep.remove_edges_for_sectors import remove_edges_through_sectors
from equinox.training.prep.prep_routes import filter_routes_by_origin_dest

@dataclass
class TrajectoryEnsemble:
    """Container for trajectory ensemble results"""
    trajectories: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    config: Dict[str, Any]

class ConductorPipeline:
    """Main prediction pipeline implementation"""
    
    def __init__(self, temp_dir: str = "equinox_temp"):
        self.temp_dir = Path(temp_dir)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.path_prefix = '/Volumes/CrucialX/project-akrav/'
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_temp_directory(self, case_name: str) -> Path:
        """Setup temporary directory structure"""
        case_dir = self.temp_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['graphs', 'transitions', 'samples', 'configs', 'trajectories']:
            (case_dir / subdir).mkdir(exist_ok=True)
            
        return case_dir
        
    def check_existing_resources(self, case_dir: Path, case_name: str) -> Dict[str, bool]:
        """Check which resources already exist"""
        resources = {}
        
        # Graph files
        graph_path = case_dir / "graphs" / "routes.gml"
        resources['graph'] = graph_path.exists()
        
        # Distance matrix
        dist_path = case_dir / "graphs" / "routes_distances.npy"
        resources['distances'] = dist_path.exists()
        
        # Charges matrix
        charges_path = case_dir / "graphs" / "routes_charges.npy"
        resources['charges'] = charges_path.exists()
        
        # Routes data
        routes_path = case_dir / "all_routes.csv"
        resources['routes'] = routes_path.exists()
        
        # Sculpted routes
        sculpted_path = case_dir / "all_routes_sculpted.csv"
        resources['sculpted_routes'] = sculpted_path.exists()
        
        return resources
        
    def prep_graph(self, case_dir: Path, origin: str, destination: str, 
                   recycle_resources: bool = False) -> bool:
        """Prepare graph from prep_all.py implementation"""
        
        graph_path = case_dir / "graphs" / "routes.gml"
        
        if recycle_resources and graph_path.exists():
            self.logger.info(f"Recycling existing graph: {graph_path}")
            return True
            
        self.logger.info(f"Preparing graph for {origin} to {destination}")
        
        try:
            nodes_only_graph_path = os.path.join(
                self.path_prefix, "data", "graphs", "ats_fra_nodes_only.gml"
            )
            routes_dir = os.path.join(self.path_prefix, "matched_filtered_data")
            
            # Create output directory
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sectors to avoid
            sectors_2_avoid = []
            
            start_time = time.time()
            Gno = process_and_save_graph(
                nodes_only_graph_path,
                origin,
                destination,
                routes_dir,
                output_path=str(graph_path),
                minimum_detour_allowed=0.075,
                n_iter=15,
                max_allowed_deviation_angle=60,
                remove_collinear_edges_option=False,
                sectors_to_avoid=sectors_2_avoid,
                improve_connectivity_option=False
            )
            
            self.logger.info(f"Graph processing completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"Final graph has {Gno.number_of_nodes()} nodes and {Gno.number_of_edges()} edges")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare graph: {e}")
            return False
            
    def prep_routes_data(self, case_dir: Path, origin: str, destination: str,
                        recycle_resources: bool = False) -> bool:
        """Prepare routes data from prep_all.py implementation"""
        
        routes_path = case_dir / "all_routes.csv"
        
        if recycle_resources and routes_path.exists():
            self.logger.info(f"Recycling existing routes data: {routes_path}")
            return True
            
        self.logger.info(f"Preparing routes data for {origin} to {destination}")
        
        try:
            input_directory = os.path.join(self.path_prefix, "matched_filtered_data")
            output_directory = str(case_dir)
            
            filtered_flights = filter_routes_by_origin_dest(
                input_dir=input_directory,
                origin=origin,
                dest=destination,
                output_dir=output_directory,
            )
            
            self.logger.info(f"Filtered {len(filtered_flights)} flights")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare routes data: {e}")
            return False
            
    def prep_airspace_charges(self, case_dir: Path, recycle_resources: bool = False) -> bool:
        """Prepare airspace charges from prep_all.py implementation"""
        
        charges_path = case_dir / "graphs" / "routes_charges.npy"
        
        if recycle_resources and charges_path.exists():
            self.logger.info(f"Recycling existing charges: {charges_path}")
            return True
            
        self.logger.info("Preparing airspace charges")
        
        try:
            charges_df = pd.read_csv(self.project_root / "data" / "ufir" / "fir_charges.csv")
            graph_path = case_dir / "graphs" / "routes.gml"
            
            if not graph_path.exists():
                self.logger.error(f"Graph file not found: {graph_path}")
                return False
                
            graph = nx.read_gml(graph_path)
            charge_graph = compute_charges_for_graph(graph, charges_df)
            
            # Create cost matrix
            node_mapping = enumerate_nodes(charge_graph)
            node_ids = node_names_to_ids(charge_graph, list(charge_graph.nodes()))
            cost_matrix = np.zeros((len(node_ids), len(node_ids)))
            
            for u, v, data in charge_graph.edges(data=True):
                cost_matrix[node_mapping[u], node_mapping[v]] = data.get('airspace_charge', 0)
                
            # Save cost matrix
            np.save(charges_path, cost_matrix)
            self.logger.info(f"Saved cost matrix to {charges_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare airspace charges: {e}")
            return False
            
    def prep_distance_matrix(self, case_dir: Path, recycle_resources: bool = False) -> bool:
        """Prepare distance matrix from prep_all.py implementation"""
        
        dist_path = case_dir / "graphs" / "routes_distances.npy"
        
        if recycle_resources and dist_path.exists():
            self.logger.info(f"Recycling existing distance matrix: {dist_path}")
            return True
            
        self.logger.info("Preparing distance matrix")
        
        try:
            graph_path = case_dir / "graphs" / "routes.gml"
            
            if not graph_path.exists():
                self.logger.error(f"Graph file not found: {graph_path}")
                return False
                
            graph = nx.read_gml(graph_path)
            dist_matrix = haversine_distance_matrix(graph)
            
            # Save distance matrix
            np.save(dist_path, dist_matrix)
            self.logger.info(f"Saved distance matrix to {dist_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare distance matrix: {e}")
            return False
            
    def prep_sculpting_routes(self, case_dir: Path, recycle_resources: bool = False) -> bool:
        """Sculpt existing routes to new graph from prep_all.py implementation"""
        
        sculpted_path = case_dir / "all_routes_sculpted.csv"
        
        if recycle_resources and sculpted_path.exists():
            self.logger.info(f"Recycling existing sculpted routes: {sculpted_path}")
            return True
            
        self.logger.info("Sculpting routes to graph")
        
        try:
            graph_path = case_dir / "graphs" / "routes.gml"
            routes_csv_path = case_dir / "all_routes.csv"
            
            if not graph_path.exists():
                self.logger.error(f"Graph file not found: {graph_path}")
                return False
                
            if not routes_csv_path.exists():
                self.logger.error(f"Routes file not found: {routes_csv_path}")
                return False
                
            # Load graphs
            Gwf = nx.read_gml(graph_path)
            
            nodes_only_graph_path = os.path.join(
                self.path_prefix, "data", "graphs", "ats_fra_nodes_only.gml"
            )
            Gwp = nx.read_gml(nodes_only_graph_path)
            
            # Add length_nm attribute
            for u, v in Gwf.edges():
                lat1, lon1 = Gwf.nodes[u]["lat"], Gwf.nodes[u]["lon"]
                lat2, lon2 = Gwf.nodes[v]["lat"], Gwf.nodes[v]["lon"]
                Gwf.edges[u, v]["length_nm"] = haversine_nm(lat1, lon1, lat2, lon2)
                
            # Load routes
            df = pd.read_csv(routes_csv_path)
            sculpted_routes = []
            
            for index, row in df.iterrows():
                flight_id = row.get("flight_id", f"flight_{index}")
                
                real_waypoints = row.get("real_waypoints", "")
                if not real_waypoints or not isinstance(real_waypoints, str):
                    continue
                    
                orig_wp_names = real_waypoints.split()
                obs_pts = []
                
                for n in orig_wp_names:
                    if n in Gwp.nodes:
                        obs_pts.append((Gwp.nodes[n]["lat"], Gwp.nodes[n]["lon"]))
                        
                if len(obs_pts) < 2:
                    continue
                    
                try:
                    best_nodes, best_edges = viterbi_match(Gwf, obs_pts, k=10, beta=0.5)
                    if not best_edges:
                        continue
                        
                    full_nodes = [best_edges[0][0]] + [edge[1] for edge in best_edges]
                    sculpted_route_str = " ".join(full_nodes)
                    
                    sculpted_routes.append({
                        "flight_id": flight_id,
                        "route": sculpted_route_str,
                        "takeoff_time": row.get("takeoff"),
                        "landing_time": row.get("landing"),
                        "cruise_altitude": max(map(float, row.get("alts", "0").split())) 
                                        if row.get("alts") and isinstance(row.get("alts"), str) 
                                        and row.get("alts").strip() else None,
                        "origin": row.get("origin"),
                        "destination": row.get("destination"),
                        "flight_time_s": row.get("flight_time_s")
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to sculpt route for flight {flight_id}: {e}")
                    continue
                    
            if sculpted_routes:
                output_df = pd.DataFrame(sculpted_routes)
                output_df.to_csv(sculpted_path, index=False)
                self.logger.info(f"Saved {len(output_df)} sculpted routes to {sculpted_path}")
                return True
            else:
                self.logger.warning("No routes were sculpted successfully")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to sculpt routes: {e}")
            return False
            
    def create_config_yaml(self, case_dir: Path, case_name: str, 
                          origin: str, destination: str, 
                          crz_fl: float, crz_spd: float, date: str, gamma: float = 1.0) -> Path:
        """Create configuration YAML with cruise flight level, speed, and date for wind model"""
        
        config_path = case_dir / "config.yaml"
        
        # Convert cruise flight level to altitude in feet
        cruise_altitude_ft = crz_fl * 100.0  # FL350 -> 35000 ft
        
        # Build paths relative to case directory
        graph_path = case_dir / "graphs" / "routes.gml"
        charges_path = case_dir / "graphs" / "routes_charges.npy"
        distances_path = case_dir / "graphs" / "routes_distances.npy"
        output_dir = case_dir / "transitions"
        wind_data_dir = self.project_root / "data" / "era5"
        
        yaml_content = {
            'aircraft_model': 'NARROW_BODY_JET',
            'charges_file_path': str(charges_path),
            'climb_phase_switch_allowance_climb_time_bins': 10,
            'cost_model_beta0': 0.0,
            'cost_model_beta1': 1.0,
            'cost_model_beta2': 1.0,
            'cost_model_beta3': 1.0,
            'cruise_altitude_ft': cruise_altitude_ft,
            'cruise_speed_kts': crz_spd,
            'delta_t_seconds': 600,
            'device_preference': 'cuda',
            'distances_file_path': str(distances_path),
            'etto_delta_t_seconds': 30,
            'file_prefix': case_name,
            'goal_elevation_ft': 0.0,
            'goal_node': destination,
            'graph_file_path': str(graph_path),
            'initial_alt_ft': 0.0,
            'max_elapsed_time_since_takeoff_hours': 0.75,
            'max_flight_duration_hours': 5.0,
            'origin_node': origin,
            'output_dir': str(output_dir),
            'source_elevation_ft': 0.0,
            'wind_data_dir': str(wind_data_dir),
            'wind_date': date,
            'disable_config_wind_model': False,
            'gamma': gamma,
            'alpha_pref_reg': 1.0,
            'cost_model_version': '4'
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            
        self.logger.info(f"Created config file: {config_path}")
        return config_path
        
    def run_sampling_pipeline(self, config_path: Path, case_dir: Path) -> bool:
        """Run the sampling pipeline from sample1_modified.py"""
        
        try:
            from equinox.sampling.trespass.sampler_auto import run_full_pipeline
            
            self.logger.info("Starting sampling pipeline")
            success = run_full_pipeline(str(config_path))
            
            if success:
                self.logger.info("Sampling pipeline completed successfully")
                return True
            else:
                self.logger.error("Sampling pipeline failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running sampling pipeline: {e}")
            return False
            
    def load_trajectory_ensemble(self, case_dir: Path, case_name: str) -> Optional[List[Dict]]:
        """Load trajectory ensemble from output files"""
        
        try:
            # Look for trajectory file
            trajectory_file = case_dir / "trajectories" / f"{case_name}_CLB_trajectories.txt"
            
            if not trajectory_file.exists():
                # Try alternative location
                trajectory_file = case_dir / "transitions" / f"{case_name}_CLB_trajectories.txt"
                
            if not trajectory_file.exists():
                self.logger.error(f"Trajectory file not found: {trajectory_file}")
                return None
                
            trajectories = []
            
            with open(trajectory_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Parse line format: "cost,waypoint1 waypoint2 ..."
                    parts = line.split(',', 1)
                    if len(parts) != 2:
                        continue
                        
                    cost = float(parts[0])
                    waypoints = parts[1].split()
                    
                    trajectories.append({
                        'trajectory_id': line_idx,
                        'cost': cost,
                        'waypoints': waypoints,
                        'num_waypoints': len(waypoints)
                    })
                    
            self.logger.info(f"Loaded {len(trajectories)} trajectories")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Failed to load trajectory ensemble: {e}")
            return None

def predict(origin_airport: str, destination_airport: str, 
           crz_fl: float, crz_spd: float, date: str,
           gamma: float = 1.0,
           recycle_resources: bool = False) -> Optional[TrajectoryEnsemble]:
    """
    Main prediction function that generates trajectory ensembles
    
    Args:
        origin_airport: ICAO code of origin airport
        destination_airport: ICAO code of destination airport
        crz_fl: Cruise flight level (e.g., 350 for FL350)
        crz_spd: Cruise speed in knots
        date: Date string for wind model (e.g., '2023-04-01')
        gamma: Temperature parameter for soft value iteration (default: 1.0)
        recycle_resources: Whether to reuse existing resources
        
    Returns:
        TrajectoryEnsemble object containing the generated trajectories
    """
    
    # Create case name
    case_name = f"{origin_airport}_{destination_airport}"
    
    # Initialize pipeline
    pipeline = ConductorPipeline()
    
    try:
        # Setup temporary directory
        case_dir = pipeline.setup_temp_directory(case_name)
        pipeline.logger.info(f"Working in directory: {case_dir}")
        
        # Check existing resources
        if recycle_resources:
            existing = pipeline.check_existing_resources(case_dir, case_name)
            pipeline.logger.info(f"Existing resources: {existing}")
            
        # Run preparation pipeline
        stages = [
            ("Graph preparation", lambda: pipeline.prep_graph(case_dir, origin_airport, destination_airport, recycle_resources)),
            ("Routes data preparation", lambda: pipeline.prep_routes_data(case_dir, origin_airport, destination_airport, recycle_resources)),
            ("Airspace charges preparation", lambda: pipeline.prep_airspace_charges(case_dir, recycle_resources)),
            ("Distance matrix preparation", lambda: pipeline.prep_distance_matrix(case_dir, recycle_resources)),
            ("Route sculpting", lambda: pipeline.prep_sculpting_routes(case_dir, recycle_resources)),
        ]
        
        for stage_name, stage_func in stages:
            pipeline.logger.info(f"Starting {stage_name}")
            if not stage_func():
                pipeline.logger.error(f"Failed at {stage_name}")
                return None
                
        # Create configuration
        config_path = pipeline.create_config_yaml(case_dir, case_name, origin_airport, destination_airport, crz_fl, crz_spd, date, gamma)
        
        # Run sampling pipeline
        if not pipeline.run_sampling_pipeline(config_path, case_dir):
            pipeline.logger.error("Sampling pipeline failed")
            return None
            
        # Load results
        trajectories = pipeline.load_trajectory_ensemble(case_dir, case_name)
        if trajectories is None:
            pipeline.logger.error("Failed to load trajectory ensemble")
            return None
            
        # Create ensemble object
        ensemble = TrajectoryEnsemble(
            trajectories=trajectories,
            metadata={
                'origin_airport': origin_airport,
                'destination_airport': destination_airport,
                'cruise_flight_level': crz_fl,
                'cruise_speed_kts': crz_spd,
                'date': date,
                'num_trajectories': len(trajectories),
                'case_name': case_name,
                'case_dir': str(case_dir)
            },
            config=yaml.safe_load(config_path.read_text())
        )
        
        pipeline.logger.info(f"Prediction completed successfully with {len(trajectories)} trajectories")
        return ensemble
        
    except Exception as e:
        pipeline.logger.error(f"Prediction failed: {e}")
        return None