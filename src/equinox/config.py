import yaml
import torch
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, Type
from pathlib import Path


def get_cost_model_class(cost_model_version: str) -> Type[torch.nn.Module]:
    """Returns the cost model class based on the version string."""
    # Local imports to avoid circular dependencies if cost models import config
    from equinox.cost.cost_rev2_reg import CostRev2
    from equinox.cost.cost_rev3 import CostRev3
    from equinox.cost.cost_rev4 import CostRev4
    from equinox.cost.cost_rev4_lite import CostRev4Lite
    from equinox.cost.cost_rev4_ronbun1 import CostRev4Ronbun1

    if cost_model_version == "2reg":
        return CostRev2
    elif cost_model_version == "3":
        return CostRev3
    elif cost_model_version == "4":
        return CostRev4
    elif cost_model_version == "4lite":
        return CostRev4Lite
    elif cost_model_version == '4rb1':
        return CostRev4Ronbun1
    else:
        raise ValueError(f"Unsupported cost model version: {cost_model_version}")


@dataclass
class RunConfiguration:
    """Configuration class for storing algorithm parameters."""
    
    # Graph and file paths
    graph_file_path: str = "data/graph/LEMD_EGLL_2023_04_01.gml"
    distances_file_path: str = "data/graph/LEMD_EGLL_2023_04_01_distances.npy"
    charges_file_path: str = "data/graph/LEMD_EGLL_2023_04_01_charges.npy"
    wind_avg_file_path: str = "data/graph/wind_averages/LEMD_EGLL_2023_04_01_wind_avg.pt"
    
    # Wind model parameters
    wind_date: Optional[str] = "2023-04-01"
    wind_data_dir: str = "data/era5"
    
    # Cost model parameters
    cost_model_beta0: float = 1.0
    cost_model_beta1: float = 1.0
    cost_model_beta2: float = 1.0
    cost_model_beta3: float = 1.0
    
    # Performance model parameters
    aircraft_model: str = "NARROW_BODY_JET"
    cruise_altitude_ft: float = 35000.0
    cruise_speed_kts: float = 450.0
    
    # Flight parameters
    landing_time_str: str = "2023-04-01 12:00:00"
    takeoff_time_str: str = "2023-04-01 10:15:00"
    estimated_takeoff_time_str: str = "2023-04-01 10:15:00"
    estimated_landing_time_str: str = "2023-04-01 12:00:00"
    source_elevation_ft: float = 0.0
    goal_elevation_ft: float = 0.0
    initial_alt_ft: float = 0.0
    
    # Time parameters
    delta_t_seconds: int = 600
    max_flight_duration_hours: float = 5.0
    etto_delta_t_seconds: int = 30
    max_elapsed_time_since_takeoff_hours: float = 0.75
    climb_phase_switch_allowance_climb_time_bins: int = 10
    
    # Route parameters
    origin_node: str = "LEMD"
    goal_node: str = "EGLL"
    
    # Output parameters
    output_dir: str = "data/graph/transitions"
    file_prefix: str = "LEMD_EGLL_2023_04_01"
    tres_forward_output_file_name: str = file_prefix + "_CLB_WIND"
    tres_backward_output_file_name: str = file_prefix + "_CLSR_WIND"
    thinning_output_file_name: str = file_prefix + "_REACHABLE_WIND"
    
    # Device configuration
    device_preference: str = "cuda"  # "cuda" or "cpu"

    # Cost model version
    cost_model_version: str = "3"

    # Temperature
    gamma: float = 0.01

    # Regularization parameters
    alpha_pref_reg: float = 1.0

    # Wind model configuration
    disable_config_wind_model: bool = False

    # Checkpoint path
    checkpoint_path: str = None
    
    def get_device(self) -> torch.device:
        """Get the appropriate torch device based on availability and preference."""
        if self.device_preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load_graph(self) -> Tuple[nx.Graph, Dict[str, int], Dict[int, str], torch.Tensor]:
        """Load the route graph and create node mappings and coordinates."""
        G = nx.read_gml(self.graph_file_path)
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        idx_to_node = {i: node for i, node in enumerate(G.nodes())}
        
        # Node coordinates
        node_coords_deg = torch.zeros((len(G.nodes()), 2), dtype=torch.float32)
        for node_name, node_idx_val in node_to_idx.items():
            node_data = G.nodes[node_name]
            lat = node_data['lat']
            lon = node_data['lon']
            node_coords_deg[node_idx_val, 0] = lat
            node_coords_deg[node_idx_val, 1] = lon
        
        return G, node_to_idx, idx_to_node, node_coords_deg
    
    def initialize_cost_model(self, num_waypoints: int, cost_model_version: str = None) -> torch.nn.Module:
        """Initialize the cost model with configuration parameters."""
        _cost_model_version = cost_model_version if cost_model_version is not None else self.cost_model_version
        cost_model_class = get_cost_model_class(_cost_model_version)

        cost_model_instance = cost_model_class(
            beta0=self.cost_model_beta0,
            beta1=self.cost_model_beta1,
            beta2=self.cost_model_beta2,
            beta3=self.cost_model_beta3,
            num_waypoints=num_waypoints,
            alpha_pref_reg=self.alpha_pref_reg,
            device=self.get_device()
        )
        
        print(f"Cost model version {_cost_model_version} initialized with {num_waypoints} waypoints")
        return cost_model_instance
    
    def initialize_wind_model(self) -> Any:
        """Initialize the wind model based on configuration."""
        if self.wind_date:
            from equinox.wind.wind_date import WindDate

            print(f"Initializing WindDate model for date: {self.wind_date}")
            return WindDate(date_str=self.wind_date, data_dir=self.wind_data_dir)
        else:
            from equinox.wind.wind_free import WindFree

            print("Initializing WindFree model")
            return WindFree()
    
    def initialize_performance_model(self) -> 'Performance':
        """Initialize the performance model with configuration parameters."""
        from equinox.vnav.vnav_performance import Performance
        import equinox.vnav.vnav_profiles_rev1 as vnav_profiles

        model_key = self.aircraft_model.upper()

        try:
            climb_speed_profile = getattr(vnav_profiles, f"{model_key}_CLIMB_PROFILE")
            descent_speed_profile = getattr(vnav_profiles, f"{model_key}_DESCENT_PROFILE")
            climb_vertical_speed_profile = getattr(vnav_profiles, f"{model_key}_CLIMB_VS_PROFILE")
            descent_vertical_speed_profile = getattr(vnav_profiles, f"{model_key}_DESCENT_VS_PROFILE")
        except AttributeError as e:
            raise ValueError(
                f"Invalid aircraft model: {self.aircraft_model}. "
                f"Could not find corresponding profiles in vnav_profiles_rev1.py"
            ) from e
        
        performance_model = Performance(
            climb_speed_profile=climb_speed_profile,
            descent_speed_profile=descent_speed_profile,
            climb_vertical_speed_profile=climb_vertical_speed_profile,
            descent_vertical_speed_profile=descent_vertical_speed_profile,
            cruise_altitude_ft=self.cruise_altitude_ft,
            cruise_speed_kts=self.cruise_speed_kts,
        )
        return performance_model
    
    def load_distance_matrix(self) -> np.ndarray:
        """Load the distance matrix."""
        return np.load(self.distances_file_path)
    
    def load_charges_matrix(self) -> np.ndarray:
        """Load the airspace charges matrix."""
        return np.load(self.charges_file_path)
    
    def initialize_all_components(self, cost_model_version: str = None, manual_cost_model_init: bool = False) -> Dict[str, Any]:
        """Initialize all components and return them in a dictionary."""
        # Load graph and create mappings
        G, node_to_idx, idx_to_node, node_coords_deg = self.load_graph()
        
        # Initialize models
        if manual_cost_model_init:
            cost_model = None
        else:
            cost_model = self.initialize_cost_model(len(G.nodes()), cost_model_version)
        if not self.disable_config_wind_model:
            wind_model = self.initialize_wind_model()
        else:
            wind_model = None
        performance_model = self.initialize_performance_model()
        
        # Load matrices
        dist_matrix = self.load_distance_matrix()
        ac_matrix = self.load_charges_matrix()
        
        # Get device
        device = self.get_device()
        
        # Get node indices
        origin_node_idx = self.get_origin_node_idx(node_to_idx)
        goal_node_idx = self.get_goal_node_idx(node_to_idx)
        
        components = {
            'graph': G,
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'node_coords_deg': node_coords_deg,
            'cost_model': cost_model,
            'wind_model': wind_model,
            'performance_model': performance_model,
            'dist_matrix': dist_matrix,
            'ac_matrix': ac_matrix,
            'device': device,
            'origin_node_idx': origin_node_idx,
            'goal_node_idx': goal_node_idx,
            'num_nodes': len(G.nodes())
        }
        
        print(f"All components initialized successfully:")
        print(f"  - Graph loaded with {len(G.nodes())} nodes")
        print(f"  - Origin: {self.origin_node} (idx: {origin_node_idx})")
        print(f"  - Goal: {self.goal_node} (idx: {goal_node_idx})")
        print(f"  - Device: {device}")
        print(f"  - Distance matrix shape: {dist_matrix.shape}")
        print(f"  - Charges matrix shape: {ac_matrix.shape}")
        
        return components
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RunConfiguration':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'RunConfiguration':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def get_origin_node_idx(self, node_to_idx: Dict[str, int]) -> int:
        """Get the index of the origin node."""
        return node_to_idx[self.origin_node]
    
    def get_goal_node_idx(self, node_to_idx: Dict[str, int]) -> int:
        """Get the index of the goal node."""
        return node_to_idx[self.goal_node]
    
def save_default_config_to_yaml(file_path: str) -> None:
    """Save the default configuration to a YAML file."""
    config = RunConfiguration()
    config.save_to_yaml(file_path)
    print(f"Default configuration saved to {file_path}")

def demo_configuration_usage():
    """Demonstration of how to use the AlgorithmConfiguration class."""
    
    # Create a new configuration with default values
    config = RunConfiguration()
    
    # Modify some parameters
    config.cruise_altitude_ft = 37000.0
    config.origin_node = "KJFK"
    config.goal_node = "EGLL"
    
    # Save to YAML
    config.save_to_yaml("data/profiles/demo_config.yaml")
    print("Configuration saved to demo_config.yaml")
    
    # Load from YAML
    loaded_config = RunConfiguration.load_from_yaml("data/profiles/demo_config.yaml")
    print(f"Loaded configuration - Origin: {loaded_config.origin_node}, Goal: {loaded_config.goal_node}")
    print(f"Cruise altitude: {loaded_config.cruise_altitude_ft} ft")
    
    # Get device
    device = loaded_config.get_device()
    print(f"Using device: {device}")
    
    return loaded_config


def demo_initialization():
    """Demonstration of how to initialize all components using the configuration."""
    
    # Load configuration from YAML
    config = RunConfiguration.load_from_yaml("data/profiles/nbjet_35450_egll_lemd_2023_04_01.yaml")
    
    # Initialize all components
    components = config.initialize_all_components()
    
    # Access individual components
    G = components['graph']
    cost_model = components['cost_model']
    performance_model = components['performance_model']
    wind_model = components['wind_model']
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    device = components['device']
    
    print(f"\nExample usage:")
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Cost model device: {cost_model.device}")
    print(f"Performance model cruise altitude: {performance_model.cruise_altitude_ft} ft")
    
    return components


if __name__ == "__main__":
    # demo_configuration_usage()
    # print("\n" + "="*50 + "\n")
    # demo_initialization()
    response = input("Do you want to save the config to a yaml file? (y/n): ")
    if response.lower() in ['y', 'yes']:
        save_default_config_to_yaml("data/profiles/nbjet_35450_egll_lemd_2023_04_01.yaml")