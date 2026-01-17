import yaml
import torch
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, Type
from pathlib import Path

from equinox.cost.cost_linear_disentangled import CostLinearDisentangled

@dataclass
class RunConfiguration:
    """Configuration class for storing algorithm parameters."""
    
    # Graph and file paths
    graph_file_path: str = None
    distances_file_path: str = None
    charges_file_path: str = None
    wind_avg_file_path: str = None 

    # Wind model parameters
    wind_date: Optional[str] = None
    wind_data_dir: str = "data/era5"

    
    # Performance model parameters
    aircraft_model: str = None
    cruise_altitude_ft: float = None
    cruise_speed_kts: float = None
    
    # Flight parameters are not applicable because `RunConfiguration` is 
    # landing_time_str: Optional[str] = None
    # takeoff_time_str: Optional[str] = None
    # estimated_takeoff_time_str: Optional[str] = None
    # estimated_landing_time_str: Optional[str] = None
    source_elevation_ft: Optional[float] = None
    goal_elevation_ft: Optional[float] = None
    initial_alt_ft: Optional[float] = None

    # Time bin, time limit parameters
    delta_t_seconds: Optional[int] = None
    max_flight_duration_hours: Optional[float] = None
    etto_delta_t_seconds: Optional[int] = None
    max_elapsed_time_since_takeoff_hours: Optional[float] = None
    climb_phase_switch_allowance_climb_time_bins: Optional[int] = None

    # Route parameters
    origin_node: Optional[str] = None
    goal_node: Optional[str] = None
    
    # Output parameters
    output_dir: str = None
    file_prefix: str = None
    tres_forward_output_file_name: Optional[str] = None
    tres_backward_output_file_name: Optional[str] = None
    thinning_output_file_name: Optional[str] = None
    
    # Device configuration
    device_preference: str = "cpu"  # "cuda" or "cpu"

    # Cost model version
    cost_model_version: str = None

    # Regularization parameters
    alpha_pref_reg: float = None

    # Wind model configuration
    disable_config_wind_model: bool = False

    # Checkpoint path
    checkpoint_path: str = None

    # lin-disent cost model specific params
    common_weights: list = None 
    preference_weight: float = None 
    
    # training config
    gamma: float = None
    training_batch_size: int = None
    common_features_learning_rate: float = None
    preference_feature_learning_rate: float = None
    preference_projection_ridge: float = None
    max_iters: int = None
    convergence_threshold: float = None
    checkpoint_interval: int = None
    
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
        
        if _cost_model_version == "lin_disent":
            # The configured cost model is linear disentangled: `cost_linear_disentangled.py`
            cost_model_instance = CostLinearDisentangled(
                common_weights = self.common_weights,
                preference_weights = self.preference_weight,
                alpha_pref_reg = None, # regularization of preference params not supported at this stage yet 
                cruise_speed_kts = self.cruise_speed_kts,
                num_waypoints=num_waypoints
            )
        else:
            raise Exception("At this point, only lin_disent cost model is supported.")
        
        print(f"Cost model version {_cost_model_version} initialized with {num_waypoints} waypoints")
        return cost_model_instance

    def build_cost_model_from_state_dict(
        self,
        cost_model_state: Dict[str, torch.Tensor],
        num_waypoints: int,
        device: torch.device,
        cost_model_version: str = None,
    ) -> torch.nn.Module:
        """Reconstruct a lin_disent cost model from a state_dict using config defaults."""
        _cost_model_version = cost_model_version if cost_model_version is not None else self.cost_model_version
        if _cost_model_version != "lin_disent":
            raise ValueError("Only lin_disent is supported for cost model reconstruction.")

        common_weights = self.common_weights
        if common_weights is None:
            state_weights = cost_model_state.get("common_weights")
            if isinstance(state_weights, torch.Tensor):
                common_weights = tuple(float(v) for v in state_weights.detach().cpu().tolist())
            else:
                common_weights = (0.0, 0.0, 0.0)

        preference_weight = self.preference_weight if self.preference_weight is not None else 1.0

        cost_model_instance = CostLinearDisentangled(
            common_weights=common_weights,
            preference_weights=preference_weight,
            alpha_pref_reg=None,
            cruise_speed_kts=self.cruise_speed_kts,
            num_waypoints=num_waypoints,
            device=device,
        )
        cost_model_instance.load_state_dict(cost_model_state)
        cost_model_instance.to(device)
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
    # Set default values for dependent paths
    config.tres_forward_output_file_name: str = config.file_prefix + "_CLB_WIND"
    config.tres_backward_output_file_name: str = config.file_prefix + "_CLSR_WIND"
    config.thinning_output_file_name: str = config.file_prefix + "_REACHABLE_WIND"
    
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
