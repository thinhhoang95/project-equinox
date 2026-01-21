#!/usr/bin/env python3
"""
Modified Sample1 - Systematic Route Sampling Pipeline
=====================================================

This is a modified version of sample1.py that works with systematic file naming
and configuration management for the automated testing framework.

Key modifications:
- Accepts configuration file path as parameter
- Handles errors gracefully and returns success/failure status
- Uses systematic output directory structure
- Optimized for batch processing
"""

import os
import sys
import torch
import numpy as np
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import all required modules from the original sample1.py
from equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from equinox.dp.trespass.tres_forward import tres_forward, save_transitions
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.dp.trespass.sparse_io_utils import save_sparse_coo_tensor_with_convention, load_sparse_coo_tensor_with_convention
from equinox.config import RunConfiguration
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
from equinox.dp.trespass.tres_backward import tres_backward
from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from equinox.sampling.trespass.sampler_log import sample_tres_trajectory
import networkx as nx

# Import all the functions from the original sample1.py
def forward_tres(config: RunConfiguration, components: dict):
    """Forward dynamic programming pass."""
    print('Running Forward Dynamic Programming (forward_dp_vec2)')

    device = components['device']
    print(f"Using device: {device}")

    G = components['graph']
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    wind_model = components['wind_model']
    performance_model = components['performance_model']
    cost_model_instance = components['cost_model']
    num_actual_nodes = components['num_nodes']
    
    takeoff_time_str = config.takeoff_time_str

    try:
        eta_final, alt_final, phase_final, transitions_list = tres_forward(
            graph=G,
            source_node_id=config.origin_node,
            takeoff_time_str=takeoff_time_str,
            source_elevation_ft=config.source_elevation_ft,
            goal_elevation_ft=config.goal_elevation_ft,
            cost_model=cost_model_instance,
            wind_model=wind_model,
            performance_model=performance_model,
            dist_matrix_np=dist_matrix,
            ac_matrix_np=ac_matrix,
            initial_alt_ft=config.initial_alt_ft,
            delta_t_seconds=config.delta_t_seconds,
            max_flight_duration_hours=config.max_flight_duration_hours,
            etto_delta_t_seconds=config.etto_delta_t_seconds,
            max_elapsed_time_since_takeoff_hours=config.max_elapsed_time_since_takeoff_hours,
            device=device
        )

        print(f'Number of nodes: {num_actual_nodes}')
        print(f'Total number of unique transitions: {len(transitions_list)}')

        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        save_transitions(transitions_list, output_dir, f"{config.file_prefix}_{config.tres_forward_output_file_name}")

        return transitions_list

    except Exception as e:
        print(f"Error in forward tres: {e}")
        import traceback
        traceback.print_exc()
        return None

def backward_tres(config: RunConfiguration, components: dict):
    """Backward dynamic programming pass."""
    forward_pass_output_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.tres_forward_output_file_name}.pkl")
    
    if not os.path.exists(forward_pass_output_path):
        print(f"Forward pass output not found: {forward_pass_output_path}")
        return None
        
    transitions_list = pickle.load(open(forward_pass_output_path, "rb"))
    
    G = components['graph']
    wind_model = components['wind_model']
    performance_model = components['performance_model']
    device = components['device']

    estimated_landing_time_str = config.estimated_landing_time_str
    estimated_takeoff_time_str = config.estimated_takeoff_time_str

    try:
        state_closure_list = tres_backward(
            graph=G,
            goal_node_id=config.goal_node,
            estimated_landing_time_str=estimated_landing_time_str,
            origin_elevation_ft=config.source_elevation_ft,
            destination_elevation_ft=config.goal_elevation_ft,
            wind_model=wind_model,
            performance_model=performance_model,
            transitions_list=transitions_list,
            eta_takeoff_str=estimated_takeoff_time_str,
            final_alt_ft=0.0,
            delta_t_seconds_wall_clock=300,
            delta_t_seconds_climb=30,
            max_flight_duration_hours=config.max_flight_duration_hours,
            climb_phase_switch_allowance_climb_time_bins=config.climb_phase_switch_allowance_climb_time_bins,
            device=device
        )

        save_transitions(state_closure_list, config.output_dir, f"{config.file_prefix}_{config.tres_backward_output_file_name}")
        return state_closure_list
        
    except Exception as e:
        print(f"Error in backward tres: {e}")
        import traceback
        traceback.print_exc()
        return None

def thinning(config: RunConfiguration, components: dict):
    """Thinning pass to reduce state space."""
    backward_pass_output_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.tres_backward_output_file_name}.pkl")
    
    if not os.path.exists(backward_pass_output_path):
        print(f"Backward pass output not found: {backward_pass_output_path}")
        return None
        
    closure_list = pickle.load(open(backward_pass_output_path, "rb"))

    G = components['graph']
    node_to_idx = components['node_to_idx']

    try:
        # Option A: infer max_rho directly from the closure tuples.
        thinned_closures = thin_closures(
            node_to_idx[config.origin_node],
            node_to_idx[config.goal_node],
            None,
            G,
            closure_list,
            wallclock_time_bin_k_tolerance_s=config.delta_t_seconds,
            delta_t_seconds_wall_clock=config.delta_t_seconds,
            include_wait_edges_in_output=True,
        )
        max_rho_val = max(
            max(t[2] for t in closure_list),
            max(t[7] for t in closure_list),
        )
        print(f"Loaded {len(closure_list)} closures from backward tres pass, max rho value: {max_rho_val}")
        print(f"After thinning: {len(thinned_closures)} closures remain from backward tres pass")
        save_transitions(thinned_closures, config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}")
        return thinned_closures
        
    except Exception as e:
        print(f"Error in thinning: {e}")
        import traceback
        traceback.print_exc()
        return None

def amortize_wind_average(config: RunConfiguration, components: dict):
    """Pre-compute wind averages for all transitions."""
    print("Pre-computing wind averages on all transitions")
    wind_model = components['wind_model']

    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    
    if not os.path.exists(thinned_transitions_path):
        print(f"Thinned transitions not found: {thinned_transitions_path}")
        return None
        
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    delta_t_wall_clock_sec = 300.0
    max_flight_duration_hours = config.max_flight_duration_hours
    num_time_bins_wall_clock = int(max_flight_duration_hours * 3600 / delta_t_wall_clock_sec) + 1
    # CRITICAL FIX: The transitions come from tres_backward which uses estimated_landing_time - max_flight_duration
    # as the time reference. We must use the SAME time reference here to calculate correct wind times.
    estimated_landing_time_str = config.estimated_landing_time_str
    estimated_landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str)
    min_wall_clock_time_sec = float(estimated_landing_ssm - max_flight_duration_hours * 3600)

    node_coords_deg = components['node_coords_deg']

    try:
        wind_avg = wind_model.get_average_tailwind_on_edges_knots(
            transitions, node_coords_deg, min_wall_clock_time_sec, 
            delta_t_wall_clock_sec, num_integration_steps=3
        )
        
        wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
        os.makedirs(os.path.dirname(wind_avg_file_path), exist_ok=True)
        torch.save(wind_avg, wind_avg_file_path)
        print(f"Saved wind averages tensor of shape {wind_avg.shape} to {wind_avg_file_path}")
        return wind_avg
        
    except Exception as e:
        print(f"Error in wind averaging: {e}")
        import traceback
        traceback.print_exc()
        return None

def forward_svi(config: RunConfiguration, components: dict):
    """Forward soft value iteration."""
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    G = components['graph']
    idx_to_node = components['idx_to_node']
    cost_model_instance = components['cost_model']
    origin_node_idx = components['origin_node_idx']
    device = components['device']
    num_nodes = components['num_nodes']

    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    
    if not os.path.exists(wind_avg_file_path):
        print(f"Wind averages file not found: {wind_avg_file_path}")
        return None
        
    avg_tailwind_knots_per_transition = torch.load(wind_avg_file_path)
    print(f"Loaded wind averages with shape {avg_tailwind_knots_per_transition.shape}")

    estimated_takeoff_time_str = config.estimated_takeoff_time_str
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    
    if not os.path.exists(thinned_transitions_path):
        print(f"Thinned transitions not found: {thinned_transitions_path}")
        return None
        
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # Determine dimensions from transitions
    max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions)) if transitions else 0
    max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions)) if transitions else 0
    max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions)) if transitions else 0

    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1
    min_wall_clock_time_sec = float(takeoff_ssm)

    print(f"Forward SVI dimensions: nodes={num_nodes}, time_bins={num_time_bins_wall_clock}, "
          f"rho_bins={num_rho_bins}, phases={num_phases}")

    try:
        V_soft = forward_soft_value_iteration(
            state_transitions=transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots_per_transition.to(device),
            G=G,
            idx_to_node=idx_to_node,
            origin_node_idx=origin_node_idx,
            cost_model=cost_model_instance,
            num_nodes=num_nodes,
            num_time_bins_wall_clock=num_time_bins_wall_clock,
            num_rho_bins=num_rho_bins,
            num_phases=num_phases,
            distance_matrix_d=distance_matrix_d,
            airspace_charge_matrix_ac=airspace_charge_matrix_ac,
            device=device,
            verbose=False,
            gamma=config.gamma
        )
        
        print(f"Forward SVI completed. V_soft shape: {V_soft.shape}")
        print(f"Finite values: {torch.isfinite(V_soft).sum().item()}")
        
        # Save sparse tensor
        finite_mask = torch.isfinite(V_soft)
        sparse_indices = finite_mask.nonzero(as_tuple=False).transpose(0, 1)
        sparse_values = V_soft[finite_mask]

        V_soft_sparse = torch.sparse_coo_tensor(
            indices=sparse_indices,
            values=sparse_values,
            size=V_soft.shape,
            dtype=V_soft.dtype,
            device=V_soft.device
        ).coalesce()

        sparse_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_FWD_SPRSE_WIND.pt")
        save_sparse_coo_tensor_with_convention(
            V_soft_sparse, 
            sparse_file_path,
            "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
        )
        
        return V_soft.cpu().numpy()
        
    except Exception as e:
        print(f"Error in forward SVI: {e}")
        import traceback
        traceback.print_exc()
        return None

def backward_svi(config: RunConfiguration, components: dict):
    """Backward soft value iteration."""
    num_nodes = components['num_nodes']
    dist_matrix = components['dist_matrix']
    ac_matrix = components['ac_matrix']
    G = components['graph']
    idx_to_node = components['idx_to_node']
    goal_node_idx = components['goal_node_idx']
    cost_model_instance = components['cost_model']
    device = components['device']

    wind_avg_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_wind_avg.pt")
    
    if not os.path.exists(wind_avg_file_path):
        print(f"Wind averages file not found: {wind_avg_file_path}")
        return None
        
    avg_tailwind_knots_per_transition = torch.load(wind_avg_file_path)

    estimated_takeoff_time_str = config.estimated_takeoff_time_str
    thinned_transitions_path = os.path.join(config.output_dir, f"{config.file_prefix}_{config.thinning_output_file_name}.pkl")
    
    if not os.path.exists(thinned_transitions_path):
        print(f"Thinned transitions not found: {thinned_transitions_path}")
        return None
        
    transitions = pickle.load(open(thinned_transitions_path, "rb"))

    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)

    takeoff_ssm = datestr_to_seconds_since_midnight(estimated_takeoff_time_str)
    
    # Determine dimensions from transitions
    max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions)) if transitions else 0
    max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions)) if transitions else 0
    max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions)) if transitions else 0

    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1
    min_wall_clock_time_sec = float(takeoff_ssm)

    print(f"Backward SVI dimensions: nodes={num_nodes}, time_bins={num_time_bins_wall_clock}, "
          f"rho_bins={num_rho_bins}, phases={num_phases}")

    try:
        V_soft_bwd, edge_costs = backward_soft_value_iteration(
            state_transitions=transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots_per_transition.to(device),
            G=G,
            idx_to_node=idx_to_node,
            goal_node_idx=goal_node_idx,
            cost_model=cost_model_instance,
            num_nodes=num_nodes,
            num_time_bins_wall_clock=num_time_bins_wall_clock,
            num_rho_bins=num_rho_bins,
            num_phases=num_phases,
            distance_matrix_d=distance_matrix_d,
            airspace_charge_matrix_ac=airspace_charge_matrix_ac,
            device=device,
            verbose=False,
            gamma=config.gamma
        )
        
        print(f"Backward SVI completed. V_soft_bwd shape: {V_soft_bwd.shape}")
        print(f"Finite values: {torch.isfinite(V_soft_bwd).sum().item()}")
        
        # Save sparse tensors
        finite_mask = torch.isfinite(V_soft_bwd)
        sparse_indices = finite_mask.nonzero(as_tuple=False).transpose(0, 1)
        sparse_values = V_soft_bwd[finite_mask]

        V_soft_bwd_sparse = torch.sparse_coo_tensor(
            indices=sparse_indices,
            values=sparse_values,
            size=V_soft_bwd.shape,
            dtype=V_soft_bwd.dtype,
            device=V_soft_bwd.device
        ).coalesce()

        sparse_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_BWD_SPRSE_WIND.pt")
        save_sparse_coo_tensor_with_convention(
            V_soft_bwd_sparse, 
            sparse_file_path,
            "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
        )
        
        edge_costs_path = os.path.join(config.output_dir, f"{config.file_prefix}_COST_WIND.pt")
        save_sparse_coo_tensor_with_convention(edge_costs, edge_costs_path)
        
        return V_soft_bwd.cpu().numpy()
        
    except Exception as e:
        print(f"Error in backward SVI: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tres_sampler(config: RunConfiguration, components: dict):
    """Sample trajectories using the TRes sampler."""
    print("Running TRes Sampler")
    device = components['device']

    G = components['graph']
    node_to_idx = components['node_to_idx']
    idx_to_node = components['idx_to_node']

    # Load backward value function
    v_bwd_sparse_path = os.path.join(config.output_dir, f"{config.file_prefix}_V_BWD_SPRSE_WIND.pt")
    
    if not os.path.exists(v_bwd_sparse_path):
        print(f"Backward value function not found: {v_bwd_sparse_path}")
        return None
        
    try:
        V_soft_bwd_sparse, _ = load_sparse_coo_tensor_with_convention(v_bwd_sparse_path, target_device=device)
        V_soft_bwd_sparse_coalesced = V_soft_bwd_sparse.coalesce()
        
        # Convert to dense with inf fill
        V_soft_bwd_dense_filled = torch.full(
            V_soft_bwd_sparse_coalesced.shape,
            float('inf'),
            dtype=V_soft_bwd_sparse_coalesced.dtype,
            device=V_soft_bwd_sparse_coalesced.device
        )

        indices = V_soft_bwd_sparse_coalesced.indices()
        values = V_soft_bwd_sparse_coalesced.values()

        if values.numel() > 0:
            V_soft_bwd_dense_filled[tuple(indices)] = values
            
    except Exception as e:
        print(f"Error loading backward value function: {e}")
        return None

    # Load edge costs
    edge_costs_path = os.path.join(config.output_dir, f"{config.file_prefix}_COST_WIND.pt")
    
    if not os.path.exists(edge_costs_path):
        print(f"Edge costs not found: {edge_costs_path}")
        return None
        
    try:
        edge_costs_tensor, _ = load_sparse_coo_tensor_with_convention(edge_costs_path, target_device=device)
    except Exception as e:
        print(f"Error loading edge costs: {e}")
        return None

    origin_node_id = config.origin_node
    goal_node_id = config.goal_node

    # Initial state parameters
    initial_k = 0
    initial_rho = V_soft_bwd_dense_filled.shape[2] - 1 if V_soft_bwd_dense_filled.shape[2] > 0 else 0
    initial_phase = 0

    print(f"Sampling from {origin_node_id} to {goal_node_id}")
    print(f"Initial state: k={initial_k}, rho={initial_rho}, phase={initial_phase}")

    num_samples = 100
    successful_samples = 0
    trajectories = []
    all_trajectory_costs = []

    try:
        for i in range(num_samples):
            trajectory, trajectory_costs = sample_tres_trajectory(
                G=G,
                node_to_idx=node_to_idx,
                idx_to_node=idx_to_node,
                origin_node_id=origin_node_id,
                goal_node_id=goal_node_id,
                initial_rho=initial_rho,
                initial_phase=initial_phase,
                soft_cost_to_go=V_soft_bwd_dense_filled,
                edge_costs_uv=edge_costs_tensor,
                max_steps=200,
                gamma=config.gamma
            )
            
            if trajectory:
                successful_samples += 1
                trajectories.append(trajectory)
                all_trajectory_costs.append(trajectory_costs)
                
                if i % 20 == 0:
                    print(f"Sampled {i+1}/{num_samples} trajectories...")

        print(f"Sampling completed: {successful_samples}/{num_samples} successful")

        if trajectories:
            # Save trajectories
            trajectories_file_path = os.path.join(config.output_dir, f"{config.file_prefix}_CLB_trajectories.txt")
            
            with open(trajectories_file_path, "w") as f:
                for i, trajectory in enumerate(trajectories):
                    waypoint_names = [str(step[0]) for step in trajectory]
                    trajectory_line = " ".join(waypoint_names)
                    total_cost = np.sum(all_trajectory_costs[i])
                    f.write(f"{total_cost},{trajectory_line}\n")
            
            print(f"Saved {len(trajectories)} trajectories to {trajectories_file_path}")
            return trajectories
        else:
            print("No successful trajectories generated")
            return None
            
    except Exception as e:
        print(f"Error in trajectory sampling: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_full_pipeline(config_file_path: str) -> bool:
    """Run the complete pipeline for a single flight configuration."""
    try:
        print(f"Loading configuration from: {config_file_path}")
        config = RunConfiguration.load_from_yaml(config_file_path)
        
        print(f"Processing flight: {config.file_prefix}")
        print(f"Route: {config.origin_node} -> {config.goal_node}")
        print(f"Output directory: {config.output_dir}")
        
        # Initialize components
        components = config.initialize_all_components(manual_cost_model_init=True)
        
        # Manual wind model initialization (as in original)
        components['wind_model'] = WindDate(date_str=config.wind_date, data_dir=config.wind_data_dir)
        
        # Initialize cost model
        cost_model = config.initialize_cost_model(num_waypoints=components['num_nodes'])
        components['cost_model'] = cost_model
        
        print("Running pipeline stages...")
        
        # Stage 1: Forward tres
        print("Stage 1/7: Forward tres")
        if not forward_tres(config, components):
            print("Forward tres failed")
            return False
            
        # Stage 2: Backward tres  
        print("Stage 2/7: Backward tres")
        if not backward_tres(config, components):
            print("Backward tres failed")
            return False
            
        # Stage 3: Thinning
        print("Stage 3/7: Thinning")
        if not thinning(config, components):
            print("Thinning failed")
            return False
            
        # Stage 4: Wind averaging
        print("Stage 4/7: Wind averaging")
        wind_avg_result = amortize_wind_average(config, components)
        if wind_avg_result is None:
            print("Wind averaging failed")
            return False
            
        # Stage 5: Forward SVI
        print("Stage 5/7: Forward SVI")
        fwd_svi_result = forward_svi(config, components)
        if fwd_svi_result is None:
            print("Forward SVI failed")
            return False
            
        # Stage 6: Backward SVI
        print("Stage 6/7: Backward SVI")
        bwd_svi_result = backward_svi(config, components)
        if bwd_svi_result is None:
            print("Backward SVI failed")
            return False
            
        # Stage 7: Trajectory sampling
        print("Stage 7/7: Trajectory sampling")
        if not test_tres_sampler(config, components):
            print("Trajectory sampling failed")
            return False
            
        print(f"Pipeline completed successfully for flight {config.file_prefix}")
        return True
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sample1_modified.py <config_file_path>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
        
    success = run_full_pipeline(config_file)
    sys.exit(0 if success else 1)
