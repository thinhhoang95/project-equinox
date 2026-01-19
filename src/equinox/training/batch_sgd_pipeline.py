# To run: python src/equinox/training/batch_sgd_pipeline.py --case-dir data/cases/LEMD_EGLL --batch-size 5 --learning-rate 1e-6
"""
Batch Stochastic Gradient Descent Pipeline for Maximum Entropy Inverse Learning

This module implements a data pipeline for batch stochastic gradient descent for Maximum Entropy 
Inverse Learning of airline routing preferences. The pipeline processes multiple flights in parallel,
computes gradients for each flight, and updates the cost model parameters using batch SGD.

Key Features:
- Loads pre-computed TRES results for multiple flights
- Processes flights through thinning, forward/backward SVI, and gradient computation
- Implements batch SGD with gradient queuing and model updates
- Supports multiprocessing for parallel SVI computation
- Includes convergence checking and checkpointing
- Handles empirical count computation from actual flight routes

The pipeline follows these steps for each flight:
1. Load TRES forward/backward results
2. Perform thinning to get reachable states
3. Run forward and backward soft value iteration (SVI) in parallel
4. Compute gradients using Maximum Entropy Inverse Learning
5. Queue gradients and apply batch updates to the cost model

Usage:
    python batch_sgd_pipeline.py --case-dir data/cases/LEMD_EGLL --batch-size 5 --learning-rate 1e-6
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import yaml

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.equinox.config import RunConfiguration
from src.equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass
from src.equinox.wind.batch_wind_model import get_flight_batches

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_sgd_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FlightGradientResult:
    """Result of gradient computation for a single flight."""
    flight_id: str
    takeoff_timestamp: int
    gradient: torch.Tensor
    empirical_counts: torch.Tensor
    expected_counts: torch.Tensor
    log_likelihood: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchLearningConfig:
    """Configuration for batch learning parameters."""
    batch_size: int = 5
    learning_rate: float = 1e-6
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    gradient_queue_size: int = 1
    checkpoint_interval: int = 10
    num_workers: int = None
    device: str = "cuda"
    gamma: float = 1.0
    debug_single_process: bool = False
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)


def load_flight_tres_results(case_dir: str, flight_id: str, takeoff_timestamp: int, 
                            components: Dict[str, Any]) -> Tuple[List, List, List]:
    """
    Load pre-computed TRES results for a specific flight.
    
    Args:
        case_dir: Directory containing the case data
        flight_id: Flight identifier
        takeoff_timestamp: Takeoff timestamp
        components: Shared components containing graph and node mappings
        
    Returns:
        Tuple of (forward_transitions, backward_transitions, thinned_transitions)
    """
    tres_dir = Path(case_dir) / "tres_runs"
    
    # Find the batch directory containing this flight
    forward_transitions = None
    backward_transitions = None
    thinned_transitions = None
    
    for batch_dir in tres_dir.glob("batch*"):
        fw_file = batch_dir / f"FW_{flight_id}_{takeoff_timestamp}.pkl"
        bw_file = batch_dir / f"BW_{flight_id}_{takeoff_timestamp}.pkl"
        
        if fw_file.exists() and bw_file.exists():
            try:
                with open(fw_file, 'rb') as f:
                    forward_transitions = pickle.load(f)
                with open(bw_file, 'rb') as f:
                    backward_transitions = pickle.load(f)
                
                # Check for pre-computed thinned transitions
                thinned_file = batch_dir / f"CLSR_{flight_id}_{takeoff_timestamp}.pkl"
                if thinned_file.exists():
                    with open(thinned_file, 'rb') as f:
                        thinned_transitions = pickle.load(f)
                else:
                    # Compute thinning on the fly using the correct function
                    logger.info(f"Computing thinning for flight {flight_id}")
                    
                    # Get flight-specific origin and destination from CSV data
                    flights_csv = os.path.join(case_dir, "all_routes.csv")
                    flights_df = pd.read_csv(flights_csv)
                    if "takeoff_time" in flights_df.columns:
                        takeoff_col = "takeoff_time"
                    elif "takeoff" in flights_df.columns:
                        takeoff_col = "takeoff"
                    else:
                        raise ValueError("No takeoff_time/takeoff column found in all_routes.csv")

                    flight_row = flights_df[
                        (flights_df["flight_id"] == flight_id)
                        & (flights_df[takeoff_col] == takeoff_timestamp)
                    ]
                    
                    if flight_row.empty:
                        raise ValueError(
                            f"Flight {flight_id} with takeoff {takeoff_timestamp} not found in all_routes.csv"
                        )
                    
                    origin_node = flight_row.iloc[0]['origin']
                    destination_node = flight_row.iloc[0]['destination']
                    
                    source_node_idx = components['node_to_idx'][origin_node]
                    goal_node_idx = components['node_to_idx'][destination_node]
                    # Use thin_closures with correct parameters
                    from src.equinox.dp.trespass.thinning import thin_closures
                    thinned_transitions = thin_closures(
                        # Option A: infer max_rho directly from the closure tuples.
                        source_node_idx, goal_node_idx, None,
                        components['graph'], backward_transitions
                    )
                    
                    # Save for future use
                    with open(thinned_file, 'wb') as f:
                        pickle.dump(thinned_transitions, f)
                
                break
                
            except Exception as e:
                logger.error(f"Error loading TRES results for flight {flight_id}: {e}")
                continue
    
    if forward_transitions is None or backward_transitions is None:
        raise FileNotFoundError(f"TRES results not found for flight {flight_id}_{takeoff_timestamp}")
    
    return forward_transitions, backward_transitions, thinned_transitions


def compute_empirical_counts_for_flight(flight_data: pd.Series, node_to_idx: Dict[str, int], 
                                       num_nodes: int) -> torch.Tensor:
    """
    Compute empirical counts (actual route usage) for a single flight.
    
    Args:
        flight_data: Flight data from the CSV
        node_to_idx: Mapping from waypoint names to indices
        num_nodes: Total number of waypoints
        
    Returns:
        Empirical counts tensor of shape (num_nodes, num_nodes)
    """
    empirical_counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    
    # Parse the actual waypoints from the flight data
    waypoints_str = flight_data['real_waypoints']
    waypoints = waypoints_str.split()
    
    # Count traversals between consecutive waypoints
    for i in range(len(waypoints) - 1):
        from_wp = waypoints[i]
        to_wp = waypoints[i + 1]
        
        if from_wp in node_to_idx and to_wp in node_to_idx:
            from_idx = node_to_idx[from_wp]
            to_idx = node_to_idx[to_wp]
            empirical_counts[from_idx, to_idx] += 1.0
    
    return empirical_counts


def _run_forward_svi_wrapper(args):
    """Wrapper function for multiprocessing forward SVI."""
    (state_transitions, avg_tailwind_knots, G, idx_to_node, origin_node_idx, 
     cost_model_state, num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases, 
     distance_matrix_d, airspace_charge_matrix_ac, device_str, gamma) = args
    
    device = torch.device(device_str)
    
    # Reconstruct cost model from state dict
    from src.equinox.cost.cost_rev2_reg import CostRev2
    cost_model = CostRev2(
        beta0=0.0, beta1=1e-2, beta2=0.0, beta3=1.0,
        num_waypoints=num_nodes, alpha_pref_reg=1.0, device=device
    )
    cost_model.load_state_dict(cost_model_state)
    cost_model.to(device)

    # Convert np.arrays to tensors if needed
    if isinstance(avg_tailwind_knots, np.ndarray):
        avg_tailwind_knots = torch.from_numpy(avg_tailwind_knots)
    if isinstance(distance_matrix_d, np.ndarray):
        distance_matrix_d = torch.from_numpy(distance_matrix_d)
    if isinstance(airspace_charge_matrix_ac, np.ndarray):
        airspace_charge_matrix_ac = torch.from_numpy(airspace_charge_matrix_ac)
    
    # Move tensors to device
    avg_tailwind_knots = avg_tailwind_knots.to(device)
    distance_matrix_d = distance_matrix_d.to(device)   
    airspace_charge_matrix_ac = airspace_charge_matrix_ac.to(device)
    
    v_f = forward_soft_value_iteration(
        state_transitions=state_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=G,
        idx_to_node=idx_to_node,
        origin_node_idx=origin_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        gamma=gamma,
        verbose=False
    )
    
    return v_f.cpu()


def _run_backward_svi_wrapper(args):
    """Wrapper function for multiprocessing backward SVI."""
    (state_transitions, avg_tailwind_knots, G, idx_to_node, goal_node_idx, 
     cost_model_state, num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases, 
     distance_matrix_d, airspace_charge_matrix_ac, device_str, gamma) = args
    
    device = torch.device(device_str)
    
    # Reconstruct cost model from state dict
    from src.equinox.cost.cost_rev2_reg import CostRev2
    cost_model = CostRev2(
        beta0=0.0, beta1=1e-2, beta2=0.0, beta3=1.0,
        num_waypoints=num_nodes, alpha_pref_reg=1.0, device=device
    )
    cost_model.load_state_dict(cost_model_state)
    cost_model.to(device)

    # Convert np.arrays to tensors if needed
    if isinstance(avg_tailwind_knots, np.ndarray):
        avg_tailwind_knots = torch.from_numpy(avg_tailwind_knots)
    if isinstance(distance_matrix_d, np.ndarray):
        distance_matrix_d = torch.from_numpy(distance_matrix_d)
    if isinstance(airspace_charge_matrix_ac, np.ndarray):
        airspace_charge_matrix_ac = torch.from_numpy(airspace_charge_matrix_ac)
    
    # Move tensors to device
    avg_tailwind_knots = avg_tailwind_knots.to(device)
    distance_matrix_d = distance_matrix_d.to(device)
    airspace_charge_matrix_ac = airspace_charge_matrix_ac.to(device)
    
    v_b, _ = backward_soft_value_iteration(
        state_transitions=state_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=G,
        idx_to_node=idx_to_node,
        goal_node_idx=goal_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        gamma=gamma,
        verbose=False
    )
    
    return v_b.cpu()


def process_single_flight(flight_data: pd.Series, config: RunConfiguration, 
                         components: Dict[str, Any], batch_config: BatchLearningConfig,
                         case_dir: str) -> FlightGradientResult:
    """
    Process a single flight through the complete pipeline.
    
    Args:
        flight_data: Flight data from CSV
        config: Run configuration
        components: Shared components (graph, models, etc.)
        batch_config: Batch learning configuration
        case_dir: Case directory path
        
    Returns:
        FlightGradientResult containing gradients and metadata
    """
    start_time = time.time()
    flight_id = flight_data['flight_id']
    takeoff_timestamp = int(flight_data['takeoff'])
    
    try:
        # 1. Load TRES results
        logger.info(f"Processing flight {flight_id}")
        forward_transitions, backward_transitions, thinned_transitions = load_flight_tres_results(
            case_dir, flight_id, takeoff_timestamp, components
        )
        
        if not thinned_transitions:
            return FlightGradientResult(
                flight_id=flight_id,
                takeoff_timestamp=takeoff_timestamp,
                gradient=torch.zeros(1),
                empirical_counts=torch.zeros(1),
                expected_counts=torch.zeros(1),
                log_likelihood=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message="No thinned transitions available"
            )
        
        # 2. Derive state space dimensions
        max_k_val = max(max(t[1] for t in thinned_transitions), max(t[6] for t in thinned_transitions))
        max_rho_val = max(max(t[2] for t in thinned_transitions), max(t[7] for t in thinned_transitions))
        max_phase_val = max(max(t[4] for t in thinned_transitions), max(t[9] for t in thinned_transitions))
        
        num_time_bins_wall_clock = max_k_val + 1
        num_rho_bins = max_rho_val + 1
        num_phases = max_phase_val + 1
        num_nodes = components['num_nodes']
        
        # 3. Prepare wind data (simplified - using average wind)
        avg_tailwind_knots = torch.zeros(len(thinned_transitions), dtype=torch.float32)
        logger.debug(f"Flight {flight_id}: Using {len(thinned_transitions)} thinned transitions")
        
        # 4. Get origin and goal indices for this flight
        origin_node_idx = components['node_to_idx'][flight_data['origin']]
        goal_node_idx = components['node_to_idx'][flight_data['destination']]
        
        # 5. Prepare arguments for parallel SVI
        device_str = str(components['device'])
        cost_model_state = components['cost_model'].state_dict()
        
        forward_args = (
            thinned_transitions, avg_tailwind_knots, components['graph'], 
            components['idx_to_node'], origin_node_idx, cost_model_state,
            num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases,
            components['dist_matrix'], components['ac_matrix'], device_str, batch_config.gamma
        )
        
        backward_args = (
            thinned_transitions, avg_tailwind_knots, components['graph'], 
            components['idx_to_node'], goal_node_idx, cost_model_state,
            num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases,
            components['dist_matrix'], components['ac_matrix'], device_str, batch_config.gamma
        )
        
        # 6. Run forward and backward SVI
        if batch_config.debug_single_process:
            logger.info(f"Running SVI sequentially for flight {flight_id} (debug mode)")
            v_f = _run_forward_svi_wrapper(forward_args)
            v_b = _run_backward_svi_wrapper(backward_args)
        else:
            with ProcessPoolExecutor(max_workers=2) as executor:
                future_forward = executor.submit(_run_forward_svi_wrapper, forward_args)
                future_backward = executor.submit(_run_backward_svi_wrapper, backward_args)
                
                v_f = future_forward.result()
                v_b = future_backward.result()
        
        
        # 7. Move results to device and compute empirical counts
        device = components['device']
        v_f = v_f.to(device)
        v_b = v_b.to(device)
        
        empirical_counts = compute_empirical_counts_for_flight(
            flight_data, components['node_to_idx'], num_nodes
        ).to(device)
        
        # 8. Compute gradients using backward gradient pass
        expected_counts, gradient = backward_gradient_pass(
            state_transitions=thinned_transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots.to(device),
            V_f=v_f,
            V_b=v_b,
            cost_model=components['cost_model'],
            empirical_counts=empirical_counts,
            origin_node_idx=origin_node_idx,
            num_nodes=num_nodes,
            distance_matrix_d=components['dist_matrix'],
            airspace_charge_matrix_ac=components['ac_matrix'],
            device=device,
            gamma=batch_config.gamma,
            verbose=False
        )

        # For debugging, save the expected counts and the soft value functions to a file to be inspected externally
        if batch_config.debug_single_process:
            import pickle
            with open(f"expected_counts_{flight_id}.pkl", "wb") as f:
                pickle.dump(expected_counts.cpu().numpy(), f)
            with open(f"soft_value_functions_f_{flight_id}.pkl", "wb") as f:
                pickle.dump(v_f.cpu().numpy(), f)
            with open(f"soft_value_functions_b_{flight_id}.pkl", "wb") as f:
                pickle.dump(v_b.cpu().numpy(), f)

        # raise Exception("Stop here")

        # 9. Compute log likelihood (simplified)
        log_likelihood = -torch.sum((empirical_counts - expected_counts) ** 2).item()
        
        # Validate gradient dimensions
        expected_grad_size = sum(p.numel() for p in components['cost_model'].parameters() if p.requires_grad)
        if gradient.numel() != expected_grad_size:
            logger.warning(f"Gradient size mismatch for flight {flight_id}: expected {expected_grad_size}, got {gradient.numel()}")
        
        processing_time = time.time() - start_time
        
        return FlightGradientResult(
            flight_id=flight_id,
            takeoff_timestamp=takeoff_timestamp,
            gradient=gradient.cpu(),
            empirical_counts=empirical_counts.cpu(),
            expected_counts=expected_counts.cpu(),
            log_likelihood=log_likelihood,
            processing_time=processing_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing flight {flight_id}: {e}")
        raise Exception(f"Error processing flight {flight_id}: {e}")
        return FlightGradientResult(
            flight_id=flight_id,
            takeoff_timestamp=takeoff_timestamp,
            gradient=torch.zeros(1),
            empirical_counts=torch.zeros(1),
            expected_counts=torch.zeros(1),
            log_likelihood=0.0,
            processing_time=time.time() - start_time,
            success=False,
            error_message=str(e)
        )


def run_batch_sgd_pipeline(case_dir: str, config_path: str, batch_config: BatchLearningConfig,
                          output_dir: str = None) -> Dict[str, Any]:
    """
    Main function to run the batch SGD pipeline for Maximum Entropy Inverse Learning.
    
    Args:
        case_dir: Directory containing case data (flights, TRES results, etc.)
        config_path: Path to the configuration YAML file
        batch_config: Batch learning configuration
        output_dir: Directory to save results and checkpoints
        
    Returns:
        Dictionary containing training results and statistics
    """
    logger.info("Starting Batch SGD Pipeline for Maximum Entropy Inverse Learning")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(case_dir, "batch_sgd_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load configuration and initialize components
    logger.info(f"Loading configuration from {config_path}")
    config = RunConfiguration.load_from_yaml(config_path)
    components = config.initialize_all_components()
    
    device = torch.device(batch_config.device if torch.cuda.is_available() else "cpu")
    components['device'] = device
    components['cost_model'] = components['cost_model'].to(device)
    
    # Ensure cost model parameters require gradients
    for param in components['cost_model'].parameters():
        param.requires_grad = True
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, components['cost_model'].parameters()),
        lr=batch_config.learning_rate
    )
    
    logger.info(f"Using device: {device}")
    num_trainable_params = sum(p.numel() for p in components['cost_model'].parameters() if p.requires_grad)
    logger.info(f"Cost model has {num_trainable_params} trainable parameters")
    
    # 2. Load flight data and create batches
    flights_csv = os.path.join(case_dir, "all_routes.csv")
    logger.info(f"Loading flights from {flights_csv} and creating batches...")
    flight_batches = get_flight_batches(flights_csv, batch_config.batch_size)
    num_batches = len(flight_batches)
    if num_batches == 0:
        logger.error("No flight batches were created. Please check the routes file and batch size.")
        return
    logger.info(f"Created {num_batches} batches of flights.")
    
    # 3. Initialize tracking variables
    gradient_queue = []
    iteration = 0
    converged = False
    training_history = {
        'iterations': [],
        'avg_log_likelihood': [],
        'gradient_norms': [],
        'processing_times': [],
        'successful_flights': [],
        'failed_flights': []
    }
    
    # 4. Main training loop
    logger.info("Starting main training loop")
    
    while iteration < batch_config.max_iterations and not converged:
        iteration += 1
        iteration_start_time = time.time()
        
        logger.info(f"\n--- Iteration {iteration}/{batch_config.max_iterations} ---")
        
        # Get the next batch of flights sequentially, cycling through the batches
        batch_idx = (iteration - 1) % num_batches
        batch_flights = flight_batches[batch_idx]
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_flights)} flights")
        
        # Process flights in the batch
        batch_results = []
        successful_flights = 0
        failed_flights = 0
        
        for _, flight_data in batch_flights.iterrows():
            result = process_single_flight(
                flight_data, config, components, batch_config, case_dir
            )
            batch_results.append(result)
            
            if result.success:
                successful_flights += 1
                gradient_queue.append(result.gradient)
            else:
                failed_flights += 1
                logger.warning(f"Flight {result.flight_id} failed: {result.error_message}")
        
        # Apply gradient updates when queue is full or at end of iteration
        if len(gradient_queue) >= batch_config.gradient_queue_size or iteration == batch_config.max_iterations:
            if gradient_queue:
                # Average gradients in the queue
                avg_gradient = (
                    torch.stack(gradient_queue)
                         .mean(dim=0)
                         .to(device=device, dtype=torch.float32)
                )
                # gradient_norm = torch.norm(avg_gradient).item()
                gradient_norm = torch.max(torch.abs(avg_gradient)).item()
                
                # Apply gradient update
                optimizer.zero_grad()
                
                # Manually set gradients
                param_idx = 0
                for param in components['cost_model'].parameters():
                    if param.requires_grad:
                        param_size = param.numel()
                        param.grad = (
                            avg_gradient[param_idx: param_idx + param_size]
                            .view(param.shape)
                            .to(dtype=param.dtype)
                        )
                        param_idx += param_size
                
                optimizer.step()
                
                logger.info(f"Applied gradient update with norm L_inf: {gradient_norm:.6f}")
                
                # Check convergence
                if gradient_norm < batch_config.convergence_threshold:
                    logger.info(f"Convergence achieved! Gradient norm L_inf: {gradient_norm:.6f} < {batch_config.convergence_threshold}")
                    converged = True
                
                # Clear gradient queue
                gradient_queue.clear()
            else:
                gradient_norm = 0.0
        else:
            gradient_norm = 0.0
        
        # Compute iteration statistics
        avg_log_likelihood = np.mean([r.log_likelihood for r in batch_results if r.success])
        iteration_time = time.time() - iteration_start_time
        
        # Update training history
        training_history['iterations'].append(iteration)
        training_history['avg_log_likelihood'].append(avg_log_likelihood)
        training_history['gradient_norms'].append(gradient_norm)
        training_history['processing_times'].append(iteration_time)
        training_history['successful_flights'].append(successful_flights)
        training_history['failed_flights'].append(failed_flights)
        
        logger.info(f"Iteration {iteration} completed:")
        logger.info(f"  - Successful flights: {successful_flights}/{len(batch_flights)}")
        logger.info(f"  - Average log likelihood: {avg_log_likelihood:.6f}")
        logger.info(f"  - Gradient norm: {gradient_norm:.6f}")
        logger.info(f"  - Processing time: {iteration_time:.2f}s")
        
        # Save checkpoint
        if iteration % batch_config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_iter_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'model_state_dict': components['cost_model'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': training_history,
                'batch_config': asdict(batch_config)
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # 5. Save final results
    final_results = {
        'converged': converged,
        'final_iteration': iteration,
        'training_history': training_history,
        'final_model_state': components['cost_model'].state_dict(),
        'batch_config': asdict(batch_config)
    }
    
    results_path = os.path.join(output_dir, "final_results.pt")
    torch.save(final_results, results_path)
    
    # Save training history as CSV for analysis
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    logger.info(f"Training completed after {iteration} iterations")
    logger.info(f"Final results saved to {results_path}")
    
    return final_results


def validate_implementation():
    """
    Validate the implementation by checking key components and dependencies.
    """
    logger.info("Validating batch SGD pipeline implementation...")
    
    # Check imports
    try:
        from src.equinox.dp.trespass.thinning import thin_closures
        from src.equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass
        logger.info("✓ All required imports are available")
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    
    # Check function signatures
    import inspect
    
    # Check thin_closures signature
    sig = inspect.signature(thin_closures)
    expected_params = ['source_node_idx', 'goal_node_idx', 'max_rho', 'G', 'closures']
    actual_params = list(sig.parameters.keys())
    if actual_params == expected_params:
        logger.info("✓ thin_closures function signature is correct")
    else:
        logger.error(f"✗ thin_closures signature mismatch. Expected: {expected_params}, Got: {actual_params}")
        return False
    
    # Check backward_gradient_pass return type
    sig = inspect.signature(backward_gradient_pass)
    logger.info("✓ backward_gradient_pass function is available")
    
    logger.info("✓ Implementation validation passed")
    return True


def main():
    """Command-line interface for the batch SGD pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch SGD Pipeline for Maximum Entropy Inverse Learning"
    )
    
    parser.add_argument(
        "--case-dir", 
        required=True,
        help="Directory containing case data (e.g., data/cases/LEMD_EGLL)"
    )
    parser.add_argument(
        "--config", 
        help="Path to configuration YAML file (default: case-dir/default.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: case-dir/batch_sgd_results)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="Number of flights per batch (default: 5)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-6,
        help="Learning rate for SGD (default: 1e-6)"
    )
    parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=100,
        help="Maximum number of iterations (default: 100)"
    )
    parser.add_argument(
        "--convergence-threshold", 
        type=float, 
        default=1e-4,
        help="Convergence threshold for gradient norm (default: 1e-4)"
    )
    parser.add_argument(
        "--gradient-queue-size", 
        type=int, 
        default=1,
        help="Size of gradient queue before applying updates (default: 10)"
    )
    parser.add_argument(
        "--checkpoint-interval", 
        type=int, 
        default=10,
        help="Interval for saving checkpoints (default: 10)"
    )
    parser.add_argument(
        "--num-workers", 
        type=int,
        help="Number of worker processes (default: CPU count - 1)"
    )
    parser.add_argument(
        "--device", 
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=1.0,
        help="Temperature parameter (default: 1.0)"
    )
    parser.add_argument(
        "--debug-single-process",
        action="store_true",
        help="Run in a single process for debugging purposes."
    )
    
    args = parser.parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = os.path.join(args.case_dir, "default.yaml")
    
    # Create batch configuration
    batch_config = BatchLearningConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        gradient_queue_size=args.gradient_queue_size,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        device=args.device,
        gamma=args.gamma,
        debug_single_process=args.debug_single_process
    )
    
    # Validate implementation first
    if not validate_implementation():
        logger.error("Implementation validation failed. Exiting.")
        return
    
    # Run the pipeline
    try:
        results = run_batch_sgd_pipeline(
            case_dir=args.case_dir,
            config_path=args.config,
            batch_config=batch_config,
            output_dir=args.output_dir
        )
        
        logger.info("Pipeline completed successfully!")
        if results['converged']:
            logger.info("Model converged!")
        else:
            logger.info("Model did not converge within the maximum iterations.")
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 
