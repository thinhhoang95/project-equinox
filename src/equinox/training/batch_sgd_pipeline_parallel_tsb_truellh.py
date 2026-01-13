# To run: python src/equinox/training/batch_sgd_pipeline_parallel.py --case-dir data/cases/LEMD_EGLL --batch-size 5 --learning-rate 1e-6
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
- Comprehensive TensorBoard logging for training monitoring and visualization

The pipeline follows these steps for each flight:
1. Load TRES forward/backward results
2. Perform thinning to get reachable states
3. Run forward and backward soft value iteration (SVI) in parallel
4. Compute gradients using Maximum Entropy Inverse Learning
5. Queue gradients and apply batch updates to the cost model

Usage:
    python batch_sgd_pipeline_parallel_w_tensorboard.py --case-dir data/cases/LEMD_EGLL --batch-size 5 --learning-rate 1e-6
    
    To monitor training with TensorBoard:
    tensorboard --logdir data/cases/LEMD_EGLL/batch_sgd_results/tensorboard_logs
    
    To start fresh training without resuming from checkpoint:
    python batch_sgd_pipeline_parallel_w_tensorboard.py --case-dir data/cases/LEMD_EGLL --no-resume
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
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Mapping
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import yaml
from collections import defaultdict
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.equinox.config import RunConfiguration, get_cost_model_class
from src.equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass
from src.equinox.wind.batch_wind_model import get_flight_batches
from src.equinox.preferences.disentanglement import (
    build_edge_list,
    build_feature_matrix,
    compute_empirical_counts_from_routes,
    d_weighted_normalize_features,
    PreferenceProjector,
)

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

_WORKER_CONTEXT: Dict[str, Any] = {}


def _init_worker(config_path: str, case_dir: str, device_str: str, gamma: float, debug_single_process: bool) -> None:
    """Initialize per-worker shared context to avoid pickling large objects per task."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    config = RunConfiguration.load_from_yaml(config_path)
    graph, node_to_idx, idx_to_node, _ = config.load_graph()
    dist_matrix = config.load_distance_matrix()
    ac_matrix = config.load_charges_matrix()

    device = torch.device(device_str)
    if isinstance(dist_matrix, np.ndarray):
        dist_matrix = torch.from_numpy(dist_matrix)
    if isinstance(ac_matrix, np.ndarray):
        ac_matrix = torch.from_numpy(ac_matrix)

    dist_matrix = dist_matrix.to(device)
    ac_matrix = ac_matrix.to(device)

    pref_enabled = config.cost_model_version == "lin_disent"
    edge_u = None
    edge_v = None
    if pref_enabled:
        edge_u, edge_v = build_edge_list(graph, node_to_idx)
        edge_u = edge_u.to(device)
        edge_v = edge_v.to(device)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "graph": graph,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "dist_matrix": dist_matrix,
        "ac_matrix": ac_matrix,
        "num_nodes": len(graph.nodes()),
        "device": device,
        "case_dir": case_dir,
        "cost_model_version": config.cost_model_version,
        "gamma": gamma,
        "debug_single_process": debug_single_process,
        "pref_enabled": pref_enabled,
        "edge_u": edge_u,
        "edge_v": edge_v,
    }


def _get_worker_context() -> Dict[str, Any]:
    if not _WORKER_CONTEXT:
        raise RuntimeError("Worker context is not initialized. Pass _init_worker to ProcessPoolExecutor.")
    return _WORKER_CONTEXT


@dataclass
class FlightGradientResult:
    """Result of gradient computation for a single flight."""
    flight_id: str
    takeoff_timestamp: int
    # Per-parameter gradients keyed by parameter name. Using names avoids any reliance on
    # flattened-vector ordering, which can silently break if parameter registration order changes.
    gradient: Dict[str, torch.Tensor]
    pref_grad_e: Optional[torch.Tensor]
    log_likelihood: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchLearningConfig:
    """Configuration for batch learning parameters."""
    batch_size: int = 5
    learning_rate: float = 1e-6
    pref_learning_rate: float = 1e-2
    pref_projection_ridge: float = 1e-8
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    checkpoint_interval: int = 10
    num_workers: int = None
    device: str = "cuda"
    gamma: float = 1.0
    debug_single_process: bool = False
    tensorboard_log_dir: str = None
    log_interval: int = 1  # Log every iteration by default
    resume_from_checkpoint: bool = True  # Resume from last checkpoint by default
    randomized: bool = False
    random_seed: int = None  # Random seed for deterministic randomization
    fixed_batch_index: Optional[int] = None # Use a fixed batch for all iterations
    
    def __post_init__(self):
        if self.num_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            self.num_workers = max(1, min(self.batch_size, max_workers))


def load_flight_tres_results(case_dir: str, flight_id: str, takeoff_timestamp: int) -> Tuple[List, List, List, torch.Tensor]:
    """
    Load pre-computed TRES results for a specific flight.
    
    Args:
        case_dir: Directory containing the case data
        flight_id: Flight identifier
        takeoff_timestamp: Takeoff timestamp
    Returns:
        Tuple of (forward_transitions, backward_transitions, thinned_transitions, avg_tailwind_knots)
    """
    tres_dir = Path(case_dir) / "tres_runs"
    
    # Find the batch directory containing this flight
    forward_transitions = None
    backward_transitions = None
    thinned_transitions = None
    avg_tailwind_knots = None
    
    for batch_dir in tres_dir.glob("batch*"):
        fw_file = batch_dir / f"FW_{flight_id}_{takeoff_timestamp}.pkl"
        bw_file = batch_dir / f"BW_{flight_id}_{takeoff_timestamp}.pkl"
        
        if fw_file.exists() and bw_file.exists():
            try:
                with open(fw_file, 'rb') as f:
                    forward_transitions = pickle.load(f)
                with open(bw_file, 'rb') as f:
                    backward_transitions = pickle.load(f)
                
                # Check for pre-computed thinned transitions and wind data
                thinned_file = batch_dir / f"CLSR_{flight_id}_{takeoff_timestamp}.pkl"
                wind_file = batch_dir / f"WIND_{flight_id}_{takeoff_timestamp}.pt"
                
                if thinned_file.exists() and wind_file.exists():
                    with open(thinned_file, 'rb') as f:
                        thinned_transitions = pickle.load(f)
                    avg_tailwind_knots = torch.load(wind_file, weights_only=False)
                    # Found all required files, break the loop
                    break
                else:
                    # If any of the derived files are missing, this batch is incomplete.
                    # Continue to check other batch directories.
                    missing_files = []
                    if not thinned_file.exists():
                        missing_files.append(str(thinned_file))
                    if not wind_file.exists():
                        missing_files.append(str(wind_file))
                    logger.debug(f"Missing derived files in {batch_dir} for flight {flight_id}: {', '.join(missing_files)}. Checking other batch directories.")
                    forward_transitions = None
                    backward_transitions = None
                    continue

            except Exception as e:
                logger.error(f"Error loading TRES results for flight {flight_id} from {batch_dir}. Expected files: {fw_file}, {bw_file}, {thinned_file}, {wind_file}. Error: {e}")
                # Reset and check next directory
                forward_transitions = None
                backward_transitions = None
                thinned_transitions = None
                avg_tailwind_knots = None
                continue
    
    if forward_transitions is None or backward_transitions is None or thinned_transitions is None or avg_tailwind_knots is None:
        # Provide detailed information about what files were expected
        expected_files = []
        for batch_dir in tres_dir.glob("batch*"):
            fw_file = batch_dir / f"FW_{flight_id}_{takeoff_timestamp}.pkl"
            bw_file = batch_dir / f"BW_{flight_id}_{takeoff_timestamp}.pkl"
            thinned_file = batch_dir / f"CLSR_{flight_id}_{takeoff_timestamp}.pkl"
            wind_file = batch_dir / f"WIND_{flight_id}_{takeoff_timestamp}.pt"
            
            missing_in_batch = []
            if not fw_file.exists():
                missing_in_batch.append(f"FW file: {fw_file}")
            if not bw_file.exists():
                missing_in_batch.append(f"BW file: {bw_file}")
            if not thinned_file.exists():
                missing_in_batch.append(f"CLSR file: {thinned_file}")
            if not wind_file.exists():
                missing_in_batch.append(f"WIND file: {wind_file}")
            
            if missing_in_batch:
                expected_files.append(f"In {batch_dir}: Missing {', '.join(missing_in_batch)}")
        
        error_msg = f"Complete TRES, Thinned (CLSR), and Wind (WIND) results not found for flight {flight_id}_{takeoff_timestamp}."
        if expected_files:
            error_msg += f"\nDetailed missing files:\n" + "\n".join(expected_files)
        else:
            error_msg += f" No batch directories found in {tres_dir}."
        
        raise FileNotFoundError(error_msg)
    
    return forward_transitions, backward_transitions, thinned_transitions, avg_tailwind_knots


def compute_empirical_counts_for_flight(flight_data: Mapping[str, Any], node_to_idx: Dict[str, int],
                                       num_nodes: int) -> torch.Tensor:
    """
    Compute empirical counts (actual route usage) for a single flight.
    
    Args:
        flight_data: Flight data mapping with a 'route' field
        node_to_idx: Mapping from waypoint names to indices
        num_nodes: Total number of waypoints
        
    Returns:
        Empirical counts tensor of shape (num_nodes, num_nodes)
    """
    empirical_counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    
    # Parse the actual waypoints from the flight data
    waypoints_str = flight_data['route']
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


def _scalar_from_state_dict(state_dict: Dict[str, torch.Tensor], key: str, default: float) -> float:
    """Best-effort extraction of a scalar float from a model state_dict."""
    v = state_dict.get(key, None)
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.detach().cpu().item())
    return float(default)


def _build_cost_model_from_state_dict(
    *,
    cost_model_version: str,
    num_waypoints: int,
    device: torch.device,
    cost_model_state: Dict[str, torch.Tensor],
) -> torch.nn.Module:
    """
    Construct a cost model instance and load weights from `cost_model_state`.

    Note on betas:
    - The beta coefficients are intentionally kept FIXED (requires_grad=False) to avoid an
      ill-defined scale ambiguity between beta weights and the learned functionals (PLMs / preferences).
    - We therefore treat betas as part of the model definition/state and do not thread them through
      the training pipeline as separate "hyperparameters" that might accidentally drift.
    """
    cost_model_class = get_cost_model_class(cost_model_version)

    # Provide placeholder constructor arguments. For cost model versions that register these
    # as Parameters/Buffers, `load_state_dict` will overwrite them with the authoritative values.
    # For versions that ignore betas (e.g., CostRev3/4), these are ignored by design.
    beta0 = _scalar_from_state_dict(cost_model_state, "beta0", 0.0)
    beta1 = _scalar_from_state_dict(cost_model_state, "beta1", 1.0)
    beta2 = _scalar_from_state_dict(cost_model_state, "beta2", 1.0)
    beta3 = _scalar_from_state_dict(cost_model_state, "beta3", 0.0)
    alpha_pref_reg = _scalar_from_state_dict(cost_model_state, "alpha_pref_reg", 1.0)

    cost_model = cost_model_class(
        beta0=beta0,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        num_waypoints=num_waypoints,
        alpha_pref_reg=alpha_pref_reg,
        device=device,
    )
    cost_model.load_state_dict(cost_model_state)
    cost_model.to(device)
    return cost_model


def _vector_to_named_grads(
    grad_vector: torch.Tensor,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Convert a flattened gradient vector to a name-keyed gradient dict based on the model's
    current `named_parameters()` order (restricted to `requires_grad=True`).

    This must match the flattening order used by `backward_gradient_pass`, which iterates
    `cost_model.parameters()`; PyTorch guarantees `parameters()` and `named_parameters()`
    traverse parameters in the same registration order.
    """
    grads: Dict[str, torch.Tensor] = {}
    idx = 0
    total = int(grad_vector.numel())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = int(param.numel())
        if idx + n > total:
            raise ValueError(
                f"Gradient vector too short while slicing '{name}': need {idx+n} elems, have {total}"
            )
        grads[name] = grad_vector[idx: idx + n].view_as(param).detach()
        idx += n

    if idx != total:
        raise ValueError(f"Gradient vector has {total} elems but only consumed {idx} elems from model parameters")

    return grads


def _run_forward_svi_wrapper(args):
    """Wrapper function for multiprocessing forward SVI."""
    (state_transitions, avg_tailwind_knots, G, idx_to_node, origin_node_idx, 
     cost_model_state, num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases, 
     distance_matrix_d, airspace_charge_matrix_ac, device_str, gamma,
     cost_model_version) = args
    
    device = torch.device(device_str)
    
    # Reconstruct cost model from state dict
    cost_model = _build_cost_model_from_state_dict(
        cost_model_version=cost_model_version,
        num_waypoints=num_nodes,
        device=device,
        cost_model_state=cost_model_state,
    )

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
     distance_matrix_d, airspace_charge_matrix_ac, device_str, gamma,
     cost_model_version) = args
    
    device = torch.device(device_str)
    
    # Reconstruct cost model from state dict
    cost_model = _build_cost_model_from_state_dict(
        cost_model_version=cost_model_version,
        num_waypoints=num_nodes,
        device=device,
        cost_model_state=cost_model_state,
    )

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


def process_single_flight(
    flight_data: Mapping[str, Any],
    cost_model_state: Optional[Dict[str, torch.Tensor]] = None,
    debug_this_flight: bool = False,
) -> FlightGradientResult:
    """
    Process a single flight through the complete pipeline.
    
    Args:
        flight_data: Flight data mapping
        cost_model_state: Serialized state of the cost model for multiprocessing
        
    Returns:
        FlightGradientResult containing gradients and metadata
    """
    start_time = time.time()
    flight_id = flight_data['flight_id']
    takeoff_timestamp = int(flight_data['takeoff_timestamp'])

    ctx = _get_worker_context()
    device = ctx['device']
    gamma = ctx['gamma']
    case_dir = ctx['case_dir']

    if cost_model_state is None:
        raise ValueError("cost_model_state is required to process a flight in worker mode.")

    # Reconstruct the cost model from the provided state dict.
    cost_model_version = ctx['cost_model_version']
    cost_model = _build_cost_model_from_state_dict(
        cost_model_version=cost_model_version,
        num_waypoints=ctx['num_nodes'],
        device=device,
        cost_model_state=cost_model_state,
    )

    try:
        # 1. Load TRES results
        logger.info(f"Processing flight {flight_id}")
        forward_transitions, backward_transitions, thinned_transitions, avg_tailwind_knots = load_flight_tres_results(
            case_dir, flight_id, takeoff_timestamp
        )
        
        if not thinned_transitions:
            return FlightGradientResult(
                flight_id=flight_id,
                takeoff_timestamp=takeoff_timestamp,
                gradient={},
                pref_grad_e=None,
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
        num_nodes = ctx['num_nodes']
        
        # 3. Wind data is now pre-loaded, so this section is simplified.
        logger.debug(f"Flight {flight_id}: Using pre-computed wind for {len(thinned_transitions)} thinned transitions")
        
        # 4. Get origin and goal indices for this flight
        origin_node_idx = ctx['node_to_idx'][flight_data['origin']]
        goal_node_idx = ctx['node_to_idx'][flight_data['destination']]

        # 5. Run forward and backward SVI sequentially within this worker
        if isinstance(avg_tailwind_knots, np.ndarray):
            avg_tailwind_knots = torch.from_numpy(avg_tailwind_knots)
        avg_tailwind_knots = avg_tailwind_knots.to(device)

        v_f = forward_soft_value_iteration(
            state_transitions=thinned_transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots,
            G=ctx['graph'],
            idx_to_node=ctx['idx_to_node'],
            origin_node_idx=origin_node_idx,
            cost_model=cost_model,
            num_nodes=num_nodes,
            num_time_bins_wall_clock=num_time_bins_wall_clock,
            num_rho_bins=num_rho_bins,
            num_phases=num_phases,
            distance_matrix_d=ctx['dist_matrix'],
            airspace_charge_matrix_ac=ctx['ac_matrix'],
            device=device,
            gamma=gamma,
            verbose=False
        )

        v_b, _ = backward_soft_value_iteration(
            state_transitions=thinned_transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots,
            G=ctx['graph'],
            idx_to_node=ctx['idx_to_node'],
            goal_node_idx=goal_node_idx,
            cost_model=cost_model,
            num_nodes=num_nodes,
            num_time_bins_wall_clock=num_time_bins_wall_clock,
            num_rho_bins=num_rho_bins,
            num_phases=num_phases,
            distance_matrix_d=ctx['dist_matrix'],
            airspace_charge_matrix_ac=ctx['ac_matrix'],
            device=device,
            gamma=gamma,
            verbose=False
        )
        
        # 6. Compute empirical counts
        
        empirical_counts = compute_empirical_counts_for_flight(
            flight_data, ctx['node_to_idx'], num_nodes
        ).to(device)
        
        # 7. Compute gradients using backward gradient pass
        expected_counts, gradient, log_partition_z_tensor = backward_gradient_pass(
            state_transitions=thinned_transitions,
            avg_tailwind_knots_per_transition=avg_tailwind_knots,
            V_f=v_f,
            V_b=v_b,
            cost_model=cost_model,
            empirical_counts=empirical_counts,
            origin_node_idx=origin_node_idx,
            num_nodes=num_nodes,
            distance_matrix_d=ctx['dist_matrix'],
            airspace_charge_matrix_ac=ctx['ac_matrix'],
            device=device,
            gamma=gamma,
            verbose=False
        )

        pref_grad_e = None
        if ctx.get("pref_enabled"):
            edge_u = ctx.get("edge_u")
            edge_v = ctx.get("edge_v")
            if edge_u is None or edge_v is None:
                raise RuntimeError("Preference edges missing in worker context.")
            pref_grad_e = (empirical_counts[edge_u, edge_v] - expected_counts[edge_u, edge_v]) / gamma

        # 8.5 Debugging: Print top links with highest empirical counts and their expected traversals
        # Only print for the designated debug flight to avoid spam
        # if debug_this_flight:
        #     logger.info(f"=== DEBUGGING COUNTS FOR FLIGHT {flight_id} ===")
            
        #     # Find all non-zero empirical counts
        #     nonzero_mask = empirical_counts > 0
        #     nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False)
            
        #     if len(nonzero_indices) > 0:
        #         # Get empirical and expected counts for non-zero empirical links
        #         empirical_values = empirical_counts[nonzero_mask]
        #         expected_values = expected_counts[nonzero_mask]
                
        #         # Sort by empirical counts (descending)
        #         sorted_indices = torch.argsort(empirical_values, descending=True)
                
        #         # Show top 10 links (or all if fewer than 10)
        #         top_k = min(10, len(sorted_indices))
        #         logger.info(f"Top {top_k} links with highest empirical counts:")
                
        #         for i in range(top_k):
        #             idx = sorted_indices[i]
        #             from_idx = nonzero_indices[idx, 0].item()
        #             to_idx = nonzero_indices[idx, 1].item()
        #             emp_count = empirical_values[idx].item()
        #             exp_count = expected_values[idx].item()
                    
        #             from_node = ctx['idx_to_node'][from_idx]
        #             to_node = ctx['idx_to_node'][to_idx]
                    
        #             # Calculate ratio to see if expected is catching up to empirical
        #             ratio = exp_count / emp_count if emp_count > 0 else 0.0
                    
        #             logger.info(f"  {i+1:2d}. {from_node:8s} -> {to_node:8s} | "
        #                        f"Empirical: {emp_count:.3e} | Expected: {exp_count:.3e} | "
        #                        f"Ratio: {ratio:.3e}")
        #     else:
        #         logger.warning(f"No empirical traversals found for flight {flight_id}")
            
        #     # Also show some high expected counts that don't correspond to empirical usage
        #     logger.info(f"Links with highest expected counts (regardless of empirical usage):")
        #     expected_flat = expected_counts.flatten()
        #     top_expected_indices = torch.topk(expected_flat, k=min(5, len(expected_flat)), largest=True).indices
            
        #     for i, flat_idx in enumerate(top_expected_indices):
        #         from_idx = flat_idx // expected_counts.shape[1]
        #         to_idx = flat_idx % expected_counts.shape[1]
        #         emp_count = empirical_counts[from_idx, to_idx].item()
        #         exp_count = expected_counts[from_idx, to_idx].item()
                
        #         from_node = ctx['idx_to_node'][from_idx.item()]
        #         to_node = ctx['idx_to_node'][to_idx.item()]
                
        #         logger.info(f"  {i+1}. {from_node:8s} -> {to_node:8s} | "
        #                    f"Empirical: {emp_count:.3e} | Expected: {exp_count:.3e}")
            
        #     logger.info(f"=== END DEBUGGING COUNTS FOR FLIGHT {flight_id} ===\n")

        # For debugging, save the expected counts and the soft value functions to a file to be inspected externally
        if ctx['debug_single_process'] and debug_this_flight:
            import pickle
            with open(f"expected_counts_{flight_id}.pkl", "wb") as f:
                pickle.dump(expected_counts.cpu().numpy(), f)
            with open(f"soft_value_functions_f_{flight_id}.pkl", "wb") as f:
                pickle.dump(v_f.cpu().numpy(), f)
            with open(f"soft_value_functions_b_{flight_id}.pkl", "wb") as f:
                pickle.dump(v_b.cpu().numpy(), f)

        # raise Exception("Stop here")

        # 9. Compute true log likelihood
        log_partition_z = log_partition_z_tensor.item()

        # Calculate cost of the empirical trajectory c(xi)
        waypoints_str = flight_data['route']
        waypoints = waypoints_str.split()
        empirical_route_links = []
        for i in range(len(waypoints) - 1):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]
            if from_wp in ctx['node_to_idx'] and to_wp in ctx['node_to_idx']:
                from_idx = ctx['node_to_idx'][from_wp]
                to_idx = ctx['node_to_idx'][to_wp]
                empirical_route_links.append((from_idx, to_idx))

        transitions_by_link = defaultdict(list)
        for i, t in enumerate(thinned_transitions):
            transitions_by_link[(t[0], t[5])].append(i)

        c_xi = 0.0
        with torch.no_grad():
            for u_idx, v_idx in empirical_route_links:
                if (u_idx, v_idx) in transitions_by_link:
                    transition_indices = transitions_by_link[(u_idx, v_idx)]
                    
                    # Average tailwind for this link, from the model's perspective
                    avg_tailwind_for_link = avg_tailwind_knots[transition_indices].mean()
                    
                    edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
                    edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)
                    
                    link_cost = cost_model(
                        (edge_u_indices, edge_v_indices),
                        ctx['dist_matrix'],
                        ctx['ac_matrix'],
                        avg_tailwind_for_link.to(device).unsqueeze(0)
                    ).item()
                    
                    c_xi += link_cost
                else:
                    # This link from the empirical route is not in our model's reachable graph.
                    # This means the model assigns it zero probability.
                    logger.warning(f"Link ({u_idx}, {v_idx}) from empirical route for flight {flight_id} not found in thinned transitions. Skipping this link in log-likelihood computation.")
                    # c_xi = float('inf')
                    # break
                    pass 
        
        if torch.isinf(torch.tensor(c_xi)) or torch.isinf(log_partition_z_tensor):
            log_likelihood = -float('inf')
        else:
            # log p(xi) = -c(xi) - log Z. Note that log_partition_z is already V, so it's -gamma*logZ.
            # We want log(p(xi)) = -c(xi)/gamma - log(Z). log_partition_z = V_b(origin), and log(Z) = -V_b(origin)/gamma
            log_likelihood = (-c_xi / gamma) - (-log_partition_z / gamma)
            log_likelihood = (-c_xi + log_partition_z) / gamma
        
        # Convert to name-keyed gradients for robust application in the main process.
        try:
            gradient_by_name = _vector_to_named_grads(gradient, cost_model)
        except Exception as e:
            return FlightGradientResult(
                flight_id=flight_id,
                takeoff_timestamp=takeoff_timestamp,
                gradient={},
                pref_grad_e=None,
                log_likelihood=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=f"Gradient vector-to-name mapping failed: {e}"
            )
        
        processing_time = time.time() - start_time
        
        return FlightGradientResult(
            flight_id=flight_id,
            takeoff_timestamp=takeoff_timestamp,
            gradient={k: v.cpu() for k, v in gradient_by_name.items()},
            pref_grad_e=pref_grad_e.detach().cpu() if pref_grad_e is not None else None,
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
            gradient={},
            pref_grad_e=None,
            log_likelihood=0.0,
            processing_time=time.time() - start_time,
            success=False,
            error_message=str(e)
        )


def _process_flight_wrapper(args):
    """Wrapper for process_single_flight to be used with ProcessPoolExecutor."""
    return process_single_flight(*args)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in the output directory.
    
    Args:
        output_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        logger.debug(f"Output directory does not exist: {output_dir}")
        return None
    
    logger.debug(f"Searching for checkpoints in directory: {output_dir}")
    
    # Get all files in the directory for debugging
    try:
        all_files = os.listdir(output_dir)
        logger.debug(f"All files in {output_dir}: {all_files}")
    except Exception as e:
        logger.error(f"Error listing directory {output_dir}: {e}")
        return None
    
    checkpoint_files = []
    for filename in all_files:
        if filename.startswith("checkpoint_iter_") and filename.endswith(".pt"):
            logger.debug(f"Found potential checkpoint file: {filename}")
            # Extract iteration number from filename
            try:
                # More robust parsing: extract the number between "checkpoint_iter_" and ".pt"
                prefix = "checkpoint_iter_"
                suffix = ".pt"
                if filename.startswith(prefix) and filename.endswith(suffix):
                    iter_str = filename[len(prefix):-len(suffix)]
                    iter_num = int(iter_str)
                    full_path = os.path.join(output_dir, filename)
                    checkpoint_files.append((iter_num, full_path))
                    logger.debug(f"Valid checkpoint found: iteration {iter_num}, path: {full_path}")
            except ValueError as e:
                logger.warning(f"Could not parse iteration number from filename {filename}: {e}")
                continue
    
    if not checkpoint_files:
        logger.debug("No valid checkpoint files found")
        return None
    
    # Sort by iteration number and get the latest
    checkpoint_files.sort(key=lambda x: x[0])
    latest_checkpoint = checkpoint_files[-1]
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint(s). Latest: iteration {latest_checkpoint[0]}, path: {latest_checkpoint[1]}")
    
    # Verify the file actually exists and is readable
    if os.path.exists(latest_checkpoint[1]) and os.path.isfile(latest_checkpoint[1]):
        return latest_checkpoint[1]
    else:
        logger.error(f"Latest checkpoint file does not exist or is not readable: {latest_checkpoint[1]}")
        return None


def load_checkpoint(checkpoint_path: str, cost_model, optimizer, device: torch.device) -> Tuple[int, Dict]:
    """
    Load a checkpoint and restore model, optimizer, and training state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cost_model: Cost model to load state into
        optimizer: Optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Tuple of (start_iteration, training_history)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    cost_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get starting iteration (add 1 to continue from next iteration)
    start_iteration = checkpoint['iteration'] + 1
    
    # Load training history
    training_history = checkpoint.get('training_history', {
        'iterations': [],
        'avg_log_likelihood': [],
        'gradient_norms': [],
        'processing_times': [],
        'successful_flights': [],
        'failed_flights': []
    })
    
    logger.info(f"Checkpoint loaded successfully. Resuming from iteration {start_iteration}")
    logger.info(f"Previous training history contains {len(training_history['iterations'])} iterations")
    
    return start_iteration, training_history


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
    
    # Setup TensorBoard logging
    tensorboard_writer = None
    if TENSORBOARD_AVAILABLE:
        if batch_config.tensorboard_log_dir is not None:
            tb_log_dir = batch_config.tensorboard_log_dir
        else:
            tb_log_dir = os.path.join(output_dir, "tensorboard_logs")
        
        os.makedirs(tb_log_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging enabled. Log directory: {tb_log_dir}")
        logger.info(f"To view logs, run: tensorboard --logdir {tb_log_dir}")
    else:
        logger.warning("TensorBoard not available. Training will proceed without TensorBoard logging.")
    
    # 1. Load configuration and initialize components
    logger.info(f"Loading configuration from {config_path}")
    config = RunConfiguration.load_from_yaml(config_path)
    components = config.initialize_all_components(cost_model_version=config.cost_model_version)
    components['cost_model_version'] = config.cost_model_version
    components['etto_delta_t_seconds'] = config.etto_delta_t_seconds
    
    device = torch.device(batch_config.device if torch.cuda.is_available() else "cpu")
    components['device'] = device
    components['cost_model'] = components['cost_model'].to(device)
    
    # IMPORTANT: Do NOT blanket-enable gradients for all parameters.
    # The beta coefficients are intentionally kept fixed to avoid an ill-defined scaling
    # between betas and the learned functionals. Cost model implementations should mark
    # betas (and other fixed scalars) with requires_grad=False.
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, components['cost_model'].parameters()),
        lr=batch_config.learning_rate
        # maximize=True # I have personally verified that we need to minimize the cost function!
    )
    
    logger.info(f"Using device: {device}")
    num_trainable_params = sum(p.numel() for p in components['cost_model'].parameters() if p.requires_grad)
    logger.info(f"Cost model has {num_trainable_params} trainable parameters")
    if num_trainable_params == 0:
        logger.warning("Cost model has 0 trainable parameters (requires_grad=True). Training will be a no-op.")
    
    # 2. Load flight data and create batches
    flights_csv = os.path.join(case_dir, "tres_runs", "all_routes_feasibly_snapped.csv")
    logger.info(f"Loading flights from {flights_csv} and creating batches...")
    flight_batches = get_flight_batches(flights_csv, batch_config.batch_size)
    num_batches = len(flight_batches)
    if num_batches == 0:
        logger.error("No flight batches were created. Please check the routes file and batch size.")
        return
    logger.info(f"Created {num_batches} batches of flights.")

    pref_enabled = components["cost_model_version"] == "lin_disent"
    pref_projector = None
    pref_edge_u = None
    pref_edge_v = None
    pref_support_mask = None
    if pref_enabled:
        routes_df = pd.read_csv(flights_csv)
        edge_u_cpu, edge_v_cpu = build_edge_list(components["graph"], components["node_to_idx"])
        pref_edge_u = edge_u_cpu.to(device)
        pref_edge_v = edge_v_cpu.to(device)

        global_counts = compute_empirical_counts_from_routes(
            routes_df["route"],
            components["node_to_idx"],
            components["num_nodes"],
        )
        d_e = global_counts[edge_u_cpu, edge_v_cpu].to(device=device, dtype=torch.float64)

        X_raw = build_feature_matrix(
            pref_edge_u,
            pref_edge_v,
            components["dist_matrix"],
            components["ac_matrix"],
            device=device,
            dtype=torch.float64,
        )
        X_norm, feature_means, feature_scales, manual_scales = d_weighted_normalize_features(
            X_raw,
            d_e,
            bias_index=0,
        )
        pref_projector = PreferenceProjector(
            X_norm,
            d_e,
            ridge=batch_config.pref_projection_ridge,
            feature_means=feature_means,
            feature_scales=feature_scales,
            manual_scales=manual_scales,
        )
        pref_support_mask = d_e > 0

        if pref_projector.condition_number is not None:
            logger.info(f"Preference projector condition number: {pref_projector.condition_number:.3e}")

        with torch.no_grad():
            pref_matrix = components["cost_model"].preference_matrix_p
            p_e = pref_matrix[pref_edge_u, pref_edge_v].to(dtype=X_norm.dtype)
            p_e = pref_projector.project(p_e)
            pref_matrix.zero_()
            pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)
    
    # 3. Initialize tracking variables and handle checkpoint resumption
    gradient_queue: List[Dict[str, torch.Tensor]] = []
    pref_grad_queue: List[torch.Tensor] = []
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
    
    # Check for existing checkpoint and resume if requested
    if batch_config.resume_from_checkpoint:
        logger.info("Checkpoint resumption is enabled. Searching for existing checkpoints...")
        logger.info(f"Checkpoint search directory: {output_dir}")
        logger.info(f"Expected checkpoint pattern: checkpoint_iter_<number>.pt")
        
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint:
            try:
                iteration, training_history = load_checkpoint(
                    latest_checkpoint, components['cost_model'], optimizer, device
                )
                logger.info(f"✓ Successfully resumed training from checkpoint. Starting at iteration {iteration}")
                if pref_enabled and pref_projector is not None:
                    with torch.no_grad():
                        pref_matrix = components["cost_model"].preference_matrix_p
                        p_e = pref_matrix[pref_edge_u, pref_edge_v].to(dtype=pref_projector.X.dtype)
                        p_e = pref_projector.project(p_e)
                        pref_matrix.zero_()
                        pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)
            except Exception as e:
                logger.warning(f"✗ Failed to load checkpoint {latest_checkpoint}: {e}")
                logger.info("Starting training from scratch")
                iteration = 1  # Start from iteration 1 for fresh training
        else:
            logger.info("No existing checkpoints found. Starting training from scratch")
            logger.info(f"Note: Checkpoints will be saved to {output_dir}")
            iteration = 1  # Start from iteration 1 for fresh training
    else:
        logger.info("Checkpoint resumption disabled. Starting training from scratch")
        iteration = 1  # Start from iteration 1 for fresh training
    
    # 4. Set random seed for deterministic behavior if specified
    if batch_config.random_seed is not None:
        random.seed(batch_config.random_seed)
        np.random.seed(batch_config.random_seed)
        torch.manual_seed(batch_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(batch_config.random_seed)
            torch.cuda.manual_seed_all(batch_config.random_seed)
        logger.info(f"Random seed set to {batch_config.random_seed} for deterministic behavior")
    
    # 5. Main training loop
    logger.info("Starting main training loop")
    
    # Create a single pool of workers for parallel flight processing
    with ProcessPoolExecutor(
        max_workers=batch_config.num_workers,
        initializer=_init_worker,
        initargs=(config_path, case_dir, str(device), batch_config.gamma, batch_config.debug_single_process),
    ) as executor:
        while iteration <= batch_config.max_iterations and not converged:
            iteration_start_time = time.time()
            
            logger.info(f"\n--- Iteration {iteration}/{batch_config.max_iterations} ---")
            
            # Get the next batch of flights (fixed, randomly or sequentially)
            if batch_config.fixed_batch_index is not None:
                batch_idx = batch_config.fixed_batch_index
                if batch_idx >= num_batches:
                    logger.error(f"fixed_batch_index {batch_idx} is out of bounds. Number of batches is {num_batches}.")
                    raise IndexError(f"fixed_batch_index {batch_idx} is out of bounds. Number of batches is {num_batches}.")
            elif batch_config.randomized:
                batch_idx = random.randint(0, num_batches - 1)
            else:
                batch_idx = (iteration - 1) % num_batches
            
            batch_flights = flight_batches[batch_idx]
            
            if batch_config.fixed_batch_index is not None:
                logger.info(f"Processing fixed batch {batch_idx + 1}/{num_batches} with {len(batch_flights)} flights")
            elif batch_config.randomized:
                logger.info(f"Processing random batch {batch_idx + 1}/{num_batches} with {len(batch_flights)} flights")
            else:
                logger.info(f"Processing sequential batch {batch_idx + 1}/{num_batches} with {len(batch_flights)} flights")
            
            # Get current model state to be used by all flights in this batch
            cost_model_state = {k: v.detach().cpu() for k, v in components['cost_model'].state_dict().items()}

            # Process flights in the batch in parallel
            batch_results = []
            
            # Prepare arguments for each flight in the batch
            tasks = []
            flight_list = list(batch_flights.iterrows())
            for i, (_, flight_data) in enumerate(flight_list):
                # Set debug flag for only the last flight in the batch
                debug_this_flight = (i == len(flight_list) - 1)
                flight_payload = {
                    "flight_id": flight_data["flight_id"],
                    "takeoff_timestamp": int(flight_data["takeoff_time"]),
                    "origin": flight_data["origin"],
                    "destination": flight_data["destination"],
                    "route": flight_data["route"],
                }
                tasks.append((flight_payload, cost_model_state, debug_this_flight))

            # Use the executor to run flight processing in parallel
            future_results = executor.map(_process_flight_wrapper, tasks)
            
            # Collect results
            batch_results = list(tqdm(future_results, total=len(tasks), desc=f"Processing batch {batch_idx+1}"))

            successful_flights = 0
            failed_flights = 0
            gradient_queue.clear() # Clear queue for each new batch
            pref_grad_queue.clear()

            for result in batch_results:
                if result.success:
                    successful_flights += 1
                    gradient_queue.append(result.gradient)
                    if pref_enabled and result.pref_grad_e is not None:
                        pref_grad_queue.append(result.pref_grad_e)
                else:
                    failed_flights += 1
                    logger.warning(f"Flight {result.flight_id} failed: {result.error_message}")
            
            # Apply gradient update after the entire batch is processed
            gradient_norm = 0.0
            if gradient_queue:
                # Average gradients by parameter NAME to avoid any dependence on parameter ordering.
                trainable_named_params = [
                    (name, param) for name, param in components['cost_model'].named_parameters()
                    if param.requires_grad
                ]
                trainable_names = [n for n, _ in trainable_named_params]

                # Initialize sums on the target device
                grad_sums: Dict[str, torch.Tensor] = {
                    name: torch.zeros_like(param, device=device, dtype=torch.float32)
                    for name, param in trainable_named_params
                }

                # Accumulate
                for grad_dict in gradient_queue:
                    for name, _ in trainable_named_params:
                        if name not in grad_dict:
                            raise KeyError(
                                f"Missing gradient for parameter '{name}'. "
                                f"Expected keys: {trainable_names}. Got keys: {list(grad_dict.keys())}"
                            )
                        grad_sums[name] += grad_dict[name].to(device=device, dtype=torch.float32)

                denom = float(len(gradient_queue))
                avg_grads: Dict[str, torch.Tensor] = {name: g / denom for name, g in grad_sums.items()}

                # Gradient norm (L2 over concatenated parameters)
                gradient_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in avg_grads.values())).item())

                # Apply gradient update
                optimizer.zero_grad()

                for name, param in trainable_named_params:
                    param.grad = avg_grads[name].to(dtype=param.dtype)
                
                optimizer.step()
                
                logger.info(f"✓ Applied gradient update with norm L2: {gradient_norm:.6f}")
                
                # Check convergence
                if gradient_norm < batch_config.convergence_threshold:
                    logger.info(f"✓ Convergence achieved! Gradient norm L2: {gradient_norm:.6f} < {batch_config.convergence_threshold}")
                    converged = True

            pref_grad_norm = None
            pref_violation = None
            pref_min = None
            pref_max = None
            pref_mean = None
            if pref_enabled and pref_projector is not None and pref_grad_queue:
                pref_grad_sum = torch.zeros_like(
                    pref_grad_queue[0],
                    device=device,
                    dtype=pref_projector.X.dtype,
                )
                for grad in pref_grad_queue:
                    pref_grad_sum += grad.to(device=device, dtype=pref_projector.X.dtype)

                pref_grad_avg = pref_grad_sum / float(len(pref_grad_queue))

                pref_matrix = components["cost_model"].preference_matrix_p
                p_e = pref_matrix[pref_edge_u, pref_edge_v].to(
                    device=device,
                    dtype=pref_projector.X.dtype,
                )

                alpha_pref_reg = float(
                    components["cost_model"].alpha_pref_reg.detach().cpu().item()
                )
                if alpha_pref_reg != 0.0:
                    pref_grad_avg = pref_grad_avg + 2.0 * alpha_pref_reg * p_e

                pref_grad_proj = pref_projector.project(pref_grad_avg)
                p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
                p_e = pref_projector.project(p_e)

                with torch.no_grad():
                    pref_matrix.zero_()
                    pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)

                pref_grad_norm = float(torch.linalg.norm(pref_grad_proj).item())
                pref_violation = float(pref_projector.constraint_violation(p_e).item())

                support_mask = pref_support_mask
                if support_mask is not None and support_mask.any():
                    p_support = p_e[support_mask]
                else:
                    p_support = p_e
                pref_min = float(p_support.min().item())
                pref_max = float(p_support.max().item())
                pref_mean = float(p_support.mean().item())

                logger.info(
                    "Preference update: grad_norm=%.6f | constraint=%.3e | min=%.6f max=%.6f mean=%.6f",
                    pref_grad_norm,
                    pref_violation,
                    pref_min,
                    pref_max,
                    pref_mean,
                )
            
            # Compute iteration statistics
            avg_log_likelihood = np.mean([r.log_likelihood for r in batch_results if r.success]) if successful_flights > 0 else 0.0
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
            
            # Log to TensorBoard
            if tensorboard_writer is not None and iteration % batch_config.log_interval == 0:
                # Training metrics
                tensorboard_writer.add_scalar('Training/Average_Log_Likelihood', avg_log_likelihood, iteration)
                tensorboard_writer.add_scalar('Training/Gradient_Norm_L2', gradient_norm, iteration)
                tensorboard_writer.add_scalar('Training/Learning_Rate', batch_config.learning_rate, iteration)
                
                # Flight processing metrics
                tensorboard_writer.add_scalar('Flights/Successful_Count', successful_flights, iteration)
                tensorboard_writer.add_scalar('Flights/Failed_Count', failed_flights, iteration)
                tensorboard_writer.add_scalar('Flights/Success_Rate', successful_flights / len(batch_flights), iteration)
                
                # Performance metrics
                tensorboard_writer.add_scalar('Performance/Iteration_Time_Seconds', iteration_time, iteration)
                tensorboard_writer.add_scalar('Performance/Flights_Per_Second', len(batch_flights) / iteration_time, iteration)

                if pref_enabled and pref_grad_norm is not None:
                    tensorboard_writer.add_scalar('Preferences/Grad_Norm_L2', pref_grad_norm, iteration)
                    tensorboard_writer.add_scalar('Preferences/Constraint_Violation', pref_violation, iteration)
                    tensorboard_writer.add_scalar('Preferences/Mean', pref_mean, iteration)
                    tensorboard_writer.add_scalar('Preferences/Min', pref_min, iteration)
                    tensorboard_writer.add_scalar('Preferences/Max', pref_max, iteration)
                
                # Model parameters statistics
                for name, param in components['cost_model'].named_parameters():
                    if param.requires_grad:
                        tensorboard_writer.add_scalar(f'Parameters/{name}_mean', param.data.mean().item(), iteration)
                        tensorboard_writer.add_scalar(f'Parameters/{name}_std', param.data.std().item(), iteration)
                        tensorboard_writer.add_scalar(f'Parameters/{name}_max', param.data.max().item(), iteration)
                        tensorboard_writer.add_scalar(f'Parameters/{name}_min', param.data.min().item(), iteration)
                        
                        # Log gradient statistics if available
                        if param.grad is not None:
                            tensorboard_writer.add_scalar(f'Gradients/{name}_mean', param.grad.data.mean().item(), iteration)
                            tensorboard_writer.add_scalar(f'Gradients/{name}_std', param.grad.data.std().item(), iteration)
                            tensorboard_writer.add_scalar(f'Gradients/{name}_norm', param.grad.data.norm().item(), iteration)
                
                # Log histograms less frequently to avoid cluttering
                if iteration % (batch_config.log_interval * 5) == 0:
                    for name, param in components['cost_model'].named_parameters():
                        if param.requires_grad:
                            tensorboard_writer.add_histogram(f'Parameters_Hist/{name}', param.data, iteration)
                            if param.grad is not None:
                                tensorboard_writer.add_histogram(f'Gradients_Hist/{name}', param.grad.data, iteration)
                
                # Log batch-specific metrics if available
                flight_processing_times = [r.processing_time for r in batch_results if r.success]
                if flight_processing_times:
                    tensorboard_writer.add_scalar('Performance/Avg_Flight_Processing_Time', 
                                                 np.mean(flight_processing_times), iteration)
                    tensorboard_writer.add_scalar('Performance/Max_Flight_Processing_Time', 
                                                 np.max(flight_processing_times), iteration)
                
                # Log individual flight log likelihoods distribution
                flight_log_likelihoods = [r.log_likelihood for r in batch_results if r.success]
                if flight_log_likelihoods:
                    tensorboard_writer.add_histogram('Training/Flight_Log_Likelihoods', 
                                                    np.array(flight_log_likelihoods), iteration)
                
                # Flush to ensure data is written
                tensorboard_writer.flush()
            
            # Save checkpoint before incrementing iteration
            if iteration % batch_config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_iter_{iteration}.pt")
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': components['cost_model'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_history': training_history,
                        'batch_config': asdict(batch_config)
                    }, checkpoint_path)
                    
                    # Verify the checkpoint was saved successfully
                    if os.path.exists(checkpoint_path):
                        file_size = os.path.getsize(checkpoint_path)
                        logger.info(f"✓ Checkpoint saved successfully to {checkpoint_path} (size: {file_size:,} bytes)")
                    else:
                        logger.error(f"✗ Checkpoint save failed - file does not exist: {checkpoint_path}")
                        
                except Exception as e:
                    logger.error(f"✗ Error saving checkpoint to {checkpoint_path}: {e}")
                
                # Log checkpoint save to TensorBoard
                if tensorboard_writer is not None:
                    tensorboard_writer.add_text('Training/Checkpoint', 
                                               f"Checkpoint saved at iteration {iteration}: {checkpoint_path}", 
                                               iteration)
            
            # Increment iteration for next loop
            iteration += 1

    # 5. Save final results and checkpoint
    final_results = {
        'converged': converged,
        'final_iteration': iteration,
        'training_history': training_history,
        'final_model_state': components['cost_model'].state_dict(),
        'batch_config': asdict(batch_config)
    }
    
    results_path = os.path.join(output_dir, "final_results.pt")
    torch.save(final_results, results_path)
    
    # Save a final checkpoint
    final_checkpoint_path = os.path.join(output_dir, f"checkpoint_iter_{iteration}.pt")
    try:
        torch.save({
            'iteration': iteration,
            'model_state_dict': components['cost_model'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': training_history,
            'batch_config': asdict(batch_config)
        }, final_checkpoint_path)
        
        if os.path.exists(final_checkpoint_path):
            file_size = os.path.getsize(final_checkpoint_path)
            logger.info(f"✓ Final checkpoint saved to {final_checkpoint_path} (size: {file_size:,} bytes)")
        else:
            logger.error(f"✗ Final checkpoint save failed - file does not exist: {final_checkpoint_path}")
            
    except Exception as e:
        logger.error(f"✗ Error saving final checkpoint to {final_checkpoint_path}: {e}")
    
    # Save training history as CSV for analysis
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    # Log final training summary to TensorBoard
    if tensorboard_writer is not None:
        # Add final summary statistics
        tensorboard_writer.add_text('Training/Summary', 
                                   f"Training completed after {iteration} iterations. "
                                   f"Converged: {converged}. "
                                   f"Final gradient norm: {gradient_norm:.6f}. "
                                   f"Final avg log likelihood: {avg_log_likelihood:.6f}")
        
        # Log hyperparameters and final metrics
        hparams = {
            'batch_size': batch_config.batch_size,
            'learning_rate': batch_config.learning_rate,
            'pref_learning_rate': batch_config.pref_learning_rate,
            'pref_projection_ridge': batch_config.pref_projection_ridge,
            'gamma': batch_config.gamma,
            'max_iterations': batch_config.max_iterations,
            'convergence_threshold': batch_config.convergence_threshold
        }
        
        metrics = {
            'final_gradient_norm': gradient_norm,
            'final_avg_log_likelihood': avg_log_likelihood,
            'converged': int(converged),
            'total_iterations': iteration
        }
        
        tensorboard_writer.add_hparams(hparams, metrics)
        tensorboard_writer.close()
        logger.info(f"TensorBoard logs saved to {tb_log_dir}")
    
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
        description="Batch SGD Pipeline for Maximum Entropy Inverse Learning with automatic checkpoint resumption"
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
        default=1e-5,
        help="Learning rate for SGD (default: 1e-5)"
    )
    parser.add_argument(
        "--pref-learning-rate",
        type=float,
        default=1e-2,
        help="Preference learning rate for SGD updates (default: 1e-2)"
    )
    parser.add_argument(
        "--pref-projection-ridge",
        type=float,
        default=1e-8,
        help="Ridge added to X^T D X for the preference projector (default: 1e-8)"
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
    parser.add_argument(
        "--tensorboard-log-dir",
        help="Directory for TensorBoard logs (default: output-dir/tensorboard_logs)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Interval for logging to TensorBoard (default: 1, log every iteration)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume from checkpoint (default: resume is enabled)"
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the order of flights in each batch"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for deterministic randomization (default: None for non-deterministic)"
    )
    parser.add_argument(
        "--fixed-batch-index",
        type=int,
        default=None,
        help="Use a fixed batch index for all iterations to check for convergence on a single batch."
    )
    
    args = parser.parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = os.path.join(args.case_dir, "default.yaml")
    
    # Create batch configuration
    batch_config = BatchLearningConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pref_learning_rate=args.pref_learning_rate,
        pref_projection_ridge=args.pref_projection_ridge,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        device=args.device,
        gamma=args.gamma,
        debug_single_process=args.debug_single_process,
        tensorboard_log_dir=args.tensorboard_log_dir,
        log_interval=args.log_interval,
        resume_from_checkpoint=not args.no_resume,  # Resume by default unless --no-resume is specified
        randomized=args.randomize,
        random_seed=getattr(args, 'random_seed', None),  # Handle hyphenated argument name
        fixed_batch_index=args.fixed_batch_index
    )
    
    # Validate implementation first
    if not validate_implementation():
        logger.error("Implementation validation failed. Exiting.")
        return
    
    # Delete tensorboard log directory if it exists
    # Remove existing TensorBoard log directory
    if batch_config.tensorboard_log_dir:
        tensorboard_dir = batch_config.tensorboard_log_dir
    elif args.output_dir:
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard_logs")
    else:
        tensorboard_dir = None
    
    import shutil
    if tensorboard_dir:
        try:
            if os.path.exists(tensorboard_dir):
                logger.info(f"Removing existing TensorBoard log directory: {tensorboard_dir}")
                shutil.rmtree(tensorboard_dir)
        except FileNotFoundError:
            logger.warning(f"TensorBoard log directory {tensorboard_dir} was not found when attempting to remove it. Skipping deletion.")
        except Exception as e:
            logger.error(f"Error while removing TensorBoard log directory {tensorboard_dir}: {e}")
            raise
    
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
