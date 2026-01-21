# To run: python src/equinox/training/batch_sgd_pipeline_parallel.py --case-dir data/cases/LEMD_EGLL
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
    python batch_sgd_pipeline_parallel_w_tensorboard.py --case-dir data/cases/LEMD_EGLL
    
    To monitor training with TensorBoard:
    tensorboard --logdir data/cases/LEMD_EGLL/batch_sgd_results/tensorboard_logs
    
    To keep runs separate, set a run name (default uses a timestamp):
    python batch_sgd_pipeline_parallel_w_tensorboard.py --case-dir data/cases/LEMD_EGLL \
        --tensorboard-run-name run_01
    
    To clear existing TensorBoard logs (optional):
    python batch_sgd_pipeline_parallel_w_tensorboard.py --case-dir data/cases/LEMD_EGLL \
        --clear-tensorboard-logs
    
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
import hashlib
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import yaml
from collections import defaultdict
import re
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.equinox.config import RunConfiguration
from src.equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass
from src.equinox.preferences.disentanglement import (
    build_edge_list,
    build_feature_matrix,
    compute_empirical_counts_from_routes,
    d_weighted_normalize_features,
    GaugeFixedPreferenceProjector,
    _EPS
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


def _edge_list_fingerprint(edge_u: torch.Tensor, edge_v: torch.Tensor) -> str:
    if edge_u.numel() != edge_v.numel():
        raise ValueError("edge_u and edge_v must have the same number of elements.")
    edge_u_cpu = edge_u.detach().to(device="cpu", dtype=torch.int64).contiguous()
    edge_v_cpu = edge_v.detach().to(device="cpu", dtype=torch.int64).contiguous()
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(edge_u_cpu.numpy().tobytes())
    hasher.update(b"|")
    hasher.update(edge_v_cpu.numpy().tobytes())
    return f"{edge_u_cpu.numel()}:{hasher.hexdigest()}"


def _support_component_stats(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    num_nodes: int,
    support_mask: torch.Tensor,
) -> dict[str, float]:
    if num_nodes <= 0:
        return {
            "edge_count": 0.0,
            "node_count": 0.0,
            "component_count": 0.0,
            "edge_fraction": 0.0,
            "node_fraction": 0.0,
        }

    edge_u_cpu = edge_u.detach().to(device="cpu", dtype=torch.int64)
    edge_v_cpu = edge_v.detach().to(device="cpu", dtype=torch.int64)
    support_cpu = support_mask.detach().to(device="cpu")
    if support_cpu.numel() != edge_u_cpu.numel():
        raise ValueError("support_mask must match edge list length.")

    support_edge_u = edge_u_cpu[support_cpu]
    support_edge_v = edge_v_cpu[support_cpu]
    support_edge_count = int(support_edge_u.numel())
    total_edges = int(edge_u_cpu.numel())

    support_nodes = set(support_edge_u.tolist()) | set(support_edge_v.tolist())
    support_node_count = len(support_nodes)

    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for left, right in zip(support_edge_u.tolist(), support_edge_v.tolist()):
        union(left, right)

    if support_node_count > 0:
        component_count = len({find(node) for node in support_nodes})
    else:
        component_count = 0

    edge_fraction = support_edge_count / total_edges if total_edges > 0 else 0.0
    node_fraction = support_node_count / num_nodes if num_nodes > 0 else 0.0

    return {
        "edge_count": float(support_edge_count),
        "node_count": float(support_node_count),
        "component_count": float(component_count),
        "edge_fraction": float(edge_fraction),
        "node_fraction": float(node_fraction),
    }


def _project_pref_features_only(
    pref_projector: GaugeFixedPreferenceProjector,
    v_e: torch.Tensor,
) -> torch.Tensor:
    """Project v_e to satisfy X^T W v = 0 without enforcing B W v = 0."""
    if v_e.ndim != 1:
        raise ValueError("v_e must be 1D (m,).")
    if v_e.shape[0] != pref_projector.w_e.shape[0]:
        raise ValueError("v_e must match edge dimension.")
    if pref_projector.X is None:
        raise RuntimeError("Preference projector features are not initialized.")

    if v_e.device != pref_projector.w_e.device or v_e.dtype != pref_projector.w_e.dtype:
        v_e = v_e.to(device=pref_projector.w_e.device, dtype=pref_projector.w_e.dtype)

    X = pref_projector.X
    w_e = pref_projector.w_e
    weighted_X = w_e[:, None] * X
    A = X.t().matmul(weighted_X)
    if pref_projector.feature_ridge > 0.0:
        A = A + pref_projector.feature_ridge * torch.eye(
            X.shape[1],
            device=X.device,
            dtype=X.dtype,
        )
    rhs = X.t().matmul(w_e * v_e)
    alpha = torch.linalg.solve(A, rhs)
    return v_e - X.matmul(alpha)


def _index_tres_batch_files(tres_dir: Path) -> dict[tuple[str, int], tuple[Optional[Path], Optional[Path]]]:
    """
    Build an index from (flight_id, takeoff_timestamp) to (CLSR_path, WIND_path).
    """
    pattern = re.compile(r"^(?P<prefix>CLSR|WIND)_(?P<flight_id>.+)_(?P<ts>\d+)\.(?P<ext>pkl|pt)$")
    index: dict[tuple[str, int], tuple[Optional[Path], Optional[Path]]] = {}
    for batch_dir in tres_dir.glob("batch*"):
        if not batch_dir.is_dir():
            continue
        for path in batch_dir.iterdir():
            match = pattern.match(path.name)
            if match is None:
                continue
            flight_id = match.group("flight_id")
            ts = int(match.group("ts"))
            key = (flight_id, ts)
            clsr_path, wind_path = index.get(key, (None, None))
            if match.group("prefix") == "CLSR":
                clsr_path = path
            else:
                wind_path = path
            index[key] = (clsr_path, wind_path)
    return index


def _compute_mean_tailwind_per_edge(
    *,
    routes_df: pd.DataFrame,
    case_dir: str,
    edge_u_cpu: torch.Tensor,
    edge_v_cpu: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Compute a fixed per-edge mean tailwind (knots) over the training dataset.

    This provides a deterministic per-edge statistic suitable for building the
    disentanglement feature matrix X (which must be fixed for precomputing X^T D X).
    """
    if routes_df.empty:
        raise RuntimeError("Routes CSV is empty; cannot compute mean tailwind per edge for projector features.")

    if "flight_id" not in routes_df.columns:
        raise RuntimeError(
            "Routes CSV missing column 'flight_id'; cannot compute mean tailwind per edge."
        )

    # Timestamp column naming has drifted across datasets/pipelines.
    # Accept common aliases and treat values as unix seconds.
    ts_col = None
    for candidate in ("takeoff_timestamp", "takeoff_time", "takeoff"):
        if candidate in routes_df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise RuntimeError(
            "Routes CSV missing a takeoff time column (expected one of "
            "['takeoff_timestamp', 'takeoff_time', 'takeoff']); cannot compute mean tailwind per edge."
        )

    tres_dir = Path(case_dir) / "tres_runs"
    file_index = _index_tres_batch_files(tres_dir)
    if not file_index:
        raise RuntimeError(
            f"No batch directories or CLSR/WIND files found under {tres_dir}; "
            "cannot compute mean tailwind per edge. Ensure TRES thinning and wind files exist."
        )

    edge_u_list = edge_u_cpu.detach().to(device="cpu", dtype=torch.int64).tolist()
    edge_v_list = edge_v_cpu.detach().to(device="cpu", dtype=torch.int64).tolist()
    edge_to_id = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(edge_u_list, edge_v_list))}

    import numpy as np

    sum_tail = np.zeros((len(edge_u_list),), dtype=np.float64)
    cnt_tail = np.zeros((len(edge_u_list),), dtype=np.int64)
    used_flights = 0
    missing_flights = 0

    for row in routes_df.itertuples(index=False):
        flight_id = getattr(row, "flight_id")
        takeoff_timestamp = int(getattr(row, ts_col))
        clsr_path, wind_path = file_index.get((flight_id, takeoff_timestamp), (None, None))
        if clsr_path is None or wind_path is None:
            missing_flights += 1
            continue

        try:
            with open(clsr_path, "rb") as f:
                thinned_transitions = pickle.load(f)
            tailwind = torch.load(wind_path, weights_only=False)
        except Exception as e:
            logger.debug(f"Failed to load wind/thinned data for {flight_id}_{takeoff_timestamp}: {e}")
            missing_flights += 1
            continue

        tailwind_np = np.asarray(tailwind, dtype=np.float64).reshape(-1)
        if len(thinned_transitions) != tailwind_np.shape[0]:
            logger.debug(
                f"Tailwind length mismatch for {flight_id}_{takeoff_timestamp}: "
                f"{len(thinned_transitions)} transitions vs {tailwind_np.shape[0]} tailwind values."
            )
            missing_flights += 1
            continue

        for i, transition in enumerate(thinned_transitions):
            try:
                u_idx = int(transition[0])
                v_idx = int(transition[5])
            except Exception:
                continue
            edge_id = edge_to_id.get((u_idx, v_idx))
            if edge_id is None:
                continue
            sum_tail[edge_id] += float(tailwind_np[i])
            cnt_tail[edge_id] += 1

        used_flights += 1

    if used_flights == 0:
        raise RuntimeError(
            "No CLSR/WIND files matched the flights in the routes CSV; cannot compute mean tailwind per edge. "
            "This indicates tailwind is not plumbed/available for the training set."
        )

    avg_tail = np.zeros_like(sum_tail)
    mask = cnt_tail > 0
    avg_tail[mask] = sum_tail[mask] / cnt_tail[mask]
    coverage = float(mask.mean()) if mask.size else 0.0
    if missing_flights > 0:
        logger.warning(
            "Mean tailwind per edge computed from %d flights; %d flights missing CLSR/WIND files.",
            used_flights,
            missing_flights,
        )
    logger.info("Mean tailwind per edge coverage %.1f%% (edges observed in CLSR transitions).", 100.0 * coverage)
    return torch.from_numpy(avg_tail).to(dtype=torch.float64)


def _init_worker(
    config_path: str,
    case_dir: str,
    device_str: str,
    gamma: float,
    debug_single_process: bool,
    pref_edge_signature: Optional[str],
    disable_edge_preference: bool,
) -> None:
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

    pref_enabled = config.cost_model_version == "lin_disent" and not disable_edge_preference
    edge_u = None
    edge_v = None
    if pref_enabled:
        edge_u, edge_v = build_edge_list(graph, node_to_idx)
        worker_signature = _edge_list_fingerprint(edge_u, edge_v)
        if pref_edge_signature is not None and worker_signature != pref_edge_signature:
            raise RuntimeError(
                "Preference edge ordering mismatch between main and worker processes. "
                f"Expected signature {pref_edge_signature}, got {worker_signature}."
            )
        edge_u = edge_u.to(device)
        edge_v = edge_v.to(device)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "config": config,
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
    n_expected_e: Optional[torch.Tensor] = None
    t_expected_e: Optional[torch.Tensor] = None
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
    tensorboard_run_name: Optional[str] = None
    clear_tensorboard_logs: bool = False
    log_interval: int = 1  # Log every iteration by default
    randomized: bool = False
    random_seed: int = None  # Random seed for deterministic randomization
    batch_shuffling: str = "none"
    fixed_batch_index: Optional[int] = None # Use a fixed batch for all iterations
    disable_edge_preference: bool = False
    disable_pref_bw_constraint: bool = False
    
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


def _build_cost_model_from_state_dict(
    *,
    cost_model_version: str,
    num_waypoints: int,
    device: torch.device,
    cost_model_state: Dict[str, torch.Tensor],
    config: Optional[RunConfiguration] = None,
) -> torch.nn.Module:
    """Construct a lin_disent cost model instance and load weights from `cost_model_state`."""
    if cost_model_version != "lin_disent":
        raise ValueError("Only lin_disent is supported for cost model reconstruction.")

    if config is None:
        config = RunConfiguration(
            cost_model_version="lin_disent",
            cruise_speed_kts=450.0,
            common_weights=(0.0, 0.0, 0.0),
            preference_weight=1.0,
        )

    return config.build_cost_model_from_state_dict(
        cost_model_state=cost_model_state,
        num_waypoints=num_waypoints,
        device=device,
        cost_model_version=cost_model_version,
    )


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
    config = ctx['config']

    if cost_model_state is None:
        raise ValueError("cost_model_state is required to process a flight in worker mode.")

    # Reconstruct the cost model from the provided state dict.
    cost_model_version = ctx['cost_model_version']
    cost_model = _build_cost_model_from_state_dict(
        cost_model_version=cost_model_version,
        num_waypoints=ctx['num_nodes'],
        device=device,
        cost_model_state=cost_model_state,
        config=config,
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
        edge_time_sums = None
        if ctx.get("pref_enabled"):
            expected_counts, gradient, log_partition_z_tensor, edge_time_sums = backward_gradient_pass(
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
                verbose=False,
                return_edge_time_sums=True,
                cruise_speed_kts=float(cost_model.cruise_speed_kts),
            )
        else:
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
        n_expected_e = None
        t_expected_e = None
        if ctx.get("pref_enabled"):
            edge_u = ctx.get("edge_u")
            edge_v = ctx.get("edge_v")
            if edge_u is None or edge_v is None:
                raise RuntimeError("Preference edges missing in worker context.")
            expected_e = expected_counts[edge_u, edge_v]
            # Gradient of the negative log-likelihood w.r.t. the per-edge cost offset p(e):
            #   ∂NLL/∂p(e) = (N_empirical(e) - N_expected(e)) / gamma
            # With the update rule p <- p - lr * ∂NLL/∂p, this increases p(e) for edges
            # the model over-uses (expected > empirical), making them more avoided.
            pref_grad_e = (empirical_counts[edge_u, edge_v] - expected_e) / gamma
            n_expected_e = expected_e.detach()
            if edge_time_sums is None:
                raise RuntimeError("edge_time_sums was requested but missing from backward_gradient_pass.")
            t_expected_e = edge_time_sums[edge_u, edge_v].detach()

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
            success=True,
            n_expected_e=n_expected_e.cpu() if n_expected_e is not None else None,
            t_expected_e=t_expected_e.cpu() if t_expected_e is not None else None,
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
            tb_log_base_dir = batch_config.tensorboard_log_dir
        else:
            tb_log_base_dir = os.path.join(output_dir, "tensorboard_logs")

        if batch_config.tensorboard_run_name:
            tb_run_dir = os.path.join(tb_log_base_dir, batch_config.tensorboard_run_name)
        else:
            tb_run_dir = os.path.join(tb_log_base_dir, time.strftime("%Y%m%d_%H%M%S"))

        os.makedirs(tb_run_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=tb_run_dir)
        logger.info(f"TensorBoard logging enabled. Log directory: {tb_run_dir}")
        logger.info(f"To view logs, run: tensorboard --logdir {tb_log_base_dir}")
    else:
        logger.warning("TensorBoard not available. Training will proceed without TensorBoard logging.")
    
    # 1. Load configuration and initialize components
    logger.info(f"Loading configuration from {config_path}")
    config = RunConfiguration.load_from_yaml(config_path)
    components = config.initialize_all_components(cost_model_version=config.cost_model_version)
    components['cost_model_version'] = config.cost_model_version
    components['etto_delta_t_seconds'] = config.etto_delta_t_seconds

    graph_signature = None
    try:
        edge_u_cpu, edge_v_cpu = build_edge_list(components["graph"], components["node_to_idx"])
        graph_signature = _edge_list_fingerprint(edge_u_cpu, edge_v_cpu)
    except Exception as e:
        logger.warning(f"Failed to compute graph fingerprint: {e}")
    
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
    routes_df = pd.read_csv(flights_csv)
    if routes_df.empty:
        logger.error("No flights were found in the routes file. Please check the routes file and batch size.")
        return

    def make_batches(df: pd.DataFrame) -> List[pd.DataFrame]:
        return [df.iloc[i:i + batch_config.batch_size] for i in range(0, len(df), batch_config.batch_size)]

    num_batches = int(np.ceil(len(routes_df) / batch_config.batch_size))
    if num_batches == 0:
        logger.error("No flight batches were created. Please check the routes file and batch size.")
        return
    steps_per_epoch = num_batches
    epoch_idx = 0

    def shuffled_df_for_epoch(epoch_idx: int) -> pd.DataFrame:
        if batch_config.random_seed is None:
            return routes_df.sample(frac=1).reset_index(drop=True)
        return routes_df.sample(
            frac=1,
            random_state=batch_config.random_seed + epoch_idx,
        ).reset_index(drop=True)

    def batch_order_signature(batches: List[pd.DataFrame]) -> Tuple[str, List[str]]:
        hasher = hashlib.blake2b(digest_size=8)
        batch_heads: List[str] = []
        for batch in batches:
            if batch.empty:
                batch_heads.append("<empty>")
                hasher.update(b"<empty>|")
                continue
            first_row = batch.iloc[0]
            head = f"{first_row['flight_id']}:{int(first_row['takeoff_time'])}"
            batch_heads.append(head)
            hasher.update(head.encode("utf-8"))
            hasher.update(b"|")
        return hasher.hexdigest(), batch_heads

    if batch_config.batch_shuffling == "shuffle":
        flight_batches = make_batches(shuffled_df_for_epoch(epoch_idx))
        logger.info("Batch shuffling enabled: rebuilding batches each epoch.")
        batch_signature, batch_heads = batch_order_signature(flight_batches)
        logger.info(
            "Epoch %d shuffle signature=%s batch_heads=%s",
            epoch_idx,
            batch_signature,
            batch_heads,
        )
    else:
        flight_batches = make_batches(routes_df)
    logger.info(f"Created {num_batches} batches of flights.")

    pref_enabled = (
        components["cost_model_version"] == "lin_disent"
        and not batch_config.disable_edge_preference
    )
    pref_projector = None
    pref_project = None
    pref_edge_u = None
    pref_edge_v = None
    pref_support_mask = None
    pref_edge_signature = None
    pref_feature_bias_e = None
    pref_feature_ac_dist_e = None
    pref_time_fallback_e = None
    pref_laplacian_min_eig = None
    pref_laplacian_max_eig = None
    pref_laplacian_cond = None
    pref_laplacian_method = None
    pref_support_edge_fraction = None
    pref_support_node_fraction = None
    pref_support_component_count = None
    pref_support_edge_count = None
    pref_support_node_count = None
    if batch_config.disable_edge_preference and components["cost_model_version"] == "lin_disent":
        with torch.no_grad():
            components["cost_model"].preference_matrix_p.zero_()
        logger.info("Edge preference learning disabled; preference matrix fixed at 0.")
    if pref_enabled:
        edge_u_cpu, edge_v_cpu = build_edge_list(components["graph"], components["node_to_idx"])
        pref_edge_signature = _edge_list_fingerprint(edge_u_cpu, edge_v_cpu)
        pref_edge_u = edge_u_cpu.to(device)
        pref_edge_v = edge_v_cpu.to(device)

        global_counts = compute_empirical_counts_from_routes(
            routes_df["route"],
            components["node_to_idx"],
            components["num_nodes"],
        )
        d_e = global_counts[edge_u_cpu, edge_v_cpu].to(device=device, dtype=torch.float64)
        pref_projection_eps = _EPS
        w_e = d_e + pref_projection_eps

        mean_tailwind_e = _compute_mean_tailwind_per_edge(
            routes_df=routes_df,
            case_dir=case_dir,
            edge_u_cpu=edge_u_cpu,
            edge_v_cpu=edge_v_cpu,
        )
        mean_tailwind_e = mean_tailwind_e.to(device=device, dtype=torch.float64)

        X_raw = build_feature_matrix(
            pref_edge_u,
            pref_edge_v,
            components["dist_matrix"],
            components["ac_matrix"],
            cruise_speed_kts=float(components["cost_model"].cruise_speed_kts),
            tailwind_values_w=mean_tailwind_e,
            device=device,
            dtype=torch.float64,
        )
        pref_feature_bias_e = X_raw[:, 0]
        pref_feature_ac_dist_e = X_raw[:, 1]
        pref_time_fallback_e = X_raw[:, 2]
        X_norm, feature_means, feature_scales, manual_scales = d_weighted_normalize_features(
            X_raw,
            d_e,
            bias_index=0,
        )
        pref_projector = GaugeFixedPreferenceProjector(
            edge_u_cpu,
            edge_v_cpu,
            components["num_nodes"],
            w_e,
            feature_ridge=batch_config.pref_projection_ridge,
        )
        pref_projector.update_features(X_norm)
        if batch_config.disable_pref_bw_constraint:
            pref_project = lambda v_e: _project_pref_features_only(pref_projector, v_e)
            logger.info(
                "Preference projection ablation: enforcing X^T W p = 0 only (BW constraint disabled)."
            )
        else:
            pref_project = pref_projector.project
        pref_support_mask = d_e > 0

        support_stats = _support_component_stats(
            edge_u_cpu, edge_v_cpu, components["num_nodes"], pref_support_mask
        )
        pref_support_edge_count = support_stats["edge_count"]
        pref_support_node_count = support_stats["node_count"]
        pref_support_component_count = support_stats["component_count"]
        pref_support_edge_fraction = support_stats["edge_fraction"]
        pref_support_node_fraction = support_stats["node_fraction"]

        with torch.no_grad():
            (
                pref_laplacian_min_eig,
                pref_laplacian_max_eig,
                pref_laplacian_cond,
                pref_laplacian_method,
            ) = pref_projector.estimate_laplacian_spectrum()

        if pref_projector.condition_number is not None:
            logger.info(f"Preference projector A_eff condition number: {pref_projector.condition_number:.3e}")
        if pref_support_edge_fraction is not None:
            logger.info(
                "Preference support: edges=%d (%.1f%%) nodes=%d (%.1f%%) components=%d",
                int(pref_support_edge_count),
                100.0 * pref_support_edge_fraction,
                int(pref_support_node_count),
                100.0 * pref_support_node_fraction,
                int(pref_support_component_count),
            )
        if pref_laplacian_min_eig is not None and pref_laplacian_max_eig is not None:
            logger.info(
                "Preference Laplacian spectrum (%s): min=%.3e max=%.3e cond=%.3e",
                pref_laplacian_method,
                pref_laplacian_min_eig,
                pref_laplacian_max_eig,
                pref_laplacian_cond,
            )

        with torch.no_grad():
            pref_matrix = components["cost_model"].preference_matrix_p
            p_e = pref_matrix[pref_edge_u, pref_edge_v].to(dtype=X_norm.dtype)
            p_e = pref_project(p_e)
            pref_matrix.zero_()
            pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)

        with torch.no_grad():
            cost_model = components["cost_model"]
            phi_bias = pref_projector.compute_node_potential(pref_feature_bias_e)
            phi_ac = pref_projector.compute_node_potential(pref_feature_ac_dist_e)
            phi_time = pref_projector.compute_node_potential(pref_time_fallback_e)
            cost_model.phi_bias.copy_(phi_bias.to(device=cost_model.phi_bias.device, dtype=cost_model.phi_bias.dtype))
            cost_model.phi_ac_dist.copy_(phi_ac.to(device=cost_model.phi_ac_dist.device, dtype=cost_model.phi_ac_dist.dtype))
            cost_model.phi_time.copy_(phi_time.to(device=cost_model.phi_time.device, dtype=cost_model.phi_time.dtype))
    
    # 3. Initialize tracking variables
    gradient_queue: List[Dict[str, torch.Tensor]] = []
    pref_grad_queue: List[torch.Tensor] = []
    iteration = 1
    converged = False
    training_history = {
        'iterations': [],
        'avg_log_likelihood': [],
        'gradient_norms': [],
        'processing_times': [],
        'successful_flights': [],
        'failed_flights': []
    }
    
    logger.info("Starting training from scratch")
    logger.info(f"Checkpoints will be saved to {output_dir}")
    
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
        initargs=(
            config_path,
            case_dir,
            str(device),
            batch_config.gamma,
            batch_config.debug_single_process,
            pref_edge_signature,
            batch_config.disable_edge_preference,
        ),
    ) as executor:
        while iteration <= batch_config.max_iterations and not converged:
            iteration_start_time = time.time()
            
            logger.info(
                "\n--- Iteration %d/%d (epoch %d/%d) ---",
                iteration,
                batch_config.max_iterations,
                epoch_idx + 1,
                int(np.ceil(batch_config.max_iterations / steps_per_epoch)),
            )

            if batch_config.batch_shuffling == "shuffle":
                new_epoch_idx = (iteration - 1) // steps_per_epoch
                if new_epoch_idx != epoch_idx:
                    epoch_idx = new_epoch_idx
                    flight_batches = make_batches(shuffled_df_for_epoch(epoch_idx))
                    batch_signature, batch_heads = batch_order_signature(flight_batches)
                    logger.info(
                        "Reshuffled for epoch %d: signature=%s batch_heads=%s",
                        epoch_idx,
                        batch_signature,
                        batch_heads,
                    )

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
            n_expected_sum_e = None
            t_expected_sum_e = None
            pref_edge_stats_flights = 0
            if pref_enabled and pref_edge_u is not None:
                n_expected_sum_e = torch.zeros(
                    (pref_edge_u.shape[0],),
                    device=device,
                    dtype=torch.float64,
                )
                t_expected_sum_e = torch.zeros_like(n_expected_sum_e)

            for result in batch_results:
                if result.success:
                    successful_flights += 1
                    gradient_queue.append(result.gradient)
                    if pref_enabled and result.pref_grad_e is not None:
                        pref_grad_queue.append(result.pref_grad_e)
                        if result.n_expected_e is not None and result.t_expected_e is not None:
                            n_expected_sum_e += result.n_expected_e.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            t_expected_sum_e += result.t_expected_e.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            pref_edge_stats_flights += 1
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
            pref_grad_avg_norm = None
            pref_grad_proj_ratio = None
            pref_violation_features = None
            pref_violation_cycle = None
            pref_grad_cycle_residual = None
            pref_grad_potential_energy = None
            pref_grad_cycle_fraction = None
            common_cycle_violation_raw = None
            common_cycle_violation_cyc = None
            common_cost_raw_potential_energy = None
            common_cost_raw_cycle_fraction = None
            pref_min = None
            pref_max = None
            pref_mean = None
            if pref_enabled and pref_grad_queue:
                if pref_feature_bias_e is None or pref_feature_ac_dist_e is None or pref_time_fallback_e is None:
                    raise RuntimeError("Preference features not initialized; cannot rebuild preference projector.")
                if pref_projector is None:
                    raise RuntimeError("Preference projector not initialized; cannot update features.")
                if n_expected_sum_e is None or t_expected_sum_e is None:
                    n_expected_sum_e = torch.zeros_like(pref_time_fallback_e)
                    t_expected_sum_e = torch.zeros_like(pref_time_fallback_e)

                eps = _EPS
                time_support = n_expected_sum_e > eps
                x_time_batch = pref_time_fallback_e.clone()
                if time_support.any():
                    x_time_batch[time_support] = t_expected_sum_e[time_support] / n_expected_sum_e[time_support]
                nonfinite_mask = ~torch.isfinite(x_time_batch)
                if nonfinite_mask.any():
                    x_time_batch[nonfinite_mask] = pref_time_fallback_e[nonfinite_mask]

                with torch.no_grad():
                    phi_time = pref_projector.compute_node_potential(x_time_batch)
                    cost_model = components["cost_model"]
                    cost_model.phi_time.copy_(
                        phi_time.to(device=cost_model.phi_time.device, dtype=cost_model.phi_time.dtype)
                    )

                X_raw_batch = torch.stack(
                    [pref_feature_bias_e, pref_feature_ac_dist_e, x_time_batch],
                    dim=1,
                )
                X_norm, feature_means, feature_scales, manual_scales = d_weighted_normalize_features(
                    X_raw_batch,
                    d_e,
                    bias_index=0,
                )
                pref_projector.update_features(X_norm)

                stats_support = time_support
                if pref_support_mask is not None:
                    stats_support = stats_support & pref_support_mask

                common_w = components["cost_model"].common_weights.detach().to(
                    device=X_raw_batch.device, dtype=X_raw_batch.dtype
                )
                common_cost_raw_e = X_raw_batch.matmul(common_w)
                common_cycle_violation_raw = float(
                    pref_projector.cycle_violation(common_cost_raw_e).item()
                )
                with torch.no_grad():
                    common_cost_raw_potential_energy = float(
                        pref_projector.potential_energy(common_cost_raw_e).item()
                    )
                    common_cost_raw_cycle_fraction = float(
                        pref_projector.cycle_fraction(common_cost_raw_e).item()
                    )

                cost_model = components["cost_model"]
                phi_bias = cost_model.phi_bias.to(device=X_raw_batch.device, dtype=X_raw_batch.dtype)
                phi_ac = cost_model.phi_ac_dist.to(device=X_raw_batch.device, dtype=X_raw_batch.dtype)
                phi_time = cost_model.phi_time.to(device=X_raw_batch.device, dtype=X_raw_batch.dtype)
                phi_w = (
                    common_w[0] * phi_bias
                    + common_w[1] * phi_ac
                    + common_w[2] * phi_time
                )
                common_cost_cyc_e = common_cost_raw_e - (
                    phi_w[pref_edge_u] - phi_w[pref_edge_v]
                )
                common_cycle_violation_cyc = float(
                    pref_projector.cycle_violation(common_cost_cyc_e).item()
                )

                if pref_support_mask is not None and pref_support_mask.any():
                    support_frac = float(stats_support.sum().item() / pref_support_mask.sum().item())
                else:
                    support_frac = float(time_support.to(dtype=torch.float64).mean().item())

                if stats_support.any():
                    time_vals = x_time_batch[stats_support]
                    time_min = float(time_vals.min().item())
                    time_max = float(time_vals.max().item())
                    time_mean = float(time_vals.mean().item())
                else:
                    time_min = None
                    time_max = None
                    time_mean = None

                if time_mean is not None:
                    if pref_projector.condition_number is not None:
                        logger.info(
                            "Preference projector (batch): A_eff cond=%.3e | time_support=%.1f%% | time[min/mean/max]=%.6f/%.6f/%.6f",
                            pref_projector.condition_number,
                            100.0 * support_frac,
                            time_min,
                            time_mean,
                            time_max,
                        )
                    else:
                        logger.info(
                            "Preference projector (batch): time_support=%.1f%% | time[min/mean/max]=%.6f/%.6f/%.6f",
                            100.0 * support_frac,
                            time_min,
                            time_mean,
                            time_max,
                        )

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

                alpha_pref_reg = float(components["cost_model"].alpha_pref_reg.detach().cpu().item())
                if alpha_pref_reg != 0.0:
                    pref_grad_avg = pref_grad_avg + 2.0 * alpha_pref_reg * p_e

                with torch.no_grad():
                    pref_grad_cycle_residual = float(
                        pref_projector.cycle_violation(pref_grad_avg).item()
                    )
                    pref_grad_potential_energy = float(
                        pref_projector.potential_energy(pref_grad_avg).item()
                    )
                    pref_grad_cycle_fraction = float(
                        pref_projector.cycle_fraction(pref_grad_avg).item()
                    )

                pref_grad_proj = pref_project(pref_grad_avg)
                p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
                p_e = pref_project(p_e)

                with torch.no_grad():
                    pref_matrix.zero_()
                    pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)

                pref_grad_avg_norm = float(torch.linalg.norm(pref_grad_avg).item())
                pref_grad_norm = float(torch.linalg.norm(pref_grad_proj).item())
                pref_grad_proj_ratio = pref_grad_norm / (pref_grad_avg_norm + 1e-12)
                pref_violation_features = float(pref_projector.violation_features(p_e).item())
                pref_violation_cycle = float(pref_projector.violation_cycle(p_e).item())

                support_mask = pref_support_mask
                if support_mask is not None and support_mask.any():
                    p_support = p_e[support_mask]
                else:
                    p_support = p_e
                pref_min = float(p_support.min().item())
                pref_max = float(p_support.max().item())
                pref_mean = float(p_support.mean().item())

                logger.info(
                    "Preference update: grad_norm=%.6f | proj_ratio=%.6f | features=%.3e | cycle=%.3e | min=%.6f max=%.6f mean=%.6f",
                    pref_grad_norm,
                    pref_grad_proj_ratio,
                    pref_violation_features,
                    pref_violation_cycle,
                    pref_min,
                    pref_max,
                    pref_mean,
                )
                if common_cycle_violation_raw is not None:
                    logger.info(
                        "Common gauge (raw): BW(Xw) norm=%.3e",
                        common_cycle_violation_raw,
                    )
                if common_cycle_violation_cyc is not None:
                    logger.info(
                        "Common gauge (cyc): BW(X_cyc w) norm=%.3e",
                        common_cycle_violation_cyc,
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
                    if pref_grad_avg_norm is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/RawGrad_Norm_L2',
                            pref_grad_avg_norm,
                            iteration,
                        )
                    if pref_grad_proj_ratio is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Grad_Proj_Ratio',
                            pref_grad_proj_ratio,
                            iteration,
                        )
                    if pref_grad_cycle_residual is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/RawGrad_CycleResidual_L2',
                            pref_grad_cycle_residual,
                            iteration,
                        )
                    if pref_grad_potential_energy is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/RawGrad_PotentialEnergy',
                            pref_grad_potential_energy,
                            iteration,
                        )
                    if pref_grad_cycle_fraction is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/RawGrad_CycleFraction',
                            pref_grad_cycle_fraction,
                            iteration,
                        )
                    if pref_violation_features is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Feature_Constraint_Violation',
                            pref_violation_features,
                            iteration,
                        )
                    if pref_violation_cycle is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Cycle_Constraint_Violation',
                            pref_violation_cycle,
                            iteration,
                        )
                    tensorboard_writer.add_scalar('Preferences/Mean', pref_mean, iteration)
                    tensorboard_writer.add_scalar('Preferences/Min', pref_min, iteration)
                    tensorboard_writer.add_scalar('Preferences/Max', pref_max, iteration)
                    if common_cycle_violation_raw is not None:
                        tensorboard_writer.add_scalar(
                            'Common/Feature_Cycle_Violation_Raw',
                            common_cycle_violation_raw,
                            iteration,
                        )
                    if common_cost_raw_potential_energy is not None:
                        tensorboard_writer.add_scalar(
                            'Common/CommonCostRaw_PotentialEnergy',
                            common_cost_raw_potential_energy,
                            iteration,
                        )
                    if common_cost_raw_cycle_fraction is not None:
                        tensorboard_writer.add_scalar(
                            'Common/CommonCostRaw_CycleFraction',
                            common_cost_raw_cycle_fraction,
                            iteration,
                        )
                    if common_cycle_violation_cyc is not None:
                        tensorboard_writer.add_scalar(
                            'Common/Feature_Cycle_Violation_Cyc',
                            common_cycle_violation_cyc,
                            iteration,
                        )

                if pref_enabled:
                    if pref_laplacian_min_eig is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_MinEig_Est',
                            pref_laplacian_min_eig,
                            iteration,
                        )
                    if pref_laplacian_max_eig is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_MaxEig_Est',
                            pref_laplacian_max_eig,
                            iteration,
                        )
                    if pref_laplacian_cond is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_Cond_Est',
                            pref_laplacian_cond,
                            iteration,
                        )
                    if pref_support_edge_fraction is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_Support_Edge_Fraction',
                            pref_support_edge_fraction,
                            iteration,
                        )
                    if pref_support_node_fraction is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_Support_Node_Fraction',
                            pref_support_node_fraction,
                            iteration,
                        )
                    if pref_support_component_count is not None:
                        tensorboard_writer.add_scalar(
                            'Preferences/Laplacian_Support_Components',
                            pref_support_component_count,
                            iteration,
                        )
                
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
                        'batch_config': asdict(batch_config),
                        'run_config': config.to_dict(),
                        'edge_list_fingerprint': graph_signature,
                        'cost_model_version': config.cost_model_version,
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
        'batch_config': asdict(batch_config),
        'run_config': config.to_dict(),
        'edge_list_fingerprint': graph_signature,
        'cost_model_version': config.cost_model_version,
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
            'batch_config': asdict(batch_config),
            'run_config': config.to_dict(),
            'edge_list_fingerprint': graph_signature,
            'cost_model_version': config.cost_model_version,
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
        logger.info(f"TensorBoard logs saved to {tb_run_dir}")
    
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
    expected_params = [
        'source_node_idx',
        'goal_node_idx',
        'max_rho',
        'G',
        'closures',
        'wallclock_time_bin_k_tolerance_s',
        'delta_t_seconds_wall_clock',
        'include_wait_edges_in_output',
    ]
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


def _find_case_yaml(case_dir: str) -> str:
    case_path = Path(case_dir)
    if not case_path.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    default_path = case_path / "default.yaml"
    if default_path.exists():
        return str(default_path)
    yaml_paths = sorted(case_path.glob("*.yaml"))
    if not yaml_paths:
        raise FileNotFoundError(
            f"No YAML configuration file found in case directory: {case_dir}"
        )
    return str(yaml_paths[0])


def _load_training_params(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file) or {}
    training_casts = {
        "training_batch_size": int,
        "max_iters": int,
        "checkpoint_interval": int,
        "common_features_learning_rate": float,
        "preference_feature_learning_rate": float,
        "preference_projection_ridge": float,
        "convergence_threshold": float,
        "gamma": float,
    }
    for key, caster in training_casts.items():
        if key in config_data and isinstance(config_data[key], str):
            try:
                config_data[key] = caster(config_data[key])
            except ValueError as exc:
                raise ValueError(
                    f"Training param '{key}' must be {caster.__name__}, got {config_data[key]!r}."
                ) from exc
    return config_data


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
        "--num-workers", 
        type=int,
        help="Number of worker processes (default: CPU count - 1)"
    )
    parser.add_argument(
        "--device", 
        default="cpu",
        help="Device to use (only cpu is supported for the moment)"
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
        "--tensorboard-run-name",
        help="Run subdirectory name for TensorBoard logs (default: timestamp)"
    )
    parser.add_argument(
        "--clear-tensorboard-logs",
        action="store_true",
        help="Delete existing TensorBoard log directory before starting"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Interval for logging to TensorBoard (default: 1, log every iteration)"
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize which precomputed batch index is selected each iteration"
    )
    parser.add_argument(
        "--batch-shuffling",
        choices=["none", "shuffle"],
        default="none",
        help="Shuffle flights once per epoch and rebuild batches (default: none)",
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
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override max_iters from the case YAML (useful for smoke tests).",
    )
    parser.add_argument(
        "--disable-edge-preference",
        action="store_true",
        help="Disable edge preference learning and fix preference matrix at 0 (lin_disent only).",
    )
    parser.add_argument(
        "--disable-pref-bw-constraint",
        action="store_true",
        help="Disable BW preference constraint; only enforce X^T W p = 0 projection.",
    )
    
    args = parser.parse_args()

    # Set default config path if not provided
    if args.config is None:
        args.config = _find_case_yaml(args.case_dir)
    training_params = _load_training_params(args.config)

    default_batch_config = BatchLearningConfig()
    batch_size = training_params.get("training_batch_size", default_batch_config.batch_size)
    learning_rate = training_params.get("common_features_learning_rate", default_batch_config.learning_rate)
    pref_learning_rate = training_params.get("preference_feature_learning_rate", default_batch_config.pref_learning_rate)
    pref_projection_ridge = training_params.get("preference_projection_ridge", default_batch_config.pref_projection_ridge)
    max_iterations = training_params.get("max_iters", default_batch_config.max_iterations)
    if args.max_iters is not None:
        max_iterations = args.max_iters
    convergence_threshold = training_params.get("convergence_threshold", default_batch_config.convergence_threshold)
    checkpoint_interval = training_params.get("checkpoint_interval", default_batch_config.checkpoint_interval)
    gamma = training_params.get("gamma", default_batch_config.gamma)
    
    # Create batch configuration
    batch_config = BatchLearningConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        pref_learning_rate=pref_learning_rate,
        pref_projection_ridge=pref_projection_ridge,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        checkpoint_interval=checkpoint_interval,
        num_workers=args.num_workers,
        device=args.device,
        gamma=gamma,
        debug_single_process=args.debug_single_process,
        tensorboard_log_dir=args.tensorboard_log_dir,
        tensorboard_run_name=args.tensorboard_run_name,
        clear_tensorboard_logs=args.clear_tensorboard_logs,
        log_interval=args.log_interval,
        randomized=args.randomize,
        random_seed=getattr(args, 'random_seed', None),  # Handle hyphenated argument name
        batch_shuffling=args.batch_shuffling,
        fixed_batch_index=args.fixed_batch_index,
        disable_edge_preference=args.disable_edge_preference,
        disable_pref_bw_constraint=args.disable_pref_bw_constraint,
    )
    
    # Validate implementation first
    if not validate_implementation():
        logger.error("Implementation validation failed. Exiting.")
        return
    
    # Optionally delete TensorBoard log directory if requested
    if batch_config.clear_tensorboard_logs:
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
