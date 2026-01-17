import logging
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import networkx as nx

logger = logging.getLogger(__name__)


def build_edge_list(
    graph: nx.DiGraph, node_to_idx: dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic edge list (u_idx, v_idx) from the base waypoint graph."""
    edges = []
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))
    edges.sort()
    edge_u = torch.tensor([e[0] for e in edges], dtype=torch.long)
    edge_v = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return edge_u, edge_v


def compute_empirical_counts_from_routes(
    routes: Iterable[str],
    node_to_idx: dict[str, int],
    num_nodes: int,
) -> torch.Tensor:
    """Compute dense empirical counts for all routes in the dataset."""
    counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    for route in routes:
        if not isinstance(route, str):
            continue
        waypoints = route.split()
        for i in range(len(waypoints) - 1):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]
            if from_wp in node_to_idx and to_wp in node_to_idx:
                counts[node_to_idx[from_wp], node_to_idx[to_wp]] += 1.0
    return counts


def _ensure_tensor(
    matrix: Union[torch.Tensor, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        return matrix.to(device=device, dtype=dtype)
    if isinstance(matrix, np.ndarray):
        return torch.from_numpy(matrix).to(device=device, dtype=dtype)
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(matrix)}")


def build_feature_matrix(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    distance_matrix_d: Union[torch.Tensor, np.ndarray],
    airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
    *,
    cruise_speed_kts: float = 450.0,
    tailwind_values_w: Optional[Union[torch.Tensor, np.ndarray, float]] = None,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build per-edge feature matrix X for the linear common cost model.

    Feature order must match CostLinearDisentangled:
      [bias, ac_dist, time]

    Notes:
      - The cost model defines time as:
          time = dist / (60 * (cruise_speed_kts + tailwind))
      - If you want a *static* projector, `tailwind_values_w` must be a fixed per-edge statistic
        (e.g., climatology mean tailwind per edge) so that X is time-invariant.
      - If your workflow rebuilds the projector over time, prefer supplying a precomputed per-edge
        time column via `build_feature_matrix_from_time(...)`.
      - If omitted, tailwind is assumed to be 0, making time proportional to distance.
    """
    dist = _ensure_tensor(distance_matrix_d, device=device, dtype=dtype)[edge_u, edge_v]
    ac = _ensure_tensor(airspace_charge_matrix_ac, device=device, dtype=dtype)[edge_u, edge_v]
    ac_dist = ac * dist / 100.0
    if cruise_speed_kts <= 0:
        raise ValueError("cruise_speed_kts must be positive.")

    if tailwind_values_w is None:
        tailwind_e = torch.zeros_like(dist)
    elif isinstance(tailwind_values_w, (float, int)):
        tailwind_e = torch.full_like(dist, float(tailwind_values_w))
    else:
        tailwind_tensor = _ensure_tensor(tailwind_values_w, device=device, dtype=dtype)
        if tailwind_tensor.ndim == 2:
            tailwind_e = tailwind_tensor[edge_u, edge_v]
        elif tailwind_tensor.ndim == 1:
            if tailwind_tensor.shape[0] != edge_u.shape[0]:
                raise ValueError(
                    "tailwind_values_w must be 1D with length matching the edge list, "
                    "or 2D with shape (num_nodes, num_nodes)."
                )
            tailwind_e = tailwind_tensor
        elif tailwind_tensor.ndim == 0:
            tailwind_e = tailwind_tensor.expand_as(dist)
        else:
            raise ValueError("tailwind_values_w must be a scalar, 1D, or 2D tensor/array.")

    time_e = dist / (60.0 * (cruise_speed_kts + tailwind_e))
    ones = torch.ones_like(dist)
    return torch.stack([ones, ac_dist, time_e], dim=1)


def build_feature_matrix_from_time(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    distance_matrix_d: Union[torch.Tensor, np.ndarray],
    airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
    *,
    time_values: Union[torch.Tensor, np.ndarray, float],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build per-edge feature matrix X when the time column is already computed per edge.

    Feature order matches CostLinearDisentangled:
      [bias, ac_dist, time]
    """
    dist = _ensure_tensor(distance_matrix_d, device=device, dtype=dtype)[edge_u, edge_v]
    ac = _ensure_tensor(airspace_charge_matrix_ac, device=device, dtype=dtype)[edge_u, edge_v]
    ac_dist = ac * dist / 100.0

    if isinstance(time_values, (float, int)):
        time_e = torch.full_like(dist, float(time_values))
    else:
        time_tensor = _ensure_tensor(time_values, device=device, dtype=dtype)
        if time_tensor.ndim == 2:
            time_e = time_tensor[edge_u, edge_v]
        elif time_tensor.ndim == 1:
            if time_tensor.shape[0] != edge_u.shape[0]:
                raise ValueError(
                    "time_values must be 1D with length matching the edge list, "
                    "or 2D with shape (num_nodes, num_nodes)."
                )
            time_e = time_tensor
        elif time_tensor.ndim == 0:
            time_e = time_tensor.expand_as(dist)
        else:
            raise ValueError("time_values must be a scalar, 1D, or 2D tensor/array.")

    ones = torch.ones_like(dist)
    return torch.stack([ones, ac_dist, time_e], dim=1)


def d_weighted_normalize_features(
    X: torch.Tensor,
    d_e: torch.Tensor,
    *,
    bias_index: int = 0,
    eps: float = 1e-12,
    manual_scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Center/scale columns under D-weighted statistics, keeping the bias intact."""
    weights = d_e
    weight_sum = weights.sum()
    if weight_sum.item() <= 0:
        logger.warning("Empirical weights sum to 0; using uniform weights for normalization.")
        weights = torch.ones_like(weights)
        weight_sum = weights.sum()

    means = (weights[:, None] * X).sum(dim=0) / weight_sum
    centered = X - means
    variances = (weights[:, None] * centered.pow(2)).sum(dim=0) / weight_sum
    scales = torch.sqrt(variances.clamp_min(eps))

    if 0 <= bias_index < X.shape[1]:
        means = means.clone()
        scales = scales.clone()
        means[bias_index] = 0.0
        scales[bias_index] = 1.0

    X_norm = (X - means) / scales

    if manual_scales is not None:
        manual_scales = manual_scales.to(device=X.device, dtype=X.dtype)
        X_norm = X_norm * manual_scales

    return X_norm, means, scales, manual_scales


class PreferenceProjector:
    """Applies the D-weighted projection operator P_perp,D without forming a full matrix."""

    def __init__(
        self,
        X: torch.Tensor,
        d_e: torch.Tensor,
        *,
        ridge: float = 1e-8,
        feature_means: Optional[torch.Tensor] = None,
        feature_scales: Optional[torch.Tensor] = None,
        manual_scales: Optional[torch.Tensor] = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D (m, d).")
        if d_e.ndim != 1:
            raise ValueError("d_e must be 1D (m,).")
        if X.shape[0] != d_e.shape[0]:
            raise ValueError("X and d_e must agree on the edge dimension.")

        self.X = X
        self.d_e = d_e
        self.ridge = ridge
        self.feature_means = feature_means
        self.feature_scales = feature_scales
        self.manual_scales = manual_scales

        weighted_X = d_e[:, None] * X
        dim = X.shape[1]
        self.M = X.t().matmul(weighted_X) + ridge * torch.eye(dim, device=X.device, dtype=X.dtype)

        self._chol = None
        try:
            self._chol = torch.linalg.cholesky(self.M)
        except RuntimeError:
            logger.warning("Cholesky failed for X^T D X; falling back to solve().")

        self.condition_number = None
        try:
            self.condition_number = float(torch.linalg.cond(self.M).item())
        except RuntimeError:
            logger.warning("Could not compute condition number for X^T D X.")

    def project(self, v_e: torch.Tensor) -> torch.Tensor:
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.X.shape[0]:
            raise ValueError("v_e must match X edge dimension.")

        rhs = self.X.t().matmul(self.d_e * v_e)
        if self._chol is not None:
            alpha = torch.cholesky_solve(rhs.unsqueeze(1), self._chol).squeeze(1)
        else:
            alpha = torch.linalg.solve(self.M, rhs)
        return v_e - self.X.matmul(alpha)

    def constraint_violation(self, p_e: torch.Tensor) -> torch.Tensor:
        if p_e.ndim != 1:
            raise ValueError("p_e must be 1D (m,).")
        return torch.linalg.norm(self.X.t().matmul(self.d_e * p_e))
