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
          time = 60.0 * dist / (cruise_speed_kts + tailwind_e)
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

    time_e = 60.0 * dist / (cruise_speed_kts + tailwind_e)
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


def _infer_reference_nodes(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    num_nodes: int,
    ref_nodes: Optional[Iterable[int]] = None,
) -> tuple[list[int], int]:
    edge_u_cpu = edge_u.detach().to(device="cpu", dtype=torch.int64).tolist()
    edge_v_cpu = edge_v.detach().to(device="cpu", dtype=torch.int64).tolist()

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

    for left, right in zip(edge_u_cpu, edge_v_cpu):
        union(left, right)

    root_to_comp: dict[int, int] = {}
    comp_for_node: list[int] = []
    comp_to_rep: list[int] = []
    for node in range(num_nodes):
        root = find(node)
        comp_id = root_to_comp.get(root)
        if comp_id is None:
            comp_id = len(root_to_comp)
            root_to_comp[root] = comp_id
            comp_to_rep.append(node)
        comp_for_node.append(comp_id)

    ref_list: list[int] = []
    if ref_nodes is not None:
        if isinstance(ref_nodes, torch.Tensor):
            ref_list = ref_nodes.detach().to(device="cpu", dtype=torch.int64).tolist()
        else:
            ref_list = list(ref_nodes)
    ref_set = {int(node) for node in ref_list}
    for node in ref_set:
        if node < 0 or node >= num_nodes:
            raise ValueError(f"Reference node {node} is out of range for {num_nodes} nodes.")

    pinned_components = {comp_for_node[node] for node in ref_set} if ref_set else set()
    pins = set(ref_set)
    for comp_id, rep in enumerate(comp_to_rep):
        if comp_id not in pinned_components:
            pins.add(rep)

    if ref_nodes is not None and len(pins) > len(ref_set):
        logger.warning(
            "Reference nodes did not cover all components; added %d extra pins.",
            len(pins) - len(ref_set),
        )

    return sorted(pins), len(comp_to_rep)


class GaugeFixedPreferenceProjector:
    """Projects onto X^T W p = 0 and B W p = 0 using a stable Schur complement."""

    def __init__(
        self,
        edge_u: torch.Tensor,
        edge_v: torch.Tensor,
        num_nodes: int,
        w_e: torch.Tensor,
        *,
        ref_nodes: Optional[Iterable[int]] = None,
        laplacian_ridge: float = 0.0,
        feature_ridge: float = 1e-8,
    ) -> None:
        if edge_u.ndim != 1 or edge_v.ndim != 1:
            raise ValueError("edge_u and edge_v must be 1D (m,).")
        if edge_u.shape != edge_v.shape:
            raise ValueError("edge_u and edge_v must have the same shape.")
        if w_e.ndim != 1:
            raise ValueError("w_e must be 1D (m,).")
        if w_e.shape[0] != edge_u.shape[0]:
            raise ValueError("w_e must match edge_u/edge_v length.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if laplacian_ridge < 0.0:
            raise ValueError("laplacian_ridge must be non-negative.")
        if feature_ridge < 0.0:
            raise ValueError("feature_ridge must be non-negative.")

        self.num_nodes = num_nodes
        self.w_e = w_e
        self.laplacian_ridge = float(laplacian_ridge)
        self.feature_ridge = float(feature_ridge)

        device = w_e.device
        self.edge_u = edge_u.to(device=device, dtype=torch.long)
        self.edge_v = edge_v.to(device=device, dtype=torch.long)

        pins, num_components = _infer_reference_nodes(
            self.edge_u, self.edge_v, num_nodes, ref_nodes
        )
        if num_components > 1:
            logger.warning("Preference graph has %d connected components.", num_components)
        self.ref_nodes = pins

        node_to_reduced = [-1] * num_nodes
        reduced_idx = 0
        pin_set = set(pins)
        for node in range(num_nodes):
            if node in pin_set:
                continue
            node_to_reduced[node] = reduced_idx
            reduced_idx += 1
        if reduced_idx == 0:
            raise ValueError("All nodes are pinned; projection is ill-defined.")

        self.node_to_reduced = torch.tensor(
            node_to_reduced, device=device, dtype=torch.long
        )
        self.reduced_node_count = reduced_idx

        self._row_u = self.node_to_reduced[self.edge_u]
        self._row_v = self.node_to_reduced[self.edge_v]
        self._mask_u = self._row_u >= 0
        self._mask_v = self._row_v >= 0
        self._mask_both = self._mask_u & self._mask_v

        self._laplacian = None
        self._laplacian_chol = None
        self._build_laplacian()

        self.X = None
        self._C = None
        self._Z = None
        self._A_eff = None
        self._A_eff_chol = None
        self.condition_number = None

    def _build_laplacian(self) -> None:
        device = self.w_e.device
        dtype = self.w_e.dtype
        laplacian = torch.zeros(
            (self.reduced_node_count, self.reduced_node_count),
            device=device,
            dtype=dtype,
        )
        if self._mask_u.any():
            idx_u = self._row_u[self._mask_u]
            laplacian.index_put_((idx_u, idx_u), self.w_e[self._mask_u], accumulate=True)
        if self._mask_v.any():
            idx_v = self._row_v[self._mask_v]
            laplacian.index_put_((idx_v, idx_v), self.w_e[self._mask_v], accumulate=True)
        if self._mask_both.any():
            idx_u = self._row_u[self._mask_both]
            idx_v = self._row_v[self._mask_both]
            weights = self.w_e[self._mask_both]
            laplacian.index_put_((idx_u, idx_v), -weights, accumulate=True)
            laplacian.index_put_((idx_v, idx_u), -weights, accumulate=True)

        if self.laplacian_ridge > 0.0:
            laplacian.diagonal().add_(self.laplacian_ridge)

        self._laplacian = laplacian
        self._laplacian_chol = None
        try:
            self._laplacian_chol = torch.linalg.cholesky(laplacian)
        except RuntimeError:
            logger.warning("Cholesky failed for the Laplacian; falling back to solve().")

    def _solve_laplacian(self, rhs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if rhs.ndim == 1:
            rhs = rhs.unsqueeze(1)
            squeeze = True
        if self._laplacian_chol is not None:
            sol = torch.cholesky_solve(rhs, self._laplacian_chol)
        else:
            sol = torch.linalg.solve(self._laplacian, rhs)
        return sol.squeeze(1) if squeeze else sol

    def _solve_aeff(self, rhs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if rhs.ndim == 1:
            rhs = rhs.unsqueeze(1)
            squeeze = True
        if self._A_eff_chol is not None:
            sol = torch.cholesky_solve(rhs, self._A_eff_chol)
        else:
            sol = torch.linalg.solve(self._A_eff, rhs)
        return sol.squeeze(1) if squeeze else sol

    def _bw_apply(self, v_e: torch.Tensor) -> torch.Tensor:
        weighted_v = self.w_e * v_e
        bwv = torch.zeros(
            self.reduced_node_count, device=v_e.device, dtype=weighted_v.dtype
        )
        if self._mask_u.any():
            bwv.index_add_(0, self._row_u[self._mask_u], weighted_v[self._mask_u])
        if self._mask_v.any():
            bwv.index_add_(0, self._row_v[self._mask_v], -weighted_v[self._mask_v])
        return bwv

    def _bt_apply(self, psi: torch.Tensor) -> torch.Tensor:
        bpsi = torch.zeros_like(self.w_e, dtype=psi.dtype, device=psi.device)
        if self._mask_u.any():
            bpsi[self._mask_u] += psi[self._row_u[self._mask_u]]
        if self._mask_v.any():
            bpsi[self._mask_v] -= psi[self._row_v[self._mask_v]]
        return bpsi

    def update_features(self, X: torch.Tensor) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D (m, d).")
        if X.shape[0] != self.w_e.shape[0]:
            raise ValueError("X must match edge dimension.")
        if X.device != self.w_e.device or X.dtype != self.w_e.dtype:
            X = X.to(device=self.w_e.device, dtype=self.w_e.dtype)

        self.X = X
        weighted_X = self.w_e[:, None] * X
        A = X.t().matmul(weighted_X)

        C = torch.zeros(
            (self.reduced_node_count, X.shape[1]),
            device=X.device,
            dtype=X.dtype,
        )
        if self._mask_u.any():
            C.index_add_(0, self._row_u[self._mask_u], weighted_X[self._mask_u])
        if self._mask_v.any():
            C.index_add_(0, self._row_v[self._mask_v], -weighted_X[self._mask_v])

        Z = self._solve_laplacian(C)
        A_eff = A - C.t().matmul(Z)
        if self.feature_ridge > 0.0:
            A_eff = A_eff + self.feature_ridge * torch.eye(
                X.shape[1], device=X.device, dtype=X.dtype
            )

        self._C = C
        self._Z = Z
        self._A_eff = A_eff
        self._A_eff_chol = None
        try:
            self._A_eff_chol = torch.linalg.cholesky(A_eff)
        except RuntimeError:
            logger.warning("Cholesky failed for A_eff; falling back to solve().")

        self.condition_number = None
        try:
            self.condition_number = float(torch.linalg.cond(A_eff).item())
        except RuntimeError:
            logger.warning("Could not compute condition number for A_eff.")

    def project(self, v_e: torch.Tensor) -> torch.Tensor:
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if self.X is None or self._A_eff is None or self._C is None or self._Z is None:
            raise RuntimeError("update_features() must be called before project().")

        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)

        rhs1 = self.X.t().matmul(self.w_e * v_e)
        rhs2 = self._bw_apply(v_e)
        z2 = self._solve_laplacian(rhs2)
        rhs1_eff = rhs1 - self._C.t().matmul(z2)
        alpha = self._solve_aeff(rhs1_eff)
        psi = z2 - self._Z.matmul(alpha)
        return v_e - self.X.matmul(alpha) - self._bt_apply(psi)

    def compute_node_potential(self, v_e: torch.Tensor) -> torch.Tensor:
        """Solve L phi = B W v and return full node potentials with pinned nodes set to 0."""
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)

        rhs = self._bw_apply(v_e)
        phi_reduced = self._solve_laplacian(rhs)
        phi_full = torch.zeros(self.num_nodes, device=phi_reduced.device, dtype=phi_reduced.dtype)
        mask = self.node_to_reduced >= 0
        if mask.any():
            phi_full[mask] = phi_reduced[self.node_to_reduced[mask]]
        return phi_full

    def cycle_project_vector(self, v_e: torch.Tensor) -> torch.Tensor:
        """Project v_e onto ker(B W) by subtracting a gradient component."""
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)

        psi = self._solve_laplacian(self._bw_apply(v_e))
        return v_e - self._bt_apply(psi)

    def cycle_project_features(self, X: torch.Tensor) -> torch.Tensor:
        """Project each column of X onto ker(B W) without forming dense matrices."""
        if X.ndim != 2:
            raise ValueError("X must be 2D (m, d).")
        if X.shape[0] != self.w_e.shape[0]:
            raise ValueError("X must match edge dimension.")
        if X.device != self.w_e.device or X.dtype != self.w_e.dtype:
            X = X.to(device=self.w_e.device, dtype=self.w_e.dtype)

        weighted_X = self.w_e[:, None] * X
        C = torch.zeros(
            (self.reduced_node_count, X.shape[1]),
            device=X.device,
            dtype=X.dtype,
        )
        if self._mask_u.any():
            C.index_add_(0, self._row_u[self._mask_u], weighted_X[self._mask_u])
        if self._mask_v.any():
            C.index_add_(0, self._row_v[self._mask_v], -weighted_X[self._mask_v])

        Z = self._solve_laplacian(C)
        btx = torch.zeros_like(X)
        if self._mask_u.any():
            btx[self._mask_u] += Z[self._row_u[self._mask_u]]
        if self._mask_v.any():
            btx[self._mask_v] -= Z[self._row_v[self._mask_v]]

        return X - btx

    def cycle_violation(self, v_e: torch.Tensor) -> torch.Tensor:
        """Return ||B W v_e||_2 for diagnostics."""
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)
        return torch.linalg.norm(self._bw_apply(v_e))

    def potential_energy(self, v_e: torch.Tensor) -> torch.Tensor:
        """Return (B W v)^T L^{-1} (B W v) for diagnostics."""
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)
        rhs = self._bw_apply(v_e)
        phi = self._solve_laplacian(rhs)
        return torch.dot(rhs, phi)

    def w_energy(self, v_e: torch.Tensor) -> torch.Tensor:
        """Return v^T W v for diagnostics."""
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)
        return torch.dot(self.w_e * v_e, v_e)

    def cycle_fraction(self, v_e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Return 1 - E_pot / (v^T W v + eps) for diagnostics."""
        if eps < 0.0:
            raise ValueError("eps must be non-negative.")
        if v_e.ndim != 1:
            raise ValueError("v_e must be 1D (m,).")
        if v_e.shape[0] != self.w_e.shape[0]:
            raise ValueError("v_e must match edge dimension.")
        if v_e.device != self.w_e.device or v_e.dtype != self.w_e.dtype:
            v_e = v_e.to(device=self.w_e.device, dtype=self.w_e.dtype)
        rhs = self._bw_apply(v_e)
        phi = self._solve_laplacian(rhs)
        potential = torch.dot(rhs, phi)
        w_energy = torch.dot(self.w_e * v_e, v_e)
        denom = w_energy + w_energy.new_tensor(eps)
        return w_energy.new_tensor(1.0) - potential / denom

    def violation_features(self, p_e: torch.Tensor) -> torch.Tensor:
        if p_e.ndim != 1:
            raise ValueError("p_e must be 1D (m,).")
        return torch.linalg.norm(self.X.t().matmul(self.w_e * p_e))

    def violation_cycle(self, p_e: torch.Tensor) -> torch.Tensor:
        if p_e.ndim != 1:
            raise ValueError("p_e must be 1D (m,).")
        return torch.linalg.norm(self._bw_apply(p_e))
