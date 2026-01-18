import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union

import logging

# Use __name__ to ensure the logger is named after the module path
logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES = ("bias", "ac_dist", "time")


class CostLinearDisentangled(nn.Module):
    r"""
    Explicitly linear common cost with additive per-edge preferences.

    Common cost: c_common(e) = x(e)^T w
    Total cost: c(e) = c_common(e) + p(e)

    Feature definition (fixed per waypoint-edge):
      - bias = 1
      - ac_dist = AC(e) * d(e) / 100.0
      - time = 60.0 * dist / (cruise_speed_kts + tailwind_e)

    Units:
      - AC: per 100km of Boeing 737 weight load
      - cruise_speed, and w_tail in knots
      - ac_dist is in range of hundreds, time is in range of hundreds to a thousand as well.
      - This unit system will ensure that the weights will be at roughly the same magnitude.
    """
    def __init__(
        self,
        common_weights: tuple[float, float, float],
        preference_weights: float,
        alpha_pref_reg: float,
        num_waypoints: int, # total number of waypoint nodes in the city pair route graph
        device: torch.device = None,
        cruise_speed_kts: float = None,
        feature_names: Tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cpu") # forcing cpu
            logger.warning("Using CPU for training because device is None.")
        else:
            if device.type != "cpu":
                raise ValueError("device must be None to force cpu. We do not support GPU for training.")
            self.device = device

        if len(feature_names) != 3:
            raise ValueError("feature_names must have length 3 to match the built-in feature map.")

        if len(common_weights) != len(feature_names):
            raise ValueError("weights must have the same length as feature_names.")

        if cruise_speed_kts is None:
            logger.warning("Cruise speed is not provided. Using default value of 450.0 kts.")
            cruise_speed_kts = 450.0

        self.cruise_speed_kts = cruise_speed_kts

        # Initialize linear weights from the `weights` tuple.
        init_weights = torch.tensor(common_weights, dtype=torch.float32)
        self.common_weights = nn.Parameter(init_weights, requires_grad=True)
        self.preference_weights = preference_weights
        alpha_pref_value = 0.0 if alpha_pref_reg is None else alpha_pref_reg
        if isinstance(alpha_pref_value, torch.Tensor):
            alpha_pref_value = alpha_pref_value.detach().clone().to(dtype=torch.float32)
        else:
            alpha_pref_value = torch.tensor(alpha_pref_value, dtype=torch.float32)
        self.alpha_pref_reg = nn.Parameter(alpha_pref_value, requires_grad=False)
        self.feature_names = feature_names

        # Preferences live as a buffer so they are saved in the state_dict but not optimized by autograd.
        self.register_buffer(
            "preference_matrix_p",
            torch.zeros((num_waypoints, num_waypoints), dtype=torch.float32),
        )

        self.to(self.device)

    def _get_edge_metric_batched(
        self,
        u_indices: torch.Tensor,
        v_indices: torch.Tensor,
        metric_matrix: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Batched retrieval of edge-wise metrics from a square metric matrix.

        Given arrays of source (`u_indices`) and target (`v_indices`) indices,
        this function extracts the corresponding entries from `metric_matrix`,
        yielding a 1D tensor of metric values for the batch of edges.

        Args:
            u_indices (torch.Tensor): Indices of start nodes for each edge.
            v_indices (torch.Tensor): Indices of end nodes for each edge.
            metric_matrix (Union[torch.Tensor, np.ndarray]): A 2D matrix (tensor or ndarray)
                representing the metric (e.g., distance or cost) between nodes.

        Returns:
            torch.Tensor: The batch of edge metrics for the specified edges.

        Raises:
            TypeError: If `metric_matrix` is neither a torch.Tensor nor a numpy.ndarray.
            ValueError: If `metric_matrix` is not 2-dimensional.
        """
        if not isinstance(metric_matrix, torch.Tensor):
            try:
                metric_matrix = torch.from_numpy(metric_matrix)
            except (TypeError, AttributeError) as exc:
                raise TypeError(
                    f"Metric matrix must be torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}"
                ) from exc

        if metric_matrix.device != self.device:
            metric_matrix = metric_matrix.to(self.device)

        if metric_matrix.ndim != 2:
            raise ValueError("Metric matrix must be 2-dimensional.")

        return metric_matrix[u_indices, v_indices]

    def _compute_edge_features(
        self,
        u_indices: torch.Tensor,
        v_indices: torch.Tensor,
        distance_matrix_d: Union[torch.Tensor, np.ndarray],
        airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
        tailwind_values_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature vectors for the given batch of waypoint edges.

        For each edge, this method constructs a 3-dimensional feature vector consisting of:
            - 1 (bias term),
            - airspace charge per distance unit multiplied by edge distance (ac_e * dist_e / 100.0),
            - edge time in hours based on cruise speed and tailwind.

        These features are stacked for each edge to produce a batch of shape (num_edges, 3).

        Args:
            u_indices (torch.Tensor): Source node indices for the edges (shape: [num_edges]).
            v_indices (torch.Tensor): Target node indices for the edges (shape: [num_edges]).
            distance_matrix_d (Union[torch.Tensor, np.ndarray]): Matrix of pairwise node distances.
            airspace_charge_matrix_ac (Union[torch.Tensor, np.ndarray]): Matrix of airspace charges between nodes.
            tailwind_values_w (torch.Tensor): Tailwind values (knots) for each edge in the batch.

        Returns:
            torch.Tensor: Stacked feature tensor of shape (num_edges, 3) where each row is:
                [1.0, (airspace_charge * distance / 100.0), time_hours]
        """
        dist_e = self._get_edge_metric_batched(u_indices, v_indices, distance_matrix_d)
        ac_e = self._get_edge_metric_batched(u_indices, v_indices, airspace_charge_matrix_ac)
        ac_dist = ac_e * dist_e / 100.0
        tailwind_tensor = tailwind_values_w.to(device=self.device, dtype=torch.float32)
        time_e = 60.0 * dist_e / (self.cruise_speed_kts + tailwind_tensor)
        ones = torch.ones_like(dist_e)
        return torch.stack([ones, ac_dist, time_e], dim=-1)

    def get_preference_score_batched(
        self, u_indices: torch.Tensor, v_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieves preference scores P(e) from the preference matrix p for a batch of edges e = (u, v).
        """
        return self._get_edge_metric_batched(u_indices, v_indices, self.preference_matrix_p)

    def forward(
        self,
        edge_indices: Tuple[torch.Tensor, torch.Tensor],
        distance_matrix_d: Union[torch.Tensor, np.ndarray],
        airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
        tailwind_values_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total cost for a batch of waypoint edges.

        This method combines both the common linear cost and the per-edge preference for each edge in the batch:
            total_cost(e) = common_weights @ features(e) + preference_matrix_p[u, v]

        Args:
            edge_indices (Tuple[torch.Tensor, torch.Tensor]): Tuple of (u_indices, v_indices) representing the source and target node indices of each edge in the batch.
            distance_matrix_d (Union[torch.Tensor, np.ndarray]): Pairwise distance matrix between nodes.
            airspace_charge_matrix_ac (Union[torch.Tensor, np.ndarray]): Pairwise airspace charge matrix between nodes.
            tailwind_values_w (torch.Tensor): Tailwind values (knots) for each edge in the batch.
        
        Returns:
            torch.Tensor: Total cost for each edge in the batch (shape: [num_edges,]).
        """
        u_indices, v_indices = edge_indices

        features = self._compute_edge_features(
            u_indices,
            v_indices,
            distance_matrix_d,
            airspace_charge_matrix_ac,
            tailwind_values_w,
        )

        common_cost = features @ self.common_weights.to(dtype=features.dtype)
        pref_cost = self.get_preference_score_batched(u_indices, v_indices)
        total_cost = common_cost + pref_cost

        dist_e = self._get_edge_metric_batched(u_indices, v_indices, distance_matrix_d)
        inf_mask = torch.isinf(dist_e)
        total_cost = total_cost.clone()
        total_cost[inf_mask] = float("inf")

        return total_cost
