import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union

DEFAULT_FEATURE_NAMES = ("bias", "ac_dist", "dist")


class CostLinearDisentangled(nn.Module):
    r"""
    Explicitly linear common cost with additive per-edge preferences.

    Common cost: c_common(e) = x(e)^T w
    Total cost: c(e) = c_common(e) + p(e)

    Feature definition (fixed per waypoint-edge):
      - bias = 1
      - ac_dist = AC(e) * d(e) / 100.0
      - dist = d(e)
    """
    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,  # unused, kept for API compatibility
        alpha_pref_reg: float,
        num_waypoints: int,
        device: torch.device = None,
        feature_names: Tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if len(feature_names) != 3:
            raise ValueError("feature_names must have length 3 to match the built-in feature map.")

        if isinstance(alpha_pref_reg, torch.Tensor):
            self.alpha_pref_reg = nn.Parameter(
                alpha_pref_reg.detach().clone().to(dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.alpha_pref_reg = nn.Parameter(
                torch.tensor(alpha_pref_reg, dtype=torch.float32),
                requires_grad=False,
            )

        # Initialize linear weights from beta0/beta1/beta2 for convenience.
        init_weights = torch.tensor([beta0, beta1, beta2], dtype=torch.float32)
        self.common_weights = nn.Parameter(init_weights, requires_grad=True)
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
    ) -> torch.Tensor:
        dist_e = self._get_edge_metric_batched(u_indices, v_indices, distance_matrix_d)
        ac_e = self._get_edge_metric_batched(u_indices, v_indices, airspace_charge_matrix_ac)
        ac_dist = ac_e * dist_e / 100.0
        ones = torch.ones_like(dist_e)
        return torch.stack([ones, ac_dist, dist_e], dim=-1)

    def get_preference_score_batched(
        self, u_indices: torch.Tensor, v_indices: torch.Tensor
    ) -> torch.Tensor:
        return self._get_edge_metric_batched(u_indices, v_indices, self.preference_matrix_p)

    def forward(
        self,
        edge_indices: Tuple[torch.Tensor, torch.Tensor],
        distance_matrix_d: Union[torch.Tensor, np.ndarray],
        airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
        tailwind_values_w: torch.Tensor,
    ) -> torch.Tensor:
        u_indices, v_indices = edge_indices

        features = self._compute_edge_features(
            u_indices,
            v_indices,
            distance_matrix_d,
            airspace_charge_matrix_ac,
        )

        # Ignore tailwind_values_w: features are fixed per waypoint-edge.
        common_cost = features @ self.common_weights
        pref_cost = self.get_preference_score_batched(u_indices, v_indices)
        total_cost = common_cost + pref_cost

        dist_e = features[:, 2]
        inf_mask = torch.isinf(dist_e)
        total_cost = total_cost.clone()
        total_cost[inf_mask] = float("inf")

        return total_cost
