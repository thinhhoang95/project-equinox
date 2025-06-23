import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

class CostRev4Lite(nn.Module):
    r"""
    Cost function c(edge) for an edge 'e'.
    The cost function is defined as:
    cost(edge) = distance * (airspace_charge/100 - 1.25 * tailwind_plus/450) 
    where tailwind_plus = max(0, tailwind).

    This module is designed to be a "plug-in" replacement for CostRev4,
    but with a fixed cost function and no learnable parameters.
    """
    def __init__(self, device=None, **kwargs):
        """
        Initializes the CostRev4Lite module.
        Args:
            device (torch.device, optional): The device to run the model on. Defaults to None.
            **kwargs: Catches unused arguments for API compatibility with other cost functions.
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # No parameters to learn.
        # The kwargs are there to absorb other parameters from CostRev4 for compatibility
        
        self.to(self.device)

    def _get_edge_metric_batched(self, u_indices: torch.Tensor, v_indices: torch.Tensor, metric_matrix: torch.Tensor) -> torch.Tensor:
        """
        Retrieves metrics for a batch of edges (u, v) from a 2D metric matrix.
        u_indices and v_indices are 1D tensors of the same length.
        """
        if not isinstance(metric_matrix, torch.Tensor):
            try:
                metric_matrix = torch.from_numpy(metric_matrix).to(dtype=torch.float32, device=self.device)
            except (TypeError, AttributeError):
                 raise TypeError(f"Metric matrix (e.g., D, AC) must be a torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}")

        if metric_matrix.device != self.device:
            metric_matrix = metric_matrix.to(self.device)
            
        if metric_matrix.ndim != 2:
            raise ValueError("Metric matrix must be 2-dimensional.")
            
        return metric_matrix[u_indices, v_indices]

    def get_distance_batched(self, u_indices: torch.Tensor, v_indices: torch.Tensor, distance_matrix_d: torch.Tensor) -> torch.Tensor:
        """
        Retrieves distances d(e) for a batch of edges e = (u, v).
        """
        return self._get_edge_metric_batched(u_indices, v_indices, distance_matrix_d)

    def get_airspace_charge_batched(self, u_indices: torch.Tensor, v_indices: torch.Tensor, airspace_charge_matrix_ac: torch.Tensor) -> torch.Tensor:
        """
        Retrieves airspace charges AC(e) for a batch of edges e = (u, v).
        """
        return self._get_edge_metric_batched(u_indices, v_indices, airspace_charge_matrix_ac)

    def forward(self,
                edge_indices: Tuple[torch.Tensor, torch.Tensor],
                distance_matrix_d: Union[torch.Tensor, np.ndarray],
                airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
                tailwind_values_w: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Calculates the cost for a batch of edges.

        cost(edge) = distance * (airspace_charge/100 - 1.25 * tailwind_plus/450)
        where tailwind_plus = max(0, tailwind)

        Args:
            edge_indices: Tuple (u_indices, v_indices) of 1D integer tensors for start and end nodes.
            distance_matrix_d: 2D torch.Tensor or numpy.ndarray for distances D. D[i,j] = d(i,j).
            airspace_charge_matrix_ac: 2D torch.Tensor or numpy.ndarray for airspace charges AC.
            tailwind_values_w: 1D torch.Tensor for tailwind w_tail(e, t_e) for each edge.
                               Positive for tailwind, negative for headwind.
            **kwargs: Catches unused arguments for API compatibility with other cost functions.

        Returns:
            A 1D torch.Tensor representing the costs of the edges.
        """
        u_indices, v_indices = edge_indices

        dist_e_batch = self.get_distance_batched(u_indices, v_indices, distance_matrix_d)

        # For edges with infinite distance, cost should be infinite.
        inf_mask = torch.isinf(dist_e_batch)

        ac_e_batch = self.get_airspace_charge_batched(u_indices, v_indices, airspace_charge_matrix_ac)
        
        tailwind_tensor_batch = tailwind_values_w.to(device=self.device, dtype=torch.float32)
        
        tailwind_plus = torch.relu(tailwind_tensor_batch)

        cost_component_ac = ac_e_batch / 100.0
        cost_component_wind = 1.25 * tailwind_plus / 450.0

        total_cost_batch = dist_e_batch * (cost_component_ac - cost_component_wind)
        
        total_cost_batch[inf_mask] = float('inf')
        
        return total_cost_batch

    