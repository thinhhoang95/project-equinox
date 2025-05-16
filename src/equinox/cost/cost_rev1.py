import torch
import torch.nn as nn
from equinox.cost.plf_mono import PiecewiseLinearMonoModel
import numpy as np
from typing import Union, Tuple

# Default knot points
DEFAULT_KNOTS_AC_DIST = [100.0, 500.0, 2000.0, 5000.0, 10000.0] # euros
DEFAULT_KNOTS_WIND = [-80.0, -40.0, -10.0, 0.0, 10.0, 40.0, 80.0] # knots

class CostRev1(nn.Module):
    r"""
    Cost function c(e, t_e, beta) for an edge 'e'.
    The cost implicitly depends on time 't_e' through the 'tailwind_value_w' input,
    which should be w_tail(e, t_e).

    The formula is:
    c(e, t_e, beta) = beta_0 + beta_1 * PLM_ac(AC(e) * d(e)) + beta_2 * PLM_wind(w_tail(e, t_e))

    Where:
    - beta_i are coefficients (learnable or fixed).
    - PLM_ac is a monotonically non-decreasing piecewise linear model for the airspace charge component.
    - PLM_wind is a monotonically non-increasing piecewise linear model for the tailwind component.
    - AC(e) is the airspace charge for edge e.
    - d(e) is the great circle distance of edge e.
    - w_tail(e, t_e) is the tailwind on edge e at time t_e.
    """
    def __init__(self,
                 beta0: float,
                 beta1: float,
                 beta2: float,
                 knots_ac_dist: list[float] = None,
                 knots_wind: list[float] = None,
                 device: torch.device = None):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Coefficients (nn.Parameter for flexibility, can be fixed with requires_grad=False)
        self.beta0 = nn.Parameter(torch.tensor(beta0, dtype=torch.float32), requires_grad=False)
        self.beta1 = nn.Parameter(torch.tensor(beta1, dtype=torch.float32), requires_grad=False)
        self.beta2 = nn.Parameter(torch.tensor(beta2, dtype=torch.float32), requires_grad=False)

        _knots_ac_dist = knots_ac_dist if knots_ac_dist is not None else DEFAULT_KNOTS_AC_DIST
        _knots_wind = knots_wind if knots_wind is not None else DEFAULT_KNOTS_WIND

        self.plm_ac_dist = PiecewiseLinearMonoModel(
            knot_points=_knots_ac_dist,
            monotonic_type="non_decreasing"  # Higher AC*dist -> higher cost contribution
        )
        self.plm_wind = PiecewiseLinearMonoModel(
            knot_points=_knots_wind,
            monotonic_type="non_increasing"  # Higher tailwind -> lower cost contribution
        )
        
        self.to(self.device) # Move all parameters and buffers to the specified device

    def _get_edge_metric_batched(self, u_indices: torch.Tensor, v_indices: torch.Tensor, metric_matrix: torch.Tensor) -> torch.Tensor:
        """
        Retrieves metrics for a batch of edges (u, v) from a 2D metric matrix.
        u_indices and v_indices are 1D tensors of the same length.
        """
        if not isinstance(metric_matrix, torch.Tensor):
            try:
                metric_matrix = torch.from_numpy(metric_matrix).to(dtype=torch.float32, device=self.device)
            except TypeError:
                 raise TypeError(f"Metric matrix (e.g., D, AC) must be a torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}")
            except AttributeError:
                 raise TypeError(f"Metric matrix (e.g., D, AC) must be a torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}")

        if metric_matrix.device != self.device:
            metric_matrix = metric_matrix.to(self.device)
            
        if metric_matrix.ndim != 2:
            raise ValueError("Metric matrix must be 2-dimensional.")
        
        # No extensive bound checks for speed in batch mode, assume valid indices.
        # If needed, add:
        # num_nodes_dim0 = metric_matrix.shape[0]
        # num_nodes_dim1 = metric_matrix.shape[1]
        # if not (torch.all(0 <= u_indices) and torch.all(u_indices < num_nodes_dim0) and \
        #         torch.all(0 <= v_indices) and torch.all(v_indices < num_nodes_dim1)):
        #     raise IndexError("Node indices out of bounds for metric matrix.")
            
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
                edge_indices: Tuple[torch.Tensor, torch.Tensor], # Tuple of (u_indices, v_indices)
                distance_matrix_d: Union[torch.Tensor, np.ndarray],
                airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
                tailwind_values_w: torch.Tensor # Batch of tailwind values
               ) -> torch.Tensor:
        """
        Calculates the cost c(e, t_e, beta) for a batch of edges.

        Args:
            edge_indices: Tuple (u_indices, v_indices) of 1D integer tensors for start and end nodes.
            distance_matrix_d: 2D torch.Tensor or numpy.ndarray for distances D. D[i,j] = d(i,j).
            airspace_charge_matrix_ac: 2D torch.Tensor or numpy.ndarray for airspace charges AC.
            tailwind_values_w: 1D torch.Tensor for tailwind w_tail(e, t_e) for each edge.
                               Positive for tailwind, negative for headwind.

        Returns:
            A 1D torch.Tensor representing the costs of the edges.
        """
        u_indices, v_indices = edge_indices

        dist_e_batch = self.get_distance_batched(u_indices, v_indices, distance_matrix_d)

        # For edges with infinite distance, cost should be infinite.
        # Other calculations might lead to NaN or errors if inf is not handled.
        inf_mask = torch.isinf(dist_e_batch)

        ac_e_batch = self.get_airspace_charge_batched(u_indices, v_indices, airspace_charge_matrix_ac)
        
        ac_dist_product_batch = ac_e_batch * dist_e_batch
        # Ensure tailwind_values_w is on the correct device and dtype for PLM input
        tailwind_tensor_batch = tailwind_values_w.to(device=self.device, dtype=torch.float32)

        cost_component_ac_dist = self.plm_ac_dist(ac_dist_product_batch) 
        cost_component_wind = self.plm_wind(tailwind_tensor_batch)

        total_cost_batch = self.beta0 + self.beta1 * cost_component_ac_dist + self.beta2 * cost_component_wind
        
        # Apply infinite cost where distance was infinite
        total_cost_batch[inf_mask] = float('inf')
        
        return total_cost_batch

# Sanity Tests / Example Usage
# Import numpy for example matrices if needed for testing numpy input handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
from typing import Union # For type hints in forward


def run_sanity_tests():
    print("Running Sanity Tests for CostRev1...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    beta0, beta1, beta2 = 0.1, 1.0, 0.5 # Example coefficients

    cost_model = CostRev1(beta0, beta1, beta2, device=device)
    print(f"Model initialized on device: {cost_model.device}")
    print(f"Default knots for AC*Dist: {cost_model.plm_ac_dist.knot_points.tolist()}")
    print(f"Default knots for Wind: {cost_model.plm_wind.knot_points.tolist()}")


    # Example matrices (Torch Tensors)
    distance_matrix_torch = torch.tensor([
        [0.0, 150.0, float('inf'), 300.0],
        [150.0, 0.0, 200.0, 250.0],
        [float('inf'), 200.0, 0.0, 100.0],
        [300.0, 250.0, 100.0, 0.0]
    ], dtype=torch.float32, device=device)

    airspace_charge_matrix_torch = torch.tensor([
        [0.0, 10.0, 0.0, 5.0],
        [10.0, 0.0, 12.0, 8.0],
        [0.0, 12.0, 0.0, 15.0],
        [5.0, 8.0, 15.0, 0.0]
    ], dtype=torch.float32, device=device)

    # Test Case 1: Batch of edges with Torch tensor inputs
    print(f"\n--- Test Case: Batch of Edges with Torch Tensors ---")
    u_indices_batch = torch.tensor([0, 0, 1, 2], device=device, dtype=torch.long)
    v_indices_batch = torch.tensor([1, 3, 2, 3], device=device, dtype=torch.long)
    # Edges: (0,1), (0,3), (1,2), (2,3)
    # Dists: 150, 300, 200, 100
    # ACs:   10,  5,   12,  15
    # AC*Dist: 1500, 1500, 2400, 1500

    tailwind_batch_pos = torch.tensor([30.0, 10.0, 20.0, 40.0], dtype=torch.float32, device=device)
    tailwind_batch_neg = torch.tensor([-20.0, -5.0, -15.0, -25.0], dtype=torch.float32, device=device)

    costs_batch_pos = cost_model((u_indices_batch, v_indices_batch), distance_matrix_torch, airspace_charge_matrix_torch, tailwind_batch_pos)
    print(f"Edges: {[(u.item(),v.item()) for u,v in zip(u_indices_batch, v_indices_batch)]}")
    print(f"Tailwinds (pos): {tailwind_batch_pos.tolist()}")
    print(f"Costs (pos): {costs_batch_pos.tolist()}")
    
    costs_batch_neg = cost_model((u_indices_batch, v_indices_batch), distance_matrix_torch, airspace_charge_matrix_torch, tailwind_batch_neg)
    print(f"Tailwinds (neg): {tailwind_batch_neg.tolist()}")
    print(f"Costs (neg): {costs_batch_neg.tolist()}")

    if torch.all(costs_batch_pos < costs_batch_neg):
        print("OK: Positive tailwinds result in lower costs for the batch.")
    else:
        print("WARNING: Positive tailwind did NOT consistently result in lower cost. Check PLM_wind and beta2.")
        for i in range(len(costs_batch_pos)):
            if costs_batch_pos[i] >= costs_batch_neg[i]:
                print(f"  Problem at edge index {i}: cost_pos={costs_batch_pos[i]}, cost_neg={costs_batch_neg[i]}")


    # Test Case 2: Batch including a non-existent edge
    print(f"\n--- Test Case: Batch with Non-existent Edge (0,2) ---")
    u_indices_nonexist = torch.tensor([0, 0], device=device, dtype=torch.long) # Edge (0,2) is inf distance
    v_indices_nonexist = torch.tensor([1, 2], device=device, dtype=torch.long)
    tailwind_nonexist = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
    
    costs_nonexist = cost_model((u_indices_nonexist, v_indices_nonexist), distance_matrix_torch, airspace_charge_matrix_torch, tailwind_nonexist)
    print(f"Edges: {[(u.item(),v.item()) for u,v in zip(u_indices_nonexist, v_indices_nonexist)]}")
    print(f"Costs: {costs_nonexist.tolist()}")
    if torch.isinf(costs_nonexist[1]):
        print("OK: Cost for non-existent edge (0,2) in batch is infinite.")
    else:
        print("WARNING: Cost for non-existent edge (0,2) in batch is NOT infinite.")
    if not torch.isinf(costs_nonexist[0]):
        print("OK: Cost for existent edge (0,1) in batch is finite.")
    else:
        print("WARNING: Cost for existent edge (0,1) in batch is infinite.")


    if NUMPY_AVAILABLE:
        print(f"\n--- Test Case: Batch with NumPy array inputs for matrices ---")
        distance_matrix_np = distance_matrix_torch.cpu().numpy()
        airspace_charge_matrix_np = airspace_charge_matrix_torch.cpu().numpy()
        
        # Using same u_indices_batch, v_indices_batch, tailwind_batch_pos from Test Case 1
        costs_batch_np_inputs = cost_model((u_indices_batch, v_indices_batch), distance_matrix_np, airspace_charge_matrix_np, tailwind_batch_pos)
        print(f"Edges: {[(u.item(),v.item()) for u,v in zip(u_indices_batch, v_indices_batch)]}")
        print(f"Tailwinds: {tailwind_batch_pos.tolist()}")
        print(f"Costs (NumPy matrix inputs): {costs_batch_np_inputs.tolist()}")
        
        if isinstance(costs_batch_np_inputs, torch.Tensor) and \
           not torch.any(torch.isnan(costs_batch_np_inputs)) and \
           not torch.all(torch.isinf(costs_batch_np_inputs[~torch.isinf(distance_matrix_torch[u_indices_batch, v_indices_batch])])) : # Check non-inf distances didn't become inf costs
             print("OK: NumPy matrix inputs processed for batch.")
        else:
             print("WARNING: Problem with NumPy matrix input processing for batch.")
    else:
        print("\nSkipping NumPy input test as NumPy is not available.")

    print("\nReminder: Altitude-related edge cases for tailwind generation are external to this module.")
    print("\nSanity Tests Complete.")

if __name__ == '__main__':
    run_sanity_tests()
