import torch
import torch.nn as nn
from equinox.cost.plf_mono import PiecewiseLinearMonoModel
import numpy as np
from typing import Union

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

    def _get_edge_metric(self, u_idx: int, v_idx: int, metric_matrix: torch.Tensor) -> torch.Tensor:
        """
        Retrieves a metric for an edge (u, v) from a 2D metric matrix.
        """
        if not isinstance(metric_matrix, torch.Tensor):
            # Convert to tensor if numpy array, assuming it's on CPU
            # For robust handling, it's better if caller passes tensors.
            # This is a fallback.
            try:
                metric_matrix = torch.from_numpy(metric_matrix).to(dtype=torch.float32, device=self.device)
            except TypeError: # handles cases where it might already be a tensor but wrong type, or other error
                 raise TypeError(f"Metric matrix (e.g., D, AC) must be a torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}")
            except AttributeError: # Not a numpy array
                 raise TypeError(f"Metric matrix (e.g., D, AC) must be a torch.Tensor or numpy.ndarray. Got {type(metric_matrix)}")


        if metric_matrix.device != self.device:
            metric_matrix = metric_matrix.to(self.device)
            
        if metric_matrix.ndim != 2:
            raise ValueError("Metric matrix must be 2-dimensional.")
        
        num_nodes_dim0 = metric_matrix.shape[0]
        num_nodes_dim1 = metric_matrix.shape[1]
        if not (0 <= u_idx < num_nodes_dim0 and 0 <= v_idx < num_nodes_dim1):
            raise IndexError(
                f"Node indices ({u_idx}, {v_idx}) out of bounds for matrix with shape ({num_nodes_dim0}, {num_nodes_dim1})."
            )
            
        return metric_matrix[u_idx, v_idx]

    def get_distance(self, u_idx: int, v_idx: int, distance_matrix_d: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the distance d(e) for an edge e = (u, v) from the distance matrix D.
        The matrix D should have +inf for non-existent links.
        Input matrix is expected to be a torch.Tensor or numpy.ndarray.
        """
        return self._get_edge_metric(u_idx, v_idx, distance_matrix_d)

    def get_airspace_charge(self, u_idx: int, v_idx: int, airspace_charge_matrix_ac: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the airspace charge AC(e) for an edge e = (u, v) from the AC matrix.
        Input matrix is expected to be a torch.Tensor or numpy.ndarray.
        """
        return self._get_edge_metric(u_idx, v_idx, airspace_charge_matrix_ac)

    def forward(self,
                edge_indices: tuple[int, int],
                distance_matrix_d: Union[torch.Tensor, np.ndarray],
                airspace_charge_matrix_ac: Union[torch.Tensor, np.ndarray],
                tailwind_value_w: Union[torch.Tensor, float]
               ) -> torch.Tensor:
        """
        Calculates the cost c(e, t_e, beta) for the given edge.

        Args:
            edge_indices: Tuple (u_idx, v_idx) of integer indices for the start and end nodes.
            distance_matrix_d: 2D torch.Tensor or numpy.ndarray for distances D. D[i,j] = d(i,j).
            airspace_charge_matrix_ac: 2D torch.Tensor or numpy.ndarray for airspace charges AC.
            tailwind_value_w: Scalar torch.Tensor or float for tailwind w_tail(e, t_e).
                              Positive for tailwind, negative for headwind.

        Returns:
            A scalar torch.Tensor representing the cost of the edge.
        """
        u_idx, v_idx = edge_indices

        dist_e = self.get_distance(u_idx, v_idx, distance_matrix_d)

        if torch.isinf(dist_e):
            return torch.tensor(float('inf'), dtype=torch.float32, device=self.device)

        ac_e = self.get_airspace_charge(u_idx, v_idx, airspace_charge_matrix_ac)
        
        ac_dist_product = ac_e * dist_e

        # Ensure tailwind_value_w is a tensor on the correct device for PLM input
        if not isinstance(tailwind_value_w, torch.Tensor):
            tailwind_tensor = torch.tensor(tailwind_value_w, dtype=torch.float32, device=self.device)
        else:
            tailwind_tensor = tailwind_value_w.to(device=self.device, dtype=torch.float32)


        # PLMs expect torch.Tensor inputs. Their internal handling should manage device and dtype
        # based on their parameters, but it's good practice to ensure inputs are tensors.
        # The PLM's forward method converts list/float inputs to tensors using its parameters' device/dtype.
        cost_component_ac_dist = self.plm_ac_dist(ac_dist_product) 
        cost_component_wind = self.plm_wind(tailwind_tensor)

        total_cost = self.beta0 + self.beta1 * cost_component_ac_dist + self.beta2 * cost_component_wind
        
        return total_cost

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
        [0.0, 150.0, float('inf')],
        [150.0, 0.0, 200.0],
        [float('inf'), 200.0, 0.0]
    ], dtype=torch.float32, device=device)

    airspace_charge_matrix_torch = torch.tensor([
        [0.0, 10.0, 0.0],
        [10.0, 0.0, 12.0],
        [0.0, 12.0, 0.0]
    ], dtype=torch.float32, device=device)

    # Test Case 1: Edge (0, 1) with Torch tensor inputs
    print(f"\n--- Test Case: Edge (0,1) with Torch Tensors ---")
    edge1 = (0, 1)
    tailwind1_pos = torch.tensor(30.0, dtype=torch.float32, device=device)
    tailwind1_neg = torch.tensor(-20.0, dtype=torch.float32, device=device)

    cost1_pos = cost_model(edge1, distance_matrix_torch, airspace_charge_matrix_torch, tailwind1_pos)
    print(f"Edge {edge1}, Tailwind {tailwind1_pos.item():.1f}: Cost = {cost1_pos.item():.4f}")
    
    cost1_neg = cost_model(edge1, distance_matrix_torch, airspace_charge_matrix_torch, tailwind1_neg)
    print(f"Edge {edge1}, Tailwind {tailwind1_neg.item():.1f}: Cost = {cost1_neg.item():.4f}")

    if cost1_pos < cost1_neg:
        print("OK: Positive tailwind results in lower cost.")
    else:
        print("WARNING: Positive tailwind did NOT result in lower cost. Check PLM_wind (non-increasing) and beta2 (positive).")

    # Test Case 2: Non-existent edge (0, 2)
    print(f"\n--- Test Case: Non-existent Edge (0,2) ---")
    edge3 = (0, 2)
    cost3 = cost_model(edge3, distance_matrix_torch, airspace_charge_matrix_torch, tailwind1_pos)
    print(f"Edge {edge3}: Cost = {cost3.item()}")
    if torch.isinf(cost3):
        print("OK: Cost for non-existent edge is infinite.")
    else:
        print("WARNING: Cost for non-existent edge is NOT infinite.")

    # Test Case 3: Edge with zero AC*Dist product (0,0)
    print(f"\n--- Test Case: Edge (0,0) (Zero AC*Dist) ---")
    edge4 = (0,0) # d(0,0)=0, ac(0,0)=0
    tailwind4 = torch.tensor(0.0, device=device)
    cost4 = cost_model(edge4, distance_matrix_torch, airspace_charge_matrix_torch, tailwind4)
    # Expected: beta0 + beta1*PLM_ac(0) + beta2*PLM_wind(0)
    # PLM(0) = initial_intercept + first_slope * 0 + sum(d_i * ReLU(-k_i))
    # If all knots are > 0, then ReLU(-k_i) = 0. So PLM(0) = initial_intercept.
    # If some knots are < 0, it's more complex.
    print(f"Edge {edge4}, AC*Dist=0, Wind=0: Cost = {cost4.item():.4f}")
    # To verify this, one would need to inspect the initialized parameters of the PLMs or set them.
    # For example:
    # plm_ac_at_0 = cost_model.plm_ac_dist(torch.tensor(0.0, device=device))
    # plm_wind_at_0 = cost_model.plm_wind(torch.tensor(0.0, device=device))
    # expected_cost4 = cost_model.beta0 + cost_model.beta1 * plm_ac_at_0 + cost_model.beta2 * plm_wind_at_0
    # print(f"Calculated PLM_ac(0): {plm_ac_at_0.item()}, PLM_wind(0): {plm_wind_at_0.item()}")
    # print(f"Expected cost for edge4 (approx): {expected_cost4.item():.4f}")


    if NUMPY_AVAILABLE:
        print(f"\n--- Test Case: Edge (1,2) with NumPy array inputs ---")
        distance_matrix_np = distance_matrix_torch.cpu().numpy()
        airspace_charge_matrix_np = airspace_charge_matrix_torch.cpu().numpy()
        edge2 = (1,2) # d=200, ac=12 -> ac_dist = 2400
        tailwind2_np = -10.0 # float input for tailwind

        cost2_np_inputs = cost_model(edge2, distance_matrix_np, airspace_charge_matrix_np, tailwind2_np)
        print(f"Edge {edge2} (NumPy inputs), Tailwind {tailwind2_np:.1f}: Cost = {cost2_np_inputs.item():.4f}")
        # Check if it ran without errors and produced a value.
        if isinstance(cost2_np_inputs, torch.Tensor) and not torch.isnan(cost2_np_inputs) and not torch.isinf(cost2_np_inputs):
             print("OK: NumPy inputs processed.")
        else:
             print("WARNING: Problem with NumPy input processing.")
    else:
        print("\nSkipping NumPy input test as NumPy is not available.")

    print("\nReminder: Altitude-related edge cases for tailwind generation are external to this module.")
    print("\nSanity Tests Complete.")

if __name__ == '__main__':
    run_sanity_tests()
