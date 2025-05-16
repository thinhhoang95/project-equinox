import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Literal

class PiecewiseLinearMonoModel(nn.Module):
    r"""
    A PyTorch module implementing a continuous piecewise linear function
    with monotonically non-decreasing or non-increasing slopes.

    The function is defined by a set of knot points, an initial intercept,
    a slope for the first segment, and subsequent slope increments that
    ensure overall slope monotonicity.

    The mathematical form is:
    `f(x) = initial_intercept + first_slope * x + \sum_{i=0}^{M-1} d_i * ReLU(x - knot_points[i])`
    where:
    - `M` is the number of knot points.
    - `knot_points` (`k_i`) are fixed locations dividing the input domain.
    - `initial_intercept` is a learnable scalar.
    - `first_slope` (`s_0`) is a learnable scalar, the slope of the first segment.
    - `d_i` is the effective slope increment at `knot_points[i]`.
        - If `monotonic_type` is "non_decreasing" (for convexity):
          `d_i = softplus(unconstrained_slope_increments[i])`
        - If `monotonic_type` is "non_increasing" (for concavity):
          `d_i = -softplus(unconstrained_slope_increments[i])`
    - `unconstrained_slope_increments` (`\alpha_i`) are learnable parameters.
    - `softplus(z) = log(1 + exp(z))`.
    - `ReLU(z) = max(0, z)`.

    This construction ensures that the sequence of slopes for the segments
    (`s_0`, `s_0 + d_0`, `s_0 + d_0 + d_1`, ...) is monotonic.
    Non-decreasing slopes result in a convex function, while non-increasing
    slopes result in a concave function.

    Parameters:
        knot_points (Union[List[float], torch.Tensor]): A list of numbers
            or a 1D PyTorch tensor for knot point locations. Sorted internally.
            Considered fixed (not learnable). An empty list/tensor means
            a single linear function (defined by `initial_intercept` and `first_slope`).
        monotonic_type (Literal["non_decreasing", "non_increasing"], optional):
            Specifies the type of slope monotonicity.
            "non_decreasing" ensures slopes are `s_i <= s_{i+1}` (convex function).
            "non_increasing" ensures slopes are `s_i >= s_{i+1}` (concave function).
            Defaults to "non_decreasing".

    Learnable Parameters:
        initial_intercept (torch.nn.Parameter): Scalar intercept.
        first_slope (torch.nn.Parameter): Scalar slope of the first segment.
        unconstrained_slope_increments (torch.nn.Parameter): 1D tensor of size `M`
            (number of knots). These are transformed by `softplus` (or `-softplus`)
            to get the actual slope increments `d_i`.

    Input:
        x (Union[torch.Tensor, List[float], float]): Input tensor of any shape or
            convertible type. The function is applied element-wise.

    Output:
        torch.Tensor: A tensor with the same shape as input `x`, containing
            the function values.
    """
    def __init__(self,
                 knot_points: Union[List[float], torch.Tensor],
                 monotonic_type: Literal["non_decreasing", "non_increasing"] = "non_decreasing"):
        super().__init__()

        if isinstance(knot_points, list):
            if knot_points and not all(isinstance(k, (int, float)) for k in knot_points):
                raise TypeError("If knot_points is a non-empty list, all elements must be numbers (int or float).")
            knot_points_tensor = torch.tensor(knot_points, dtype=torch.float32)
        elif isinstance(knot_points, torch.Tensor):
            knot_points_tensor = knot_points.float() # Ensure float
        else:
            raise TypeError("knot_points must be a list of numbers or a PyTorch Tensor.")

        # Validate tensor shape: must be 1D
        if knot_points_tensor.ndim != 1:
            # Allow 0-dim tensor only if it's truly empty and can be reshaped to 1D empty.
            if knot_points_tensor.ndim == 0 and knot_points_tensor.numel() == 0:
                knot_points_tensor = knot_points_tensor.reshape(0) # Normalize to 1D, shape (0,)
            else:
                raise ValueError(
                    f"knot_points tensor must be 1D (e.g., torch.tensor([1.0, 2.0])) "
                    f"or an empty 1D tensor (torch.tensor([])). "
                    f"Got ndim={knot_points_tensor.ndim} with numel={knot_points_tensor.numel()}."
                )
        
        # Sort knot points if any exist
        if knot_points_tensor.numel() > 0:
            knot_points_tensor, _ = torch.sort(knot_points_tensor)
        
        self.register_buffer('knot_points', knot_points_tensor)

        if monotonic_type not in ["non_decreasing", "non_increasing"]:
            raise ValueError("monotonic_type must be 'non_decreasing' or 'non_increasing'.")
        self.monotonic_type = monotonic_type

        num_knots = self.knot_points.numel()
        
        self.initial_intercept = nn.Parameter(torch.randn(1))
        self.first_slope = nn.Parameter(torch.randn(1)) # This is s_0
        
        # These are the 'alpha_i' parameters before softplus
        # If num_knots is 0, this creates a Parameter of shape (0,), which is fine.
        self.unconstrained_slope_increments = nn.Parameter(torch.randn(num_knots))

    def forward(self, x: Union[torch.Tensor, List[float], float]) -> torch.Tensor:
        """
        Applies the monotonic piecewise linear function element-wise to the input.
        """
        if not isinstance(x, torch.Tensor):
            try:
                # Infer dtype and device from a parameter to ensure consistency
                target_dtype = self.initial_intercept.dtype
                target_device = self.initial_intercept.device
                x_tensor = torch.tensor(x, dtype=target_dtype, device=target_device)
            except Exception as e:
                raise TypeError(
                    f"Input x must be a torch.Tensor or convertible to one (e.g., list of numbers, float). "
                    f"Attempted conversion failed. Original error: {e}"
                )
        else:
            x_tensor = x
            
        # Base linear function: y = initial_intercept + first_slope * x
        y = self.initial_intercept + self.first_slope * x_tensor
        
        # Add segments: sum_{i=0}^{M-1} d_i * ReLU(x - k_i)
        num_knots = self.knot_points.numel()
        if num_knots > 0:
            for i in range(num_knots):
                # Calculate effective_slope_increment (d_i)
                # d_i = softplus(alpha_i) or -softplus(alpha_i)
                slope_increment = F.softplus(self.unconstrained_slope_increments[i])
                if self.monotonic_type == "non_increasing":
                    slope_increment = -slope_increment
                
                # Add term: d_i * ReLU(x - k_i)
                # self.knot_points is on the correct device due to register_buffer and model.to(device)
                # Broadcasting: x_tensor (any shape) - self.knot_points[i] (scalar)
                y = y + slope_increment * torch.relu(x_tensor - self.knot_points[i])
            
        return y

    def get_slopes(self) -> torch.Tensor:
        """
        Computes and returns the actual slopes of all segments.

        The slopes are s_0, s_1, ..., s_M.
        s_0 = first_slope
        s_{j+1} = s_j + d_j
        where d_j is the effective slope increment at knot_points[j].

        Returns:
            torch.Tensor: A 1D tensor containing the M+1 slopes.
        """
        num_knots = self.knot_points.numel()
        slopes = torch.empty(num_knots + 1, dtype=self.first_slope.dtype, device=self.first_slope.device)
        
        current_slope = self.first_slope.clone()
        slopes[0] = current_slope

        if num_knots > 0:
            for i in range(num_knots):
                increment = F.softplus(self.unconstrained_slope_increments[i])
                if self.monotonic_type == "non_increasing":
                    increment = -increment
                current_slope = current_slope + increment
                slopes[i+1] = current_slope
        
        return slopes

# Example usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    # Test case 1: Non-decreasing (convex)
    knots1 = [0.0, 1.0, 2.0]
    model_convex = PiecewiseLinearMonoModel(knots1, monotonic_type="non_decreasing")
    
    # Initialize parameters for predictability (optional)
    # model_convex.initial_intercept.data.fill_(0.1)
    # model_convex.first_slope.data.fill_(0.5)
    # model_convex.unconstrained_slope_increments.data.fill_(0.0) # softplus(0) = ln(2) ~ 0.693

    print("Convex Model Parameters:")
    for name, param in model_convex.named_parameters():
        print(f"{name}: {param.data}")
    
    x_test = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    y_convex = model_convex(x_test)
    print(f"\nInput x: {x_test}")
    print(f"Output y (convex): {y_convex}")
    slopes_convex = model_convex.get_slopes()
    print(f"Slopes (convex): {slopes_convex}")
    # Check if slopes are non-decreasing
    if slopes_convex.numel() > 1:
        print(f"Slopes non-decreasing: {torch.all(slopes_convex[1:] >= slopes_convex[:-1] - 1e-6)}")


    # Test case 2: Non-increasing (concave)
    knots2 = [-1.0, 1.0]
    model_concave = PiecewiseLinearMonoModel(knots2, monotonic_type="non_increasing")

    # model_concave.initial_intercept.data.fill_(-0.2)
    # model_concave.first_slope.data.fill_(1.0)
    # model_concave.unconstrained_slope_increments.data.fill_(0.5) # -softplus(0.5) ~ -0.974

    print("\nConcave Model Parameters:")
    for name, param in model_concave.named_parameters():
        print(f"{name}: {param.data}")

    y_concave = model_concave(x_test) # Using same x_test
    print(f"\nInput x: {x_test}")
    print(f"Output y (concave): {y_concave}")
    slopes_concave = model_concave.get_slopes()
    print(f"Slopes (concave): {slopes_concave}")
    # Check if slopes are non-increasing
    if slopes_concave.numel() > 1:
        print(f"Slopes non-increasing: {torch.all(slopes_concave[1:] <= slopes_concave[:-1] + 1e-6)}")

    # Test case 3: No knots (single linear function)
    model_linear = PiecewiseLinearMonoModel([])
    # model_linear.initial_intercept.data.fill_(1.0)
    # model_linear.first_slope.data.fill_(-2.0)

    print("\nLinear Model (no knots) Parameters:")
    for name, param in model_linear.named_parameters():
        print(f"{name}: {param.data}")
    
    y_linear = model_linear(x_test)
    print(f"\nInput x: {x_test}")
    print(f"Output y (linear): {y_linear}")
    slopes_linear = model_linear.get_slopes()
    print(f"Slopes (linear): {slopes_linear}") # Should be just [first_slope]
    
    # Example of training
    # optimizer = torch.optim.Adam(model_convex.parameters(), lr=0.01)
    # criterion = nn.MSELoss()
    # # Dummy training data
    # x_train = torch.randn(100, 1) * 2
    # y_target_convex = 0.5 * x_train**2 + 0.1 * x_train # A convex function
    #
    # print("\nStarting dummy training for convex model...")
    # for epoch in range(10): # Small number of epochs for example
    #     optimizer.zero_grad()
    #     y_pred = model_convex(x_train.squeeze()) # model expects 1D or scalar elements if not tensor
    #     loss = criterion(y_pred.unsqueeze(-1), y_target_convex)
    #     loss.backward()
    #     optimizer.step()
    #     if (epoch + 1) % 2 == 0:
    #         current_slopes = model_convex.get_slopes()
    #         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Slopes: {current_slopes.data}")
    #         if current_slopes.numel() > 1:
    #             is_non_decreasing = torch.all(current_slopes[1:] >= current_slopes[:-1] - 1e-6)
    #             print(f"  Slopes still non-decreasing: {is_non_decreasing}")
    #             if not is_non_decreasing:
    #                 print("  WARNING: Slopes became non-monotonic!")
    # print("Dummy training finished.")
