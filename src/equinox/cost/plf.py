import torch
import torch.nn as nn
from typing import List, Union

class PiecewiseLinearFunction(nn.Module):
    """
    A PyTorch module implementing a continuous piecewise linear function.

    The function is defined by a set of knot points, an initial intercept,
    and slopes for each segment. The knot points divide the input domain
    into segments, and each segment has a distinct linear function.
    The overall function is continuous at the knot points.

    The function is of the form:
    f(x) = initial_intercept + slopes[0] * x +
           sum_{i=0}^{M-1} (slopes[i+1] - slopes[i]) * ReLU(x - knot_points[i])
    where M is the number of knot points.

    Parameters:
        knot_points (Union[List[float], torch.Tensor]): A list of floating-point numbers
            or a 1D PyTorch tensor representing the knot point locations.
            These points define the boundaries between linear segments.
            They will be sorted internally and are considered fixed (not learnable).
            An empty list or tensor means no knot points, resulting in a single linear function.

    Learnable Parameters:
        initial_intercept (torch.nn.Parameter): A scalar tensor representing the
            y-intercept of the first linear segment (for x < first knot point,
            or for all x if no knot points are given).
        slopes (torch.nn.Parameter): A 1D tensor of slopes. `slopes[0]` is the
            slope of the first segment. `slopes[i+1]` is the slope of the
            segment starting after `knot_points[i]`. There are `M+1` slopes
            for `M` knot points.

    Input:
        x (torch.Tensor): An input tensor of any shape. The piecewise
            linear function is applied element-wise. If not a tensor,
            an attempt will be made to convert it.

    Output:
        torch.Tensor: A tensor with the same shape as the input `x`.
            Each element `y_i` is the result of applying the piecewise
            linear function to the corresponding element `x_i`.
            The piecewise linear function itself maps a scalar input to a
            scalar output.
    """
    def __init__(self, knot_points: Union[List[float], torch.Tensor]):
        super().__init__()

        if isinstance(knot_points, list):
            if knot_points and not all(isinstance(k, (int, float)) for k in knot_points):
                raise TypeError("If knot_points is a non-empty list, all elements must be numbers (int or float).")
            knot_points_tensor = torch.tensor(knot_points, dtype=torch.float32)
        elif isinstance(knot_points, torch.Tensor):
            knot_points_tensor = knot_points.float()
        else:
            raise TypeError("knot_points must be a list of numbers or a PyTorch Tensor.")

        # Validate tensor shape: must be 1D
        if knot_points_tensor.ndim != 1:
            # Allow 0-dim tensor only if it's truly empty and can be reshaped to 1D empty.
            # Note: torch.tensor([]) is 1D with shape (0,).
            # A scalar like torch.tensor(5.0) is 0D with 1 element.
            if knot_points_tensor.ndim == 0 and knot_points_tensor.numel() == 0:
                # This case (0D, 0 elements) is rare for well-formed empty tensors.
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

        num_knots = self.knot_points.numel()
        
        # M+1 slopes: s_0, s_1, ..., s_M
        self.slopes = nn.Parameter(torch.randn(num_knots + 1)) 
        self.initial_intercept = nn.Parameter(torch.randn(1)) # b_0

    def forward(self, x: Union[torch.Tensor, List[float], float]) -> torch.Tensor:
        """
        Applies the piecewise linear function element-wise to the input.
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
            
        # Ensure parameters and buffers are on the same device as input x_tensor.
        # This is implicit if module.to(device) and x_tensor.to(device) were called correctly by user.
        # Accessing them directly uses their stored device.
        # If devices mismatch, PyTorch will raise an error, which is standard.
        
        # y = b0 + s0*x
        y = self.initial_intercept + self.slopes[0] * x_tensor
        
        # Add subsequent segments: sum_{i=0}^{M-1} (slope[i+1] - slope[i]) * ReLU(x - knot_point[i])
        # self.knot_points is already on the correct device due to register_buffer and model.to(device)
        # self.slopes is already on the correct device due to nn.Parameter and model.to(device)
        for i in range(self.knot_points.numel()):
            delta_slope = self.slopes[i+1] - self.slopes[i]
            # Broadcasting: x_tensor (any shape) - self.knot_points[i] (scalar)
            y = y + delta_slope * torch.relu(x_tensor - self.knot_points[i])
            
        return y

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    # Test case 1: No knot points (should be a simple linear function)
    plf_no_knots = PiecewiseLinearFunction(knot_points=[])
    print("PLF with no knots (should be linear):")
    print("Parameters:", list(plf_no_knots.named_parameters()))
    test_input_1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    output_1 = plf_no_knots(test_input_1)
    print(f"Input: {test_input_1.numpy()}, Output: {output_1.detach().numpy()}")
    # Manually check: y = intercept + slope[0]*x
    intercept_1 = plf_no_knots.initial_intercept.item()
    slope_1_0 = plf_no_knots.slopes[0].item()
    expected_1 = intercept_1 + slope_1_0 * test_input_1
    print(f"Expected: {expected_1.detach().numpy()}")
    assert torch.allclose(output_1, expected_1), "Test Case 1 Failed"
    print("-" * 30)

    # Test case 2: With knot points
    knot_points_2 = [-1.0, 1.0]
    plf_with_knots = PiecewiseLinearFunction(knot_points=knot_points_2)
    # Manually set parameters for predictability in test
    plf_with_knots.initial_intercept.data.fill_(1.0) # b0 = 1.0
    plf_with_knots.slopes.data = torch.tensor([0.5, -1.0, 2.0]) # s0=0.5, s1=-1.0, s2=2.0
                                                               # knots: k0=-1, k1=1
    print(f"PLF with knots {knot_points_2}:")
    print("Initial Intercept (b0):", plf_with_knots.initial_intercept.item())
    print("Slopes (s0, s1, s2):", plf_with_knots.slopes.data.numpy())
    
    test_input_2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    # Expected calculation:
    # y = b0 + s0*x + (s1-s0)*relu(x-k0) + (s2-s1)*relu(x-k1)
    # b0=1.0, s0=0.5, s1=-1.0, s2=2.0, k0=-1.0, k1=1.0
    # (s1-s0) = -1.0 - 0.5 = -1.5
    # (s2-s1) = 2.0 - (-1.0) = 3.0
    # For x = -2.0: y = 1.0 + 0.5*(-2) + (-1.5)*relu(-2 - (-1)) + (3.0)*relu(-2 - 1)
    #                y = 1.0 - 1.0    + (-1.5)*relu(-1)      + (3.0)*relu(-3)
    #                y = 0.0          + 0                    + 0                = 0.0
    # For x = -1.0: y = 1.0 + 0.5*(-1) + (-1.5)*relu(-1 - (-1)) + (3.0)*relu(-1 - 1)
    #                y = 1.0 - 0.5    + (-1.5)*relu(0)       + (3.0)*relu(-2)
    #                y = 0.5          + 0                    + 0                = 0.5
    # For x = 0.0:  y = 1.0 + 0.5*(0)  + (-1.5)*relu(0 - (-1))  + (3.0)*relu(0 - 1)
    #                y = 1.0 + 0      + (-1.5)*relu(1)       + (3.0)*relu(-1)
    #                y = 1.0          + (-1.5)*1             + 0                = -0.5
    # For x = 1.0:  y = 1.0 + 0.5*(1)  + (-1.5)*relu(1 - (-1))  + (3.0)*relu(1 - 1)
    #                y = 1.0 + 0.5    + (-1.5)*relu(2)       + (3.0)*relu(0)
    #                y = 1.5          + (-1.5)*2             + 0                = 1.5 - 3.0 = -1.5
    # For x = 2.0:  y = 1.0 + 0.5*(2)  + (-1.5)*relu(2 - (-1))  + (3.0)*relu(2 - 1)
    #                y = 1.0 + 1.0    + (-1.5)*relu(3)       + (3.0)*relu(1)
    #                y = 2.0          + (-1.5)*3             + (3.0)*1          = 2.0 - 4.5 + 3.0 = 0.5
    
    output_2 = plf_with_knots(test_input_2)
    print(f"Input: {test_input_2.numpy()}, Output: {output_2.detach().numpy()}")
    expected_output_2 = torch.tensor([0.0, 0.5, -0.5, -1.5, 0.5, 2.5]) # Recalculate x=3.0
    # For x = 3.0:  y = 1.0 + 0.5*(3)  + (-1.5)*relu(3 - (-1))  + (3.0)*relu(3 - 1)
    #                y = 1.0 + 1.5    + (-1.5)*relu(4)       + (3.0)*relu(2)
    #                y = 2.5          + (-1.5)*4             + (3.0)*2          = 2.5 - 6.0 + 6.0 = 2.5
    print(f"Expected: {expected_output_2.numpy()}")
    assert torch.allclose(output_2, expected_output_2, atol=1e-6), "Test Case 2 Failed"
    print("-" * 30)

    # Test case 3: Input as list
    input_list = [-2.0, 0.0, 2.0]
    output_3 = plf_with_knots(input_list)
    print(f"Input list: {input_list}, Output: {output_3.detach().numpy()}")
    expected_output_3 = torch.tensor([0.0, -0.5, 0.5])
    assert torch.allclose(output_3, expected_output_3, atol=1e-6), "Test Case 3 Failed"
    print("-" * 30)

    # Test case 4: Multi-dimensional input
    input_md = torch.tensor([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
    output_md = plf_with_knots(input_md)
    print(f"Input MD: \n{input_md.numpy()}, \nOutput MD: \n{output_md.detach().numpy()}")
    expected_output_md = torch.tensor([[0.0, 0.5], [-0.5, -1.5], [0.5, 2.5]])
    assert torch.allclose(output_md, expected_output_md, atol=1e-6), "Test Case 4 Failed"
    print("-" * 30)
    
    # Test case 5: Knot points as tensor
    plf_tensor_knots = PiecewiseLinearFunction(knot_points=torch.tensor([-1.0, 1.0]))
    plf_tensor_knots.initial_intercept.data.fill_(1.0)
    plf_tensor_knots.slopes.data = torch.tensor([0.5, -1.0, 2.0])
    output_5 = plf_tensor_knots(test_input_2) # Using same input as test_input_2
    print(f"PLF with tensor knots, Output: {output_5.detach().numpy()}")
    assert torch.allclose(output_5, expected_output_2, atol=1e-6), "Test Case 5 Failed"
    print("-" * 30)

    print("All basic tests passed!")
