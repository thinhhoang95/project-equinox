import torch

def batched_interp1d_torch(
    x_new_batched: torch.Tensor,
    x_known_batched: torch.Tensor,
    y_known_batched: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs batched 1D linear interpolation.
    x_new_batched: [B], points to interpolate at for each batch item.
    x_known_batched: [B, P], sorted x-coordinates for each batch item.
    y_known_batched: [B, P], y-coordinates corresponding to x_known_batched.
    device: torch.device to use for new tensors.
    Returns y_new_batched: [B]
    """
    # Ensure x_new is clipped to the range of each row in x_known_batched
    min_x_known = x_known_batched[:, 0]
    max_x_known = x_known_batched[:, -1]
    # Ensure x_new_batched has same shape as min_x_known for clamp if it's a scalar
    if x_new_batched.ndim == 0:
        x_new_batched_expanded = x_new_batched.expand_as(min_x_known)
    else:
        x_new_batched_expanded = x_new_batched

    x_new_clipped = torch.clamp(x_new_batched_expanded, min_x_known, max_x_known)

    idx_right = torch.searchsorted(
        x_known_batched, x_new_clipped.unsqueeze(1), right=True
    )
    idx_right = torch.clamp(idx_right, 1, x_known_batched.shape[1] - 1)
    idx_left = idx_right - 1

    batch_indices = torch.arange(x_known_batched.shape[0], device=device).unsqueeze(1)

    x_left = torch.gather(x_known_batched, 1, idx_left)
    x_right = torch.gather(x_known_batched, 1, idx_right)
    y_left = torch.gather(y_known_batched, 1, idx_left)
    y_right = torch.gather(y_known_batched, 1, idx_right)

    denom = x_right - x_left
    # Avoid division by zero: if denom is very small, assume x_new_clipped is at x_left or x_right
    # If x_left == x_right, weight is 0 if x_new_clipped == x_left, or could be nan if not handled.
    # A simple safe way: if denom is zero, result is y_left.
    weight_right = torch.where(
        denom > 1e-9,
        (x_new_clipped.unsqueeze(1) - x_left) / denom,
        torch.zeros_like(denom),
    )

    y_new_interp = y_left + weight_right * (y_right - y_left)
    return y_new_interp.squeeze(1)