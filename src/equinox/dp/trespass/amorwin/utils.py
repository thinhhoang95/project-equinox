import torch
import os

def load_value_function(path: str, shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Loads a soft value function from a .pt file.
    The file is expected to contain a sparse COO tensor.
    This function converts it to a dense tensor, filling missing values with infinity.

    Args:
        path (str): Path to the .pt file.
        shape (tuple): The desired shape of the dense tensor.
        device (torch.device): The device to load the tensor onto.

    Returns:
        torch.Tensor: The dense value function tensor.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Value function file not found at {path}")

    # Load the sparse tensor
    sparse_tensor = torch.load(path, map_location=device)

    # Coalesce is important to sum duplicate indices and sort them
    sparse_tensor_coalesced = sparse_tensor.coalesce()

    # Create a dense tensor filled with infinity
    dense_tensor = torch.full(
        shape,
        fill_value=float('inf'),
        dtype=sparse_tensor_coalesced.dtype,
        device=device
    )

    # Get indices and values from the sparse tensor
    indices = sparse_tensor_coalesced.indices()
    values = sparse_tensor_coalesced.values()

    # Fill the dense tensor with values from the sparse tensor
    if values.numel() > 0:
        dense_tensor[tuple(indices)] = values

    return dense_tensor 