import torch
import warnings

# Dictionary to map string representations of torch dtypes back to torch.dtype objects
_STR_TO_DTYPE = {
    "torch.float16": torch.float16,
    "torch.half": torch.half,
    "torch.float32": torch.float32,
    "torch.float": torch.float,
    "torch.float64": torch.float64,
    "torch.double": torch.double,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int16": torch.int16,
    "torch.short": torch.short,
    "torch.int32": torch.int32,
    "torch.int": torch.int,
    "torch.int64": torch.int64,
    "torch.long": torch.long,
    "torch.bool": torch.bool,
    "torch.complex32": torch.complex32,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
    "torch.cfloat": torch.cfloat, # alias for complex64
    "torch.cdouble": torch.cdouble, # alias for complex128
    "torch.bfloat16": torch.bfloat16,
}

def _parse_dtype_str(dtype_str: str) -> torch.dtype:
    """Converts a string representation of a torch.dtype back to a torch.dtype object."""
    dtype = _STR_TO_DTYPE.get(dtype_str)
    if dtype is None:
        try:
            # Fallback for dtypes not explicitly in the map, e.g. 'float32' instead of 'torch.float32'
            # or custom/future dtypes if their string representation is directly getattr-able from torch
            attr_name = dtype_str.split('.')[-1]
            dtype = getattr(torch, attr_name)
            if not isinstance(dtype, torch.dtype):
                raise AttributeError(f"Attribute {attr_name} is not a torch.dtype.")
        except AttributeError as e:
            raise ValueError(f"Unsupported or unknown dtype string: {dtype_str}. Original error: {e}")
    return dtype

def save_sparse_coo_tensor_with_convention(
    tensor: torch.Tensor,
    file_path: str,
    interpretation_note: str = "Implicit zeros should be treated as float('inf'). Only explicitly stored values are actual costs."
) -> None:
    """
    Saves a sparse COO tensor to a file, along with metadata including an interpretation convention.

    The tensor data (indices and values) is moved to the CPU before saving to ensure portability.

    Args:
        tensor (torch.Tensor): The sparse COO tensor to save.
        file_path (str): The path to the file where the tensor will be saved.
        interpretation_note (str, optional): A note regarding how to interpret
            values not explicitly stored in the sparse tensor. Defaults to a standard
            message about treating implicit zeros as infinity.
    """
    if not tensor.is_sparse:
        warnings.warn("The provided tensor is not sparse. Saving as a dense tensor's components might be inefficient or unintended.")

    if tensor.layout != torch.sparse_coo:
        raise ValueError(f"Tensor layout must be torch.sparse_coo, but got {tensor.layout}. Consider converting with .to_sparse_coo()")

    data_to_save = {
        "format_version": "1.0.0",
        "tensor_representation": "sparse_coo",
        "indices": tensor.indices().cpu(),
        "values": tensor.values().cpu(),
        "size": list(tensor.size()),
        "dtype_str": str(tensor.dtype), # Storing dtype of the tensor, not necessarily values.dtype
        "value_interpretation_note": interpretation_note
    }
    torch.save(data_to_save, file_path)
    if hasattr(torch, 'mps') and torch.mps.is_available(): # Temp fix for potential MPS issue
        torch.mps.synchronize()


def load_sparse_coo_tensor_with_convention(
    file_path: str,
    target_device: torch.device | str | None = None
) -> tuple[torch.Tensor, str]:
    """
    Loads a sparse COO tensor and its interpretation convention from a file.

    Args:
        file_path (str): The path to the file from which to load the tensor.
        target_device (torch.device | str | None, optional): The device to move the loaded tensor to.
            If None, the tensor remains on the CPU (as it was saved).

    Returns:
        tuple[torch.Tensor, str]: A tuple containing:
            - The loaded sparse COO tensor.
            - The interpretation note string that was saved with the tensor.
    """
    if target_device is None:
        map_location = 'cpu'
    elif isinstance(target_device, str):
        map_location = target_device
    else: # torch.device object
        map_location = target_device

    # Load to CPU first to handle potential device mismatches during load, then move.
    loaded_data = torch.load(file_path, map_location='cpu')

    if not isinstance(loaded_data, dict):
        raise TypeError(f"Expected saved data to be a dictionary, but got {type(loaded_data)}.")

    # Basic validation of the loaded data structure
    required_keys = ["format_version", "tensor_representation", "indices", "values", "size", "dtype_str", "value_interpretation_note"]
    for key in required_keys:
        if key not in loaded_data:
            raise ValueError(f"Saved data is missing required key: {key}")

    if loaded_data["tensor_representation"] != "sparse_coo":
        raise ValueError(f"Expected tensor_representation to be 'sparse_coo', but got '{loaded_data['tensor_representation']}'.")

    indices = loaded_data["indices"]
    values = loaded_data["values"]
    size = torch.Size(loaded_data["size"])
    parsed_dtype = _parse_dtype_str(loaded_data["dtype_str"])
    interpretation_note = loaded_data["value_interpretation_note"]

    # Ensure values tensor has the parsed dtype if it's different
    # This can happen if indices are empty, values might get a default dtype.
    if values.dtype != parsed_dtype:
        values = values.to(parsed_dtype)
        
    reconstructed_tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        dtype=parsed_dtype # Explicitly set dtype for the sparse tensor itself
    )

    if target_device is not None:
        reconstructed_tensor = reconstructed_tensor.to(target_device)
    
    # Coalesce after potential device move and ensuring dtype consistency
    reconstructed_tensor = reconstructed_tensor.coalesce()

    return reconstructed_tensor, interpretation_note

if __name__ == '__main__':
    # Example Usage
    print("Running example usage of sparse_io_utils...")

    # Create a dummy sparse tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    indices = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2], [0,1,1,0]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    size = (3, 3, 2) # 3D sparse tensor for example
    
    # Ensure indices are within bounds for the given size
    valid_indices = []
    valid_values = []
    for i in range(indices.shape[1]):
        idx_tuple = indices[:, i]
        if all(idx_tuple[d] < size[d] for d in range(len(size))):
            valid_indices.append(idx_tuple.tolist())
            valid_values.append(values[i].item())
            
    if not valid_indices:
        print("Warning: No valid indices for the sparse tensor based on the size. Creating an empty sparse tensor.")
        # Create an empty sparse tensor if no valid indices
        indices_for_sparse = torch.empty((len(size), 0), dtype=torch.long, device=device)
        values_for_sparse = torch.empty((0,), dtype=torch.float64, device=device)
    else:
        indices_for_sparse = torch.tensor(valid_indices, dtype=torch.long, device=device).t()
        values_for_sparse = torch.tensor(valid_values, dtype=torch.float64, device=device)

    sparse_tensor_original = torch.sparse_coo_tensor(indices_for_sparse, values_for_sparse, size, dtype=torch.float64, device=device)
    sparse_tensor_original = sparse_tensor_original.coalesce() # Good practice

    print(f"Original sparse tensor (on {sparse_tensor_original.device}):\n", sparse_tensor_original)
    print(f"Original values dtype: {sparse_tensor_original.values().dtype}, Tensor dtype: {sparse_tensor_original.dtype}")


    # Save the tensor
    file_path = "temp_sparse_tensor.pt"
    custom_note = "This is a test sparse tensor. Zeros mean +infinity."
    save_sparse_coo_tensor_with_convention(sparse_tensor_original, file_path, interpretation_note=custom_note)
    print(f"Tensor saved to {file_path}")

    # Load the tensor (to CPU for this example)
    loaded_tensor_cpu, note_cpu = load_sparse_coo_tensor_with_convention(file_path, target_device='cpu')
    print(f"Loaded sparse tensor (on {loaded_tensor_cpu.device}):\n", loaded_tensor_cpu)
    print(f"Interpretation note (CPU): {note_cpu}")
    print(f"Loaded values dtype: {loaded_tensor_cpu.values().dtype}, Tensor dtype: {loaded_tensor_cpu.dtype}")


    # Verify content (optional)
    assert torch.allclose(sparse_tensor_original.to_dense().cpu(), loaded_tensor_cpu.to_dense()), "Tensor content mismatch (CPU)"
    assert note_cpu == custom_note, "Interpretation note mismatch (CPU)"


    if torch.cuda.is_available():
        # Load the tensor (to CUDA if available)
        loaded_tensor_cuda, note_cuda = load_sparse_coo_tensor_with_convention(file_path, target_device='cuda')
        print(f"Loaded sparse tensor (on {loaded_tensor_cuda.device}):\n", loaded_tensor_cuda)
        print(f"Interpretation note (CUDA): {note_cuda}")
        print(f"Loaded values dtype: {loaded_tensor_cuda.values().dtype}, Tensor dtype: {loaded_tensor_cuda.dtype}")
        assert torch.allclose(sparse_tensor_original.to_dense(), loaded_tensor_cuda.to_dense()), "Tensor content mismatch (CUDA)"
        assert note_cuda == custom_note, "Interpretation note mismatch (CUDA)"
        
    # Test with an empty sparse tensor
    print("\nTesting with an empty sparse tensor...")
    empty_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.empty((len(size),0), dtype=torch.long, device=device),
        values=torch.empty((0,), dtype=torch.float64, device=device),
        size=size,
        dtype=torch.float64,
        device=device
    )
    empty_file_path = "temp_empty_sparse_tensor.pt"
    save_sparse_coo_tensor_with_convention(empty_sparse_tensor, empty_file_path)
    print(f"Empty tensor saved to {empty_file_path}")
    loaded_empty_tensor, empty_note = load_sparse_coo_tensor_with_convention(empty_file_path, target_device=device)
    print(f"Loaded empty sparse tensor (on {loaded_empty_tensor.device}):\n", loaded_empty_tensor)
    print(f"Interpretation note: {empty_note}")
    assert loaded_empty_tensor.is_sparse
    assert loaded_empty_tensor.values().numel() == 0
    assert loaded_empty_tensor.indices().numel() == 0
    assert loaded_empty_tensor.size() == size
    assert loaded_empty_tensor.dtype == torch.float64

    print("\nExample usage complete. Cleaning up temporary files...")
    import os
    os.remove(file_path)
    os.remove(empty_file_path)
    print("Temporary files removed.") 