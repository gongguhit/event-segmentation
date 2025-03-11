import torch
import platform
import logging

def get_device(logger=None):
    """
    Get the best available device for PyTorch.
    For Apple Silicon (M1/M2/M3), use MPS if available.
    Falls back to CUDA if available, otherwise CPU.
    
    Args:
        logger: Optional logger to record device information
        
    Returns:
        torch.device: The selected device
    """
    # Check if we're on macOS with Apple Silicon
    is_mac_apple_silicon = (
        platform.system() == "Darwin" and 
        (platform.processor() == "arm" or "Apple M" in platform.processor())
    )
    
    # Check for MPS (Metal Performance Shaders)
    if is_mac_apple_silicon and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple Metal (MPS)"
    # Fall back to CUDA if available
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    # Otherwise use CPU
    else:
        device = torch.device("cpu")
        device_info = "CPU"
    
    # Log device information if logger is provided
    if logger:
        logger.info(f"Using device: {device_info}")
    else:
        print(f"Using device: {device_info}")
    
    return device

def to_device(data, device):
    """
    Recursively move data to the specified device.
    Handles dictionaries, lists, tuples, and tensors.
    
    Args:
        data: The data to move (can be a complex nested structure)
        device: The target device
        
    Returns:
        Data on the target device
    """
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def mps_fix_for_training():
    """
    Apply fixes and workarounds for MPS training if needed.
    Some operations might not be supported in the current MPS implementation.
    
    This function should be updated when new issues are found or when PyTorch updates fix them.
    """
    # Current PyTorch version with MPS support has some limitations:
    # 1. Some operations may not be supported or may behave differently
    # 2. Some advanced operations may fall back to CPU silently
    
    # Check if we're actually using MPS
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        return
    
    # Set environment variables for better MPS performance
    import os
    
    # Enable the 'eager' mode for Metal operations
    # This can improve the performance on M1/M2/M3 chips
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Apply specific fixes to PyTorch functions with known issues
    # These will be updated as PyTorch's MPS backend matures
    
    # Patch torch functions if needed
    original_cat = torch.cat
    
    def safe_cat(tensors, *args, **kwargs):
        """
        Safely concatenate tensors on MPS device, falling back to CPU if needed.
        Some concatenation operations may not work correctly on MPS.
        """
        try:
            return original_cat(tensors, *args, **kwargs)
        except Exception as e:
            if any(t.device.type == 'mps' for t in tensors if isinstance(t, torch.Tensor)):
                # Move to CPU, concatenate, then move back to MPS
                cpu_tensors = [t.cpu() if isinstance(t, torch.Tensor) else t for t in tensors]
                result = original_cat(cpu_tensors, *args, **kwargs)
                # Get the original MPS device
                mps_device = next(t.device for t in tensors if isinstance(t, torch.Tensor) and t.device.type == 'mps')
                return result.to(mps_device)
            else:
                # Re-raise the exception if it's not MPS related
                raise
    
    # Only patch if we're using MPS to avoid affecting other backends
    if torch.device("mps").type == "mps":
        torch.cat = safe_cat 