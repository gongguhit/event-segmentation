import torch
import sys
from utils.device_utils import get_device

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Get device using our utility
    device = get_device()
    print(f"Selected device: {device}")
    
    # Create a test tensor
    x = torch.randn(3, 3)
    print(f"CPU tensor: {x}")
    
    # Move to selected device
    x = x.to(device)
    print(f"Device tensor: {x}")
    print(f"Tensor device: {x.device}")
    
    # Try a basic operation
    y = x @ x
    print(f"Matrix multiplication result: {y}")
    print(f"Result device: {y.device}")
    
    print("\nIf you see 'mps' in the device outputs above, MPS is working correctly!")

if __name__ == "__main__":
    main() 