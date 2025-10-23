"""
Verify the Installation
After the installation completes, run the following Python code
to confirm PyTorch is installed and can detect your GPU.
"""

import torch

print("Torch version:", torch.__version__)

# 1. Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")
print("CUDA version built with PyTorch:", torch.version.cuda)

# 2. Check the number of devices (should be > 0)
print(f"Device Count: {torch.cuda.device_count()}")

# 3. Print the name of your first GPU
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 4. Check the PyTorch-compiled CUDA version
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# 5. Optional: Run a test tensor on the GPU
if torch.cuda.is_available():
    x = torch.rand(2, 3, device='cuda')
    print(f"Test Tensor on GPU:\n{x}")

"""
Expected output:
CUDA Available: True
Device Count: 1 (or more, depending on your setup)
Device Name: [Your GPU Model Name]
PyTorch CUDA Version: 12.4 (or whatever version you installed)
"""