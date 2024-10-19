import torch

print(torch.cuda.is_available())  # This should return True if your GPU is accessible
print(torch.cuda.get_device_name(0))  # This will show the name of the GPU detected by PyTorch
