import torch

assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."

print(torch.cuda.is_available())