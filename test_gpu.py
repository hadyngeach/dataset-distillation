import time
import torch

# Select GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch version: {torch.__version__}")
print(f"Selected device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Create large matrices directly on the GPU
size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Synchronise before timing GPU work
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

# Perform matrix multiplication
c = a @ b

# GPU operations are asynchronous, so wait for completion
if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print(f"Result device: {c.device}")
print(f"Matrix multiplication took: {elapsed:.3f} seconds")
print(f"Result sample: {c[0, 0].item():.4f}")

if device.type == "cuda":
    print(f"Peak allocated memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")