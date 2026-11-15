
import torch
from app.layer_norm import LayerNormalization

embedding_dim = 16
batch_size = 2
sequence_length = 64

# Create sample input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

# Create Layer Normalization
layer_norm = LayerNormalization(embedding_dim)

# Apply normalization
output = layer_norm(x)

print("========== LAYER NORMALIZATION TEST ==========")
print("Input shape :", x.shape)
print("Output shape:", output.shape)

# Check output shape
assert output.shape == (2, 64, 16)

# Check normalized mean and variance
mean = output.mean(dim=-1)
variance = output.var(dim=-1, unbiased=False)

print("Mean of first token:", mean[0, 0].item())
print("Variance of first token:", variance[0, 0].item())

assert torch.allclose(
    mean,
    torch.zeros_like(mean),
    atol=1e-5
)

assert torch.allclose(
    variance,
    torch.ones_like(variance),
    atol=1e-4
)

print("\nLayer normalization test: PASSED")
