
import torch

from app.attention import QKVProjection


# Configuration
embedding_dim = 16
batch_size = 2
sequence_length = 64

# Create sample input embeddings
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Initialize QKV projection
qkv = QKVProjection(embedding_dim)

# Generate Q, K and V
q, k, v = qkv(x)

# Display shapes
print("========== QKV PROJECTION TEST ==========")

print("Input shape  :", x.shape)
print("Query shape  :", q.shape)
print("Key shape    :", k.shape)
print("Value shape  :", v.shape)

# Verify shapes
expected_shape = (batch_size, sequence_length, embedding_dim)

assert q.shape == expected_shape
assert k.shape == expected_shape
assert v.shape == expected_shape

# Verify numerical validity
assert torch.isfinite(q).all()
assert torch.isfinite(k).all()
assert torch.isfinite(v).all()

print("\nQKV projection test: PASSED")
