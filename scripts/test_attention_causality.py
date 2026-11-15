
import torch
from app.attention import MultiHeadSelfAttention

torch.manual_seed(42)

embedding_dim = 16
num_heads = 4
block_size = 8

attention = MultiHeadSelfAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    block_size=block_size
)

attention.eval()

# Create a sample input sequence
x1 = torch.randn(1, 5, embedding_dim)

# Make a copy and modify only the final token
x2 = x1.clone()
x2[:, 4, :] += 100

# Get attention outputs
with torch.no_grad():
    output1 = attention(x1)
    output2 = attention(x2)

# Compare outputs for the first four positions
same_outputs = torch.allclose(
    output1[:, :4, :],
    output2[:, :4, :],
    atol=1e-6
)

print("========== CAUSAL ATTENTION TEST ==========")
print("Original output shape:", output1.shape)
print("Modified output shape:", output2.shape)
print("Earlier positions unchanged:", same_outputs)

assert same_outputs, "Causal attention test failed!"

print("\nCausal attention test: PASSED")
