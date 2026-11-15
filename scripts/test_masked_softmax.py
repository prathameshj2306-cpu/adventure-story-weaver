
import torch

from app.attention import MaskedAttentionSoftmax

sequence_length = 5

# Create sample scores
scores = torch.randn(1, sequence_length, sequence_length)

# Apply causal masking and softmax
masked_softmax = MaskedAttentionSoftmax(sequence_length)
weights = masked_softmax(scores)

print("========== MASKED SOFTMAX TEST ==========")
print("Attention weights:")
print(weights[0])

# Check shape
assert weights.shape == (1, sequence_length, sequence_length)

# Check future positions have zero attention
for i in range(sequence_length):
    for j in range(i + 1, sequence_length):
        assert weights[0, i, j].item() == 0.0

# Check every row sums to 1
row_sums = weights.sum(dim=-1)

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6
)

print("\nRow sums:", row_sums)
print("\nMasked softmax test: PASSED")
