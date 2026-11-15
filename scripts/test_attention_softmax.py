
import torch

from app.attention import (
    QKVProjection,
    AttentionScores,
    ScaledAttentionScores,
    AttentionSoftmax
)

embedding_dim = 16
batch_size = 2
sequence_length = 64

x = torch.randn(batch_size, sequence_length, embedding_dim)

projection = QKVProjection(embedding_dim)
q, k, v = projection(x)

scores = AttentionScores()(q, k)

scaled_scores = ScaledAttentionScores(
    head_size=embedding_dim
)(scores)

weights = AttentionSoftmax()(scaled_scores)

print("========== SOFTMAX TEST ==========")
print("Scaled scores shape :", scaled_scores.shape)
print("Attention weights   :", weights.shape)

# Check shape
assert weights.shape == (batch_size, sequence_length, sequence_length)

# Every row must sum to approximately 1
row_sums = weights.sum(dim=-1)

print("First row weight sum:", row_sums[0, 0].item())

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6
)

# Check valid probability range
assert torch.all(weights >= 0)
assert torch.all(weights <= 1)

print("\nSoftmax test: PASSED")
