
import torch

from app.attention import (
    QKVProjection,
    AttentionScores,
    ScaledAttentionScores
)

embedding_dim = 16
batch_size = 2
sequence_length = 64

x = torch.randn(batch_size, sequence_length, embedding_dim)

projection = QKVProjection(embedding_dim)
q, k, v = projection(x)

attention = AttentionScores()
scores = attention(q, k)

scaler = ScaledAttentionScores(head_size=embedding_dim)
scaled_scores = scaler(scores)

print("========== SCALED ATTENTION TEST ==========")
print("Raw scores shape    :", scores.shape)
print("Scaled scores shape :", scaled_scores.shape)
print("Scaling factor      :", scaler.scale)

assert scaled_scores.shape == scores.shape
assert torch.allclose(scaled_scores, scores / 4)
assert torch.isfinite(scaled_scores).all()

print("\nScaled attention test: PASSED")
