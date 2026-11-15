
import torch

from app.attention import QKVProjection, AttentionScores

embedding_dim = 16
batch_size = 2
sequence_length = 64

# Sample input
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Generate Q and K
projection = QKVProjection(embedding_dim)
q, k, v = projection(x)

# Calculate attention scores
attention = AttentionScores()
scores = attention(q, k)

print("========== ATTENTION SCORES TEST ==========")
print("Query shape          :", q.shape)
print("Key shape            :", k.shape)
print("Transposed Key shape :", k.transpose(-2, -1).shape)
print("Attention scores     :", scores.shape)

# Verify expected shape
assert scores.shape == (batch_size, sequence_length, sequence_length)

assert torch.isfinite(scores).all()

print("\nAttention scores test: PASSED")
