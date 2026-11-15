
import torch

from app.attention import (
    QKVProjection,
    AttentionScores,
    ScaledAttentionScores,
    MaskedAttentionSoftmax,
    WeightedValueOutput
)

embedding_dim = 16
batch_size = 2
sequence_length = 64

# Generate sample embeddings
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Step 1: Generate Q, K, V
projection = QKVProjection(embedding_dim)
q, k, v = projection(x)

# Step 2: Calculate attention scores
scores = AttentionScores()(q, k)

# Step 3: Scale scores
scaled_scores = ScaledAttentionScores(embedding_dim)(scores)

# Step 4: Apply causal mask and softmax
weights = MaskedAttentionSoftmax(sequence_length)(scaled_scores)

# Step 5: Calculate weighted Value output
output = WeightedValueOutput()(weights, v)

print("========== WEIGHTED VALUE TEST ==========")
print("Input shape            :", x.shape)
print("Attention weights shape:", weights.shape)
print("Value shape            :", v.shape)
print("Final output shape     :", output.shape)

# Verify dimensions
assert output.shape == (batch_size, sequence_length, embedding_dim)

# Verify numerical validity
assert torch.isfinite(output).all()

print("\nWeighted Value test: PASSED")
