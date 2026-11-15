
import torch

from app.attention import CausalMask

sequence_length = 5

# Create sample attention scores
scores = torch.randn(1, sequence_length, sequence_length)

# Apply causal mask
causal_mask = CausalMask(block_size=sequence_length)
masked_scores = causal_mask(scores)

print("========== CAUSAL MASK TEST ==========")

print("Original scores shape:", scores.shape)
print("Masked scores shape  :", masked_scores.shape)

print("\nMasked score matrix:")
print(masked_scores[0])

# Verify all future positions are negative infinity
for i in range(sequence_length):
    for j in range(i + 1, sequence_length):
        assert torch.isneginf(masked_scores[0, i, j])

# Verify allowed positions remain unchanged
for i in range(sequence_length):
    for j in range(i + 1):
        assert masked_scores[0, i, j] == scores[0, i, j]

print("\nCausal masking test: PASSED")
