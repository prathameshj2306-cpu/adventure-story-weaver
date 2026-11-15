
import torch
from app.feed_forward import FeedForward

embedding_dim = 16
batch_size = 2
sequence_length = 64

# Sample input
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Create feed-forward network
feed_forward = FeedForward(embedding_dim)

# Forward pass
output = feed_forward(x)

print("========== FEED-FORWARD TEST ==========")
print("Input shape :", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 64, 16)
assert torch.isfinite(output).all()

print("\nFeed-forward test: PASSED")
