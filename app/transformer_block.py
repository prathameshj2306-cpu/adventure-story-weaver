
import torch
import torch.nn as nn

from app.attention import MultiHeadSelfAttention
from app.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        # First Layer Normalization
        self.ln1 = nn.LayerNorm(embedding_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size
        )

        # Second Layer Normalization
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Feed-Forward Network
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x):

        # Attention + Residual Connection
        x = x + self.attention(self.ln1(x))

        # Feed-Forward + Residual Connection
        x = x + self.feed_forward(self.ln2(x))

        return x
