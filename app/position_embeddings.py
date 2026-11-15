
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

    def forward(self, sequence_length, device):
        positions = torch.arange(
            sequence_length,
            device=device
        )

        # Convert position IDs into vectors
        return self.embedding(positions)
