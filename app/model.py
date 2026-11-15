
import torch
import torch.nn as nn

from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding
from app.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        block_size
    ):
        super().__init__()

        self.block_size = block_size

        # 1. Token Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # 2. Positional Embeddings
        self.position_embedding = PositionalEmbedding(
            block_size,
            embedding_dim
        )

        # 3. Multiple Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                block_size=block_size
            )
            for _ in range(num_layers)
        ])

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(embedding_dim)

        # 5. Language Model Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                "Input sequence exceeds block size"
            )

        # Token embeddings
        token_vectors = self.token_embedding(token_ids)

        # Positional embeddings
        position_vectors = self.position_embedding(
            sequence_length,
            token_ids.device
        )

        # Combine embeddings
        x = token_vectors + position_vectors

        # Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Convert representations into vocabulary scores
        logits = self.lm_head(x)

        return logits
