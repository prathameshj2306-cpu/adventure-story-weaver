
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding
from app.transformer_block import TransformerBlock

torch.manual_seed(42)

embedding_dim = 16
num_heads = 4
block_size = 64
batch_size = 2

# Load actual adventure dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Get sample token IDs
sample_ids = dataset.train_data[:batch_size * block_size]
sample_ids = sample_ids.view(batch_size, block_size)

# Create embeddings
token_embedding = TokenEmbedding(
    dataset.tokenizer.vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    block_size,
    embedding_dim
)

token_vectors = token_embedding(sample_ids)

position_vectors = position_embedding(
    sequence_length=block_size,
    device=sample_ids.device
)

x = token_vectors + position_vectors

# Create Transformer Block
transformer = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    block_size=block_size
)

# Forward pass
output = transformer(x)

print("========== TRANSFORMER BLOCK TEST ==========")
print("Input shape :", x.shape)
print("Output shape:", output.shape)

assert output.shape == (batch_size, block_size, embedding_dim)
assert torch.isfinite(output).all()

print("\nTransformer Block test: PASSED")
