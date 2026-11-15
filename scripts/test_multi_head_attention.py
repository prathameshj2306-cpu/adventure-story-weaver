
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding
from app.attention import MultiHeadSelfAttention

# Configuration
embedding_dim = 16
num_heads = 4
block_size = 64
batch_size = 2

# Load actual project dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Prepare input token IDs
sample_ids = dataset.train_data[:batch_size * block_size]
sample_ids = sample_ids.view(batch_size, block_size)

# Generate embeddings
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

# Initialize Multi-Head Attention
attention = MultiHeadSelfAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    block_size=block_size
)

# Forward pass
output = attention(x)

print("========== MULTI-HEAD ATTENTION TEST ==========")
print("Input shape       :", x.shape)
print("Number of heads   :", num_heads)
print("Head size         :", embedding_dim // num_heads)
print("Output shape      :", output.shape)

# Verify output dimensions
assert output.shape == (batch_size, block_size, embedding_dim)

# Verify numerical validity
assert torch.isfinite(output).all()

print("\nMulti-head attention test: PASSED")
