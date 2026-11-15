
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Model configuration
vocab_size = dataset.tokenizer.vocab_size
embedding_dim = 16
block_size = 64

# Create embedding layers
token_embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    block_size,
    embedding_dim
)

# Take 10 token IDs and add a batch dimension
sample_ids = dataset.train_data[:10].unsqueeze(0)

# Shape: [1, 10]
print("Input shape:", sample_ids.shape)

# Convert token IDs into vectors
token_vectors = token_embedding(sample_ids)

# Generate positional vectors
position_vectors = position_embedding(
    sequence_length=sample_ids.shape[1],
    device=sample_ids.device
)

# Combine both embeddings
combined = token_vectors + position_vectors

print("Token embedding shape:", token_vectors.shape)
print("Position embedding shape:", position_vectors.shape)
print("Combined embedding shape:", combined.shape)

# Verify dimensions
assert token_vectors.shape == (1, 10, embedding_dim)
assert position_vectors.shape == (10, embedding_dim)
assert combined.shape == (1, 10, embedding_dim)

print("\nPositional embedding test: PASSED")
