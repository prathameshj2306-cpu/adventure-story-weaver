
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Configuration
vocab_size = dataset.tokenizer.vocab_size
embedding_dim = 16

# Create embedding layer
embedding_layer = TokenEmbedding(
    vocab_size,
    embedding_dim
)

# Take a small sample of token IDs
sample_ids = dataset.train_data[:10]

# Convert IDs into embedding vectors
embedded = embedding_layer(sample_ids)

print("Vocabulary size:", vocab_size)
print("Embedding dimension:", embedding_dim)
print("Input shape:", sample_ids.shape)
print("Output shape:", embedded.shape)

# Verify dimensions
assert embedded.shape == (10, embedding_dim)

# Verify output is floating-point
assert torch.is_floating_point(embedded)

print("\nToken embedding test: PASSED")
