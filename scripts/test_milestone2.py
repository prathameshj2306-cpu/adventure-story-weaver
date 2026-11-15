
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding

# Configuration
embedding_dim = 16
block_size = 64
batch_size = 2

# 1. Load and encode the corpus
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# 2. Create embedding layers
token_embedding = TokenEmbedding(
    dataset.tokenizer.vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    block_size,
    embedding_dim
)

# 3. Prepare a batch of two sequences, each 64 tokens long
sample_ids = dataset.train_data[:batch_size * block_size]
sample_ids = sample_ids.view(batch_size, block_size)

# 4. Generate token embeddings
token_vectors = token_embedding(sample_ids)

# 5. Generate positional embeddings
position_vectors = position_embedding(
    sequence_length=block_size,
    device=sample_ids.device
)

# 6. Combine both embeddings
combined = token_vectors + position_vectors

# 7. Display results
print("========== MILESTONE 2 INTEGRATION ==========")
print("Vocabulary size:", dataset.tokenizer.vocab_size)
print("Training data shape:", dataset.train_data.shape)
print("Validation data shape:", dataset.val_data.shape)
print("Input batch shape:", sample_ids.shape)
print("Token embeddings:", token_vectors.shape)
print("Position embeddings:", position_vectors.shape)
print("Final combined output:", combined.shape)

# 8. Verify everything
assert combined.shape == (batch_size, block_size, embedding_dim)
assert torch.is_floating_point(combined)
assert torch.isfinite(combined).all()

print("\nAll integration checks passed!")
