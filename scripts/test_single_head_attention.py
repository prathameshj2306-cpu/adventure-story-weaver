
import torch

from app.dataset import AdventureDataset
from app.embeddings import TokenEmbedding
from app.position_embeddings import PositionalEmbedding
from app.attention import SingleHeadSelfAttention

# Configuration
embedding_dim = 16
block_size = 64
batch_size = 2

# Load actual project dataset
dataset = AdventureDataset("data/processed/adventure_corpus.txt")

# Generate sample token IDs
sample_ids = dataset.train_data[:batch_size * block_size]
sample_ids = sample_ids.view(batch_size, block_size)

# Generate combined embeddings
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

# Run single-head attention
attention = SingleHeadSelfAttention(
    embedding_dim=embedding_dim,
    block_size=block_size
)

output = attention(x)

print("========== SINGLE HEAD ATTENTION TEST ==========")
print("Input embeddings :", x.shape)
print("Attention output :", output.shape)

assert output.shape == (batch_size, block_size, embedding_dim)
assert torch.isfinite(output).all()

print("\nSingle-head self-attention test: PASSED")
