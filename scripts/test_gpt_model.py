
import torch

from app.dataset import AdventureDataset
from app.model import GPTLanguageModel

torch.manual_seed(42)

# Model configuration
embedding_dim = 64
num_heads = 4
num_layers = 4
block_size = 128
batch_size = 2

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

vocab_size = dataset.tokenizer.vocab_size

# Sample input token IDs
sample_ids = dataset.train_data[:batch_size * block_size]
sample_ids = sample_ids.view(batch_size, block_size)

# Create GPT model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size
)

# Forward pass
logits = model(sample_ids)

print("========== GPT MODEL TEST ==========")
print("Vocabulary size:", vocab_size)
print("Number of Transformer blocks:", num_layers)
print("Input shape:", sample_ids.shape)
print("Output logits shape:", logits.shape)

# Verify output dimensions
assert logits.shape == (
    batch_size,
    block_size,
    vocab_size
)

# Verify numerical stability
assert torch.isfinite(logits).all()

print("\nGPT model test: PASSED")
