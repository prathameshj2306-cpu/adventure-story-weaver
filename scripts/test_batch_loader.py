
from app.dataset import AdventureDataset
from app.batch_loader import get_batch
import torch
# Configuration
batch_size = 2
block_size = 8

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Generate a batch
x, y = get_batch(
    dataset.train_data,
    batch_size,
    block_size
)

print("========== BATCH LOADER TEST ==========")
print("Input shape :", x.shape)
print("Target shape:", y.shape)

print("\nInput tokens:")
print(x)

print("\nTarget tokens:")
print(y)

# Verify dimensions
assert x.shape == (batch_size, block_size)
assert y.shape == (batch_size, block_size)

# Verify next-character shifting
assert torch.equal(x[:, 1:], y[:, :-1])

print("\nBatch loader test: PASSED")
