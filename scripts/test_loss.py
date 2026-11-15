
import torch
import torch.nn.functional as F

from app.dataset import AdventureDataset
from app.batch_loader import get_batch
from app.model import GPTLanguageModel

# Configuration
batch_size = 2
block_size = 64
embedding_dim = 16
num_heads = 4
num_layers = 2

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Get a training batch
x, y = get_batch(
    dataset.train_data,
    batch_size,
    block_size
)

# Create GPT model
model = GPTLanguageModel(
    vocab_size=dataset.tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size
)

# Get predictions
logits = model(x)

# Calculate loss
loss = F.cross_entropy(
    logits.reshape(-1, dataset.tokenizer.vocab_size),
    y.reshape(-1)
)

print("========== LOSS TEST ==========")
print("Logits shape:", logits.shape)
print("Targets shape:", y.shape)
print("Loss:", loss.item())

assert torch.isfinite(loss)

print("\nLoss calculation: PASSED")
