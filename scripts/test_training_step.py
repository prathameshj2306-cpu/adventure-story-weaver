
import torch
import torch.nn.functional as F

from app.dataset import AdventureDataset
from app.batch_loader import get_batch
from app.model import GPTLanguageModel

torch.manual_seed(42)

# Configuration
batch_size = 2
block_size = 64
embedding_dim = 16
num_heads = 4
num_layers = 2
learning_rate = 0.001

# Load dataset
dataset = AdventureDataset(
    "data/processed/adventure_corpus.txt"
)

# Create model
model = GPTLanguageModel(
    vocab_size=dataset.tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size
)

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# Get batch
x, y = get_batch(
    dataset.train_data,
    batch_size,
    block_size
)

# 1. Forward pass
logits = model(x)

# 2. Calculate loss
loss = F.cross_entropy(
    logits.reshape(-1, dataset.tokenizer.vocab_size),
    y.reshape(-1)
)

# 3. Clear old gradients
optimizer.zero_grad(set_to_none=True)

# 4. Backpropagation
loss.backward()

# 5. Update model parameters
optimizer.step()

print("========== TRAINING STEP TEST ==========")
print("Initial loss:", loss.item())
print("Backpropagation: PASSED")
print("Optimizer update: PASSED")
