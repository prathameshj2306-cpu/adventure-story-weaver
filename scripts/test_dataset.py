
from app.dataset import AdventureDataset

dataset = AdventureDataset(
    corpus_path="data/processed/adventure_corpus.txt",
    train_split=0.9
)

dataset.summary()

# Verify the split
assert len(dataset.data) == (
    len(dataset.train_data) + len(dataset.val_data)
)

assert len(dataset.train_data) > len(dataset.val_data)

assert dataset.train_data.dtype == __import__("torch").long

print("\nDataset split test: PASSED")
