
from app.dataset import AdventureDataset

dataset = AdventureDataset(
    train_corpus_path="data/processed/train_corpus.txt",
    val_corpus_path="data/processed/val_corpus.txt",
    tokenizer_path="tokenizer/adventure_bpe_v2.json"
)

dataset.summary()

# Verify that both datasets contain tokens
assert len(dataset.train_data) > 0
assert len(dataset.val_data) > 0

# Verify token IDs are within the tokenizer vocabulary
assert dataset.train_data.min() >= 0
assert dataset.val_data.min() >= 0

assert dataset.train_data.max() < dataset.tokenizer.vocab_size
assert dataset.val_data.max() < dataset.tokenizer.vocab_size

print("\n========== BPE DATASET TEST PASSED ==========")
