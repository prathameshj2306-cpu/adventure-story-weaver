
from app.bpe_tokenizer import BPETokenizer

CORPUS_PATH = "data/processed/train_corpus.txt"
TOKENIZER_PATH = "tokenizer/adventure_bpe_v2.json"

print("========== BPE TOKENIZER TRAINING ==========")

tokenizer = BPETokenizer.train(
    corpus_path=CORPUS_PATH,
    vocab_size=1000
)

tokenizer.save(TOKENIZER_PATH)

print("\n========== TRAINING COMPLETE ==========")
print("Vocabulary size:", tokenizer.vocab_size)
print("Saved tokenizer:", TOKENIZER_PATH)
