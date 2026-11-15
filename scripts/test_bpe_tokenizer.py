
from pathlib import Path

from app.bpe_tokenizer import BPETokenizer

CORPUS_PATH = "data/processed/adventure_corpus.txt"
TOKENIZER_PATH = "tokenizer/adventure_bpe.json"

# Load the trained tokenizer
tokenizer = BPETokenizer.load(TOKENIZER_PATH)

# Read the original corpus
original_text = Path(CORPUS_PATH).read_text(encoding="utf-8")

# Encode the complete corpus
encoded = tokenizer.encode(original_text)

# Decode it back
decoded = tokenizer.decode(encoded)

print("========== BPE TOKENIZER TEST ==========")

print("Vocabulary size:", tokenizer.vocab_size)
print("Original characters:", len(original_text))
print("BPE tokens:", len(encoded))
print("Decoded characters:", len(decoded))

print("\nOriginal and decoded text match:", original_text == decoded)

print("\nSample original text:")
print(original_text[:200])

print("\nSample encoded token IDs:")
print(encoded[:30])

print("\nSample decoded text:")
print(decoded[:200])

assert original_text == decoded, "Text reconstruction failed!"

print("\nBPE TOKENIZER TEST PASSED!")
