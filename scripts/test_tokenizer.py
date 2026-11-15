
from pathlib import Path
from app.tokenizer import CharacterTokenizer

# Load our processed adventure corpus
text = Path("data/processed/adventure_corpus.txt").read_text(
    encoding="utf-8"
)

# Initialize tokenizer
tokenizer = CharacterTokenizer(text)

# Test with a sample sentence
sample = "The treasure is hidden."

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Original text :", sample)
print("Encoded IDs   :", encoded)
print("Decoded text  :", decoded)

print("\n========== TOKENIZER SUMMARY ==========")
print("Vocabulary size:", tokenizer.vocab_size)
print("Unique characters:", len(tokenizer.chars))

# Verify exact reconstruction
assert decoded == sample

# Verify entire corpus can be encoded and decoded
all_ids = tokenizer.encode(text)
reconstructed = tokenizer.decode(all_ids)

assert reconstructed == text

print("\nSample test: PASSED")
print("Full corpus round-trip test: PASSED")
