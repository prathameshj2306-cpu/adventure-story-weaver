
from pathlib import Path
from collections import Counter
import re

# Load processed corpus
corpus_path = Path("data/processed/adventure_corpus.txt")
text = corpus_path.read_text(encoding="utf-8")

# Basic statistics
words = text.split()
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

print("========== DATASET QUALITY REPORT ==========")

print(f"Total characters : {len(text):,}")
print(f"Total words      : {len(words):,}")
print(f"Total paragraphs: {len(paragraphs):,}")
print(f"Unique characters: {len(set(text)):,}")

# Paragraph analysis
lengths = [len(p.split()) for p in paragraphs]

print(f"\nAverage paragraph length: {sum(lengths) / len(lengths):.2f} words")
print(f"Shortest paragraph      : {min(lengths)} words")
print(f"Longest paragraph       : {max(lengths)} words")

# Most frequent characters
print("\n========== TOP 15 CHARACTERS ==========")

for char, count in Counter(text).most_common(15):
    display_char = repr(char)
    print(f"{display_char:10} : {count:,}")

# Check for Gutenberg metadata
print("\n========== METADATA CHECK ==========")

patterns = [
    r"PROJECT GUTENBERG",
    r"END OF THE PROJECT GUTENBERG",
    r"START OF THE PROJECT GUTENBERG"
]

for pattern in patterns:
    found = bool(re.search(pattern, text, re.IGNORECASE))
    print(f"{pattern}: {'FOUND' if found else 'NOT FOUND'}")
