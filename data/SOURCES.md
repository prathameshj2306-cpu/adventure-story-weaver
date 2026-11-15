# Dataset Sources

The Adventure Story Weaver project uses public-domain literary texts from Project Gutenberg for training and experimentation.

## 1. Treasure Island

- **Author:** Robert Louis Stevenson
- **Project Gutenberg eBook ID:** 120
- **Source:** https://www.gutenberg.org/cache/epub/120/pg120.txt
- **Local file:** `data/raw/treasure_island.txt`

## 2. The Wonderful Wizard of Oz

- **Author:** L. Frank Baum
- **Project Gutenberg eBook ID:** 55
- **Source:** https://www.gutenberg.org/cache/epub/55/pg55.txt
- **Local file:** `data/raw/wizard_of_oz.txt`

## Preprocessing

Both source texts are cleaned and combined into:

`data/processed/adventure_corpus.txt`

The preprocessing script:
- Removes Project Gutenberg headers and footers.
- Normalizes line endings and whitespace.
- Preserves paragraph boundaries.
- Combines the cleaned texts into one corpus.

**Dataset statistics:**
- Books: 2
- Total words: 108,286
- Total characters: 570,612

**Retrieval date:** 15 November 2026

Refer to Project Gutenberg's terms and the individual ebook notices before redistributing source texts or derived datasets.
