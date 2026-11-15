
import re
from pathlib import Path

# Project directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Input books
BOOKS = [
    "treasure_island.txt",
    "wizard_of_oz.txt"
]


def clean_gutenberg_text(file_path):
    """Read and clean a Project Gutenberg text file."""

    text = file_path.read_text(encoding="utf-8")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Project Gutenberg header
    start_pattern = re.compile(
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        re.IGNORECASE | re.DOTALL
    )

    # Remove Project Gutenberg footer
    end_pattern = re.compile(
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        re.IGNORECASE | re.DOTALL
    )

    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    if not start_match or not end_match:
        raise ValueError(
            f"Gutenberg start/end markers not found in {file_path.name}"
        )

    if end_match.start() <= start_match.end():
        raise ValueError(f"Invalid Gutenberg markers in {file_path.name}")

    text = text[start_match.end():end_match.start()]

    # Remove form-feed characters
    text = text.replace("\f", "\n")

    # Clean individual lines
    lines = [line.strip() for line in text.split("\n")]

    # Preserve paragraph boundaries
    paragraphs = []
    current_paragraph = []

    for line in lines:
        if line:
            current_paragraph.append(line)
        elif current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    cleaned_text = "\n\n".join(paragraphs)

    # Remove excessive spaces
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text.strip()



def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_books = []
    train_books = []
    val_books = []

    for book in BOOKS:
        file_path = RAW_DIR / book

        print(f"\nProcessing: {book}")

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        cleaned_text = clean_gutenberg_text(file_path)

        cleaned_books.append(cleaned_text)

        # Split at paragraph boundaries
        paragraphs = cleaned_text.split("\n\n")

        split_index = int(len(paragraphs) * 0.9)

        train_text = "\n\n".join(paragraphs[:split_index])
        val_text = "\n\n".join(paragraphs[split_index:])

        train_books.append(train_text)
        val_books.append(val_text)

        print(f"Total paragraphs: {len(paragraphs):,}")
        print(f"Training paragraphs: {len(paragraphs[:split_index]):,}")
        print(f"Validation paragraphs: {len(paragraphs[split_index:]):,}")

    # Combine training and validation sections separately
    full_corpus = "\n\n".join(cleaned_books)
    train_corpus = "\n\n".join(train_books)
    val_corpus = "\n\n".join(val_books)

    # Save all three datasets
    (PROCESSED_DIR / "adventure_corpus.txt").write_text(
        full_corpus,
        encoding="utf-8"
    )

    (PROCESSED_DIR / "train_corpus.txt").write_text(
        train_corpus,
        encoding="utf-8"
    )

    (PROCESSED_DIR / "val_corpus.txt").write_text(
        val_corpus,
        encoding="utf-8"
    )

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total characters: {len(full_corpus):,}")
    print(f"Training characters: {len(train_corpus):,}")
    print(f"Validation characters: {len(val_corpus):,}")

    print("\nFiles saved:")
    print(" - adventure_corpus.txt")
    print(" - train_corpus.txt")
    print(" - val_corpus.txt")

    print("\nDataset preparation complete!")



if __name__ == "__main__":
    main()
