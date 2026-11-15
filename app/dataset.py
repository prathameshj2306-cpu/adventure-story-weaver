
from pathlib import Path

import torch

from app.bpe_tokenizer import BPETokenizer


class AdventureDataset:
    def __init__(
        self,
        train_corpus_path,
        val_corpus_path,
        tokenizer_path
    ):
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        train_text = Path(train_corpus_path).read_text(
            encoding="utf-8"
        )

        val_text = Path(val_corpus_path).read_text(
            encoding="utf-8"
        )

        train_encoded = self.tokenizer.encode(train_text)
        val_encoded = self.tokenizer.encode(val_text)

        self.train_data = torch.tensor(
            train_encoded,
            dtype=torch.long
        )

        self.val_data = torch.tensor(
            val_encoded,
            dtype=torch.long
        )

    def summary(self):
        print("========== DATASET SUMMARY ==========")

        print("Vocabulary size :", self.tokenizer.vocab_size)
        print("Training tokens :", len(self.train_data))
        print("Validation tokens:", len(self.val_data))

        print("\nTraining tensor shape  :", self.train_data.shape)
        print("Validation tensor shape:", self.val_data.shape)
