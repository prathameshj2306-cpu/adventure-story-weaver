
from pathlib import Path

from tokenizers import Tokenizer, models, trainers
from tokenizers import pre_tokenizers, decoders


class BPETokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(cls, corpus_path, vocab_size=1000):
        tokenizer = Tokenizer(
            models.BPE(unk_token="[UNK]")
        )

        # Split text into byte-level units while preserving spaces.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )

        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        tokenizer.train([str(corpus_path)], trainer)

        return cls(tokenizer)

    @classmethod
    def load(cls, tokenizer_path):
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer)

    def save(self, tokenizer_path):
        Path(tokenizer_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self.tokenizer.save(str(tokenizer_path))

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
