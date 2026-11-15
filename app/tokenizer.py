
class CharacterTokenizer:
    def __init__(self, text):
        # Find every unique character and sort them
        self.chars = sorted(set(text))

        # Vocabulary size
        self.vocab_size = len(self.chars)

        # Character -> Integer ID
        self.stoi = {
            ch: idx for idx, ch in enumerate(self.chars)
        }

        # Integer ID -> Character
        self.itos = {
            idx: ch for idx, ch in enumerate(self.chars)
        }

    def encode(self, text):
        """Convert a string into a list of integer IDs."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as error:
            raise ValueError(
                f"Unknown character: {error.args[0]!r}"
            ) from error

    def decode(self, ids):
        """Convert a list of integer IDs back into a string."""
        try:
            return "".join(self.itos[idx] for idx in ids)
        except KeyError as error:
            raise ValueError(
                f"Invalid token ID: {error.args[0]}"
            ) from error
