
import pytest
import torch
import torch.nn as nn

from app.generate import generate


# ==========================================
# 1. DUMMY GPT MODEL FOR TESTING
# ==========================================

class DummyGPT(nn.Module):

    def __init__(self, vocab_size=20):
        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_lengths = []

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        self.sequence_lengths.append(sequence_length)

        # Return logits with shape:
        # [batch_size, sequence_length, vocab_size]

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size
        )

        return logits


# ==========================================
# 2. TEST BASIC GENERATION
# ==========================================

def test_generation_output_shape():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2, 3]])

    output = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=5,
        block_size=8,
        temperature=0.8,
        top_k=10
    )

    assert output.shape == (1, 8)


# ==========================================
# 3. TEST TOKEN GENERATION COUNT
# ==========================================

def test_correct_number_of_tokens_generated():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2]])

    output = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=4,
        block_size=8
    )

    assert output.shape[1] - input_ids.shape[1] == 4


# ==========================================
# 4. TEST BATCH GENERATION
# ==========================================

def test_batch_generation():

    model = DummyGPT()

    input_ids = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

    output = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=3,
        block_size=8
    )

    assert output.shape == (2, 6)


# ==========================================
# 5. TEST ZERO NEW TOKENS
# ==========================================

def test_zero_new_tokens():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2, 3]])

    output = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=0,
        block_size=8
    )

    assert torch.equal(output, input_ids)


# ==========================================
# 6. TEST CONTEXT WINDOW LIMIT
# ==========================================

def test_context_window_limit():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=2,
        block_size=4
    )

    assert all(
        length <= 4
        for length in model.sequence_lengths
    )


# ==========================================
# 7. TEST INVALID TEMPERATURE
# ==========================================

def test_invalid_temperature():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=8,
            temperature=0
        )


# ==========================================
# 8. TEST INVALID TOP-K
# ==========================================

def test_invalid_top_k():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=8,
            top_k=0
        )


# ==========================================
# 9. TEST INVALID REPETITION PENALTY
# ==========================================

def test_invalid_repetition_penalty():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=8,
            repetition_penalty=0
        )


# ==========================================
# 10. TEST EMPTY INPUT
# ==========================================

def test_empty_input():

    model = DummyGPT()

    input_ids = torch.empty((1, 0), dtype=torch.long)

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=8
        )


# ==========================================
# 11. TEST INVALID INPUT DIMENSION
# ==========================================

def test_invalid_input_dimension():

    model = DummyGPT()

    input_ids = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=8
        )


# ==========================================
# 12. TEST NEGATIVE TOKEN COUNT
# ==========================================

def test_negative_new_tokens():

    model = DummyGPT()

    input_ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError):

        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=-1,
            block_size=8
        )


# ==========================================
# 13. TEST MODEL EVALUATION MODE
# ==========================================

def test_model_evaluation_mode():

    model = DummyGPT()

    model.train()

    input_ids = torch.tensor([[1, 2]])

    generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=2,
        block_size=8
    )

    assert model.training is False
