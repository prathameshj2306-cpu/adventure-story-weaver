
import torch


def get_batch(data, batch_size, block_size, device="cpu"):
    """
    Generate a batch of input and target sequences.

    data:       Encoded text tensor
    batch_size: Number of sequences in one batch
    block_size: Number of characters in each sequence
    """

    # Random starting positions
    start_positions = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        data[start:start + block_size]
        for start in start_positions
    ])

    # Target sequences (shifted by one character)
    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in start_positions
    ])

    return x.to(device), y.to(device)
