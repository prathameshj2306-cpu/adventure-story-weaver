
import torch
import torch.nn as nn
import math


class AttentionSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scaled_scores):
        weights = torch.softmax(scaled_scores, dim=-1)
        return weights


class AttentionScores(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        # Transpose the last two dimensions of Key
        k_transposed = k.transpose(-2, -1)

        # Calculate attention scores
        scores = torch.matmul(q, k_transposed)

        return scores


class ScaledAttentionScores(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.scale = math.sqrt(head_size)

    def forward(self, scores):
        return scores / self.scale


class QKVProjection(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        # Three separate learnable linear transformations
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        # Convert input embeddings into Q, K and V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        return q, k, v


class CausalMask(nn.Module):
    def __init__(self, block_size):
        super().__init__()

        # Create a lower triangular matrix
        mask = torch.tril(torch.ones(block_size, block_size))

        # Store mask as a model buffer
        self.register_buffer("mask", mask)

    def forward(self, scores):
        sequence_length = scores.size(-1)

        # Block future positions
        masked_scores = scores.masked_fill(
            self.mask[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        return masked_scores


class MaskedAttentionSoftmax(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.causal_mask = CausalMask(block_size)

    def forward(self, scaled_scores):
        # Apply causal mask first
        masked_scores = self.causal_mask(scaled_scores)

        # Apply softmax after masking
        weights = torch.softmax(masked_scores, dim=-1)

        return weights


class WeightedValueOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_weights, v):
        # Combine Value vectors using attention weights
        output = torch.matmul(attention_weights, v)

        return output


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, block_size):
        super().__init__()

        self.qkv = QKVProjection(embedding_dim)
        self.scale = ScaledAttentionScores(embedding_dim)
        self.masked_softmax = MaskedAttentionSoftmax(block_size)
        self.weighted_output = WeightedValueOutput()

    def forward(self, x):
        # 1. Generate Query, Key and Value
        q, k, v = self.qkv(x)

        # 2. Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))

        # 3. Scale scores
        scaled_scores = self.scale(scores)

        # 4. Apply causal masking and softmax
        weights = self.masked_softmax(scaled_scores)

        # 5. Calculate weighted Value output
        output = self.weighted_output(weights, v)

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

        # Project input into Query, Key and Value
        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False
        )

        # Combine information from all heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Causal mask shared by all heads
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        # 1. Generate Q, K and V together
        qkv = self.qkv_projection(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Split into multiple heads
        q = q.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        # 3. Calculate scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_size ** 0.5)

        # 4. Apply causal mask
        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # 5. Calculate attention weights
        weights = torch.softmax(scores, dim=-1)

        # 6. Combine Value vectors
        output = torch.matmul(weights, v)

        # 7. Join all heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        # 8. Final output projection
        output = self.output_projection(output)

        return output
