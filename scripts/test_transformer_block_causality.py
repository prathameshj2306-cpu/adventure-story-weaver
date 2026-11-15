
import torch
from app.transformer_block import TransformerBlock

torch.manual_seed(42)

transformer = TransformerBlock(
    embedding_dim=16,
    num_heads=4,
    block_size=8
)

transformer.eval()

# Original input
x1 = torch.randn(1, 5, 16)

# Modify only the final token
x2 = x1.clone()
x2[:, 4, :] += 100

with torch.no_grad():
    output1 = transformer(x1)
    output2 = transformer(x2)

# Earlier positions must remain unchanged
same_outputs = torch.allclose(
    output1[:, :4, :],
    output2[:, :4, :],
    atol=1e-6
)

print("========== TRANSFORMER CAUSALITY TEST ==========")
print("Original output shape:", output1.shape)
print("Modified output shape:", output2.shape)
print("Earlier positions unchanged:", same_outputs)

assert same_outputs, "Transformer causality test failed!"

print("\nTransformer causality test: PASSED")
