
import torch
from app.residual import residual_add

torch.manual_seed(42)

x = torch.randn(2, 64, 16)
sublayer_output = torch.randn(2, 64, 16)

output = residual_add(x, sublayer_output)

print("========== RESIDUAL CONNECTION TEST ==========")
print("Input shape:", x.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Final output shape:", output.shape)

assert output.shape == x.shape
assert torch.allclose(output, x + sublayer_output)

print("\nResidual connection test: PASSED")

import torch
from app.residual import residual_add

torch.manual_seed(42)

x = torch.randn(2, 64, 16)
sublayer_output = torch.randn(2, 64, 16)

output = residual_add(x, sublayer_output)

print("========== RESIDUAL CONNECTION TEST ==========")
print("Input shape:", x.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Final output shape:", output.shape)

assert output.shape == x.shape
assert torch.allclose(output, x + sublayer_output)

print("\nResidual connection test: PASSED")
