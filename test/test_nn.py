import torch
from axgrad import tensor

input_data = [[1.0, 2.0], [3.0, 4.0]]
weight_data = [[0.5, -0.5], [1.5, -1.0]]

torch_input = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)
torch_weight = torch.tensor(weight_data, requires_grad=True, dtype=torch.float32)

ax_input = tensor(input_data, requires_grad=True, dtype='float32')
ax_weight = tensor(weight_data, requires_grad=True, dtype='float32')

# PyTorch: Perform a simple matrix multiplication and then a sum
torch_output = torch.matmul(torch_input, torch_weight).sum()
# axgrad: Perform the same operation using axgrad tensors
ax_output = (ax_input @ ax_weight).sum()
torch_output.backward()
ax_output.backward()
print("Torch Output:", torch_output.item())
print("AxGrad Output:", ax_output.data)
print("\nTorch Input Grad:\n", torch_input.grad)
print("AxGrad Input Grad:\n", ax_input.grad.data)
print("\nTorch Weight Grad:\n", torch_weight.grad)
print("AxGrad Weight Grad:\n", ax_weight.grad.data)

def check_gradients(tensor1, tensor2, tol=1e-6):
  if isinstance(tensor1, list):
    return all(check_gradients(t1, t2, tol) for t1, t2 in zip(tensor1, tensor2))
  return abs(tensor1 - tensor2) < tol

print("\nGradients match:", check_gradients(ax_input.grad.data, torch_input.grad.tolist()))
print("Weights Gradients match:", check_gradients(ax_weight.grad.data, torch_weight.grad.tolist()))