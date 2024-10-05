# X = axgrad.randn(shape=(4, 4))
# Y = axgrad.tensor(axgrad.randn(shape=(4, 1)))
# out = model(X)
# loss = ((Y - out) ** 2).sum()
# loss.backward()

# class SGD:
#   def __init__(self, parameters, lr=0.01):
#     self.parameters = parameters
#     self.lr = lr
  
#   def _mul(self, grad):
#     if isinstance(grad, list):
#       return [self._mul(g) for g in grad]
#     return grad * self.lr
  
#   def _sub(self, param, grad):
#     if isinstance(grad, list):
#       return [self._sub(p, g) for p, g in zip(param, grad)]
#     return param - grad
  
#   def step(self):
#     for param in self.parameters:
#       if param.grad is not None:
#         param.data = self._sub(param.data, self._mul(param.grad.data))

# optimizer = SGD(parameters=model.parameters(), lr=0.001)
# optimizer.step()

# out = model(X)
# new_loss = ((Y - out) ** 2).sum()
# print("Loss after one step:", new_loss)

import axgrad
import axgrad.nn as nn

class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.GELU()
    self.layer2 = nn.Linear(_hid, _out, bias)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.gelu(out)
    out = self.layer2(out)
    return out

model = MLP(4, 10, 1)

X = axgrad.tensor(axgrad.randn(shape=(4, 4)), requires_grad=True)  # Input tensor of shape (batch_size=4, features=4)
Y = axgrad.tensor(axgrad.randn(shape=(4, 1)), requires_grad=True)  # Target tensor of shape (batch_size=4, 1)

out = model.forward(X)
print(f"Output: {out}")
loss = ((Y - out) ** 2).sum()
print(f"Initial loss: {loss}")

loss.backward()

lr = 0.01
for param in model.parameters():
  if param.grad is not None:
    param.data = (param - (param.grad * lr).data).data

model.zero_grad()

out = model.forward(X)
print(f"Output: {out}")
loss = ((Y - out) ** 2).sum()
print(f"Initial loss: {loss}")

loss.backward()
lr = 0.01
for param in model.parameters():
  if param.grad is not None:
    param.data = (param - (param.grad * lr).data).data