import axgrad
import axgrad.nn as nn

# Example tensor
t = axgrad.tensor([[[3, -4, 5], [1, 5, 6], [-1, 5, 0]], [[3, -4, 5], [1, 5, 6], [-1, 5, 0]]])

# Create a norm instance
l2_norm = axgrad.norm(t, p=2)
# Access the computed norm value
print(f"L2 Norm: {l2_norm}")

# Create an L1 norm instance
l1_norm = axgrad.norm(t, p=1)
print(f"L1 Norm: {l1_norm}")

# Create an L3 norm instance
l3_norm = axgrad.norm(t, p=3)
print(f"L3 Norm: {l3_norm}")

import axgrad
import axgrad.nn as nn
from axgrad.nn import functional as F

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

optimizer = nn.SGD(parameters=model.parameters(), lr=0.001)
epoch = 10
lr = 0.1

for n in range(epoch):
  out = model.forward(X)
  loss = F.mae(out, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print(f"{n+1}th step, loss: {loss.data[0]:.6f}")