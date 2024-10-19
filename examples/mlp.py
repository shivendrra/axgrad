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

optimizer = nn.Optim.SGD(parameters=model.parameters(), lr=0.001)
epoch = 100
lr = 0.1
for n in range(epoch):
  out = model.forward(X)
  loss = F.mse(out, Y)
  model.zero_grad()
  loss.backward()
  optimizer.step()
  print(f"{n+1}th step, loss: {loss.data[0]:.6f}")