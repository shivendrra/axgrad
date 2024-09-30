import axgrad
import axgrad.nn as nn

class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.ReLU()
    self.layer2 = nn.Linear(_hid, _out, bias)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.gelu(out)
    out = self.layer2(out)
    return out

model = MLP(4, 10, 1)
X = axgrad.randn(shape=(4, 4))
Y = axgrad.tensor(axgrad.randn(shape=(4, 1)))
out = model(X)

print(model)
loss = ((Y - X) ** 2).sum()
print(loss)
loss.backward()
loss.zero_grad()