import axgrad.nn as nn
from axgrad.utils import randn

class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.Tanh()
    self.layer2 = nn.Linear(_hid, _out, bias)

  def forward(self, x):
    out = self.layer1(x)
    out = self.gelu(out)
    out = self.layer2(out)
    return out

model = MLP(20, 30, 10)
mse = nn.MSELoss()

x = randn(10, 20)
y = randn(10, 10)

yp = model(x)
loss = mse(yp, y)
loss.backward()

print("input grads: ", x.grad)
print("input grads: ", y.grad)

print("w1 grad: ", model.layer1.weight.grad)
print("w2 grad: ", model.layer2.weight.grad)