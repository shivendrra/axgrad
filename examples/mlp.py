import axgrad.nn as nn
from axgrad.utils import randn, uniform
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

model = MLP(20, 10, 10)
optimizer = nn.SGD(parameters=model.parameters(), lr=1e-1)
x = randn(10, 20)
y = randn(10, 10)

epoch = 500
losses = []
steps = []

for n in range(1, epoch + 1):
  out = model.forward(x)
  loss = F.mae(out, y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if n % 50 == 0:
    losses.append(loss.tolist())
    steps.append(n)
    print(f"{n}th step, loss: {loss.tolist():.6f}")

import matplotlib.pyplot as plt

y_list = y.flatten().tolist()
out_list = out.flatten().tolist()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, losses, marker='o', linestyle='-', color='b')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.xticks(steps)

plt.subplot(1, 2, 2)
plt.scatter(range(y.size), y_list, color='b', label='Actual Values')  # Actual values in blue
plt.scatter(range(out.size), out_list, color='r', label='Predicted Values')  # Predicted values in red
plt.title('Predictions vs Targets')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()