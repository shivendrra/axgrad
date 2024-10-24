# import axgrad
# import axgrad.nn as nn

# # t1 = axgrad.tensor([1, 2, 3], requires_grad=True)
# # t2 = axgrad.tensor([4, 5, 6], requires_grad=True)

# # stacked = axgrad.concat([t1, t2], axis=0)
# # d = stacked.tanh()
# # out = d.sum()
# # print(out)
# # print(stacked)

# # out.backward()
# # print(out.grad)
# # print(d.grad)
# # print(stacked.grad)
# # print(t1.grad)  # Should show gradients for t1
# # print(t2.grad)  # Should show gradients for t2

# input_tensor = axgrad.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0)
# output = conv(input_tensor)
# d = output.sum()

# d.backward()
# print(d.grad)
# print(output.grad)
# for p in conv.parameters():
#   print(p, p.grad)

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