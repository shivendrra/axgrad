import axon
from axon import tensor

a = [[1, -4],
     [-2, 7]]

b = [[5, 1],
     [-9, -5]]

a, b = tensor(a, requires_grad=True, dtype=axon.float32), tensor(b, requires_grad=True, dtype=axon.float32)

c = a @ b
d = c ** 2
e = d.sum()

e.backward()

print(a, a.grad)
print(b, b.grad)
print(c, c.grad)
print(d, d.grad)
print(e, e.grad)

# import axgrad
# import axgrad.nn as nn


# class MLP(nn.Module):
#   def __init__(self, _in, _hidden, _out, bias=False) -> None:
#     super().__init__()
#     self.layer1 = nn.Linear(_in, _hidden, bias)
#     self.gelu = nn.GELU()
#     self.layer2 = nn.Linear(_hidden, _out, bias)
  
#   def forward(self, x):
#     out = self.layer1(x)
#     out = self.gelu(out)
#     out = self.layer2(out)
#     return out

# model = MLP(_in=4, _hidden=20, _out=1, bias=False)

# xs = [
#   [1, 3, 4, 5],
#   [0, -1, 0, -1],
#   [1, 0.3, -4, 0.9],
#   [0, 9, -2, 1]
# ]

# out = model(xs)
# print(out)
# print(out.shape)

# ys = axgrad.tensor([1, 0, -1, 0], requires_grad=True, dtype='float32')
# ys = ys.reshape((-1, 1))

# def loss(target, outputs):
#   loss = outputs - target
#   loss = (-loss ** 2).sum() / 2
#   return loss

# loss_out = loss(ys, out)
# print(loss_out)