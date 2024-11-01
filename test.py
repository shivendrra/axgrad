import axgrad
import axgrad.nn as nn

x = axgrad.tensor([[0.0, -2.0, -3.0], [-1.0, -5.2, 1.0]])
y = axgrad.tensor([[-1.1, -0.1, -1.9], [4.9, -10.1, 5.1]])

rms = nn.RMSNorm(dim=3)
out = rms(x)
e = out * y
g = e.sum()
g.backward()

print(g.grad)
print(out.grad)

for p in rms.parameters():
  print(p.grad)

# ln = nn.LayerNorm(normalized_shape=3, eps=1e-5)
# x = axgrad.tensor([[0.0, -2.0, -3.0], [-1.0, -5.2, 1.0]])
# y = axgrad.tensor([[-1.1, -0.1, -1.9], [4.9, -10.1, 5.1]])

# output = ln(x)
# e = output * y
# g = e.sum()
# g.backward()

# print(g.grad)
# print(output.grad)

# for p in ln.parameters():
#   print(p.grad)

# x = axgrad.tensor(axgrad.randn(shape=(2, 3)))
# y = axgrad.tensor(axgrad.randn(shape=(2, 3)))

# bn = nn.BatchNorm(num_features=3, eps=1e-5, momentum=0.1)

# x = axgrad.tensor([[0.0, -2.0, -3.0], [-1.0, -5.2, 1.0]])
# y = axgrad.tensor([[-1.1, -0.1, -1.9], [4.9, -10.1, 5.1]])

# output = bn(x)
# e = output * y
# g = e.sum()
# g.backward()

# print(g.grad)
# print(output.grad)

# for p in bn.parameters():
#   print(p.grad)