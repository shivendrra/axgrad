import axgrad
import axgrad.nn as nn

ln = nn.LayerNorm(normalized_shape=3, eps=1e-5)
x = axgrad.tensor([[1.9, -2.1, 3.3], [-4.0, -5.0, 5.0]])
y = axgrad.tensor([[1.0, -0.1, -3.9], [4.9, -0.1, 5.1]])

output = ln(x)
e = output * y
g = e.sum()
g.backward()

print(g.grad)
print(output.grad)

for p in ln.parameters():
  print(p.grad)