import axgrad
import axgrad.nn as nn

x = axgrad.tensor(axgrad.randn(shape=(2, 3)))
y = axgrad.tensor(axgrad.randn(shape=(2, 3)))

rms = nn.RMSNorm(dim=3)
out = rms(x)
e = out * y
g = e.sum()
g.backward()

print(g.grad)
print(out.grad)

for p in rms.parameters():
  print(p.grad)