import axon
from axon import tensor

x = tensor([[1, 2], [3, 4]])
y = tensor([[5, 6], [7, 8]])

c = x * y
c.backward()

print(y.grad)