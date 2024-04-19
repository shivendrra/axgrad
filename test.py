import axgrad
from axon import tensor
import axon.modules.nn as nn

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[9, 8, -7],
     [6, -5, 4],
     [3, -2, -1]]

C = [[9, -8, -7.8],
     [-1.5, -5, 4],
     [-0.4, 2, -1]]

x = tensor(A)
y = tensor(B)
c = tensor(C)

# Addition
z = x + y
e = z*c.transpose()
f = e.relu()
print(e)
print(f)
f.backward()
print("f", f)

from axgrad.engine import Value

a = Value(1)
b = Value(9)
c = b

z = a + b
e = z*c
print(z)
print(e)
e.backward()
print(e)
print('\n')

x = tensor([
  [1.0, 2.0, 3.0, 8.0],
  [-0.6, 2.0, -3.0, 0.7],
  [-4.0, -2.0, 3.0, -5.0],
])

linear = nn.Linear(4, 5, bias=True)
wei, bias = linear.weight, linear.bias
print(linear(x))

print('\n')
seq = nn.Sequence(_in=4, _out=2)
print(seq(x))