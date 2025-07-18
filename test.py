# import axgrad.nn as nn
# from axgrad.utils import randn

# x = randn(10, 20)
# y = randn(10, 10)
# linear = nn.Linear(20, 10)
# yp = linear(x)
# loss = ((y - yp) ** 2 / x.size).sum()
# loss.backward()

# print(x.grad)
# print(y.grad)
# print(linear.weight.grad)

import axgrad as ax

a, b = ax.randn(2, 5), ax.randn(2, 5)

print(a == b)
print(a != b)
print(a > b)
print(a < b)
print(a >= b)
print(a <= b)

print(a == 2.0)
print(a != 2.0)
print(a > 2.0)
print(a < 2.0)
print(a >= 2.0)
print(a <= 2.0)