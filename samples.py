# from axon import tensor, zeros
# import numpy as np

x = [[1, 4, 4], [1, 5, 6], [1, 5, 7]]
y = [[1, 4, 4], [1, 5, 6], [1, 5, 7], [1, 4, 5]]
# z = [[[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]], [[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]]]

# a, b, c = tensor(x), tensor(y), tensor(z)

# x = a * a * a
# print(x.sum())

# x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])
# print(x + y)

# x = tensor([[1, 4, 4], [1, 5, 6], [1, 5, 7]])
# z = tensor([[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]])

# print(tensor.matmul(x, z))

import torch

# a = torch.tensor(x)
# for i in range(10):
#   y = torch.stack((x))
# print(a.shape)
# print(y.shape)
# print(y)

# x = torch.randn(2, 3)
# print(x)

# y = torch.cat((x, x, x), 0)
# print(y)

# z = torch.cat((x, x, x), 1)
# print(z)

x = torch.randn(2, 3)
print(x)
print(x.size())

y = torch.stack((x, x))
print(y)
print(y.size())

z = torch.stack((x, x), dim=1)
print(z)
print(z.size())

a = torch.stack((x, x), dim=2)
print(a)
print(a.size())

b = torch.stack((x, x), dim=-1)
print(b)
print(b.size())