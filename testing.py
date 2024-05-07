import numpy as np

a = [[2, 4, 4], [1, 5, 6]]
b = [[2, 4], [1, 5], [-1, 5]]

A, B = np.array(a), np.array(b)
c = np.matmul(A, B)
print(c)
print(np.transpose(c))

from axon import tensor

x, y = tensor(a), tensor(b)
c = tensor.matmul_2d(x, y)
print(c)
print(c.transpose())