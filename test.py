a = [[2, 3, 5], [6, 4, 9]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]
c = [[1, 2, 3], [4, 5, 6]]
import axon
from axon import tensor

a, b = tensor(a), tensor(b)
c = tensor(c)
c = a.copy()
a[0][1] = 5
print(c)
print(a)

# arr1 = tensor([[1, 2], [3, 4]])
# arr2 = tensor([[5, 6], [7, 8]])

# out = axon.stack(arr1, arr2, arr1, arr2, axis=2)
# print(out)

# import numpy as np

# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5, 6], [7, 8]])

# out = np.stack((arr1, arr2, arr1, arr2), axis=2)
# print(out)