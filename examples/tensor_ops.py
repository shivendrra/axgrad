import axon
from axon import tensor

a = [[1, -4],
     [-2, 7]]

b = [[5, 1],
     [-9, -5]]

a, b = tensor(a), tensor(b)

''' element level '''
print("add: ", a + b)
print("sub: ", a - b)
print("mul: ", a * b)
print("div: ", a / b)
print("pow: ", a ** 2)

''' more operations '''
print("shape: ", a.shape)
print("transpose", a.transpose())
print("sum: ", a.sum())
print("matmul: ", axon.matmul(a, b))