import axon
from axon import tensor

a = [[1, -4],
     [-2, 7]]

b = [[5, 1],
     [-9, -5]]

a, b = tensor(a, requires_grad=True), tensor(b, requires_grad=True)

''' element level '''
print("add: ", a + [2, 0])
print("mul: ", a * [2, 0])
print("sub: ", a - [2, 0])
print("div: ", a / [2, 0])
print("pow: ", a ** 2)

''' more operations '''
print("shape: ", a.shape)
print("transpose", a.T)
print("sum: ", a.sum())
print("matmul: ", axon.matmul(a, b))