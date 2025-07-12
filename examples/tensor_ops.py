import axgrad
from axgrad import Tensor, randn

a = [[1, -4], [-2, 7]]
b = [[5, 1], [-9, -5]]

a, b = Tensor(a, dtype="float32", requires_grad=True), Tensor(b, dtype="float32", requires_grad=True)

''' element level '''
print("add: ", a + 2)
print("mul: ", a * 2)
print("sub: ", a - 2)
print("div: ", a / 2)
print("pow: ", a ** 2)

''' more operations '''
print("shape: ", a.shape)
print("transpose", a.transpose())
print("sum: ", a.sum())
print("matmul: ", a @ b)

a, b = randn(2, 4), randn(2, 4)
a.requires_grad = True
b.requires_grad = True

c = a + b
d = c * a
e = d.transpose()
f = e.sum(axis=-1)
g = f.sum()

g.backward()

print("a:\ndata:", a ,"grad: ", a.grad, "\n")
print("b:\ndata:", b ,"grad:", b.grad, "\n")
print("c:\ndata:", c ,"grad:", c.grad, "\n")
print("d:\ndata:", d ,"grad: ", d.grad, "\n")
print("e:\ndata:", e ,"grad: ", e.grad, "\n")
print("g:\ndata:", g ,"grad: ", g.grad, "\n")