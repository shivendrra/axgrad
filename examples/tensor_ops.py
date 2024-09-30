import axgrad
from axgrad import tensor, randn

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
print("matmul: ", axgrad.matmul(a, b))

a, b = tensor(randn(shape=(2, 4))), tensor(randn(shape=(2, 4)))

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