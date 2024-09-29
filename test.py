from axgrad import tensor, randn

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