a = [[1, 4, 6], [1, 5, 8]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

from axon import tensor

a, b = tensor(a), tensor(b)

c = a + b
c.backward()
print(c)
print(c.grad)
print(a.grad)
print(b.grad)

a = a.broadcast(b)
print(a)
print(a.sum())