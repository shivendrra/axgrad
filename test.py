a = [[2, 3, 5], [6, 4, 9]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

from axon import tensor

a, b = tensor(a), tensor(b)

c = a + a
d = c.tanh()
# for prev in c.prev:
#   print(prev.grad)
d.backward()
print(d.prev)
print("d: ", d)
print("d.grad: ", d.grad)
print("c.grad: ", c.grad)
print("a.grad: ", a.grad)
print("b.grad: ", b.grad)

# import torch
# a, b = torch.tensor(a, requires_grad=True, dtype=torch.float64), torch.tensor(b, requires_grad=True, dtype=torch.float64)
# c = a + b
# d = c.sum()

# c.retain_grad()
# d.retain_grad()

# d.backward()

# print(d)
# print("d grad: ", d.grad)
# print("c grad: ", c.grad)
# print("b grad: ", b.grad)
# print("a grad: ", a.grad)