a = [[1, 4, 6], [1, 5, 8]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

# from axgrad import tensor

# a, b = tensor(a), tensor(b)

# c = a + b
# print(c)
# print(c.prev)

# a = a.broadcast(b)
# print(a.sum())

import torch

a, b = torch.tensor(a, requires_grad=True, dtype=torch.float64), torch.tensor(b, requires_grad=True, dtype=torch.float64)
c = a + b
d = c ** 2
e = d.sum()

c.retain_grad()
d.retain_grad()
e.retain_grad()
e.backward()

print("e grad: ", e.grad)
print("d grad: ", d.grad)
print("c grad: ", c.grad)
print("b grad: ", b.grad)
print("a grad: ", a.grad)