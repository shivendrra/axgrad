# a = [[2, 3, 5], [6, 4, 9]]
# x = [[1, 4], [5, 8], [9, 0]]
# b = [[[1, 4, 6], [1, 5, 8]], [[1, 4, 6], [1, 5, 8]], [[1, 4, 6], [1, 5, 8]], [[1, 4, 6], [1, 5, 8]]]

# import axon
# from axon import tensor

# a, b, x = tensor(a), tensor(b), tensor(x)
# c = axon.matmul(a, x)
# # c.backward()
# d = c.sum()
# d.backward()

# print(d)
# print("d grad:", d.grad)
# print(c)
# print("c grad:", c.grad)
# print(a)
# print("a grad:", a.grad)
# print(x)
# print("x grad:", x.grad)

a = [[2, 3, 5], [6, 4, 9]]
from axon import tensor
a = tensor(a)
print(a + a)
print(a - a)
print(a * a)
print(a / a)