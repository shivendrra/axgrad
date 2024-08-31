# import axgrad
# from axgrad import tensor

# a = [[1, -4],
#      [-2, 7]]

# b = [[5, 1],
#      [-9, -5]]

# a, b = tensor(a, requires_grad=True, dtype=axgrad.float32), tensor(b, requires_grad=True, dtype=axgrad.float32)

# c = a * [[1, 4]]
# d = c ** 2

# d.backward()

# print(a, a.grad)
# print(b, b.grad)
# print(c, c.grad)
# print(d, d.grad)