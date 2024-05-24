from axon import tensor, nn

a = tensor([[-1, 2, -3, 4], [5, -6, 7, -8]])
b = tensor([[1, 4, 6, 6], [4, 7, -2, -3]])

c = a*b
print(c)