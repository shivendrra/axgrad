from axon import tensor

x = tensor([[1, 2], [3, 4]])
y = tensor([[5, 6], [7, 8]])

res = x * y
print(res.sigmoid())