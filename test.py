import axgrad as ax

a = ax.Tensor([[4, 5], [-1, -4]], dtype="float32")
b = ax.Tensor([[2, -1], [0, -1]], dtype="float32")

c = (a @ b) ** 2
d = c.transpose()

print(a, a.dtype, a.strides, a.shape)
print(b, b.dtype, b.strides, b.shape)
print(c, c.dtype, c.strides, c.shape)
print(d, d.dtype, d.strides, d.shape)