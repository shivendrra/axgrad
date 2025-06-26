import axgrad as ax

a = ax.Tensor([[2, 4, 5], [0, -1, -4]], dtype="int32")
b = ax.Tensor([[-4, 2, -1], [2, 0, -1]], dtype="int32")

c = a + b - 10

print(a, a.dtype, a.strides, a.shape, a.requires_grad, a.hooks)
print(b, b.dtype, b.strides, b.shape, b.requires_grad, b.hooks)
print(c, c.dtype, c.strides, c.shape, c.requires_grad, c.hooks)
print(c.abs(), c.dtype, c.strides, c.shape, c.requires_grad, c.hooks)
print(c.abs().sqrt(), c.dtype, c.strides, c.shape, c.requires_grad, c.hooks)