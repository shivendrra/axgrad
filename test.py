import axgrad as ax

a = ax.Tensor([[4, 5], [-1, -4]], dtype="float32", requires_grad=True)
b = ax.Tensor([[2, -1], [0, -1]], dtype="float32", requires_grad=True)

c = a + b
d = c * 10
e = d.sum()

e.backward()

print("actual tensor:")
print(a)
print("grad: ")
print(a.grad)
print("actual tensor:")
print(b)
print("grad: ")
print(b.grad)
print("actual tensor:")
print(c)
print("grad: ")
print(c.grad)
print("actual tensor:")
print(d)
print("grad: ")
print(d.grad)
print("actual tensor:")
print(e)
print("grad: ")
print(e.grad)