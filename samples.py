from axon import tensor, nn

a = tensor([[-1, 2, -3, 4], [5, -6, 7, -8]])

print("Original tensor:")
print(a)
print(a.shape)

reshaped = a.reshape((4, 2))
print("Reshaped tensor:")
print(reshaped)
print(reshaped.shape)

a = tensor([[[1], [2], [3]], [[4], [5], [6]]])
print("Original tensor:")
print(a)
print(a.shape)

squeezed = a.squeeze()
print("Squeezed tensor:")
print(squeezed)
print(squeezed.shape)

a = tensor([[[1]], [[2]], [[3]]])
print("Original tensor:")
print(a)
print(a.shape)

squeezed = a.squeeze(dim=1)
print("Squeezed tensor along dimension 1:")
print(squeezed)
print(squeezed.shape)