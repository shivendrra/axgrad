import axon

tensor1 = axon.tensor([[1, 2], [3, 4]])
tensor2 = axon.tensor([[5, 6], [7, 8]])

result = axon.stack((tensor1, tensor2, tensor1), axis=0)
print(result)
print(result.shape)

result = axon.concat((tensor1, tensor2), axis=1)
print(result)
print(result.shape)