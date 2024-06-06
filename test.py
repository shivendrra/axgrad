a = [[1, 4, 6], [1, 5, 8]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

from axgrad import tensor

tensor1 = tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

out = tensor1 * tensor2
print(out)
print(out ** 2)

print(tensor1.flatten())
print(tensor2.flatten())
print(tensor1.flatten(0, -1))
print(tensor2.flatten(1, 1))