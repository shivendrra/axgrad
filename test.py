# Assuming your tensor class accepts a list of lists (or nested lists) as data
import random
from axgrad import tensor, randn

# Example tensor instances
tensor_2d = tensor(randn(shape=[13, 10]))  # 3x3 tensor
tensor_3d = tensor(randn(shape=[2, 10, 12]))  # 2x2x2 tensor

# Test some operations
print("Original tensor 2D:", tensor_2d)
print("ReLU operation result:", tensor_2d.relu())
print("Power operation result (raise to 2):", (tensor_2d ** 2))
print("Division result:", (tensor_2d / tensor([5])))

# Similar tests for higher dimensional tensors
print("Original tensor 3D:", tensor_3d)
print("ReLU on 3D tensor:", tensor_3d.relu())