# import axgrad

# array1_4d = axgrad.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
# array2_4d = axgrad.tensor([[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]])
# result_4d = array2_4d - array1_4d

# print(result_4d)
# result_4d[1] = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
# print(result_4d)

# import axgrad.nn as nn
# linear_layer = nn.Linear(4, 5)
# input_data = [1.0, 2.0, 3.0, 8.0]
# output = linear_layer(input_data)

# print(output)

from axgrad import tensor
import axgrad.modules.nn as nn

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[9, 8, -7],
     [6, -5, 4],
     [3, -2, -1]]

C = [[9, 8, -7],
     [6, -5, 4],
     [3, -2, -1]]

x = tensor(A)
y = tensor(B)
c = tensor(C)

# Addition
z = x + y
e = z*c.transpose()
f = tensor.relu(e)
print(e)
print(f)
f.backward()
print(f)

print('\n')

x = [
  [1.0, 2.0, 3.0, 8.0],
  [-0.6, 2.0, -3.0, 0.7],
  [-4.0, -2.0, 3.0, -5.0],
]

linear = nn.Linear(4, 5, bias=True)
wei, bias = linear.weight, linear.bias
print(linear(x))

print('\n')
seq = nn.Sequence(_in=4, _out=2)
print(seq(x))