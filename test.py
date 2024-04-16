# from axgrad.modules.matrices import ones
# from axgrad.arrays import array as arr

# zero_list = ones((1, 3), dtype=float)
# one_list = ones((1, 3), dtype=float)

# zero_list = arr(zero_list)
# one_list = arr(one_list)

# import axgrad.nn as nn

# linear_layer = nn.Linear(4, 5)
# input_data = [1.0, 2.0, 3.0, 8.0]
# output = linear_layer(input_data)

# print(output)


from axgrad import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

from axgrad import nn_mods

n = nn_mods.Neuron(2)
# x = [Value(1.0), Value(-2.0)]
x = [a, b]
y = n(x)
print(y)
y.backward()
print(y)