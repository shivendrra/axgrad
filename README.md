# axgrad
![axon.jpg](https://github.com/shivendrra/axgrad/blob/main/axon.jpg)
A gradient engine along with it's own matrix operation library like PyTorch. It's supposed to be a fast & lightweight, C and Python based deep learning framework, which it's not, as of now.

## Overview
It contains a gradient engine exactly like Karpathy's [micrograd](https://github.com/karpathy/micrograd) and a deep learning framework like PyTorch or [TinyGrad](https://github.com/tinygrad/tinygrad) that has `Module`, `Linear` & `Sequence` layers similar to pytorch's `nn.Module`,  `nn.Linear` &`nn.Sequential`. It still work in progress though.

### Features
It has basic building blocks required to build a neural network: 
1. Basic tensor unary operations framework that could easily so matrix addition, multiplication, transpose, subtraction except for division(idk how to do that).
2. A gradient engine that could compute and update gradients, automatically, much like micrograd, but on a tensor level.
3. Optimizer & loss computation blocks to compute and optimize.
4. Basic blocks of network like Linear layer similar to `nn.Linear`, Sequence layer similar to `nn.Sequential` and others.
i'll be adding more things in future...

### Usage
This shows basic usage of `axgrad.engine` & few of the `axon`'s modules to preform tensor operations and build a sample neural network
#### Axgrad

```python
from axgrad import Value

a = Value(-4.0)
b = Value(2.0)

c = a + b
d = a * b + b**3
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**4
g = f / 125.25
g.backward()

print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```
#### Tensor operations

```python
from axgrad import tensor

# initializing 2-d matrices
x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])

z = x + y # addition
e = z*c.transpose() # mat_mul along with transpose
f = e.relu() # non-linearity
f.backward() # gradient pass

print(f) # axon.tensor(data=[41.2, 0, 20.0], [12.0, 25.0, 0], [0, 0, 0],
		     # grad=[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0])
```
#### Neural Network

```python
import axgrad.modules.nn as nn
x = tensor([[1.0, 2.0, 3.0, 8.0],[-0.6, 2.0, -3.0, 0.7],[-4.0, -2.0, 3.0, -5.0]])

linear = nn.Linear(4, 5, bias=True)
seq = nn.Sequence(4,2)

print(linear(x))
print(seq(x))
```
#### Matrix Functions

```python
import axgrad
zeros = axgrad.zeros([1, 4, 5])
ones = axgrad.ones([3, 4])

print(zeros) # 3-d matrix containing zeros
print(ones) # 2-d matrix containing ones
```

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. But it's still a work in progress.

## License
None!