# axgrad
![axon.jpg](https://github.com/shivendrra/axgrad/blob/main/axon.jpg)
My attempt to make something like [TinyGrad](https://github.com/tinygrad/tinygrad) or [PyTorch](https://github.com/pytorch/pytorch)
A framework like PyTorch & MicroGrad written fully in python(i will add the c & cpp components for faster implementation though).
It's supposed to be a good and lightweight C and Python based deep learning framework, which it's not, as of now(still building).

## Overview
It contains a framework similar to Numpy which allows to do basic matrix operations like element-wise add/mul + matrix multiplication + broadcasting and many more things in near future.

## Features
It has basic building blocks required to build a neural network:
1. Basic tensor ops framework that could easily so matrix add/mul (element-wise), transpose, broadcasting, matmul, etc.
2. A gradient engine that could compute and update gradients, automatically, much like micrograd, but on a tensor level ~ autograd like.
3. Optimizer & loss computation blocks to compute and optimize.
i'll be adding more things in future...

### Progress

| Development                  | Status      | Feature                                                                |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| Base Class                   | in progress | <ul><li>[x] tensor.py</li><li>[x] Axgrad</li><li>[x] Broadcasting</li></ul>                 |
| Versions                         | in progress | <ul><li>[ ] cpp version</li><li>[ ] c version</li><li>[ ] final version</li></ul>    |
| Loss                         | in progress | <ul><li>[ ] MSE</li><li>[ ] Cross Entropy</li><li>[ ] MAE</li></ul>    |
| Language Transformer | in progress    | <ul><li>[x] Matmul</li><li>[ ] Embeddings</li></ul> |
| Convolutional Neural Network | in progress    | <ul><li>[ ] Conv2d</li><li>[ ] MaxPool2d</li><li>[ ] Dropout</li></ul> |
| Neural Network Components                  | in progress | <ul><li>[ ] Module</li><li>[ ] Sequential</li><li>[ ] ModuleList</li><li>[x] Linear</li></ul>

## Usage
This shows basic usage of `axgrad.engine` & few of the `axon`'s modules to preform tensor operations and build a sample neural network

anyway, prefer documentation for detailed usage guide:
1. [axon.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axonDoc.md)
2. [axgrad.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axgradDoc.md)

### Axgrad
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

#### Neural Network
```python
import axgrad.nn as nn
x = tensor([[1.0, 2.0, 3.0, 8.0],[-0.6, 2.0, -3.0, 0.7],[-4.0, -2.0, 3.0, -5.0]])

linear = nn.Linear(4, 5, bias=True)
seq = nn.Sequence(4,2)

print(linear(x))
print(seq(x))
```

### Axon
It's similar to NumPy, for now. I'm trying to add more functions/methods to make it equivalent to PyTorch, at-least to some extent. It supports a few basic functions for now, like element wise ops: add/sub/mul/div/pow; tensor ops: matmul, flatten, 2d-convolution, transpose, shape, etc.

#### Tensor operations
```python
from axon import tensor

# initializing 2-d matrices
x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])

z = x + y # addition
print(z) # output: axon.tensor([10, 10, -4], [10, 0, 10], [10, 6, 8])
```

#### Matrix Functions
```python
import axon

zeros = axon.zeros([1, 4, 5], dtype=float) # 3-d matrix containg zeros, (float)
ones = axgon.ones([3, 4], dtype=int) # 2-d matrix containing ones, (int)

x = tensor([[1, 4, 4], [1, 5, 6], [1, 5, 7]])
z = tensor([[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]])
print(tensor.matmul(x, z)) # out: axon.tensor([33, 45, 49, 37], [45, 62, 68, 51], [49, 68, 75, 56])
```

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. But it's still a work in progress.
## License
None!