# axgrad
![axon.jpg](https://github.com/shivendrra/axgrad/blob/main/axon.jpg)
My attempt to make something like [TinyGrad](https://github.com/tinygrad/tinygrad) or [PyTorch](https://github.com/pytorch/pytorch)
A framework like PyTorch & MicroGrad written fully in python(i will add the c & cpp components for faster implementation though).
It's supposed to be a good and lightweight C and Python based deep learning framework, which it's not, as of now(still building).

## Overview
It contains a framework similar to Numpy which allows to do basic matrix operations like element-wise add/mul + matrix multiplication + broadcasting. Also building pytorch like auto-differentiation engine: axgrad (work in progress!)

## Features
It has basic building blocks required to build a neural network:
1. Basic tensor ops framework that could easily so matrix add/mul (element-wise), transpose, broadcasting, matmul, etc.
2. A gradient engine that could compute and update gradients, automatically, much like micrograd, but on a tensor level ~ autograd like (work in progress!).
3. Optimizer & loss computation blocks to compute and optimize (work in progress!).
i'll be adding more things in future...

### Progress

| Development                  | Status      | Feature                                                                |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| Base Class                   | in progress | <ul><li>[x] Axon</li><li>[ ] Auto-diff (axgrad)</li><li>[x] Broadcasting</li></ul>                 |
| Versions                         | in progress | <ul><li>[x] python version</li><li>[ ] c version</li><li>[ ] final version</li></ul>    |
| Loss                         | in progress | <ul><li>[ ] MSE</li><li>[ ] Cross Entropy</li><li>[ ] MAE</li></ul>    |
| Language Transformer | in progress    | <ul><li>[x] Matmul</li><li>[ ] Embeddings</li></ul> |
| Convolutional Neural Network | in progress    | <ul><li>[ ] Conv2d</li><li>[ ] MaxPool2d</li><li>[ ] Dropout</li></ul> |
| Neural Network Components                  | in progress | <ul><li>[x] Module</li><li>[ ] Sequential</li><li>[ ] ModuleList</li><li>[x] Linear</li></ul>

## Usage
This shows basic usage of `axgrad.engine` & few of the `axon`'s modules to preform tensor operations and build a sample neural network

anyway, prefer documentation for detailed usage guide:
1. [axon.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axonDoc.md): for using like numpy
2. [axgrad.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axgradDoc.md): for building neural network from axon library (incomplete for now)

### Axon
It's similar to NumPy, for now. I'm trying to add more functions/methods to make it equivalent to PyTorch, at-least to some extent. It supports a few basic functions for now, like element wise ops: add/sub/mul/div/pow; tensor ops: matmul, flatten, 2d-convolution, transpose, shape, etc.

#### Tensor operations
```python
from axgrad import tensor

# initializing 2-d matrices
x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])

z = x + y # addition
print(z) # output: axon.tensor([10, 10, -4], [10, 0, 10], [10, 6, 8])
```

#### Matrix Functions
```python
import axgrad

zeros = agrad.zeros([1, 4, 5]) # 3-d matrix containg zeros, (float)
ones = axgrad.ones([3, 4]) # 2-d matrix containing ones, (int)

x = tensor([[1, 4, 4], [1, 5, 6], [1, 5, 7]])
z = tensor([[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]])
print(axgrad.matmul(x, z)) # out: axon.tensor([33, 45, 49, 37], [45, 62, 68, 51], [49, 68, 75, 56])
```

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. But it's still a work in progress.
## License
None!