# axgrad
![axon.jpg](https://github.com/shivendrra/axgrad/blob/main/axon.jpg)
A gradient engine along with it's own matrix operation library like PyTorch. It's supposed to be a good and lightweight C and Python based deep learning framework, which it's not, as of now.

## Overview
It contains two gradient engines, one exactly like Karpathy's [micrograd](https://github.com/karpathy/micrograd) and other is something like PyTorch or [TinyGrad](https://github.com/tinygrad/tinygrad) along with a basic neural net framework that has `Module`, `Linear` & `Sequence` layers similar to pytorch's `nn.Module`,  `nn.Linear` &`nn.Sequential`.

### Features
It has basic building blocks required to build a neural network: 
1. Basic tensor operation framework that could easily so matrix addition, multiplication, transpose, subtraction except for division(idk how to do that).
2. A gradient engine that could compute and update gradients, automatically, much like micrograd, but on a tensor level.
3. Optimizer & loss computation blocks to compute and optimize.
4. Basic blocks of network like Linear layer similar to `nn.Linear`, Sequence layer similar to `nn.Sequential` and others.
add more things in future...

## Usage

### Tensor operations

```python
from axgrad import tensor

# initializing 2-d matrices
x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])

z = x + y # addition
e = z*c.transpose() # mat_mul along with transpose
f = e.relu() # non-linearity
f.backward() # gradient pass

print(f) # axon.tensor(data=[41.2, 0, 20.0], [12.0, 25.0, 0], [0, 0, 0],  grad=[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0])
```
### Neural Network

```python
import axgrad.modules.nn as nn
x = tensor([[1.0, 2.0, 3.0, 8.0],[-0.6, 2.0, -3.0, 0.7],[-4.0, -2.0, 3.0, -5.0]])

linear = nn.Linear(4, 5, bias=True)
seq = nn.Sequence(4,2)

print(linear(x))
print(seq(x))
```
### Matrix Functions

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