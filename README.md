# axgrad
![axon.png](https://github.com/shivendrra/axgrad/blob/main/axon.png)
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

## Usage
This shows basic usage of `axgrad.engine` & few of the `axon`'s modules to preform tensor operations and build a sample neural network

anyway, prefer documentation for detailed usage guide:
1. [axon.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axonDoc.md): for using like numpy
2. [axgrad.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axgradDoc.md): for building neural network from axon library (incomplete for now)

## Creating a MLP

To create a multi-layer perceptron in ``axgrad``, you'll just need to follow the steps you followed in PyTorch. Very basic, initiallize two linear layers & a basic activation layer.

```python
import axgrad
import axgrad.nn as nn

class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.GELU()
    self.layer2 = nn.Linear(_hid, _out, bias)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.gelu(out)
    out = self.layer2(out)
    return out
```

refer to this [Example](https://github.com/shivendrra/axgrad/blob/main/examples/mlp.py) for detailed info on making mlp

btw, here's the outputs i got from my simple implementation, that ran till 500 iters:

![result](https://github.com/shivendrra/axgrad/blob/main/examples/mlp.png)

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. But it's still a work in progress.

## License
None!
