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

## Usage
This shows basic usage of `axgrad.engine` & few of the `axon`'s modules to preform tensor operations and build a sample neural network

anyway, prefer documentation for detailed usage guide:
1. [axon.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axonDoc.md): for using like numpy
2. [axgrad.doc](https://github.com/shivendrra/axgrad/blob/main/docs/axgradDoc.md): for building neural network from axon library (incomplete for now)

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. But it's still a work in progress.

## License
None!