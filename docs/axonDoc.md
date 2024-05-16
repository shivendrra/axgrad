# AxGrad Documentation

Guide to create a small deep learning framework using the `Value` class to handle automatic differentiation and custom neural network layers.

## Neural Network Framework Documentation

### Class: `Value`

Represents a scalar value and supports automatic differentiation.

- **Attributes**:
  - `data` (float): The scalar value.
  - `grad` (float): The gradient of the value (initialized to 0.0).
  - `_backward` (function): The function to compute the gradient for this value.
  - `_prev` (set): The set of preceding `Value` objects (for backpropagation).
  - `_op` (str): The operation that produced this value (for debugging).

- **Methods**:
  - `__add__(self, other)`: Adds two `Value` objects.
  - `__mul__(self, other)`: Multiplies two `Value` objects.
  - `__pow__(self, other)`: Raises `self` to the power of `other`.
  - `tanh(self)`: Applies the hyperbolic tangent function.
  - `sigmoid(self)`: Applies the sigmoid function.
  - `relu(self)`: Applies the ReLU function.
  - `backward(self)`: Computes the gradient of all preceding `Value` objects.

**Example**:
```python
a = Value(2.0)
b = Value(3.0)
c = a + b
c.backward()
print(a.grad)  # Output: 1.0
print(b.grad)  # Output: 1.0
```

### Class: `Module`

Base class for all neural network modules.

- **Methods**:
  - `zero_grad(self)`: Resets the gradients of all parameters to zero.
  - `parameters(self)`: Returns a list of parameters (to be overridden by subclasses).
  - `children(self)`: Yields all child modules.
  - `train(self, mode)`: Sets the module to training mode.
  - `eval(self)`: Sets the module to evaluation mode.

### Class: `Linear`

Represents a single linear neuron.

- **Attributes**:
  - `wei` (list of `Value`): The weights of the neuron.
  - `b` (`Value`): The bias of the neuron.
  - `nonlin` (bool): Whether to apply a ReLU non-linearity.

- **Methods**:
  - `__call__(self, x)`: Applies the neuron to an input.
  - `parameters(self)`: Returns the weights and bias as parameters.
  
**Example**:
```python
neuron = Linear(3, True)
output = neuron([Value(1.0), Value(2.0), Value(3.0)])
print(output)
```

### Class: `Layer`

Represents a layer of linear neurons.

- **Attributes**:
  - `neurons` (list of `Linear`): The neurons in the layer.

- **Methods**:
  - `__call__(self, x)`: Applies the layer to an input.
  - `parameters(self)`: Returns the parameters of all neurons.
  
**Example**:
```python
layer = Layer(3, 2)
output = layer([Value(1.0), Value(2.0), Value(3.0)])
print(output)
```

### Class: `MLP`

Represents a multi-layer perceptron (MLP).

- **Attributes**:
  - `layers` (list of `Layer`): The layers in the MLP.

- **Methods**:
  - `__call__(self, x)`: Applies the MLP to an input.
  - `parameters(self)`: Returns the parameters of all layers.
  
**Example**:
```python
mlp = MLP(3, [4, 2])
output = mlp([Value(1.0), Value(2.0), Value(3.0)])
print(output)
```

### Class: `Linear2d`

Represents a 2D linear layer.

- **Attributes**:
  - `wei` (list of list of `Value`): The weights of the layer.
  - `b` (list of `Value` or None): The biases of the layer.

- **Methods**:
  - `__call__(self, x)`: Applies the layer to an input.
  - `parameters(self)`: Returns the weights and biases as parameters.
  
**Example**:
```python
linear2d = Linear2d(3, 2, bias=True)
output = linear2d([[Value(1.0), Value(2.0), Value(3.0)]])
print(output)
```

### Class: `FeedForward`

Represents a simple feedforward neural network.

- **Attributes**:
  - `layer1` (`Linear2d`): The first linear layer.
  - `relu` (`ReLU`): The ReLU activation function.
  - `layer2` (`Linear2d`): The second linear layer.

- **Methods**:
  - `__call__(self, x)`: Applies the feedforward network to an input.
  - `parameters(self)`: Returns the parameters of all layers.
  
**Example**:
```python
ff = FeedForward(3, 2)
output = ff([[Value(1.0), Value(2.0), Value(3.0)]])
print(output)
```

### Import and Usage

To use these classes and functions, you can either import them individually:

```python
from axon import Value, Linear, Layer, MLP, Linear2d, FeedForward
```

Or import the module and use the classes with the module prefix:

```python
import axon
a = axon.Value(2.0)
neuron = axon.Linear(3, True)
layer = axon.Layer(3, 2)
mlp = axon.MLP(3, [4, 2])
linear2d = axon.Linear2d(3, 2, bias=True)
ff = axon.FeedForward(3, 2)
```

These classes and functions provide the foundation for building and training simple neural networks with automatic differentiation support.

## Example Neural Network

```python
from axon.axgrad import MLP, Optim, Loss

n = MLP(3, [4, 4, 1])

xs = [
  [-2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, -1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [-1.0, 1.0, -1.0, 1.0]
l_arr = []

optimizer = Optim.sgd(n.parameters(), learn_rate=0.05)
loss_f = Loss()

for k in range(10):
  ypred = [n(x) for x in xs]
  loss = loss_f.mse_loss(ys, ypred)
  l_arr.append(loss.data)
  n.zero_grad()
  loss.backward()
  optimizer.step()

print(ypred)
print(loss.data)

import matplotlib.pyplot as plt

plt.plot(range(10), l_arr)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
```

### trained output:
![trained example](https://github.com/shivendrra/axgrad/tree/main/docs/example_net.jpg)