# Tensor Library Documentation

A Python tensor library with automatic differentiation capabilities, built on top of a C backend for performance.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
- [Tensor Creation](#tensor-creation)
- [Basic Operations](#basic-operations)
- [Mathematical Functions](#mathematical-functions)
- [Shape Operations](#shape-operations)
- [Automatic Differentiation](#automatic-differentiation)
- [Data Types](#data-types)
- [Examples](#examples)

## Installation

```python
# Import the main classes
from tensor import Tensor
from utils import zeros, ones, randn, randint, uniform, fill, zeros_like, ones_like
```

## Quick Start

```python
# Create tensors
x = Tensor([1, 2, 3, 4], dtype="float32")
y = Tensor([[1, 2], [3, 4]], dtype="float32")

# Basic operations
z = x + 5
w = x * y  # Element-wise multiplication
result = x @ y  # Matrix multiplication

# With gradients
x = Tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # Gradient of y with respect to x
```

## Core Classes

### Tensor

The main tensor class that supports n-dimensional arrays with automatic differentiation.

#### Constructor

```python
Tensor(data, dtype="float32", requires_grad=False)
```

**Parameters:**
- `data`: Input data (list, nested list, int, float, or another Tensor)
- `dtype`: Data type (default: "float32")
- `requires_grad`: Whether to track gradients (default: False)

**Properties:**
- `shape`: Tuple representing tensor dimensions
- `size`: Total number of elements
- `ndim`: Number of dimensions
- `dtype`: Data type
- `requires_grad`: Whether gradients are tracked
- `grad`: Accumulated gradients (None if requires_grad=False)

## Tensor Creation

### From Data

```python
# From list
x = Tensor([1, 2, 3])

# From nested list (2D)
y = Tensor([[1, 2], [3, 4]])

# From scalar
z = Tensor(5.0)
```

### Utility Functions

```python
# Create tensors filled with zeros
zeros_tensor = zeros(3, 4)  # 3x4 tensor of zeros
zeros_like_tensor = zeros_like(existing_tensor)

# Create tensors filled with ones
ones_tensor = ones(2, 3, 4)  # 2x3x4 tensor of ones
ones_like_tensor = ones_like(existing_tensor)

# Random tensors
random_normal = randn(3, 3)  # Normal distribution
random_int = randint(0, 10, 3, 3)  # Random integers
random_uniform = uniform(0.0, 1.0, 3, 3)  # Uniform distribution

# Fill with specific value
filled = fill(7.5, 2, 2)  # 2x2 tensor filled with 7.5
```

## Basic Operations

### Arithmetic Operations

```python
x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])

# Element-wise operations
addition = x + y        # or x + 5 (scalar)
subtraction = x - y     # or x - 2 (scalar)
multiplication = x * y  # or x * 3 (scalar)
division = x / y        # or x / 2 (scalar)

# Power operations
power = x ** 2          # Element-wise power
base_power = 2 ** x     # Scalar base to tensor power
```

### Matrix Operations

```python
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])

# Matrix multiplication
matmul = x @ y

# Dot product (for vectors)
dot_product = x.dot(y)
```

### Comparison and Sign

```python
x = Tensor([-1, 0, 1])

# Sign function
signs = x.sign()  # [-1, 0, 1]

# Negation
negated = -x      # [1, 0, -1]
```

## Mathematical Functions

### Basic Math Functions

```python
x = Tensor([1, 4, 9])

# Square root
sqrt_x = x.sqrt()

# Absolute value
abs_x = x.abs()

# Exponential and logarithm
exp_x = x.exp()
log_x = x.log()
```

### Trigonometric Functions

```python
x = Tensor([0, 1.57, 3.14])  # 0, π/2, π

# Basic trigonometric
sin_x = x.sin()
cos_x = x.cos()
tan_x = x.tan()

# Hyperbolic functions
sinh_x = x.sinh()
cosh_x = x.cosh()
tanh_x = x.tanh()
```

## Shape Operations

### Reshaping

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

# Reshape to different dimensions
reshaped = x.reshape([3, 2])  # Shape: (3, 2)
reshaped = x.reshape((6,))    # Shape: (6,)

# Flatten to 1D
flattened = x.flatten()       # Shape: (6,)
```

### Transpose

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
transposed = x.transpose()           # Shape: (3, 2)
```

### Reduction Operations

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])

# Sum all elements
total_sum = x.sum()  # Scalar

# Sum along axis
sum_axis0 = x.sum(axis=0)  # Sum along rows
sum_axis1 = x.sum(axis=1)  # Sum along columns

# Keep dimensions
sum_keepdims = x.sum(axis=0, keepdims=True)
```

## Automatic Differentiation

### Basic Gradient Computation

```python
# Enable gradient tracking
x = Tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Compute gradients
y.backward()
print(x.grad)  # dy/dx = 2*x + 3 = 7.0
```

### Multi-variable Gradients

```python
x = Tensor([1.0], requires_grad=True)
y = Tensor([2.0], requires_grad=True)
z = x * y + x ** 2

z.backward()
print(x.grad)  # dz/dx = y + 2*x = 4.0
print(y.grad)  # dz/dy = x = 1.0
```

### Gradient Hooks

```python
def print_grad(grad):
    print(f"Gradient: {grad}")
    return grad

x = Tensor([1.0], requires_grad=True)
x.register_hook(print_grad)

y = x ** 2
y.backward()  # Will print the gradient when computed
```

## Data Types

Supported data types:

### Integer Types
- `int8`, `int16`, `int32`, `int64`, `long`
- `uint8`, `uint16`, `uint32`, `uint64`

### Floating Point Types
- `float32`, `float64`, `double`

### Boolean Type
- `boolean` (or `bool`)

### Type Conversion

```python
x = Tensor([1, 2, 3], dtype="int32")
x_float = x.astype("float64")
```

## Examples

### Linear Regression Example

```python
# Generate data
X = randn(100, 1)
y = 3 * X + 2 + 0.1 * randn(100, 1)

# Initialize parameters
W = Tensor([[0.0]], requires_grad=True)
b = Tensor([0.0], requires_grad=True)

# Training loop
learning_rate = 0.01
for epoch in range(1000):
    # Forward pass
    y_pred = X @ W + b
    loss = ((y_pred - y) ** 2).sum() / len(y)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    W = W - learning_rate * W.grad
    b = b - learning_rate * b.grad
    
    # Reset gradients
    W.grad = None
    b.grad = None
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.tolist()}")
```

### Neural Network Layer Example

```python
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = randn(in_features, out_features, dtype="float32") * 0.1
        self.b = zeros(out_features, dtype="float32")
        self.W.requires_grad = True
        self.b.requires_grad = True
    
    def forward(self, x):
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]

# Usage
layer = LinearLayer(784, 128)
x = randn(32, 784)  # Batch of 32 samples
output = layer.forward(x)
```

### Matrix Operations Example

```python
# Create matrices
A = Tensor([[1, 2], [3, 4]], dtype="float32")
B = Tensor([[5, 6], [7, 8]], dtype="float32")

# Various operations
C = A @ B           # Matrix multiplication
D = A * B           # Element-wise multiplication
E = A.transpose()   # Transpose
F = A.sum(axis=0)   # Column sums

print(f"A @ B = {C.tolist()}")
print(f"A * B = {D.tolist()}")
print(f"A^T = {E.tolist()}")
print(f"Column sums = {F.tolist()}")
```

## Best Practices

1. **Memory Management**: The library uses C backend, so be mindful of large tensor operations
2. **Gradient Tracking**: Only enable `requires_grad=True` when necessary to save memory
3. **Data Types**: Choose appropriate data types for your use case (float32 is usually sufficient)
4. **Shape Consistency**: Ensure tensor shapes are compatible for operations
5. **Gradient Reset**: Remember to reset gradients to None after parameter updates in training loops

## Error Handling

Common errors and solutions:

- **Shape Mismatch**: Ensure tensors have compatible shapes for operations
- **Scalar Backward**: Only scalar tensors (0-d or 1-element tensors) can call `backward()`
- **Reshape Error**: New shape must have the same total number of elements
- **Transpose Limitation**: Transpose is limited to tensors with ≤3 dimensions

## Performance Tips

1. Use appropriate batch sizes for matrix operations
2. Prefer in-place operations when gradients aren't needed
3. Use the correct data type (float32 vs float64) based on precision requirements
4. Leverage vectorized operations instead of loops when possible
