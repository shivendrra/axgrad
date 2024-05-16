# Axon framework documentation

The `tensor` class provides an implementation of multi-dimensional arrays (tensors) and various operations on them. It supports element-wise arithmetic, matrix multiplication, broadcasting, transposition, and more.

## Initialization

### __init__
Initializes a tensor object.
- **Args**:
  - `data` (list): The data to initialize the tensor with. Can be nested lists.
  - `requires_grad` (bool, optional): Indicates if gradients are required for this tensor. Default is `False`.
  - `child` (set, optional): Internal use for tracking gradient computation.

**Example**:
```python
t1 = tensor([[1, 2], [3, 4]])
t2 = tensor([5, 6, 7])
```

## Basic Operations

### __add__
Element-wise addition of two tensors.
- **Args**:
  - `other` (tensor): The tensor to add.
- **Returns**: A new tensor with the element-wise sum.

**Example**:
```python
result = t1 + tensor([[1, 1], [1, 1]])
print(result)  # Output: tensor(data=[[2, 3], [4, 5]])
```

### __mul__
Element-wise multiplication of two tensors.
- **Args**:
  - `other` (tensor): The tensor to multiply.
- **Returns**: A new tensor with the element-wise product.

**Example**:
```python
result = t1 * tensor([[2, 2], [2, 2]])
print(result)  # Output: tensor(data=[[2, 4], [6, 8]])
```

### __sub__
Element-wise subtraction of two tensors.
- **Args**:
  - `other` (tensor): The tensor to subtract.
- **Returns**: A new tensor with the element-wise difference.

**Example**:
```python
result = t1 - tensor([[1, 1], [1, 1]])
print(result)  # Output: tensor(data=[[0, 1], [2, 3]])
```

### __truediv__
Element-wise division of two tensors.
- **Args**:
  - `other` (tensor): The tensor to divide by.
- **Returns**: A new tensor with the element-wise quotient.

**Example**:
```python
result = t1 / tensor([[1, 2], [3, 4]])
print(result)  # Output: tensor(data=[[1.0, 1.0], [1.0, 1.0]])
```

### __pow__
Raises each element in the tensor to a specified power.
- **Args**:
  - `pow` (int, float): The power to raise each element to.
- **Returns**: A new tensor with each element raised to the specified power.

**Example**:
```python
result = t1 ** 2
print(result)  # Output: tensor(data=[[1, 4], [9, 16]])
```

### Matrix Operations

### .transpose()
Transposes the tensor (swaps rows and columns).
- **Args**: None.
- **Returns**: A new transposed tensor.

**Example**:
```python
result = t1.transpose()
print(result)  # Output: tensor(data=[[1, 3], [2, 4]])
```

### .matmul()
Performs matrix multiplication on two tensors.
- **Args**:
  - `x` (tensor): The first tensor.
  - `y` (tensor): The second tensor.
- **Returns**: A new tensor resulting from the matrix multiplication.

**Example**:
```python
t1 = tensor([[1, 2], [3, 4]])
t2 = tensor([[5, 6], [7, 8]])
result = tensor.matmul(t1, t2)
print(result)  # Output: tensor(data=[[19, 22], [43, 50]])
```

### .convolution_2d()
Performs 2D convolution on an image with a given kernel.
- **Args**:
  - `image` (list): The 2D list representing the image.
  - `kernel` (list): The 2D list representing the convolution kernel.
- **Returns**: A 2D list representing the convolved image.

**Example**:
```python
image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
kernel = [[1, 0], [0, -1]]
result = tensor.convolution_2d(image, kernel)
print(result)  # Output: [[-3, -3], [-3, -3]]
```

## Utility Functions

The following utility functions create and manipulate arrays (lists) with various properties. These functions do not rely on external libraries and are simple and efficient for generating and manipulating basic arrays, providing a core functionality similar to numpy for small-scale applications.

### .flatten()
Flattens a multi-dimensional tensor into a 1D list.
- **Args**: None.
- **Returns**: A 1D list containing all the elements of the tensor.

**Example**:
```python
result = t1.flatten()
print(result)  # Output: [1, 2, 3, 4]
```

### .sum()
Sums all elements in the tensor.
- **Args**:
  - `dtype` (optional): The data type of the output sum (int or float).
- **Returns**: The sum of all elements in the tensor.

**Example**:
```python
result = t1.sum()
print(result)  # Output: 10
```

### .broadcast()
Broadcasts two tensors to a common shape.
- **Args**:
  - `other` (tensor): The tensor to broadcast with.
- **Returns**: Two new tensors with the broadcasted shape.

**Example**:
```python
t3 = tensor([1, 2, 3])
t4 = tensor([[1], [2], [3]])
b1, b2 = t3.broadcast(t4)
print(b1)  # Output: tensor(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])
print(b2)  # Output: tensor(data=[[1, 1, 1], [2, 2, 2], [3, 3, 3]])
```


### .zeros_like()

Creates an array of zeros with the same shape and data type as the input array.

- **Args**:
  - `arr` (list or tuple): Input array to copy the shape from.
  - `dtype` (type, optional): Data type of the output array elements. Default is `int`.

- **Returns**: 
  - `list`: Array of zeros with the same shape and data type as the input array.

**Example**:
```python
input_array = [[1, 2], [3, 4]]
output_array = axon.zeros_like(input_array)
print(output_array)  # Output: [[0, 0], [0, 0]]
```

### .zeros()

Creates an array of zeros with the specified shape and data type.

- **Args**:
  - `shape` (list or tuple): Shape of the output array.
  - `dtype` (type, optional): Data type of the output array elements. Default is `int`.

- **Returns**: 
  - `list`: Array of zeros with the specified shape and data type.

**Example**:
```python
output_array = axon.zeros((2, 3))
print(output_array)  # Output: [[0, 0, 0], [0, 0, 0]]
```

### .ones()

Creates an array of ones with the specified shape and data type.

- **Args**:
  - `shape` (list or tuple): Shape of the output array.
  - `dtype` (type, optional): Data type of the output array elements. Default is `int`.

- **Returns**: 
  - `list`: Array of ones with the specified shape and data type.

**Example**:
```python
output_array = axon.ones((2, 3))
print(output_array)  # Output: [[1, 1, 1], [1, 1, 1]]
```

### .ns()

Creates an array filled with a specified value `n`, with the specified shape and data type.

- **Args**:
  - `shape` (list or tuple): Shape of the output array.
  - `n` (int or float): The value to fill the array with.
  - `dtype` (type, optional): Data type of the output array elements. Default is `int`.

- **Returns**: 
  - `list`: Array filled with the specified value `n`, with the specified shape and data type.

**Example**:
```python
output_array = axon.ns((2, 3), 5)
print(output_array)  # Output: [[5, 5, 5], [5, 5, 5]]
```

### .randint()

Generates an array of random integers within a specified range.

- **Args**:
  - `low` (int): Lower bound of the range (inclusive).
  - `high` (int): Upper bound of the range (inclusive).
  - `size` (int, optional): Number of random integers to generate. If `None`, returns a single integer. Default is `None`.
  - `dtype` (type, optional): Data type of the output array elements. Default is `int`.

- **Returns**: 
  - `list` or `int`: Array of random integers if `size` is specified, otherwise a single random integer.

**Example**:
```python
output_array = axon.randint(1, 10, size=5)
print(output_array)  # Output: [3, 7, 2, 8, 6] (example output, actual values will vary)
```

### .arange()

Generates a list of evenly spaced values within a specified range.

- **Args**:
  - `start` (int or float): Start of the interval.
  - `end` (int or float): End of the interval.
  - `step` (int or float): Step size between values.

- **Returns**: 
  - `list`: List of evenly spaced values.

**Example**:
```python
output_array = axon.arange(0, 10, 2)
print(output_array)  # Output: [0, 2, 4, 6, 8]
```

### Conclusion

The `tensor` class is a versatile and powerful tool for numerical computations involving multi-dimensional arrays. It supports various mathematical operations and provides utility functions for reshaping and transforming data. The class can be easily extended and integrated into larger projects involving machine learning, data analysis, and more.