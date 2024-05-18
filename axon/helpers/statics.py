import random
import math

def relu(x:float) -> float:
  return max(0, x)

def relu_derivative(x:float) -> float:
  return 1 if x > 0 else 0

def LeakyRELU(x:float, alpha: float=0.03) -> float:
  return x if x >= 0 else alpha * x

def LeakyRELU_derivative(x:float, alpha:float= 0.03) -> float:
  return 1 if x > 0 else alpha

def tanh(x:float) -> float:
  return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanh_derivative(x:float) -> float:
  return 1 - (tanh(x)**2)

def sigmoid(x:float) -> float:
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x:float) -> float:
  return sigmoid(x)(1 - sigmoid(x))

def gelu(x:float) -> float:
  return 

def softmax(x):
  max_x = max(x)
  exp_values = [math.exp(i - max_x) for i in x]
  exp_sum = sum(exp_values)
  return [exp_value / exp_sum for exp_value in exp_values]

def zeros_like(arr, dtype=int):
  """
    Create an array of zeros with the same shape and dtype as the input array.
    Args:
      arr (list or tuple): Input array.
    Returns:
      list: Array of zeros with the same shape and dtype as the input array.
  """
  if isinstance(arr, list):
    return [zeros_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(zeros_like(elem) for elem in arr)
  else:
    return dtype(0)

def zeros(shape, dtype=int):
  """
    creates an array of zeros according to the given input
    Args:
      shape (list or tuple): shape array.
      dtype (int or float): determines the datatype
    Returns:
      list: array of zeros with the same shape as input_shape and in same dtype.
  """
  if len(shape) == 1:
    return [0] * shape[0] if dtype is None else [dtype(0)] * shape[0]
  else:
    return [zeros(shape[1:], dtype=dtype) for _ in range(shape[0])]

def ones(shape, dtype=int):
  """
    creates an array of ones according to the given input
    Args:
      shape (list or tuple): shape array.
      dtype (int or float): determines the datatype
    Returns:
      list: array of ones with the same shape as input_shape and in same dtype.
  """
  if len(shape) == 1:
    return [1] * shape[0] if dtype is None else [dtype(1)] * shape[0]
  else:
    return [ones(shape[1:], dtype=dtype) for _ in range(shape[0])]

def ns(shape, n, dtype=int):
  """
    creates an array of 'n' according to the given input number
    Args:
      shape (list or tuple): shape array
      n (int or float): sets the number for outputs
      dtype (int or float): determines the datatype
    Returns:
      list: array of number equal to 'n' with the same shape as input_shape and in same dtype.
  """
  if len(shape) == 1:
    return [n] * shape[0] if dtype is None else [dtype(n)] * shape[0]
  else:
    return [ns(shape[1:], dtype=dtype) for _ in range(shape[0])]

def randint(low, high, size=None, dtype=int):
  if size is None:
    if dtype is None:
      return random.randint(low, high)
    else:
      return dtype(random.randint(low, high))
  else:
    if dtype is None:
      return [random.randint(low, high) for _ in range(size)]
    else:
      return [dtype(random.randint(low, high)) for _ in range(size)]

def arange(start, end, step):
  return [start + i * step for i in range(int((end - start) / step))]