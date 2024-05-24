import random

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

def randn(domain=(1, -1), shape=None):
  if len(shape) == 1:
    return [random.uniform(domain[0], domain[1]) for _ in range(shape[0])]
  else:
    return [randn(domain=domain, shape=shape[1:]) for _ in range(shape[0])]