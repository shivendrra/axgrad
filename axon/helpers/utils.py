import random

def zeros(shape):
  if len(shape) == 1:
    return [0] * shape[0]
  return [zeros(shape[1:]) for _ in range(shape[0])]

def ones(shape):
  if len(shape) == 1:
    return [1] * shape[0]
  return [ones(shape[1:]) for _ in range(shape[0])]

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
  return [start + i * step for i in range(int((end-start)/step))]

def randn(domain=(1, -1), shape=None):
  if len(shape) == 1:
    return [random.uniform(domain[0], domain[1]) for _ in range(shape[0])]
  else:
    return [randn(domain=domain, shape=shape[1:]) for _ in range(shape[0])]

def zeros_like(arr, dtype=int):
  if isinstance(arr, list):
    return [zeros_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(zeros_like(elem) for elem in arr)
  else:
    return dtype(0)

def ones_like(arr, dtype=int):
  if isinstance(arr, list):
    return [ones_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(ones_like(elem) for elem in arr)
  else:
    return dtype(1)

def generate_random_list(shape):
  if len(shape) == 0:
    return []
  else:
    inner_shape = shape[1:]
    if len(inner_shape) == 0:
      return [random.uniform(-1, 1) for _ in range(shape[0])]
    else:
      return [generate_random_list(inner_shape) for _ in range(shape[0])]