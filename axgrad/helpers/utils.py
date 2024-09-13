import random

def _zeros(shape):
  if len(shape) == 1:
    return [0] * shape[0]
  return [_zeros(shape[1:]) for _ in range(shape[0])]

def _ones(shape):
  if len(shape) == 1:
    return [1] * shape[0]
  return [_ones(shape[1:]) for _ in range(shape[0])]

def _randint(low, high, size=None, dtype=int):
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

def _arange(start, end, step):
  return [start + i * step for i in range(int((end-start)/step))]

def _randn(domain=(1, -1), shape=None):
  if len(shape) == 1:
    return [random.uniform(domain[0], domain[1]) for _ in range(shape[0])]
  else:
    return [_randn(domain=domain, shape=shape[1:]) for _ in range(shape[0])]

def _zeros_like(arr, dtype=int):
  if isinstance(arr, list):
    return [_zeros_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(_zeros_like(elem) for elem in arr)
  else:
    return dtype(0)

def _ones_like(arr, dtype=int):
  if isinstance(arr, list):
    return [_ones_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(_ones_like(elem) for elem in arr)
  else:
    return dtype(1)