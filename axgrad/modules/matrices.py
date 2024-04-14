import random

def zeros(shape, dtype=None):
  if len(shape) == 1:
    return [0] * shape[0] if dtype is None else [dtype(0)] * shape[0]
  else:
    return [zeros(shape[1:], dtype=dtype) for _ in range(shape[0])]

def ones(shape, dtype=None):
  if len(shape) == 1:
    return [1] * shape[0] if dtype is None else [dtype(1)] * shape[0]
  else:
    return [ones(shape[1:], dtype=dtype) for _ in range(shape[0])]

def ns(shape, n, dtype=None):
  if len(shape) == 1:
    return [n] * shape[0] if dtype is None else [dtype(n)] * shape[0]
  else:
    return [ns(shape[1:], dtype=dtype) for _ in range(shape[0])]

def randint(low, high, size=None, dtype=None):
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