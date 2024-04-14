import random

def zeros(shape, dtype=None):
  if len(shape) == 1:
    return [0] * shape[0]
  else:
    result = []
    for _ in range(shape[0]):
      result.append(zeros(shape[1:]))
    return result

def ones(shape, dtype=None):
  if len(shape) == 1:
    return [1] * shape[0]
  else:
    result = []
    for _ in range(shape[0]):
      result.append(zeros(shape[1:]))
    return result

def ns(shape, n, dtype=None):
  if len(shape) == 1:
    return [n] * shape[0]
  else:
    result = []
    for _ in range(shape[0]):
      result.append(zeros(shape[1:]))
    return result

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