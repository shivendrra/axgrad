from .shape import transpose, get_shape
from .utils import _zeros

def sum_axis0(data):
  if not isinstance(data[0], list):
    return sum(data)

  result = []
  for i in range(len(data[0])):
    result.append(sum_axis0([d[i] for d in data]))
  return result

def mean_axis0(data):
  if not isinstance(data[0], list):
    return sum(data) / len(data)

  result = []
  for i in range(len(data[0])):
    result.append(mean_axis0([d[i] for d in data]))
  return result

def var_axis0(data, ddof=0):
  mean_values = mean_axis0(data)
  if not isinstance(data[0], list):
    return sum((x - mean_values) ** 2 for x in data) / (len(data) - ddof)

  result = []
  for i in range(len(data[0])):
    result.append(var_axis0([d[i] for d in data], ddof))
  return result

def mean_axis(data, axis, keepdims):
  if axis == 0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    mean_vals = [mean_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) / len(d) for d in transposed]
  else:
    mean_vals = [mean_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) / len(d) for d in data]
  if keepdims:
    mean_vals = [mean_vals]
  return mean_vals

def var_axis(data, mean_values, axis, ddof, keepdims):
  if axis == 0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    variance = [var_axis(d, mean_values[i], axis - 1, ddof, keepdims) if isinstance(d[0], list) else sum((x - mean_values[i]) ** 2 for x in d) / (len(d) - ddof) for i, d in enumerate(transposed)]
  else:
    variance = [var_axis(d, mean_values[i], axis - 1, ddof, keepdims) if isinstance(d[0], list) else sum((x - mean_values[i]) ** 2 for x in d) / (len(d) - ddof) for i, d in enumerate(data)]
  if keepdims:
    variance = [variance]
  return variance

def sum_axis(data, axis, keepdims):
  if axis==0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in transposed]
  else:
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in data]
  if keepdims:
    mean_vals = [mean_vals]
  return mean_vals

def matmul(a, b):
  def _remul(a, b):
    if len(get_shape(a)) == 2 and len(get_shape(b)) == 2:
      out = _zeros((len(a), len(b[0])))
      b_t = transpose(b)
      for i in range(len(a)):
        for j in range(len(b_t)):
          out[i][j] = sum(a[i][k] * b_t[j][k] for k in range(len(a[0])))
      return out
    else:
      out_shape = get_shape(a)[:-1] + (get_shape(b)[-1],)
      out = _zeros(out_shape)
      for i in range(len(a)):
        out[i] = _remul((a[i]), (b[i]))
      return out

  if get_shape(a)[-1] != get_shape(b)[-2]:
    raise ValueError("Matrices have incompatible dimensions for matmul")
  return _remul(a, b)

def dot_product(a, b):
  def dot_product_1d(v1, v2):
    if len(v1) != len(v2):
      raise ValueError("Vectors must have the same length")
    return sum(x * y for x, y in zip(v1, v2))

  def dot_product_2d(m1, m2):
    if len(m1[0]) != len(m2):
      raise ValueError("Incompatible dimensions for matrix multiplication")
    result = [[0] * len(m2[0]) for _ in range(len(m1))]
    for i in range(len(m1)):
      for j in range(len(m2[0])):
        result[i][j] = sum(m1[i][k] * m2[k][j] for k in range(len(m2)))
    return result

  if isinstance(a[0], list) and isinstance(b[0], list):
    if len(a[0]) == len(b):
      return dot_product_2d(a, b)
    else:
      raise ValueError("Incompatible dimensions for 2-D dot product")
  elif isinstance(a[0], list) or isinstance(b[0], list):
    if isinstance(a[0], list):
      if len(a[0]) == len(b):
        return [sum(x * y for x, y in zip(row, b)) for row in a]
      else:
        raise ValueError("Incompatible dimensions for 2-D and 1-D dot product")
    else:
      if len(a) == len(b):
        return [sum(x * y for x, y in zip(a, row)) for row in b]
      else:
        raise ValueError("Incompatible dimensions for 1-D and 2-D dot product")
  else:
    if len(a) != len(b):
      raise ValueError("Vectors must have the same length")
    return dot_product_1d(a, b)

def determinant(data):
  def minor(matrix, row, col):
    """Calculate the minor of the matrix after removing the specified row and column."""
    return [row[:col] + row[col+1:] for row in (matrix[:row] + matrix[row+1:])]

  def determinant_2d(matrix):
    """Calculate the determinant of a 2x2 matrix."""
    if len(matrix) != 2 or len(matrix[0]) != 2:
      raise ValueError("Matrix must be 2x2")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

  def determinant_nd(matrix):
    n = len(matrix)
    if n == 2:
      return determinant_2d(matrix)
    elif n == 1:
      return matrix[0][0]
    elif n == 0:
      return 1
    
    det = 0
    for c in range(n):
      det += ((-1) ** c) * matrix[0][c] * determinant_nd(minor(matrix, 0, c))
    return det

  if not data:
    raise ValueError("Matrix is empty")
  if not all(len(row) == len(data) for row in data):
    raise ValueError("Matrix must be square (number of rows must equal number of columns)")

  return determinant_nd(data)