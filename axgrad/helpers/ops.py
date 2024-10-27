"""
  @helpers/ops.py function file
  @breif Code contains various functions for mathematical functions
"""

from .shape import get_shape, broadcast_shape, broadcast, get_element, set_element, flatten
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
    for _ in range(axis):
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
    for _ in range(axis):
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
    for _ in range(axis):
      mean_vals = [mean_vals]
  return mean_vals

def _l1_norm(zeros, data):
  if isinstance(data, list):
    for d in data:
      zeros = _l1_norm(zeros, d)
  else:
    zeros += abs(data)
  return zeros

def _l2_norm(zeros, data):
  if isinstance(data, list):
    for d in data:
      zeros = _l2_norm(zeros, d)
  else:
    zeros += data * data
  return zeros

def _l3_norm(zeros, data):
  if isinstance(data, list):
    for d in data:
      zeros = _l3_norm(zeros, d)
  else:
    zeros += abs(data) ** 3
  return zeros

def compute_norm(data, p: int = 2):
  if data is None:
    raise ValueError("Data is None.")
  if p < 1:
    raise ValueError("p must be greater than or equal to 1.")

  norm_value = 0.0
  if p == 1:
    norm_value = _l1_norm(norm_value, data)
  elif p == 2:
    norm_value = _l2_norm(norm_value, data) ** 0.5
  else:
    norm_value = _l3_norm(norm_value, data) ** (1 / 3)
  return norm_value

def dedup(x): return list(dict.fromkeys(x))

def _stack(data, axis: int=0) -> list:
  if not data:
    raise ValueError("Need atleast one tensor to stack")

  # shape checking
  base_shape = data[0].shape
  for d in data:
    if d.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  
  # new shape after stacking & initilization
  new_shape = list(base_shape[:])
  new_shape.insert(axis, len(data))
  new_data = _zeros(new_shape)

  def insert_data(new_data, tensors, axis, indices=[]):
    if len(indices) == len(new_shape):
      for idx, tensor in enumerate(tensors):
        data_idx = indices[:]
        data_idx[axis] = idx
        sub_arr = new_data
        for k in data_idx[:-1]:
          sub_arr = sub_arr[k]
        sub_arr[data_idx[-1]] = get_element(tensor.data, indices[:axis] + indices[axis+1:])
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, tensors, axis, indices + [i])
  
  insert_data(new_data, data, axis)
  return new_data

def _concat(data, axis) -> list:
  if not data:
    raise ValueError("Need atleast one tensor to stack")
  
  # shape checking
  base_shape = list(data[0].shape) # shape of first tensor for target tensor
  for arr in data:
    if list(arr.shape)[:axis] + list(arr.shape)[axis+1:] != base_shape[:axis] + base_shape[axis+1:]:
      raise ValueError("All input tensors must have the same shape except for the concatenation axis")
  
  new_shape = base_shape[:]
  new_shape[axis] *= len(data)
  new_data = _zeros(new_shape)

  def insert_data(new_data, tensors, axis, indices=[]):
    if len(indices) == len(new_shape):
      current_offset = 0
      for tensor in tensors:
        if current_offset <= indices[axis] < current_offset + tensor.shape[axis]:
          local_indices = indices[:]
          local_indices[axis] -= current_offset
          ele = get_element(tensor.data, local_indices)
          set_element(new_data, indices, ele)
          break
        current_offset += tensor.shape[axis]
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, tensors, axis, indices + [i])
  
  insert_data(new_data, data, axis)
  return new_data

def matmul(A, B):
  def matmul_2d(A, B):
    assert len(A[0]) == len(B), "Incompatible dimensions for matrix multiplication"
    result = [[0] * len(B[0]) for _ in range(len(A))]

    for i in range(len(A)):
      for j in range(len(B[0])):
        for k in range(len(B)):
          result[i][j] += A[i][k] * B[k][j]
          
    return result

  if len(get_shape(A)) == 2 and len(get_shape(B)) == 2:
    return matmul_2d(A, B)

  target_shape, _ = broadcast_shape(get_shape(A), get_shape(B), ops="<MATMUL>")
  A = broadcast(A, target_shape)
  B = broadcast(B, target_shape)

  if len(get_shape(A)) > 2 or len(get_shape(B)) > 2:
    return [matmul(a, b) for a, b in zip(A, B)]
  
  return matmul_2d(A, B)

def _conv2d(input_data, kernel_data, stride):
  input_h, input_w = len(input_data), len(input_data[0])
  kernel_h, kernel_w = len(kernel_data), len(kernel_data[0])
  output_h, output_w = (input_h - kernel_h) // stride + 1, (input_w - kernel_w) // stride + 1
  output = _zeros((output_h, output_w))
  for i in range(0, output_h):
    for j in range(0, output_w):
      for m in range(kernel_h):
        for n in range(kernel_w):
          output[i][j] += (
            input_data[i * stride + m][j * stride + n] * kernel_data[m][n]
          )
  return output

def _apply_padding(input_data, padding):
  if padding == 0:
    return input_data
  padded_shape = (len(input_data) + 2 * padding, len(input_data[0]) + 2 * padding)
  padded_input = _zeros(padded_shape)
  for i in range(len(input_data)):
    for j in range(len(input_data[0])):
      padded_input[i + padding][j + padding] = input_data[i][j]
  return padded_input

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