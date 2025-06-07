"""
  @helpers/shape.py
  @breif Contains helper functions needed for shaping a tensor
"""

from .utils import _zeros

def get_element(data, indices):
  for idx in indices:
    data = data[idx]
  return data

def set_element(data, indices, value):
  for idx in indices[:-1]:
    data = data[idx]
  data[indices[-1]] = value

def get_shape(data):
  if not isinstance(data, list):  # Base case: not a list, return empty shape.
    return []
  if len(data) == 0:  # Handle empty lists.
    return [0]
  return [len(data)] + get_shape(data[0])

def get_strides(shape:tuple) -> list:
  strides = [1]
  for size in reversed(shape[:-1]):
    strides.append(strides[-1] * size)
  return list(reversed(strides))

def get_size(shape:tuple) -> list:
  out = 1
  for dim in shape:
    out *= dim
  return out

# returns a flatten tensor by appending elements recursively
def flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  return [data]

def flatten_recursive(data:list, start_dim:int=0, end_dim:int=-1) -> list:
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return flatten(data)
    else:
      return data
  if end_dim == -1:
    end_dim = len(get_shape(data)) - 1
  return _recurse_flatten(data, 0)

# transpose by swaping the extreme dims/axes like np.transpose
def transpose(data:list) -> list:
  def fill_transposed(original, transposed, current_indices):
    """Recursively fill the transposed array."""
    if not isinstance(original, list):
      transposed_ref = transposed
      for i in range(len(current_indices) - 1):
        transposed_ref = transposed_ref[current_indices[::-1][i]]
      transposed_ref[current_indices[::-1][-1]] = original
      return

    for i in range(len(original)):
      fill_transposed(original[i], transposed, current_indices + [i])
  if not data or not isinstance(data, list):
    raise ValueError("Input must be a non-empty n-dimensional list.")
  shape = get_shape(data)
  transposed_shape = shape[::-1]
  transposed = _zeros(transposed_shape)
  fill_transposed(data, transposed, [])
  return transposed

# swaps axes, basically recursive swaping of n-d tensor like np.swapaxes
def swap_axes(array:list, axis1:int, axis2:int) -> list:
  def fill_swapped(original, swapped, axis1, axis2, indices): # recursively fill the swapped data
    if not isinstance(original, list):
      swapped_ref = swapped
      for i in range(len(indices) - 1):
        swapped_ref = swapped_ref[indices[i] if i not in [axis1, axis2] else indices[axis2 if i == axis1 else axis1]]
      swapped_ref[indices[-1] if len(indices) - 1 not in [axis1, axis2] else indices[axis2 if len(indices) - 1 == axis1 else axis1]] = original
      return
    for i in range(len(original)):
      fill_swapped(original[i], swapped, axis1, axis2, indices + [i])

  if not array or not isinstance(array, list):
    raise ValueError("Input must be a non-empty n-dimensional list.")
  shape, ndim = get_shape(array), len(shape)
  
  if axis1 >= ndim or axis2 >= ndim:
    raise ValueError(f"Axes {axis1} and {axis2} are out of bounds for array of dimension {ndim}.")
  new_shape = shape[:] # creates a target shape
  new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]
  swapped = _zeros(new_shape)
  fill_swapped(array, swapped, axis1, axis2, [])
  return swapped

# reshapes the input data into new shape
# flattens it & the builds a new shape
def reshape(data:list, new_shape:tuple) -> list:
  assert type(new_shape) == tuple, "new shape must be a tuple"
  def _shape_numel(shape):
    numel = 1
    for dim in shape:
      numel *= dim
    return numel

  def unflatten(flat, shape):
    if len(shape) == 1:
      return flat[:shape[0]]
    size = shape[0]
    return [unflatten(flat[i*int(len(flat)/size):(i+1)*int(len(flat)/size)], shape[1:]) for i in range(size)]

  def infer_shape(shape, total_size):
    if shape.count(-1) > 1:
      raise ValueError("Only one dimension can be -1")
    unknown_dim, known_dims, known_size = shape.index(-1) if -1 in shape else None, [dim for dim in shape if dim != -1], 1
    for dim in known_dims:
      known_size *= dim      
    if unknown_dim is not None:
      inferred_size = total_size // known_size
      if inferred_size * known_size != total_size:
        raise ValueError(f"Cannot reshape array to shape {shape}")
      shape = list(shape)
      shape[unknown_dim] = inferred_size
    return shape
  original_size = _shape_numel(get_shape(data))
  new_shape, new_size = infer_shape(new_shape, original_size), _shape_numel(new_shape)
  if original_size != new_size:
    raise ValueError(f"Cannot reshape array of size {original_size} to shape {new_shape}")
  flat_data = flatten(data)
  return unflatten(flat_data, new_shape)

# checks the broadcastability
# creates new target shape for broadcasting
def broadcast_shape(shape1:tuple, shape2:tuple, ops=None) -> tuple:
  if ops == "<MATMUL>":
    if len(shape1) < 2 or len(shape2) < 2:
      raise ValueError("Both tensors must have at least two dimensions for matmul")
    if shape1[-1] != shape2[-2]:
      raise ValueError(f"Shapes {shape1} and {shape2} are incompatible for matmul (dimensions must align)")
    matmul_shape = (shape1[-2], shape2[-1]) # shape1[-2] x shape1[-1] and shape2[-2] x shape2[-1] -> result shape should be shape1[-2] x shape2[-1]
    batch_shape1, batch_shape2 = shape1[:-2], shape2[:-2] # broadcast the remaining batch dimensions, excluding the last two dims
    
    # broadcast batch dimensions, if necessary
    result_shape = []
    max_len = max(len(batch_shape1), len(batch_shape2))
    batch_shape1, batch_shape2 = [1] * (max_len - len(batch_shape1)) + batch_shape1, [1] * (max_len - len(batch_shape2)) + batch_shape2
    
    for dim1, dim2 in zip(batch_shape1, batch_shape2):
      if dim1 != dim2 and dim1 != 1 and dim2 != 1:
        raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
      result_shape.append(max(dim1, dim2))
    return tuple(result_shape + list(matmul_shape)), True

  else:
    res_shape = []
    if shape1 == shape2:
      return shape1, False # returns false if same shape
    max_len = max(len(shape1), len(shape2))
    shape1, shape2 = [1] * (max_len - len(shape1)) + shape1, [1] * (max_len - len(shape2)) + shape2
    for dim1, dim2 in zip(shape1, shape2):
      if dim1 != dim2 and dim1 != 1 and dim2 != 1:
        raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
      res_shape.append(max(dim1, dim2)) # appends max of the each axis/dim
    return tuple(res_shape), True

# main broadcasting function, broadcasts by copying data to match target shape
def broadcast(array, target_shape):
  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array

  def expand_dims(array, current_shape, target_shape): # expands dims by copying the dim data
    if not current_shape:
      return array
    if len(current_shape) < len(target_shape):
      array, current_shape = [array], [1] + current_shape
    if current_shape == target_shape:
      return array
    if current_shape[0] == 1:
      array = array * target_shape[0]
    result = []
    for subarray in array:
      result.append(expand_dims(subarray, current_shape[1:], target_shape[1:]))
    return result
  return expand_dims(array, current_shape, target_shape)

def unsqueeze(data:list, dim:int=0) -> list:
  if dim == 0:
    return [item for sublist in data for item in unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [unsqueeze(d, dim-1) for d in data]
    return [data]

def squeeze(data:list, dim:int) -> list:
  if dim is None:
    if isinstance(data, list):
      squeezed = [squeeze(d, None) for d in data]
      return squeezed if len(squeezed) > 1 else squeezed[0]
    return data
  if isinstance(data, list):
    if dim == 0:
      return data[0] if len(data) == 1 else data
    return [squeeze(d, dim - 1) for d in data]
  return data