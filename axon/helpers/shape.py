def get_shape(data):
  return [len(data),] + get_shape(data[0]) if isinstance(data, list) else []

def _unsqueeze(data, dim=0):
  if dim == 0:
    return [item for sublist in data for item in _unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [_unsqueeze(d, dim-1) for d in data]
    return [data]

def flatten(input_tensor, start_dim=0, end_dim=-1):
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return _flatten(data)
    else:
      return data
  
  if end_dim == -1:
    end_dim = len(input_tensor) - 1

  return [_recurse_flatten(input_tensor, 0)]

def transpose(arr):
  return list(map(list, zip(*arr)))

def re_transpose(data, dim0, dim1, ndim, depth=0):
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [re_transpose(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

def broadcasted_shape(shape1, shape2):
  res_shape = []
  if shape1 == shape2:
    return shape1, False
  
  max_len = max(len(shape1), len(shape2))
  shape1 = [1] * (max_len - len(shape1)) + shape1
  shape2 = [1] * (max_len - len(shape2)) + shape2

  for dim1, dim2 in zip(shape1, shape2):
    if dim1 != dim2 and dim1 != 1 and dim2 != 1:
      raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
    res_shape.append(max(dim1, dim2))
  return res_shape, True

def broadcast_array(array, target_shape):
  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array

  def expand_dims(array, current_shape, target_shape):
    if len(current_shape) < len(target_shape):
      array = [array]
      current_shape = [1,] + current_shape
    if current_shape == target_shape:
      return array

    if current_shape[0] == 1:
      array = array * target_shape[0]
    result = []
    for subarray in array:
      result.append(expand_dims(subarray, current_shape[1:], target_shape[1:]))
    return result

  return expand_dims(array, current_shape, target_shape)

def _flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in _flatten(sublist)]
  else:
    return [data]

def _reshape(data, new_shape):
  flat_data = _flatten(data)
  shape_size = _shape_size(new_shape)
  if shape_size != len(flat_data):
    raise ValueError("Total size of new array must be unchanged")

  def _recursive_reshape(data, shape):
    if len(shape) == 1:
      return data[:shape[0]]
    size = shape[0]
    sub_size = _shape_size(shape[1:])
    return [_recursive_reshape(data[i * sub_size:(i + 1) * sub_size], shape[1:]) for i in range(size)]

  return _recursive_reshape(flat_data, new_shape)

def _shape_size(shape):
  size = 1
  for dim in shape:
    size *= dim
  return size

def _squeeze(data, dim):
  if dim is None:
    if isinstance(data, list):
      squeezed = [_squeeze(d, None) for d in data]
      return squeezed if len(squeezed) > 1 else squeezed[0]
    return data
  if isinstance(data, list):
    if dim == 0:
      return data[0] if len(data) == 1 else data
    return [_squeeze(d, dim - 1) for d in data]
  return data

def transpose(matrix):
  return list(map(list, zip(*matrix)))

def re_transpose(data, dim0, dim1, ndim, depth=0):
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [re_transpose(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]