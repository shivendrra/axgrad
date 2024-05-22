""" tensor's basic shape operations, all in one file """

def get_shape(arr):
  if isinstance(arr, list):
    return (len(arr),) + get_shape(arr[0])
  else:
    return ()

def check_arr(arr1, arr2):
  if get_shape(arr1)[1] == get_shape(arr2)[0] and len(get_shape(arr2)) == 2:
    return True

def view_arr(arr):
  raise NotImplementedError("view function is empty")

def _flatten(arr, new=None):
  if new is None:
    new = []
  if isinstance(arr, list):
    for i in arr:
      _flatten(i, new)
  elif isinstance(arr, int) or isinstance(arr, float):
    new.append(arr)
  return new

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

def _unsqueeze(data, dim):
  if dim == 0:
    return [data]
  if isinstance(data, list):
    return [_unsqueeze(d, dim-1) for d in data]
  return [data]

def _reshape(data, flat_data, new_shape):
  reshaped = []
  shape_size = _shape_size(data, new_shape)
  if shape_size != len(flat_data):
    raise ValueError("Total size of new array must be unchanged")

  def _recursive_reshape(data, shape):
    if len(shape) == 1:
      return data[:shape[0]]
    size = shape[0]
    sub_size = _shape_size(data, shape[1:])
    return [_recursive_reshape(data[i * sub_size:(i + 1) * sub_size], shape[1:]) for i in range(size)]

  return _recursive_reshape(flat_data, new_shape)

def _shape_size(data, shape):
  size = 1
  for dim in shape:
    size *= dim
  return size

def broadcast_shapes(shape1, shape2):
  result_shape = []
  for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
    if dim1 == 1 or dim2 == 1 or dim1 == dim2:
      result_shape.append(max(dim1, dim2))
    else:
      raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
  result_shape.extend(reversed(shape1[len(shape2):]))
  result_shape.extend(reversed(shape2[len(shape1):]))
  return tuple(reversed(result_shape))

def broadcast_array(array, target_shape):
  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array

  def expand_dims(array, current_shape, target_shape):
    if len(current_shape) < len(target_shape):
      array = [array]
      current_shape = (1,) + current_shape
    if current_shape == target_shape:
      return array

    if current_shape[0] == 1:
      array = array * target_shape[0]
    result = []
    for subarray in array:
      result.append(expand_dims(subarray, current_shape[1:], target_shape[1:]))
    return result
  
  return expand_dims(array, current_shape, target_shape)

def transpose(arr):
  if isinstance(arr[0], list):
    return [list(row) for row in zip(*arr)]
  else:
    return arr