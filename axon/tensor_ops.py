from .arrays import tensor
from .helpers.statics import zeros

def get_element(data, indices):
  for idx in indices:
    data = data[idx]
  return data

def matmul(x:tensor, y:tensor) -> tensor:
  x = x if isinstance(x, tensor) else tensor(x)
  y = y if isinstance(y, tensor) else tensor(y)
  if len(x.data[0]) != len(y.data):
    raise ValueError(f"Matrices have incompatible dimensions for multiplication. {x.shape} != {y.shape}")

  out = zeros((len(x.data), len(y.data[0])))
  y_t = y.transpose().data
  for i in range(len(x.data)):
    for j in range(len(y_t)):
      out[i][j] = sum(x.data[i][k] * y_t[j][k] for k in range(len(y.data)))
  return tensor(out)

def stack(array: tuple, axis: int=0) -> tensor:
  if not array:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = array[0].shape
  for arr in array:
    if arr.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  
  # new shape after stacking & initilization
  new_shape = list(base_shape[:])
  new_shape.insert(axis, len(array))
  new_data = zeros(new_shape)

  def insert_data(new_data, arrays, axis, indices=[]):
    if len(indices) == len(new_shape):
      for idx, array in enumerate(arrays):
        data_idx = indices[:]
        data_idx[axis] = idx
        sub_arr = new_data
        for k in data_idx[:-1]:
          sub_arr = sub_arr[k]
        sub_arr[data_idx[-1]] = get_element(array.data, indices[:axis] + indices[axis+1:])
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, arrays, axis, indices + [i])
    
  insert_data(new_data, array, axis)
  return tensor(new_data)

def concat(array: tuple, axis: int=0) -> tensor:
  if not array:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = list(array[0].shape) # shape of first array for target array
  for arr in array:
    if list(arr.shape)[:axis] + list(arr.shape)[axis+1:] != base_shape[:axis] + base_shape[axis+1:]:
      raise ValueError("All input arrays must have the same shape except for the concatenation axis")
  
  new_shape = base_shape[:]
  new_shape[axis] *= len(array)
  new_data = zeros(new_shape)

  def set_element(data, indices, value):
    for idx in indices[:-1]:
      data = data[idx]
    data[indices[-1]] = value

  def insert_data(new_data, arrays, axis, indices=[]):
    if len(indices) == len(new_shape):
      current_offset = 0
      for array in arrays:
        if current_offset <= indices[axis] < current_offset + array.shape[axis]:
          local_indices = indices[:]
          local_indices[axis] -= current_offset
          ele = get_element(array.data, local_indices)
          set_element(new_data, indices, ele)
          break
        current_offset += array.shape[axis]
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, arrays, axis, indices + [i])
  
  insert_data(new_data, array, axis)
  return tensor(new_data)