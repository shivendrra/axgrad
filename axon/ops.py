from .tensor import tensor
from .helpers.statics import zeros
from .helpers.shape import transpose

def get_element(data, indices):
  for idx in indices:
    data = data[idx]
  return data

def matmul(a, b):
  def _remul(a, b):
    if len(a.shape) == 2 and len(b.shape) == 2:
      out = zeros((len(a.data), len(b.data[0])))
      b_t = transpose(b.data)
      for i in range(len(a.data)):
        for j in range(len(b_t)):
          out[i][j] = sum(a.data[i][k] * b_t[j][k] for k in range(len(a.data[0])))
      return out
    else:
      out_shape = a.shape[:-1] + (b.shape[-1],)
      out = zeros(out_shape)
      for i in range(len(a.data)):
        out[i] = _remul(tensor(a.data[i]), tensor(b.data[i]))
      return out

  if a.shape[-1] != b.shape[-2]:
    raise ValueError("Matrices have incompatible dimensions for matmul")

  a = a if isinstance(a, tensor) else tensor(a)
  b = b if isinstance(b, tensor) else tensor(b)
  out = _remul(a, b)
  return tensor(out, child=(a, b), _ops='<Matmul>')

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