from .tensor import tensor
from .helpers.utils import _zeros
from .helpers.shape import squeeze, unsqueeze, get_shape
from typing import *

def matmul(a:Union[tensor, list], b:Union[tensor, list], dtype=None) -> tensor:
  a = a if isinstance(a, tensor) else tensor(a, dtype=dtype)
  b = b if isinstance(b, tensor) else tensor(b, dtype=dtype)
  return (a @ b)

def dot(a:Union[tensor, list], b:Union[tensor, list]) -> tensor:
  a = a if isinstance(a, tensor) else tensor(a)
  b = b if isinstance(b, tensor) else tensor(b)
  return a.dot(b)

def stack(data: tuple[tensor, tensor], axis: int=0) -> tensor:
  if not data:
    raise ValueError("Need atleast one tensor to stack")

  def get_element(data, indices):
    for idx in indices:
      data = data[idx]
    return data

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
  return tensor(new_data, dtype=data[0].dtype)

def concat(data: tuple[tensor, tensor], axis: int=0) -> tensor:
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

  def set_element(data, indices, value):
    for idx in indices[:-1]:
      data = data[idx]
    data[indices[-1]] = value

  def get_element(data, indices):
    for idx in indices:
      data = data[idx]
    return data

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
  return tensor(new_data, dtype=data[0].dtype)

def split(data:Union[tensor, list], idx:int, axis:Optional[int]=None) -> list:
  def _get_slices(start_idx, end_idx, data):
    slices = []
    for start, end in zip(start_idx, end_idx):
      slices.append(data[start:end])
    return slices
  
  if isinstance(idx, int):
    N = idx
    total_len = len(data) if axis == 0 else len(data[0])
    if total_len % N != 0:
      raise ValueError("tensor split doesn't results in an equal division")
    step = total_len // N
    indices = [i*step for i in range(1, N)]
  else:
    indices = idx
  
  start_idx = [0] + indices
  end_idx = indices + [len(data) if axis==0 else len(data)]

  if axis == 0:
    return _get_slices(start_idx, end_idx, data)
  else:
    result = []
    for row in data:
      result.append(_get_slices(start_idx, end_idx, row))
    return [list(col) for col in zip(*result)]
  
def mean(data:Union[tensor, list], axis:Optional[int]=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, tensor) else tensor(data, dtype)
  return data.mean(axis=axis, keepdims=keepdims)

def var(data:Union[tensor, list], axis:Optional[int]=None, ddof:int=0, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, tensor) else tensor(data, dtype)
  return data.var(axis=axis, ddof=ddof, keepdims=keepdims)

def std(data:Union[tensor, list], axis:Optional[int]=None, ddof:int=0, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, tensor) else tensor(data, dtype)
  return data.std(axis=axis, ddof=ddof, keepdims=keepdims)

def squeeze(*data, dim:int=0) -> tensor:
  for _data in data:
    if dim is not None and dim>=len(get_shape(_data)):
      dim = dim if dim > 0 else len(get_shape(_data)) + dim
      raise IndexError(f"Dimension out of range (expected to be in range of {len(get_shape(_data))} dimensions)")
    else:
      return squeeze(_data, dim)

def unsqueeze(*data, dim:int=0):
  for _data in data:
    dim = dim if dim > 0 else len(get_shape(_data)) + dim
    return unsqueeze(_data, dim)

def clip(data:Union[tensor, list], min, max, out=None) -> tensor:
  data = data if isinstance(data, tensor) else tensor(data)
  if out is not None:
    return data.clip(min_value=min, max_value=max)
  else:
    out = data.clip(min_value=min, max_value=max)
    return out

def reshape(data:Union[tensor, list], new_shape:tuple) -> tensor:
  data = data if isinstance(data, tensor) else tensor(data)
  return data.reshape(new_shape)

def det(data:Union[tensor, list]) -> tensor:
  data = data if isinstance(data, tensor) else tensor(data)
  return data.det()

def swap_axes(data:Union[tensor, list], axis1:int, axis2:int) -> tensor:
  data = data if isinstance(data, tensor) else tensor(data)
  return data.swap_axes(axis1, axis2)