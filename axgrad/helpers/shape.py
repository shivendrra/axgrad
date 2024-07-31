from typing import *

def get_shape(data:list) -> list:
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def flatten(data:list) -> list:
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  else:
    return [data]

def flatten_recursive(data:list, start_dim:int=0, end_dim:int=-1) -> list:
  raise NotImplementedError("Not written yet")

def transpose(data:list) -> list:
  return list(map(list, zip(*data)))

def transpose_recursive(data:list, dim:int) -> list:
  raise NotImplementedError("not written")

def swap_axes(data:list, dim0:int, dim1:int, ndim:int, depth:int=0) -> list:
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [swap_axes(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

def broadcast_shape(shape1:tuple, shape2:tuple) -> tuple:
  res_shape = []
  if shape1 == shape2:
    return shape1, False
  
  else:
    max_len = max(len(shape1), len(shape2))
    shape1 = [1] * (max_len - len(shape1)) + shape1
    shape2 = [1] * (max_len - len(shape2)) + shape2

    for dim1, dim2 in zip(shape1, shape2):
      if dim1 != dim2 and dim2 != 1 and dim1 != 1:
        raise ValueError(f"shape {shape1} and {shape2} are not compatible for broadcasting")
      res_shape.append(dim1, dim2)
    return res_shape, True

def broadcast(data:list, shape:tuple) -> list:
  raise NotImplementedError("Not yet written")

def reshape(data:list, new_shape:tuple) -> list:
  
  raise NotImplementedError("Not yet written")

def unsqueeze(data:list, dim:int) -> list:
  raise NotImplementedError("Not yet written")

def squeeze(data:list, dim:int) -> list:
  raise NotImplementedError("Not yet written")