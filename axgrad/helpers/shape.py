def get_shape(data):
  return (len(data),) + get_shape(data[0]) if isinstance(data, list) else ()

def _unsqueeze(data, dim=0):
  if dim == 0:
    return [item for sublist in data for item in _unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [_unsqueeze(d, dim-1) for d in data]
    return [data]

def _flatten(data):
  if isinstance(data, list):
    result = []
    for item in data:
      result.extend(_flatten(item))
    return result
  else:
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