def get_shape(data):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def transpose(data):
  return list(map(list, zip(*data)))

def swap_axes(data, dim0, dim1, ndim, depth=0):
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [swap_axes(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

def _broadcast_shape(shape1, shape2):
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

def broadcast(data, shape):
  raise NotImplementedError("Not yet written")