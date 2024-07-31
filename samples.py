
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

def tensor_sum(data:list, axis:int=None, keepdim:bool=False) -> list:
  def _re_sum(data, axis):
    if axis is None:
      return [sum(flatten(data))]
    elif axis == 0:
      return [sum(row[i] for row in data) for i in range(len(data[0]))]
    else:
      for i in range(len(data[0])):
        for row in data:
          if isinstance(row[i], list):
            return _re_sum(row[i], axis-1)
          return [_re_sum(data[j], None) for j in range(len(data))]
  
  if axis is not None and (axis >= len(get_shape(data))):
    raise ValueError("Axis out of range for the tensor")
  axis = axis + len(get_shape(data)) if axis < 0 else axis
  out = _re_sum(data, axis)
  if keepdim:
    if isinstance(out[0], list):
      out = [item for item in out]
  else:
    out = flatten(out)
  return out

x = [[[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]]]

print("shape: ", get_shape(x))
print("sum: ", tensor_sum(x, 0))
print("sum: ", tensor_sum(x, 1))
print("sum: ", tensor_sum(x, 2))
print("sum: ", tensor_sum(x, -1))