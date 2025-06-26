type_dtypes = ["int8", "int16", "int32", "int64", "long", "float32", "float64", "double", "uint8", "uint16", "uint32", "uint64", "bool"]

def get_dtypes(): return type_dtypes

def get_shape(data: list):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def flatten(subdata):
  if isinstance(subdata, list):
    return [item for sub in subdata for item in flatten(sub)]
  else:
    return [subdata]

def get_size(shape:tuple) -> list:
  out = 1
  for dim in shape:
    out *= dim
  return out

def get_strides(shape:tuple) -> list:
  strides = [1]
  for size in reversed(shape[:-1]):
    strides.append(strides[-1] * size)
  return list(reversed(strides))

def transposed_shape(shape):
  ndim = len(shape)
  if ndim == 1:
    return shape  # no transpose for 1D
  elif ndim == 2:
    rows, cols = shape
    return [cols, rows]
  elif ndim == 3:
    batch, rows, cols = shape
    return [batch, cols, rows]
  else:
    raise ValueError(f"Unsupported shape dimension: {ndim}")

def reshape_list(flat_list, shape):
  if len(shape) == 1:
    return flat_list[:shape[0]]

  size = shape[0]
  stride = len(flat_list) // size
  result = []
  for i in range(size):
    start = i * stride
    end = start + stride
    result.append(reshape_list(flat_list[start:end], shape[1:]))
  return result