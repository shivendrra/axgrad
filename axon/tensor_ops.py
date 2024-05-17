from .arrays import tensor
from .helpers.statics import zeros

def convolution_2d(image:list, kernel:list) -> list:
  img_h, img_w = len(image), len(image[0])
  ker_h, ker_w = len(kernel), len(kernel[0])

  output_height = img_h - ker_h + 1
  output_width = img_w - ker_w + 1

  output = [[0] * output_width for _ in range(output_height)]

  for i in range(output_height):
    for j in range(output_width):
      for k in range(ker_h):
        for l in range(ker_w):
          output[i][j] += image[i+k][j+l] * kernel[k][l]
  return output

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

def stack(array: tuple, dim: int=0) -> tensor:
  if not array:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = array[0].shape
  for arr in array:
    if arr.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  
  # new shape after stacking & initilization
  new_shape = list(base_shape[:])
  new_shape.insert(dim, len(array))
  new_data = zeros(new_shape)
  
  def get_element(data, indices):
    for idx in indices:
        data = data[idx]
    return data

  def insert_data(new_data, arrays, dim, indices=[]):
    if len(indices) == len(new_shape):
      for idx, array in enumerate(arrays):
        data_idx = indices[:]
        data_idx[dim] = idx
        sub_arr = new_data
        for k in data_idx[:-1]:
          sub_arr = sub_arr[k]
        sub_arr[data_idx[-1]] = get_element(array.data, indices[:dim] + indices[dim+1:])
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, arrays, dim, indices + [i])
    
  insert_data(new_data, array, dim)
  return tensor(new_data)

def cat(array: tuple, dim: int=0) -> tensor:
  if not array:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = array[0].shape # shape of first array for target array
  for arr in array:
    if arr.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  new_shape = list(base_shape[:])
  print(new_shape)
  new_shape.insert(dim, len(array))
  print(new_shape)
  pass